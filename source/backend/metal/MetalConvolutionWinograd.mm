//
//  MetalConvolutionWinograd.mm
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "MetalConvolutionWinograd.hpp"
#import "Macro.h"
#import "Macro.h"
#import "MetalBackend.hpp"
#import "MetalConvolution.hpp"
#import "WingoradGenerater.hpp"

#if MNN_METAL_ENABLED

#define UNIT 2

namespace MNN {
bool MetalConvolutionWinograd::isValid(const Convolution2D *conv, const Tensor *input) {
    if (conv->quanParameter() != nullptr || conv->common()->group() != 1) {
        return false;
    }
    auto common = conv->common();
    if (input->batch() != 1
        || !((common->kernelX() == common->kernelY()) && ((common->kernelX() == 3) || (common->kernelX() == 5)))
        || common->dilateX() != 1
        || common->dilateY() != 1
        || common->strideX() != 1
        || common->strideY() != 1) {
        return false;
    }
    auto iw = input->width(), ih = input->height();
    auto ic = ROUND_UP(input->channel(), 4), oc = ROUND_UP(common->outputCount(), 4);
    return ic * oc * ih / iw >= 2048;
}

MetalConvolutionWinograd::MetalConvolutionWinograd(Backend *backend, const Tensor *input, const MNN::Op *op)
    : MetalConvolutionCommon(backend, op) {
    auto conv = op->main_as_Convolution2D();
    mSrcUnit  = UNIT + conv->common()->kernelY() - 1;
    mDstUnit  = UNIT;
    loadWeight(conv);
}

ErrorCode MetalConvolutionWinograd::onResize(const std::vector<Tensor *> &inputs,
                                             const std::vector<Tensor *> &outputs) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto input   = inputs[0];
    auto output  = outputs[0];

    auto ow  = output->width();
    auto oh  = output->height();
    auto uw  = UP_DIV(ow, mDstUnit);
    auto uh  = UP_DIV(oh, mDstUnit);
    auto us  = UP_DIV(uw * uh, 4);
    auto iz  = UP_DIV(input->channel(), 4);
    auto oz  = UP_DIV(output->channel(), 4);
    int padX = mPadX;
    int padY = mPadY;

    if (mPadMode == PadMode_SAME) {
        int kernelWidthSize = (mKernelX - 1) * mDilateX + 1;
        int kernelHeightSize = (mKernelY - 1) * mDilateY + 1;
        int pw = (output->width() - 1) * mStrideX + kernelWidthSize - input->width();
        int ph = (output->height() - 1) * mStrideY + kernelHeightSize - input->height();
        padX   = pw / 2;
        padY   = ph / 2;
    }

    // create const buffer
    struct TransformBuffer {
        int inputSize[4];
        int outputSize[4];
        int padX;
        int padY;
        int unitWidth;
        int unitHeight;
        int unit;
        int activation;
        int remain[2];
    };
    TransformBuffer transform;
    transform.inputSize[0]  = input->width();
    transform.inputSize[1]  = input->height();
    transform.inputSize[2]  = iz;
    transform.inputSize[3]  = input->batch();
    transform.outputSize[0] = output->width();
    transform.outputSize[1] = output->height();
    transform.outputSize[2] = oz;
    transform.outputSize[3] = output->batch();
    transform.padX          = padX;
    transform.padY          = padY;
    transform.unitWidth     = uw;
    transform.unitHeight    = uh;
    transform.unit          = mDstUnit;
    transform.activation    = mActivationType;
    mConstBuffer            = [context newDeviceBuffer:sizeof(transform) bytes:&transform access:CPUWriteOnly];

    // create matmul buffer
    int shapes[] = {us, oz, iz, mSrcUnit * mSrcUnit};
    mShapeBuffer = [context newDeviceBuffer:sizeof(shapes) bytes:shapes access:CPUWriteOnly];

    // save threads size
    mInputTransformThreads.width   = uw;
    mInputTransformThreads.height  = uh;
    mInputTransformThreads.depth   = iz;
    mMatMulThreads.width           = us;
    mMatMulThreads.height          = oz;
    mMatMulThreads.depth           = mSrcUnit * mSrcUnit;
    mOutputTransformThreads.width  = uw;
    mOutputTransformThreads.height = uh;
    mOutputTransformThreads.depth  = oz;

    // accquire space
    int is = mSrcUnit * mSrcUnit * us * iz * 16 * sizeof(metal_float) / sizeof(uint8_t);
    int os = mSrcUnit * mSrcUnit * us * oz * 16 * sizeof(metal_float) / sizeof(uint8_t);
    mTempSrc.reset(Tensor::createDevice<uint8_t>(std::vector<int>{is}));
    mTempDst.reset(Tensor::createDevice<uint8_t>(std::vector<int>{os}));
    backend->onAcquireBuffer(mTempSrc.get(), Backend::DYNAMIC);
    backend->onAcquireBuffer(mTempDst.get(), Backend::DYNAMIC);
    backend->onReleaseBuffer(mTempSrc.get(), Backend::DYNAMIC);
    backend->onReleaseBuffer(mTempDst.get(), Backend::DYNAMIC);

    return NO_ERROR;
}

ErrorCode MetalConvolutionWinograd::onQuantized(const Tensor *input, const Tensor *output) {
    return NOT_SUPPORT;
}
ErrorCode MetalConvolutionWinograd::onFloat(const Tensor *input, const Tensor *output) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto encoder = [context encoder];
    { // transform
        auto bandwidth = [context load:mKernelX == 3 ? @"winograd_transform_source2_3_1" : @"winograd_transform_source2_5_1" encoder:encoder];
        [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input->deviceId() offset:0 atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)mTempSrc->deviceId() offset:0 atIndex:1];
        [encoder setBuffer:mConstBuffer offset:0 atIndex:2];
        [context dispatchEncoder:encoder threads:mInputTransformThreads bandwidth:bandwidth];
    }
    { // gemm
        auto bandwidth = [context load:@"matmul4x4" encoder:encoder];
        [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)mTempSrc->deviceId() offset:0 atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)mTempDst->deviceId() offset:0 atIndex:1];
        [encoder setBuffer:mWeight offset:0 atIndex:2];
        [encoder setBuffer:mShapeBuffer offset:0 atIndex:3];
        [context dispatchEncoder:encoder threads:mMatMulThreads bandwidth:bandwidth];
    }
    { // transform
        auto bandwidth = [context load:mKernelX == 3 ? @"winograd_transform_dest2_3_1" : @"winograd_transform_dest2_5_1" encoder:encoder];
        [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)mTempDst->deviceId() offset:0 atIndex:0];
        [encoder setBuffer:mBias offset:0 atIndex:1];
        [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output->deviceId() offset:0 atIndex:2];
        [encoder setBuffer:mConstBuffer offset:0 atIndex:3];
        [context dispatchEncoder:encoder threads:mOutputTransformThreads bandwidth:bandwidth];
    }
    [encoder endEncoding];
    MNN_PRINT_ENCODER(context, encoder);

    return NO_ERROR;
}

id<MTLBuffer> MetalConvolutionWinograd::weightForQuantized(int group, int oc, int ic, int kh, int kw,
                                                           const int8_t *src) {
    return nil;
}

id<MTLBuffer> MetalConvolutionWinograd::weightForFloat(int group, int oc, int ic, int kh, int kw, const float *src) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)static_cast<MetalBackend *>(backend)->context();

    std::shared_ptr<Tensor> srcWeight(Tensor::create<float>(std::vector<int>{oc, ic, kh, kh}, (void *)src, Tensor::CAFFE));
    Math::WinogradGenerater generater(mDstUnit, kh, 1.0f);
    std::shared_ptr<Tensor> dstWeight = generater.allocTransformWeight(srcWeight.get(), 4, 4);
    generater.transformWeight(dstWeight.get(), srcWeight.get());

#if MNN_METAL_FULL_PRECISION
    auto bytes = dstWeight->host<metal_float>();
#else
    std::shared_ptr<Tensor> dstWeightHalf(Tensor::create<int16_t>(dstWeight->shape()));
    auto f32 = dstWeight->host<float>();
    auto f16 = dstWeightHalf->host<metal_float>();
    for (int i = 0; i < dstWeight->elementSize(); ++i) {
        f16[i] = f32[i];
    }
    auto bytes = dstWeightHalf->host<metal_float>();
#endif
    return [context newDeviceBuffer:4 * UP_DIV(ic, 4) * UP_DIV(oc, 4) * mSrcUnit * mSrcUnit * 4 * sizeof(metal_float)
                              bytes:bytes
                             access:CPUWriteOnly];
}

} // namespace MNN
#endif /* MNN_METAL_ENABLED */
