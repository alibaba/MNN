//
//  MetalConvolutionWinograd.mm
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "backend/metal/MetalConvolutionWinograd.hpp"
#import "core/Macro.h"
#import "core/Macro.h"
#import "backend/metal/MetalBackend.hpp"
#import "backend/metal/MetalConvolution.hpp"
#import "math/WingoradGenerater.hpp"

#if MNN_METAL_ENABLED

#define UNIT 2

namespace MNN {
bool MetalConvolutionWinograd::isValid(Backend *backend, const Convolution2D *conv, const Tensor* input, const Tensor *output) {
    auto common = conv->common();
    if (!((common->kernelX() == common->kernelY()) && ((common->kernelX() == 3) || (common->kernelX() == 5)))
        || common->dilateX() != 1
        || common->dilateY() != 1
        || common->strideX() != 1
        || common->strideY() != 1) {
        return false;
    }
    
    int ow = output->width();
    int oh = output->height();
    int oc = output->channel();
    int ic = input->channel();
    
    auto winogradMemoryLevel = static_cast<MetalBackend *>(backend)->getRuntime()->hint().winogradMemoryUsed;
    // 0 means not use winograd
    if (winogradMemoryLevel == 0) {
        return false;
    } else if(winogradMemoryLevel < 3) {
        auto block =  UNIT + common->kernelY() - 1;
        size_t tw = block * block * ic * oc;
        // memory care mode & transformed weight element size bigger than 32M
        auto magic = 8.0;
        if(winogradMemoryLevel == 2) {
            magic = 32.0;
        }
        if(tw / 1024.0 / 1024.0 > magic) {
            return false;
        }
    }


    if(oc >= 16 && ic >= 16) {
        return true;
    }
    return (ow <= 16 && oh <= 16);
}

MetalConvolutionWinograd::MetalConvolutionWinograd(Backend *backend, const MNN::Op *op)
    : MetalConvolutionCommon(backend, op, nullptr) {
    auto conv = op->main_as_Convolution2D();
    mSrcUnit  = UNIT + conv->common()->kernelY() - 1;
    mDstUnit  = UNIT;
    loadWeight(op);
}
MetalConvolutionWinograd::MetalConvolutionWinograd(Backend *backend, const MNN::Op *op, std::shared_ptr<Tensor> weight, std::shared_ptr<Tensor> bias) : MetalConvolutionCommon(backend, op, bias) {
    auto conv = op->main_as_Convolution2D();
    mSrcUnit  = UNIT + conv->common()->kernelY() - 1;
    mDstUnit  = UNIT;
    mWeight = weight;
}
bool MetalConvolutionWinograd::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (!mValid) {
        return false;
    }
    if (nullptr == dst) {
        return true;
    }
    *dst = new MetalConvolutionWinograd(bn, op, mWeight, mBias);
    return true;
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

    auto pads = ConvolutionCommon::convolutionPad(input, output, mOp->main_as_Convolution2D()->common());
    auto padX = pads.first;
    auto padY = pads.second;

    int bytes = backend->useFp16InsteadFp32() ? 2 : 4;

    // accquire space
    int is = mSrcUnit * mSrcUnit * us * iz * 16 * bytes;
    int os = mSrcUnit * mSrcUnit * us * oz * 16 * bytes;

    // single tensor need over 128MB memory, split in block
    float mbytes = ALIMAX(is / 1024.0 / 1024.0, os / 1024.0 / 1024.0);
    mSplitNum = 1;
    if(mbytes > 32.0) {
        mSplitNum = mbytes > 512.0 ? 16 : (mbytes < 64.0 ? 4 : 8);
        uh = UP_DIV(uh, mSplitNum);
        us  = UP_DIV(uw * uh, 4);
        is = mSrcUnit * mSrcUnit * us * iz * 16 * bytes;
        os = mSrcUnit * mSrcUnit * us * oz * 16 * bytes;
    }

    
    // create const buffer
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
    mConstBuffer = backend->getConstBuffer(sizeof(transform));
    ::memcpy(mConstBuffer.contents, &transform, sizeof(transform));

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
    
    mTempSrc.reset(Tensor::createDevice<uint8_t>(std::vector<int>{is}));
    mTempDst.reset(Tensor::createDevice<uint8_t>(std::vector<int>{os}));
    backend->onAcquireBuffer(mTempSrc.get(), Backend::DYNAMIC);
    backend->onAcquireBuffer(mTempDst.get(), Backend::DYNAMIC);
    backend->onReleaseBuffer(mTempSrc.get(), Backend::DYNAMIC);
    backend->onReleaseBuffer(mTempDst.get(), Backend::DYNAMIC);

    return NO_ERROR;
}

void MetalConvolutionWinograd::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, id<MTLComputeCommandEncoder> encoder) {
    auto input = inputs[0];
    auto output = outputs[0];
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    for(int b = 0; b < input->batch(); b++) {
        int batch_idx = b;
        for(int split_idx = 0; split_idx < mSplitNum; split_idx++) {
            { // transform
                auto pipeline = [context pipelineWithName:mKernelX == 3 ? @"winograd_transform_source2_3_1" : @"winograd_transform_source2_5_1" fp16:backend->useFp16InsteadFp32()];
                [encoder setComputePipelineState:pipeline];
                // [ci/4, batch, height, width, ci_4]
                MetalBackend::setTensor(input, encoder, 0);
                // [mSrcUnit * mSrcUnit, UP_DIV(uw * wh, 4), UP_DIV(ci, 4), (uw * wh)_4, ci_4]
                MetalBackend::setTensor(mTempSrc.get(), encoder, 1);
                [encoder setBuffer:mConstBuffer offset:0 atIndex:2];
                [encoder setBytes:&batch_idx length:sizeof(batch_idx) atIndex:3];
                [encoder setBytes:&split_idx length:sizeof(split_idx) atIndex:4];
                auto gl = [context computeBestGroupAndLocal:pipeline threads:mInputTransformThreads];
                [encoder dispatchThreadgroups:gl.first threadsPerThreadgroup:gl.second];
            }
            { // gemm
                auto pipeline = [context pipelineWithName:@"matmul4x4" fp16:backend->useFp16InsteadFp32()];
                [encoder setComputePipelineState:pipeline];
                // [mSrcUnit * mSrcUnit, UP_DIV(uw * wh, 4), UP_DIV(ci, 4), (uw * wh)_4, ci_4]
                MetalBackend::setTensor(mTempSrc.get(), encoder, 0);
                // [co/4, (uw * wh)_4, mSrcUnit * mSrcUnit, UP_DIV(uw * wh, 4), co_4]
                MetalBackend::setTensor(mTempDst.get(), encoder, 1);
                // [mSrcUnit * mSrcUnit, UP_DIV(co, 4), UP_DIV(ci, 4), co_4, ci_4]
                MetalBackend::setTensor(mWeight.get(), encoder, 2);
                [encoder setBuffer:mShapeBuffer offset:0 atIndex:3];
                auto gl = [context computeBestGroupAndLocal:pipeline threads:mMatMulThreads];
                [encoder dispatchThreadgroups:gl.first threadsPerThreadgroup:gl.second];
            }
            { // transform
                auto pipeline = [context pipelineWithName:mKernelX == 3 ? @"winograd_transform_dest2_3_1" : @"winograd_transform_dest2_5_1" fp16:backend->useFp16InsteadFp32()];
                [encoder setComputePipelineState:pipeline];
                // [co/4, (uw * wh)_4, mSrcUnit * mSrcUnit, UP_DIV(uw * wh, 4), co_4]
                MetalBackend::setTensor(mTempDst.get(), encoder, 0);
                MetalBackend::setTensor(mBias.get(), encoder, 1);
                // [co/4, batch, height, width, co_4]
                MetalBackend::setTensor(output, encoder, 2);
                [encoder setBuffer:mConstBuffer offset:0 atIndex:3];
                [encoder setBytes:&batch_idx length:sizeof(batch_idx) atIndex:4];
                [encoder setBytes:&split_idx length:sizeof(split_idx) atIndex:5];
                auto gl = [context computeBestGroupAndLocal:pipeline threads:mOutputTransformThreads];
                [encoder dispatchThreadgroups:gl.first threadsPerThreadgroup:gl.second];
            }
        }
    }
}
std::shared_ptr<MNN::Tensor> MetalConvolutionWinograd::weightTransform(int group, int oc, int ic, int kh, int kw, const float *src, bool int8Weight, bool int4Weight, id<MTLBuffer> srcGpuBuffer) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)static_cast<MetalBackend *>(backend)->context();

    std::shared_ptr<Tensor> srcWeight(Tensor::create<float>(std::vector<int>{oc, ic, kh, kh}, (void *)src, Tensor::CAFFE));
    Math::WinogradGenerater generater(mDstUnit, kh, 1.0f);
    std::shared_ptr<Tensor> dstWeight = generater.allocTransformWeight(srcWeight.get(), 4, 4);
    if (nullptr == dstWeight->host<float>()) {
        // Alloc cpu memory error
        MNN_ERROR("Alloca cpu memory error in MetalConvolutionWinograd.mm\n");
        return nullptr;
    }
    generater.transformWeight(dstWeight.get(), srcWeight.get());
    std::shared_ptr<Tensor> dstWeightGpu = generater.allocTransformWeight(srcWeight.get(), 4, 4, false);
    auto res = backend->onAcquireBuffer(dstWeightGpu.get(), Backend::STATIC);
    if (!res) {
        MNN_ERROR("Alloca GPU memory error in MetalConvolutionWinograd.mm\n");
        return nullptr;
    }

    auto buffer = MetalBackend::getBuffer(dstWeightGpu.get());
    uint8_t* bytes = (uint8_t*)[buffer.first contents] + buffer.second;
    auto length = dstWeight->elementSize();
    if (backend->useFp16InsteadFp32()) {
        auto f32 = dstWeight->host<float>();
        auto f16 = (__fp16*)bytes;
        for (int i = 0; i < length; ++i) {
            f16[i] = f32[i];
        }
    } else {
        ::memcpy(bytes, dstWeight->host<float>(), length * sizeof(float));
    }
    return dstWeightGpu;
}

} // namespace MNN
#endif /* MNN_METAL_ENABLED */
