//
//  MetalConvolutionDepthwise.mm
//  MNN
//
//  Created by MNN on 2019/02/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "MetalConvolutionDepthwise.hpp"
#import "Macro.h"
#import "MetalBackend.hpp"

#if MNN_METAL_ENABLED

namespace MNN {
MetalConvolutionDepthwise::MetalConvolutionDepthwise(Backend *backend, const MNN::Op *op)
    : MetalConvolutionCommon(backend, op) {
    loadWeight(op->main_as_Convolution2D());
}

ErrorCode MetalConvolutionDepthwise::onResize(const std::vector<Tensor *> &inputs,
                                              const std::vector<Tensor *> &outputs) {
    MetalConvolutionCommon::onResize(inputs, outputs);

    // prepare
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto input = inputs[0], output = outputs[0];
    auto iw = input->width(), ih = input->height(), iz = UP_DIV(input->channel(), 4);
    auto ow = output->width(), oh = output->height(), ob = output->batch();

    // pad mode support
    int padX = mPadX, padY = mPadY;
    if (mPadMode == PadMode_SAME) {
        int kernelWidthSize = (mKernelX - 1) * mDilateX + 1;
        int kernelHeightSize = (mKernelY - 1) * mDilateY + 1;
        int pw = (ow - 1) * mStrideX + kernelWidthSize - iw;
        int ph = (oh - 1) * mStrideY + kernelHeightSize - ih;
        padX   = pw / 2;
        padY   = ph / 2;
    }

    // create const buffer
    int constants[] = {iw,
                       ih,
                       iw * ih,
                       ow,
                       oh,
                       ow * oh,
                       iz,
                       ob,

                       mKernelX,
                       mKernelY,
                       mKernelX * mKernelY,
                       mStrideX,
                       mStrideY,
                       padX,
                       padY,
                       mDilateX,
                       mDilateY,
                       mActivationType};
    mConstBuffer = [context newDeviceBuffer:sizeof(constants) bytes:constants access:CPUWriteOnly];
    return NO_ERROR;
}

ErrorCode MetalConvolutionDepthwise::onQuantized(const Tensor *input, const Tensor *output) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto w = output->width(), h = output->height(), z = UP_DIV(output->channel(), 4), b = output->batch();
    
    auto encoder   = [context encoder];
    auto bandwidth = [context load:@"qntconv_depthwise" encoder:encoder];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input->deviceId() offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output->deviceId() offset:0 atIndex:1];
    [encoder setBuffer:mConstBuffer offset:0 atIndex:2];
    [encoder setBuffer:mWeight offset:0 atIndex:3];
    [encoder setBuffer:mBias offset:0 atIndex:4];
    [encoder setBuffer:mAlpha offset:0 atIndex:5];
    [context dispatchEncoder:encoder threads:{ (NSUInteger)w, (NSUInteger)h, (NSUInteger)z * b } bandwidth:bandwidth];
    [encoder endEncoding];
    MNN_PRINT_ENCODER(context, encoder);
    return NO_ERROR;
}

ErrorCode MetalConvolutionDepthwise::onFloat(const Tensor *input, const Tensor *output) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto w = output->width(), h = output->height(), z = UP_DIV(output->channel(), 4), b = output->batch();
    
    auto encoder   = [context encoder];
    auto bandwidth = [context load:@"conv_depthwise" encoder:encoder];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input->deviceId() offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output->deviceId() offset:0 atIndex:1];
    [encoder setBuffer:mConstBuffer offset:0 atIndex:2];
    [encoder setBuffer:mWeight offset:0 atIndex:3];
    [encoder setBuffer:mBias offset:0 atIndex:4];
    [context dispatchEncoder:encoder threads:{ (NSUInteger)w, (NSUInteger)h, (NSUInteger)z * b } bandwidth:bandwidth];
    [encoder endEncoding];
    MNN_PRINT_ENCODER(context, encoder);
    return NO_ERROR;
}

template <typename FType, typename TType>
static id<MTLBuffer> weightInBlock(MNNMetalContext *context, int group, int kh, int kw, const FType *src) {
    auto buffer = [context newDeviceBuffer:UP_DIV(group, 4) * 4 * kw * kh * sizeof(TType) access:CPUWriteOnly];
    auto dst    = (TType *)buffer.contents;
    for (int g = 0; g < group; g++) {
        auto z = g / 4, r = g % 4;
        auto z_dst = dst + z * kh * kw * 4 + r;
#pragma clang loop vectorize(enable)
        for (int h = 0; h < kh; h++) {
#pragma clang loop vectorize(enable) unroll(enable)
            for (int w = 0; w < kw; w++) {
                // to   [g/4][h][w][4]
                // from [g][h][w]
                // dst[(z * kh * kw + h * kw + w) * 4 + r] =
                // src[ g * kh * kw + h * kw + w];
                z_dst[(h * kw + w) * 4] = *src++;
            }
        }
    }
    return buffer;
}

id<MTLBuffer> MetalConvolutionDepthwise::weightForQuantized(int group, int oc, int ic, int kh, int kw,
                                                            const int8_t *src) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)static_cast<MetalBackend *>(backend)->context();
    return weightInBlock<int8_t, int8_t>(context, group, kh, kw, src);
}

id<MTLBuffer> MetalConvolutionDepthwise::weightForFloat(int group, int oc, int ic, int kh, int kw, const float *src) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)static_cast<MetalBackend *>(backend)->context();
    return weightInBlock<float, metal_float>(context, group, kh, kw, src);
}

class MetalConvolutionDepthwiseCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend) const {
        return new MetalConvolutionDepthwise(backend, op);
    }
};
REGISTER_METAL_OP_CREATOR(MetalConvolutionDepthwiseCreator, OpType_ConvolutionDepthwise);
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
