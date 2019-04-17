//
//  MetalConvolutionCommon.mm
//  MNN
//
//  Created by MNN on 2019/02/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "MetalConvolutionCommon.hpp"
#import "Macro.h"
#import "MetalBackend.hpp"
#import "MetalConvolution1x1.hpp"
#import "MetalConvolutionWinograd.hpp"
#import "TensorUtils.hpp"

#if MNN_METAL_ENABLED
namespace MNN {

static id<MTLBuffer> biasForConv(MNNMetalContext *context, const Convolution2D *conv) {
    auto bias   = conv->bias();
    auto oc     = conv->common()->outputCount();
    auto buffer = [context newDeviceBuffer:UP_DIV(oc, 4) * 4 * sizeof(metal_float) access:CPUWriteOnly];
    auto src    = bias->data();
    auto dst    = (metal_float *)buffer.contents;
#pragma clang loop vectorize(enable) unroll(enable)
    for (int i = 0; i < oc; i++) {
        dst[i] = src[i];
    }
    return buffer;
}

static id<MTLBuffer> alphaForConv(MNNMetalContext *context, const Convolution2D *conv) {
    auto alpha  = conv->quanParameter()->alpha();
    auto oc     = conv->common()->outputCount();
    auto buffer = [context newDeviceBuffer:UP_DIV(oc, 4) * 4 * sizeof(float) access:CPUWriteOnly];
    auto src    = alpha->data();
    auto dst    = (float *)buffer.contents;
#pragma clang loop vectorize(enable) unroll(enable)
    for (int i = 0; i < oc; i++) {
        dst[i] = src[i];
    }
    return buffer;
}

MetalConvolutionCommon::MetalConvolutionCommon(Backend *backend, const MNN::Op *op) : Execution(backend) {
    auto context    = (__bridge MNNMetalContext *)static_cast<MetalBackend *>(backend)->context();
    auto conv       = op->main_as_Convolution2D();
    auto common     = conv->common();
    mDepthwise      = op->type() == OpType_ConvolutionDepthwise;
    mGroups         = common->group();
    mKernelX        = common->kernelX();
    mKernelY        = common->kernelY();
    mPadMode        = common->padMode();
    mPadX           = common->padX();
    mPadY           = common->padY();
    mStrideX        = common->strideX();
    mStrideY        = common->strideY();
    mDilateX        = common->dilateX();
    mDilateY        = common->dilateY();
    mBias           = biasForConv(context, conv);
    mActivationType = common->relu() ? 1 : (common->relu6() ? 2 : 0);
}

ErrorCode MetalConvolutionCommon::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    // reserve space for qnt input
    if (mQnt) {
        auto backend = static_cast<MetalBackend *>(this->backend());
        mQntInput.reset(new Tensor);
        TensorUtils::copyShape(inputs[0], mQntInput.get());
        mQntInput->setType(DataType_DT_INT8);
        backend->onAcquireBuffer(mQntInput.get(), Backend::DYNAMIC);
        backend->onReleaseBuffer(mQntInput.get(), Backend::DYNAMIC);
    }
    return NO_ERROR;
}

ErrorCode MetalConvolutionCommon::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    if (mQnt) {
        auto backend   = static_cast<MetalBackend *>(this->backend());
        auto context   = (__bridge MNNMetalContext *)backend->context();
        auto encoder   = [context encoder];
        auto bandwidth = [context load:@"conv_quantize" encoder:encoder];
        [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)inputs[0]->deviceId() offset:0 atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)mQntInput->deviceId() offset:0 atIndex:1];
        [encoder setBuffer:[context newDeviceBuffer:sizeof(mQntScale) bytes:&mQntScale access:CPUWriteOnly]
                    offset:0
                   atIndex:2];
        [encoder setBuffer:[context newDeviceBuffer:sizeof(mQntRange) bytes:&mQntRange access:CPUWriteOnly]
                    offset:0
                   atIndex:3];
        [context dispatchEncoder:encoder
                         threads:{ (NSUInteger) mQntInput->elementSize() / 4, 1, 1 }
                       bandwidth:bandwidth];
        [encoder endEncoding];
        MNN_PRINT_ENCODER(context, encoder);
        return onQuantized(mQntInput.get(), outputs[0]);
    } else {
        return onFloat(inputs[0], outputs[0]);
    }
}

template <typename FType, typename TType>
static id<MTLBuffer> weightInBlock(MNNMetalContext *context, int group, int oc, int ic, int kh, int kw,
                                   const FType *src) {
    auto goc    = oc / group;
    auto gic    = ic / group;
    auto goc_4  = UP_DIV(goc, 4);
    auto gic_4  = UP_DIV(gic, 4);
    auto buffer = [context newDeviceBuffer:group * goc_4 * gic_4 * kw * kh * 16 * sizeof(TType) access:CPUWriteOnly];
    auto dst    = (TType *)buffer.contents;

    for (int g = 0; g < group; g++) {
        auto g_dst = dst + g * goc_4 * gic_4 * kh * kw * 16; // g
#pragma clang loop vectorize(enable)
        for (int o = 0; o < goc; o++) {
            auto zo = o / 4, ro = o % 4;
            auto o_dst = g_dst + zo * gic_4 * kh * kw * 16 + ro * 4; // o/4 x 4
#pragma clang loop vectorize(enable)
            for (int i = 0; i < gic; i++) {
                auto zi = i / 4, ri = i % 4;
                auto i_dst = o_dst + zi * kh * kw * 16 + ri; // i/4 x 4
#pragma clang loop vectorize(enable)
                for (int h = 0; h < kh; h++) {
#pragma clang loop vectorize(enable) unroll(enable)
                    for (int w = 0; w < kw; w++) {
                        // to   [g][o/4][i/4][h][w][16]
                        // from [g][o][i][h][w]
                        i_dst[(h * kw + w) * 16] = *src++;
                    }
                }
            }
        }
    }
    return buffer;
}

void MetalConvolutionCommon::loadWeight(const MNN::Convolution2D *conv) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)static_cast<MetalBackend *>(backend)->context();

    std::shared_ptr<ConvolutionIntFactory::Int8Common> qnt = NULL;
    if (conv->quanParameter()) {
        qnt          = ConvolutionIntFactory::load(conv->quanParameter(), false);
        mQnt         = qnt->weight.size() > 0;
        mQntRange[0] = conv->quanParameter()->aMin();
        mQntRange[1] = conv->quanParameter()->aMax();
        mQntScale    = conv->quanParameter()->quantScale();
        mAlpha       = alphaForConv(context, conv);
    }
    mWeight = weightForConv(conv, qnt.get(), mDepthwise);
}

id<MTLBuffer> MetalConvolutionCommon::weightForQuantized(int group, int oc, int ic, int kh, int kw, const int8_t *src) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)static_cast<MetalBackend *>(backend)->context();
    return weightInBlock<int8_t, int8_t>(context, group, oc, ic, kh, kw, src);
}

id<MTLBuffer> MetalConvolutionCommon::weightForFloat(int group, int oc, int ic, int kh, int kw, const float *src) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)static_cast<MetalBackend *>(backend)->context();
    return weightInBlock<float, metal_float>(context, group, oc, ic, kh, kw, src);
}

id<MTLBuffer> MetalConvolutionCommon::weightForConv(const Convolution2D *conv, ConvolutionIntFactory::Int8Common *qnt,
                                                    bool depthwise) {
    // param
    auto size   = qnt ? MAX(qnt->weight.size(), qnt->weightFloat.size()) : conv->weight()->size();
    auto common = conv->common();
    auto kw     = common->kernelX();
    auto kh     = common->kernelY();
    auto group  = common->group();
    auto oc     = common->outputCount();
    auto ic     = size / kw / kh / (oc / group);

    // convert
    if (qnt && qnt->weight.size() > 0) {
        return weightForQuantized(group, oc, ic, kh, kw, qnt->weight.get());
    } else if (qnt && qnt->weightFloat.size() > 0) {
        return weightForFloat(group, oc, ic, kh, kw, qnt->weightFloat.get());
    } else {
        return weightForFloat(group, oc, ic, kh, kw, conv->weight()->data());
    }
}
} // namespace MNN

#endif
