//
//  MetalTFQuantizedConv2D.mm
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "MetalTFQuantizedConv2D.hpp"
#import "CPUQuantizationUtils.hpp"
#import "MNNMetalContext.h"
#import "Macro.h"
#import "MetalBackend.hpp"

#if MNN_METAL_ENABLED
namespace MNN {

static id<MTLBuffer> weightForDepthwise(MNNMetalContext *context, const TfQuantizedConv2D *conv, int32_t zeropoint) {
    auto common = conv->common();
    auto kw     = common->kernelX();
    auto kh     = common->kernelY();
    auto group  = common->outputCount();

    auto buffer = [context newDeviceBuffer:UP_DIV(group, 4) * 4 * kw * kh * sizeof(int16_t) access:CPUWriteOnly];
    auto src    = conv->weight()->data();
    auto dst    = (int16_t *)buffer.contents;
    for (int h = 0; h < kh; h++) {
        for (int w = 0; w < kw; w++) {
#pragma clang loop unroll(enable)
            for (int g = 0; g < group; g++) {
                auto z = g / 4, r = g % 4;
                auto z_dst = dst + z * kh * kw * 4 + r;
                // to   [g/4][h][w][4]
                // from [h][w][g]
                z_dst[(h * kw + w) * 4] = (int16_t)((int32_t)*src++ - zeropoint);
            }
        }
    }
    return buffer;
}

static id<MTLBuffer> weightForConv(MNNMetalContext *context, const TfQuantizedConv2D *conv, int32_t zeropoint) {
    auto size   = conv->weight()->size();
    auto common = conv->common();
    auto kw     = common->kernelX();
    auto kh     = common->kernelY();
    auto group  = common->group();
    auto oc     = common->outputCount();
    auto ic     = size / kw / kh / (oc / group);

    auto goc    = oc / group;
    auto gic    = ic / group;
    auto goc_4  = UP_DIV(goc, 4);
    auto gic_4  = UP_DIV(gic, 4);
    auto buffer = [context newDeviceBuffer:group * goc_4 * gic_4 * kw * kh * 16 * sizeof(int16_t) access:CPUWriteOnly];
    auto src    = conv->weight()->data();
    auto dst    = (int16_t *)buffer.contents;

    for (int g = 0; g < group; g++) {
        auto g_dst = dst + g * goc_4 * gic_4 * kh * kw * 16; // g
        for (int h = 0; h < kh; h++) {
            for (int w = 0; w < kw; w++) {
#pragma clang loop vectorize(enable)
                for (int i = 0; i < gic; i++) {
                    auto zi = i / 4, ri = i % 4;
                    auto i_dst = g_dst + zi * kh * kw * 16 + ri; // i/4 x 4
#pragma clang loop unroll(enable)
                    for (int o = 0; o < goc; o++) {
                        auto zo = o / 4, ro = o % 4;
                        auto o_dst = i_dst + zo * gic_4 * kh * kw * 16 + ro * 4; // o/4 x 4
                        // to   [g][o/4][i/4][h][w][16]
                        // from [g][h][w][i][o]
                        o_dst[(h * kw + w) * 16] = (int16_t)((int32_t)*src++ - zeropoint);
                    }
                }
            }
        }
    }
    return buffer;
}

static id<MTLBuffer> biasForConv(MNNMetalContext *context, const TfQuantizedConv2D *conv) {
    auto bias   = conv->bias();
    auto oc     = conv->common()->outputCount();
    auto buffer = [context newDeviceBuffer:UP_DIV(oc, 4) * 4 * sizeof(int) access:CPUWriteOnly];
    auto src    = bias->data();
    auto dst    = (int *)buffer.contents;
#pragma clang loop vectorize(enable) unroll(enable)
    for (int i = 0; i < oc; i++)
        dst[i] = src[i];
    return buffer;
}

MetalTFQuantizedConv2D::MetalTFQuantizedConv2D(Backend *backend, const MNN::Op *op) : Execution(backend) {
    auto context = (__bridge MNNMetalContext *)static_cast<MetalBackend *>(backend)->context();
    auto conv    = op->main_as_TfQuantizedConv2D();
    auto common  = conv->common();
    mDepthwise   = op->type() == OpType_QuantizedDepthwiseConv2D;
    mPadMode     = common->padMode();
    mGroups      = common->group();
    mKernelX     = common->kernelX();
    mKernelY     = common->kernelY();
    mStrideX     = common->strideX();
    mStrideY     = common->strideY();
    mDilateX     = common->dilateX();
    mDilateY     = common->dilateY();

    mInputZeroPoint            = conv->inputQuantizedParam()->zeroPoint();
    mOutputZeroPoint           = conv->outputQuantizedParam()->zeroPoint();
    float inputScale           = conv->inputQuantizedParam()->scale();
    float filter_scale         = conv->filterQuantizedParam()->scale();
    float output_scale         = conv->outputQuantizedParam()->scale();
    double input_product_scale = inputScale * filter_scale;
    double real_multiplier     = input_product_scale / output_scale;
    int shift                  = 0;
    QuantizeMultiplierSmallerThanOne(real_multiplier, &mOutputMultiplier, &shift);
    shift = -shift;
    if (shift < 0) {
        mOutputShiftBefore = 0;
        mOutputShiftAfter  = shift;
    } else {
        mOutputShiftBefore = shift;
        mOutputShiftAfter  = 0;
    }
    CalculateActivationRangeUint8(conv->activationType(), mOutputZeroPoint, output_scale, &mOutputActivationMin,
                                  &mOutputActivationMax);
    mWeight = mDepthwise ? weightForDepthwise(context, conv, conv->filterQuantizedParam()->zeroPoint())
                         : weightForConv(context, conv, conv->filterQuantizedParam()->zeroPoint());
    mBias = biasForConv(context, conv);
}

ErrorCode MetalTFQuantizedConv2D::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    // prepare
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto input = inputs[0], output = outputs[0];
    auto iw = input->width(), ih = input->height(), igz = UP_DIV(input->channel(), 4) / mGroups;
    auto ow = output->width(), oh = output->height(), ogz = UP_DIV(output->channel(), 4) / mGroups;

    // pad mode support
    int pw = (ow - 1) * mStrideX + mKernelX - iw;
    int ph = (oh - 1) * mStrideY + mKernelY - ih;
    if (mPadMode == PadMode_VALID) {
        pw += 1;
        ph += 1;
    }
    int padX = pw / 2;
    int padY = ph / 2;

    // create const buffer
    int constants[] = {
        iw,
        ih,
        iw * ih,
        igz,
        ow,
        oh,
        ow * oh,
        ogz,

        mKernelX,
        mKernelY,
        mKernelX * mKernelY,
        mStrideX,
        mStrideY,
        padX,
        padY,
        mDilateX,
        mDilateY,

        mInputZeroPoint,
        mOutputZeroPoint,
        mOutputShiftBefore,
        mOutputMultiplier,
        mOutputShiftAfter,
        mOutputActivationMin,
        mOutputActivationMax,
    };
    mConstBuffer = [context newDeviceBuffer:sizeof(constants) bytes:constants access:CPUWriteOnly];
    return NO_ERROR;
}

ErrorCode MetalTFQuantizedConv2D::onConv(const Tensor *input, const Tensor *output) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto iw = input->width(), ih = input->height(), iz = UP_DIV(input->channel(), 4);
    auto ow = output->width(), oh = output->height(), oz = UP_DIV(output->channel(), 4), ogz = oz / mGroups;
    auto ib = iw * ih * iz * 4 * sizeof(uint8_t), ig = ib / mGroups;
    auto ob = ow * oh * oz * 4 * sizeof(uint8_t), og = ob / mGroups;

    auto encoder = [context encoder];
    if (ogz > 16) {
        auto bandwidth = [context load:@"tfqntconv_z4" encoder:encoder];
        for (int b = 0; b < input->batch(); b++) {
            for (int g = 0; g < mGroups; g++) {
                [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input->deviceId() offset:b * ib + g * ig atIndex:0];
                [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output->deviceId() offset:b * ob + g * og atIndex:1];
                [encoder setBuffer:mConstBuffer offset:0 atIndex:2];
                [encoder setBuffer:mWeight offset:g * mWeight.length / mGroups atIndex:3];
                [encoder setBuffer:mBias offset:g * mBias.length / mGroups atIndex:4];
                [context dispatchEncoder:encoder
                                 threads:{ (NSUInteger) ow, (NSUInteger)oh, (NSUInteger)UP_DIV(ogz, 4) }
                               bandwidth:bandwidth];
            }
        }
    } else {
        auto bandwidth = [context load:@"tfqntconv" encoder:encoder];
        for (int b = 0; b < input->batch(); b++) {
            for (int g = 0; g < mGroups; g++) {
                [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input->deviceId() offset:b * ib + g * ig atIndex:0];
                [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output->deviceId() offset:b * ob + g * og atIndex:1];
                [encoder setBuffer:mConstBuffer offset:0 atIndex:2];
                [encoder setBuffer:mWeight offset:g * mWeight.length / mGroups atIndex:3];
                [encoder setBuffer:mBias offset:g * mBias.length / mGroups atIndex:4];
                [context dispatchEncoder:encoder
                                 threads:{ (NSUInteger) ow, (NSUInteger)oh, (NSUInteger)ogz }
                               bandwidth:bandwidth];
            }
        }
    }
    [encoder endEncoding];
    MNN_PRINT_ENCODER(context, encoder);
    return NO_ERROR;
}

ErrorCode MetalTFQuantizedConv2D::onDepthwise(const Tensor *input, const Tensor *output) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto iw = input->width(), ih = input->height(), iz = UP_DIV(input->channel(), 4);
    auto ow = output->width(), oh = output->height(), oz = UP_DIV(output->channel(), 4);
    auto ib = iw * ih * iz * 4 * (int)sizeof(uint8_t);
    auto ob = ow * oh * oz * 4 * (int)sizeof(uint8_t);

    auto encoder   = [context encoder];
    auto bandwidth = [context load:@"tfqntconv_depthwise" encoder:encoder];
    for (int b = 0; b < input->batch(); b++) {
        [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input->deviceId() offset:b * ib atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output->deviceId() offset:b * ob atIndex:1];
        [encoder setBuffer:mConstBuffer offset:0 atIndex:2];
        [encoder setBuffer:mWeight offset:0 atIndex:3];
        [encoder setBuffer:mBias offset:0 atIndex:4];
        [context dispatchEncoder:encoder
                         threads:{ (NSUInteger) ow, (NSUInteger)oh, (NSUInteger)oz }
                       bandwidth:bandwidth];
    }
    [encoder endEncoding];
    MNN_PRINT_ENCODER(context, encoder);
    return NO_ERROR;
}

ErrorCode MetalTFQuantizedConv2D::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    if (mDepthwise) {
        return onDepthwise(inputs[0], outputs[0]);
    } else {
        return onConv(inputs[0], outputs[0]);
    }
}

class MetalTFQuantizedConv2DCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend) const {
        return new MetalTFQuantizedConv2D(backend, op);
    }
};
REGISTER_METAL_OP_CREATOR(MetalTFQuantizedConv2DCreator, OpType_TfQuantizedConv2D);
REGISTER_METAL_OP_CREATOR(MetalTFQuantizedConv2DCreator, OpType_QuantizedDepthwiseConv2D);
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
