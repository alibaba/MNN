//
//  MetalDeconvolution.mm
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "MetalDeconvolution.hpp"
#import "ConvolutionIntFactory.hpp"
#import "MNNMetalContext.h"
#import "Macro.h"
#import "MetalBackend.hpp"

#if MNN_METAL_ENABLED
namespace MNN {
    
static int leastCommonMultiple(int m, int n) {
    int a = m, b = n;
    while(a != b){
        if (a > b){
            a = a - b;
        } else {
            b = b - a;
        }
    }
    return m * n / a;
}

template <typename FType, typename TType>
static id<MTLBuffer> weightForDeconv(MNNMetalContext *context, int group, int oc, int ic, int kh, int kw,
                                     const FType *src) {
    auto goc    = oc / group;
    auto gic    = ic / group;
    auto goc_4  = UP_DIV(goc, 4);
    auto gic_4  = UP_DIV(gic, 4);
    auto buffer = [context newDeviceBuffer:group * goc_4 * gic_4 * kw * kh * 16 * sizeof(TType) access:CPUWriteOnly];
    auto dst    = (TType *)buffer.contents;

    for (int g = 0; g < group; g++) {
#pragma clang loop vectorize(enable)
        for (int i = 0; i < gic; i++) {
#pragma clang loop vectorize(enable)
            for (int o = 0; o < goc; o++) {
#pragma clang loop vectorize(enable)
                for (int h = 0; h < kh; h++) {
#pragma clang loop vectorize(enable) unroll(enable)
                    for (int w = 0; w < kw; w++) {
                        auto zo = o / 4, ro = o % 4;
                        auto zi = i / 4, ri = i % 4;
                        // to   [g][o/4][i/4][h][w][16]
                        dst[(g * goc_4 * gic_4 * kh * kw + zo * gic_4 * kh * kw + zi * kh * kw + h * kw + w) * 16 +
                            ro * 4 + ri] =
                            // from [g][i][o][h][w]
                            //                            src[ g * goc   * gic   * kh * kw +  i * goc   * kh * kw +  o *
                            //                            kh * kw + h * kw + w];
                            *src++;
                    }
                }
            }
        }
    }
    return buffer;
}

template <typename FType, typename TType>
static id<MTLBuffer> weightForDepthwise(MNNMetalContext *context, int group, int kh, int kw, const FType *src) {
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

static id<MTLBuffer> weightForDeconv(MNNMetalContext *context, bool depthwise, const Convolution2D *deconv,
                                     ConvolutionIntFactory::Int8Common *qnt) {
    auto size   = qnt ? qnt->weightFloat.size() : deconv->weight()->size();
    auto common = deconv->common();
    auto kw     = common->kernelX();
    auto kh     = common->kernelY();
    auto group  = common->group();
    auto oc     = common->outputCount();
    auto ic     = size / kw / kh / (oc / group);
    if (depthwise) {
        return weightForDepthwise<float, metal_float>(context, group, kh, kw,
                                                      qnt ? qnt->weightFloat.get() : deconv->weight()->data());
    } else {
        return weightForDeconv<float, metal_float>(context, group, oc, ic, kh, kw,
                                                   qnt ? qnt->weightFloat.get() : deconv->weight()->data());
    }
}

static id<MTLBuffer> biasForDeconv(MNNMetalContext *context, const Convolution2D *deconv) {
    auto bias = deconv->bias();
    if (!bias || bias->size() == 0)
        return [context newDeviceBuffer:0 access:CPUTransparent];

    auto oc     = deconv->common()->outputCount();
    auto buffer = [context newDeviceBuffer:UP_DIV(oc, 4) * 4 * sizeof(metal_float) access:CPUWriteOnly];
    auto src    = bias->data();
    auto dst    = (metal_float *)buffer.contents;
#pragma clang loop vectorize(enable) unroll(enable)
    for (int i = 0; i < oc; i++)
        dst[i] = src[i];
    return buffer;
}

MetalDeconvolution::MetalDeconvolution(Backend *backend, const MNN::Op *op) : Execution(backend) {
    auto context = (__bridge MNNMetalContext *)static_cast<MetalBackend *>(backend)->context();
    auto deconv  = op->main_as_Convolution2D();
    auto common  = deconv->common();
    mDepthwise   = op->type() == MNN::OpType_DeconvolutionDepthwise;
    mGroup       = common->group();
    mKernelX     = common->kernelX();
    mKernelY     = common->kernelY();
    mPadMode     = common->padMode();
    mPadX        = common->padX();
    mPadY        = common->padY();
    mStrideX     = common->strideX();
    mStrideY     = common->strideY();
    mDilateX     = common->dilateX();
    mDilateY     = common->dilateY();
    // forcy downgrade to float like what CPU does
    std::shared_ptr<ConvolutionIntFactory::Int8Common> qnt = NULL;
    if (deconv->quanParameter()) {
        qnt = ConvolutionIntFactory::load(deconv->quanParameter(), true);
    }
    mWeight = weightForDeconv(context, mDepthwise, deconv, qnt.get());
    mBias   = biasForDeconv(context, deconv);
}

ErrorCode MetalDeconvolution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto input = inputs[0], output = outputs[0];
    int iw = input->width(), ih = input->height(), iz = UP_DIV(input->channel(), 4);
    int ow = output->width(), oh = output->height(), oz = UP_DIV(output->channel(), 4);

    // pad mode support
    int padX = mPadX, padY = mPadY;
    if (mPadMode == PadMode_SAME) {
        int pw = (iw - 1) * mStrideX + mKernelX - ow;
        int ph = (ih - 1) * mStrideY + mKernelY - oh;
        padX   = pw / 2;
        padY   = ph / 2;
    }

    // const buffer
    auto deltaKy = leastCommonMultiple(mDilateY, mStrideY) / mDilateY;
    auto deltaKx = leastCommonMultiple(mDilateX, mStrideX) / mDilateX;
    
    int consts[] = {
        iw,
        ih,
        iw * ih,
        iz,
        ow,
        oh,
        ow * oh,
        oz,
        mKernelX,
        mKernelY,
        mKernelX * mKernelY,
        mStrideX,
        mStrideY,
        padX,
        padY,
        mDilateX,
        mDilateY,
        deltaKy,
        deltaKx,
        deltaKy * mDilateY / mStrideY,
        deltaKx * mDilateX / mStrideX,
        mBias.length > 0,
    };
    mConstBuffer = [context newDeviceBuffer:sizeof(consts) bytes:consts access:CPUWriteOnly];
    return NO_ERROR;
}

ErrorCode MetalDeconvolution::onDepthwise(const Tensor *input, const Tensor *output) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    int unit     = sizeof(metal_float);
    auto iw = input->width(), ih = input->height(), iz = UP_DIV(input->channel(), 4), ib = iw * ih * iz * 4 * unit;
    auto ow = output->width(), oh = output->height(), oz = UP_DIV(output->channel(), 4), ob = ow * oh * oz * 4 * unit;

    // run
    auto encoder   = [context encoder];
    auto bandwidth = [context load:@"deconv_depthwise" encoder:encoder];
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
ErrorCode MetalDeconvolution::onDeconv(const Tensor *input, const Tensor *output) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    int ow = output->width(), oh = output->height(), oc = output->channel(), oz = UP_DIV(oc, 4), ob = output->batch();

    // run
    auto encoder   = [context encoder];
    auto bandwidth = [context load:@"deconv" encoder:encoder];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)input->deviceId() offset:0 atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output->deviceId() offset:0 atIndex:1];
    [encoder setBuffer:mConstBuffer offset:0 atIndex:2];
    [encoder setBuffer:mWeight offset:0 atIndex:3];
    [encoder setBuffer:mBias offset:0 atIndex:4];
    [context dispatchEncoder:encoder
                     threads:{ (NSUInteger) ow, (NSUInteger)oh, (NSUInteger)oz * ob }
                   bandwidth:bandwidth];
    [encoder endEncoding];
    MNN_PRINT_ENCODER(context, encoder);
    return NO_ERROR;
}

ErrorCode MetalDeconvolution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input = inputs[0], output = outputs[0];
    if (mDepthwise) {
        return onDepthwise(input, output);
    } else {
        return onDeconv(input, output);
    }
}

class MetalDeconvolutionCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend) const {
        return new MetalDeconvolution(backend, op);
    }
};
REGISTER_METAL_OP_CREATOR(MetalDeconvolutionCreator, OpType_Deconvolution);
REGISTER_METAL_OP_CREATOR(MetalDeconvolutionCreator, OpType_DeconvolutionDepthwise);
} // namespace MNN
#endif /* MNN_METAL_ENABLED */
