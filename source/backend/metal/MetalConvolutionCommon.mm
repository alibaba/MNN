//
//  MetalConvolutionCommon.mm
//  MNN
//
//  Created by MNN on 2019/02/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "backend/metal/MetalConvolutionCommon.hpp"
#import "core/Macro.h"
#import "backend/metal/MetalBackend.hpp"
#import "backend/metal/MetalConvolution1x1.hpp"
#import "backend/metal/MetalConvolutionWinograd.hpp"
#import "core/TensorUtils.hpp"

#if MNN_METAL_ENABLED
namespace MNN {

static std::shared_ptr<MNN::Tensor> biasForConv(Backend *bn, const Convolution2D *conv, bool fp16) {
    auto bias   = conv->bias();
    auto oc     = conv->common()->outputCount();
    int bytes = 4;
    if (fp16) {
        bytes = 2;
    }
    auto bias_size_unit = UP_DIV(oc, 16) * 16;
    std::shared_ptr<MNN::Tensor> t(MNN::Tensor::createDevice<float>({bias_size_unit}));
    auto res = bn->onAcquireBuffer(t.get(), Backend::STATIC);
    if (!res) {
        return nullptr;
    }
    auto bias_size = bias_size_unit *bytes;
    auto buffer = MetalBackend::getBuffer(t.get());
    auto src    = bias->data();
    auto dstOrigin = (uint8_t*)[buffer.first contents] + buffer.second;
    ::memset(dstOrigin, 0, bias_size);
    if (fp16) {
        auto dst    = (__fp16 *)dstOrigin;
    #pragma clang loop vectorize(enable) unroll(enable)
        for (int i = 0; i < oc; i++) {
            dst[i] = src[i];
        }
    } else {
        ::memcpy(dstOrigin, src, oc * sizeof(float));
    }
    return t;
}

MetalConvolutionCommon::MetalConvolutionCommon(Backend *backend, const MNN::Op *op, std::shared_ptr<MNN::Tensor> bias) : MetalExecution(backend) {
    auto mtbn = static_cast<MetalBackend*>(backend);
    auto conv       = op->main_as_Convolution2D();
    auto common     = conv->common();
    mOp             = op;
    mKernelX        = common->kernelX();
    mKernelY        = common->kernelY();
    mStrideX        = common->strideX();
    mStrideY        = common->strideY();
    mDilateX        = common->dilateX();
    mDilateY        = common->dilateY();
    if (nullptr != bias) {
        mBias = bias;
    } else {
        mBias = biasForConv(backend, conv, mtbn->useFp16InsteadFp32());
    }
    mActivationType = common->relu() ? 1 : (common->relu6() ? 2 : 0);
    if (nullptr == mBias) {
        mValid = false;
    }
}

template <typename FType, typename TType>
void weightInBlock(int group, int oc, int ic, int kh, int kw, const FType *src, uint8_t* dstOrigion) {
    auto goc    = oc / group;
    auto gic    = ic / group;
    auto goc_4  = UP_DIV(goc, 4);
    auto gic_4  = UP_DIV(gic, 4);
    TType* dst = (TType*)dstOrigion;
    auto weight_len = group * ROUND_UP(goc_4, 4) * gic_4 * kw * kh * 16 * sizeof(TType);
    ::memset(dst, 0, weight_len);
    for (int g = 0; g < group; g++) {
        auto g_dst = dst + g * goc_4 * gic_4 * kh * kw * 16; // g
        for (int o = 0; o < goc; o++) {
            auto zo = o / 4, ro = o % 4;
            auto o_dst = g_dst + zo * gic_4 * kh * kw * 16 + ro * 4; // o/4 x 4
            for (int i = 0; i < gic; i++) {
                auto zi = i / 4, ri = i % 4;
                auto i_dst = o_dst + zi * kh * kw * 16 + ri; // i/4 x 4
                for (int h = 0; h < kh; h++) {
                    for (int w = 0; w < kw; w++) {
                        // to   [g][o/4][i/4][h][w][16]
                        // from [g][o][i][h][w]
                        i_dst[(h * kw + w) * 16] = *src++;
                    }
                }
            }
        }
    }
}

template<typename DType>
static std::pair<std::shared_ptr<MNN::Tensor>, float> getDequantScale(const float* scale, int size, MetalBackend *backend, bool asymmetric, int oc) {
    int totalCount = 0;
    if (asymmetric) {
        totalCount = size / 2;
    } else {
        totalCount = size;
    }
    int blockSize = totalCount / oc;
    int alignOutputCount = ALIGN_UP4(oc);
    std::shared_ptr<MNN::Tensor> dequantScale(MNN::Tensor::createDevice<uint8_t>({alignOutputCount, blockSize, (int)(sizeof(DType) * 2)}));
    bool res = backend->onAcquireBuffer(dequantScale.get(), Backend::STATIC);
    if (!res) {
        MNN_ERROR("Buffer allocated error!\n");
        return std::make_pair(nullptr, 1.0);
    }
    auto buffer0 = MetalBackend::getBuffer(dequantScale.get());
    DType* dst_scale = (DType*)((uint8_t*)[buffer0.first contents] + buffer0.second);
    ::memset(dst_scale, 0, dequantScale->usize());
    
    float coef = 1.0;
    if(std::is_same<DType, __fp16>::value) {
        float max_data = 0.0;
        for (int z=0; z<oc; ++z) {
            auto srcZ = scale + z * blockSize * 2;
            for (int bi=0; bi<blockSize; ++bi) {
                float s = fabs(srcZ[2*bi+1]);
                float b = fabs(srcZ[2*bi+0]);
                float temp = ALIMAX(s, b);
                if(temp > max_data) {
                    max_data = temp;
                }
            }
        }
        coef = 65504.0 / max_data;
    }
    if (asymmetric) {
        for (int z=0; z<oc; ++z) {
            int zo = z / 4;
            int zi = z % 4;
            auto srcZ = scale + z * blockSize * 2;
            auto dstSZ = dst_scale + zo * blockSize * 8 + zi;
            auto dstBZ = dst_scale + zo * blockSize * 8 + zi + 4;
            for (int bi=0; bi<blockSize; ++bi) {
                float s = srcZ[2*bi+1];
                float b = srcZ[2*bi+0];
                dstSZ[bi * 8] = (DType)(s * coef);
                dstBZ[bi * 8] = (DType)(b * coef);
            }
        }
    } else {
        for (int z=0; z<oc; ++z) {
            int zo = z / 4;
            int zi = z % 4;
            auto srcZ = scale + z * blockSize;
            auto dstSZ = dst_scale + zo * blockSize * 8 + zi;
            auto dstBZ = dst_scale + zo * blockSize * 8 + zi + 4;
            for (int bi=0; bi<blockSize; ++bi) {
                float s = srcZ[bi];
                float b = 0.0f;
                dstSZ[bi * 8] = (DType)(s * coef);
                dstBZ[bi * 8] = b;
            }
        }
    }
    return std::make_pair(dequantScale, coef);
}
void MetalConvolutionCommon::loadWeight(const MNN::Op *op, bool loadWeightInt8) {
    auto conv = op->main_as_Convolution2D();
    std::shared_ptr<ConvolutionCommon::Int8Common> qnt = NULL;
    if (loadWeightInt8) {
        qnt = ConvolutionCommon::load(op, backend(), false, true);
    } else if (conv->quanParameter()) {
        qnt = ConvolutionCommon::load(op, backend(), true);
    }
    // param
    auto size   = qnt ? MAX(qnt->weight.size(), qnt->weightFloat.size()) : conv->weight()->size();
    auto common = conv->common();
    auto kw     = common->kernelX();
    auto kh     = common->kernelY();
    auto group  = common->group();
    auto oc     = common->outputCount();
    int ic     = common->inputCount();
    if (0 == ic) {
        ic = size / kw / kh / (oc / group);
    }

    // convert
    if (loadWeightInt8 && qnt->weight.get() != nullptr) {
        auto backend = static_cast<MetalBackend *>(this->backend());
        mWeight = weightTransform(group, oc, ic, kh, kw, (float*)qnt->weight.get(), !qnt->canUseInt4, qnt->canUseInt4);
        if(backend->useFp16InsteadFp32()) {
            auto dequantParams = getDequantScale<__fp16>(qnt->alpha.get(), qnt->alpha.size(), backend, qnt->asymmetric, oc);
            mDequantScaleBias = dequantParams.first;
            mScaleCoef = dequantParams.second;
        } else {
            auto dequantParams = getDequantScale<float>(qnt->alpha.get(), qnt->alpha.size(), backend, qnt->asymmetric, oc);
            mDequantScaleBias = dequantParams.first;
            mScaleCoef = dequantParams.second;
        }

        mDequantBits = qnt->canUseInt4 ? 4:8;
    } else if (qnt && qnt->weightFloat.size() > 0) {
        mWeight = weightTransform(group, oc, ic, kh, kw, qnt->weightFloat.get(), false, false);
    } else {
        mWeight = weightTransform(group, oc, ic, kh, kw, conv->weight()->data(), false, false);
    }
}


std::shared_ptr<MNN::Tensor> MetalConvolutionCommon::weightTransform(int group, int oc, int ic, int kh, int kw, const float *src, bool int8Weight, bool int4Weight) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)static_cast<MetalBackend *>(backend)->context();
    auto goc    = oc / group;
    auto gic    = ic / group;
    auto goc_4  = UP_DIV(goc, 4);
    auto gic_4  = UP_DIV(gic, 4);
    auto weight_len = group * ROUND_UP(goc_4, 4) * gic_4 * kw * kh * 16;

    if (int4Weight) {
        weight_len = UP_DIV(weight_len, 2);
        std::shared_ptr<MNN::Tensor> weightLow(MNN::Tensor::createDevice<int8_t>({weight_len}));
        auto res = backend->onAcquireBuffer(weightLow.get(), Backend::STATIC);
        if (!res) {
            MNN_ERROR("Memory alloc error!\n");
            return nullptr;
        }
        auto srcPtr = (int8_t*)src;
        auto buf = MetalBackend::getBuffer(weightLow.get());
        auto dstPtr = (uint8_t*)[buf.first contents] + buf.second;
        auto oc_4  = UP_DIV(oc, 4);
        auto ic_4  = UP_DIV(ic, 4);
        if (group == 1 && kh == 1 && kw == 1 && ic % 2 == 0) {
            // fast int4 reorder
            for (int i = 0; i < oc; i++) {
                auto zo = i / 4, ro = i % 4;
                for (int j = 0; j < ic; j++) {
                    auto zi = j / 4, ri = j % 4;
                    // [co, ci] -> [co/4, ci/4, co_4, ci_4]
                    dstPtr[((zo * ic_4 + zi) * 16 + ro * 4 + ri) / 2] = srcPtr[(i * ic + j) / 2];
                }
            }
        } else {
            // slow int4 reorder
            int sx = 0;
            auto goc_4  = UP_DIV(goc, 4);
            auto gic_4  = UP_DIV(gic, 4);
            ::memset(dstPtr, 0, weight_len);
            for (int g = 0; g < group; g++) {
                for (int o = 0; o < goc; o++) {
                    auto zo = o / 4, ro = o % 4;
                    for (int i = 0; i < gic; i++) {
                        auto zi = i / 4, ri = i % 4;
                        for (int h = 0; h < kh; h++) {
                            for (int w = 0; w < kw; w++) {
                                // to   [g][o/4][i/4][h][w][16]
                                // from [g][o][i][h][w]
                                int dx = g * goc_4 * gic_4 * kh * kw * 16 + zo * gic_4 * kh * kw * 16 + ro * 4 + zi * kh * kw * 16 + ri + (h * kw + w) * 16;
                                uint8_t s = srcPtr[sx/2];
                                s = (sx % 2) ? (s & 0xf) : (s >> 4);
                                s = (dx % 2) ? s : (s << 4);
                                dstPtr[dx/2] |= s;
                                sx++;
                            }
                        }
                    }
                }
            }
        }
        return weightLow;
    }

    std::shared_ptr<MNN::Tensor> t(MNN::Tensor::createDevice<float>({weight_len}));
    if (int8Weight || int4Weight) {
        t.reset(MNN::Tensor::createDevice<int8_t>({weight_len}));
    }
    bool res = backend->onAcquireBuffer(t.get(), Backend::STATIC);
    if (!res) {
        return nullptr;
    }
    auto buffer = MetalBackend::getBuffer(t.get());
    auto dst = (uint8_t*)[buffer.first contents] + buffer.second;
    if (int8Weight) {
        weightInBlock<int8_t, int8_t>(group, oc, ic, kh, kw, (int8_t*)src, dst);
    } else if (backend->useFp16InsteadFp32()) {
        weightInBlock<float, __fp16>(group, oc, ic, kh, kw, src, dst);
    } else {
        weightInBlock<float, float>(group, oc, ic, kh, kw, src, dst);
    }

    return t;
}

} // namespace MNN

#endif
