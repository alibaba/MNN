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
    
static const char* gWeightTrans = R"metal(
#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;
struct weight_shape {
    int group;
    int goc;
    int goc_4;
    int gic;
    int gic_4;
    int kh;
    int kw;
};
kernel void int4_weight_transform_fast(const device uint8_t* src [[buffer(0)]],
    device uint8_t* dst [[buffer(1)]],
    constant weight_shape &uConstant [[buffer(2)]], 
    uint3 gid [[thread_position_in_grid]]
) {
    if ((int)gid.x < uConstant.goc && (int)gid.y < uConstant.gic) {
        auto zo = gid.x / 4, ro = gid.x % 4;
        auto zi = gid.y / 4, ri = gid.y % 4;

        dst[((zo * uConstant.gic_4 + zi) * 16 + ro * 4 + ri) / 2] = src[(gid.x * uConstant.gic + gid.y) / 2];
    }
}

kernel void int4_weight_transform_c4_fast(const device uint16_t* src [[buffer(0)]],
    device uint16_t* dst [[buffer(1)]],
    constant weight_shape &uConstant [[buffer(2)]], 
    uint3 gid [[thread_position_in_grid]]
) {
    if ((int)gid.x < uConstant.goc && (int)gid.y < uConstant.gic_4) {
        auto zo = gid.x / 4, ro = gid.x % 4;

        dst[(zo * uConstant.gic_4 + gid.y) * 4 + ro] = src[gid.x * uConstant.gic_4 + gid.y];
    }
}

kernel void weight_transform_common(const device IType* src [[buffer(0)]],
    device OType* dst [[buffer(1)]],
    constant weight_shape &uConstant [[buffer(2)]], 
    uint3 gid [[thread_position_in_grid]]
) {
    if ((int)gid.x < uConstant.group * uConstant.goc && (int)gid.y < uConstant.gic && (int)gid.z < uConstant.kh * uConstant.kw) {

        auto g = gid.x / uConstant.goc;
        auto goc = gid.x % uConstant.goc;
        auto zo = goc / 4, ro = goc % 4;
        auto zi = gid.y / 4, ri = gid.y % 4;
        auto h = gid.z / uConstant.kw;
        auto w = gid.z % uConstant.kw;

        // to   [g][o/4][i/4][h][w][16]
        // from [g][o][i][h][w]
        int dx = g * uConstant.goc_4 * uConstant.gic_4 * uConstant.kh * uConstant.kw * 16 + zo * uConstant.gic_4 * uConstant.kh * uConstant.kw * 16 + ro * 4 + zi * uConstant.kh * uConstant.kw * 16 + ri + (h * uConstant.kw + w) * 16;
        int sx = (gid.x * uConstant.gic + gid.y) * uConstant.kh * uConstant.kw + gid.z;

        dst[dx] = (OType)src[sx];
    }
}
)metal";
    
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
    void MetalConvolutionCommon::convertWeightFormat(int group, int oc, int ic, int kh, int kw, const FType *src, TType* dstOrigion, Tensor* dstTensor, id<MTLBuffer> srcGpuBuffer) {
    auto goc    = oc / group;
    auto gic    = ic / group;
    auto goc_4  = UP_DIV(goc, 4);
    auto gic_4  = UP_DIV(gic, 4);

    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)static_cast<MetalBackend *>(backend)->context();
    if(srcGpuBuffer == nil) {
        srcGpuBuffer  = [context newDeviceBuffer:group * goc * gic * kh * kw * sizeof(FType) access:CPUReadWrite];
        ::memcpy((void *)srcGpuBuffer.contents, (void *)src, group * goc * gic * kh * kw * sizeof(FType));
    }
    std::string IType = "float";
    std::string OType = "float";
    if(std::is_same<FType, int8_t>::value) {
        IType = "int8_t";
    }
    if(std::is_same<TType, __fp16>::value) {
        OType = "half";
    } else if(std::is_same<TType, int8_t>::value) {
        OType = "int8_t";
    }
    MTLCompileOptions *option = [[MTLCompileOptions alloc] init];
    auto dic = [NSMutableDictionary dictionaryWithCapacity:0];
    [dic setValue:@(IType.c_str()) forKey:@"IType"];
    [dic setValue:@(OType.c_str()) forKey:@"OType"];
    option.preprocessorMacros = dic;
    // create const buffer
    int constants[] = {group,
                       goc,
                       goc_4,
                       gic,
                       gic_4,
                       kh,
                       kw};
    auto constBuffer = backend->getConstBuffer(sizeof(constants));
    ::memcpy(constBuffer.contents, constants, sizeof(constants));
    
    auto encoder = [backend->getCommandBufferForBufferCopy() computeCommandEncoder];
    auto pipeline = backend->makeComputePipelineWithSourceOption(gWeightTrans, "weight_transform_common", option);
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:srcGpuBuffer offset:0 atIndex:0];
    MetalBackend::setTensor(dstTensor, encoder, 1);
    [encoder setBuffer:constBuffer offset:0 atIndex:2];
    auto gl = [context computeBestGroupAndLocal:pipeline threads:MTLSizeMake((NSInteger)group * goc, (NSInteger)gic, (NSInteger)kh * kw)];
    [encoder dispatchThreadgroups:gl.first threadsPerThreadgroup:gl.second];
    [encoder endEncoding];
    // just commit, don‘t wait for not block
    backend->commit();
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
        if(asymmetric) {
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
        } else {
            for (int z=0; z<oc; ++z) {
                auto srcZ = scale + z * blockSize;
                for (int bi=0; bi<blockSize; ++bi) {
                    float s = srcZ[bi];
                    if(s > max_data) {
                        max_data = s;
                    }
                }
            }
        }
        // too big scale may cause half precision loss
        coef = 1000.0 / max_data;
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
    auto common = conv->common();
    auto kw     = common->kernelX();
    auto kh     = common->kernelY();
    auto group  = common->group();
    auto oc     = common->outputCount();
    int ic     = common->inputCount();
    
    void* weightMemPtr = nullptr;
    id<MTLBuffer> srcGpuBuffer = nil;
    bool preAllocGpuMem = ic != 0 && conv->quanParameter();
    int quantBit;
    // only for weight int4/int8 now.
    if(loadWeightInt8) {
        quantBit = conv->quanParameter()->aMaxOrBits();
        // 3.1.2 and after has aMaxOrBits for quant bits
        if (quantBit == 0) {
            // support old model for external weight file with int4/int8 quant
            quantBit = ConvolutionCommon::getQuantBitFromExternalFile(op);
        }
        if(quantBit != 4 && quantBit != 8) {
            preAllocGpuMem = false;
        }
    }
    if (preAllocGpuMem) {
        size_t size = oc * ic * kh * kw / group;
        
        if (loadWeightInt8) {
            if(quantBit == 4) {
                size = UP_DIV(size, 2);
            }
        } else {
            size *= sizeof(float);
        }

        auto backend = static_cast<MetalBackend *>(this->backend());
        auto context = (__bridge MNNMetalContext *)static_cast<MetalBackend *>(backend)->context();
        
        srcGpuBuffer = [context newDeviceBuffer:size access:CPUReadWrite];
    }

    std::shared_ptr<ConvolutionCommon::Int8Common> qnt = NULL;
    if (loadWeightInt8) {
        qnt = ConvolutionCommon::load(op, backend(), false, true, (void *)srcGpuBuffer.contents);
    } else if (conv->quanParameter()) {
        qnt = ConvolutionCommon::load(op, backend(), true, false, (void *)srcGpuBuffer.contents);
    }
    // param
    auto size   = qnt ? MAX(qnt->weight.size(), qnt->weightFloat.size()) : conv->weight()->size();
    if (0 == ic) {
        ic = size / kw / kh / (oc / group);
    }
    // convert
    if (loadWeightInt8 && qnt->weight.get() != nullptr) {
        auto backend = static_cast<MetalBackend *>(this->backend());
        mWeight = weightTransform(group, oc, ic, kh, kw, (float*)qnt->weight.get(), !qnt->canUseInt4, qnt->canUseInt4, srcGpuBuffer);
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
    } else if (qnt && qnt->weightFloat.get()) {
        mWeight = weightTransform(group, oc, ic, kh, kw, qnt->weightFloat.get(), false, false, srcGpuBuffer);
    } else {
        mWeight = weightTransform(group, oc, ic, kh, kw, conv->weight()->data(), false, false, srcGpuBuffer);
    }
}

std::shared_ptr<MNN::Tensor> MetalConvolutionCommon::weightTransform(int group, int oc, int ic, int kh, int kw, const float *src, bool int8Weight, bool int4Weight, id<MTLBuffer> srcGpuBuffer) {
    if(srcGpuBuffer != nil) {
        MNN_ASSERT((void*)src == (void*)srcGpuBuffer.contents);
    }
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)static_cast<MetalBackend *>(backend)->context();
    auto goc    = oc / group;
    auto gic    = ic / group;
    auto goc_4  = UP_DIV(goc, 4);
    auto gic_4  = UP_DIV(gic, 4);
    auto weight_len = group * ROUND_UP(goc_4, 4) * gic_4 * kw * kh * 16;
    auto ori_len = group * goc * gic * kh * kw;
    bool needMemset = (goc % 4 != 0 || gic % 4 != 0);
#ifdef MNN_LOW_MEMORY
    if (int4Weight) {
        weight_len = UP_DIV(weight_len, 2);
        std::shared_ptr<MNN::Tensor> weightLow(MNN::Tensor::createDevice<int8_t>({weight_len}));
        auto res = backend->onAcquireBuffer(weightLow.get(), Backend::STATIC);
        if (!res) {
            MNN_ERROR("Memory alloc error!\n");
            return nullptr;
        }
        
        auto buf = MetalBackend::getBuffer(weightLow.get());
        auto dstPtr = (uint8_t*)[buf.first contents] + buf.second;
        if(needMemset) {
            ::memset(dstPtr, 0, weight_len);
        }
        bool fastBlit = (group == 1 && kh == 1 && kw == 1 && ic % 2 == 0);
        
        auto oc_4  = UP_DIV(oc, 4);
        auto ic_4  = UP_DIV(ic, 4);
        // fast int4 reorder
        if (fastBlit) {
            if(srcGpuBuffer == nil) {
                srcGpuBuffer  = [context newDeviceBuffer:UP_DIV(ori_len, 2) access:CPUReadWrite];
                ::memcpy((void *)srcGpuBuffer.contents, (void *)src, UP_DIV(ori_len, 2));
            }

            MTLCompileOptions *option = [[MTLCompileOptions alloc] init];
            auto dic = [NSMutableDictionary dictionaryWithCapacity:0];
            [dic setValue:@"uint8_t" forKey:@"IType"];
            [dic setValue:@"uint8_t" forKey:@"OType"];
            option.preprocessorMacros = dic;
            // create const buffer
            int constants[] = {group,
                               goc,
                               goc_4,
                               gic,
                               gic_4,
                               kh,
                               kw};
            auto constBuffer = backend->getConstBuffer(sizeof(constants));
            ::memcpy(constBuffer.contents, constants, sizeof(constants));
            
            auto encoder = [backend->getCommandBufferForBufferCopy() computeCommandEncoder];
            id<MTLComputePipelineState> pipeline;
            bool c4_fast = ic % 4 == 0;
            if(c4_fast) {
                pipeline = backend->makeComputePipelineWithSourceOption(gWeightTrans, "int4_weight_transform_c4_fast", option);
            } else {
                pipeline = backend->makeComputePipelineWithSourceOption(gWeightTrans, "int4_weight_transform_fast", option);
            }
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:srcGpuBuffer offset:0 atIndex:0];
            MetalBackend::setTensor(weightLow.get(), encoder, 1);
            [encoder setBuffer:constBuffer offset:0 atIndex:2];
            MTLSize totalThread;
            if(c4_fast) {
                totalThread = MTLSizeMake((NSInteger)goc, (NSInteger)gic_4, (NSInteger)1);
            } else {
                totalThread = MTLSizeMake((NSInteger)goc, (NSInteger)gic, (NSInteger)1);
            }
            auto gl = [context computeBestGroupAndLocal:pipeline threads:totalThread];
            [encoder dispatchThreadgroups:gl.first threadsPerThreadgroup:gl.second];
            [encoder endEncoding];
            // just commit, don‘t wait for not block
            backend->commit();
        } else {
            auto srcPtr = (int8_t*)src;
            // slow int4 reorder
            int sx = 0;
            auto goc_4  = UP_DIV(goc, 4);
            auto gic_4  = UP_DIV(gic, 4);
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
#endif
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
        if(needMemset) {
            ::memset(dst, 0, weight_len);
        }
        convertWeightFormat<int8_t, int8_t>(group, oc, ic, kh, kw, (const int8_t*)src, (int8_t *)dst, t.get(), srcGpuBuffer);
    } else if (backend->useFp16InsteadFp32()) {
        if(needMemset) {
            ::memset(dst, 0, weight_len * sizeof(__fp16));
        }
        convertWeightFormat<float, __fp16>(group, oc, ic, kh, kw, (const float*)src, (__fp16 *)dst, t.get(), srcGpuBuffer);
    } else {
        if(needMemset) {
            ::memset(dst, 0, weight_len * sizeof(float));
        }
        convertWeightFormat<float, float>(group, oc, ic, kh, kw, (const float*)src, (float *)dst, t.get(), srcGpuBuffer);
    }

    return t;
}

} // namespace MNN

#endif
