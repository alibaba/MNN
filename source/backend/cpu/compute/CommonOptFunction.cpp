//
//  CommonOptFunction.cpp
//  MNN
//
//  Created by MNN on 2018/09/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CommonOptFunction.h"
#include "ConvOpt.h"
#include "WinogradOptFunction.hpp"
#include "Int8FunctionsOpt.h"
#include "ImageProcessFunction.hpp"
#include <string.h>
#include <algorithm>
#include <cmath>
#include <math.h>
#include "math/Vec.hpp"
#include <vector>
#include "../CPURuntime.hpp"
#include "core/MemoryFormater.h"
// TODO: Find better way to optimize it
#include "../CPUBinary.hpp"
#include "../CPUUnary.hpp"
#include "../CPUPool.hpp"
#define PACK 4
#define FLOAT float
using Vec = MNN::Math::Vec<float, 4>;
#include "../GridSampler.hpp"

#ifndef MNN_USE_SSE
void MNNInt8ToInt16(int16_t* dest, const int8_t* source, size_t count) {
    // Should not be called
    MNN_ASSERT(false);
}
#endif

#ifdef MNN_LOW_MEMORY
#ifndef __aarch64__
static void _MNNPackedMatMulRemain_int4(float* C, const float* A, const float* fB, size_t eSize, const size_t* parameter, const float* postParameters, const float* bias, int aStride, const float* k, const float* b) {
    auto B = reinterpret_cast<const uint8_t*>(fB);
    auto h = parameter[2];
    auto l = parameter[1];
    auto cStride = parameter[3] / sizeof(float);
    auto hRemain = parameter[4];
    float weightBytes = 0.5; // sizeof(int4_t)
    auto bExtraStride = static_cast<int32_t>(parameter[5] / weightBytes);
    auto bStride = bExtraStride + 4 * l;
    auto hC4 = UP_DIV(h, 4);
    float minValue = -std::numeric_limits<float>().max();
    float maxValue = std::numeric_limits<float>().max();
    if (nullptr != postParameters) {
        minValue = postParameters[2];
        maxValue = postParameters[3];
    }
    int blockId = parameter[6];

    for (int x=0; x<eSize; ++x) {
        auto dst = C + 4 * x;
        auto src = A + x;
        for (int y=0; y<hC4; ++y) {
            auto dstY = dst + y * cStride;
            auto weight = B + y * bStride / 2;
            auto alpha = k + y * 4;
            auto qbias  = b + y * 4;
            float summer[4] = {
                0.0f,
                0.0f,
                0.0f,
                0.0f,
            };
            if (blockId > 0) {
                summer[0] = dstY[0];
                summer[1] = dstY[1];
                summer[2] = dstY[2];
                summer[3] = dstY[3];
            }
            if (nullptr != bias && nullptr != postParameters) {
                for (int v=0; v<4; ++v) {
                    summer[v] += bias[4 * y + v];
                }
            }
            for (int z=0; z<l; ++z) {
                auto aZ = src + z * aStride;
                auto i4wZ = weight + z * 2;
                float wZ[4];
                {
                    auto w01    = i4wZ[0];
                    auto w23    = i4wZ[1];
                    int iw01    = w01;
                    int iw23    = w23;
                    int iw0     = iw01 / 16;
                    int iw1     = iw01 % 16;
                    int iw2     = iw23 / 16;
                    int iw3     = iw23 % 16;
                    wZ[0]       = iw0 * alpha[0] + qbias[0];
                    wZ[1]       = iw1 * alpha[1] + qbias[1];
                    wZ[2]       = iw2 * alpha[2] + qbias[2];
                    wZ[3]       = iw3 * alpha[3] + qbias[3];
                }
                summer[0] += wZ[0] * aZ[0];
                summer[1] += wZ[1] * aZ[0];
                summer[2] += wZ[2] * aZ[0];
                summer[3] += wZ[3] * aZ[0];
            }
            for (int v=0; v<4; ++v) {
                auto dstValue = std::min(summer[v], maxValue);
                dstValue = std::max(dstValue, minValue);
                dstY[v] = dstValue;
            }
        }
    }
}
static void _MNNPackedMatMulRemain_int8(float* C, const float* A, const float* fB, size_t eSize, const size_t* parameter, const float* postParameters, const float* bias, int aStride, const float* k, const float* b) {
    auto B = reinterpret_cast<const int8_t*>(fB);
    auto h = parameter[2];
    auto l = parameter[1];
    auto cStride = parameter[3] / sizeof(float);
    auto hRemain = parameter[4];
    float weightBytes = 1; // sizeof(int8_t)
    auto bExtraStride = static_cast<int32_t>(parameter[5] / weightBytes);
    auto bStride = bExtraStride + 4 * l;
    auto hC4 = UP_DIV(h, 4);
    float minValue = -std::numeric_limits<float>().max();
    float maxValue = std::numeric_limits<float>().max();
    if (nullptr != postParameters) {
        minValue = postParameters[2];
        maxValue = postParameters[3];
    }
    int blockId = parameter[6];

    for (int x=0; x<eSize; ++x) {
        auto dst = C + 4 * x;
        auto src = A + x;
        for (int y=0; y<hC4; ++y) {
            auto dstY = dst + y * cStride;
            auto weight = B + y * bStride;
            auto alpha = k + y * 4;
            auto qbias  = b + y * 4;
            float summer[4] = {
                0.0f,
                0.0f,
                0.0f,
                0.0f,
            };
            if (blockId > 0) {
                summer[0] = dstY[0];
                summer[1] = dstY[1];
                summer[2] = dstY[2];
                summer[3] = dstY[3];
            }
            if (nullptr != bias && nullptr != postParameters) {
                for (int v=0; v<4; ++v) {
                    summer[v] += bias[4 * y + v];
                }
            }
            for (int z=0; z<l; ++z) {
                auto aZ = src + z * aStride;
                auto i8wZ = weight + z * 4;
                float wZ[4];
                {
                    wZ[0]       = i8wZ[0] * alpha[0] + qbias[0];
                    wZ[1]       = i8wZ[1] * alpha[1] + qbias[1];
                    wZ[2]       = i8wZ[2] * alpha[2] + qbias[2];
                    wZ[3]       = i8wZ[3] * alpha[3] + qbias[3];
                }
                summer[0] += wZ[0] * aZ[0];
                summer[1] += wZ[1] * aZ[0];
                summer[2] += wZ[2] * aZ[0];
                summer[3] += wZ[3] * aZ[0];
            }
            for (int v=0; v<4; ++v) {
                auto dstValue = std::min(summer[v], maxValue);
                dstValue = std::max(dstValue, minValue);
                dstY[v] = dstValue;
            }
        }
    }
}
void MNNPackedMatMul_int4(float* C, const float* A, const float* B, const size_t* parameter, const float* postParameters, const float* bias, const float* k, const float* b) {
    _MNNPackedMatMulRemain_int4(C, A, B, 16, parameter, postParameters, bias, 16, k, b);
}
void MNNPackedMatMulRemain_int4(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter, const float* postParameters, const float* bias, const float* k, const float* b) {
    auto aStride = parameter[0] / sizeof(float);
    _MNNPackedMatMulRemain_int4(C, A, B, eSize, parameter, postParameters, bias, aStride, k, b);
}
void MNNPackedMatMul_int8(float* C, const float* A, const float* B, const size_t* parameter, const float* postParameters, const float* bias, const float* k, const float* b) {
    _MNNPackedMatMulRemain_int8(C, A, B, 16, parameter, postParameters, bias, 16, k, b);
}
void MNNPackedMatMulRemain_int8(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter, const float* postParameters, const float* bias, const float* k, const float* b) {
    auto aStride = parameter[0] / sizeof(float);
    _MNNPackedMatMulRemain_int8(C, A, B, eSize, parameter, postParameters, bias, aStride, k, b);
}
void MNNAbsMaxFP32(const float* source, float* absmax, size_t src_depth_quad, size_t realSize, int pack) {
    // source: (ic/4, N, 4)
    auto srcStep = pack * realSize;
    for (int i = 0; i < realSize; ++i) {
        float absmaxVal = 0.f; // absmaxVal>=0
        for (int c = 0; c < src_depth_quad; ++c) {
            auto src = source + c * srcStep + i * pack;
            for (int k = 0; k < pack; ++k) {
                absmaxVal = std::max(absmaxVal, std::abs(src[k]));
            }
        }
        absmax[i] = absmaxVal;
    }
}
void MNNQuantScaleFP32(float* absmax, float* quant_scale, float* dequant_scale, size_t thread, size_t batch) {
    for (int i = 0; i < batch; ++i) {
        auto absmaxPtr = absmax + i;
        float absVal = 0.f;
        for (int t = 0; t < thread; ++t) {
            absVal = std::max(absVal, absmaxPtr[t * batch]);
        }
        quant_scale[i] = 127.0f / absVal;
        dequant_scale[i] = absVal / 127.0f;
    }
}
void MNNQuantSumFP32(float* sum, const float* dequant_scale, size_t thread, size_t batch) {
    for (int i = 0; i < batch; ++i) {
        auto sumPtr = reinterpret_cast<int*>(sum) + i;
        int sumVal = 0.f;
        for (int t = 0; t < thread; ++t) {
            sumVal += sumPtr[t * batch];
        }
        sum[i] = sumVal * dequant_scale[i];
    }
}
void MNNDynamicQuantFP32(const float* src, int8_t* dst, const float* scale, size_t src_depth_quad, size_t realSize, int pack) {
#ifdef MNN_USE_SSE
    uint8_t* dstPtr = reinterpret_cast<uint8_t*>(dst);
    int offset = 128;
#else
    int8_t* dstPtr = dst;
    int offset = 0;
#endif
    for (int i = 0; i < realSize; ++i) {
        auto scaleVal = scale[i];
        for (int c = 0; c < src_depth_quad; ++c) {
            auto srcZ = src + c * pack * realSize + i * pack;
            auto dstZ = dstPtr + c * pack * realSize + i * pack;
            for (int k = 0; k < pack; ++k) {
                int val = (int)roundf(srcZ[k] * scaleVal);
                dstZ[k] = val + offset;
            }
        }
    }
}

void MNNDynamicUpdateConvBiasScale(float* newbias, float* newscale, float* oldbias, float* weightScale, float* inputScale, float* weightKernelSum, float* inputZero, size_t ocQuad, size_t scaleSize) {
    int ocUp4 = 4 * ocQuad;
    int pack = 4;
    int blockNum = scaleSize / ocUp4;
    for (int i = 0; i < ocUp4; ++i) {
        newbias[i] = oldbias[i] - weightKernelSum[i] * inputZero[0];
    }
    for (int k = 0; k < blockNum; ++k) {
        for (int i = 0; i < ocUp4; ++i) {
           newscale[i + k * ocUp4] = weightScale[i + k * ocUp4] * inputScale[0];
        }
    }
}

#endif // not __aarch64__
#endif // LOW_MEMORY


static void MNNSumByAxisLForMatmul_A(float* dest, int8_t* source, const float* scale, ssize_t realDstCount, SumByAxisParams sumParams) {
#ifdef MNN_USE_SSE
    uint8_t* srcInt8 = reinterpret_cast<uint8_t*>(source);
#else
    int8_t* srcInt8 = source;
#endif
    auto scalePtr = scale;
    auto kernelCountUnitDouble = sumParams.kernelCountUnitDouble;
    auto blockNum = sumParams.blockNum;
    auto EP = sumParams.DST_XUNIT;
    auto LP = sumParams.SRC_UNIT;
    auto blockSizeQuad = kernelCountUnitDouble / blockNum;
    auto col_buffer_unit_size = sumParams.col_buffer_unit_size;
    auto oneScale = sumParams.oneScale;
    do {
        int step = ALIMIN(EP, realDstCount);

        for (int k = 0; k < blockNum; ++k) {
            // const auto src_x   = srcInt8 + w * LP;
            const auto src_x = srcInt8 + k * (EP * LP * blockSizeQuad);
            for (int w = 0; w < step; ++w) {
                float dequantScale = scale[0];
                if (oneScale == 0) {
                    dequantScale = scalePtr[w];
                }
                int sumint32 = 0;
                const auto src_y = src_x + w * LP;
                for (int j = 0; j < blockSizeQuad; ++j) {
                    const auto src_z = src_y + j * (EP * LP);
                    for (int i = 0; i < LP; ++i) {
                        sumint32 += src_z[i];
                    }
                }
                dest[w + k * step] = dequantScale * static_cast<float>(sumint32);
            }
        }
        scalePtr += step;

        dest += (step * blockNum);
        realDstCount -= step;
        srcInt8 += col_buffer_unit_size;
    } while(realDstCount > 0); 
}

template<typename T>
void MNNPackC4Common(T* dst, const T* src, size_t area, size_t depth, int* areaOffset) {
    int depthC4     = depth / 4;
    int depthRemain = depthC4 * 4;
    int remain      = depth - depthRemain;
    int z, x, y;
    const T* srcChannel[4];
    const T* srcOffset = src;
    for(z = 0; z < depthC4; ++z) {
        auto dstZ = dst + z * areaOffset[1] * 4;
        for(y = 0; y < 4; ++y) {
            srcChannel[y] = srcOffset + areaOffset[0] * y;
        }
        for(x = 0; x < area; ++x) {
            for(y = 0; y < 4; ++y) {
                dstZ[0] = srcChannel[y][x];
                dstZ++;
            }
        }
        srcOffset += areaOffset[0] * 4;
    }
    if(remain > 0){
        auto dstZ = dst + depthC4 * areaOffset[1] * 4;
        for(y = 0; y < remain; ++y) {
            srcChannel[y] = srcOffset + areaOffset[0] * y;
        }
        for(x = 0; x < area; ++x) {
            for(y = 0; y < remain; ++y) {
                dstZ[0] = srcChannel[y][x];
                dstZ++;
            }
            for(y = remain; y < 4; ++y) {
                dstZ[0] = 0;
                dstZ++;
            }
        }
    }
}

template<typename T>
void MNNUnpackC4Common(T* dst, const T* src, size_t area, size_t depth, int* areaOffset) {
    int depthC4     = depth / 4;
    int depthRemain = depthC4 * 4;
    int remain      = depth - depthRemain;
    int z, x, y;
    const T* srcChannel[4];
    const T* srcOffset = src;
    for(z = 0; z < depthC4; ++z) {
        for(y = 0; y < 4; ++y) {
            auto dstZ = dst + (z * 4 + y) * areaOffset[1];
            srcChannel[y] = srcOffset + y;
            for(x = 0; x < area; ++x) {
                dstZ[x] = srcChannel[y][0];
                srcChannel[y] += 4;
            }
        }
        srcOffset += areaOffset[0] * 4;
    }
    if(remain > 0){
        auto dstZ = dst + depthC4 * areaOffset[1] * 4;
        for(y = 0; y < remain; ++y) {
            srcChannel[y] = srcOffset + y;
            for(x = 0; x < area; ++x) {
                dstZ[x] = srcChannel[y][0];
                srcChannel[y] += 4;
            }
            dstZ += areaOffset[1];
        }
    }
}

template<typename T>
void MNNPackC2Common(T* dst, const T* src, size_t area, size_t depth, int* areaOffset) {
    int depthC2     = depth / 2;
    int depthRemain = depthC2 * 2;
    int remain      = depth - depthRemain;
    int z, x, y;
    const T* srcChannel[2];
    const T* srcOffset = src;
    for(z = 0; z < depthC2; ++z) {
        auto dstZ = dst + z * areaOffset[1] * 2;
        for(y = 0; y < 2; ++y) {
            srcChannel[y] = srcOffset + areaOffset[0] * y;
        }
        for(x = 0; x < area; ++x) {
            for(y = 0; y < 2; ++y) {
                dstZ[0] = srcChannel[y][x];
                dstZ++;
            }
        }
        srcOffset += areaOffset[0] * 2;
    }
    if(remain > 0){
        auto dstZ = dst + depthC2 * areaOffset[1] * 2;
        for(y = 0; y < remain; ++y) {
            srcChannel[y] = srcOffset + areaOffset[0] * y;
        }
        for(x = 0; x < area; ++x) {
            for(y = 0; y < remain; ++y) {
                dstZ[0] = srcChannel[y][x];
                dstZ++;
            }
            for(y = remain; y < 2; ++y) {
                dstZ[0] = 0;
                dstZ++;
            }
        }
    }
}

template<typename T>
void MNNUnpackC2Common(T* dst, const T* src, size_t area, size_t depth, int* areaOffset, int pack = 1) {
    int depthC2     = depth / 2;
    int depthRemain = depthC2 * 2;
    int remain      = depth - depthRemain;
    int z, x, y;
    const T* srcChannel[2];
    const T* srcOffset = src;
    for(z = 0; z < depthC2; ++z) {
        for(y = 0; y < 2; ++y) {
            auto dstZ = dst + (z * 2 + y) * areaOffset[1] * pack;
            srcChannel[y] = srcOffset + y * pack;
            for(x = 0; x < area; ++x) {
                for (int p = 0; p < pack; ++p) {
                    dstZ[x * pack + p] = srcChannel[y][p];
                }
                srcChannel[y] += (2 * pack);
            }
        }
        srcOffset += areaOffset[0] * 2 * pack;
    }
    if(remain > 0){
        auto dstZ = dst + depthC2 * areaOffset[1] * 2 * pack;
        for(y = 0; y < remain; ++y) {
            srcChannel[y] = srcOffset + y * pack;
            for(x = 0; x < area; ++x) {
                for (int p = 0; p < pack; ++p) {
                    dstZ[x * pack + p] = srcChannel[y][p];
                }
                srcChannel[y] += 2 * pack;
            }
            dstZ += areaOffset[1] * pack;
        }
    }
}

void MNN4BitcopyWithStride (uint8_t* dstO, const uint8_t* srcO, int size, int stride, int ds) {
    auto src = (uint32_t*)srcO;
    auto dst = (uint32_t*)dstO;
    for (int i = 0; i < size; ++i) {
        dst[0] = *src;
        dst += ds;
        src += stride;
    }
}

void MNN4BitcopyFast (uint8_t* dstO, const uint8_t* srcO, int size, int stride, int ds) {
    // ds=1, stride=0||1
    auto src = (float*)srcO;
    auto dst = (float*)dstO;
    int cnt  = size;
    if (stride == 1) { // stride=1
#ifdef MNN_USE_NEON
        for (; cnt >= 8; cnt -= 8) {
            auto v4 = vld1q_f32(src);
            auto u4 = vld1q_f32(src + 4);
            vst1q_f32(dst, v4);
            vst1q_f32(dst + 4, u4);
            dst += 8;
            src += 8;
        }
        for (; cnt >= 4; cnt -= 4) {
            auto v4 = vld1q_f32(src);
            vst1q_f32(dst, v4);
            dst += 4;
            src += 4;
        }
#elif defined(MNN_USE_SSE)
        for (; cnt >= 8; cnt -= 8) {
            __m128 v4 = _mm_loadu_ps(src);
            __m128 u4 = _mm_loadu_ps(src + 4);
            _mm_storeu_ps(dst, v4);
            _mm_storeu_ps(dst + 4, u4);
            dst += 8;
            src += 8;
        }
        for (; cnt >= 4; cnt -= 4) {
            __m128 v4 = _mm_loadu_ps(src);
            _mm_storeu_ps(dst, v4);
            dst += 4;
            src += 4;
        }
#endif
    } else { // stride=0
        int i = 0;
        float val = *src;
#ifdef MNN_USE_NEON
        auto val4 = vdupq_n_f32(val);
        for (; cnt >= 8; cnt -= 8) {
            vst1q_f32(dst, val4);
            vst1q_f32(dst + 4, val4);
            dst += 8;
        }
        for (; cnt >= 4; cnt -= 4) {
            vst1q_f32(dst, val4);
            dst += 4;
        }
#elif defined(MNN_USE_SSE)
        __m128 val4 = _mm_set_ps(val, val, val, val);
        for (; cnt >= 8; cnt -= 8) {
            _mm_storeu_ps(dst, val4);
            _mm_storeu_ps((dst + 4), val4);
            dst += 8;
        }
        for (; cnt >= 4; cnt -= 4) {
            _mm_storeu_ps(dst, val4);
            dst += 4;
        }
#endif
    }
    for (; cnt > 0; --cnt) {
        dst[0] = *src;
        dst += ds;
        src += stride;
    }
}

void MNN2BitcopyWithStride(uint8_t* dstO, const uint8_t* srcO, int size, int stride, int ds) {
    auto src = (uint16_t*)srcO;
    auto dst = (uint16_t*)dstO;
    for (int i=0; i<size; ++i) {
        *dst = *src;
        src+=stride;
        dst+=ds;
    }
}

void MNN2BitcopyFast(uint8_t* dstO, const uint8_t* srcO, int size, int stride, int ds) {
    auto src = (uint16_t*)srcO;
    auto dst = (uint16_t*)dstO;
    int cnt  = size;
    uint16_t val = *src;
    if (stride == 1) {
#ifdef MNN_USE_NEON
        for (; cnt >= 8; cnt-=8) {
            auto val8 = vld1q_u16(src);
            vst1q_u16(dst, val8);
            src += 8;
            dst += 8;
        }
        for (; cnt >= 4; cnt-=4) {
            auto val4 = vld1_u16(src);
            vst1_u16(dst, val4);
            src += 4;
            dst += 4;
        }
#elif defined(MNN_USE_SSE)
        for (; cnt >= 8; cnt-=8) {
            auto tmp = _mm_loadu_ps((float*)src);
            _mm_storeu_ps((float*)dst, tmp);
            src += 8;
            dst += 8;
        }
#endif
    } else { // stride=0
#ifdef MNN_USE_NEON
        auto val4 = vdup_n_u16(val);
        auto val8 = vdupq_n_u16(val);
        for (; cnt >= 8; cnt-=8) {
            vst1q_u16(dst, val8);
            dst += 8;
        }
        for (; cnt >= 4; cnt-=4) {
            vst1_u16(dst, val4);
            dst += 4;
        }
#elif defined(MNN_USE_SSE)
        uint16_t arr[8] = {val, val, val, val, val, val, val, val};
        auto val8 = _mm_loadu_ps((float*)arr);
        for (; cnt >= 8; cnt-=8) {
            _mm_storeu_ps((float*)dst, val8);
            dst += 8;
        }
#endif
    }
    for (; cnt > 0; --cnt) {
        *dst = *src;
        src += stride;
        dst += ds;
    }
}

void MNN1BitcopyWithStride (uint8_t* dstO, const uint8_t* srcO, int size, int stride, int ds) {
    for (int i = 0; i < size; ++i) {
        dstO[0] = *srcO;
        dstO += ds;
        srcO += stride;
    }

}

void MNN1BitCopyFast (uint8_t* dstO, const uint8_t* srcO, int size, int stride, int ds) {
    int cnt = size;
    uint8_t val = *srcO;
    if (stride == 1) {
#ifdef MNN_USE_SSE
        for (; cnt >= 16; cnt-=16) {
            auto tmp = _mm_loadu_ps((float*)srcO);
            _mm_storeu_ps((float*)dstO, tmp);
            srcO += 16;
            dstO += 16;
        }
#elif defined(MNN_USE_NEON)
        for (; cnt >= 16; cnt-=16) {
            auto val16 = vld1q_u8(srcO);
            vst1q_u8(dstO, val16);
            srcO += 16;
            dstO += 16;
        }
        for (; cnt >= 8; cnt-=8) {
            auto val8 = vld1_u8(srcO);
            vst1_u8(dstO, val8);
            srcO += 8;
            dstO += 8;
        }
#endif
    } else { // stride=0
#ifdef MNN_USE_SSE
        std::vector<uint8_t> arr(16, val);
        auto val16 = _mm_loadu_ps((float*)arr.data());

        for (; cnt >= 16; cnt-=16) {
            _mm_storeu_ps((float*)dstO, val16);
            dstO += 16;
        }
#elif defined(MNN_USE_NEON)
        auto val16 = vdupq_n_u8(val);
        auto val8 = vdup_n_u8(val);
        for (; cnt >= 16; cnt-=16) {
            vst1q_u8(dstO, val16);
            dstO += 16;
        }
        for (; cnt >= 8; cnt-=8) {
            vst1_u8(dstO, val8);
            dstO += 8;
        }
#endif
    }
    for (; cnt > 0; --cnt) {
        dstO[0] = *srcO;
        dstO += ds;
        srcO += stride;
    }
}

void MNNAccumulateSequenceNumber (float* dst, const float* src, int size) {
    // mode: 0:Add, 1:Sub, 2:Min, 3:Max
    int size8 = (size / 8) * 8;
    int i = 0;
    float sum = 0.f;
    float tmp[4];
#ifdef MNN_USE_NEON
    if (size >= 8) {
        auto sum4_1 = vdupq_n_f32(0.f);
        auto sum4_2 = vdupq_n_f32(0.f);
        for (; i < size8; i += 8) {
            auto v4 = vld1q_f32(src);
            auto u4 = vld1q_f32(src + 4);
            sum4_1 = vaddq_f32(sum4_1, v4);
            sum4_2 = vaddq_f32(sum4_2, u4);
            src += 8;
        }
        sum4_1 = vaddq_f32(sum4_1, sum4_2);
        for (int j = 0;j < 4; ++j) {
            sum += sum4_1[j];
        }
    }
#elif defined(MNN_USE_SSE)
    if (size >= 8) {
        auto sum4_1 = _mm_set_ps1(0.f);
        auto sum4_2 = _mm_set_ps1(0.f);

        for (; i < size8; i += 8) {
            auto v4 = _mm_loadu_ps(src);
            auto u4 = _mm_loadu_ps(src + 4);
            sum4_1 = _mm_add_ps(sum4_1, v4);
            sum4_2 = _mm_add_ps(sum4_2, u4);
            src += 8;
        }

        sum4_1 = _mm_add_ps(sum4_1, sum4_2);
        _mm_storeu_ps(tmp, sum4_1);
        sum += (tmp[0] + tmp[1] + tmp[2] + tmp[3]);
    }
#endif
    for (; i < size; ++i) {
        sum += (*src);
        src += 1;
    }
    *dst = sum;
}

void MNNCountMaxMinValue(float* source, float* minVal, float* maxVal, size_t size) {
    int pack = 4;
    float max_ = source[0], min_ = source[0];
    for (int i = 1; i < size; ++i) {
        if (max_ < source[i]) {
            max_ = source[i];
        }
        if (min_ > source[i]) {
            min_ = source[i];
        }
    }
    *minVal = min_;
    *maxVal = max_;
}

#ifndef MNN_USE_NEON
void MNNGetMatMulPackMode(int* eP, int *lP, int* hP) {
    *eP = 16;
    *lP = 1;
    *hP = 4;
}

void MNNGetSparseMatMulPackMode(int* eP, int *lP, int* hP) {
    *eP = 16;
    *lP = 1;
    *hP = 4;
    // hp is corresponding to sparse block along right matrix colum dimension. in ramdom sparse, it is 1.
    return;
}

void MNNPackForMatMul_B(float* dest, const float* source, size_t h, size_t l, bool transpose) {
    auto hP = h / 4;
    auto hR = hP * 4;
    if (hR != h) {
        ::memset(dest, 0, UP_DIV(h, 4)*4*l*sizeof(float));
    }
    if (!transpose) {
        for (int y=0; y<hP; ++y) {
            auto destY = dest + y * 4 * l;
            auto sourceY = source + y * 4;
            for (int x=0; x<l; ++x) {
                ::memcpy(destY + 4 * x, sourceY + x * h, 4 * sizeof(float));
            }
        }
        auto hRemain = h - hR;
        if (hRemain > 0) {
            auto destY = dest + hP * 4 * l;
            auto sourceY = source + hP * 4;
            for (int x=0; x<l; ++x) {
                ::memcpy(destY + 4 * x, sourceY + x * h, hRemain * sizeof(float));
            }
        }
        return;
    }
    int offset[] = {
        (int)l,
        (int)l
    };
    MNNPackC4(dest, source, l, h, offset);
}

static void _MNNPackedMatMulRemain(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter, const float* postParameters, const float* bias, int aStride) {
    auto h = parameter[2];
    auto l = parameter[1];
    auto cStride = parameter[3] / sizeof(float);
    auto hRemain = parameter[4];
    auto bExtraStride = parameter[5] / sizeof(float);
    auto bStride = bExtraStride + l * 4;
    auto hC4 = UP_DIV(h, 4);
    for (int y=0; y<hC4; ++y) {
        ::memset(C + y * cStride, 0, eSize * 4 * sizeof(float));
    }
    float alpha = 1.0f;
    float beta = 0.0f;
    float minValue = -std::numeric_limits<float>().max();
    float maxValue = std::numeric_limits<float>().max();
    if (nullptr != postParameters) {
        minValue = postParameters[2];
        maxValue = postParameters[3];
        alpha = postParameters[0];
        beta = postParameters[1];
    }

    for (int x=0; x<eSize; ++x) {
        auto dst = C + 4 * x;
        auto src = A + x;
        for (int y=0; y<hC4; ++y) {
            auto dstY = dst + y * cStride;
            auto weight = B + y * bStride;
            float summer[4] = {
                0.0f,
                0.0f,
                0.0f,
                0.0f,
            };
            if (nullptr != bias) {
                for (int v=0; v<4; ++v) {
                    summer[v] = bias[4 * y + v];
                }
            }
            for (int z=0; z<l; ++z) {
                auto aZ = src + z * aStride;
                auto wZ = weight + z * 4;
                summer[0] += wZ[0] * aZ[0];
                summer[1] += wZ[1] * aZ[0];
                summer[2] += wZ[2] * aZ[0];
                summer[3] += wZ[3] * aZ[0];
            }
            for (int v=0; v<4; ++v) {
                auto dstValue = std::min(summer[v], maxValue);
                dstValue = std::max(dstValue, minValue);
                dstY[v] = dstValue;
            }
        }
    }
}

void MNNPackedMatMul(float* C, const float* A, const float* B, const size_t* parameter, const float* postParameters, const float* bias, const float* k, const float* b) {
    return _MNNPackedMatMulRemain(C, A, B, 16, parameter, postParameters, bias, 16);
}

void MNNPackedMatMulRemain(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter, const float* postParameters, const float* bias, const float* k, const float* b) {
    auto aStride = parameter[0] / sizeof(float);
    _MNNPackedMatMulRemain(C, A, B, eSize, parameter, postParameters, bias, aStride);
}

void MNNPackC4ForMatMul_A(float* destOrigin, float const** sourceGroup, const int32_t* info, const int32_t* el) {
    int number = info[0];
    int eReal = info[1];
    int eDest = info[2];
    int offset = info[3];
    for (int n=0; n<number; ++n) {
        int e = el[4 * n + 0];
        int l = el[4 * n + 1];
        int eOffset = el[4 * n + 2];
        int lOffset = el[4 * n + 3];
        auto dest = destOrigin + lOffset * eDest + eOffset;
        auto source = sourceGroup[n];

        for (int y=0; y<e; ++y) {
            auto yR = y % eDest;
            for (int x=0; x<l; ++x) {
                auto xR = x % 4;
                auto xC = x / 4;
                dest[(x) * eDest + yR] = source[xC * eReal * 4 + y * 4 * offset + xR];
            }
        }
    }
}

void MNNPackedSparseMatMulEpx1(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter, const float* postParameters, const float* bias, unsigned int* NNZMap, int* dataOffsetMap) {

    auto eP = parameter[0] / sizeof(float);
    MNN_ASSERT((eP & 0x03) == 0); // In sparse calculate, eP should be evenly divided by 4
    auto h = parameter[2];
    auto l = parameter[1];
    auto cStride = parameter[3] / sizeof(float);
    auto aStride = eP * l;
    auto hRemain = parameter[4];
    auto bExtraStride = parameter[5] / sizeof(float);
    auto bStride = bExtraStride + l * 4;
    auto hC4 = UP_DIV(h, 4);
    float minValue = -std::numeric_limits<float>().max();
    float maxValue = std::numeric_limits<float>().max();
    if (nullptr != postParameters) {
        minValue = postParameters[2];
        maxValue = postParameters[3];
    }
    // MNN_PRINT("MNNPackedSparseMatMul eP:%lu, eSize:%lu, l:%lu, h:%lu, cStride:%lu, aStride:%lu\n", eP, eSize, l, h, cStride, aStride);

    const float* a = A;
    size_t ie = 0;
    for (ie = 0; ie < eSize && eP <= eSize; ie += eP) {
        const int* dataOffset = dataOffsetMap;
        const int diff = *dataOffset++;
        a += diff;
        const float* w = B;
        float* blockC = C + (ie << 2);
        const unsigned int* nnz = NNZMap;
        for (auto ih = 0; ih < h; ih++) {
            auto ihPack = ih >> 2;
            auto ihSubIndex = ih & 0x03;
            auto c = blockC + ihPack * cStride + ihSubIndex;
            const float initValue = nullptr != bias ? bias[ih] : 0;
            float acc0 = initValue;
            float acc1 = initValue;
            float acc2 = initValue;
            float acc3 = initValue;
            float acc4 = initValue;
            float acc5 = initValue;
            float acc6 = initValue;
            float acc7 = initValue;
            float acc8 = initValue;
            float acc9 = initValue;
            float acc10 = initValue;
            float acc11 = initValue;
            float acc12 = initValue;
            float acc13 = initValue;
            float acc14 = initValue;
            float acc15 = initValue;
            const int lElement = *nnz++;
            for (auto il = 0; il < lElement; il++) {

                const int diff = *dataOffset++;
                const float a0 = a[0];
                const float a1 = a[1];
                const float a2 = a[2];
                const float a3 = a[3];
                const float a4 = a[4];
                const float a5 = a[5];
                const float a6 = a[6];
                const float a7 = a[7];
                const float a8 = a[8];
                const float a9 = a[9];
                const float a10 = a[10];
                const float a11 = a[11];
                const float a12 = a[12];
                const float a13 = a[13];
                const float a14 = a[14];
                const float a15 = a[15];

                const float oneW = *w++;

                // MNN_PRINT("16-loop: ie:%zu, a offset:%ld, w offset:%ld, c offset:%ld, w value:%f, a value[0-15]:", ie, a - A, w - B - 1, c - C, oneW);
                // formatMatrix(a, {16});
                // MNN_PRINT("\n");
                a = a + diff;
                acc0 += a0 * oneW;
                acc1 += a1 * oneW;
                acc2 += a2 * oneW;
                acc3 += a3 * oneW;
                acc4 += a4 * oneW;
                acc5 += a5 * oneW;
                acc6 += a6 * oneW;
                acc7 += a7 * oneW;
                acc8 += a8 * oneW;
                acc9 += a9 * oneW;
                acc10 += a10 * oneW;
                acc11 += a11 * oneW;
                acc12 += a12 * oneW;
                acc13 += a13 * oneW;
                acc14 += a14 * oneW;
                acc15 += a15 * oneW;
            }
            acc0  = std::max(std::min(maxValue, acc0), minValue);
            acc1  = std::max(std::min(maxValue, acc1), minValue);
            acc2  = std::max(std::min(maxValue, acc2), minValue);
            acc3  = std::max(std::min(maxValue, acc3), minValue);
            acc4  = std::max(std::min(maxValue, acc4), minValue);
            acc5  = std::max(std::min(maxValue, acc5), minValue);
            acc6  = std::max(std::min(maxValue, acc6), minValue);
            acc7  = std::max(std::min(maxValue, acc7), minValue);
            acc8  = std::max(std::min(maxValue, acc8), minValue);
            acc9  = std::max(std::min(maxValue, acc9), minValue);
            acc10 = std::max(std::min(maxValue, acc10), minValue);
            acc11 = std::max(std::min(maxValue, acc11), minValue);
            acc12 = std::max(std::min(maxValue, acc12), minValue);
            acc13 = std::max(std::min(maxValue, acc13), minValue);
            acc14 = std::max(std::min(maxValue, acc14), minValue);
            acc15 = std::max(std::min(maxValue, acc15), minValue);

            // how to store faster: st4 / transpose /
            c[0] = acc0;
            c[4] = acc1;
            c[4 * 2] = acc2;
            c[4 * 3] = acc3;
            c[4 * 4] = acc4;
            c[4 * 5] = acc5;
            c[4 * 6] = acc6;
            c[4 * 7] = acc7;
            c[4 * 8] = acc8;
            c[4 * 9] = acc9;
            c[4 * 10] = acc10;
            c[4 * 11] = acc11;
            c[4 * 12] = acc12;
            c[4 * 13] = acc13;
            c[4 * 14] = acc14;
            c[4 * 15] = acc15;
        }
        a += aStride;
    }
    // const float* blockA = A + ie * l;
    if (eSize & 0x08) {
        const int* dataOffset = dataOffsetMap;
        const int diff = *dataOffset++;
        // a = blockA + diff;
        a += diff;
        const float* w = B;
        float* blockC = C + (ie << 2);
        const unsigned int* nnz = NNZMap;
        for (auto ih = 0; ih < h; ih++) {
            auto ihPack = ih >> 2;
            auto ihSubIndex = ih & 0x03;
            auto c = blockC + ihPack * cStride + ihSubIndex;
            const float initValue = nullptr != bias ? bias[ih] : 0;
            float acc0 = initValue;
            float acc1 = initValue;
            float acc2 = initValue;
            float acc3 = initValue;
            float acc4 = initValue;
            float acc5 = initValue;
            float acc6 = initValue;
            float acc7 = initValue;

            const int lElement = *nnz++;
            for (auto il = 0; il < lElement; il++) {
                const int diff = *dataOffset++;
                const float a0 = a[0];
                const float a1 = a[1];
                const float a2 = a[2];
                const float a3 = a[3];
                const float a4 = a[4];
                const float a5 = a[5];
                const float a6 = a[6];
                const float a7 = a[7];
                const float oneW = *w++;
                // MNN_PRINT("8-loop: ie:%zu, a offset:%ld, w offset:%ld, c offset:%ld, w value:%f, a value[0-7]:", ie, a - A, w - B - 1, c - C, oneW);
                // formatMatrix(a, {8});
                // MNN_PRINT("\n");
                a = a + diff;
                acc0 += a0 * oneW;
                acc1 += a1 * oneW;
                acc2 += a2 * oneW;
                acc3 += a3 * oneW;
                acc4 += a4 * oneW;
                acc5 += a5 * oneW;
                acc6 += a6 * oneW;
                acc7 += a7 * oneW;
            }
            acc0  = std::max(std::min(maxValue, acc0), minValue);
            acc1  = std::max(std::min(maxValue, acc1), minValue);
            acc2  = std::max(std::min(maxValue, acc2), minValue);
            acc3  = std::max(std::min(maxValue, acc3), minValue);
            acc4  = std::max(std::min(maxValue, acc4), minValue);
            acc5  = std::max(std::min(maxValue, acc5), minValue);
            acc6  = std::max(std::min(maxValue, acc6), minValue);
            acc7  = std::max(std::min(maxValue, acc7), minValue);
            // how to store faster: st4 / transpose /
            c[0] = acc0;
            c[4] = acc1;
            c[4 * 2] = acc2;
            c[4 * 3] = acc3;
            c[4 * 4] = acc4;
            c[4 * 5] = acc5;
            c[4 * 6] = acc6;
            c[4 * 7] = acc7;
        }
        ie += 8;
        a += 8;
    }

    if (eSize & 0x04) {
        const int* dataOffset = dataOffsetMap;
        const int diff = *dataOffset++;
        // const float* a = blockA + diff;
        a += diff;
        const float* w = B;
        float* blockC = C + (ie << 2);
        const unsigned int* nnz = NNZMap;
        for (auto ih = 0; ih < h; ih++) {
            auto ihPack = ih >> 2;
            auto ihSubIndex = ih & 0x03;
            auto c = blockC + ihPack * cStride + ihSubIndex;
            const float initValue = nullptr != bias ? bias[ih] : 0;
            float acc0 = initValue;
            float acc1 = initValue;
            float acc2 = initValue;
            float acc3 = initValue;

            const int lElement = *nnz++;
            for (auto il = 0; il < lElement; il++) {
                const int diff = *dataOffset++;
                const float a0 = a[0];
                const float a1 = a[1];
                const float a2 = a[2];
                const float a3 = a[3];
                const float oneW = *w++;
                // MNN_PRINT("4-loop: ie:%zu, a offset:%ld, w offset:%ld, c offset:%ld, w value:%f, a value[0-3]:", ie, a - A, w - B - 1, c - C, oneW);
                // formatMatrix(a, {4});
                // MNN_PRINT("\n");
                a = a + diff;
                acc0 += a0 * oneW;
                acc1 += a1 * oneW;
                acc2 += a2 * oneW;
                acc3 += a3 * oneW;
            }
            acc0  = std::max(std::min(maxValue, acc0), minValue);
            acc1  = std::max(std::min(maxValue, acc1), minValue);
            acc2  = std::max(std::min(maxValue, acc2), minValue);
            acc3  = std::max(std::min(maxValue, acc3), minValue);
            // how to store faster: st4 / transpose /
            c[0] = acc0;
            c[4] = acc1;
            c[4 * 2] = acc2;
            c[4 * 3] = acc3;
        }
        ie += 4;
        a += 4;
    }
    if (eSize & 0x02) {
        const int* dataOffset = dataOffsetMap;
        const int diff = *dataOffset++;
        // const float* a = blockA + diff;
        a += diff;
        const float* w = B;
        float* blockC = C + (ie << 2);
        const unsigned int* nnz = NNZMap;
        for (auto ih = 0; ih < h; ih++) {
            auto ihPack = ih >> 2;
            auto ihSubIndex = ih & 0x03;
            auto c = blockC + ihPack * cStride + ihSubIndex;
            const float initValue = nullptr != bias ? bias[ih] : 0;
            float acc0 = initValue;
            float acc1 = initValue;

            const int lElement = *nnz++;
            for (auto il = 0; il < lElement; il++) {
                const int diff = *dataOffset++;
                const float a0 = a[0];
                const float a1 = a[1];
                const float oneW = *w++;
                // MNN_PRINT("2-loop: ie:%zu, a offset:%ld, w offset:%ld, c offset:%ld, w value:%f, a value[0-1]:", ie, a - A, w - B - 1, c - C, oneW);
                // formatMatrix(a, {2});
                // MNN_PRINT("\n");
                a = a + diff;
                acc0 += a0 * oneW;
                acc1 += a1 * oneW;
            }
            acc0  = std::max(std::min(maxValue, acc0), minValue);
            acc1  = std::max(std::min(maxValue, acc1), minValue);
            // how to store faster: st4 / transpose /
            c[0] = acc0;
            c[4] = acc1;
        }
        ie += 2;
        a += 2;
    }
    if (eSize & 0x01) {
        const int* dataOffset = dataOffsetMap;
        const int diff = *dataOffset++;
        // const float* a = blockA + diff;
        a += diff;
        const float* w = B;
        float* blockC = C + (ie << 2);
        const unsigned int* nnz = NNZMap;
        for (auto ih = 0; ih < h; ih++) {
            auto ihPack = ih >> 2;
            auto ihSubIndex = ih & 0x03;
            auto c = blockC + ihPack * cStride + ihSubIndex;
            const float initValue = nullptr != bias ? bias[ih] : 0;
            float acc0 = initValue;

            const int lElement = *nnz++;
            for (auto il = 0; il < lElement; il++) {
                const int diff = *dataOffset++;
                const float a0 = a[0];
                const float oneW = *w++;

                // MNN_PRINT("1-loop: ie:%zu, a offset:%ld, c offset:%ld, w offset:%ld, w value:%f, a value[0]:", ie, a - A, w - B - 1, c - C, oneW);
                // formatMatrix(a, {1});
                // MNN_PRINT("\n");
                a = a + diff;
                acc0 += a0 * oneW;
            }
            acc0  = std::max(std::min(maxValue, acc0), minValue);
            // how to store faster: st4 / transpose /
            c[0] = acc0;
        }
        ie += 1;
        // a += 1;
    }

    return;
}

void MNNPackedSparseMatMulEpx4(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter, const float* postParameters, const float* bias, unsigned int* NNZMap, int* dataOffsetMap) {

    auto eP = parameter[0] / sizeof(float);
    MNN_ASSERT((eP & 0x03) == 0); // In sparse calculate, eP should be evenly divided by 4
    auto h = parameter[2];
    auto l = parameter[1];
    auto cStride = parameter[3] / sizeof(float);
    auto aStride = eP * l;
    auto hRemain = parameter[4];
    auto bExtraStride = parameter[5] / sizeof(float);
    auto bStride = bExtraStride + l * 4;
    auto hC4 = UP_DIV(h, 4);
    float minValue = -std::numeric_limits<float>().max();
    float maxValue = std::numeric_limits<float>().max();
    if (nullptr != postParameters) {
        minValue = postParameters[2];
        maxValue = postParameters[3];
    }
    // MNN_PRINT("MNNPackedSparseMatMul 16x4 eP:%lu, eSize:%lu, l:%lu, h:%lu, cStride:%lu, aStride:%lu\n", eP, eSize, l, h, cStride, aStride);
    const int sparseBlockOC = 4;
    const float* a = A;
    size_t ie = 0;
    for (ie = 0; ie < eSize && eP <= eSize; ie += eP) {
        const int* dataOffset = dataOffsetMap;
        const int diff = *dataOffset++;
        a += diff;
        const float* w = B;
        float* blockC = C + (ie << 2);
        const unsigned int* nnz = NNZMap;

        size_t ih = 0;
        for (; ih < (h & (~0x03)); ih += sparseBlockOC) {
            auto ihPack = ih >> 2;
            auto c = blockC + ihPack * cStride;

            float initValue[4] = {0, 0, 0, 0};
            if (nullptr != bias) {
                memcpy(initValue, bias + ih, 4 * sizeof(float));
            }
            float acc0[4];
            float acc1[4];
            float acc2[4];
            float acc3[4];
            float acc4[4];
            float acc5[4];
            float acc6[4];
            float acc7[4];
            float acc8[4];
            float acc9[4];
            float acc10[4];
            float acc11[4];
            float acc12[4];
            float acc13[4];
            float acc14[4];
            float acc15[4];
            memcpy(acc0, initValue, 4 * sizeof(float));
            memcpy(acc1, initValue, 4 * sizeof(float));
            memcpy(acc2, initValue, 4 * sizeof(float));
            memcpy(acc3, initValue, 4 * sizeof(float));
            memcpy(acc4, initValue, 4 * sizeof(float));
            memcpy(acc5, initValue, 4 * sizeof(float));
            memcpy(acc6, initValue, 4 * sizeof(float));
            memcpy(acc7, initValue, 4 * sizeof(float));
            memcpy(acc8, initValue, 4 * sizeof(float));
            memcpy(acc9, initValue, 4 * sizeof(float));
            memcpy(acc10, initValue, 4 * sizeof(float));
            memcpy(acc11, initValue, 4 * sizeof(float));
            memcpy(acc12, initValue, 4 * sizeof(float));
            memcpy(acc13, initValue, 4 * sizeof(float));
            memcpy(acc14, initValue, 4 * sizeof(float));
            memcpy(acc15, initValue, 4 * sizeof(float));

            const int lElement = *nnz++;
            for (auto il = 0; il < lElement; il++) {

                const int diff = *dataOffset++;
                const float a0 = a[0];
                const float a1 = a[1];
                const float a2 = a[2];
                const float a3 = a[3];
                const float a4 = a[4];
                const float a5 = a[5];
                const float a6 = a[6];
                const float a7 = a[7];
                const float a8 = a[8];
                const float a9 = a[9];
                const float a10 = a[10];
                const float a11 = a[11];
                const float a12 = a[12];
                const float a13 = a[13];
                const float a14 = a[14];
                const float a15 = a[15];

                const float wv[4] = {*w++, *w++, *w++, *w++};

                // MNN_PRINT("16-loop: ie:%zu, a offset:%ld, w offset:%ld, c offset:%ld, w value:%f, a value[0-15]:", ie, a - A, w - B - 1, c - C, oneW);
                // formatMatrix(a, {16});
                // MNN_PRINT("\n");
                a = a + diff;
                for (int lane = 0; lane < 4; lane++) {
                    acc0[lane] += a0 * wv[lane];
                    acc1[lane] += a1 * wv[lane];
                    acc2[lane] += a2 * wv[lane];
                    acc3[lane] += a3 * wv[lane];
                    acc4[lane] += a4 * wv[lane];
                    acc5[lane] += a5 * wv[lane];
                    acc6[lane] += a6 * wv[lane];
                    acc7[lane] += a7 * wv[lane];
                    acc8[lane] += a8 * wv[lane];
                    acc9[lane] += a9 * wv[lane];
                    acc10[lane] += a10 * wv[lane];
                    acc11[lane] += a11 * wv[lane];
                    acc12[lane] += a12 * wv[lane];
                    acc13[lane] += a13 * wv[lane];
                    acc14[lane] += a14 * wv[lane];
                    acc15[lane] += a15 * wv[lane];
                }
            }

            for (int lane = 0; lane < 4; lane++) {
                acc0[lane]  = std::max(std::min(maxValue, acc0[lane]), minValue);
                acc1[lane]  = std::max(std::min(maxValue, acc1[lane]), minValue);
                acc2[lane]  = std::max(std::min(maxValue, acc2[lane]), minValue);
                acc3[lane]  = std::max(std::min(maxValue, acc3[lane]), minValue);
                acc4[lane]  = std::max(std::min(maxValue, acc4[lane]), minValue);
                acc5[lane]  = std::max(std::min(maxValue, acc5[lane]), minValue);
                acc6[lane]  = std::max(std::min(maxValue, acc6[lane]), minValue);
                acc7[lane]  = std::max(std::min(maxValue, acc7[lane]), minValue);
                acc8[lane]  = std::max(std::min(maxValue, acc8[lane]), minValue);
                acc9[lane]  = std::max(std::min(maxValue, acc9[lane]), minValue);
                acc10[lane] = std::max(std::min(maxValue, acc10[lane]), minValue);
                acc11[lane] = std::max(std::min(maxValue, acc11[lane]), minValue);
                acc12[lane] = std::max(std::min(maxValue, acc12[lane]), minValue);
                acc13[lane] = std::max(std::min(maxValue, acc13[lane]), minValue);
                acc14[lane] = std::max(std::min(maxValue, acc14[lane]), minValue);
                acc15[lane] = std::max(std::min(maxValue, acc15[lane]), minValue);
            }

            memcpy(c, acc0, 4 * sizeof(float));  // store continuous c
            memcpy(c + 4, acc1, 4 * sizeof(float));
            memcpy(c + 4 * 2, acc2, 4 * sizeof(float));
            memcpy(c + 4 * 3, acc3, 4 * sizeof(float));
            memcpy(c + 4 * 4, acc4, 4 * sizeof(float));
            memcpy(c + 4 * 5, acc5, 4 * sizeof(float));
            memcpy(c + 4 * 6, acc6, 4 * sizeof(float));
            memcpy(c + 4 * 7, acc7, 4 * sizeof(float));
            memcpy(c + 4 * 8, acc8, 4 * sizeof(float));
            memcpy(c + 4 * 9, acc9, 4 * sizeof(float));
            memcpy(c + 4 * 10, acc10, 4 * sizeof(float));
            memcpy(c + 4 * 11, acc11, 4 * sizeof(float));
            memcpy(c + 4 * 12, acc12, 4 * sizeof(float));
            memcpy(c + 4 * 13, acc13, 4 * sizeof(float));
            memcpy(c + 4 * 14, acc14, 4 * sizeof(float));
            memcpy(c + 4 * 15, acc15, 4 * sizeof(float));
        }

        blockC += (h >> 2) * cStride;
        for (; ih < h; ih++) {
            auto ihSubIndex = ih & 0x03;
            auto c = blockC + ihSubIndex;
            const float initValue = nullptr != bias ? bias[ih] : 0;
            float acc0 = initValue;
            float acc1 = initValue;
            float acc2 = initValue;
            float acc3 = initValue;
            float acc4 = initValue;
            float acc5 = initValue;
            float acc6 = initValue;
            float acc7 = initValue;
            float acc8 = initValue;
            float acc9 = initValue;
            float acc10 = initValue;
            float acc11 = initValue;
            float acc12 = initValue;
            float acc13 = initValue;
            float acc14 = initValue;
            float acc15 = initValue;
            const int lElement = *nnz++;
            for (auto il = 0; il < lElement; il++) {

                const int diff = *dataOffset++;
                const float a0 = a[0];
                const float a1 = a[1];
                const float a2 = a[2];
                const float a3 = a[3];
                const float a4 = a[4];
                const float a5 = a[5];
                const float a6 = a[6];
                const float a7 = a[7];
                const float a8 = a[8];
                const float a9 = a[9];
                const float a10 = a[10];
                const float a11 = a[11];
                const float a12 = a[12];
                const float a13 = a[13];
                const float a14 = a[14];
                const float a15 = a[15];

                const float oneW = *w++;

                // MNN_PRINT("16-loop: ie:%zu, a offset:%ld, w offset:%ld, c offset:%ld, w value:%f, a value[0-15]:", ie, a - A, w - B - 1, c - C, oneW);
                // formatMatrix(a, {16});
                // MNN_PRINT("\n");
                a = a + diff;
                acc0 += a0 * oneW;
                acc1 += a1 * oneW;
                acc2 += a2 * oneW;
                acc3 += a3 * oneW;
                acc4 += a4 * oneW;
                acc5 += a5 * oneW;
                acc6 += a6 * oneW;
                acc7 += a7 * oneW;
                acc8 += a8 * oneW;
                acc9 += a9 * oneW;
                acc10 += a10 * oneW;
                acc11 += a11 * oneW;
                acc12 += a12 * oneW;
                acc13 += a13 * oneW;
                acc14 += a14 * oneW;
                acc15 += a15 * oneW;
            }
            acc0  = std::max(std::min(maxValue, acc0), minValue);
            acc1  = std::max(std::min(maxValue, acc1), minValue);
            acc2  = std::max(std::min(maxValue, acc2), minValue);
            acc3  = std::max(std::min(maxValue, acc3), minValue);
            acc4  = std::max(std::min(maxValue, acc4), minValue);
            acc5  = std::max(std::min(maxValue, acc5), minValue);
            acc6  = std::max(std::min(maxValue, acc6), minValue);
            acc7  = std::max(std::min(maxValue, acc7), minValue);
            acc8  = std::max(std::min(maxValue, acc8), minValue);
            acc9  = std::max(std::min(maxValue, acc9), minValue);
            acc10 = std::max(std::min(maxValue, acc10), minValue);
            acc11 = std::max(std::min(maxValue, acc11), minValue);
            acc12 = std::max(std::min(maxValue, acc12), minValue);
            acc13 = std::max(std::min(maxValue, acc13), minValue);
            acc14 = std::max(std::min(maxValue, acc14), minValue);
            acc15 = std::max(std::min(maxValue, acc15), minValue);

            // how to store faster: st4 / transpose /
            c[0] = acc0;
            c[4] = acc1;
            c[4 * 2] = acc2;
            c[4 * 3] = acc3;
            c[4 * 4] = acc4;
            c[4 * 5] = acc5;
            c[4 * 6] = acc6;
            c[4 * 7] = acc7;
            c[4 * 8] = acc8;
            c[4 * 9] = acc9;
            c[4 * 10] = acc10;
            c[4 * 11] = acc11;
            c[4 * 12] = acc12;
            c[4 * 13] = acc13;
            c[4 * 14] = acc14;
            c[4 * 15] = acc15;
        }
        a += aStride;
    }
    // const float* blockA = A + ie * l;
    if (eSize & 0x08) {
        const int* dataOffset = dataOffsetMap;
        const int diff = *dataOffset++;
        // a = blockA + diff;
        a += diff;
        const float* w = B;
        float* blockC = C + (ie << 2);
        const unsigned int* nnz = NNZMap;

        size_t ih = 0;
        for (; ih < (h & (~0x03)); ih += sparseBlockOC) {
            auto ihPack = ih >> 2;
            auto c = blockC + ihPack * cStride;
            float initValue[4] = {0, 0, 0, 0};
            if (nullptr != bias) {
                memcpy(initValue, bias + ih, 4 * sizeof(float));
            }
            float acc0[4];
            float acc1[4];
            float acc2[4];
            float acc3[4];
            float acc4[4];
            float acc5[4];
            float acc6[4];
            float acc7[4];
            memcpy(acc0, initValue, 4 * sizeof(float));
            memcpy(acc1, initValue, 4 * sizeof(float));
            memcpy(acc2, initValue, 4 * sizeof(float));
            memcpy(acc3, initValue, 4 * sizeof(float));
            memcpy(acc4, initValue, 4 * sizeof(float));
            memcpy(acc5, initValue, 4 * sizeof(float));
            memcpy(acc6, initValue, 4 * sizeof(float));
            memcpy(acc7, initValue, 4 * sizeof(float));
            const int lElement = *nnz++;
            for (auto il = 0; il < lElement; il++) {

                const int diff = *dataOffset++;
                const float a0 = a[0];
                const float a1 = a[1];
                const float a2 = a[2];
                const float a3 = a[3];
                const float a4 = a[4];
                const float a5 = a[5];
                const float a6 = a[6];
                const float a7 = a[7];
                const float wv[4] = {*w++, *w++, *w++, *w++};
                // MNN_PRINT("16-loop: ie:%zu, a offset:%ld, w offset:%ld, c offset:%ld, w value:%f, a value[0-15]:", ie, a - A, w - B - 1, c - C, oneW);
                // formatMatrix(a, {16});
                // MNN_PRINT("\n");
                a = a + diff;
                for (int lane = 0; lane < 4; lane++) {
                    acc0[lane] += a0 * wv[lane];
                    acc1[lane] += a1 * wv[lane];
                    acc2[lane] += a2 * wv[lane];
                    acc3[lane] += a3 * wv[lane];
                    acc4[lane] += a4 * wv[lane];
                    acc5[lane] += a5 * wv[lane];
                    acc6[lane] += a6 * wv[lane];
                    acc7[lane] += a7 * wv[lane];
                }
            }

            for (int lane = 0; lane < 4; lane++) {
                acc0[lane]  = std::max(std::min(maxValue, acc0[lane]), minValue);
                acc1[lane]  = std::max(std::min(maxValue, acc1[lane]), minValue);
                acc2[lane]  = std::max(std::min(maxValue, acc2[lane]), minValue);
                acc3[lane]  = std::max(std::min(maxValue, acc3[lane]), minValue);
                acc4[lane]  = std::max(std::min(maxValue, acc4[lane]), minValue);
                acc5[lane]  = std::max(std::min(maxValue, acc5[lane]), minValue);
                acc6[lane]  = std::max(std::min(maxValue, acc6[lane]), minValue);
                acc7[lane]  = std::max(std::min(maxValue, acc7[lane]), minValue);
            }

            memcpy(c, acc0, 4 * sizeof(float));  // store continuous c
            memcpy(c + 4, acc1, 4 * sizeof(float));
            memcpy(c + 4 * 2, acc2, 4 * sizeof(float));
            memcpy(c + 4 * 3, acc3, 4 * sizeof(float));
            memcpy(c + 4 * 4, acc4, 4 * sizeof(float));
            memcpy(c + 4 * 5, acc5, 4 * sizeof(float));
            memcpy(c + 4 * 6, acc6, 4 * sizeof(float));
            memcpy(c + 4 * 7, acc7, 4 * sizeof(float));
        }
        blockC += (ih >> 2) * cStride;
        for (; ih < h; ih++) {
            auto ihSubIndex = ih & 0x03;
            auto c = blockC + ihSubIndex;
            const float initValue = nullptr != bias ? bias[ih] : 0;
            float acc0 = initValue;
            float acc1 = initValue;
            float acc2 = initValue;
            float acc3 = initValue;
            float acc4 = initValue;
            float acc5 = initValue;
            float acc6 = initValue;
            float acc7 = initValue;

            const int lElement = *nnz++;
            for (auto il = 0; il < lElement; il++) {
                const int diff = *dataOffset++;
                const float a0 = a[0];
                const float a1 = a[1];
                const float a2 = a[2];
                const float a3 = a[3];
                const float a4 = a[4];
                const float a5 = a[5];
                const float a6 = a[6];
                const float a7 = a[7];
                const float oneW = *w++;
                // MNN_PRINT("8-loop: ie:%zu, a offset:%ld, w offset:%ld, c offset:%ld, w value:%f, a value[0-7]:", ie, a - A, w - B - 1, c - C, oneW);
                // formatMatrix(a, {8});
                // MNN_PRINT("\n");
                a = a + diff;
                acc0 += a0 * oneW;
                acc1 += a1 * oneW;
                acc2 += a2 * oneW;
                acc3 += a3 * oneW;
                acc4 += a4 * oneW;
                acc5 += a5 * oneW;
                acc6 += a6 * oneW;
                acc7 += a7 * oneW;
            }
            acc0  = std::max(std::min(maxValue, acc0), minValue);
            acc1  = std::max(std::min(maxValue, acc1), minValue);
            acc2  = std::max(std::min(maxValue, acc2), minValue);
            acc3  = std::max(std::min(maxValue, acc3), minValue);
            acc4  = std::max(std::min(maxValue, acc4), minValue);
            acc5  = std::max(std::min(maxValue, acc5), minValue);
            acc6  = std::max(std::min(maxValue, acc6), minValue);
            acc7  = std::max(std::min(maxValue, acc7), minValue);
            // how to store faster: st4 / transpose /
            c[0] = acc0;
            c[4] = acc1;
            c[4 * 2] = acc2;
            c[4 * 3] = acc3;
            c[4 * 4] = acc4;
            c[4 * 5] = acc5;
            c[4 * 6] = acc6;
            c[4 * 7] = acc7;
        }
        ie += 8;
        a += 8;
    }

    if (eSize & 0x04) {
        const int* dataOffset = dataOffsetMap;
        const int diff = *dataOffset++;
        // const float* a = blockA + diff;
        a += diff;
        const float* w = B;
        float* blockC = C + (ie << 2);
        const unsigned int* nnz = NNZMap;

        size_t ih = 0;
        for (; ih < (h & (~0x03)); ih += sparseBlockOC) {
            auto ihPack = ih >> 2;
            auto c = blockC + ihPack * cStride;
            float initValue[4] = {0, 0, 0, 0};
            if (nullptr != bias) {
                memcpy(initValue, bias + ih, 4 * sizeof(float));
            }
            float acc0[4];
            float acc1[4];
            float acc2[4];
            float acc3[4];
            memcpy(acc0, initValue, 4 * sizeof(float));
            memcpy(acc1, initValue, 4 * sizeof(float));
            memcpy(acc2, initValue, 4 * sizeof(float));
            memcpy(acc3, initValue, 4 * sizeof(float));

            const int lElement = *nnz++;
            for (auto il = 0; il < lElement; il++) {
                const int diff = *dataOffset++;
                const float a0 = a[0];
                const float a1 = a[1];
                const float a2 = a[2];
                const float a3 = a[3];
                const float wv[4] = {*w++, *w++, *w++, *w++};
                // MNN_PRINT("16-loop: ie:%zu, a offset:%ld, w offset:%ld, c offset:%ld, w value:%f, a value[0-15]:", ie, a - A, w - B - 1, c - C, oneW);
                // formatMatrix(a, {16});
                // MNN_PRINT("\n");
                a = a + diff;
                for (int lane = 0; lane < 4; lane++) {
                    acc0[lane] += a0 * wv[lane];
                    acc1[lane] += a1 * wv[lane];
                    acc2[lane] += a2 * wv[lane];
                    acc3[lane] += a3 * wv[lane];
                }
            }

            for (int lane = 0; lane < 4; lane++) {
                acc0[lane]  = std::max(std::min(maxValue, acc0[lane]), minValue);
                acc1[lane]  = std::max(std::min(maxValue, acc1[lane]), minValue);
                acc2[lane]  = std::max(std::min(maxValue, acc2[lane]), minValue);
                acc3[lane]  = std::max(std::min(maxValue, acc3[lane]), minValue);
            }

            memcpy(c, acc0, 4 * sizeof(float));  // store continuous c
            memcpy(c + 4, acc1, 4 * sizeof(float));
            memcpy(c + 4 * 2, acc2, 4 * sizeof(float));
            memcpy(c + 4 * 3, acc3, 4 * sizeof(float));
        }
        blockC += (ih >> 2) * cStride;
        for (; ih < h; ih++) {
            auto ihSubIndex = ih & 0x03;
            auto c = blockC + ihSubIndex;
            const float initValue = nullptr != bias ? bias[ih] : 0;
            float acc0 = initValue;
            float acc1 = initValue;
            float acc2 = initValue;
            float acc3 = initValue;

            const int lElement = *nnz++;
            for (auto il = 0; il < lElement; il++) {
                const int diff = *dataOffset++;
                const float a0 = a[0];
                const float a1 = a[1];
                const float a2 = a[2];
                const float a3 = a[3];
                const float oneW = *w++;
                // MNN_PRINT("4-loop: ie:%zu, a offset:%ld, w offset:%ld, c offset:%ld, w value:%f, a value[0-3]:", ie, a - A, w - B - 1, c - C, oneW);
                // formatMatrix(a, {4});
                // MNN_PRINT("\n");
                a = a + diff;
                acc0 += a0 * oneW;
                acc1 += a1 * oneW;
                acc2 += a2 * oneW;
                acc3 += a3 * oneW;
            }
            acc0  = std::max(std::min(maxValue, acc0), minValue);
            acc1  = std::max(std::min(maxValue, acc1), minValue);
            acc2  = std::max(std::min(maxValue, acc2), minValue);
            acc3  = std::max(std::min(maxValue, acc3), minValue);
            // how to store faster: st4 / transpose /
            c[0] = acc0;
            c[4] = acc1;
            c[4 * 2] = acc2;
            c[4 * 3] = acc3;
        }
        ie += 4;
        a += 4;
    }
    if (eSize & 0x02) {
        const int* dataOffset = dataOffsetMap;
        const int diff = *dataOffset++;
        // const float* a = blockA + diff;
        a += diff;
        const float* w = B;
        float* blockC = C + (ie << 2);
        const unsigned int* nnz = NNZMap;

        size_t ih = 0;
        for (; ih < (h & (~0x03)); ih += sparseBlockOC) {
            auto ihPack = ih >> 2;
            auto c = blockC + ihPack * cStride;
            float initValue[4] = {0, 0, 0, 0};
            if (nullptr != bias) {
                memcpy(initValue, bias + ih, 4 * sizeof(float));
            }
            float acc0[4];
            float acc1[4];
            memcpy(acc0, initValue, 4 * sizeof(float));
            memcpy(acc1, initValue, 4 * sizeof(float));
            const int lElement = *nnz++;
            for (auto il = 0; il < lElement; il++) {

                const int diff = *dataOffset++;
                const float a0 = a[0];
                const float a1 = a[1];
                const float wv[4] = {*w++, *w++, *w++, *w++};
                // MNN_PRINT("16-loop: ie:%zu, a offset:%ld, w offset:%ld, c offset:%ld, w value:%f, a value[0-15]:", ie, a - A, w - B - 1, c - C, oneW);
                // formatMatrix(a, {16});
                // MNN_PRINT("\n");
                a = a + diff;
                for (int lane = 0; lane < 4; lane++) {
                    acc0[lane] += a0 * wv[lane];
                    acc1[lane] += a1 * wv[lane];
                }
            }

            for (int lane = 0; lane < 4; lane++) {
                acc0[lane]  = std::max(std::min(maxValue, acc0[lane]), minValue);
                acc1[lane]  = std::max(std::min(maxValue, acc1[lane]), minValue);
            }

            memcpy(c, acc0, 4 * sizeof(float));  // store continuous c
            memcpy(c + 4, acc1, 4 * sizeof(float));
        }
        blockC += (ih >> 2) * cStride;
        for (; ih < h; ih++) {
            auto ihPack = ih >> 2;
            auto ihSubIndex = ih & 0x03;
            auto c = blockC + ihSubIndex;
            const float initValue = nullptr != bias ? bias[ih] : 0;
            float acc0 = initValue;
            float acc1 = initValue;

            const int lElement = *nnz++;
            for (auto il = 0; il < lElement; il++) {
                const int diff = *dataOffset++;
                const float a0 = a[0];
                const float a1 = a[1];
                const float oneW = *w++;
                // MNN_PRINT("2-loop: ie:%zu, a offset:%ld, w offset:%ld, c offset:%ld, w value:%f, a value[0-1]:", ie, a - A, w - B - 1, c - C, oneW);
                // formatMatrix(a, {2});
                // MNN_PRINT("\n");
                a = a + diff;
                acc0 += a0 * oneW;
                acc1 += a1 * oneW;
            }
            acc0  = std::max(std::min(maxValue, acc0), minValue);
            acc1  = std::max(std::min(maxValue, acc1), minValue);
            // how to store faster: st4 / transpose /
            c[0] = acc0;
            c[4] = acc1;
        }
        ie += 2;
        a += 2;
    }
    if (eSize & 0x01) {
        const int* dataOffset = dataOffsetMap;
        const int diff = *dataOffset++;
        // const float* a = blockA + diff;
        a += diff;
        const float* w = B;
        float* blockC = C + (ie << 2);
        const unsigned int* nnz = NNZMap;

        size_t ih = 0;
        for (; ih < (h & (~0x03)); ih += sparseBlockOC) {
            auto ihPack = ih >> 2;
            auto c = blockC + ihPack * cStride;
            float initValue[4] = {0, 0, 0, 0};
            if (nullptr != bias) {
                memcpy(initValue, bias + ih, 4 * sizeof(float));
            }
            float acc0[4];
            memcpy(acc0, initValue, 4 * sizeof(float));
            const int lElement = *nnz++;
            for (auto il = 0; il < lElement; il++) {

                const int diff = *dataOffset++;
                const float a0 = a[0];
                const float wv[4] = {*w++, *w++, *w++, *w++};
                // MNN_PRINT("16-loop: ie:%zu, a offset:%ld, w offset:%ld, c offset:%ld, w value:%f, a value[0-15]:", ie, a - A, w - B - 1, c - C, oneW);
                // formatMatrix(a, {16});
                // MNN_PRINT("\n");
                a = a + diff;
                for (int lane = 0; lane < 4; lane++) {
                    acc0[lane] += a0 * wv[lane];
                }
            }

            for (int lane = 0; lane < 4; lane++) {
                acc0[lane]  = std::max(std::min(maxValue, acc0[lane]), minValue);
            }
            memcpy(c, acc0, 4 * sizeof(float));  // store continuous c
        }
        blockC += (ih >> 2) * cStride;
        for (; ih < h; ih++) {
            auto ihSubIndex = ih & 0x03;
            auto c = blockC + ihSubIndex;
            const float initValue = nullptr != bias ? bias[ih] : 0;
            float acc0 = initValue;

            const int lElement = *nnz++;
            for (auto il = 0; il < lElement; il++) {
                const int diff = *dataOffset++;
                const float a0 = a[0];
                const float oneW = *w++;

                // MNN_PRINT("1-loop: ie:%zu, a offset:%ld, c offset:%ld, w offset:%ld, w value:%f, a value[0]:", ie, a - A, w - B - 1, c - C, oneW);
                // formatMatrix(a, {1});
                // MNN_PRINT("\n");
                a = a + diff;
                acc0 += a0 * oneW;
            }
            acc0  = std::max(std::min(maxValue, acc0), minValue);
            // how to store faster: st4 / transpose /
            c[0] = acc0;
        }
        ie += 1;
        // a += 1;
    }

    return;
}

#endif

#ifndef MNN_USE_SSE
#ifndef MNN_USE_NEON
void MNNTranspose32Bit(int32_t* dstO, const int32_t* srcO, int32_t* dim) {
    int w = dim[0];
    int h = dim[1];
    int srcStride = dim[2];
    int dstStride = dim[3];
    for (int i=0; i<h; ++i) {
        auto si = srcO + i;
        auto di = dstO + i * dstStride;
        for (int j=0; j<w; ++j) {
            auto sj = si + j * srcStride;
            auto dj = di + j;
            *dj = *sj;
        }
    }
}
void MNNTranspose16Bit(int16_t* dstO, const int16_t* srcO, int32_t* dim) {
    int w = dim[0];
    int h = dim[1];
    int srcStride = dim[2];
    int dstStride = dim[3];
    for (int i=0; i<h; ++i) {
        auto si = srcO + i;
        auto di = dstO + i * dstStride;
        for (int j=0; j<w; ++j) {
            auto sj = si + j * srcStride;
            auto dj = di + j;
            *dj = *sj;
        }
    }
}
#endif
void MNNFunctionInit() {
    // Do nothing
}
#endif

#ifdef MNN_USE_NEON
#include <arm_neon.h>
#endif

#define UNIT 4
using Vec4 = MNN::Math::Vec<float, 4>;

#ifndef MNN_USE_NEON

#ifndef MNN_USE_SSE

void MNNCopyC4WithStride(const float* source, float* dest, size_t srcStride, size_t dstStride, size_t count) {
    for (int i = 0; i < count; ++i) {
        auto s = source + i * srcStride;
        auto d = dest + i * dstStride;
        for (int j = 0; j < 4; ++j) {
            d[j] = s[j];
        }
    }
}

void MNNAddC4WithStride(const float* source, float* dest, size_t srcStride, size_t dstStride, size_t count) {
    for (int i = 0; i < count; ++i) {
        auto s = source + i * srcStride;
        auto d = dest + i * dstStride;
        for (int j = 0; j < 4; ++j) {
            d[j] += s[j];
        }
    }
}

void MNNReluWithSlopeChannel(float* dst, const float* src, const float* slope, size_t sizeQuad, size_t depthQuad) {
    for (int j = 0; j < depthQuad; j++) {
        const float* slopeZ = slope + 4 * j;
        const float* srcZ   = src + 4 * j * sizeQuad;
        float* dstZ         = dst + 4 * j * sizeQuad;
        for (int i = 0; i < sizeQuad; i++) {
            for (int c = 0; c < 4; c++) {
                if (srcZ[4 * i + c] < 0) {
                    dstZ[4 * i + c] = srcZ[4 * i + c] * slopeZ[c];
                } else {
                    dstZ[4 * i + c] = srcZ[4 * i + c];
                }
            }
        }
    }
}

void MNNPackC4(float* dst, const float* src, size_t area, size_t depth, int* areaOffset) {
    MNNPackC4Common<float>(dst, src, area, depth, areaOffset);
}

void MNNUnpackC4(float* dst, const float* src, size_t area, size_t depth, int* areaOffset) {
    MNNUnpackC4Common<float>(dst, src, area, depth, areaOffset);
}

void MNNExpC8(float* dest, const float* source, float* offset, const float* parameters, size_t countC8) {
    auto count = countC8 * 8;
    auto param = parameters[0];
    float xLimit = 87;
    float summer = offset[3];
    for (int i = 0; i < count; ++i) {
        auto x         = source[i] * offset[0] + offset[2];
        x = ALIMAX(x, -xLimit);
        x = ALIMIN(x, xLimit);
        int div        = (x * parameters[1]);
        int div2       = (div + 127) << 23;
        auto xReamin   = x - div * param;
        float expBasic = *(float*)(&div2);
        auto t = xReamin;
        auto expRemain =
            ((((parameters[7] * t + parameters[6]) * t + parameters[5]) * t + parameters[4]) * t + parameters[3]) * t +
            parameters[2];
        dest[i] = expBasic * expRemain + offset[1];
        summer+= dest[i];
    }
    offset[3] = summer;
}

void MNNSoftmax(float* dest, const float* source, size_t size) {
    float maxValue = ALIMAX(source[0], source[1]);
    for (int i = 2; i < size; ++i) {
        maxValue = ALIMAX(maxValue, source[i]);
    }
    float xLimit = 87, param = 0.6931471805599453, sumValue = 0.f;
    for (int i = 0; i < size; ++i) {
        auto x         = source[i] - maxValue;
        x = x > -xLimit ? x : -xLimit;
        x = x < xLimit ? x : xLimit;

        int div        = (x / param);
        int div2       = (div + 127) << 23;
        auto xReamin   = x - div * param;
        float expBasic = *(float*)(&div2);

        auto t         = xReamin;
        auto expRemain = ((((1.0f / 120 * t + 1.0f / 24) * t + 1.0f / 6) * t + 0.5f) * t + 1.0f) * t + 1.0f;
        dest[i]  = expBasic * expRemain;
        sumValue += dest[i];
    }
    sumValue = 1.f / sumValue;
    for (int i = 0; i < size; ++i) {
        dest[i] *= sumValue;
    }
}

void MNNReluInt8(int8_t* dst, const int8_t* src, size_t size, ssize_t zeroPoint) {
    for (int i = 0; i < size; ++i) {
        if (src[i] < zeroPoint) {
            dst[i] = zeroPoint;
        } else {
            dst[i] = src[i];
        }
    }
}
#endif // no MNN_USE_SSE

void MNNMaxFloat(float* input, float* maxBuffer, int32_t inputCountUnit) {
    for (int i = 0; i < inputCountUnit; i++) {
        for (int j = 0; j < UNIT; j++) {
            for (int m = 0; m < 2; m++) {
                maxBuffer[j] = std::max(input[i * UNIT * 2 + j * 2 + m], maxBuffer[j]);
            }
        }
    }
}
void MNNMinFloat(float* input, float* minBuffer, int32_t inputCountUnit) {
    for (int i = 0; i < inputCountUnit; i++) {
        for (int j = 0; j < UNIT; j++) {
            for (int m = 0; m < 2; m++) {
                minBuffer[j] = std::min(input[i * UNIT * 2 + j * 2 + m], minBuffer[j]);
            }
        }
    }
}
void MNNScaleAndAddBias(float* dst, const float* src, const float* bias, const float* alpha, size_t planeNumber,
                        size_t biasNumber) {
    for (int z = 0; z < biasNumber; ++z) {
        float* dstZ         = dst + planeNumber * 4 * z;
        const float* srcZ   = src + planeNumber * 4 * z;
        auto biasZ = Vec4::load(bias + 4 * z);
        auto alphaZ = Vec4::load(alpha + 4 * z);
        for (int p = 0; p < planeNumber; ++p) {
            float* dstX       = dstZ + 4 * p;
            const float* srcX = srcZ + 4 * p;
            Vec4::save(dstX, (Vec4::load(srcX) * alphaZ) + biasZ);
        }
    }
}



void MNNUInt8ToInt16WithOffsetC4Common(int16_t* dst, const uint8_t* src, size_t zeroPoint, size_t sizeQuad,
                                       size_t dstStride, size_t srcStride) {
    dstStride /= sizeof(int16_t);
    srcStride /= sizeof(uint8_t);
    for (int z = 0; z < sizeQuad; ++z) {
        auto dstZ = dst + dstStride * z;
        auto srcZ = src + srcStride * z;
        for (int j = 0; j < 4; ++j) {
            dstZ[j] = (int16_t)((int32_t)srcZ[j] - (int32_t)zeroPoint);
        }
    }
}

void MNNUInt8ToInt16WithOffsetC4Fast(int16_t* colAddr, const uint8_t* srcStart, size_t zeroPoint, size_t sizeQuad,
                                     size_t depthQuad, size_t dstZStep, size_t srcZStep) {
    dstZStep /= sizeof(int16_t);
    srcZStep /= sizeof(uint8_t);
    for (int sz = 0; sz < depthQuad; ++sz) {
        auto dstZ = colAddr + sz * dstZStep;
        auto srcZ = srcStart + sz * srcZStep;
        MNNUInt8ToInt16WithOffsetC4Common(dstZ, srcZ, zeroPoint, sizeQuad, 4 * sizeof(int16_t), 4 * sizeof(uint8_t));
    }
}

void MNNPowC8(float* dest, const float* source, const float* powfParam, size_t betaInt, size_t countC8) {
    const int count          = countC8 * 8;
    const float powfConstant = powfParam[6];
    for (int i = 0; i < count; ++i) {
        float result = 1, x, xInv = 1 / source[i];
        for (int j = 0; j < betaInt; result *= xInv, ++j)
            ;
        for (x = source[i]; x >= 1.25; x /= 1.5, result *= powfConstant)
            ;
        float t = x - 1;
        float powRemain =
            powfParam[0] +
            t * (powfParam[1] + t * (powfParam[2] + t * (powfParam[3] + t * (powfParam[4] + t * powfParam[5]))));
        result *= powRemain;
        dest[i] = result;
    }
}
#endif // no MNN_USE_NEON

void MNNGridSampleComputeCord(float* dst, const float* src, size_t inH, size_t inW, size_t outH, size_t outW, size_t stride, bool alignCorners) {
    float a = alignCorners ? 1.0f : 0.0f;
    float b = alignCorners ? 0.0f : 1.0f;
    for (auto h = 0; h < outH; ++h) {
        auto __gridPtr = src + h * stride;
        auto cordH = dst + h * outW * 2;
        for (auto w = 0; w < outW; ++w) {
            auto x = __gridPtr[2 * w + 0];
            auto y = __gridPtr[2 * w + 1];
            cordH[2 * w + 0] = ((1 + x) * (inW - a) - b) * 0.5f;
            cordH[2 * w + 1] = ((1 + y) * (inH - a) - b) * 0.5f;
        }
    }
}
void MNNGridSampleComputeCord3D(float* dst, const float* src, size_t inD, size_t inH, size_t inW, size_t outD, size_t outH, size_t outW, size_t strideD, size_t strideH, bool alignCorners) {
    float a = alignCorners ? 1.0f : 0.0f;
    float b = alignCorners ? 0.0f : 1.0f;
    for (auto d = 0; d < outD; ++d) {
        for (auto h = 0; h < outH; ++h) {
            auto __gridPtr = src + d * strideD + h * strideH;
            auto cordH = dst + (d * outH + h) * outW * 3;
            for (auto w = 0; w < outW; ++w) {
                auto x = __gridPtr[3 * w + 0];
                auto y = __gridPtr[3 * w + 1];
                auto z = __gridPtr[3 * w + 2];
                cordH[3 * w + 0] = ((1 + x) * (inW - a) - b) * 0.5f;
                cordH[3 * w + 1] = ((1 + y) * (inH - a) - b) * 0.5f;
                cordH[3 * w + 2] = ((1 + z) * (inD - a) - b) * 0.5f;
            }
        }
    }
}

#ifndef MNN_USE_SSE
void MNNNorm(float *dst, const float *src, const float *gamma, const float *beta, float epsilon, size_t size, bool RMSNorm) {
    float mean = 0;
    if(false == RMSNorm){
        float sum = 0.f;
        for (int j = 0; j < size; ++j) {
            sum += src[j];
        }
        mean = sum / size;
    }
    float square_sum = 0.f;
    for (int j = 0; j < size; ++j) {
        square_sum += (src[j] - mean) * (src[j] - mean);
    }
    float variable = square_sum / size;
    variable = 1.f / std::sqrt(variable + epsilon);

    if (gamma && beta) {
        for (int j = 0; j < size; ++j) {
            dst[j] = (src[j] - mean) * variable * gamma[j] + beta[j];
        }
    } else {
        for (int j = 0; j < size; ++j) {
            dst[j] = (src[j] - mean) * variable;
        }
    }
}
#endif

void MNNRoiPoolingMax(float* dst, const float* src, int hLen, int wLen, int iw) {
    Vec4 max = Vec4(-FLT_MAX);
    for (int h = 0; h < hLen; h++, src += iw * UNIT) {
        for (int w = 0; w < wLen; w++) {
            Vec4 in = Vec4::load(src + w * UNIT);
            max = Vec4::max(max, in);
        }
    }
    Vec4::save(dst, max);
}

void MNNRoiAlignMax(float* dst, const float* src, const std::vector<std::vector<int>> &vecPos, const std::vector<std::vector<float>> &vecArea, int samplingRatioArea, int pooledHeight, int pooledWidth) {
    for (int h = 0; h < pooledHeight; ++h, dst += pooledWidth * UNIT) {
        int preCalcIdx = h * pooledWidth * samplingRatioArea;
        for (int w = 0; w < pooledWidth; ++w) {
            Vec4 res = Vec4(-FLT_MAX);
            for (int i = 0; i < samplingRatioArea; ++i) {
                const std::vector<int>& pos    = vecPos[preCalcIdx];
                const std::vector<float>& area = vecArea[preCalcIdx];

                Vec4 val0 = Vec4::load(src + pos[0] * UNIT);
                Vec4 val1 = Vec4::load(src + pos[1] * UNIT);
                Vec4 val2 = Vec4::load(src + pos[2] * UNIT);
                Vec4 val3 = Vec4::load(src + pos[3] * UNIT);
                Vec4 mla  = val0 * area[0];
                mla       = Vec4::fma(mla, val1, area[1]);
                mla       = Vec4::fma(mla, val2, area[2]);
                mla       = Vec4::fma(mla, val3, area[3]);
                res       = Vec4::max(res, mla);
                preCalcIdx++;
            }
            Vec4::save(dst + w * UNIT, res);
        }
    }
}

void MNNRoiAlignAvg(float* dst, const float* src, const std::vector<std::vector<int>> &vecPos, const std::vector<std::vector<float>> &vecArea, int samplingRatioArea, int pooledHeight, int pooledWidth) {
    float invSamplingCnt = 1.f / samplingRatioArea;
    for (int h = 0; h < pooledHeight; ++h, dst += pooledWidth * UNIT) {
        int preCalcIdx = h * pooledWidth * samplingRatioArea;
        for (int w = 0; w < pooledWidth; ++w) {
            Vec4 res = Vec4(0.f);
            for (int i = 0; i < samplingRatioArea; ++i) {
                const std::vector<int>& pos    = vecPos[preCalcIdx];
                const std::vector<float>& area = vecArea[preCalcIdx];

                Vec4 val0 = Vec4::load(src + pos[0] * UNIT);
                Vec4 val1 = Vec4::load(src + pos[1] * UNIT);
                Vec4 val2 = Vec4::load(src + pos[2] * UNIT);
                Vec4 val3 = Vec4::load(src + pos[3] * UNIT);
                Vec4 mla  = val0 * area[0];
                mla       = Vec4::fma(mla, val1, area[1]);
                mla       = Vec4::fma(mla, val2, area[2]);
                mla       = Vec4::fma(mla, val3, area[3]);
                res       += mla;
                preCalcIdx++;
            }
            res = res * invSamplingCnt;
            Vec4::save(dst + w * UNIT, res);
        }
    }
}

void MNNPackC4Uint8(uint8_t* dst, const uint8_t* src, size_t area,size_t depth, int* areaOffset) {
    MNNPackC4Common(dst, src, area, depth, areaOffset);
}

void MNNUnpackC4Uint8(uint8_t* dst, const uint8_t* src, size_t area,size_t depth, int* areaOffset) {
    MNNUnpackC4Common(dst, src, area, depth, areaOffset);
}

void MNNUnpackTransposeUint8(uint8_t* dst, const uint8_t* src, size_t area,size_t depth, int* areaOffset) {
    if (depth == 4) {
        ::memcpy(dst, src, area * depth * sizeof(uint8_t));
        return;
    }
#ifdef MNN_USE_NEON
    if (depth == 3) {
        uint8x16x4_t rgba;
        rgba.val[3] = vdupq_n_u8(0);
        int sta     = 0;
        int staC16  = (int)area / 16;
        for (int i = 0; i < staC16; sta += 16, ++i) {
            auto rgb    = vld3q_u8(src + sta * 3);
            rgba.val[0] = rgb.val[0];
            rgba.val[1] = rgb.val[1];
            rgba.val[2] = rgb.val[2];
            vst4q_u8(dst + 4 * sta, rgba);
        }
        sta = staC16 * 16;

        for (; sta < area; ++sta) {
            auto s = src + sta * 3;
            auto d = dst + sta * 4;
            d[0]   = s[0];
            d[1]   = s[1];
            d[2]   = s[2];
            d[3]   = 0;
        }

        return;
    }
    if (depth == 1) {
        uint8x16x4_t rgba;
        rgba.val[1] = vdupq_n_u8(0);
        rgba.val[2] = vdupq_n_u8(0);
        rgba.val[3] = vdupq_n_u8(0);
        int sta     = 0;
        for (; sta < area; sta += 16) {
            rgba.val[0] = vld1q_u8(src + sta);
            vst4q_u8(dst + 4 * sta, rgba);
        }

        for (; sta < area; ++sta) {
            auto s = src + sta;
            auto d = dst + sta * 4;
            d[0]   = s[0];
            d[1]   = 0;
            d[2]   = 0;
            d[3]   = 0;
        }

        return;
    }
#endif
    int c      = (int)depth;
    int cDiv4  = c / 4;
    int cAlign = cDiv4 * 4;

    if (cAlign == c) {
        for (int hi = 0; hi < area; ++hi) {
            auto srcHeight = reinterpret_cast<const int32_t*>(src + hi * c);
            auto dstHeight = reinterpret_cast<int32_t*>(dst + hi * 4);
            for (int ci = 0; ci < cDiv4; ++ci) {
                dstHeight[ci * areaOffset[1]] = srcHeight[ci];
            }
        }
        return;
    } else {
        for (int hi = 0; hi < area; ++hi) {
            auto srcHeight = src + hi * c;
            auto dstHeight = dst + hi * 4;
            for (int ci = 0; ci < cDiv4; ++ci) {
                dstHeight[ci * areaOffset[1] * 4 + 0] = srcHeight[ci * 4 + 0];
                dstHeight[ci * areaOffset[1] * 4 + 1] = srcHeight[ci * 4 + 1];
                dstHeight[ci * areaOffset[1] * 4 + 2] = srcHeight[ci * 4 + 2];
                dstHeight[ci * areaOffset[1] * 4 + 3] = srcHeight[ci * 4 + 3];
            }
        }
    }

    int cReamin   = c - cAlign;
    auto srcAlign = src + cAlign;
    auto dstAlign = dst + areaOffset[1] * cAlign;

    for (int hi = 0; hi < area; ++hi) {
        auto srcHeight = srcAlign + hi * c;
        auto dstHeight = dstAlign + hi * 4;
        for (int i = 0; i < 4; ++i) {
            dstHeight[i] = 0;
        }
        for (int ci = 0; ci < cReamin; ++ci) {
            dstHeight[ci] = srcHeight[ci];
        }
    }
}

void MNNUnpackTranspose(float* dst, const float* src, size_t area, size_t depth, int* areaOffset) {
    int srcAreaOffset = areaOffset[0];
    int dstAreaOffset = areaOffset[1];
#ifdef MNN_USE_NEON
    if (1 == depth) {
        auto zeroValue = vmovq_n_f32(0.0f);
        int areaC4     = (int)area / 4;
        int remain     = areaC4 * 4;
        for (int i = 0; i < areaC4; ++i) {
            auto srcCur   = src + 4 * i;
            auto dstCur   = dst + 16 * i;
            auto srcValue = vld1q_f32(srcCur);
            float32x4x4_t dstValue;
            dstValue.val[0] = srcValue;
            dstValue.val[1] = zeroValue;
            dstValue.val[2] = zeroValue;
            dstValue.val[3] = zeroValue;
            vst4q_f32(dstCur, dstValue);
        }
        for (int i = remain; i < area; ++i) {
            dst[4 * i + 0] = src[i];
            dst[4 * i + 1] = 0.0f;
            dst[4 * i + 2] = 0.0f;
            dst[4 * i + 3] = 0.0f;
        }
        return;
    }
    if (3 == depth) {
        auto zeroValue = vmovq_n_f32(0.0f);
        int areaC4     = (int)area / 4;
        int remain     = areaC4 * 4;
        for (int i = 0; i < areaC4; ++i) {
            auto srcCur   = src + 12 * i;
            auto dstCur   = dst + 16 * i;
            auto srcValue = vld3q_f32(srcCur);
            float32x4x4_t dstValue;
            dstValue.val[0] = srcValue.val[0];
            dstValue.val[1] = srcValue.val[1];
            dstValue.val[2] = srcValue.val[2];
            dstValue.val[3] = zeroValue;
            vst4q_f32(dstCur, dstValue);
        }
        for (int i = remain; i < area; ++i) {
            dst[4 * i + 0] = src[3 * i + 0];
            dst[4 * i + 1] = src[3 * i + 1];
            dst[4 * i + 2] = src[3 * i + 2];
            dst[4 * i + 3] = 0.0f;
        }
        return;
    }
#endif
    int c      = (int)depth;
    int cDiv4  = c / 4;
    int cAlign = cDiv4 * 4;
    for (int hi = 0; hi < area; ++hi) {
        const float* srcHeight = src + hi * c;
        float* dstHeight       = dst + hi * 4;
        for (int ci = 0; ci < cDiv4; ++ci) {
            Vec4::save(dstHeight + 4 * ci * dstAreaOffset, Vec4::load(srcHeight + 4 * ci));
        }
    }

    if (cAlign == c) {
        return;
    }

    int cReamin   = c - cAlign;
    auto srcAlign = src + cAlign;
    auto dstAlign = dst + dstAreaOffset * cAlign;

#ifdef MNN_USE_NEON
    auto zeroVector = vdupq_n_f32(0.0f);
#endif

    for (int hi = 0; hi < area; ++hi) {
        const float* srcHeight = srcAlign + hi * c;
        float* dstHeight       = dstAlign + hi * 4;
#ifdef MNN_USE_NEON
        vst1q_f32(dstHeight, zeroVector);
#else
        for (int i = 0; i < 4; ++i) {
            dstHeight[i] = 0;
        }
#endif
        for (int ci = 0; ci < cReamin; ++ci) {
            dstHeight[ci] = srcHeight[ci];
        }
    }
}

void MNNPackTransposeUint8(uint8_t* dst, const uint8_t* src, size_t area,size_t depth, int* areaOffset) {
    int c      = (int)depth;
    int cDiv4  = c / 4;
    int cAlign = cDiv4 * 4;
    if (cAlign == c) {
        int32_t* dst32       = (int32_t*)dst;
        const int32_t* src32 = (int32_t*)src;
        for (int hi = 0; hi < area; ++hi) {
            auto srcHeight = src32 + hi;
            auto dstHeight = dst32 + hi * cDiv4;
            for (int ci = 0; ci < cDiv4; ++ci) {
                dstHeight[ci] = srcHeight[ci * areaOffset[0]];
            }
        }
        return;
    }

    for (int hi = 0; hi < area; ++hi) {
        auto srcHeight = src + hi * 4;
        auto dstHeight = dst + hi * c;
        for (int ci = 0; ci < cDiv4; ++ci) {
            for (int i = 0; i < 4; ++i) {
                dstHeight[ci * 4 + i] = srcHeight[4 * ci * areaOffset[0] + i];
            }
        }
    }

    int cReamin   = c - cAlign;
    auto srcAlign = src + areaOffset[0] * cAlign;
    auto dstAlign = dst + cAlign;

    for (int hi = 0; hi < area; ++hi) {
        auto srcHeight = srcAlign + hi * 4;
        auto dstHeight = dstAlign + hi * c;

        for (int ci = 0; ci < cReamin; ++ci) {
            dstHeight[ci] = srcHeight[ci];
        }
    }
}

void MNNPackTranspose(float* dst, const float* src, size_t area, size_t depth, int* areaOffset) {
#if defined(MNN_USE_NEON)
    if (3 == depth) {
        int areaC4     = (int)area / 4;
        int remain     = areaC4 * 4;
        for (int i = 0; i < areaC4; ++i) {
            auto srcCur   = src + 16 * i;
            auto dstCur   = dst + 12 * i;
            auto srcValue = vld4q_f32(srcCur);
            float32x4x3_t dstValue;
            dstValue.val[0] = srcValue.val[0];
            dstValue.val[1] = srcValue.val[1];
            dstValue.val[2] = srcValue.val[2];
            vst3q_f32(dstCur, dstValue);
        }
        for (int i = remain; i < area; ++i) {
            dst[3 * i + 0] = src[4 * i + 0];
            dst[3 * i + 1] = src[4 * i + 1];
            dst[3 * i + 2] = src[4 * i + 2];
        }
        return;
    }
#elif defined(MNN_USE_SSE)
    if (3 == depth) {
        if (area < 1) return;
        for (int i = 0; i < area - 1; ++i) {
            auto srcValue = Vec4::load(src + 4 * i);
            Vec4::save(dst + 3 * i, srcValue);
        }
        for (int i = 0; i < 3; ++i) {
            dst[3 * (area - 1) + i] = src[4 * (area - 1) + i];
        }
        return;
    }
#endif
    int c      = (int)depth;
    int cDiv4  = c / 4;
    int cAlign = cDiv4 * 4;
    auto srcArea = areaOffset[0];
    for (int hi = 0; hi < area; ++hi) {
        const float* srcHeight = src + hi * 4;
        float* dstHeight       = dst + hi * c;
        for (int ci = 0; ci < cDiv4; ++ci) {
            Vec4::save(dstHeight + 4 * ci, Vec4::load(srcHeight + 4 * ci * srcArea));
        }
    }

    if (cAlign == c) {
        return;
    }

    int cReamin   = c - cAlign;
    auto srcAlign = src + srcArea * cAlign;
    auto dstAlign = dst + cAlign;

    for (int hi = 0; hi < area; ++hi) {
        const float* srcHeight = srcAlign + hi * 4;
        float* dstHeight       = dstAlign + hi * c;

        for (int ci = 0; ci < cReamin; ++ci) {
            dstHeight[ci] = srcHeight[ci];
        }
    }
}

void MNNExp(float* dst, const float* src, float* offset, size_t dataSize) {
    int countC8        = static_cast<int32_t>(dataSize) / 8;
    int remain = static_cast<int32_t>(dataSize) % 8;
    float parameters[] = {
        (float)logf(2.0f), 1.0f / (float)logf(2.0f), 1.0f, 1.0f, 0.5f, 1.0f / 6.0f, 1.0f / 24.0f, 1.0f / 120.0f};
    if (countC8 > 0) {
        // Align to eight so asm is easier to write
        MNNExpC8(dst, src, offset, parameters, countC8);
    }
    if (remain > 0) {
        auto param = parameters[0];
        float xLimit = 87;
        float summer = offset[3];
        auto source = src + countC8 * 8;
        auto dest = dst + countC8 * 8;
        for (int i = 0; i < remain; ++i) {
            auto x         = source[i] * offset[0] + offset[2];
            x = ALIMAX(x, -xLimit);
            x = ALIMIN(x, xLimit);
            int div        = (x * parameters[1]);
            int div2       = (div + 127) << 23;
            auto xReamin   = x - div * param;
            float expBasic = *(float*)(&div2);
            auto t = xReamin;
            auto expRemain =
                ((((parameters[7] * t + parameters[6]) * t + parameters[5]) * t + parameters[4]) * t + parameters[3]) * t +
                parameters[2];
            dest[i] = expBasic * expRemain + offset[1];
            summer+= dest[i];
        }
        offset[3] = summer;
    }
}

// Lambert's series with 7 divisions
// reference from
// https://varietyofsound.wordpress.com/2011/02/14/efficient-tanh-computation-using-lamberts-continued-fraction/
inline float tanhf_poly(float value) {
    if (value > 5.0) {
        return 1.0;
    } else if (value <= -5.0) {
        return -1.0;
    } else {
        float x2 = value * value;
        float a  = value * (135135.0f + x2 * (17325.0f + x2 * (378.0f + x2)));
        float b  = 135135.0f + x2 * (62370.0f + x2 * (3150.0f + x2 * 28.0f));
        return a / b;
    }
}
void MNNTanh(float* dst, const float* src, size_t dataSize) {
    /* Origin Code
    for (int i = 0; i < dataSize; i++) {
        // outputData[i] = 1 - 2 / (expf(2 * inputData[i]) + 1);
        dst[i] = tanhf_poly(src[i]);
    }
     */
    float offset[4] = {
        -2.0f,
        0.0f,
        0.0f,
        0.0f
    };
    MNNExp(dst, src, offset, dataSize);
    for (int i = 0; i < dataSize; i++) {
        // outputData[i] = 1 - 2 / (expf(2 * inputData[i]) + 1);
        auto expX2 = dst[i];
        dst[i] = (1.0f - expX2) / (1.0f + expX2);
    }
}

void MNNReluWithSlope(float* dst, const float* src, size_t sizeQuad, float slope) {
    float slopeValue[4];
    for (int i=0; i<4; ++i) {
        slopeValue[i] = slope;
    }
    MNNReluWithSlopeChannel(dst, src, slopeValue, sizeQuad, 1);
}

void MNNReluWithSlopeCommon(float* dst, const float* src, size_t size, float slope) {
    int sizeQuad = static_cast<int32_t>(size) / 4;
    int remain = static_cast<int32_t>(size) % 4;
    if (sizeQuad > 0) {
        MNNReluWithSlope(dst, src, sizeQuad, slope);
    }
    if (remain > 0) {
        float intmp[4] = {0}, outmp[4] = {0};
        ::memcpy(intmp, src + sizeQuad * 4, remain * sizeof(float));
        MNNReluWithSlope(outmp, intmp, 1, slope);
        ::memcpy(dst + sizeQuad * 4, outmp, remain * sizeof(float));
    }
}

void MNNHardSwishCommon(float* dst, const float* src, size_t size) {
    int sizeQuad = static_cast<int32_t>(size / 4);
    int remain = static_cast<int32_t>(size) % 4;
#ifdef MNN_USE_SSE
    if (sizeQuad > 0) {
        MNNHardSwish(dst, src, sizeQuad);
    }
    if (remain > 0) {
        float intmp[4] = {0}, outmp[4] = {0};
        ::memcpy(intmp, src + sizeQuad * 4, remain * sizeof(float));
        MNNHardSwish(outmp, intmp, 1);
        ::memcpy(dst + sizeQuad * 4, outmp, remain * sizeof(float));
    }
#else
#ifdef MNN_USE_NEON
    float32x4_t zero = vdupq_n_f32(0.f);
    float32x4_t three = vdupq_n_f32(3.f);
    float32x4_t six = vdupq_n_f32(6.f);
    float32x4_t divsix = vdupq_n_f32(1.0f/6.f);
    for (int i = 0; i < sizeQuad; i++) {
        auto x = vld1q_f32(src + 4 * i);
        auto y = vmulq_f32(vmulq_f32(x, vminq_f32(vmaxq_f32(vaddq_f32(x, three), zero), six)), divsix);
        vst1q_f32(dst + 4 * i, y);
    }
    if (remain > 0) {
        float intmp[4] = {0}, outmp[4] = {0};
        ::memcpy(intmp, src + sizeQuad * 4, remain * sizeof(float));
        auto x = vld1q_f32(intmp);
        auto y = vmulq_f32(vmulq_f32(x, vminq_f32(vmaxq_f32(vaddq_f32(x, three), zero), six)), divsix);
        vst1q_f32(outmp, y);
        ::memcpy(dst + sizeQuad * 4, outmp, remain * sizeof(float));
    }
#else
    for (int j = 0; j < size; j++) {
        if (src[j] <= -3) {
            dst[j] = 0;
        } else if (src[j] >= 3){
            dst[j] = src[j];
        } else {
            dst[j] = src[j] * (src[j] + 3) / 6.f;
        }
    }
#endif
#endif
}

void MNNGeluStandardCommon(float* dst, const float* src, size_t size) {
    for (int i = 0; i < size; i++) {
        dst[i] = (erf(src[i] * 0.7071067932881648) + 1) * src[i] * 0.5;
    }
}

void MNNGeluCommon(float* dst, const float* src, size_t size) {
    int sizeQuad = static_cast<int32_t>(size / 8);
    int remain = static_cast<int32_t>(size) % 8;
#if defined(MNN_USE_SSE) || defined(MNN_USE_NEON)
    float parameters[8] = {0.044715f, 0.79788458f, 378.f, 17325.f, 135135.f, 28.f, 3150.f, 62370.f};
    if (sizeQuad > 0) {
        MNNGelu(dst, src, sizeQuad, parameters);
    }
    if (remain > 0) {
        float intmp[8] = {0};
        float outmp[8] = {0};
        ::memcpy(intmp, src + 8 * sizeQuad, remain * sizeof(float));
        MNNGelu(outmp, intmp, 1, parameters);
        ::memcpy(dst + 8 * sizeQuad, outmp, remain * sizeof(float));
    }
#else
    auto tanhf_poly = [](float value) -> float {
        if (value > 5.0f) {
            return 1.0f;
        } else if (value <= -5.0f) {
            return -1.0f;
        } else {
            float x2 = value * value;
            float a  = value * (135135.0f + x2 * (17325.0f + x2 * (378.0f + x2)));
            float b  = 135135.0f + x2 * (62370.0f + x2 * (3150.0f + x2 * 28.0f));
            return a / b;
        }
    };
    for (int i = 0; i < size; i++) {
        float temp = 0.044715f * src[i] * src[i] * src[i];
        temp = 0.79788458f * (temp + src[i]);
        dst[i] = (1.0f + tanhf_poly(temp)) * src[i] * 0.5f;
    }
#endif
}

void MNNScaleAndAddBiasScalar(float* dst, const float* src, float bias, float alpha, size_t number) {
    int numberC4 = (int)number / 4;
    int start = 0;
    if (numberC4 > 0) {
        float biasC4[4] = {
            bias,
            bias,
            bias,
            bias
        };
        float alphaC4[4] = {
            alpha,
            alpha,
            alpha,
            alpha
        };
        MNNScaleAndAddBias(dst, src, biasC4, alphaC4, numberC4, 1);
        start = numberC4 * 4;
    }
    for (int i=start; i<number; ++i) {
        dst[i] = src[i] * alpha + bias;
    }
}
#ifndef MNN_USE_NEON
void MNNAxByClampBroadcastUnit(float* C, const float* A, const float* B, size_t width, size_t cStride, size_t aStride, size_t height, const float* parameters) {
    auto minF = Vec4(parameters[2]);
    auto maxF = Vec4(parameters[3]);
    auto beta = Vec4(parameters[1]);
    for (int y = 0; y < height; ++y) {
        auto a = A + aStride * y;
        auto b = B + 4 * y;
        auto bv = Vec4::load(b);
        auto c = C + cStride * y;
        for (int x = 0; x < width; ++x) {
            auto av = Vec4::load(a + 4 * x);
            auto cv = av + bv * beta;
            cv = Vec4::min(cv, maxF);
            cv = Vec4::max(cv, minF);
            Vec4::save(c + 4 * x, cv);
        }
    }
}
void MNNVectorTop1Float(float* input, float* maxValue, int32_t* maxIndex, size_t inputCountUnit) {
    float maxV = input[0];
    int maxIdx = 0;
    for (int i = 0; i < inputCountUnit; i++) {
        int offset = i * UNIT;
        for (int j = 0; j < UNIT; j++) {
            if (input[offset + j] > maxV) {
                maxV = input[offset + j];
                maxIdx = offset + j;
            }
        }
    }
    maxValue[0] = maxV;
    maxIndex[0] = maxIdx;
}

void MNNVectorTop1Int32(int32_t* input, int32_t* maxValue, int32_t* maxIndex, size_t inputCountUnit) {
    int32_t maxV = input[0];
    int maxIdx = 0;
    for (int i = 0; i < inputCountUnit; i++) {
        int offset = i * UNIT;
        for (int j = 0; j < UNIT; j++) {
            if (input[offset + j] > maxV) {
                maxV = input[offset + j];
                maxIdx = offset + j;
            }
        }
    }
    maxValue[0] = maxV;
    maxIndex[0] = maxIdx;
}

#endif

void MNNComputeMatMulForE_1(const float* A, const float* B, float* C, const float* biasPtr, const MatMulParam* param, size_t tId) {
    auto l = param->l;
    auto h = param->h;
    auto numberThread = param->numberThread;
    auto lC4 = l / 4;
    auto lR = lC4 * 4;
    if (param->BTranspose) {
        for (int y=tId; y<h; y+=numberThread) {
            Vec4 sumValue = Vec4(0.0f);
            auto by = B + y * l;
            for (int x=0; x<lC4; ++x) {
                sumValue = sumValue + Vec4::load(A + x * 4) * Vec4::load(by + x * 4);
            }
            float sumRemain = 0.0f;
            for (int x=lR; x<l; ++x) {
                sumRemain = sumRemain + A[x] * by[x];
            }
            if (nullptr != biasPtr) {
                sumRemain += biasPtr[y];
            }
            C[y] = sumRemain + sumValue[0] + sumValue[1] + sumValue[2] + sumValue[3];
        }
    } else {
        auto hC4 = h / 16;
        auto hR = hC4 * 16;
        for (int y=tId; y<hC4; y+=numberThread) {
            auto bs = B + 16 * y;
            Vec4 sumValue0 = Vec4(0.0f);
            Vec4 sumValue1 = Vec4(0.0f);
            Vec4 sumValue2 = Vec4(0.0f);
            Vec4 sumValue3 = Vec4(0.0f);
            if (biasPtr != nullptr) {
                sumValue0 = Vec4::load(biasPtr + 16 * y + 0);
                sumValue1 = Vec4::load(biasPtr + 16 * y + 4);
                sumValue2 = Vec4::load(biasPtr + 16 * y + 8);
                sumValue3 = Vec4::load(biasPtr + 16 * y + 12);
            }
            auto srcY = A + y * l;
            for (int x=0; x<l; ++x) {
                auto a = Vec4(A[x]);
                sumValue0 = sumValue0 + a * Vec4::load(bs + h * x);
                sumValue1 = sumValue1 + a * Vec4::load(bs + h * x + 4);
                sumValue2 = sumValue2 + a * Vec4::load(bs + h * x + 8);
                sumValue3 = sumValue3 + a * Vec4::load(bs + h * x + 12);
            }
            Vec4::save(C + 16 * y, sumValue0);
            Vec4::save(C + 16 * y + 4, sumValue1);
            Vec4::save(C + 16 * y + 8, sumValue2);
            Vec4::save(C + 16 * y + 12, sumValue3);
        }
        for (int y=hR + tId; y<h; y+=numberThread) {
            auto bs = B + y;
            float sumValue = 0.0f;
            if (biasPtr != nullptr) {
                sumValue = biasPtr[y];
            }
            auto srcY = A + y * l;
            for (int x=0; x<l; ++x) {
                sumValue = sumValue + A[x] * bs[h * x];
            }
            C[y] = sumValue;
        }
    }
}

void MNNComputeMatMulForH_1(const float* A, const float* B, float* C, const float* biasPtr, const MatMulParam* param, size_t tId) {
    int e = param->e;
    int l = param->l;
    int numberThread = param->numberThread;
    if (param->ATranspose) {
        float biasValue = 0.0f;
        if (nullptr != biasPtr) {
            biasValue = *biasPtr;
        }
        auto eC4 = e / 4;
        auto eR = eC4 * 4;
        for (int y=tId; y<eC4; y+=numberThread) {
            Vec4 sumValue = Vec4(biasValue);
            auto srcY = A + y * 4;
            for (int x=0; x<l; ++x) {
                sumValue = sumValue + Vec4::load(srcY + x * e) * Vec4(B[x]);
            }
            Vec4::save(C + 4 * y, sumValue);
        }
        if (0 == tId) {
            for (int y=eR; y<e; ++y) {
                float sumValue = biasValue;
                auto srcY = A + y;
                for (int x=0; x<l; ++x) {
                    sumValue = sumValue + srcY[x * e] * B[x];
                }
                C[y] = sumValue;
            }
        }
        return;
    }
    float biasValue = 0.0f;
    if (nullptr != biasPtr) {
        biasValue = *biasPtr;
    }
    auto lC4 = l / 4;
    auto lR = lC4 * 4;
    for (int y=tId; y<e; y+=numberThread) {
        Vec4 sumValue = Vec4(biasValue);
        auto srcY = A + y * l;
        for (int x=0; x<lC4; ++x) {
            sumValue = sumValue + Vec4::load(srcY + 4 * x) * Vec4::load(B + 4 * x);
        }
        float sumSingle = sumValue[0] + sumValue[1] + sumValue[2] + sumValue[3];
        for (int x=lR; x<l; ++x) {
            sumSingle += srcY[x] * B[x];
        }
        C[y] = sumSingle;
    }
}

void MNNPackC4Int16(int16_t* dst, const int16_t* src, size_t area,size_t depth, int* areaOffset) {
    MNNPackC4Common(dst, src, area, depth, areaOffset);
}

void MNNUnpackC4Int16(int16_t* dst, const int16_t* src, size_t area,size_t depth, int* areaOffset) {
    MNNUnpackC4Common(dst, src, area, depth, areaOffset);
}

void MNNUnpackTransposeInt16(int16_t* dst, const int16_t* src, size_t area,size_t depth, int* areaOffset) {
    if (depth == 4) {
        ::memcpy(dst, src, area * depth * sizeof(int16_t));
        return;
    }
    int c      = (int)depth;
    int cDiv4  = c / 4;
    int cAlign = cDiv4 * 4;
    for (int hi = 0; hi < area; ++hi) {
        auto srcHeight = (src + hi * c);
        auto dstHeight = (dst + hi * 4);
        for (int ci = 0; ci < cDiv4; ++ci) {
            for (int i = 0; i < 4; ++i) {
                dstHeight[ci * areaOffset[1] * 4 + i] = srcHeight[4 * ci + i];
            }
        }
    }

    if (cAlign == c) {
        return;
    }

    int cReamin   = c - cAlign;
    auto srcAlign = src + cAlign;
    auto dstAlign = dst + areaOffset[1] * cAlign;

    for (int hi = 0; hi < area; ++hi) {
        auto srcHeight = srcAlign + hi * c;
        auto dstHeight = dstAlign + hi * 4;
        for (int i = 0; i < 4; ++i) {
            dstHeight[i] = 0;
        }
        for (int ci = 0; ci < cReamin; ++ci) {
            dstHeight[ci] = srcHeight[ci];
        }
    }
}
void MNNPackTransposeInt16(int16_t* dst, const int16_t* src, size_t area,size_t depth, int* areaOffset) {
    int c      = (int)depth;
    int cDiv4  = c / 4;
    int cAlign = cDiv4 * 4;
    if (cAlign == c) {
        int64_t* dst32       = (int64_t*)dst;
        const int64_t* src32 = (int64_t*)src;
        for (int hi = 0; hi < area; ++hi) {
            auto srcHeight = src32 + hi;
            auto dstHeight = dst32 + hi * cDiv4;
            for (int ci = 0; ci < cDiv4; ++ci) {
                dstHeight[ci] = srcHeight[ci * areaOffset[0]];
            }
        }
        return;
    }

    for (int hi = 0; hi < area; ++hi) {
        auto srcHeight = src + hi * 4;
        auto dstHeight = dst + hi * c;
        for (int ci = 0; ci < cDiv4; ++ci) {
            for (int i = 0; i < 4; ++i) {
                dstHeight[ci * 4 + i] = srcHeight[4 * ci * areaOffset[0] + i];
            }
        }
    }

    int cReamin   = c - cAlign;
    auto srcAlign = src + areaOffset[0] * cAlign;
    auto dstAlign = dst + cAlign;

    for (int hi = 0; hi < area; ++hi) {
        auto srcHeight = srcAlign + hi * 4;
        auto dstHeight = dstAlign + hi * c;

        for (int ci = 0; ci < cReamin; ++ci) {
            dstHeight[ci] = srcHeight[ci];
        }
    }
}

void MNNCopyC4Int16WithStride(const float* sourceF, float* destF, size_t srcStride, size_t dstStride, size_t count) {
    auto source = (int16_t*)sourceF;
    auto dest = (int16_t*)destF;
    for (int i = 0; i < count; ++i) {
        auto s = source + i * srcStride;
        auto d = dest + i * dstStride;
        *(int64_t*)(d) = *((int64_t*)s);
    }
}


void MNNSin(float* dst, const float* src, size_t dataSize) {
    for (int i = 0; i < dataSize; i++) {
        dst[i] = sinf(src[i]);
    }
}

void MNNSigmoid(float* dst, const float* src, size_t dataSize) {
    float offset[4] = {
       -1.0f,
        0.0f,
        0.0f,
        0.0f
    };
    MNNExp(dst, src, offset, dataSize);
    for (int i = 0; i < dataSize; ++i) {
        dst[i] = 1.0f / (1.0f + dst[i]);
    }
}

/**
 Modified from https://github.com/alibaba/MNN/pull/1359
 Thanks for https://github.com/hroken
 */
void MNNSigmoidLowp(float* dst, const float* src, size_t dataSize) {
    float offset[4] = {
       -1.0f,
        0.0f,
        0.0f,
        0.0f
    };
    MNNExp(dst, src, offset, dataSize);
#ifdef MNN_USE_NEON
    int dataC4 = static_cast<int32_t>(dataSize) / 4;
    int remain = static_cast<int32_t>(dataSize) % 4;
    float32x4_t value = vdupq_n_f32(1.0f);

    if(dataC4 > 0) {
        float32x4_t out = vld1q_f32(dst);
        // neon optimization for sigmid cpu
        for (int i = 1; i < dataC4; ++i) {
            out = vrecpeq_f32(vaddq_f32(value,out));
            vst1q_f32(dst ,out);
            dst += 4;
            out = vld1q_f32(dst);
        }
        out = vrecpeq_f32(vaddq_f32(value,out));
        vst1q_f32(dst, out);
        dst += 4;
    }
    if (remain > 0) {
        float intmp[4] = {0};
        ::memcpy(intmp, dst, remain * sizeof(float));
        float32x4_t out = vld1q_f32(intmp);
        out = vrecpeq_f32(vaddq_f32(value,out));
        vst1q_f32(intmp, out);
        ::memcpy(dst, intmp, remain * sizeof(float));
    }
#else
    for (int i = 0; i < dataSize; ++i) {
        dst[i] = 1.0f / (1.0f + dst[i]);
    }
#endif
}

void MNNMultiAndDestTransformCommon23(float **cacheLine, const float *weigth, float *dest, int cacheLineSize, int ow, const float* bias, const float* parameters) {
    int unit = ow / 2;
    MNN_ASSERT(cacheLineSize >= 1);
    auto biasF = Vec4::load(bias);
    auto minF = Vec4(parameters[2]);
    auto maxF = Vec4(parameters[3]);
    for (int x = 0; x < unit; ++x) {
        auto offset = 4 * 4 * x;
        int i = 0;
        Vec4 m0     = Vec4::load(weigth + i * 16 + 4 * 0) * Vec4::load(cacheLine[i] + offset + 4 * 0);
        Vec4 m1     = Vec4::load(weigth + i * 16 + 4 * 1) * Vec4::load(cacheLine[i] + offset + 4 * 1);
        Vec4 m2     = Vec4::load(weigth + i * 16 + 4 * 2) * Vec4::load(cacheLine[i] + offset + 4 * 2);
        Vec4 m3     = Vec4::load(weigth + i * 16 + 4 * 3) * Vec4::load(cacheLine[i] + offset + 4 * 3);

        for (i = 1; i < cacheLineSize; ++i) {
            m0 = m0 + Vec4::load(weigth + i * 16 + 4 * 0) * Vec4::load(cacheLine[i] + offset + 4 * 0);
            m1 = m1 + Vec4::load(weigth + i * 16 + 4 * 1) * Vec4::load(cacheLine[i] + offset + 4 * 1);
            m2 = m2 + Vec4::load(weigth + i * 16 + 4 * 2) * Vec4::load(cacheLine[i] + offset + 4 * 2);
            m3 = m3 + Vec4::load(weigth + i * 16 + 4 * 3) * Vec4::load(cacheLine[i] + offset + 4 * 3);
        }

        auto o0 = m0 + m1 + m2 + biasF;
        auto o1 = m1 - m2 + m3 + biasF;
        o0 = Vec4::min(maxF, o0);
        o1 = Vec4::min(maxF, o1);
        o0 = Vec4::max(minF, o0);
        o1 = Vec4::max(minF, o1);
        Vec4::save(dest + 8 * x + 0 * 4, o0);
        Vec4::save(dest + 8 * x + 1 * 4, o1);
    }
    if (unit * 2 < ow) {
        auto offset = 4 * 4 * unit;
        int i = 0;
        Vec4 m0     = Vec4::load(weigth + i * 16 + 4 * 0) * Vec4::load(cacheLine[i] + offset + 4 * 0);
        Vec4 m1     = Vec4::load(weigth + i * 16 + 4 * 1) * Vec4::load(cacheLine[i] + offset + 4 * 1);
        Vec4 m2     = Vec4::load(weigth + i * 16 + 4 * 2) * Vec4::load(cacheLine[i] + offset + 4 * 2);

        for (i = 1; i < cacheLineSize; ++i) {
            m0 = m0 + Vec4::load(weigth + i * 16 + 4 * 0) * Vec4::load(cacheLine[i] + offset + 4 * 0);
            m1 = m1 + Vec4::load(weigth + i * 16 + 4 * 1) * Vec4::load(cacheLine[i] + offset + 4 * 1);
            m2 = m2 + Vec4::load(weigth + i * 16 + 4 * 2) * Vec4::load(cacheLine[i] + offset + 4 * 2);
        }
        auto o0 = m0 + m1 + m2 + biasF;
        o0 = Vec4::min(maxF, o0);
        o0 = Vec4::max(minF, o0);
        Vec4::save(dest + 8 * unit + 0 * 4, o0);
    }
}
extern "C" {
void MNNConvDwF23SourceTransUnit(const float *source, float *dest, size_t unit);
}

void MNNSourceTransformCommonF23(const float *source, float *dest, int unit, int iw, int pad, int su, int eu) {
    for (int x = 0; x < su; ++x) {
        auto dstX = dest + 4 * 4 * x;
        auto sx   = x * 2 - (int)pad;
        auto ex   = sx + 4;

        auto clampSx = std::max(sx, 0);
        auto clampEx = std::min(ex, (int)iw);

        Vec4 v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int i = clampSx; i < clampEx; ++i) {
            v[i - sx] = Vec4::load(source + 4 * i);
        }
        auto m0 = v[0] - v[2];
        auto m1 = v[1] + v[2];
        auto m2 = v[2] - v[1];
        auto m3 = v[3] - v[1];

        Vec4::save(dstX + 4 * 0, m0);
        Vec4::save(dstX + 4 * 1, m1);
        Vec4::save(dstX + 4 * 2, m2);
        Vec4::save(dstX + 4 * 3, m3);
    }
    MNNConvDwF23SourceTransUnit(source + 4 * (su * 2 - pad), dest + 4 * 4 * su, eu - su);

    for (int x = eu; x < unit; ++x) {
        auto dstX = dest + 4 * 4 * x;
        auto sx   = x * 2 - (int)pad;
        auto ex   = sx + 4;

        auto clampSx = std::max(sx, 0);
        auto clampEx = std::min(ex, (int)iw);

        Vec4 v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int i = clampSx; i < clampEx; ++i) {
            v[i - sx] = Vec4::load(source + 4 * i);
        }
        auto m0 = v[0] - v[2];
        auto m1 = v[1] + v[2];
        auto m2 = v[2] - v[1];
        auto m3 = v[3] - v[1];

        Vec4::save(dstX + 4 * 0, m0);
        Vec4::save(dstX + 4 * 1, m1);
        Vec4::save(dstX + 4 * 2, m2);
        Vec4::save(dstX + 4 * 3, m3);
    }
}

#ifndef MNN_USE_NEON
void MNNConvDwF23MulTransUnit(float **cacheLine, const float *weigth, float *dest, size_t ow, const float* bias, const float* parameters) {
    int unit = ow / 2;
    auto w00 = Vec4::load(weigth + 0 * 16 + 4 * 0);
    auto w01 = Vec4::load(weigth + 0 * 16 + 4 * 1);
    auto w02 = Vec4::load(weigth + 0 * 16 + 4 * 2);
    auto w03 = Vec4::load(weigth + 0 * 16 + 4 * 3);
    auto w10 = Vec4::load(weigth + 1 * 16 + 4 * 0);
    auto w11 = Vec4::load(weigth + 1 * 16 + 4 * 1);
    auto w12 = Vec4::load(weigth + 1 * 16 + 4 * 2);
    auto w13 = Vec4::load(weigth + 1 * 16 + 4 * 3);
    auto w20 = Vec4::load(weigth + 2 * 16 + 4 * 0);
    auto w21 = Vec4::load(weigth + 2 * 16 + 4 * 1);
    auto w22 = Vec4::load(weigth + 2 * 16 + 4 * 2);
    auto w23 = Vec4::load(weigth + 2 * 16 + 4 * 3);
    auto biasF = Vec4::load(bias);
    auto minF = Vec4(parameters[2]);
    auto maxF = Vec4(parameters[3]);
    for (int x = 0; x < unit; ++x) {
        auto offset = 4 * 4 * x;
        int i = 0;
        Vec4 m0     = w00 * Vec4::load(cacheLine[0] + offset + 4 * 0);
        Vec4 m1     = w01 * Vec4::load(cacheLine[0] + offset + 4 * 1);
        Vec4 m2     = w02 * Vec4::load(cacheLine[0] + offset + 4 * 2);
        Vec4 m3     = w03 * Vec4::load(cacheLine[0] + offset + 4 * 3);

        m0 = m0 + w10 * Vec4::load(cacheLine[1] + offset + 4 * 0);
        m1 = m1 + w11 * Vec4::load(cacheLine[1] + offset + 4 * 1);
        m2 = m2 + w12 * Vec4::load(cacheLine[1] + offset + 4 * 2);
        m3 = m3 + w13 * Vec4::load(cacheLine[1] + offset + 4 * 3);

        m0 = m0 + w20 * Vec4::load(cacheLine[2] + offset + 4 * 0);
        m1 = m1 + w21 * Vec4::load(cacheLine[2] + offset + 4 * 1);
        m2 = m2 + w22 * Vec4::load(cacheLine[2] + offset + 4 * 2);
        m3 = m3 + w23 * Vec4::load(cacheLine[2] + offset + 4 * 3);

        auto o0 = m0 + m1 + m2 + biasF;
        auto o1 = m1 - m2 + m3 + biasF;
        o0 = Vec4::min(maxF, o0);
        o1 = Vec4::min(maxF, o1);
        o0 = Vec4::max(minF, o0);
        o1 = Vec4::max(minF, o1);
        Vec4::save(dest + 8 * x + 0 * 4, o0);
        Vec4::save(dest + 8 * x + 1 * 4, o1);
    }
    if (unit * 2 < ow) {
        auto offset = 4 * 4 * unit;
        Vec4 m0     = w00 * Vec4::load(cacheLine[0] + offset + 4 * 0);
        Vec4 m1     = w01 * Vec4::load(cacheLine[0] + offset + 4 * 1);
        Vec4 m2     = w02 * Vec4::load(cacheLine[0] + offset + 4 * 2);

        m0 = m0 + w10 * Vec4::load(cacheLine[1] + offset + 4 * 0);
        m1 = m1 + w11 * Vec4::load(cacheLine[1] + offset + 4 * 1);
        m2 = m2 + w12 * Vec4::load(cacheLine[1] + offset + 4 * 2);

        m0 = m0 + w20 * Vec4::load(cacheLine[2] + offset + 4 * 0);
        m1 = m1 + w21 * Vec4::load(cacheLine[2] + offset + 4 * 1);
        m2 = m2 + w22 * Vec4::load(cacheLine[2] + offset + 4 * 2);
        auto o0 = m0 + m1 + m2 + biasF;
        o0 = Vec4::min(maxF, o0);
        o0 = Vec4::max(minF, o0);
        Vec4::save(dest + 8 * unit + 0 * 4, o0);
    }
}
void MNNConvDwF23SourceTransUnit(const float *source, float *dest, size_t unit) {
    if (unit <= 0) {
        return;
    }
    Vec4 v0 = Vec4::load(source + 4 * 0);
    Vec4 v1 = Vec4::load(source + 4 * 1);
    Vec4 v2;
    Vec4 v3;
    source += 8;

    for (int x = 0; x < unit; ++x) {
        v2 = Vec4::load(source + 0 * 4);
        v3 = Vec4::load(source + 1 * 4);
        auto m0 = v0 - v2;
        auto m1 = v1 + v2;
        auto m2 = v2 - v1;
        auto m3 = v3 - v1;

        Vec4::save(dest + 4 * 0, m0);
        Vec4::save(dest + 4 * 1, m1);
        Vec4::save(dest + 4 * 2, m2);
        Vec4::save(dest + 4 * 3, m3);

        source += 8;
        dest += 16;

        v0 = v2;
        v1 = v3;
    }
}
#endif

static void _MNNAdjustOptimalSparseKernel(int& sparseBlockOC, MNN::CoreFunctions::MNNPackedSparseMatMul& packedSparseMatMul) {
    if(sparseBlockOC == 4) {
        packedSparseMatMul = MNNPackedSparseMatMulEpx4;
        return;
    } else if(sparseBlockOC % 4 == 0) {
        sparseBlockOC = 4;
        packedSparseMatMul = MNNPackedSparseMatMulEpx4;
        // MNN_PRINT("common downgrade sparse to:%d\n",sparseBlockOC);
        return;
    } else {
        sparseBlockOC = 1;
        packedSparseMatMul = MNNPackedSparseMatMulEpx1;
        return;
    }
}

namespace MNN {

static CoreFunctions* gCoreFunction = nullptr;

void MNNCoreFunctionInit() {
    gCoreFunction = new CoreFunctions;
    // MatMul
    gCoreFunction->MNNGetMatMulPackMode = MNNGetMatMulPackMode;
    gCoreFunction->MNNPackC4ForMatMul_A = MNNPackC4ForMatMul_A;
    gCoreFunction->MNNPackForMatMul_B = MNNPackForMatMul_B;
    gCoreFunction->MNNPackedMatMul = MNNPackedMatMul;
    gCoreFunction->MNNPackedMatMulRemain = MNNPackedMatMulRemain;
    gCoreFunction->MNNCountMaxMinValue = MNNCountMaxMinValue;
    gCoreFunction->MNNGetSparseMatMulPackMode = MNNGetSparseMatMulPackMode;
    gCoreFunction->MNNAdjustOptimalSparseKernel = _MNNAdjustOptimalSparseKernel;

    gCoreFunction->MNNComputeMatMulForE_1 = MNNComputeMatMulForE_1;
    gCoreFunction->MNNComputeMatMulForH_1 = MNNComputeMatMulForH_1;

    // Lowp
    gCoreFunction->MNNFp32ToLowp = nullptr;
    gCoreFunction->MNNLowpToFp32 = nullptr;
    gCoreFunction->bytes = 4;// sizeof(float)

    // Packed Function
    gCoreFunction->pack = 4;
    // FIXME: MNNPackTranspose and MNNUnpackTranspose is reverted
    gCoreFunction->MNNPackCUnit = MNNPackC4;
    gCoreFunction->MNNUnpackCUnit = MNNUnpackC4;
    gCoreFunction->MNNUnpackCUnitTranspose = MNNPackTranspose;
    gCoreFunction->MNNPackCUnitTranspose = MNNUnpackTranspose;
    gCoreFunction->MNNPackCUnitInt8 = decltype(gCoreFunction->MNNPackCUnitInt8)(MNNPackC4Uint8);
    gCoreFunction->MNNUnpackCUnitInt8 = decltype(gCoreFunction->MNNUnpackCUnitInt8)(MNNUnpackC4Uint8);
    gCoreFunction->MNNPackCUnitTransposeInt8 = decltype(gCoreFunction->MNNPackCUnitTransposeInt8)(MNNUnpackTransposeUint8);
    gCoreFunction->MNNUnpackCUnitTransposeInt8 = decltype(gCoreFunction->MNNUnpackCUnitTransposeInt8)(MNNPackTransposeUint8);
    gCoreFunction->MNNPackCUnitInt16 = MNNPackC4Int16;
    gCoreFunction->MNNUnpackCUnitInt16 = MNNUnpackC4Int16;
    gCoreFunction->MNNPackCUnitTransposeInt16 = MNNUnpackTransposeInt16;
    gCoreFunction->MNNUnpackCUnitTransposeInt16 = MNNPackTransposeInt16;

    gCoreFunction->MNNAxByClampBroadcastUnit = MNNAxByClampBroadcastUnit;
    gCoreFunction->MNNConvRunForLineDepthwise = MNNConvRunForLineDepthwise;
    gCoreFunction->MNNConvRunForUnitDepthWise = MNNConvRunForUnitDepthWise;
    gCoreFunction->MNNSourceTransformCommonF23 = MNNSourceTransformCommonF23;
    gCoreFunction->MNNConvDwF23MulTransUnit = MNNConvDwF23MulTransUnit;
    gCoreFunction->MNNMultiAndDestTransformCommon23 = MNNMultiAndDestTransformCommon23;
    gCoreFunction->MNNMatrixAdd = MNNMatrixAdd;
    gCoreFunction->MNNMatrixSub = MNNMatrixSub;
    gCoreFunction->MNNStrassenMergeCFunction = MNNStrassenMergeCFunction;
    gCoreFunction->penalty = 1.5f;
    gCoreFunction->MNNScaleAndAddBias = MNNScaleAndAddBias;
    gCoreFunction->MNNGridSampleComputeCord = MNNGridSampleComputeCord;
    gCoreFunction->MNNGridSampleInterp = MNNGridSampleInterp;
    gCoreFunction->MNNGridSampleInterpGrad = MNNGridSampleInterpGrad;
    gCoreFunction->MNNGridSampleComputeCord3D = MNNGridSampleComputeCord3D;
    gCoreFunction->MNNGridSampleInterp3D = MNNGridSampleInterp3D;
    gCoreFunction->MNNRoiPoolingMax = MNNRoiPoolingMax;
    gCoreFunction->MNNRoiAlignMax = MNNRoiAlignMax;
    gCoreFunction->MNNRoiAlignAvg = MNNRoiAlignAvg;
    gCoreFunction->MNNAddC4WithStride = MNNAddC4WithStride;
    gCoreFunction->MNNCopyC4WithStride = MNNCopyC4WithStride;

    gCoreFunction->chooseWinoSourceTransformPack = WinogradFunction::chooseWinoSourceTransformPack;
    gCoreFunction->chooseWinoSourceUnrollTransform = WinogradFunction::chooseSourceUnrollTransform;
    gCoreFunction->chooseWinoDestUnrollTransform = WinogradFunction::chooseWinoDestUnrollTransform;
    gCoreFunction->MNNDeconvRunForLineDepthwise = MNNDeconvRunForLineDepthwise;
    gCoreFunction->MNNDeconvRunForUnitDepthWise = MNNDeconvRunForUnitDepthWise;
    gCoreFunction->MNNSelectBinaryFunctionForFloat = CPUBinary::selectForFloat;
    gCoreFunction->MNNSelectUnaryFunctionForFloat = CPUUnary::selectForFloat;
    gCoreFunction->MNNSelectUnaryFunctionForInt8 = CPUUnary::selectForInt8;
    gCoreFunction->MNNReluWithSlopeChannel = MNNReluWithSlopeChannel;
    gCoreFunction->MNNPoolingAvg = (decltype(gCoreFunction->MNNPoolingAvg))(poolingAvg<float, Vec4, 4>);
    // Set min value as 1 << 24
    gCoreFunction->MNNPoolingMax = (decltype(gCoreFunction->MNNPoolingMax))(poolingMax<float, Vec4, 4, -16777216>);

    gCoreFunction->MNNPoolingMaxWithRedice = (decltype(gCoreFunction->MNNPoolingMaxWithRedice))(poolingMaxWithRedice<float, -16777216>);
    // ImageProcess Functions
    gCoreFunction->MNNRGBAToBGRA = MNNRGBAToBGRA;
    gCoreFunction->MNNNV21ToRGBA = MNNNV21ToRGBA;
    gCoreFunction->MNNNV21ToRGB = MNNNV21ToRGB;
    gCoreFunction->MNNNV21ToBGRA = MNNNV21ToBGRA;
    gCoreFunction->MNNNV21ToBGR = MNNNV21ToBGR;
    gCoreFunction->MNNC1ToFloatC1 = MNNC1ToFloatC1;
    gCoreFunction->MNNC3ToFloatC3 = MNNC3ToFloatC3;
    gCoreFunction->MNNC3ToFloatRGBA = MNNC3ToFloatRGBA;
    gCoreFunction->MNNSamplerC4Nearest = MNNSamplerC4Nearest;
    gCoreFunction->MNNSamplerC4Bilinear = MNNSamplerC4Bilinear;

    gCoreFunction->MNN4BitcopyWithStride = MNN4BitcopyWithStride;
    gCoreFunction->MNN1BitcopyWithStride = MNN1BitcopyWithStride;
    gCoreFunction->MNN2BitcopyWithStride = MNN2BitcopyWithStride;
    gCoreFunction->MNN4BitcopyFast = MNN4BitcopyFast;
    gCoreFunction->MNN2BitcopyFast = MNN2BitcopyFast;
    gCoreFunction->MNN1BitcopyFast = MNN1BitCopyFast;

    gCoreFunction->MNNAccumulateSequenceNumber = MNNAccumulateSequenceNumber;

    const MNNCPUInfo& gCPUInfo = *MNNGetCPUInfo();
    gCoreFunction->supportFp16arith = gCPUInfo.fp16arith;
    gCoreFunction->supportSDot = gCPUInfo.dot;
    gCoreFunction->supportI8mm = gCPUInfo.i8mm;
    gCoreFunction->MNNSumByAxisLForMatmul_A = MNNSumByAxisLForMatmul_A;
#ifdef MNN_LOW_MEMORY
    // Weight Dequant Gemm Kernels
    gCoreFunction->MNNPackedMatMul_int4 = MNNPackedMatMul_int4;
    gCoreFunction->MNNPackedMatMulRemain_int4 = MNNPackedMatMulRemain_int4;
    gCoreFunction->MNNPackedMatMul_int8 = MNNPackedMatMul_int8;
    gCoreFunction->MNNPackedMatMulRemain_int8 = MNNPackedMatMulRemain_int8;
    // Dynamic Quant Helper Functions
    gCoreFunction->MNNAbsMax = MNNAbsMaxFP32;
    gCoreFunction->MNNDynamicQuant = MNNDynamicQuantFP32;
    gCoreFunction->MNNQuantScale = MNNQuantScaleFP32;
    gCoreFunction->MNNQuantSum = MNNQuantSumFP32;
    // Dynamic Quan Bias
    gCoreFunction->MNNDynamicUpdateConvBiasScale = MNNDynamicUpdateConvBiasScale;
#endif
    MNNCoreInt8FunctionInit();
    MNNFunctionInit();
}
CoreFunctions* MNNGetCoreFunctions() {
    return gCoreFunction;
}
};

void MNNUnpackC4Origin(float* dst, const float* src, size_t area, size_t depth, int areaOffset) {
    int offset[] = {
        areaOffset,
        areaOffset,
    };
    MNNUnpackC4(dst, src, area, depth, offset);
}
void MNNPackC4Origin(float* dst, const float* src, size_t area, size_t depth, int areaOffset) {
    int offset[] = {
        areaOffset,
        areaOffset,
    };
    MNNPackC4(dst, src, area, depth, offset);
}

void MNNPackC2(double* dst, const double* src, size_t area, size_t depth, int* areaOffset) {
    MNNPackC2Common<double>(dst, src, area, depth, areaOffset);
}

void MNNUnpackC2(double* dst, const double* src, size_t area, size_t depth, int* areaOffset) {
    MNNUnpackC2Common<double>(dst, src, area, depth, areaOffset);
}

void MNNUnpackC2Float(float* dst, const float* src, size_t area, size_t depth, int* areaOffset, int pack) {
    MNNUnpackC2Common<float>(dst, src, area, depth, areaOffset, pack);
}

void MNNPackInt8C2(float* dst, const float* src, size_t area, size_t depth, int* areaOffset) {
    MNNPackC2Common<float>(dst, src, area, depth, areaOffset);
}

void MNNUnpackInt8C2(float* dst, const float* src, size_t area, size_t depth, int* areaOffset) {
    MNNUnpackC2Common<float>(dst, src, area, depth, areaOffset);
}


void MNNUnpackC2Origin(double* dst, const double* src, size_t area, size_t depth, int areaOffset) {
    int offset[] = {
        areaOffset,
        areaOffset,
    };
    MNNUnpackC2(dst, src, area, depth, offset);
}
void MNNPackC2Origin(double* dst, const double* src, size_t area, size_t depth, int areaOffset) {
    int offset[] = {
        areaOffset,
        areaOffset,
    };
    MNNPackC2(dst, src, area, depth, offset);
}

void MNNUnpackInt8C2Origin(float* dst, const float* src, size_t area, size_t depth, int areaOffset) {
    int offset[] = {
        areaOffset,
        areaOffset,
    };
    MNNUnpackInt8C2(dst, src, area, depth, offset);
}
void MNNPackInt8C2Origin(float* dst, const float* src, size_t area, size_t depth, int areaOffset) {
    int offset[] = {
        areaOffset,
        areaOffset,
    };
    MNNPackInt8C2(dst, src, area, depth, offset);
}
