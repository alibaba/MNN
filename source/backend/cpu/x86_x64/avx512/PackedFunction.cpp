//
//  PackedFunction.cpp
//  MNN
//
//  Created by MNN on 2021/07/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <float.h>
#include <string.h>
#include <algorithm>
#include <limits>
#include <vector>
#include "FunctionSummary.hpp"
#include "core/Macro.h"
#include "backend/cpu/CPUPool.hpp"
#include "backend/cpu/BinaryUtils.hpp"
#include "Vec16.hpp"
#define PACK_UNIT 16
#define PACK PACK_UNIT
#define FLOAT float
using Vec = Vec16;
#include "backend/cpu/GridSampler.hpp"

void _AVX512_MNNCopyC4WithStride(const float* source, float* dest, size_t srcStride, size_t dstStride, size_t count) {
    for (int i = 0; i < count; ++i) {
        auto s = source + i * srcStride;
        auto d = dest + i * dstStride;
        _mm512_storeu_ps(d, _mm512_loadu_ps(s));
    }
}
void _AVX512_MNNAddC4WithStride(const float* source, float* dest, size_t srcStride, size_t dstStride, size_t count) {
    for (int i = 0; i < count; ++i) {
        auto s = source + i * srcStride;
        auto d = dest + i * dstStride;
        _mm512_storeu_ps(d, _mm512_add_ps(_mm512_loadu_ps(s), _mm512_loadu_ps(d)));
    }
}

void _AVX512_MNNCountMinMaxValue(const float* source, float* min, float* max, size_t size) {
    int pack = 16;
    int sizeDiv16 = size / pack;
    __m512 minVal = _mm512_set1_ps(source[0]);
    __m512 maxVal = minVal;
    float maxArr[16], minArr[16];
    for (int i = 0; i < sizeDiv16; ++i) {
        auto src0 = source + pack * i;
        __m512 vecA = _mm512_loadu_ps(src0);
        auto maskMax = _mm512_cmp_ps_mask(vecA, maxVal, 14);
        auto maskMin = _mm512_cmp_ps_mask(vecA, minVal, 1);
        maxVal = _mm512_mask_blend_ps(maskMax, maxVal, vecA);
        minVal = _mm512_mask_blend_ps(maskMin, minVal, vecA);
    }
    _mm512_storeu_ps(maxArr, maxVal);
    _mm512_storeu_ps(minArr, minVal);
    float max_ = maxArr[0], min_ = minArr[0];
    for (int k = 1; k < pack; ++k) {
        if (max_ < maxArr[k]) {
            max_ = maxArr[k];
        }
        if (min_ > minArr[k]) {
            min_ = minArr[k];
        }
    }
    for (int i = pack * sizeDiv16; i < size; ++i) {
        min_ = ALIMIN(min_, source[i]);
        max_ = ALIMAX(max_, source[i]);
    }
    min[0] = min_;
    max[0] = max_;
}

static void _AVX512_BatchMinMax(float* dstMin, float* dstMax, const float* source, size_t src_depth_quad, size_t realSize, int innerSide, size_t loadDstBuffer) {
    // input: [src_depth_quad, realSize, pack]
    // max,min shape: [realSize]
    // avx512: core->pack = 16, SRC_UNIT=4
    auto srcStep = realSize * innerSide;
    if (innerSide == 4) {
        float tempMax[4];
        float tempMin[4];
        for (int i = 0; i < realSize; ++i) {
            auto min_ = _mm_loadu_ps(source + i * innerSide);
            auto max_ = min_;
            for (int c = 1; c < src_depth_quad; ++c) {
                auto src0 = source + c * srcStep + i * innerSide;
                auto vecA = _mm_loadu_ps(src0);
                max_ = _mm_max_ps(max_, vecA);
                min_ = _mm_min_ps(min_, vecA);
            }
            _mm_storeu_ps(tempMax, max_);
            _mm_storeu_ps(tempMin, min_);
            float max0 = tempMax[0];
            float min0 = tempMin[0];
            for (int k = 1; k < innerSide; ++k) {
                if (max0 < tempMax[k]) {
                    max0 = tempMax[k];
                }
                if (min0 > tempMin[k]) {
                    min0 = tempMin[k];
                }
            }
            if (loadDstBuffer) {
                dstMax[i] = ALIMAX(max0, dstMax[i]);
                dstMin[i] = ALIMIN(min0, dstMin[i]);
            } else {
                dstMax[i] = max0;
                dstMin[i] = min0;
            }
        }
        return;
    }
    if (innerSide == 16) {
        float tmp[16];
        for (int i = 0; i < realSize; ++i) {
            auto min_ = _mm512_loadu_ps(source + i * innerSide);
            auto max_ = min_;
            auto src0 = source + i * innerSide;
            for (int j = 1; j < src_depth_quad; ++j) {
                auto vec = _mm512_loadu_ps(src0 + j * srcStep);
                max_ = _mm512_max_ps(max_, vec);
                min_ = _mm512_min_ps(min_, vec);
            }
            auto maxval = _mm512_reduce_max_ps(max_);
            auto minval = _mm512_reduce_min_ps(min_);
            dstMax[i] = maxval;
            dstMin[i] = minval;
        }
        return;
    }
    MNN_ERROR("batch max/min error: x86_x64 avx512 don't suppport innerSide=%d yet\n", innerSide);
}

static void _AVX512_MNNAsyQuantInfo(float* scale, float* bias, float* qscale, float* qbias, float* dstMin, float* dstMax, const float* src, const size_t* info) {
    auto blockNum = info[0];
    auto plane = info[1];        // real area for data
    auto innerSide = info[2];    // Innermost data layout, may come from backend's pack or gemmint8 units' SRC_UNIT
    auto DST_XUNIT = info[3];    // AVX512: DST_XUNIT=4
    auto kernelsize = info[5];
    auto blockLU = info[6];
    auto stride0 = blockNum * blockLU * plane * innerSide;
    auto stride1 = blockLU * plane * innerSide;

    if (info[7] == 1) { // scale&bias:[1]
        float maxval, minval;
        _AVX512_MNNCountMinMaxValue(src, &minval, &maxval, kernelsize * stride0);
        if (info[8] == 1 && (maxval -minval) > 1e-7) {
            if (minval > 0.f) {
                minval = 0;
            } else if (maxval < 0.f){
                maxval = 0;
            }
        }
        auto range = maxval - minval;
        if (range <= 1e-7) {
            scale[0] = 0.f;
            qscale[0] = 0.f;
            qbias[0] = 0.f;
            bias[0] = maxval;
        } else {
            qscale[0] = 255.f / range;
            scale[0] = range / 255.f;
            qbias[0] = roundf(-minval * 255.f / range)- 128.f;
            bias[0] = minval;
        }
        return;
    }

    // input              : [kernelsize, blockNum, blockLU, plane, pack]
    // dequant scale/bias : [EU, blockNum, step], step=ALIMIN(step, EP), EU=UP_DIV(plane, EP)
    // quant scale/bias   : [blockNum, plane]
    // max,min            : [blockNum, plane]

    for (int i = 0; i < kernelsize; ++i) {
        for (int j = 0; j < blockNum; ++j) {
            _AVX512_BatchMinMax(dstMin + j * plane, dstMax + j * plane, src + i * stride0 + j * stride1, blockLU, plane, innerSide, i);
        }
    }
    // scale,bias
    auto realDstCount = plane;
    auto thredshold4 = _mm_set1_ps(1e-6);
    auto _255f = _mm_set1_ps(255.f);
    auto _128f = _mm_set1_ps(128.f);
    auto _0f = _mm_set1_ps(0.f);
    for (int k = 0; k < blockNum; ++k) {
        auto qind = k * plane;
        auto realDstCount = plane;
        auto scalePtr = scale + k * ALIMIN(plane, DST_XUNIT);
        auto biasPtr = bias + k * ALIMIN(plane, DST_XUNIT);
        while (realDstCount >= DST_XUNIT) {
            auto step = DST_XUNIT;           // ALIMIN(realDstCount, DST_XUNIT);
            auto max4 = _mm_loadu_ps(dstMax + qind);
            auto min4 = _mm_loadu_ps(dstMin + qind);
            auto diff4 = _mm_sub_ps(max4, min4);
            auto mask = _mm_cmplt_ps(diff4, thredshold4);

            // scale,bias
            auto quantScale4 = _mm_div_ps(_255f, diff4);
            auto dequantScale4 = _mm_div_ps(diff4, _255f);
            auto quantBias4 = _mm_sub_ps(_mm_div_ps(_mm_mul_ps(_mm_sub_ps(_0f, min4), _255f), diff4), _128f);
            auto dequantBias4 = min4;

            quantScale4 = _mm_blendv_ps(quantScale4, _0f, mask);
            dequantScale4 = _mm_blendv_ps(dequantScale4, _0f, mask);
            quantBias4 = _mm_round_ps(_mm_blendv_ps(quantBias4, _0f, mask), 0);
            dequantBias4 = _mm_blendv_ps(dequantBias4, max4, mask);

            _mm_storeu_ps(scalePtr, dequantScale4);
            _mm_storeu_ps(biasPtr, dequantBias4);
            _mm_storeu_ps(qscale + qind, quantScale4);
            _mm_storeu_ps(qbias + qind, quantBias4);

            realDstCount -= DST_XUNIT;
            qind += DST_XUNIT;
            scalePtr += (blockNum * DST_XUNIT);
            biasPtr += (blockNum * DST_XUNIT);
        }
        if (realDstCount == 0) {
            continue;
        }
        auto remainE = realDstCount;
        auto stride0 = remainE * blockNum;
        scalePtr = scale + (plane / DST_XUNIT) * blockNum * DST_XUNIT + k * remainE;
        biasPtr = bias + (plane / DST_XUNIT) * blockNum * DST_XUNIT + k * remainE;
        while (realDstCount) {
            auto max_ = dstMax[qind];
            auto min_ = dstMin[qind];
            if (fabs(max_ - min_) < 1e-7) {
                qscale[qind] = 0.f;
                qbias[qind] = 0.f;
                scalePtr[0] = 0.f;
                biasPtr[0] = max_;
            } else {
                qscale[qind] = 255.f / (max_ - min_);
                qbias[qind] = roundf(-min_ * 255.f / (max_ - min_)) - 128.0f;
                scalePtr[0] = (max_ - min_) / 255.f;
                biasPtr[0] = min_;
            }
            realDstCount -= 1;
            qind += 1;
            scalePtr += 1;
            biasPtr += 1;
        }
    }
}

static void _AVX512_MNNAbsMaxFP32(const float* source, float* absmax, size_t src_depth_quad, size_t realSize, int pack) {
    auto srcStep = realSize * pack;
    if (pack == 4) {
        __m128 mask = _mm_set1_ps(-0.0f);
        float tmp[4];
        for (int i = 0; i < realSize; ++i) {
            __m128 absmax_ = _mm_loadu_ps(source + i * pack);
            absmax_ = _mm_andnot_ps(mask, absmax_);
            auto src0 = source + i * pack;
            for (int j = 1; j < src_depth_quad; ++j) {
                __m128 vec = _mm_loadu_ps(src0 + j * srcStep);
                vec = _mm_andnot_ps(mask, vec);
                absmax_ = _mm_max_ps(absmax_, vec);
            }
            _mm_storeu_ps(tmp, absmax_);
            float res = tmp[0];
            for (int j = 1; j < pack; ++j) {
                res = ALIMAX(res, tmp[j]);
            }
            absmax[i] = res;
        }
        return;
    }
    if (pack == 16) {
        float tmp[16];
        for (int i = 0; i < realSize; ++i) {
            auto absmax_ = _mm512_loadu_ps(source + i * pack);
            absmax_ = _mm512_abs_ps(absmax_);
            auto src0 = source + i * pack;
            for (int j = 1; j < src_depth_quad; ++j) {
                auto vec = _mm512_loadu_ps(src0 + j * srcStep);
                vec = _mm512_abs_ps(vec);
                absmax_ = _mm512_max_ps(absmax_, vec);
            }
            auto maxval = _mm512_reduce_max_ps(absmax_);
            absmax[i] = maxval;
        }
        return;
    }
    MNN_ERROR("absMax error: x86_x64 avx512 don't suppport pack=%d yet\n", pack);
}

static void _AVX512_DynamicQuant(const float* src, int8_t* dst, const float* scale, size_t src_depth_quad, size_t realSize, int pack, const float* bias) {
    auto srcStep = realSize * pack;
    if (pack == 16) { // core->pack=16
        auto offset = _mm512_set1_epi32(128);
        int32_t tmp[16];
        int32_t* dstPtr = reinterpret_cast<int32_t*>(dst);
        for (int i = 0; i < src_depth_quad; ++i) {
            int xcount = realSize;
            auto srcPtr = src + i * srcStep;
            auto scalePtr = scale;
            auto biasPtr = bias;
            while (xcount > 3) {
                auto scale0 = _mm512_set1_ps(scalePtr[0]);
                auto scale1 = _mm512_set1_ps(scalePtr[1]);
                auto scale2 = _mm512_set1_ps(scalePtr[2]);
                auto scale3 = _mm512_set1_ps(scalePtr[3]);
                auto data0 = _mm512_loadu_ps(srcPtr);
                auto data1 = _mm512_loadu_ps(srcPtr + pack);
                auto data2 = _mm512_loadu_ps(srcPtr + 2 * pack);
                auto data3 = _mm512_loadu_ps(srcPtr + 3 * pack);
                data0 = _mm512_mul_ps(data0, scale0);
                data1 = _mm512_mul_ps(data1, scale1);
                data2 = _mm512_mul_ps(data2, scale2);
                data3 = _mm512_mul_ps(data3, scale3);
                if (bias) {
                    auto bias0 = _mm512_set1_ps(biasPtr[0]);
                    auto bias1 = _mm512_set1_ps(biasPtr[1]);
                    auto bias2 = _mm512_set1_ps(biasPtr[2]);
                    auto bias3 = _mm512_set1_ps(biasPtr[3]);
                    data0 = _mm512_add_ps(data0, bias0);
                    data1 = _mm512_add_ps(data1, bias1);
                    data2 = _mm512_add_ps(data2, bias2);
                    data3 = _mm512_add_ps(data3, bias3);
                }
                auto r0 = _mm512_cvt_roundps_epi32(data0, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                auto r1 = _mm512_cvt_roundps_epi32(data1, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                auto r2 = _mm512_cvt_roundps_epi32(data2, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                auto r3 = _mm512_cvt_roundps_epi32(data3, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                r0 = _mm512_add_epi32(r0, offset); // int32x16
                r1 = _mm512_add_epi32(r1, offset); // int32x16
                r2 = _mm512_add_epi32(r2, offset);
                r3 = _mm512_add_epi32(r3, offset);
                auto r0_16 = _mm512_packs_epi32(r0, r1); // 00001111 00001111 00001111 00001111
                auto r1_16 = _mm512_packs_epi32(r2, r3); // 22223333 22223333 22223333 22223333
                auto r0_8  = _mm512_packus_epi16(r0_16, r1_16); // 0000111122223333 0000111122223333 0000111122223333 0000111122223333
                _mm512_storeu_si512(tmp, r0_8);
                for (int k = 0; k < 4; ++k) {
                    dstPtr[k * 4 + 0] = tmp[k + 4 * 0];
                    dstPtr[k * 4 + 1] = tmp[k + 4 * 1];
                    dstPtr[k * 4 + 2] = tmp[k + 4 * 2];
                    dstPtr[k * 4 + 3] = tmp[k + 4 * 3];
                }
                // next round
                xcount -= 4;
                scalePtr += 4;
                if (bias) {
                    biasPtr += 4;
                }
                srcPtr += (4 * pack);
                dstPtr += 16;
            }
            while (xcount) {
                auto scale0 = _mm512_set1_ps(scalePtr[0]);
                auto data0 = _mm512_loadu_ps(srcPtr);
                data0 = _mm512_mul_ps(data0, scale0);
                if (bias) {
                    auto bias0 = _mm512_set1_ps(biasPtr[0]);
                    data0 = _mm512_add_ps(data0, bias0);
                }
                auto r0 = _mm512_cvt_roundps_epi32(data0, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                r0 = _mm512_add_epi32(r0, offset); // int32x16
                auto r0_16 = _mm512_packs_epi32(r0, r0); // 00001111 00001111 00001111 00001111
                auto r0_8  = _mm512_packus_epi16(r0_16, r0_16); // 0000111122223333 0000111122223333 0000111122223333 0000111122223333
                _mm512_storeu_si512(tmp, r0_8);
                dstPtr[0] = tmp[4 * 0];
                dstPtr[1] = tmp[4 * 1];
                dstPtr[2] = tmp[4 * 2];
                dstPtr[3] = tmp[4 * 3];

                // next round
                xcount--;
                scalePtr += 1;
                if (bias) {
                    biasPtr += 1;
                }
                srcPtr += pack;
                dstPtr += 4;
            }
        }
        return;
    }
    if (pack == 4) { // LP=4;
        auto offset = _mm_set1_epi32(128);
        int32_t tmp[4];
        int32_t* dstPtr = reinterpret_cast<int32_t*>(dst);
        for (int i = 0; i < src_depth_quad; ++i) {
            int xcount = realSize;
            auto srcPtr = src + i * srcStep;
            auto scalePtr = scale;
            auto biasPtr = bias;
            while (xcount > 3) {
                auto scale0 = _mm_set1_ps(scalePtr[0]);
                auto scale1 = _mm_set1_ps(scalePtr[1]);
                auto scale2 = _mm_set1_ps(scalePtr[2]);
                auto scale3 = _mm_set1_ps(scalePtr[3]);
                auto data0 = _mm_loadu_ps(srcPtr);
                auto data1 = _mm_loadu_ps(srcPtr + pack);
                auto data2 = _mm_loadu_ps(srcPtr + 2 * pack);
                auto data3 = _mm_loadu_ps(srcPtr + 3 * pack);
                data0 = _mm_mul_ps(data0, scale0);
                data1 = _mm_mul_ps(data1, scale1);
                data2 = _mm_mul_ps(data2, scale2);
                data3 = _mm_mul_ps(data3, scale3);
                if (bias) {
                    auto bias0 = _mm_set1_ps(biasPtr[0]);
                    auto bias1 = _mm_set1_ps(biasPtr[1]);
                    auto bias2 = _mm_set1_ps(biasPtr[2]);
                    auto bias3 = _mm_set1_ps(biasPtr[3]);
                    data0 = _mm_add_ps(data0, bias0);
                    data1 = _mm_add_ps(data1, bias1);
                    data2 = _mm_add_ps(data2, bias2);
                    data3 = _mm_add_ps(data3, bias3);
                }
                data0 = _mm_round_ps(data0, 0);
                data1 = _mm_round_ps(data1, 0);
                data2 = _mm_round_ps(data2, 0);
                data3 = _mm_round_ps(data3, 0);
                auto r0 = _mm_cvtps_epi32(data0);
                auto r1 = _mm_cvtps_epi32(data1);
                auto r2 = _mm_cvtps_epi32(data2);
                auto r3 = _mm_cvtps_epi32(data3);
                r0 = _mm_add_epi32(r0, offset);
                r1 = _mm_add_epi32(r1, offset);
                r2 = _mm_add_epi32(r2, offset);
                r3 = _mm_add_epi32(r3, offset);
                auto r0_16 = _mm_packs_epi32(r0, r1); // 00001111
                auto r1_16 = _mm_packs_epi32(r2, r3); // 22223333
                auto r0_8  = _mm_packus_epi16(r0_16, r1_16); // 0000111122223333
                _mm_storeu_si128((__m128i *)dstPtr, r0_8);
                // next round
                xcount -= 4;
                scalePtr += 4;
                if (bias) {
                    biasPtr += 4;
                }
                srcPtr += (4 * pack);
                dstPtr += 4;
            }
            while (xcount) {
                auto scale0 = _mm_set1_ps(scalePtr[0]);
                auto data0 = _mm_loadu_ps(srcPtr);
                data0 = _mm_mul_ps(data0, scale0);
                if (bias) {
                    auto bias0 = _mm_set1_ps(biasPtr[0]);
                    data0 = _mm_add_ps(data0, bias0);
                }
                auto r0 = _mm_cvtps_epi32(_mm_round_ps(data0, 0));
                r0 = _mm_add_epi32(r0, offset);
                auto r0_16 = _mm_packs_epi32(r0, r0); // 00001111
                auto r0_8  = _mm_packus_epi16(r0_16, r0_16); // 0000111122223333
                _mm_storeu_si128((__m128i *)tmp, r0_8);
                dstPtr[0] = tmp[0];
                // next round
                xcount--;
                scalePtr += 1;
                if (bias) {
                    biasPtr += 1;
                }
                srcPtr += pack;
                dstPtr += 1;
            }
        }
        return;
    }
    MNN_ERROR("dynamic quant error: x86_x64 avx512 don't suppport pack=%d yet\n", pack);
    return;
}

static void _AVX512_MNNAsyQuantFunc(int8_t* dst, const float* src, float* qscale, float* qbias, const size_t* info) {
    // input shape: [kernelsize, blockNum, blockLU, EP, LP]
    auto blockNum = info[0];
    auto EP = info[1];        // real area for data
    auto LP = info[2];        // Innermost data layout, may come from backend's pack or gemmint8 units' SRC_UNIT
    auto DST_XUNIT = info[3]; // backend gemmint8 units
    auto SRC_UNIT = info[4];
    auto kernelsize = info[5];
    auto blockLU = info[6];
    auto stride0 = blockNum * blockLU * EP * LP;
    auto stride1 = blockLU * EP * LP;
    for (int k = 0; k < kernelsize; ++k) {
        for (int i = 0; i < blockNum; ++i) {
            _AVX512_DynamicQuant(src + k * stride0 + i * stride1, dst + k * stride0 + i * stride1, qscale + i * EP, blockLU, EP, LP, qbias + i * EP);
        }
    }
}

void _AVX512_MNNReluWithSlopeChannel(float* dst, const float* src, const float* slope, size_t sizeQuad, size_t depthQuad) {
    auto zero = _mm_set1_ps(0.0f);
    auto zero2 = _mm512_set1_ps(0.0f);
    int sizeC8 = sizeQuad;
    for (int j = 0; j < depthQuad; j++) {
        auto slopeZ       = _mm512_loadu_ps(slope + PACK_UNIT * j);
        const float* srcZ = src + PACK_UNIT * j * sizeQuad;
        float* dstZ       = dst + PACK_UNIT * j * sizeQuad;
        for (int i = 0; i < sizeC8; i++) {
            auto src   = _mm512_loadu_ps(srcZ);
            auto mask0 = _mm512_cmp_ps_mask(src, zero2, 0x01);
            auto other = _mm512_mul_ps(src, slopeZ);
            _mm512_storeu_ps(dstZ, _mm512_mask_blend_ps(mask0, src, other));
            srcZ += PACK_UNIT;
            dstZ += PACK_UNIT;
        }
    }
}

void _AVX512_MNNAxByClampBroadcastUnit(float* C, const float* A, const float* B, size_t width, size_t cStride, size_t aStride, size_t height, const float* parameters) {
    auto minF = _mm512_broadcastss_ps(_mm_load_ss(parameters + 2));
    auto maxF = _mm512_broadcastss_ps(_mm_load_ss(parameters + 3));
    for (int y = 0; y < height; ++y) {
        auto a = A + aStride * y;
        auto b = B + PACK_UNIT * y;
        auto bv = _mm512_loadu_ps(b);
        auto c = C + cStride * y;
        for (int x = 0; x < width; ++x) {
            auto av = _mm512_loadu_ps(a);
            auto cv = _mm512_add_ps(av, bv);
            cv = _mm512_min_ps(cv, maxF);
            cv = _mm512_max_ps(cv, minF);
            _mm512_storeu_ps(c, cv);
            a += PACK_UNIT;
            c += PACK_UNIT;
        }
    }
}

void _AVX512_MNNConvRunForLineDepthwise(float* dst, const float* src, const float* weight, size_t width, size_t src_w_setup,
                                size_t fw, size_t fh, size_t dilateX_step, size_t dilateY_step, size_t height,
                                     size_t srcHStep, size_t dstHStep, const float* bias, const float* parameters) {
    int dx, fx, fy;
    const int unit = 4;
    int widthUnit = width / unit;
    int widthRemain = width - widthUnit * unit;
    const float* weight_z = weight;
    auto minF = _mm512_broadcastss_ps(_mm_load_ss(parameters + 0));
    auto maxF = _mm512_broadcastss_ps(_mm_load_ss(parameters + 1));
    auto bv = _mm512_loadu_ps(bias);
    for (int y = 0; y < height; ++y) {
        auto srcY = src + y * srcHStep;
        auto dstY = dst + y * dstHStep;
        for (dx = 0; dx < widthUnit; ++dx) {
            auto dstValue0 = bv;
            auto dstValue1 = bv;
            auto dstValue2 = bv;
            auto dstValue3 = bv;
            for (fy = 0; fy < fh; ++fy) {
                const float* src_y    = srcY + fy * dilateY_step;
                const float* weight_y = weight_z + fy * fw * PACK_UNIT;
                for (fx = 0; fx < fw; ++fx) {
                    const float* src_x    = src_y + fx * dilateX_step;
                    const float* weight_x = weight_y + PACK_UNIT * fx;
                    auto weightValue = _mm512_loadu_ps(weight_x);
                    dstValue0 = _mm512_fmadd_ps(_mm512_loadu_ps(src_x + 0 * src_w_setup), weightValue, dstValue0);
                    dstValue1 = _mm512_fmadd_ps(_mm512_loadu_ps(src_x + 1 * src_w_setup), weightValue, dstValue1);
                    dstValue2 = _mm512_fmadd_ps(_mm512_loadu_ps(src_x + 2 * src_w_setup), weightValue, dstValue2);
                    dstValue3 = _mm512_fmadd_ps(_mm512_loadu_ps(src_x + 3 * src_w_setup), weightValue, dstValue3);
                }
            }
            dstValue0 = _mm512_min_ps(dstValue0, maxF);
            dstValue1 = _mm512_min_ps(dstValue1, maxF);
            dstValue2 = _mm512_min_ps(dstValue2, maxF);
            dstValue3 = _mm512_min_ps(dstValue3, maxF);
            dstValue0 = _mm512_max_ps(dstValue0, minF);
            dstValue1 = _mm512_max_ps(dstValue1, minF);
            dstValue2 = _mm512_max_ps(dstValue2, minF);
            dstValue3 = _mm512_max_ps(dstValue3, minF);
            _mm512_storeu_ps(dstY + PACK_UNIT * 0, dstValue0);
            _mm512_storeu_ps(dstY + PACK_UNIT * 1, dstValue1);
            _mm512_storeu_ps(dstY + PACK_UNIT * 2, dstValue2);
            _mm512_storeu_ps(dstY + PACK_UNIT * 3, dstValue3);
            dstY += PACK_UNIT * unit;
            srcY += unit * src_w_setup;
        }
        for (dx = 0; dx < widthRemain; ++dx) {
            float* dst_x          = dstY + dx * PACK_UNIT;
            auto dstValue = bv;
            const float* src_z    = srcY + src_w_setup * dx;
            const float* weight_z = weight;
            for (fy = 0; fy < fh; ++fy) {
                const float* src_y    = src_z + fy * dilateY_step;
                const float* weight_y = weight_z + fy * fw * PACK_UNIT;
                for (fx = 0; fx < fw; ++fx) {
                    const float* weight_x = weight_y + PACK_UNIT * fx;
                    const float* src_x    = src_y + fx * dilateX_step;
                    dstValue = _mm512_fmadd_ps(_mm512_loadu_ps(src_x), _mm512_loadu_ps(weight_x), dstValue);
                }
            }
            dstValue = _mm512_min_ps(dstValue, maxF);
            dstValue = _mm512_max_ps(dstValue, minF);
            _mm512_storeu_ps(dst_x, dstValue);
        }
    }
}

static MNNBinaryExecute _AVX512_MNNSelectBinaryFunctionForFloat(int opType) {
    auto vecF = MNN::selectVector<Vec16, 16, float>(opType);
    if (nullptr != vecF) {
        return vecF;
    }
    return MNN::MNNGetCoreFunctions()->MNNSelectBinaryFunctionForFloat(opType);
}

void _AVX512_MNNScaleAndAddBias(float* dst, const float* src, const float* bias, const float* alpha, size_t planeNumber,
                        size_t biasNumber) {
    for (int z = 0; z < biasNumber; ++z) {
        float* dstZ         = dst + planeNumber * PACK_UNIT * z;
        const float* srcZ   = src + planeNumber * PACK_UNIT * z;
        auto biasZ = _mm512_loadu_ps(bias + PACK_UNIT * z);
        auto alphaZ = _mm512_loadu_ps(alpha + PACK_UNIT * z);
        for (int p = 0; p < planeNumber; ++p) {
            float* dstX       = dstZ + PACK_UNIT * p;
            const float* srcX = srcZ + PACK_UNIT * p;
            _mm512_storeu_ps(dstX, _mm512_fmadd_ps(_mm512_loadu_ps(srcX), alphaZ, biasZ));
        }
    }
}

void _AVX512_MNNDeconvRunForUnitDepthWise(const float* dst, float* src, const float* weight, size_t fw, size_t fh,
                                  size_t weight_y_step, size_t dilateX_step, size_t dilateY_step) {
    int fx, fy;
    float* src_z          = src;
    const float* weight_z = weight;
    Vec16 dstV             = Vec16::load(dst);
    for (fy = 0; fy < fh; ++fy) {
        float* src_y          = src_z + fy * dilateY_step;
        const float* weight_y = weight_z + fy * weight_y_step;
        for (fx = 0; fx < fw; ++fx) {
            Vec16 weight_x = Vec16::load(weight_y + PACK_UNIT * fx);
            Vec16 src_x    = Vec16::load(src_y + fx * dilateX_step);
            Vec16::save(src_y + fx * dilateX_step, src_x + weight_x * dstV);
        }
    }
}
void _AVX512_MNNDeconvRunForLineDepthwise(const float* dst, float* src, const float* weight, size_t width, size_t src_w_setup,
                                  size_t fw, size_t fh, size_t dilateX_step, size_t dilateY_step) {
    int dx;
    for (dx = 0; dx < width; ++dx) {
        const float* dst_x = dst + dx * PACK_UNIT;
        float* src_dx      = src + src_w_setup * dx;
        _AVX512_MNNDeconvRunForUnitDepthWise(dst_x, src_dx, weight, fw, fh, fw * PACK_UNIT, dilateX_step, dilateY_step);
    }
}

void _AVX512_MNNGridSampleComputeCord(float* dst, const float* src, size_t inH, size_t inW, size_t outH, size_t outW, bool alignCorners) {
    __m512 zero = _mm512_setzero_ps();
    __m512 one = _mm512_set1_ps(1);
    __m512 half = _mm512_set1_ps(0.5f);
    __m512 a = alignCorners ? one : zero;
    __m512 b = alignCorners ? zero : one;
    __m512 inW_sub_a = _mm512_sub_ps(_mm512_set1_ps(inW), a);
    __m512 inH_sub_a = _mm512_sub_ps(_mm512_set1_ps(inH), a);

    int area = outH * outW;
    int areaC4 = area / PACK_UNIT;
    int areaRemain = area - areaC4 * PACK_UNIT;
    for (int i = 0; i < areaC4; ++i) {
        __m512 grid0 = _mm512_loadu_ps(src);
        __m512 grid1 = _mm512_loadu_ps(src + PACK_UNIT);
        __m512 x = _mm512_shuffle_ps(grid0, grid1, 0x88);
        __m512 y = _mm512_shuffle_ps(grid0, grid1, 0xdd);
        __m512 cord_x = _mm512_mul_ps(half, _mm512_sub_ps(_mm512_mul_ps(_mm512_add_ps(one, x), inW_sub_a), b));
        __m512 cord_y = _mm512_mul_ps(half, _mm512_sub_ps(_mm512_mul_ps(_mm512_add_ps(one, y), inH_sub_a), b));
        __m512 cord0 = _mm512_unpacklo_ps(cord_x, cord_y);
        __m512 cord1 = _mm512_unpackhi_ps(cord_x, cord_y);

        _mm512_storeu_ps(dst, cord0);
        _mm512_storeu_ps(dst + PACK_UNIT, cord1);

        src += PACK_UNIT * 2;
        dst += PACK_UNIT * 2;
    }

    if (areaRemain > 0) {
        __mmask16 mask = 0xffff;
        if (areaRemain > PACK_UNIT / 2) {
            int shift = areaRemain * 2 - PACK_UNIT;
            mask = (1 << shift) - 1;
            __m512 grid0 = _mm512_loadu_ps(src);
            __m512 grid1 = _mm512_maskz_loadu_ps(mask, src + PACK_UNIT);
            __m512 x = _mm512_shuffle_ps(grid0, grid1, 0x88);
            __m512 y = _mm512_shuffle_ps(grid0, grid1, 0xdd);
            __m512 cord_x = _mm512_mul_ps(half, _mm512_sub_ps(_mm512_mul_ps(_mm512_add_ps(one, x), inW_sub_a), b));
            __m512 cord_y = _mm512_mul_ps(half, _mm512_sub_ps(_mm512_mul_ps(_mm512_add_ps(one, y), inH_sub_a), b));
            __m512 cord0 = _mm512_unpacklo_ps(cord_x, cord_y);
            __m512 cord1 = _mm512_unpackhi_ps(cord_x, cord_y);

            _mm512_storeu_ps(dst, cord0);
            _mm512_mask_storeu_ps(dst + PACK_UNIT, mask, cord1);
        } else {
            int shift = areaRemain * 2;
            mask = (1 << shift) - 1;
            __m512 grid0 = _mm512_maskz_loadu_ps(mask, src);
            __m512 grid1 = zero;
            __m512 x = _mm512_shuffle_ps(grid0, grid1, 0x88);
            __m512 y = _mm512_shuffle_ps(grid0, grid1, 0xdd);
            __m512 cord_x = _mm512_mul_ps(half, _mm512_sub_ps(_mm512_mul_ps(_mm512_add_ps(one, x), inW_sub_a), b));
            __m512 cord_y = _mm512_mul_ps(half, _mm512_sub_ps(_mm512_mul_ps(_mm512_add_ps(one, y), inH_sub_a), b));
            __m512 cord0 = _mm512_unpacklo_ps(cord_x, cord_y);

            _mm512_mask_storeu_ps(dst, mask, cord0);
        }
    }
}

void _AVX512_MNNRoiPoolingMax(float* dst, const float* src, int hLen, int wLen, int iw) {
    Vec16 max = Vec16(-FLT_MAX);
    for (int h = 0; h < hLen; h++, src += iw * PACK_UNIT) {
        for (int w = 0; w < wLen; w++) {
            Vec16 in = Vec16::load(src + w * PACK_UNIT);
            max = Vec16::max(max, in);
        }
    }
    Vec16::save(dst, max);
}

void _AVX512_MNNRoiAlignMax(float* dst, const float* src, const std::vector<std::vector<int>> &vecPos, const std::vector<std::vector<float>> &vecArea, int samplingRatioArea, int pooledHeight, int pooledWidth) {
    for (int h = 0; h < pooledHeight; ++h, dst += pooledWidth * PACK_UNIT) {
        int preCalcIdx = h * pooledWidth * samplingRatioArea;
        for (int w = 0; w < pooledWidth; ++w) {
            Vec16 res = Vec16(-FLT_MAX);
            for (int i = 0; i < samplingRatioArea; ++i) {
                const std::vector<int>& pos    = vecPos[preCalcIdx];
                const std::vector<float>& area = vecArea[preCalcIdx];

                Vec16 val0 = Vec16::load(src + pos[0] * PACK_UNIT);
                Vec16 val1 = Vec16::load(src + pos[1] * PACK_UNIT);
                Vec16 val2 = Vec16::load(src + pos[2] * PACK_UNIT);
                Vec16 val3 = Vec16::load(src + pos[3] * PACK_UNIT);
                Vec16 mla  = val0 * area[0];
                mla       = Vec16::fma(mla, val1, area[1]);
                mla       = Vec16::fma(mla, val2, area[2]);
                mla       = Vec16::fma(mla, val3, area[3]);
                res       = Vec16::max(res, mla);
                preCalcIdx++;
            }
            Vec16::save(dst + w * PACK_UNIT, res);
        }
    }
}

void _AVX512_MNNRoiAlignAvg(float* dst, const float* src, const std::vector<std::vector<int>> &vecPos, const std::vector<std::vector<float>> &vecArea, int samplingRatioArea, int pooledHeight, int pooledWidth) {
    float invSamplingCnt = 1.f / samplingRatioArea;
    for (int h = 0; h < pooledHeight; ++h, dst += pooledWidth * PACK_UNIT) {
        int preCalcIdx = h * pooledWidth * samplingRatioArea;
        for (int w = 0; w < pooledWidth; ++w) {
            Vec16 res = Vec16(0.f);
            for (int i = 0; i < samplingRatioArea; ++i) {
                const std::vector<int>& pos    = vecPos[preCalcIdx];
                const std::vector<float>& area = vecArea[preCalcIdx];

                Vec16 val0 = Vec16::load(src + pos[0] * PACK_UNIT);
                Vec16 val1 = Vec16::load(src + pos[1] * PACK_UNIT);
                Vec16 val2 = Vec16::load(src + pos[2] * PACK_UNIT);
                Vec16 val3 = Vec16::load(src + pos[3] * PACK_UNIT);
                Vec16 mla  = val0 * area[0];
                mla       = Vec16::fma(mla, val1, area[1]);
                mla       = Vec16::fma(mla, val2, area[2]);
                mla       = Vec16::fma(mla, val3, area[3]);
                res       += mla;
                preCalcIdx++;
            }
            res = res * invSamplingCnt;
            Vec16::save(dst + w * PACK_UNIT, res);
        }
    }
}

void _AVX512_MNNMatrixAdd(float* C, const float* A, const float* B, size_t widthC4, size_t cStride, size_t aStride,
                       size_t bStride, size_t height) {
    for (int y = 0; y < height; ++y) {
        auto a = A + aStride * y;
        auto b = B + bStride * y;
        auto c = C + cStride * y;
        for (int x = 0; x < widthC4; ++x) {
            _mm512_storeu_ps(c + PACK_UNIT * x, _mm512_add_ps(_mm512_loadu_ps(b + PACK_UNIT * x), _mm512_loadu_ps(a + PACK_UNIT * x)));
        }
    }
}

void _AVX512_MNNStrassenMergeCFunction(float* c11, float* c12, float* c21, float* c22, float* xAddr, size_t cStride, size_t eSub, size_t hSub) {
    const int unit = PACK_UNIT;
    for (int y=0; y<hSub; ++y) {
        auto c11Y = c11 + y * cStride;
        auto c12Y = c12 + y * cStride;
        auto c22Y = c22 + y * cStride;
        auto c21Y = c21 + y * cStride;
        auto xY = xAddr + y * eSub * unit;
        for (int x=0; x<eSub; ++x) {
            auto xv = _mm512_loadu_ps(xY + unit*x);
            auto c21v = _mm512_loadu_ps(c21Y + unit*x);
            auto c11v = _mm512_loadu_ps(c11Y + unit*x);
            auto c22v = _mm512_loadu_ps(c22Y + unit*x);
            auto c12v = _mm512_loadu_ps(c12Y + unit*x);
            c12v = _mm512_add_ps(c12v, xv);
            c21v = _mm512_add_ps(c12v, c21v);
            c12v = _mm512_add_ps(c22v, c12v);
            c22v = _mm512_add_ps(c22v, c21v);
            c12v = _mm512_add_ps(c11v, c12v);
            _mm512_storeu_ps(c12Y + unit*x, c12v);
            _mm512_storeu_ps(c22Y + unit*x, c22v);
            _mm512_storeu_ps(c21Y + unit*x, c21v);
        }
    }
}

void _AVX512_MNNMatrixSub(float* C, const float* A, const float* B, size_t widthC4, size_t cStride, size_t aStride,
                       size_t bStride, size_t height) {
    for (int y = 0; y < height; ++y) {
        auto a = A + aStride * y;
        auto b = B + bStride * y;
        auto c = C + cStride * y;
        for (int x = 0; x < widthC4; ++x) {
            _mm512_storeu_ps(c + PACK_UNIT * x, _mm512_sub_ps(_mm512_loadu_ps(a + PACK_UNIT * x), _mm512_loadu_ps(b + PACK_UNIT * x)));
        }
    }
}

void _AVX512_MNNMultiAndDestTransformCommon23(float **cacheLine, const float *weigth, float *dest, int cacheLineSize, int ow, const float* bias, const float* parameter) {
    int unit = ow / 2;
    MNN_ASSERT(cacheLineSize >= 1);
    auto biasF = Vec16::load(bias);
    auto minF = Vec16(parameter[2]);
    auto maxF = Vec16(parameter[3]);
    auto SRC_TILE_UNIT = 4 * PACK_UNIT;
    auto DST_TILE_UNIT = 2 * PACK_UNIT;
    for (int x = 0; x < unit; ++x) {
        auto offset = SRC_TILE_UNIT * x;
        int i = 0;
        Vec16 m0     = Vec16::load(weigth + i * SRC_TILE_UNIT + PACK_UNIT * 0) * Vec16::load(cacheLine[i] + offset + PACK_UNIT * 0);
        Vec16 m1     = Vec16::load(weigth + i * SRC_TILE_UNIT + PACK_UNIT * 1) * Vec16::load(cacheLine[i] + offset + PACK_UNIT * 1);
        Vec16 m2     = Vec16::load(weigth + i * SRC_TILE_UNIT + PACK_UNIT * 2) * Vec16::load(cacheLine[i] + offset + PACK_UNIT * 2);
        Vec16 m3     = Vec16::load(weigth + i * SRC_TILE_UNIT + PACK_UNIT * 3) * Vec16::load(cacheLine[i] + offset + PACK_UNIT * 3);

        for (i = 1; i < cacheLineSize; ++i) {
            m0 = m0 + Vec16::load(weigth + i * SRC_TILE_UNIT + PACK_UNIT * 0) * Vec16::load(cacheLine[i] + offset + PACK_UNIT * 0);
            m1 = m1 + Vec16::load(weigth + i * SRC_TILE_UNIT + PACK_UNIT * 1) * Vec16::load(cacheLine[i] + offset + PACK_UNIT * 1);
            m2 = m2 + Vec16::load(weigth + i * SRC_TILE_UNIT + PACK_UNIT * 2) * Vec16::load(cacheLine[i] + offset + PACK_UNIT * 2);
            m3 = m3 + Vec16::load(weigth + i * SRC_TILE_UNIT + PACK_UNIT * 3) * Vec16::load(cacheLine[i] + offset + PACK_UNIT * 3);
        }
        auto o0 = m0 + m1 + m2 + biasF;
        auto o1 = m1 - m2 + m3 + biasF;
        o0 = Vec16::min(maxF, o0);
        o1 = Vec16::min(maxF, o1);
        o0 = Vec16::max(minF, o0);
        o1 = Vec16::max(minF, o1);

        Vec16::save(dest + DST_TILE_UNIT * x + 0 * PACK_UNIT, o0);
        Vec16::save(dest + DST_TILE_UNIT * x + 1 * PACK_UNIT, o1);
    }
    if (unit * 2 < ow) {
        auto offset = SRC_TILE_UNIT * unit;
        int i = 0;
        Vec16 m0     = Vec16::load(weigth + i * SRC_TILE_UNIT + PACK_UNIT * 0) * Vec16::load(cacheLine[i] + offset + PACK_UNIT * 0);
        Vec16 m1     = Vec16::load(weigth + i * SRC_TILE_UNIT + PACK_UNIT * 1) * Vec16::load(cacheLine[i] + offset + PACK_UNIT * 1);
        Vec16 m2     = Vec16::load(weigth + i * SRC_TILE_UNIT + PACK_UNIT * 2) * Vec16::load(cacheLine[i] + offset + PACK_UNIT * 2);

        for (i = 1; i < cacheLineSize; ++i) {
            m0 = m0 + Vec16::load(weigth + i * SRC_TILE_UNIT + PACK_UNIT * 0) * Vec16::load(cacheLine[i] + offset + PACK_UNIT * 0);
            m1 = m1 + Vec16::load(weigth + i * SRC_TILE_UNIT + PACK_UNIT * 1) * Vec16::load(cacheLine[i] + offset + PACK_UNIT * 1);
            m2 = m2 + Vec16::load(weigth + i * SRC_TILE_UNIT + PACK_UNIT * 2) * Vec16::load(cacheLine[i] + offset + PACK_UNIT * 2);
        }
        auto o0 = m0 + m1 + m2 + biasF;
        o0 = Vec16::min(maxF, o0);
        o0 = Vec16::max(minF, o0);
        Vec16::save(dest + DST_TILE_UNIT * unit, o0);
    }
}
static void _AVX512_MNNConvDwF23SourceTransUnit(const float *source, float *dest, size_t unit) {
    if (unit <= 0) {
        return;
    }
    Vec16 v0 = Vec16::load(source + PACK_UNIT * 0);
    Vec16 v1 = Vec16::load(source + PACK_UNIT * 1);
    Vec16 v2;
    Vec16 v3;
    source += 2 * PACK_UNIT;

    for (int x = 0; x < unit; ++x) {
        v2 = Vec16::load(source + 0 * PACK_UNIT);
        v3 = Vec16::load(source + 1 * PACK_UNIT);
        auto m0 = v0 - v2;
        auto m1 = v1 + v2;
        auto m2 = v2 - v1;
        auto m3 = v3 - v1;

        Vec16::save(dest + PACK_UNIT * 0, m0);
        Vec16::save(dest + PACK_UNIT * 1, m1);
        Vec16::save(dest + PACK_UNIT * 2, m2);
        Vec16::save(dest + PACK_UNIT * 3, m3);

        source += (2 * PACK_UNIT);
        dest += (4 * PACK_UNIT);

        v0 = v2;
        v1 = v3;
    }
}

void _AVX512_MNNSourceTransformCommonF23(const float *source, float *dest, int unit, int iw, int pad, int su, int eu) {
    for (int x = 0; x < su; ++x) {
        auto dstX = dest + 4 * PACK_UNIT * x;
        auto sx   = x * 2 - (int)pad;
        auto ex   = sx + 4;

        auto clampSx = std::max(sx, 0);
        auto clampEx = std::min(ex, (int)iw);

        Vec16 v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int i = clampSx; i < clampEx; ++i) {
            v[i - sx] = Vec16::load(source + PACK_UNIT * i);
        }
        auto m0 = v[0] - v[2];
        auto m1 = v[1] + v[2];
        auto m2 = v[2] - v[1];
        auto m3 = v[3] - v[1];

        Vec16::save(dstX + PACK_UNIT * 0, m0);
        Vec16::save(dstX + PACK_UNIT * 1, m1);
        Vec16::save(dstX + PACK_UNIT * 2, m2);
        Vec16::save(dstX + PACK_UNIT * 3, m3);
    }
    _AVX512_MNNConvDwF23SourceTransUnit(source + PACK_UNIT * (su * 2 - pad), dest + PACK_UNIT * 4 * su, eu - su);

    for (int x = eu; x < unit; ++x) {
        auto dstX = dest + PACK_UNIT * 4 * x;
        auto sx   = x * 2 - (int)pad;
        auto ex   = sx + 4;

        auto clampSx = std::max(sx, 0);
        auto clampEx = std::min(ex, (int)iw);

        Vec16 v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int i = clampSx; i < clampEx; ++i) {
            v[i - sx] = Vec16::load(source + PACK_UNIT * i);
        }
        auto m0 = v[0] - v[2];
        auto m1 = v[1] + v[2];
        auto m2 = v[2] - v[1];
        auto m3 = v[3] - v[1];

        Vec16::save(dstX + PACK_UNIT * 0, m0);
        Vec16::save(dstX + PACK_UNIT * 1, m1);
        Vec16::save(dstX + PACK_UNIT * 2, m2);
        Vec16::save(dstX + PACK_UNIT * 3, m3);
    }
}

void _AVX512_MNNConvDwF23MulTransUnit(float **cacheLine, const float *weigth, float *dest, size_t ow, const float* bias, const float* parameter) {
    int unit = ow / 2;
    auto SRC_TILE_UNIT = 4 * PACK_UNIT;
    auto DST_TILE_UNIT = 2 * PACK_UNIT;

    auto w00 = _mm512_loadu_ps(weigth + 0 * SRC_TILE_UNIT + PACK_UNIT * 0);
    auto w01 = _mm512_loadu_ps(weigth + 0 * SRC_TILE_UNIT + PACK_UNIT * 1);
    auto w02 = _mm512_loadu_ps(weigth + 0 * SRC_TILE_UNIT + PACK_UNIT * 2);
    auto w03 = _mm512_loadu_ps(weigth + 0 * SRC_TILE_UNIT + PACK_UNIT * 3);
    auto w10 = _mm512_loadu_ps(weigth + 1 * SRC_TILE_UNIT + PACK_UNIT * 0);
    auto w11 = _mm512_loadu_ps(weigth + 1 * SRC_TILE_UNIT + PACK_UNIT * 1);
    auto w12 = _mm512_loadu_ps(weigth + 1 * SRC_TILE_UNIT + PACK_UNIT * 2);
    auto w13 = _mm512_loadu_ps(weigth + 1 * SRC_TILE_UNIT + PACK_UNIT * 3);
    auto w20 = _mm512_loadu_ps(weigth + 2 * SRC_TILE_UNIT + PACK_UNIT * 0);
    auto w21 = _mm512_loadu_ps(weigth + 2 * SRC_TILE_UNIT + PACK_UNIT * 1);
    auto w22 = _mm512_loadu_ps(weigth + 2 * SRC_TILE_UNIT + PACK_UNIT * 2);
    auto w23 = _mm512_loadu_ps(weigth + 2 * SRC_TILE_UNIT + PACK_UNIT * 3);
    auto biasF = _mm512_loadu_ps(bias);
    auto minF = _mm512_broadcastss_ps(_mm_load_ss(parameter + 2));
    auto maxF = _mm512_broadcastss_ps(_mm_load_ss(parameter + 3));

    for (int x = 0; x < unit; ++x) {
        auto offset = PACK_UNIT * 4 * x;
        int i = 0;
        auto m0     = _mm512_mul_ps(w00, _mm512_loadu_ps(cacheLine[0] + offset + PACK_UNIT * 0));
        auto m1     = _mm512_mul_ps(w01, _mm512_loadu_ps(cacheLine[0] + offset + PACK_UNIT * 1));
        auto m2     = _mm512_mul_ps(w02, _mm512_loadu_ps(cacheLine[0] + offset + PACK_UNIT * 2));
        auto m3     = _mm512_mul_ps(w03, _mm512_loadu_ps(cacheLine[0] + offset + PACK_UNIT * 3));

        m0 = _mm512_fmadd_ps(w10, _mm512_loadu_ps(cacheLine[1] + offset + PACK_UNIT * 0), m0);
        m1 = _mm512_fmadd_ps(w11, _mm512_loadu_ps(cacheLine[1] + offset + PACK_UNIT * 1), m1);
        m2 = _mm512_fmadd_ps(w12, _mm512_loadu_ps(cacheLine[1] + offset + PACK_UNIT * 2), m2);
        m3 = _mm512_fmadd_ps(w13, _mm512_loadu_ps(cacheLine[1] + offset + PACK_UNIT * 3), m3);

        m0 = _mm512_fmadd_ps(w20, _mm512_loadu_ps(cacheLine[2] + offset + PACK_UNIT * 0), m0);
        m1 = _mm512_fmadd_ps(w21, _mm512_loadu_ps(cacheLine[2] + offset + PACK_UNIT * 1), m1);
        m2 = _mm512_fmadd_ps(w22, _mm512_loadu_ps(cacheLine[2] + offset + PACK_UNIT * 2), m2);
        m3 = _mm512_fmadd_ps(w23, _mm512_loadu_ps(cacheLine[2] + offset + PACK_UNIT * 3), m3);

        auto o0 = _mm512_add_ps(_mm512_add_ps(m0, _mm512_add_ps(m1, m2)), biasF);
        auto o1 = _mm512_add_ps(_mm512_add_ps(m3, _mm512_sub_ps(m1, m2)), biasF);
        o0 = _mm512_min_ps(maxF, o0);
        o1 = _mm512_min_ps(maxF, o1);
        o0 = _mm512_max_ps(minF, o0);
        o1 = _mm512_max_ps(minF, o1);
        _mm512_storeu_ps(dest + DST_TILE_UNIT * x + 0 * PACK_UNIT, o0);
        _mm512_storeu_ps(dest + DST_TILE_UNIT * x + 1 * PACK_UNIT, o1);
    }
    if (unit * 2 < ow) {
        auto offset = PACK_UNIT * 4 * unit;
        auto m0     = _mm512_mul_ps(w00, _mm512_loadu_ps(cacheLine[0] + offset + PACK_UNIT * 0));
        auto m1     = _mm512_mul_ps(w01, _mm512_loadu_ps(cacheLine[0] + offset + PACK_UNIT * 1));
        auto m2     = _mm512_mul_ps(w02, _mm512_loadu_ps(cacheLine[0] + offset + PACK_UNIT * 2));

        m0 = _mm512_fmadd_ps(w10, _mm512_loadu_ps(cacheLine[1] + offset + PACK_UNIT * 0), m0);
        m1 = _mm512_fmadd_ps(w11, _mm512_loadu_ps(cacheLine[1] + offset + PACK_UNIT * 1), m1);
        m2 = _mm512_fmadd_ps(w12, _mm512_loadu_ps(cacheLine[1] + offset + PACK_UNIT * 2), m2);

        m0 = _mm512_fmadd_ps(w20, _mm512_loadu_ps(cacheLine[2] + offset + PACK_UNIT * 0), m0);
        m1 = _mm512_fmadd_ps(w21, _mm512_loadu_ps(cacheLine[2] + offset + PACK_UNIT * 1), m1);
        m2 = _mm512_fmadd_ps(w22, _mm512_loadu_ps(cacheLine[2] + offset + PACK_UNIT * 2), m2);

        auto o0 = _mm512_add_ps(_mm512_add_ps(m0, _mm512_add_ps(m1, m2)), biasF);
        o0 = _mm512_min_ps(maxF, o0);
        o0 = _mm512_max_ps(minF, o0);
        _mm512_storeu_ps(dest + DST_TILE_UNIT * unit, o0);
    }
}
static void _8BitcopyWithStrideC4(uint8_t* dstO, const uint8_t* srcO, int size, int stride, int ds) {
    auto src = (float*)srcO;
    auto dst = (float*)dstO;
    for (int i=0; i<size; ++i) {
        _mm256_storeu_ps(dst, _mm256_loadu_ps(src));
        src+= (8 * stride);
        dst+= (8 * ds);
    }
}
static void _16BitcopyWithStrideC4(uint8_t* dstO, const uint8_t* srcO, int size, int stride, int ds) {
    auto src = (float*)srcO;
    auto dst = (float*)dstO;
    for (int i=0; i<size; ++i) {
        _mm512_storeu_ps(dst, _mm512_loadu_ps(src));
        src+= (16 * stride);
        dst+= (16 * ds);
    }
}
static MNNCopyWithStride _selectBlit(int bytesC4) {
    if (64 == bytesC4) {
        return _16BitcopyWithStrideC4;
    }
    if (32 == bytesC4) {
        return _8BitcopyWithStrideC4;
    }
    return nullptr;
}

static void _AVX512_MNNAdjustOptimalSparseKernel(int& sparseBlockOC, MNN::CoreFunctions::MNNPackedSparseMatMul& packedSparseMatMul) {
    if(sparseBlockOC == 8) {
        packedSparseMatMul = _AVX512_MNNPackedSparseMatMulEpx8;
        return;
    } else if(sparseBlockOC % 8 == 0) {
        // MNN_PRINT("avx512 downgrade sparse from:%d, ",sparseBlockOC);
        sparseBlockOC = 8;
        packedSparseMatMul = _AVX512_MNNPackedSparseMatMulEpx8;
        // MNN_PRINT(" to:%d\n",sparseBlockOC);
        return;
    }
    else if(sparseBlockOC == 4) {
        packedSparseMatMul = _AVX512_MNNPackedSparseMatMulEpx4;
        return;
    } else if(sparseBlockOC % 4 == 0) {
        // MNN_PRINT("avx512 downgrade sparse from:%d, ",sparseBlockOC);
        sparseBlockOC = 4;
        packedSparseMatMul = _AVX512_MNNPackedSparseMatMulEpx4;
        // MNN_PRINT(" to:%d\n",sparseBlockOC);
        return;
    } else {
        sparseBlockOC = 1;
        packedSparseMatMul = _AVX512_MNNPackedSparseMatMulEpx1;
        return;
    }
}

static void _AVX512_MNNSoftmax(float* softmaxDst, const float* softmaxSrc, float* runningMax, float* runningSum, float* updateScale, int outside, int reduceSize, int kvSeqOffset, int validOffset, int pack, bool mask) {
    const int packUnit = 16;
    int reduceSizeOuter = 1;
    int reduceSizeInner = reduceSize;
    int stride0         = packUnit;
    if (pack > 1) {
        reduceSizeOuter = UP_DIV(reduceSize, pack);
        reduceSizeInner = pack;
        stride0         = outside * reduceSizeInner;
    }

    float exprOffset[4] = {1.0f, 0.0f, 0.0f, 0.0f };
    for (int k = 0; k < outside; ++k) {
        exprOffset[3] = 0.f;
        if (mask && kvSeqOffset > k + validOffset) {
            if (updateScale){
                updateScale[k] = 1;
            }
            for (int j = 0; j < reduceSizeOuter; ++j) {
                auto destPtr = softmaxDst + j * stride0 + k * reduceSizeInner;
                memset(destPtr, 0, reduceSizeInner * sizeof(float));
            }
            continue;
        }

        const int validReduceSize = mask ? ALIMIN(reduceSize, k + (validOffset + 1) - kvSeqOffset) : reduceSize;
        const int remain = validReduceSize % packUnit;
        const int sizeDiv = validReduceSize / packUnit;
        const float floatLowest = std::numeric_limits<float>::lowest();

        // 1. newMax
        float oldMax = floatLowest;
        if (runningMax) {
            oldMax = runningMax[k];
        }

        __m512 maxVec = _mm512_set1_ps(floatLowest);
        for (int j = 0; j < sizeDiv; ++j) {
            auto srcPtr = softmaxSrc + j * stride0 + k * reduceSizeInner;
            __m512 srcVec = _mm512_loadu_ps(srcPtr);
            maxVec = _mm512_max_ps(maxVec, srcVec);
        }
        float newMax = _mm512_reduce_max_ps(maxVec);

        if (remain > 0) {
            auto srcPtr = softmaxSrc + sizeDiv * stride0 + k * reduceSizeInner;
            for (int i = 0; i < remain; ++i) {
                newMax = ALIMAX(newMax, srcPtr[i]);
            }
        }

        const float finalMax = ALIMAX(oldMax, newMax);
        const __m512 finalMaxVec = _mm512_set1_ps(finalMax);
        exprOffset[2] = -finalMax;

        // 2. exp(x - finalMax) and Sum
        __m512 sumVec = _mm512_setzero_ps();
        for (int j = 0; j < sizeDiv; ++j) {
            auto idx = j * stride0 + k * reduceSizeInner;
            auto srcPtr = softmaxSrc + idx;
            auto dstPtr = softmaxDst + idx;

            MNNExp(dstPtr, srcPtr, exprOffset, packUnit);
        }

        float sum = exprOffset[3];

        if (remain > 0) {
            auto idx = sizeDiv * stride0 + k * reduceSizeInner;
            auto srcPtr = softmaxSrc + idx;
            auto dstPtr = softmaxDst + idx;
            for (int i = 0; i < remain; ++i) {
                float val = expf(srcPtr[i] - finalMax);
                sum += val;
                dstPtr[i] = val;
            }
        }

        // 3. Normalization or update state
        if (runningMax != nullptr && runningSum != nullptr && updateScale != nullptr) {
            float scaleForSum = expf(oldMax - finalMax);
            runningSum[k] = runningSum[k] * scaleForSum + sum;
            runningMax[k] = finalMax;
            updateScale[k] = scaleForSum;
        } else {
            if (runningMax != nullptr && runningSum != nullptr) {
                sum += runningSum[k] * expf(oldMax - finalMax);
            }
            float scale = 1.0f / (sum + 1e-20f);
            __m512 scaleVec = _mm512_set1_ps(scale);

            for (int j = 0; j < sizeDiv; ++j) {
                auto pDest = softmaxDst + j * stride0 + k * reduceSizeInner;
                __m512 data = _mm512_loadu_ps(pDest);
                data = _mm512_mul_ps(data, scaleVec);
                _mm512_storeu_ps(pDest, data);
            }
            if (remain > 0) {
                auto pDest = softmaxDst + sizeDiv * stride0 + k * reduceSizeInner;
                for (int i = 0; i < remain; ++i) {
                    pDest[i] *= scale;
                }
            }
        }

        // 4. memset 0 for padding (逻辑不变)
        if (pack > 1) {
            if (validReduceSize % pack > 0) {
                memset(softmaxDst + (UP_DIV(validReduceSize, pack) - 1) * stride0 + k * reduceSizeInner + (validReduceSize % pack), 0, (pack - (validReduceSize % pack)) * sizeof(float));
            }
            auto validOuter = UP_DIV(validReduceSize, pack);
            auto allOuter = UP_DIV(reduceSize, pack);
            for (int j = validOuter; j < allOuter; ++j) {
                auto destPtr = softmaxDst + j * stride0 + k * reduceSizeInner;
                memset(destPtr, 0, pack * sizeof(float));
            }
        } else {
             memset(softmaxDst + k * reduceSizeInner + validReduceSize, 0, (reduceSize - validReduceSize) * sizeof(float));
        }
    }
}

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE
void _AVX512_MNNFlashAttentionUpdateBlockOutput(float* dst, float* src, float* scale, float* normalizeScale, int depthQuad, int plane, int pack, int idx, int kvBlocks, int size, int bytes, int seqStart) {
    // source shape:                 [headDim/pack, seqLen, pack]
    // scale & normalizeScale shape: [seqLen]
    // dest shape:                   [headDim/pack, seqLen, pack]
    auto stride0 = plane * pack;

    if (idx > 0) {
        for (int j = 0; j < depthQuad; ++j) {
            int i = seqStart;
            for (; i < plane; ++i) {
                auto dataNew = Vec::load(src + j * stride0 + i * pack);
                auto dataOld = Vec::load(dst + j * stride0 + i * pack);
                auto s = Vec(scale[i]);
                dataNew = Vec::fma(dataNew, dataOld, s);
                Vec::save(dst + j * stride0 + i * pack, dataNew);
            }
        }
    } else {
        memcpy(dst, src, size * bytes);
    }
    if (idx == kvBlocks - 1) { // if last subBlock, exp(xi)/sum(exp(xi))
        for (int j = 0; j < depthQuad; ++j) {
            for (int i = 0; i < plane; ++i) {
                auto dataNew = Vec::load(dst + j * stride0 + i * pack);
                auto ns = Vec(1.0f / normalizeScale[i]);
                dataNew = dataNew * ns;
                Vec::save(dst + j * stride0 + i * pack, dataNew);
            }
        }
    }
}
#endif


void _AVX512_ExtraInit(void* functions) {
    auto coreFunction = static_cast<MNN::CoreFunctions*>(functions);
    coreFunction->MNNSelectBlitFunction = _selectBlit;
    coreFunction->MNNPoolingAvg = (decltype(coreFunction->MNNPoolingAvg))(MNN::poolingAvg<float, Vec16, 16>);
    // Set min value as 1 << 24
    coreFunction->MNNPoolingMax = (decltype(coreFunction->MNNPoolingMax))(MNN::poolingMax<float, Vec16, 16, -16777216>);
    coreFunction->MNNPoolingMaxWithRedice = (decltype(coreFunction->MNNPoolingMaxWithRedice))(MNN::poolingMaxWithRedice<float, -16777216>);
    coreFunction->MNNSelectBinaryFunctionForFloat = _AVX512_MNNSelectBinaryFunctionForFloat;
    coreFunction->MNNCopyC4WithStride = _AVX512_MNNCopyC4WithStride;
    coreFunction->MNNAddC4WithStride = _AVX512_MNNAddC4WithStride;
    coreFunction->MNNScaleAndAddBias = _AVX512_MNNScaleAndAddBias;
    coreFunction->MNNMatrixAdd          = _AVX512_MNNMatrixAdd;
    coreFunction->MNNMatrixSub          = _AVX512_MNNMatrixSub;
    coreFunction->MNNAbsMax = _AVX512_MNNAbsMaxFP32;
    coreFunction->MNNDynamicQuant = _AVX512_DynamicQuant;
    coreFunction->MNNAsyQuantInfo = _AVX512_MNNAsyQuantInfo;
    coreFunction->MNNAsyQuantFunc = _AVX512_MNNAsyQuantFunc;
    coreFunction->MNNCountMaxMinValue = _AVX512_MNNCountMinMaxValue;
    coreFunction->MNNSoftmax = _AVX512_MNNSoftmax;

    coreFunction->MNNConvRunForLineDepthwise = _AVX512_MNNConvRunForLineDepthwise;
    coreFunction->MNNAxByClampBroadcastUnit = _AVX512_MNNAxByClampBroadcastUnit;
    coreFunction->MNNStrassenMergeCFunction = _AVX512_MNNStrassenMergeCFunction;
    coreFunction->MNNReluWithSlopeChannel = _AVX512_MNNReluWithSlopeChannel;
    coreFunction->MNNDeconvRunForLineDepthwise = _AVX512_MNNDeconvRunForLineDepthwise;
    coreFunction->MNNDeconvRunForUnitDepthWise = _AVX512_MNNDeconvRunForUnitDepthWise;
    coreFunction->MNNGridSampleComputeCord = _AVX512_MNNGridSampleComputeCord;
    coreFunction->MNNRoiPoolingMax = _AVX512_MNNRoiPoolingMax;
    coreFunction->MNNRoiAlignMax = _AVX512_MNNRoiAlignMax;
    coreFunction->MNNRoiAlignAvg = _AVX512_MNNRoiAlignAvg;
    coreFunction->MNNGridSampleInterp = MNNGridSampleInterp;
    coreFunction->MNNGridSampleInterp3D = MNNGridSampleInterp3D;
    coreFunction->MNNGridSampleInterpGrad = MNNGridSampleInterpGrad;

    coreFunction->MNNGetSparseMatMulPackMode = _AVX512_MNNGetSparseMatMulPackMode;
    coreFunction->MNNAdjustOptimalSparseKernel = _AVX512_MNNAdjustOptimalSparseKernel;

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE
    coreFunction->MNNFlashAttentionUpdateBlockOutput = _AVX512_MNNFlashAttentionUpdateBlockOutput;
#endif
}
