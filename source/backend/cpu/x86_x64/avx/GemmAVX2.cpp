//
//  GemmAVX2.cpp
//  MNN
//
//  Created by MNN on 2020/09/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "FunctionSummary.hpp"
#include "GemmCommon.hpp"
#include "core/Macro.h"
#define MNNAVXFMA(x, y, z) _mm256_add_ps(_mm256_mul_ps(x, y), z)
#define MNNSSEFMA(x, y, z) _mm_add_ps(_mm_mul_ps(x, y), z)
#define BROAD_LOAD(x) _mm256_broadcast_ss(x)
#define BROAD_LOAD_4(x) _mm_broadcast_ss(x)
#define LOAD8(x) _mm256_loadu_ps(x)
#define LOAD4(x) _mm_loadu_ps(x)
#define STORE_4(d, x) _mm_storeu_ps(d, x) // The memory is aligned for 4
#define STORE_8(d, x) _mm256_storeu_ps(d, x)
#include "GemmFunction.hpp"

void _AVX_MNNPackedMatMul(float* C, const float* A, const float* B, const size_t* parameter,
                          const float* postParameters, const float* bias, const float* k, const float* b) {
    _AVX_MNNPackedMatMul_Main(C, A, B, parameter);
    AVX2GemmPostTreat(C, MNN_UNIT_E, parameter, postParameters, bias);
}

void _AVX_MNNPackedMatMulRemain(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter,
                             const float* postParameters, const float* bias, const float* k, const float* b) {
    _AVX_MNNPackednMatMulRemainCommon(C, A, B, eSize, parameter);
    AVX2GemmPostTreat(C, eSize, parameter, postParameters, bias);
}

#ifdef MNN_CPU_WEIGHT_DEQUANT_GEMM
void _AVX_MNNPackedMatMul_int4(float* C, const float* A, const float* B, const size_t* parameter,
                               const float* postParameters, const float* bias, const float* k, const float* b) {
    _AVX_MNNPackedMatMul_Main_int4(C, A, B, parameter, k, b);
    if (nullptr != bias) {
        AVX2GemmPostTreat(C, MNN_UNIT_E, parameter, postParameters, bias);
    }
}
void _AVX_MNNPackedMatMulRemain_int4(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter,
                             const float* postParameters, const float* bias, const float* k, const float* b) {
    _AVX_MNNPackednMatMulRemainCommon_int4(C, A, B, eSize, parameter, k, b);
    if (nullptr != bias) {
        AVX2GemmPostTreat(C, eSize, parameter, postParameters, bias);
    }
}
void _AVX_MNNPackedMatMul_int8(float* C, const float* A, const float* B, const size_t* parameter,
                               const float* postParameters, const float* bias, const float* k, const float* b) {
    _AVX_MNNPackedMatMul_Main_int8(C, A, B, parameter, k, b);
    if (nullptr != bias) {
        AVX2GemmPostTreat(C, MNN_UNIT_E, parameter, postParameters, bias);
    }
}
void _AVX_MNNPackedMatMulRemain_int8(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter,
                             const float* postParameters, const float* bias, const float* k, const float* b) {
    _AVX_MNNPackednMatMulRemainCommon_int8(C, A, B, eSize, parameter, k, b);
    if (nullptr != bias) {
        AVX2GemmPostTreat(C, eSize, parameter, postParameters, bias);
    }
}
#endif

#ifdef MNN_LOW_MEMORY
void _AVX_MNNAbsMaxFP32(const float* source, float* absmax, size_t src_depth_quad, size_t realSize, int pack) {
    // source: (ic/8, N, 8)
    auto srcStep = pack * realSize;
    if (pack == 8) {
        float temp[8];
        auto constant = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
        for (int i = 0; i < realSize; ++i) {
            __m256 res = _mm256_setzero_ps();
            for (int c = 0; c < src_depth_quad; ++c) {
                auto src0 = source + c * srcStep + i * pack;
                __m256 vecA = _mm256_loadu_ps(src0);
                __m256 absVecA = _mm256_and_ps(vecA, constant);
                __m256 mask = _mm256_cmp_ps(absVecA, res, 1);
                res = _mm256_blendv_ps(absVecA, res, mask);
            }
            _mm256_storeu_ps(temp, res);
            float absmaxVal = temp[0];
            for (int k = 1; k < pack; ++k) {
                if (absmaxVal < temp[k]) {
                    absmaxVal = temp[k];
                }
            }
            absmax[i] = absmaxVal;
        }
        return;
    }
    if (pack == 4) {
        float tmp[4];
        __m128 mask = _mm_set1_ps(-0.0f);
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
    MNN_ERROR("absmax error: x86_x64 avx2 don't suppport pack=%d yet\n", pack);
    return;
}

static void _AVX_BatchMinMax(float* dstMin, float* dstMax, const float* source, size_t src_depth_quad, size_t realSize, int pack, size_t loadDstBuffer) {
    // input: [src_depth_quad, realSize, pack]
    // max,min shape: [realSize]

    auto srcStep = realSize * pack;
    if (pack == 8) {
        float tempMax[8];
        float tempMin[8];
        for (int i = 0; i < realSize; ++i) {
            __m256 min_ = _mm256_loadu_ps(source + i * pack);
            __m256 max_ = min_;
            for (int c = 1; c < src_depth_quad; ++c) {
                auto src0 = source + c * srcStep + i * pack;
                __m256 vecA = _mm256_loadu_ps(src0);
                max_ = _mm256_max_ps(max_, vecA);
                min_ = _mm256_min_ps(min_, vecA);
            }
            _mm256_storeu_ps(tempMax, max_);
            _mm256_storeu_ps(tempMin, min_);
            float max0 = tempMax[0];
            float min0 = tempMin[0];
            for (int k = 1; k < pack; ++k) {
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
    if (pack == 4) {
        float tempMax[4];
        float tempMin[4];
        for (int i = 0; i < realSize; ++i) {
            auto min_ = _mm_loadu_ps(source + i * pack);
            auto max_ = min_;
            for (int c = 1; c < src_depth_quad; ++c) {
                auto src0 = source + c * srcStep + i * pack;
                auto vecA = _mm_loadu_ps(src0);
                max_ = _mm_max_ps(max_, vecA);
                min_ = _mm_min_ps(min_, vecA);
            }
            _mm_storeu_ps(tempMax, max_);
            _mm_storeu_ps(tempMin, min_);
            float max0 = tempMax[0];
            float min0 = tempMin[0];
            for (int k = 1; k < pack; ++k) {
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
    MNN_ERROR("batch minmax error: x86_x64 avx2 don't suppport pack=%d yet\n", pack);
    return;
}

void _AVX_MNNAsyQuantInfo(float* scale, float* bias, float* qscale, float* qbias, float* dstMin, float* dstMax, const float* src, const size_t* info) {
    auto blockNum = info[0];
    auto plane = info[1];        // real area for data
    auto innerSide = info[2];    // Innermost data layout, may come from backend's pack or gemmint8 units' SRC_UNIT
    auto DST_XUNIT = info[3];    // AVX2: DST_XUNIT=4
    auto kernelsize = info[5];
    auto blockLU = info[6];
    auto stride0 = blockNum * blockLU * plane * innerSide;
    auto stride1 = blockLU * plane * innerSide;

    if (info[7] == 1) { // scale&bias:[1]
        float maxval, minval;
        _AVX_MNNCountMinMaxValue(src, &minval, &maxval, kernelsize * stride0);
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
            qbias[0] = -minval * 255.f / range- 128.f;
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
            _AVX_BatchMinMax(dstMin + j * plane, dstMax + j * plane, src + i * stride0 + j * stride1, blockLU, plane, innerSide, i);
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
            quantBias4 = _mm_blendv_ps(quantBias4, _0f, mask);
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
void _AVX_MNNDynamicQuant(const float* src, int8_t* dst, const float* scale, size_t src_depth_quad, size_t realSize, int pack, const float* bias) {
    auto srcStep = realSize * pack;
    if (pack == 8) { // core->pack
        auto offset = _mm256_set1_epi32(128);
        int32_t* dstPtr = reinterpret_cast<int32_t*>(dst);
        int32_t tmp[8];
        for (int i = 0; i < src_depth_quad; ++i) {
            int xcount = realSize;
            auto srcPtr = src + i * srcStep;
            auto scalePtr = scale;
            auto biasPtr = bias;
            while (xcount > 3) {
                auto scale0 = _mm256_set1_ps(scalePtr[0]);
                auto scale1 = _mm256_set1_ps(scalePtr[1]);
                auto scale2 = _mm256_set1_ps(scalePtr[2]);
                auto scale3 = _mm256_set1_ps(scalePtr[3]);
                auto data0 = _mm256_loadu_ps(srcPtr);
                auto data1 = _mm256_loadu_ps(srcPtr + pack);
                auto data2 = _mm256_loadu_ps(srcPtr + 2 * pack);
                auto data3 = _mm256_loadu_ps(srcPtr + 3 * pack);
                data0 = _mm256_mul_ps(data0, scale0);
                data1 = _mm256_mul_ps(data1, scale1);
                data2 = _mm256_mul_ps(data2, scale2);
                data3 = _mm256_mul_ps(data3, scale3);
                if (bias) {
                    auto bias0 = _mm256_set1_ps(biasPtr[0]);
                    auto bias1 = _mm256_set1_ps(biasPtr[1]);
                    auto bias2 = _mm256_set1_ps(biasPtr[2]);
                    auto bias3 = _mm256_set1_ps(biasPtr[3]);
                    data0 = _mm256_add_ps(data0, bias0);
                    data1 = _mm256_add_ps(data1, bias1);
                    data2 = _mm256_add_ps(data2, bias2);
                    data3 = _mm256_add_ps(data3, bias3);
                }
                data0 = _mm256_round_ps(data0, 0);
                data1 = _mm256_round_ps(data1, 0);
                data2 = _mm256_round_ps(data2, 0);
                data3 = _mm256_round_ps(data3, 0);
                auto r0 = _mm256_cvtps_epi32(data0);
                auto r1 = _mm256_cvtps_epi32(data1);
                auto r2 = _mm256_cvtps_epi32(data2);
                auto r3 = _mm256_cvtps_epi32(data3);
                r0 = _mm256_add_epi32(r0, offset);
                r1 = _mm256_add_epi32(r1, offset);
                r2 = _mm256_add_epi32(r2, offset);
                r3 = _mm256_add_epi32(r3, offset);
                auto r0_16 = _mm256_packs_epi32(r0, r1); // 0000111100001111
                auto r1_16 = _mm256_packs_epi32(r2, r3); // 2222333322223333
                auto r0_8 = _mm256_packus_epi16(r0_16, r1_16); // 0000111122223333 0000111122223333
                _mm256_storeu_si256((__m256i *)tmp, r0_8);
                for (int k = 0; k < 4; ++k) {
                    dstPtr[2 * k] = tmp[k];
                    dstPtr[2 * k + 1] = tmp[k + 4];
                }
                // next round
                xcount -= 4;
                scalePtr += 4;
                if (bias) {
                    biasPtr += 4;
                }
                srcPtr += (4 * pack);
                dstPtr += 8;
            }
            while (xcount) {
                auto scale0 = _mm256_set1_ps(scalePtr[0]);
                auto data0 = _mm256_loadu_ps(srcPtr);
                data0 = _mm256_mul_ps(data0, scale0);
                if (bias) {
                    auto bias0 = _mm256_set1_ps(biasPtr[0]);
                    data0 = _mm256_add_ps(data0, bias0);
                }
                data0 = _mm256_round_ps(data0, 0);
                auto r0 = _mm256_cvtps_epi32(data0);
                r0 = _mm256_add_epi32(r0, offset);
                auto r0_16 = _mm256_packs_epi32(r0, r0); // 0000111100001111
                auto r0_8 = _mm256_packus_epi16(r0_16, r0_16); // 0000111122223333 0000111122223333
                _mm256_storeu_si256((__m256i *)tmp, r0_8);
                dstPtr[0] = tmp[0];
                dstPtr[1] = tmp[4];

                // next round
                xcount--;
                scalePtr += 1;
                if (bias) {
                    biasPtr += 1;
                }
                srcPtr += pack;
                dstPtr += 2;
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
    MNN_ERROR("dynamic quant error: x86_x64 avx2 don't suppport pack=%d yet\n", pack);
    return;
}

void _AVX_MNNAsyQuantFunc(int8_t* dst, const float* src, float* qscale, float* qbias, const size_t* info) {
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
            _AVX_MNNDynamicQuant(src + k * stride0 + i * stride1, dst + k * stride0 + i * stride1, qscale + i * EP, blockLU, EP, LP, qbias + i * EP);
        }
    }
}

#endif // MNN_LOW_MEMORY

void _AVX_MNNComputeMatMulForE_1(const float* A, const float* B, float* C, const float* biasPtr, const MatMulParam* param, size_t tId) {
    auto l = param->l;
    auto h = param->h;
    auto numberThread = param->numberThread;
    auto lC4 = l / 8;
    auto lR = lC4 * 8;
    if (param->BTranspose) {
        for (int y=tId; y<h; y+=numberThread) {
            auto sumValue = _mm256_set1_ps(0.0f);
            auto by = B + y * l;
            for (int x=0; x<lC4; ++x) {
                sumValue = _mm256_add_ps(sumValue, _mm256_mul_ps(_mm256_loadu_ps(A + x * 8), _mm256_loadu_ps(by + x * 8)));
            }
            float sumRemain = 0.0f;
            for (int x=lR; x<l; ++x) {
                sumRemain = sumRemain + A[x] * by[x];
            }
            if (nullptr != biasPtr) {
                sumRemain += biasPtr[y];
            }
            sumValue = _mm256_hadd_ps(sumValue, sumValue);
            sumValue = _mm256_hadd_ps(sumValue, sumValue);
            auto s = _mm_cvtss_f32(_mm256_extractf128_ps(sumValue, 0)) + _mm_cvtss_f32(_mm256_extractf128_ps(sumValue, 1));
            C[y] = sumRemain + s;
        }
    } else {
        auto hC4 = h / 8;
        auto hR = hC4 * 8;
        for (int y=tId; y<hC4; y+=numberThread) {
            auto bs = B + 8 * y;
            auto sumValue = _mm256_set1_ps(0.0f);
            if (biasPtr != nullptr) {
                sumValue = _mm256_loadu_ps(biasPtr + 8 * y);
            }
            auto srcY = A + y * l;
            for (int x=0; x<l; ++x) {
                sumValue = _mm256_add_ps(sumValue, _mm256_mul_ps(_mm256_broadcast_ss(A + x), _mm256_loadu_ps(bs + h * x)));
            }
            _mm256_storeu_ps(C + 8 * y, sumValue);
        }
        for (int y = hR + tId; y<h; y+=numberThread) {
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

void _AVX_MNNComputeMatMulForH_1(const float* A, const float* B, float* C, const float* biasPtr, const MatMulParam* param, size_t tId) {
    int e = param->e;
    int l = param->l;
    int numberThread = param->numberThread;
    const int unit = 8;
    float biasVUnit = 0.0f;
    __m256 biasValue = _mm256_setzero_ps();
    if (nullptr != biasPtr) {
        biasValue = _mm256_broadcast_ss(biasPtr);
        biasVUnit = biasPtr[0];
    }
    if (param->ATranspose) {
        auto eC4 = e / unit;
        auto eR = eC4 * unit;
        for (int y=tId; y<eC4; y+=numberThread) {
            auto sumValue = biasValue;
            auto srcY = A + y * unit;
            for (int x=0; x<l; ++x) {
                sumValue = _mm256_add_ps(sumValue, _mm256_mul_ps(_mm256_loadu_ps(srcY + x * e), _mm256_broadcast_ss(B + x)));
            }
            _mm256_storeu_ps(C + unit * y, sumValue);
        }
        if (0 == tId) {
            for (int y=eR; y<e; ++y) {
                float sumValue = biasVUnit;
                auto srcY = A + y;
                for (int x=0; x<l; ++x) {
                    sumValue = sumValue + srcY[x * e] * B[x];
                }
                C[y] = sumValue;
            }
        }
        return;
    }
    auto lC4 = l / unit;
    auto lR = lC4 * unit;
    int eU = e / unit;
    int eR = e % unit;
    for (int y=tId; y<eU; y+=numberThread) {
        auto D0 = _mm256_setzero_ps();
        auto D1 = _mm256_setzero_ps();
        auto D2 = _mm256_setzero_ps();
        auto D3 = _mm256_setzero_ps();
        auto D4 = _mm256_setzero_ps();
        auto D5 = _mm256_setzero_ps();
        auto D6 = _mm256_setzero_ps();
        auto D7 = _mm256_setzero_ps();

        auto s0 = A + l * (y * unit + 0);
        auto s1 = A + l * (y * unit + 1);
        auto s2 = A + l * (y * unit + 2);
        auto s3 = A + l * (y * unit + 3);
        auto s4 = A + l * (y * unit + 4);
        auto s5 = A + l * (y * unit + 5);
        auto s6 = A + l * (y * unit + 6);
        auto s7 = A + l * (y * unit + 7);
        for (int x=0; x<lC4; ++x) {
            auto B0 = _mm256_loadu_ps(B + unit * x);
            auto A0 = _mm256_loadu_ps(s0);
            auto A1 = _mm256_loadu_ps(s1);
            auto A2 = _mm256_loadu_ps(s2);
            auto A3 = _mm256_loadu_ps(s3);
            auto A4 = _mm256_loadu_ps(s4);
            auto A5 = _mm256_loadu_ps(s5);
            auto A6 = _mm256_loadu_ps(s6);
            auto A7 = _mm256_loadu_ps(s7);
#define COMPUTE_TEMP(i) D##i = _mm256_add_ps(D##i, _mm256_mul_ps(A##i, B0))
            COMPUTE_TEMP(0);
            COMPUTE_TEMP(1);
            COMPUTE_TEMP(2);
            COMPUTE_TEMP(3);
            COMPUTE_TEMP(4);
            COMPUTE_TEMP(5);
            COMPUTE_TEMP(6);
            COMPUTE_TEMP(7);
            s0 += unit;
            s1 += unit;
            s2 += unit;
            s3 += unit;
            s4 += unit;
            s5 += unit;
            s6 += unit;
            s7 += unit;
        }
        if (lR < l) {
            int remain = l - lR;
            float tempB[8] = {0.0f};
            float tempA[8] = {0.0f};
            ::memcpy(tempB, B + unit * lC4, remain * sizeof(float));
            auto B0 = _mm256_loadu_ps(tempB);
            ::memcpy(tempA, s0, remain * sizeof(float));
            auto A0 = _mm256_loadu_ps(tempA);
            ::memcpy(tempA, s1, remain * sizeof(float));
            auto A1 = _mm256_loadu_ps(tempA);
            ::memcpy(tempA, s2, remain * sizeof(float));
            auto A2 = _mm256_loadu_ps(tempA);
            ::memcpy(tempA, s3, remain * sizeof(float));
            auto A3 = _mm256_loadu_ps(tempA);
            ::memcpy(tempA, s4, remain * sizeof(float));
            auto A4 = _mm256_loadu_ps(tempA);
            ::memcpy(tempA, s5, remain * sizeof(float));
            auto A5 = _mm256_loadu_ps(tempA);
            ::memcpy(tempA, s6, remain * sizeof(float));
            auto A6 = _mm256_loadu_ps(tempA);
            ::memcpy(tempA, s7, remain * sizeof(float));
            auto A7 = _mm256_loadu_ps(tempA);
            COMPUTE_TEMP(0);
            COMPUTE_TEMP(1);
            COMPUTE_TEMP(2);
            COMPUTE_TEMP(3);
            COMPUTE_TEMP(4);
            COMPUTE_TEMP(5);
            COMPUTE_TEMP(6);
            COMPUTE_TEMP(7);
        }
#undef COMPUTE_TEMP

        D0 = _mm256_hadd_ps(D0, D1);
        D2 = _mm256_hadd_ps(D2, D3);
        D4 = _mm256_hadd_ps(D4, D5);
        D6 = _mm256_hadd_ps(D6, D7);

        D0 = _mm256_hadd_ps(D0, D2);
        D4 = _mm256_hadd_ps(D4, D6);

        auto r0 = _mm_add_ps(_mm256_extractf128_ps(D0, 0), _mm256_extractf128_ps(D0, 1));
        auto r1 = _mm_add_ps(_mm256_extractf128_ps(D4, 0), _mm256_extractf128_ps(D4, 1));
        _mm_storeu_ps(C + y * unit + 0, r0);
        _mm_storeu_ps(C + y * unit + 4, r1);
    }
    for (int y=tId + eU * unit; y<e; y+=numberThread) {
        auto sumValue = _mm256_setzero_ps();
        auto srcY = A + y * l;
        for (int x=0; x<lC4; ++x) {
            sumValue = _mm256_add_ps(sumValue, _mm256_mul_ps(_mm256_loadu_ps(srcY + unit * x), _mm256_loadu_ps(B + unit * x)));
        }
        float temp[8];
        _mm256_storeu_ps(temp, sumValue);
        float sumSingle = biasVUnit;
        for (int i=0; i<8; ++i) {
            sumSingle += temp[i];
        }
        for (int x=lR; x<l; ++x) {
            sumSingle += srcY[x] * B[x];
        }
        C[y] = sumSingle;
    }
}

