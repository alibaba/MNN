//
//  GemmSSE.cpp
//  MNN
//
//  Created by MNN on 2020/09/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "FunctionSummary.hpp"
#include "GemmCommon.hpp"
#include "core/Macro.h"
#define MNNSSEFMA(x, y, z) _mm_add_ps(_mm_mul_ps(x, y), z)
#include "GemmFunction.hpp"

void _SSE_MNNPackedMatMul(float* C, const float* A, const float* B, const size_t* parameter,
                          const float* postParameters, const float* bias, const float* k, const float* b) {
    auto h       = parameter[2];
    auto hC4     = UP_DIV(h, 4);
    auto cStride = parameter[3] / sizeof(float);
    _SSE_MNNPackedMatMul_12(C, A, B, parameter);
    _SSE_GemmPostTreat(C, 12, parameter, postParameters, bias);
}

void _SSE_MNNPackedMatMulRemain(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter,
                                const float* postParameters, const float* bias, const float* k, const float* b) {
    _SSE_MNNPackednMatMulRemainCommon(C, A, B, eSize, parameter, postParameters, bias);
    _SSE_GemmPostTreat(C, eSize, parameter, postParameters, bias);
}

#ifdef MNN_CPU_WEIGHT_DEQUANT_GEMM
//----------------------- MatMul(float, int4) Functions ---------------------------//
void _SSE_MNNPackedMatMul_int4(float* C, const float* A, const float* B, const size_t* parameter,
                               const float* postParameters, const float* bias, const float* k, const float* b) {
    auto h       = parameter[2];
    auto hC4     = UP_DIV(h, 4);
    auto cStride = parameter[3] / sizeof(float);
    _SSE_MNNPackedMatMul_12_int4(C, A, B, parameter, k, b);
    if (nullptr != bias) {
        _SSE_GemmPostTreat(C, 12, parameter, postParameters, bias);
    }
}

void _SSE_MNNPackedMatMulRemain_int4(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter,
                                     const float* postParameters, const float* bias, const float* k, const float* b) {
    _SSE_MNNPackednMatMulRemainCommon_int4(C, A, B, eSize, parameter, postParameters, bias, k, b);
    if (nullptr != bias) {
        _SSE_GemmPostTreat(C, eSize, parameter, postParameters, bias);
    }
}

void _SSE_MNNPackedMatMul_int8(float* C, const float* A, const float* B, const size_t* parameter,
                               const float* postParameters, const float* bias, const float* k, const float* b) {
    auto h       = parameter[2];
    auto hC4     = UP_DIV(h, 4);
    auto cStride = parameter[3] / sizeof(float);
    _SSE_MNNPackedMatMul_12_int8(C, A, B, parameter, k, b);
    if (nullptr != bias) {
        _SSE_GemmPostTreat(C, 12, parameter, postParameters, bias);
    }
}

void _SSE_MNNPackedMatMulRemain_int8(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter,
                                     const float* postParameters, const float* bias, const float* k, const float* b) {
    _SSE_MNNPackednMatMulRemainCommon_int8(C, A, B, eSize, parameter, postParameters, bias, k, b);
    if (nullptr != bias) {
        _SSE_GemmPostTreat(C, eSize, parameter, postParameters, bias);
    }
}
#endif

#ifdef MNN_LOW_MEMORY
// Dynamic quant
void _SSE_MNNAbsMaxFP32(const float* source, float* absmax, size_t src_depth_quad, size_t realSize, int pack) {
    size_t srcStep = realSize * pack;
    __m128 mask = _mm_set1_ps(-0.0f);
    if (pack == 4) { // input c4
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
    if (pack == 16) { // (lu,ep,lp)
        float tmp[16];
        for (int i = 0; i < realSize; ++i) {
            __m128 absmax0 = _mm_loadu_ps(source + i * pack);
            __m128 absmax1 = _mm_loadu_ps(source + i * pack + 4);
            __m128 absmax2 = _mm_loadu_ps(source + i * pack + 8);
            __m128 absmax3 = _mm_loadu_ps(source + i * pack + 12);
            absmax0 = _mm_andnot_ps(mask, absmax0);
            absmax1 = _mm_andnot_ps(mask, absmax1);
            absmax2 = _mm_andnot_ps(mask, absmax2);
            absmax3 = _mm_andnot_ps(mask, absmax3);
            auto src0 = source + i * pack;
            for (int j = 1; j < src_depth_quad; ++j) {
                __m128 vec0 = _mm_loadu_ps(src0 + j * srcStep);
                __m128 vec1 = _mm_loadu_ps(src0 + j * srcStep + 4);
                __m128 vec2 = _mm_loadu_ps(src0 + j * srcStep + 8);
                __m128 vec3 = _mm_loadu_ps(src0 + j * srcStep + 12);
                vec0 = _mm_andnot_ps(mask, vec0);
                vec1 = _mm_andnot_ps(mask, vec1);
                vec2 = _mm_andnot_ps(mask, vec2);
                vec3 = _mm_andnot_ps(mask, vec3);
                absmax0 = _mm_max_ps(absmax0, vec0);
                absmax1 = _mm_max_ps(absmax1, vec1);
                absmax2 = _mm_max_ps(absmax2, vec2);
                absmax3 = _mm_max_ps(absmax3, vec3);
            }
            absmax0 = _mm_max_ps(absmax0, absmax1);
            absmax2 = _mm_max_ps(absmax2, absmax3);
            absmax0 = _mm_max_ps(absmax0, absmax2);
            _mm_storeu_ps(tmp, absmax0);
            float res = tmp[0];
            for (int j = 1; j < 4; ++j) {
                res = ALIMAX(res, tmp[j]);
            }
            absmax[i] = res;
        }
        return;
    }
    MNN_ERROR("absMax error: x86_x64 sse don't suppport pack=%d yet\n", pack);
    return;
}

void _SSE_MNNDynamicQuant(const float* src, int8_t* dst, const float* scale, size_t src_depth_quad, size_t realSize, int pack, const float* bias) {
    auto srcStep = realSize * pack;
    if (pack == 4) { // core->pack
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
    if (pack == 16) {
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
                auto data00 = _mm_loadu_ps(srcPtr);
                auto data01 = _mm_loadu_ps(srcPtr + 4);
                auto data02 = _mm_loadu_ps(srcPtr + 8);
                auto data03 = _mm_loadu_ps(srcPtr + 12);

                auto data10 = _mm_loadu_ps(srcPtr + pack);
                auto data11 = _mm_loadu_ps(srcPtr + pack + 4);
                auto data12 = _mm_loadu_ps(srcPtr + pack + 8);
                auto data13 = _mm_loadu_ps(srcPtr + pack + 12);

                auto data20 = _mm_loadu_ps(srcPtr + 2 * pack);
                auto data21 = _mm_loadu_ps(srcPtr + 2 * pack + 4);
                auto data22 = _mm_loadu_ps(srcPtr + 2 * pack + 8);
                auto data23 = _mm_loadu_ps(srcPtr + 2 * pack + 12);

                auto data30 = _mm_loadu_ps(srcPtr + 3 * pack);
                auto data31 = _mm_loadu_ps(srcPtr + 3 * pack + 4);
                auto data32 = _mm_loadu_ps(srcPtr + 3 * pack + 8);
                auto data33 = _mm_loadu_ps(srcPtr + 3 * pack + 12);

                data00 = _mm_mul_ps(data00, scale0);
                data01 = _mm_mul_ps(data01, scale0);
                data02 = _mm_mul_ps(data02, scale0);
                data03 = _mm_mul_ps(data03, scale0);
                data10 = _mm_mul_ps(data10, scale1);
                data11 = _mm_mul_ps(data11, scale1);
                data12 = _mm_mul_ps(data12, scale1);
                data13 = _mm_mul_ps(data13, scale1);
                data20 = _mm_mul_ps(data20, scale2);
                data21 = _mm_mul_ps(data21, scale2);
                data22 = _mm_mul_ps(data22, scale2);
                data23 = _mm_mul_ps(data23, scale2);
                data30 = _mm_mul_ps(data30, scale3);
                data31 = _mm_mul_ps(data31, scale3);
                data32 = _mm_mul_ps(data32, scale3);
                data33 = _mm_mul_ps(data33, scale3);
                if (bias) {
                    auto bias0 = _mm_set1_ps(biasPtr[0]);
                    auto bias1 = _mm_set1_ps(biasPtr[1]);
                    auto bias2 = _mm_set1_ps(biasPtr[2]);
                    auto bias3 = _mm_set1_ps(biasPtr[3]);
                    data00 = _mm_add_ps(data00, bias0);
                    data01 = _mm_add_ps(data01, bias0);
                    data02 = _mm_add_ps(data02, bias0);
                    data03 = _mm_add_ps(data03, bias0);
                    data10 = _mm_add_ps(data10, bias1);
                    data11 = _mm_add_ps(data11, bias1);
                    data12 = _mm_add_ps(data12, bias1);
                    data13 = _mm_add_ps(data13, bias1);
                    data20 = _mm_add_ps(data20, bias2);
                    data21 = _mm_add_ps(data21, bias2);
                    data22 = _mm_add_ps(data22, bias2);
                    data23 = _mm_add_ps(data23, bias2);
                    data30 = _mm_add_ps(data30, bias3);
                    data31 = _mm_add_ps(data31, bias3);
                    data32 = _mm_add_ps(data32, bias3);
                    data33 = _mm_add_ps(data33, bias3);
                }
                data00 = _mm_round_ps(data00, 0);
                data01 = _mm_round_ps(data01, 0);
                data02 = _mm_round_ps(data02, 0);
                data03 = _mm_round_ps(data03, 0);
                data10 = _mm_round_ps(data10, 0);
                data11 = _mm_round_ps(data11, 0);
                data12 = _mm_round_ps(data12, 0);
                data13 = _mm_round_ps(data13, 0);
                data20 = _mm_round_ps(data20, 0);
                data21 = _mm_round_ps(data21, 0);
                data22 = _mm_round_ps(data22, 0);
                data23 = _mm_round_ps(data23, 0);
                data30 = _mm_round_ps(data30, 0);
                data31 = _mm_round_ps(data31, 0);
                data32 = _mm_round_ps(data32, 0);
                data33 = _mm_round_ps(data33, 0);
                auto r00 = _mm_cvtps_epi32(data00);
                auto r01 = _mm_cvtps_epi32(data01);
                auto r02 = _mm_cvtps_epi32(data02);
                auto r03 = _mm_cvtps_epi32(data03);
                auto r10 = _mm_cvtps_epi32(data10);
                auto r11 = _mm_cvtps_epi32(data11);
                auto r12 = _mm_cvtps_epi32(data12);
                auto r13 = _mm_cvtps_epi32(data13);
                auto r20 = _mm_cvtps_epi32(data20);
                auto r21 = _mm_cvtps_epi32(data21);
                auto r22 = _mm_cvtps_epi32(data22);
                auto r23 = _mm_cvtps_epi32(data23);
                auto r30 = _mm_cvtps_epi32(data30);
                auto r31 = _mm_cvtps_epi32(data31);
                auto r32 = _mm_cvtps_epi32(data32);
                auto r33 = _mm_cvtps_epi32(data33);
                r00 = _mm_add_epi32(r00, offset);
                r01 = _mm_add_epi32(r01, offset);
                r02 = _mm_add_epi32(r02, offset);
                r03 = _mm_add_epi32(r03, offset);

                r10 = _mm_add_epi32(r10, offset);
                r11 = _mm_add_epi32(r11, offset);
                r12 = _mm_add_epi32(r12, offset);
                r13 = _mm_add_epi32(r13, offset);

                r20 = _mm_add_epi32(r20, offset);
                r21 = _mm_add_epi32(r21, offset);
                r22 = _mm_add_epi32(r22, offset);
                r23 = _mm_add_epi32(r23, offset);

                r30 = _mm_add_epi32(r30, offset);
                r31 = _mm_add_epi32(r31, offset);
                r32 = _mm_add_epi32(r32, offset);
                r33 = _mm_add_epi32(r33, offset);
                auto r00_16 = _mm_packs_epi32(r00, r01); // 00000000
                auto r01_16 = _mm_packs_epi32(r02, r03); // 00000000
                auto r0_8  = _mm_packus_epi16(r00_16, r01_16); // 0000000000000000

                auto r10_16 = _mm_packs_epi32(r10, r11);
                auto r11_16 = _mm_packs_epi32(r12, r13);
                auto r1_8  = _mm_packus_epi16(r10_16, r11_16);

                auto r20_16 = _mm_packs_epi32(r20, r21);
                auto r21_16 = _mm_packs_epi32(r22, r23);
                auto r2_8  = _mm_packus_epi16(r20_16, r21_16);

                auto r30_16 = _mm_packs_epi32(r30, r31);
                auto r31_16 = _mm_packs_epi32(r32, r33);
                auto r3_8  = _mm_packus_epi16(r30_16, r31_16);
                _mm_storeu_si128((__m128i *)dstPtr, r0_8);
                _mm_storeu_si128((__m128i *)(dstPtr + 4), r1_8);
                _mm_storeu_si128((__m128i *)(dstPtr + 8), r2_8);
                _mm_storeu_si128((__m128i *)(dstPtr + 12), r3_8);
                // next round
                xcount -= 4;
                scalePtr += 4;
                if (bias) {
                    biasPtr += 4;
                }
                srcPtr += (4 * pack);
                dstPtr += pack;
            }
            while (xcount) {
                auto scale0 = _mm_set1_ps(scalePtr[0]);
                auto data00 = _mm_loadu_ps(srcPtr);
                auto data01 = _mm_loadu_ps(srcPtr + 4);
                auto data02 = _mm_loadu_ps(srcPtr + 8);
                auto data03 = _mm_loadu_ps(srcPtr + 12);

                data00 = _mm_mul_ps(data00, scale0);
                data01 = _mm_mul_ps(data01, scale0);
                data02 = _mm_mul_ps(data02, scale0);
                data03 = _mm_mul_ps(data03, scale0);
                if (bias) {
                    auto bias0 = _mm_set1_ps(biasPtr[0]);
                    data00 = _mm_add_ps(data00, bias0);
                    data01 = _mm_add_ps(data01, bias0);
                    data02 = _mm_add_ps(data02, bias0);
                    data03 = _mm_add_ps(data03, bias0);
                }

                data00 = _mm_round_ps(data00, 0);
                data01 = _mm_round_ps(data01, 0);
                data02 = _mm_round_ps(data02, 0);
                data03 = _mm_round_ps(data03, 0);
                auto r00 = _mm_cvtps_epi32(data00);
                auto r01 = _mm_cvtps_epi32(data01);
                auto r02 = _mm_cvtps_epi32(data02);
                auto r03 = _mm_cvtps_epi32(data03);
                r00 = _mm_add_epi32(r00, offset);
                r01 = _mm_add_epi32(r01, offset);
                r02 = _mm_add_epi32(r02, offset);
                r03 = _mm_add_epi32(r03, offset);
                auto r00_16 = _mm_packs_epi32(r00, r01); // 00000000
                auto r01_16 = _mm_packs_epi32(r02, r03); // 00000000
                auto r0_8  = _mm_packus_epi16(r00_16, r01_16); // 0000000000000000

                _mm_storeu_si128((__m128i *)dstPtr, r0_8);
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
    MNN_ERROR("dynamic quant error: x86_x64 sse don't suppport pack=%d yet\n", pack);
    return;
}

static void _SSE_BatchMinMax(float* dstMin, float* dstMax, const float* source, size_t src_depth_quad, size_t realSize, int innerSide, size_t loadDstBuffer) {
    // input: [src_depth_quad, realSize, pack]
    // max,min shape: [realSize]
    // SSE: core->pack=4, LP=16
    auto srcStep = realSize * innerSide;
    if (innerSide == 16) {
        float tempMax[16];
        float tempMin[16];
        for (int i = 0; i < realSize; ++i) {
            auto min0_ = _mm_loadu_ps(source + i * innerSide);
            auto min1_ = _mm_loadu_ps(source + i * innerSide + 4);
            auto min2_ = _mm_loadu_ps(source + i * innerSide + 8);
            auto min3_ = _mm_loadu_ps(source + i * innerSide + 12);
            auto max0_ = min0_;
            auto max1_ = min1_;
            auto max2_ = min2_;
            auto max3_ = min3_;

            for (int c = 1; c < src_depth_quad; ++c) {
                auto src0 = source + c * srcStep + i * innerSide;
                auto vecA0 = _mm_loadu_ps(src0);
                auto vecA1 = _mm_loadu_ps(src0 + 4);
                auto vecA2 = _mm_loadu_ps(src0 + 8);
                auto vecA3 = _mm_loadu_ps(src0 + 12);
                max0_ = _mm_max_ps(max0_, vecA0);
                min0_ = _mm_min_ps(min0_, vecA0);
                max1_ = _mm_max_ps(max1_, vecA1);
                min1_ = _mm_min_ps(min1_, vecA1);
                max2_ = _mm_max_ps(max2_, vecA2);
                min2_ = _mm_min_ps(min2_, vecA2);
                max3_ = _mm_max_ps(max3_, vecA3);
                min3_ = _mm_min_ps(min3_, vecA3);
            }
            _mm_storeu_ps(tempMax, max0_);
            _mm_storeu_ps(tempMin, min0_);
            _mm_storeu_ps(tempMax + 4, max1_);
            _mm_storeu_ps(tempMin + 4, min1_);
            _mm_storeu_ps(tempMax + 8, max2_);
            _mm_storeu_ps(tempMin + 8, min2_);
            _mm_storeu_ps(tempMax + 12, max3_);
            _mm_storeu_ps(tempMin + 12, min3_);
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
    MNN_ERROR("batch minmax error: x86_x64 avx2 don't suppport pack=%d yet\n", innerSide);
    return;
}
void _SSE_MNNAsyQuantInfo(float* scale, float* bias, float* qscale, float* qbias, float* dstMin, float* dstMax, const float* src, const size_t* info) {
    auto blockNum = info[0];
    auto plane = info[1];        // real area for data
    auto innerSide = info[2];    // Innermost data layout, may come from backend's pack or gemmint8 units' SRC_UNIT
    auto DST_XUNIT = info[3];    // SSE: DST_XUNIT=4
    auto kernelsize = info[5];
    auto blockLU = info[6];
    auto stride0 = blockNum * blockLU * plane * innerSide;
    auto stride1 = blockLU * plane * innerSide;

    if (info[7] == 1) { // scale&bias:[1]
        float maxval, minval;
        _SSE_MNNCountMinMaxValue(src, &minval, &maxval, kernelsize * stride0);
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
            _SSE_BatchMinMax(dstMin + j * plane, dstMax + j * plane, src + i * stride0 + j * stride1, blockLU, plane, innerSide, i);
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
void _SSE_MNNAsyQuantFunc(int8_t* dst, const float* src, float* qscale, float* qbias, const size_t* info) {
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
            _SSE_MNNDynamicQuant(src + k * stride0 + i * stride1, dst + k * stride0 + i * stride1, qscale + i * EP, blockLU, EP, LP, qbias + i * EP);
        }
    }
}
#endif
