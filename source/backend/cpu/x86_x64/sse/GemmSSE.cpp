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

#ifdef MNN_LOW_MEMORY
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

void _SSE_MNNGemmHybridInt4(float* C, const int8_t* A, const int8_t* B, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad, size_t realSize, const float** param) {
    _SSE_MNNGemmHybrid_int4(C, A,  B, src_depth_quad, dst_step, dst_depth_quad, realSize, param);
}
void _SSE_MNNGemmHybridInt8(float* C, const int8_t* A, const int8_t* B, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad, size_t realSize, const float** param) {
    _SSE_MNNGemmHybrid_int8(C, A,  B, src_depth_quad, dst_step, dst_depth_quad, realSize, param);
}
// Dynamic quant
void _SSE_MNNAbsMaxFP32(const float* source, float* absmax, size_t src_depth_quad, size_t realSize, int pack) {
    // source: (ic/4, N, 4)
    auto srcStep = pack * realSize;
    auto constant = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF));
    float temp[4];
    for (int i = 0; i < realSize; ++i) {
        __m128 res = _mm_setzero_ps();
        for (int c = 0; c < src_depth_quad; ++c) {
            auto src0 = source + c * srcStep + i * pack;
            __m128 vecA = _mm_loadu_ps(src0);
            __m128 absVecA = _mm_and_ps(vecA, constant);
            __m128 mask = _mm_cmpgt_ps(res, absVecA);
            res = _mm_blendv_ps(absVecA, res, mask);
            
        }
        _mm_storeu_ps(temp, res);
        float absmaxVal = temp[0];
        for (int k = 1; k < pack; ++k) {
            if (absmaxVal < temp[k]) {
                absmaxVal = temp[k];
            }
        }
        absmax[i] = absmaxVal;
    }
}

void _SSE_MNNDynamicQuantFP32(const float* src, int8_t* dst, const float* scale, float* sum, size_t src_depth_quad, size_t realSize, int pack) {
    // SSE: pack=4
    __m128 zero = _mm_setzero_ps();
    __m128 plus = _mm_set1_ps(0.5f);
    __m128 minus = _mm_set1_ps(-0.5f);
    auto offset = _mm_set1_epi32(128);
    uint8_t* dstPtr = reinterpret_cast<uint8_t*>(dst);
    float temp[4];
    for (int i = 0; i < realSize; ++i) {
        __m128 scaleVal = _mm_load_ps1(scale + i);
        __m128 acc = _mm_setzero_ps();
        for (int c = 0; c < src_depth_quad; ++c) {
            auto srcZ = src + c * pack * realSize + i * pack;
            auto dstZ = dstPtr + c * pack * realSize + i * pack;
            __m128 f0 = _mm_loadu_ps(srcZ);
            __m128 m0 = _mm_mul_ps(f0, scaleVal);
            __m128 mask = _mm_cmplt_ps(m0, zero);
            __m128 d0 = _mm_blendv_ps(plus, minus, mask);
            d0 = _mm_add_ps(d0, m0);
            __m128 round0 = _mm_round_ps(d0, 3);
            auto d0_epi32 = _mm_cvtps_epi32(round0);
            d0_epi32 = _mm_packs_epi32(d0_epi32, d0_epi32);
            d0_epi32 = _mm_packs_epi16(d0_epi32, d0_epi32);
            *((int*)dstZ) = _mm_cvtsi128_si32(d0_epi32);
            acc = _mm_add_ps(acc, round0);
        }
        _mm_storeu_ps(temp, acc);
        int sumVal = static_cast<int32_t>(temp[0] + temp[1] + temp[2] + temp[3]);
        ((int32_t*)sum)[i] = sumVal;
    }
}
#endif
