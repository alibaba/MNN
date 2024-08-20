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
#endif
