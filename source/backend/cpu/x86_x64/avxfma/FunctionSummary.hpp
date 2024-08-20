//
//  FunctionSummary.hpp
//  MNN
//
//  Created by MNN on 2019/08/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#include <MNN/MNNDefine.h>
#include <stdint.h>

#ifndef _MM_TRANSPOSE4_PS
#define _MM_TRANSPOSE4_PS(row0, row1, row2, row3) \
    do {                                          \
        __m128 tmp3, tmp2, tmp1, tmp0;            \
        tmp0   = _mm_unpacklo_ps((row0), (row1)); \
        tmp2   = _mm_unpacklo_ps((row2), (row3)); \
        tmp1   = _mm_unpackhi_ps((row0), (row1)); \
        tmp3   = _mm_unpackhi_ps((row2), (row3)); \
        (row0) = _mm_movelh_ps(tmp0, tmp2);       \
        (row1) = _mm_movehl_ps(tmp2, tmp0);       \
        (row2) = _mm_movelh_ps(tmp1, tmp3);       \
        (row3) = _mm_movehl_ps(tmp3, tmp1);       \
    } while (0)
#endif
#include "backend/cpu/compute/Int8FunctionsOpt.h"
#include "backend/cpu/compute/CommonOptFunction.h"

// ========= CommonOptFunction.cpp ===========
extern "C" {

void _AVX_MNNPackedMatMulFMA(float* C, const float* A, const float* B, const size_t* parameter,
                             const float* postParameters, const float* bias, const float* k, const float* b);
void _AVX_MNNPackedMatMulRemainFMA(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter, const float* postParameters, const float* bias, const float* k, const float* b);
void _AVX_MNNComputeMatMulForE_1FMA(const float* A, const float* B, float* C, const float* biasPtr, const MatMulParam* param, size_t tId);
void _AVX_MNNPackedMatMulFMA_BF16(float* C, const float* A, const float* B, const size_t* parameter,
                                  const float* postParameters, const float* bias, const float* k, const float* b);
void _AVX_MNNPackedMatMulRemainFMA_BF16(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter, const float* postParameters, const float* bias, const float* k, const float* b);
void _AVX_MNNComputeMatMulForH_1FMA(const float* A, const float* B, float* C, const float* biasPtr, const MatMulParam* param, size_t tId);
void _AVX_MNNGeluFMA(float *dst, const float *src, size_t size, float* parameters);
void _AVX_MNNExpC8FMA(float* dest, const float* source, float* offset, const float* parameters, size_t countC8);
void _AVX_MNNPackedSparseMatMulEpx1NFMA(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter, const float* postParameters, const float* bias, unsigned int* NNZMap, int* dataOffsetMap);
void _AVX_MNNPackedSparseMatMulEpx4NFMA(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter, const float* postParameters, const float* bias, unsigned int* NNZMap, int* dataOffsetMap);

void _AVX_ExtraInitFMA(void* functions);

}
