//
//  FunctionSummary.hpp
//  MNN
//
//  Created by MNN on 2021/01/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#include <stdint.h>
#include "backend/cpu/compute/Int8FunctionsOpt.h"
#include "backend/cpu/compute/CommonOptFunction.h"
#include "DynamicGemm.h"

#ifndef _MM_TRANSPOSE4_PS
#define _MM_TRANSPOSE4_PS(row0, row1, row2, row3) \
do { \
  __m128 tmp3, tmp2, tmp1, tmp0; \
  tmp0 = _mm_unpacklo_ps((row0), (row1)); \
  tmp2 = _mm_unpacklo_ps((row2), (row3)); \
  tmp1 = _mm_unpackhi_ps((row0), (row1)); \
  tmp3 = _mm_unpackhi_ps((row2), (row3)); \
  (row0) = _mm_movelh_ps(tmp0, tmp2); \
  (row1) = _mm_movehl_ps(tmp2, tmp0); \
  (row2) = _mm_movelh_ps(tmp1, tmp3); \
  (row3) = _mm_movehl_ps(tmp3, tmp1); \
} while (0)
#endif

// ========= CommonOptFunction.cpp ===========
extern "C" {
void _AVX512_MNNPackForMatMul_B(float* dest, const float* source, size_t h, size_t l, bool transpose);
void _AVX512_MNNPackC8ForMatMul_A(float* destOrigin, float const** sourceGroup, const int32_t* info, const int32_t* el);

void _AVX512_MNNPackedMatMul(float* C, const float* A, const float* B, const size_t* parameter, const float* postParameters, const float* bias, const float* k, const float* b);
void _AVX512_MNNPackedMatMulRemain(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter, const float* postParameters, const float* bias, const float* k, const float* b);

void _AVX512_MNNGetSparseMatMulPackMode(int* eP, int *lP, int* hP);
void _AVX512_MNNPackedSparseMatMulEpx8(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter,
                                    const float* postParameters, const float* bias, unsigned int* NNZMap,
                                    int* dataOffsetMap);
void _AVX512_MNNPackedSparseMatMulEpx4(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter,
                                    const float* postParameters, const float* bias, unsigned int* NNZMap,
                                    int* dataOffsetMap);
void _AVX512_MNNPackedSparseMatMulEpx1(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter,
                                    const float* postParameters, const float* bias, unsigned int* NNZMap,
                                    int* dataOffsetMap);

void _AVX512_ReorderInit(void* functions);
void _AVX512_ExtraInit(void* functions);
void _AVX512_WinogradInit(void* functions);
void _AVX512_MNNInt8FunctionInit(void* functions, bool suppotVNNI);

extern MNN::CoreFunctions::MNNPackedMatMulKernel _AVX512_MNNPackedMatMulOC16Functions[AVX512_INPUT_TILE_MAX];
extern MNN::CoreFunctions::MNNPackedMatMulKernel _AVX512_MNNPackedMatMulOC32Functions[AVX512_INPUT_TILE_MAX];
extern MNN::CoreFunctions::MNNPackedMatMulKernel _AVX512_MNNPackedMatMulOC48Functions[AVX512_INPUT_TILE_MAX];

}


