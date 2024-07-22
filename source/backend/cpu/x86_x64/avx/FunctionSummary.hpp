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
void _AVX_MNNPackedMatMul(float* C, const float* A, const float* B, const size_t* parameter,
                          const float* postParameters, const float* bias, const float* k, const float* b);
void _AVX_MNNPackedMatMulRemain(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter,
                                const float* postParameters, const float* bias, const float* k, const float* b);
#ifdef MNN_LOW_MEMORY
void _AVX_MNNPackedMatMul_int4(float* C, const float* A, const float* B, const size_t* parameter,
                          const float* postParameters, const float* bias, const float* k, const float* b);
void _AVX_MNNPackedMatMulRemain_int4(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter,
                                     const float* postParameters, const float* bias, const float* k, const float* b);
void _AVX_MNNPackedMatMul_int8(float* C, const float* A, const float* B, const size_t* parameter,
                          const float* postParameters, const float* bias, const float* k, const float* b);
void _AVX_MNNPackedMatMulRemain_int8(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter,
                                     const float* postParameters, const float* bias, const float* k, const float* b);
void _AVX_MNNAbsMaxFP32(const float* source, float* absmax, size_t src_depth_quad, size_t realSize, int pack);
#endif
void _AVX_MNNPackC4ForMatMul_A(float* destOrigin, float const** sourceGroup, const int32_t* info, const int32_t* el);

void _AVX_MNNExpC8(float* dest, const float* source, float* offset, const float* parameters, size_t countC8);
void _AVX_MNNSoftmax(float* dest, const float* source, size_t size);
void _AVX_MNNFloat2Int8(const float* src, int8_t* dst, size_t sizeQuad, const float* scalep, ssize_t minV, ssize_t maxV, ssize_t zeroPoint);
void _AVX_MNNInt8ScaleToFloat(float* dst, const int8_t* src, const float* scale, size_t sizeQuad, ssize_t zeroPoint);
void _AVX_MNNLineDepthWiseInt8AddBiasScaleUnit(int8_t* dstO, const int8_t* srcO, const int8_t* weightO, const QuanPostTreatParameters* parameters, size_t width, size_t src_w_step, size_t fw, size_t fh, size_t dilateX_step, size_t dilateY_step, int8_t* idxOrder);
void _AVX_MNNComputeMatMulForE_1(const float* A, const float* B, float* C, const float* biasPtr, const MatMulParam* param, size_t tId);
void _AVX_MNNPackC4ForMatMul_A_BF16(float* destOrigin, float const** sourceGroup, const int32_t* info, const int32_t* el);
void _AVX_MNNComputeScaleZeroScalar(float* source, float* minVal, float* maxVal, size_t size);

void _AVX_MNNGetMatMulPackMode_BF16(int* eP, int *lP, int* hP);
void _AVX_MNNPackForMatMul_B_BF16(float* dest, const float* source, size_t h, size_t l, bool transpose);
void _AVX_MNNPackedSparseMatMul(float* C, const float* A, const float* B, unsigned int* NNZMap, int* dataOffsetMap, size_t eSize, const size_t* parameter, const float* postParameters, const float* bias);
void _AVX_MNNComputeMatMulForH_1(const float* A, const float* B, float* C, const float* biasPtr, const MatMulParam* param, size_t tId);

void _AVX_ReorderInit(void* functions);
void _AVX_MNNInt8FunctionInit(void* functions);
void _AVX_MNNPackCUnit(float* dst, const float* src, size_t area, size_t depth, int* areaOffset);
void _AVX_MNNUnpackCUnit(float* dst, const float* src, size_t area, size_t depth, int* areaOffset);
void _AVX_MNNPackCUnitTranspose(float* dst, const float* src, size_t area, size_t depth, int* areaOffset);
void _AVX_MNNUnpackCUnitTranspose(float* dst, const float* src, size_t area, size_t depth, int* areaOffset);
void _AVX_MNNPackForMatMul_B(float* dest, const float* source, size_t h, size_t l, bool transpose);

void _AVX_ExtraInit(void* functions);
void _AVX_WinogradInit(void* functions);

void _AVX_MNNGelu(float *dst, const float *src, size_t size, float* parameters);
void _AVX_MNNNorm(float *dst, const float *src, const float *gamma, const float *beta, float epsilon, size_t size, bool RMSNorm);
void _AVX_MNNNormInt8(int8_t* dst, const int8_t* src, const float* gamma, const float* beta, float epsilon, size_t size, QuanPrePostParameters* params, bool RMSNorm);

void _AVX_MNNGetSparseMatMulPackMode(int* eP, int *lP, int* hP);
void _AVX_MNNPackedSparseMatMulEpx1EFMA(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter, const float* postParameters, const float* bias, unsigned int* NNZMap, int* dataOffsetMap);
void _AVX_MNNPackedSparseMatMulEpx4EFMA(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter, const float* postParameters, const float* bias, unsigned int* NNZMap, int* dataOffsetMap);
}
