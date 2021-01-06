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

void _AVX_MNNAddBias(float* dst, const float* bias, size_t planeNumber, size_t biasNumber);

void _AVX_MNNAddBiasRelu(float* dst, const float* bias, size_t planeNumber, size_t biasNumber);

void _AVX_MNNAddBiasRelu6(float* dst, const float* bias, size_t planeNumber, size_t biasNumber);

// ========= MNNConvSlideWindowMiddle.cpp ===========

void _AVX_MNNConvSlideWindowMiddle(float* dst, const float* src, const float* weight, size_t width, size_t src_w_setup,
                                   size_t src_depth_quad, size_t src_depth_step, size_t fw, size_t fh,
                                   size_t dilateX_step, size_t dilateY_step, float* alpha);
void _AVX_MNNConvSlideWindowMiddleFMA(float* dst, const float* src, const float* weight, size_t width,
                                      size_t src_w_setup, size_t src_depth_quad, size_t src_depth_step, size_t fw,
                                      size_t fh, size_t dilateX_step, size_t dilateY_step, float* alpha);
void _AVX_MNNGemmFloatCommonFMA_4(float* dst, const float* src, const float* weight, size_t src_depth_quad,
                                  size_t dst_step, size_t dst_depth_quad, size_t width, size_t weight_depth_offset);

// ========= MNNGemmFloatCommon_4.cpp ===========

void _AVX_MNNGemmFloatCommon_4(float* dst, const float* src, const float* weight, size_t src_depth_quad,
                               size_t dst_step, size_t dst_depth_quad, size_t width, size_t weight_depth_offset);

void _AVX_MNNGemmFloatUnit_4(float* dstOrigin, const float* src, const float* weight, size_t src_depth_quad,
                             size_t dst_step, size_t dst_depth_quad, size_t weight_depth_offset);
void _AVX_MNNGemmFloatUnitFMA_4(float* dstOrigin, const float* src, const float* weight, size_t src_depth_quad,
                                size_t dst_step, size_t dst_depth_quad, size_t weight_depth_offset);

// ========= MNNMatrixAdd.cpp ===========

void _AVX_MNNMatrixAdd(float* C, const float* A, const float* B, size_t widthC4, size_t cStride, size_t aStride,
                       size_t bStride, size_t height);

// ========= MNNMatrixSub.cpp ===========

void _AVX_MNNMatrixSub(float* C, const float* A, const float* B, size_t widthC4, size_t cStride, size_t aStride,
                       size_t bStride, size_t height);

void _AVX_MNNStrassenMergeCFunction(float* c11, float* c12, float* c21, float* c22, float* xAddr, size_t cStride,
                                    size_t length, size_t hSub);

void _AVX_MNNPackedMatMul(float* C, const float* A, const float* B, const size_t* parameter, float* cache,
                          const float* postParameters, const float* bias);
void _AVX_MNNPackedMatMulRemain(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter,
                                float* cache, const float* postParameters, const float* bias);

void _AVX_MNNPackedMatMulFMA(float* C, const float* A, const float* B, const size_t* parameter, float* cache,
                             const float* postParameters, const float* bias);
void _AVX_MNNPackedMatMulRemainFMA(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter,
                                   float* cache, const float* postParameters, const float* bias);

void _AVX_MNNPackC4ForMatMul_A(float* dest, const float* source, size_t e, size_t l, size_t eReal);

void _AVX_MNNConvRunForLineDepthwise(float* dst, const float* src, const float* weight, size_t width, size_t src_w_setup,
                                size_t fw, size_t fh, size_t dilateX_step, size_t dilateY_step, size_t height,
                                     size_t srcHStep, size_t dstHStep);
void _AVX_MNNGemmInt8AddBiasScale_16x4_Unit(int8_t* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad, const QuanPostTreatParameters* post, size_t realDst);

void _AVX_MNNExpC8(float* dest, const float* source, const float* parameters, size_t countC8);
void _AVX_MNNFloat2Int8(const float* src, int8_t* dst, size_t sizeQuad, const float* scalep, ssize_t minV, ssize_t maxV, ssize_t zeroPoint);
void _AVX_MNNInt8ScaleToFloat(float* dst, const int8_t* src, const float* scale, size_t sizeQuad, ssize_t zeroPoint);
void _AVX_MNNLineDepthWiseInt8AddBiasScaleUnit(int8_t* dstO, const int8_t* srcO, const int8_t* weightO, const QuanPostTreatParameters* parameters, size_t width, size_t src_w_step, size_t fw, size_t fh, size_t dilateX_step, size_t dilateY_step);
void _AVX_MNNComputeMatMulForE_1(const float* A, const float* B, float* C, const float* biasPtr, const MatMulParam* param, size_t tId);
void _AVX_MNNComputeMatMulForE_1FMA(const float* A, const float* B, float* C, const float* biasPtr, const MatMulParam* param, size_t tId);

}
