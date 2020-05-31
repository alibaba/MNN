//
//  FunctionDispatcher.cpp
//  MNN
//
//  Created by MNN on 2019/08/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#include "DispatchHelper.hpp"
#include "backend/cpu/compute/CommonOptFunction.h"
#include "backend/cpu/compute/ConvOpt.h"
#include "backend/cpu/compute/Int8FunctionsOpt.h"
#include "sse/FunctionSummary.hpp"
#include "avx/FunctionSummary.hpp"

// https://stackoverflow.com/a/11230437
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
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

bool MNNReorder4x4ByPlatform(float* dst, size_t number) {
    for (int i=0; i<number; ++i) {
        auto addr = dst + 16 * i;
        auto s0 = _mm_loadu_ps(addr + 4 * 0);
        auto s1 = _mm_loadu_ps(addr + 4 * 1);
        auto s2 = _mm_loadu_ps(addr + 4 * 2);
        auto s3 = _mm_loadu_ps(addr + 4 * 3);
        _MM_TRANSPOSE4_PS(s0, s1, s2, s3);
        
        _mm_storeu_ps(addr + 4 * 0, s0);
        _mm_storeu_ps(addr + 4 * 1, s1);
        _mm_storeu_ps(addr + 4 * 2, s2);
        _mm_storeu_ps(addr + 4 * 3, s3);
    }
    return true;
}

// ========= CommonOptFunction.cpp ===========
void MNNAddBias(float* dst, const float* bias, size_t planeNumber, size_t biasNumber) {
    if(cpu_feature_available(AVX)) {
        _AVX_MNNAddBias(dst, bias, planeNumber, biasNumber);
    } else {
        _SSE_MNNAddBias(dst, bias, planeNumber, biasNumber);
    }
}

void MNNAddBiasRelu(float* dst, const float* bias, size_t planeNumber, size_t biasNumber) {
    if(cpu_feature_available(AVX)) {
        _AVX_MNNAddBiasRelu(dst, bias, planeNumber, biasNumber);
    } else {
        _SSE_MNNAddBiasRelu(dst, bias, planeNumber, biasNumber);
    }
}

void MNNAddBiasRelu6(float* dst, const float* bias, size_t planeNumber, size_t biasNumber) {
    if(cpu_feature_available(AVX)) {
        _AVX_MNNAddBiasRelu6(dst, bias, planeNumber, biasNumber);
    } else {
        _SSE_MNNAddBiasRelu6(dst, bias, planeNumber, biasNumber);
    }
}

void MNNCopyC4WithStride(const float* source, float* dest, size_t srcStride, size_t dstStride, size_t count) {
    _SSE_MNNCopyC4WithStride(source, dest, srcStride, dstStride, count);
}

void MNNAddC4WithStride(const float* source, float* dest, size_t srcStride, size_t dstStride, size_t count) {
    _SSE_MNNAddC4WithStride(source, dest, srcStride, dstStride, count);
}

// ========= MNNConvSlideWindowBorder.cpp ===========
void MNNConvSlideWindowBorder(float* dst, const float* src, const float* weight, size_t src_depth_quad,
                              size_t src_depth_step, size_t fw, size_t fh, size_t weight_y_step, size_t weight_z_step,
                              size_t dilateX_step, size_t dilateY_step, float* alpha) {
    _SSE_MNNConvSlideWindowBorder(dst, src, weight, src_depth_quad, src_depth_step, fw, fh,
                                  weight_y_step, weight_z_step, dilateX_step, dilateY_step, alpha);
}

// ========= MNNConvSlideWindowMiddle.cpp ===========
void MNNConvSlideWindowMiddle(float* dst, const float* src, const float* weight, size_t width, size_t src_w_setup,
                              size_t src_depth_quad, size_t src_depth_step, size_t fw, size_t fh, size_t dilateX_step,
                              size_t dilateY_step, float* alpha) {
    if (cpu_feature_available(AVX)) {
        _AVX_MNNConvSlideWindowMiddle(dst, src, weight, width, src_w_setup, src_depth_quad, src_depth_step, fw, fh, dilateX_step, dilateY_step, alpha);
    } else {
        _SSE_MNNConvSlideWindowMiddle(dst, src, weight, width, src_w_setup, src_depth_quad, src_depth_step, fw, fh, dilateX_step, dilateY_step, alpha);
    }
}

void MNNGemmFloatUnit_4(float* dstOrigin, const float* src, const float* weight, size_t src_depth_quad, size_t dst_step,
                        size_t dst_depth_quad, size_t weight_depth_offset) {
    if (cpu_feature_available(AVX)) {
        _AVX_MNNGemmFloatUnit_4(dstOrigin, src, weight, src_depth_quad, dst_step, dst_depth_quad, weight_depth_offset);
    } else {
        _SSE_MNNGemmFloatUnit_4(dstOrigin, src, weight, src_depth_quad, dst_step, dst_depth_quad, weight_depth_offset);
    }
}

// ========= MNNGemmFloatCommon_4.cpp ===========
void MNNGemmFloatCommon_4(float* dst, const float* src, const float* weight, size_t src_depth_quad, size_t dst_step,
                          size_t dst_depth_quad, size_t width, size_t weight_depth_offset) {
    if (cpu_feature_available(AVX)) {
        _AVX_MNNGemmFloatCommon_4(dst, src, weight, src_depth_quad, dst_step, dst_depth_quad, width, weight_depth_offset);
    } else {
        _SSE_MNNGemmFloatCommon_4(dst, src, weight, src_depth_quad, dst_step, dst_depth_quad, width, weight_depth_offset);
    }
}

// ========= MNNMatrixAdd.cpp ===========
void MNNMatrixAdd(float* C, const float* A, const float* B, size_t widthC4, size_t cStride, size_t aStride,
                  size_t bStride, size_t height) {
    if (cpu_feature_available(AVX)) {
        _AVX_MNNMatrixAdd(C, A, B, widthC4, cStride, aStride, bStride, height);
    } else {
        _SSE_MNNMatrixAdd(C, A, B, widthC4, cStride, aStride, bStride, height);
    }
}

// ========= MNNMatrixSub.cpp ===========
void MNNMatrixSub(float* C, const float* A, const float* B, size_t widthC4, size_t cStride, size_t aStride,
                  size_t bStride, size_t height) {
    if (cpu_feature_available(AVX)) {
        _AVX_MNNMatrixSub(C, A, B, widthC4, cStride, aStride, bStride, height);
    } else {
        _SSE_MNNMatrixSub(C, A, B, widthC4, cStride, aStride, bStride, height);
    }
}

#include <algorithm>
#include <cmath>
inline int8_t int32ToInt8(int data, int bias, float scale) {
    float value = (float)(data + bias) * scale;
    value       = std::max(value, -127.0f);
    value       = std::min(value, 127.0f);
    return static_cast<int8_t>(roundf(value));
}
void MNNGemmInt8AddBiasScale_16x4_Unit(int8_t* dst, const int8_t* src, const int8_t* weight, const int32_t* bias,
                                       const float* scale, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad) {
#ifdef MNN_OPTIMIZE_INT8_SSE
    if (cpu_feature_available(AVX)) {
        return _AVX_MNNGemmInt8AddBiasScale_16x4_Unit(dst, src, weight, bias, scale, src_depth_quad, dst_step, dst_depth_quad);
    } else
#endif
    {
        const auto dst_step_tmp = dst_step / sizeof(int8_t);
        for (int dz = 0; dz < dst_depth_quad; ++dz) {
            const auto weight_dz = weight + dz * src_depth_quad * (GEMM_INT8_UNIT * GEMM_INT8_SRC_UNIT);
            const auto bias_dz   = bias + dz * GEMM_INT8_UNIT;
            const auto scale_dz  = scale + dz * GEMM_INT8_UNIT;
            auto dst_z           = dst + dz * dst_step_tmp;
            for (int w = 0; w < GEMM_INT8_DST_XUNIT; ++w) {
                const auto src_x   = src + w * GEMM_INT8_SRC_UNIT;
                auto dst_x         = dst_z + w * GEMM_INT8_UNIT;
                int32_t dstTemp[4] = {0, 0, 0, 0};

                for (int sz = 0; sz < src_depth_quad; ++sz) {
                    const auto weight_sz = weight_dz + (GEMM_INT8_UNIT * GEMM_INT8_SRC_UNIT) * sz;
                    const auto src_z     = src_x + sz * GEMM_INT8_DST_XUNIT * GEMM_INT8_SRC_UNIT;

                    for (int j = 0; j < GEMM_INT8_UNIT; ++j) {
                        const auto weight_j = weight_sz + j * GEMM_INT8_SRC_UNIT;
                        for (int i = 0; i < GEMM_INT8_SRC_UNIT; ++i) {
                            dstTemp[j] += (int32_t)src_z[i] * (int32_t)weight_j[i];
                        }
                    }
                }

                for (int j = 0; j < 4; ++j) {
                    dst_x[j] = int32ToInt8(dstTemp[j], bias_dz[j], scale_dz[j]);
                }
            }
        }
    }
 }

void MNNGemmInt8AddBiasScale_16x4_Unit_FAST(int8_t* dst, const int8_t* src, const int8_t* weight, const int32_t* bias, const float* scale, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad){
    return MNNGemmInt8AddBiasScale_16x4_Unit(dst, src, weight, bias, scale, src_depth_quad, dst_step, dst_depth_quad);
}

void MNNReluWithSlopeChannel(float* dst, const float* src, const float* slope, size_t sizeQuad, size_t depthQuad) {
    return _SSE_MNNReluWithSlopeChannel(dst, src, slope, sizeQuad, depthQuad);
}

