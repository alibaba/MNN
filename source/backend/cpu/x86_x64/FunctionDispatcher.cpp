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
#include "sse/FunctionSummary.hpp"
#include "avx/FunctionSummary.hpp"

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
    if (width % 2 == 0 && cpu_feature_available(AVX)) {
        _AVX_MNNConvSlideWindowMiddle(dst, src, weight, width, src_w_setup, src_depth_quad, src_depth_step, fw, fh, dilateX_step, dilateY_step, alpha);
    } else {
        _SSE_MNNConvSlideWindowMiddle(dst, src, weight, width, src_w_setup, src_depth_quad, src_depth_step, fw, fh, dilateX_step, dilateY_step, alpha);
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
