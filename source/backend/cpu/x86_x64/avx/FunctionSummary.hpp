//
//  FunctionSummarySSE.hpp
//  MNN
//
//  Created by MNN on 2019/08/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

// ========= CommonOptFunction.cpp ===========

void _AVX_MNNAddBias(float* dst, const float* bias, size_t planeNumber, size_t biasNumber);

void _AVX_MNNAddBiasRelu(float* dst, const float* bias, size_t planeNumber, size_t biasNumber);

void _AVX_MNNAddBiasRelu6(float* dst, const float* bias, size_t planeNumber, size_t biasNumber);

// ========= MNNConvSlideWindowMiddle.cpp ===========

void _AVX_MNNConvSlideWindowMiddle(float* dst, const float* src, const float* weight, size_t width, size_t src_w_setup,
                                   size_t src_depth_quad, size_t src_depth_step, size_t fw, size_t fh, size_t dilateX_step,
                                   size_t dilateY_step, float* alpha);

// ========= MNNGemmFloatCommon_4.cpp ===========

void _AVX_MNNGemmFloatCommon_4(float* dst, const float* src, const float* weight, size_t src_depth_quad, size_t dst_step,
                               size_t dst_depth_quad, size_t width, size_t weight_depth_offset);

void _AVX_MNNGemmFloatUnit_4(float* dstOrigin, const float* src, const float* weight, size_t src_depth_quad, size_t dst_step,
                        size_t dst_depth_quad, size_t weight_depth_offset);

// ========= MNNMatrixAdd.cpp ===========

void _AVX_MNNMatrixAdd(float* C, const float* A, const float* B, size_t widthC4, size_t cStride, size_t aStride,
                       size_t bStride, size_t height);

// ========= MNNMatrixSub.cpp ===========

void _AVX_MNNMatrixSub(float* C, const float* A, const float* B, size_t widthC4, size_t cStride, size_t aStride,
                       size_t bStride, size_t height);

void _AVX_MNNGemmInt8AddBiasScale_16x4_Unit(int8_t* dst, const int8_t* src, const int8_t* weight, const int32_t* bias,
                                       const float* scale, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad);
