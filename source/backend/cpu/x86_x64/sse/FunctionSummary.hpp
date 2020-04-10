//
//  FunctionSummary.hpp
//  MNN
//
//  Created by MNN on 2019/08/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

// ========= CommonOptFunction.cpp ===========

void _SSE_MNNAddBias(float* dst, const float* bias, size_t planeNumber, size_t biasNumber);

void _SSE_MNNAddBiasRelu(float* dst, const float* bias, size_t planeNumber, size_t biasNumber);

void _SSE_MNNAddBiasRelu6(float* dst, const float* bias, size_t planeNumber, size_t biasNumber);

void _SSE_MNNCopyC4WithStride(const float* source, float* dest, size_t srcStride, size_t dstStride, size_t count);

void _SSE_MNNAddC4WithStride(const float* source, float* dest, size_t srcStride, size_t dstStride, size_t count);

// ========= MNNConvSlideWindowBorder.cpp ===========

void _SSE_MNNConvSlideWindowBorder(float* dst, const float* src, const float* weight, size_t src_depth_quad,
                                   size_t src_depth_step, size_t fw, size_t fh, size_t weight_y_step, size_t weight_z_step,
                                   size_t dilateX_step, size_t dilateY_step, float* alpha);

// ========= MNNConvSlideWindowMiddle.cpp ===========

void _SSE_MNNConvSlideWindowMiddle(float* dst, const float* src, const float* weight, size_t width, size_t src_w_setup,
                                   size_t src_depth_quad, size_t src_depth_step, size_t fw, size_t fh, size_t dilateX_step,
                                   size_t dilateY_step, float* alpha);

void _SSE_MNNGemmFloatUnit_4(float* dstOrigin, const float* src, const float* weight, size_t src_depth_quad, size_t dst_step,
                        size_t dst_depth_quad, size_t weight_depth_offset);

// ========= MNNGemmFloatCommon_4.cpp ===========

void _SSE_MNNGemmFloatCommon_4(float* dst, const float* src, const float* weight, size_t src_depth_quad, size_t dst_step,
                               size_t dst_depth_quad, size_t width, size_t weight_depth_offset);

// ========= MNNMatrixAdd.cpp ===========

void _SSE_MNNMatrixAdd(float* C, const float* A, const float* B, size_t widthC4, size_t cStride, size_t aStride,
                       size_t bStride, size_t height);

// ========= MNNMatrixSub.cpp ===========

void _SSE_MNNMatrixSub(float* C, const float* A, const float* B, size_t widthC4, size_t cStride, size_t aStride,
                       size_t bStride, size_t height);

void _SSE_MNNReluWithSlopeChannel(float* dst, const float* src, const float* slope, size_t sizeQuad, size_t depthQuad);
