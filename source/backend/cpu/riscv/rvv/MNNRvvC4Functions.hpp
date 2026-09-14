// Copyright © 2026, Alibaba Group Holding Limited
#ifndef MNN_RVV_C4_FUNCTIONS_HPP
#define MNN_RVV_C4_FUNCTIONS_HPP

#include <stddef.h>

// Distinct symbols preserve the generic functions for CPUs without vector support.
void MNNScaleAndAddBias_RVV(float* dst, const float* src, const float* bias, const float* alpha, size_t planeNumber,
                            size_t biasNumber);
void MNNReluWithSlopeChannel_RVV(float* dst, const float* src, const float* slope, size_t sizeQuad, size_t depthQuad);

#endif
