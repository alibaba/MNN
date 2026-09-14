// Copyright © 2026, Alibaba Group Holding Limited
#ifndef MNN_RVV_MATMUL_FUNCTIONS_HPP
#define MNN_RVV_MATMUL_FUNCTIONS_HPP

#include "backend/cpu/compute/CommonOptFunction.h"

void MNNComputeMatMulForE_1_RVV(const float* A, const float* B, float* C, const float* bias, const MatMulParam* param,
                                size_t tId);

#endif
