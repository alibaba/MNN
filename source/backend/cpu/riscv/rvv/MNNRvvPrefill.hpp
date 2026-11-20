//
//  MNNRvvPrefill.hpp
//  MNN
//
//  Created by MNN on 2026/11/19.
//  Copyright © 2026, Alibaba Group Holding Limited
//

#pragma once

#include "backend/cpu/compute/CommonOptFunction.h"
#include <vector>

namespace MNN {
constexpr int kRvvGatedDeltaChunk = 16;
constexpr int kRvvGatedDeltaHeadDim = 128;
void MNNRvvInitPackedNorm(CoreFunctions* core);
class Tensor;
class CPUBackend;
class Execution;
struct Op;
Execution* MNNRvvCreateLinearAttention(CPUBackend* backend, const Op* op, const std::vector<Tensor*>& inputs);
} // namespace MNN

void MNNMulSiLu_RVV(float*, const float*, const float*, size_t, bool);
void MNNDecayRankOneUpdate_RVV(float*, const float*, const float*, float, size_t, size_t);
void MNNConv4SiLuC4_RVV(float* out, const float* input, float* state, const float* weight, size_t length);
// Q, K and V contain one full 16-token block with 128 channels per head.
void MNNGatedDeltaChunk_RVV(float* state, const float* q, const float* k, float* v, float* projection, float* output,
                            const float* decay, const float* beta, size_t outputStride, bool projected);
