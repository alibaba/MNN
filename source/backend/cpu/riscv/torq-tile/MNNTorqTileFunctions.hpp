// Copyright © 2026, Alibaba Group Holding Limited
#ifndef MNN_TORQ_TILE_FUNCTIONS_HPP
#define MNN_TORQ_TILE_FUNCTIONS_HPP

#include <stddef.h>

void MNNPackedMatMulFP32_TorqTile(float* C, const float* A, const float* B, const size_t* parameter,
                                  const float* postParameters, const float* bias, const float* k, const float* b);
void MNNPackedMatMulRemainFP32_TorqTile(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter,
                                        const float* postParameters, const float* bias, const float* k, const float* b);

namespace MNN {
struct CoreFunctions;
void MNNTorqTileInitializeFunctions(CoreFunctions* core);
} // namespace MNN

#endif
