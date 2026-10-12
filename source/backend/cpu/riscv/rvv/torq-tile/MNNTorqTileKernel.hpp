//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MNN_TORQ_TILE_KERNEL_HPP
#define MNN_TORQ_TILE_KERNEL_HPP

#include <stddef.h>

// Compute one FP32 8x4 tile. A is [depth, aStride], B is [depth, 4], and
// dst is [8, 4]. Strides are in floats; depth > 0 and aStride >= 8.
// Requires RVV 1.0 with VLEN >= 128. Bias and clamp belong to the caller.
void MNNTorqTileMatMul8x4(float* dst, const float* a, const float* b, size_t depth, size_t aStride);

#endif // MNN_TORQ_TILE_KERNEL_HPP
