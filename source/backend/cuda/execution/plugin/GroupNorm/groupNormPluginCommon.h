/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef GROUPNORM_PLUGIN_COMMON_H
#define GROUPNORM_PLUGIN_COMMON_H

#include <cstdint>
#include <cuda.h>
#include <cuda_fp16.h>

#include "core/Macro.h"
#include <MNN/MNNDefine.h>
#include "backend/cuda/core/CUDABackend.hpp"

namespace MNN {
namespace CUDA {
struct GroupNormNHWCParams
{
    // The output buffer. Layout NHWC.
    __half* dst;
    // The input buffer when with prefix add. Layout NHWC.
    __half const* src_0;
    __half const* src_1;
    // The input buffer. Layout NHWC.
    __half const* src;
    // The gamma scaling factor.
    float const* gamma;
    // The beta term to add in GN.
    float const* beta;
    // The temporary buffer to do the global parallel reduction. Size:
    // BLOCKS_PER_BATCH x C x 2.
    float* redBuffer;

    // The number of instances in the batch.
    int32_t n;
    // The height and width of each activation map.
    int32_t h;
    int32_t w;
    // The number of channels.
    int32_t c;
    // The number of groups.
    int32_t groups;
    // Do we apply the Swish activation function?
    bool withSwish;

    // Precomputed values and parameters to control the execution of the kernels.

    // The number of activations per instance (h * w) and the number of
    // activations per block.
    int32_t hw;
    int32_t hwPerBlock;
    // The number of channels per group and blocks per activation in the C
    // dimension.
    int32_t cPerBlock;
    int32_t cPerGroup;

    // The precomputed stride between instances.
    int32_t hwc;
    // The inverse of hwc in floats (to compute mean/var).
    float invHWC;
    // The precomputed number of groups per block.
    int32_t groupsPerBlock;
};

} // namespace CUDA
} // namespace MNN

#endif // GROUPNORM_PLUGIN_COMMON_H
