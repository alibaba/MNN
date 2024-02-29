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
#ifndef GROUPNORM_KERNEL_H
#define GROUPNORM_KERNEL_H

#include "groupNormPluginCommon.h"

#include <cstdint>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

namespace MNN {
namespace CUDA {
using half = __half;

static inline int32_t divUp(int32_t m, int32_t n)
{
    return (m + n - 1) / n;
}
void groupNormNHWCSum(GroupNormNHWCParams const& params, cudaStream_t stream = 0);

void groupNormNHWCScale(GroupNormNHWCParams const& params, cudaStream_t stream = 0);
} // namespace CUDA
} // namespace MNN

#endif // GROUPNORM_KERNEL_H
