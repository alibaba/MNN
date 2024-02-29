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

#include "seqLen2SpatialKernel.h"
#include <MNN/MNNDefine.h>

template <typename T, int32_t C, int32_t TPB>
__global__ void SeqLen2SpatialKernel(T const* input, T const* biasInput, T const* residualInput, T* output)
{
    int32_t baseOffset = blockIdx.x * C + threadIdx.x;
    int32_t biasOffset = threadIdx.x;
#pragma unroll
    for (int32_t i = 0; i < C / TPB; ++i)
    {
        output[baseOffset] = input[baseOffset] + biasInput[biasOffset] + residualInput[baseOffset];
        baseOffset += TPB;
        biasOffset += TPB;
    }
}

template __global__ void SeqLen2SpatialKernel<float, 320, 320>(float const*, float const*, float const*, float*);
template __global__ void SeqLen2SpatialKernel<float, 640, 320>(float const*, float const*, float const*, float*);
template __global__ void SeqLen2SpatialKernel<float, 1280, 320>(float const*, float const*, float const*, float*);
template __global__ void SeqLen2SpatialKernel<half, 320, 320>(half const*, half const*, half const*, half*);
template __global__ void SeqLen2SpatialKernel<half, 640, 320>(half const*, half const*, half const*, half*);
template __global__ void SeqLen2SpatialKernel<half, 1280, 320>(half const*, half const*, half const*, half*);

int32_t launchSeqLen2SpatialKernel(void const* input0, void const* input1, void const* input2, void* output0, bool isHalf,
    int32_t gridSize, int32_t C, cudaStream_t stream)
{
    if (!isHalf)
    {
        auto const input = static_cast<float const*>(input0);
        auto const biasInput = static_cast<float const*>(input1);
        auto const residualInput = static_cast<float const*>(input2);
        auto output = static_cast<float*>(output0);
        constexpr int32_t TPB = 320; // thread per block
        switch (C)
        {
        case 320:
            (SeqLen2SpatialKernel<float, 320, TPB>) <<<gridSize, TPB, 0, stream>>>(
                input, biasInput, residualInput, output);
            break;
        case 640:
            (SeqLen2SpatialKernel<float, 640, TPB>) <<<gridSize, TPB, 0, stream>>>(
                input, biasInput, residualInput, output);
            break;
        case 1280:
            (SeqLen2SpatialKernel<float, 1280, TPB>) <<<gridSize, TPB, 0, stream>>>(
                input, biasInput, residualInput, output);
            break;
        default: MNN_ERROR("SeqLen2Spatial Unsupported number of channels!\n");
        }
        checkKernelErrors;
    }
    else
    {
        auto const input = static_cast<half const*>(input0);
        auto const biasInput = static_cast<half const*>(input1);
        auto const residualInput = static_cast<half const*>(input2);
        auto output = static_cast<half*>(output0);
        constexpr int32_t TPB = 320; // thread per block
        switch (C)
        {
        case 320:
            (SeqLen2SpatialKernel<half, 320, TPB>) <<<gridSize, TPB, 0, stream>>>(
                input, biasInput, residualInput, output);
            break;
        case 640:
            (SeqLen2SpatialKernel<half, 640, TPB>) <<<gridSize, TPB, 0, stream>>>(
                input, biasInput, residualInput, output);
            break;
        case 1280:
            (SeqLen2SpatialKernel<half, 1280, TPB>) <<<gridSize, TPB, 0, stream>>>(
                input, biasInput, residualInput, output);
            break;
        default: MNN_ERROR("SeqLen2Spatial Unsupported number of channels!\n");
        }
        checkKernelErrors;
    }
    return 0;
}
