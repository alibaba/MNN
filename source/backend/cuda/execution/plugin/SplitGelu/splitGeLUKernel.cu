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

#include "splitGeLUKernel.h"
#include <MNN/MNNDefine.h>

template <typename T, int32_t tHHS, int32_t tTPB>
__global__ void splitGeLUKernel(T const* input0, T const* input1, T* output, float const fDivRecip, float const fAdd, float const fMul)
{
    MNN_ASSERT(input0 != nullptr);
    MNN_ASSERT(output != nullptr);

    int32_t indexInput = blockIdx.x * tHHS * 2 + threadIdx.x;
    int32_t indexInput1 = threadIdx.x;
    int32_t indexOutput = blockIdx.x * tHHS + threadIdx.x;
    float valueL, valueR;
#pragma unroll
    for (int32_t i = 0; i < tHHS / tTPB; ++i)
    {
        if(input1 == nullptr) {
            valueL = static_cast<float>(input0[indexInput]);
            valueR = static_cast<float>(input0[indexInput + tHHS]);
        } else {
            valueL = static_cast<float>(input0[indexInput]) + static_cast<float>(input1[indexInput1]);
            valueR = static_cast<float>(input0[indexInput + tHHS]) + static_cast<float>(input1[indexInput1 + tHHS]);
            indexInput1 += tTPB;
        }
        float tmp = valueR;
        tmp *= fDivRecip;
        tmp = erff(tmp);
        tmp += fAdd;
        tmp *= valueR;
        tmp *= fMul;
        tmp *= valueL;
        output[indexOutput] = static_cast<T>(tmp);
        indexInput += tTPB;
        indexOutput += tTPB;
    }
    return;
}

template <typename T>
int32_t launchSplitGeLUKernel(int32_t gridSize, int32_t nHalfHiddenSize, T const* input0, T const* input1, T* output,
    float const fDiv, float const fAdd, float const fMul, cudaStream_t stream)
{
    MNN_ASSERT(fDiv != 0.F);

    auto const fDivRecip = 1.F / fDiv;
    constexpr int32_t kTPB = 256; // thread per block
    switch (nHalfHiddenSize)
    {
    case 1280: (splitGeLUKernel<T, 1280, kTPB>) <<<gridSize, kTPB, 0, stream>>>(input0, (T const*)input1, output, fDivRecip, fAdd, fMul); break;
    case 2560: (splitGeLUKernel<T, 2560, kTPB>) <<<gridSize, kTPB, 0, stream>>>(input0, (T const*)input1, output, fDivRecip, fAdd, fMul); break;
    case 5120: (splitGeLUKernel<T, 5120, kTPB>) <<<gridSize, kTPB, 0, stream>>>(input0, (T const*)input1, output, fDivRecip, fAdd, fMul); break;
    }

    checkKernelErrors;
    return 0;
}

template __global__ void splitGeLUKernel<float, 1280, 256>(float const*, float const*, float*, float const, float const, float const);
template __global__ void splitGeLUKernel<float, 2560, 256>(float const*, float const*, float*, float const, float const, float const);
template __global__ void splitGeLUKernel<float, 5120, 256>(float const*, float const*, float*, float const, float const, float const);
template __global__ void splitGeLUKernel<half, 1280, 256>(half const*, half const*, half*, float const, float const, float const);
template __global__ void splitGeLUKernel<half, 2560, 256>(half const*, half const*, half*, float const, float const, float const);
template __global__ void splitGeLUKernel<half, 5120, 256>(half const*, half const*, half*, float const, float const, float const);

template int32_t launchSplitGeLUKernel<float>(int32_t gridSize, int32_t nHalfHiddenSize,
    float const* input0, float const* input1, float* output, float const fDiv, float const fAdd, float const fMul, cudaStream_t stream);

template int32_t launchSplitGeLUKernel<half>(int32_t gridSize, int32_t nHalfHiddenSize,
    half const* input0, half const* input1, half* output, float const fDiv, float const fAdd, float const fMul, cudaStream_t stream);
