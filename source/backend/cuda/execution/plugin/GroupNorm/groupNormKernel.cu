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
#include "groupNormKernel.cuh"
// Link thrust::system, need -fexceptions
#include <cub/cub.cuh>

//#include "core/Macro.h"
#include <MNN/MNNDefine.h>
#include "backend/cuda/core/CUDABackend.hpp"

namespace MNN {
namespace CUDA {
    
static inline __device__ __host__ float sigmoid(float x)
{
    return 1.F / (1.F + expf(-x));
}

struct GroupSums
{
    // Is it the 1st element of the group?
    int32_t flag;
    // The sum.
    float sum;
    // The sum of squares.
    float sumSq;
};

struct GroupSumsOp
{
    inline __device__ GroupSums operator()(GroupSums const& a, GroupSums const& b)
    {
        GroupSums dst;
        dst.sum = b.flag ? b.sum : (a.sum + b.sum);
        dst.sumSq = b.flag ? b.sumSq : (a.sumSq + b.sumSq);
        dst.flag = a.flag + b.flag;
        return dst;
    }
};

template <int32_t tTHREADS_PER_BLOCK>
__global__ void groupNormNHWCSumKernel(GroupNormNHWCParams params)
{
    // The object in charge of doing the sums for the different blocks.
    typedef cub::BlockScan<GroupSums, tTHREADS_PER_BLOCK> BlockScan;

    // Allocate shared memory for BlockScan.
    __shared__ typename BlockScan::TempStorage tempStorage;
    // Allocate shared memory for the groups. We could reduce the amount of shared
    // memory reserved.
    __shared__ float2 smem[tTHREADS_PER_BLOCK];

    // The instance in the batch.
    int32_t ni = blockIdx.z;
    // The channel loaded by that thread (2 channels per thread for F16x2).
    int32_t ci = blockIdx.x * params.cPerBlock + threadIdx.x * 2;

    // The first activation loaded by that block.
    int32_t hwBegin = blockIdx.y * params.hwPerBlock;
    // The last activation loaded by that block.
    int32_t hwEnd = min(hwBegin + params.hwPerBlock, params.hw);

    // The sums.
    float sum = 0.F;
    float sumSq = 0.F;

    // Iterate over the activations to compute the sums.
    for (int32_t hwi = hwBegin; hwi < hwEnd; ++hwi)
    {
        // The offset.
        int64_t offset = static_cast<int64_t>(ni) * params.hwc + static_cast<int64_t>(hwi) * params.c + ci;

        // Fetch two channels per thread.
        __half2 h2(0, 0);
        float2 f2;
        f2.x = 0.0f; f2.y = 0.0f;
        if (ci < params.c)
        {
            if(params.src != nullptr) {
                h2 = *reinterpret_cast<__half2 const*>(&params.src[offset]);
                f2 = __half22float2(h2);
            } else {
                int64_t offset_1 = static_cast<int64_t>(ni) * params.c + ci;
                __half2 h2_0 = *reinterpret_cast<__half2 const*>(&params.src_0[offset]);
                __half2 h2_1 = *reinterpret_cast<__half2 const*>(&params.src_1[offset_1]);
                float2 f2_0 = __half22float2(h2_0);
                float2 f2_1 = __half22float2(h2_1);
                f2.x = f2_0.x + f2_1.x;
                f2.y = f2_0.y + f2_1.y;
            }
        }

        // Update the sum.
        sum += f2.x + f2.y;
        // Update the sum of squares.
        sumSq += f2.x * f2.x + f2.y * f2.y;
    }

    // The group that thread works on and the channel in the group (modulus).
    int32_t gi = threadIdx.x * 2 / params.cPerGroup;
    int32_t cj = threadIdx.x * 2 - params.cPerGroup * gi;

    // The data for the summations.
    GroupSums inp{cj == 0 ? 1 : 0, sum, sumSq};

    // Do the segmented scan.
    GroupSums out;
    BlockScan(tempStorage).InclusiveScan(inp, out, GroupSumsOp());

    // Store the results for the groups in shared memory (to produce coalesced
    // stores later).
    if (cj == params.cPerGroup - 2 /* 2 channels per thread */)
    {
        smem[gi] = make_float2(out.sum, out.sumSq);
    }

    // Make sure the data is in shared memory.
    __syncthreads();

    // The global group index.
    int32_t gj = blockIdx.x * params.groupsPerBlock + threadIdx.x;

    // Threads that have nothing left to do, exit.
    if (threadIdx.x >= params.groupsPerBlock || gj >= params.groups)
    {
        return;
    }

    // The first threads (those storing to global memory, load the values).
    float2 sums = smem[threadIdx.x];

    // Store to global memory.
    atomicAdd(&params.redBuffer[(2 * ni + 0) * params.groups + gj], sums.x);
    atomicAdd(&params.redBuffer[(2 * ni + 1) * params.groups + gj], sums.y);
}

void groupNormNHWCSum(GroupNormNHWCParams const& params, cudaStream_t stream)
{
    // Make sure the values are as we expect.
    MNN_ASSERT(params.c % params.cPerBlock == 0 && params.hw % params.hwPerBlock == 0);
    // Make sure a group does not span multiple blocks.
    MNN_ASSERT(params.cPerBlock % params.cPerGroup == 0);

    dim3 grid;

    // The number of blocks to compute all the channels.
    grid.x = params.c / params.cPerBlock;
    // The number of blocks to compute all the activations in a given instance.
    grid.y = divUp(params.hw, params.hwPerBlock);
    // The number of instances.
    grid.z = params.n;

    switch (params.cPerBlock)
    {
    case 320: groupNormNHWCSumKernel<160><<<grid, 160, 0, stream>>>(params); break;
    case 480: groupNormNHWCSumKernel<256><<<grid, 256, 0, stream>>>(params); break;
    case 256: groupNormNHWCSumKernel<128><<<grid, 128, 0, stream>>>(params); break;
    case 128: groupNormNHWCSumKernel<64><<<grid, 64, 0, stream>>>(params); break;
    default: MNN_ERROR("groupNormNHWCSumKernel Not implemented");
    }
    checkKernelErrors;
    return;
}

template <int32_t tTHREADS_PER_BLOCK>
__global__ void groupNormNHWCScaleKernel(GroupNormNHWCParams params)
{
    // The instance in the batch.
    int32_t ni = blockIdx.z;
    // The channel loaded by that thread (2 channels per thread for F16x2).
    int32_t ci = blockIdx.x * params.cPerBlock + threadIdx.x * 2;
    // The group that thread works on and the channel in the group (modulus).
    int32_t gi = ci / params.cPerGroup;

    // Load the sum and sum of squares for the group.
    float sum = 0.F, sumSq = 0.F;
    if (gi < params.groups)
    {
        sum = params.redBuffer[(2 * ni + 0) * params.groups + gi];
        sumSq = params.redBuffer[(2 * ni + 1) * params.groups + gi];
    }

    // Load gamma/beta.
    float2 gammaF2, betaF2;
    if (ci < params.c)
    {
        gammaF2 = *reinterpret_cast<float2 const*>(&params.gamma[ci]);
        betaF2 = *reinterpret_cast<float2 const*>(&params.beta[ci]);
    }

    // Compute the mean.
    float mean = sum * params.invHWC;
    // Compute the variance.
    float var = sumSq * params.invHWC - (mean * mean);
    // Compute the inverse of the stddev.
    float invStdDev = var <= 0.F ? 1.F : rsqrtf(var);

    // The first activation loaded by that block.
    int32_t hwBegin = blockIdx.y * params.hwPerBlock;
    // The last activation loaded by that block.
    int32_t hwEnd = min(hwBegin + params.hwPerBlock, params.hw);

    // Iterate over the activations to compute the sums.
    for (int32_t hwi = hwBegin; hwi < hwEnd; ++hwi)
    {
        // The src/dst offset.
        int64_t offset = (int64_t) ni * params.hwc + hwi * params.c + ci;

        // Fetch two channels per thread.
        __half2 h2(0, 0);
        float2 f2;
        f2.x = 0.0f; f2.y = 0.0f;
        if (ci < params.c)
        {
            if(params.src != nullptr) {
                h2 = *reinterpret_cast<__half2 const*>(&params.src[offset]);
                f2 = __half22float2(h2);
            } else {
                int64_t offset_1 = static_cast<int64_t>(ni) * params.c + ci;
                __half2 h2_0 = *reinterpret_cast<__half2 const*>(&params.src_0[offset]);
                __half2 h2_1 = *reinterpret_cast<__half2 const*>(&params.src_1[offset_1]);
                float2 f2_0 = __half22float2(h2_0);
                float2 f2_1 = __half22float2(h2_1);
                f2.x = f2_0.x + f2_1.x;
                f2.y = f2_0.y + f2_1.y;
            }
        }

        // Normalize the channels.
        f2.x = (f2.x - mean) * invStdDev;
        f2.y = (f2.y - mean) * invStdDev;

        // Scale by gamma and add beta.
        f2.x = gammaF2.x * f2.x + betaF2.x;
        f2.y = gammaF2.y * f2.y + betaF2.y;

        // Apply Swish if needed.
        if (params.withSwish)
        {
            f2.x = f2.x * sigmoid(f2.x);
            f2.y = f2.y * sigmoid(f2.y);
        }

        // Store the scaled values.
        if (ci < params.c)
        {
            *reinterpret_cast<__half2*>(&params.dst[offset]) = __float22half2_rn(f2);
        }
    }
}

void groupNormNHWCScale(GroupNormNHWCParams const& params, cudaStream_t stream)
{
    // Make sure the dimensions are aligned with what we expect.
    MNN_ASSERT(params.c % params.cPerBlock == 0);
    // Make sure a group does not span multiple blocks.
    MNN_ASSERT(params.cPerBlock % params.cPerGroup == 0);

    dim3 grid;

    // The number of blocks to compute all the channels.
    grid.x = params.c / params.cPerBlock;
    // The number of blocks to compute all the activations in a given instance.
    grid.y = divUp(params.hw, params.hwPerBlock);
    // The number of instances.
    grid.z = params.n;

    switch (params.cPerBlock)
    {
    case 320: groupNormNHWCScaleKernel<160><<<grid, 160, 0, stream>>>(params); break;
    case 480: groupNormNHWCScaleKernel<256><<<grid, 256, 0, stream>>>(params); break;
    case 256: groupNormNHWCScaleKernel<128><<<grid, 128, 0, stream>>>(params); break;
    case 128: groupNormNHWCScaleKernel<64><<<grid, 64, 0, stream>>>(params); break;
    default: MNN_ERROR("groupNormNHWCScaleKernel Not implemented");
    }
    checkKernelErrors;
    return;
}

} // namespace CUDA
} // namespace MNN
