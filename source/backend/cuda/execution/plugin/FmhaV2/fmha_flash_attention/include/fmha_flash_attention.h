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

#ifndef FMHA_FLASH_ATTENTION_H
#define FMHA_FLASH_ATTENTION_H

#include <MNN/MNNDefine.h>
#include "commonDatatype.h"
#include "sharedCubinLoader.h"
#include "cuda_fp16.h"

constexpr int32_t kSM_75 = 75;
constexpr int32_t kSM_80 = 80;
constexpr int32_t kSM_86 = 86;
constexpr int32_t kSM_87 = 87;
constexpr int32_t kSM_89 = 89;
constexpr int32_t kSM_90 = 90;

namespace
{
////////////////////////////////////////////////////////////////////////////////////////////////////
// Do not modify this, it is integrated from src/fused_multihead_attention_utils.h in fmha-flash-attention.
////////////////////////////////////////////////////////////////////////////////////////////////////
static void set_alpha(uint32_t& alpha, float norm, MNN::CUDA::MHFADataType dtype)
{
    if (dtype == MNN::CUDA::DATA_TYPE_FP16)
    {
        half x = __float2half_rn(norm);
        uint16_t h = reinterpret_cast<uint16_t const&>(x);
        ushort2 h2 = {h, h};
        alpha = reinterpret_cast<uint32_t const&>(h2);
    }
    else if (dtype == MNN::CUDA::DATA_TYPE_FP32)
    {
        alpha = reinterpret_cast<uint32_t const&>(norm);
    }
    else if (dtype == MNN::CUDA::DATA_TYPE_INT32)
    {
        int32_t inorm = static_cast<int32_t>(norm);
        alpha = reinterpret_cast<uint32_t const&>(inorm);
    }
    else
    {
        assert(false);
    }
}

static int64_t get_size_in_bytes(size_t n, MNN::CUDA::MHFADataType dtype)
{
    switch (dtype)
    {
    case MNN::CUDA::DATA_TYPE_E8M10: return n * 4;
    case MNN::CUDA::DATA_TYPE_FP32: return n * 4;
    case MNN::CUDA::DATA_TYPE_FP16: return n * 2;
    case MNN::CUDA::DATA_TYPE_INT32: return n * 4;
    case MNN::CUDA::DATA_TYPE_INT8: return n;
    case MNN::CUDA::DATA_TYPE_INT4: return n / 2U;
    case MNN::CUDA::DATA_TYPE_BOOL: return n / 8U;
    case MNN::CUDA::DATA_TYPE_E8M7: return n * 2;
    default: MNN_ASSERT(false); return 0;
    }
}
} // namespace

namespace MNN
{
namespace CUDA
{

////////////////////////////////////////////////////////////////////////////////////////////////////
// Do not modify this, it is integrated from src/fused_multihead_attention_demo_bert_params.h in fmha-flash-attention.
////////////////////////////////////////////////////////////////////////////////////////////////////

struct Fused_multihead_flash_attention_params_v2
{
    // The QKV matrices.
    void* qkv_ptr;
    // The mask to implement drop-out.
    void* packed_mask_ptr;
    // The O matrix (output).
    void* o_ptr;

    // The stride between rows of the Q, K and V matrices.
    int64_t qkv_stride_in_bytes;
    // The stride between matrices of packed mask.
    int64_t packed_mask_stride_in_bytes;
    // The stride between rows of O.
    int64_t o_stride_in_bytes;

#if defined(STORE_P)
    // The pointer to the P matrix (for debugging).
    void* p_ptr;
    // The stride between rows of the P matrix (for debugging).
    int64_t p_stride_in_bytes;
#endif // defined(STORE_P)

#if defined(STORE_S)
    // The pointer to the S matrix (for debugging).
    void* s_ptr;
    // The stride between rows of the S matrix (for debugging).
    int64_t s_stride_in_bytes;
#endif // defined(STORE_S)

    // The dimensions.
    int32_t b, h, s, d;
    // The scaling factors for the kernel.
    uint32_t scale_bmm1, scale_softmax, scale_bmm2;

    // Do we use Niall's trick to avoid I2F/F2I in the INT8 kernel.
    // See https://confluence.nvidia.com/pages/viewpage.action?pageId=302779721 for details.
    bool enable_i2f_trick;

    // array of length b+1 holding prefix sum of actual sequence lenghts
    int32_t* cu_seqlens;

    // use C/32 Format.
    bool interleaved = false;
    bool ignore_b1opt = false;
    bool force_unroll = false;
    bool use_int8_scale_max = false;

    void clear()
    {
        qkv_ptr = nullptr;
        packed_mask_ptr = nullptr;
        o_ptr = nullptr;

        qkv_stride_in_bytes = 0;
        packed_mask_stride_in_bytes = 0;
        o_stride_in_bytes = 0;
#if defined(STORE_P)
        p_ptr = nullptr;
        p_stride_in_bytes = 0
#endif // defined(STORE_P)

#if defined(STORE_S)
            s_ptr
            = nullptr;
        s_stride_in_bytes = 0;
#endif // defined(STORE_S)

        b = 0;
        h = 0;
        s = 0;
        d = 0;
        // The scaling factors for the kernel.
        scale_bmm1 = 0;
        scale_softmax = 0;
        scale_bmm2 = 0;

        enable_i2f_trick = false;

        cu_seqlens = nullptr;
        interleaved = false;
        ignore_b1opt = false;
        force_unroll = false;
        use_int8_scale_max = false;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Do not modify this, it is integrated from generated/fmha_cubin.h in fmha-flash-attention.
////////////////////////////////////////////////////////////////////////////////////////////////////
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_64_S_16_sm75_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_64_S_32_sm75_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_64_S_40_sm75_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_64_S_64_sm75_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_32_S_80_sm75_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_32_S_128_sm75_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm75_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm75_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_32_S_16_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_128_16_S_16_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_32_S_32_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_128_16_S_32_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_32_S_40_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_128_16_S_40_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_32_S_64_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_128_16_S_64_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_32_S_80_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_128_32_S_80_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_32_S_128_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_128_32_S_128_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_32_S_16_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_128_16_S_16_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_32_S_32_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_128_16_S_32_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_32_S_40_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_128_16_S_40_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_32_S_64_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_128_16_S_64_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_32_S_80_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_128_32_S_80_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_32_S_128_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_128_32_S_128_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_32_S_16_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_128_16_S_16_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_32_S_32_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_128_16_S_32_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_32_S_40_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_128_16_S_40_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_32_S_64_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_128_16_S_64_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_32_S_80_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_128_32_S_80_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_32_S_128_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_128_32_S_128_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm89_cu_cubin[];

extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_64_S_16_sm75_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_64_S_32_sm75_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_64_S_40_sm75_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_64_S_64_sm75_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_32_S_80_sm75_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_32_S_128_sm75_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm75_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm75_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_32_S_16_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_128_16_S_16_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_32_S_32_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_128_16_S_32_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_32_S_40_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_128_16_S_40_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_32_S_64_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_128_16_S_64_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_32_S_80_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_128_32_S_80_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_32_S_128_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_128_32_S_128_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_32_S_16_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_128_16_S_16_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_32_S_32_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_128_16_S_32_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_32_S_40_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_128_16_S_40_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_32_S_64_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_128_16_S_64_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_32_S_80_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_128_32_S_80_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_32_S_128_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_128_32_S_128_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_32_S_16_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_128_16_S_16_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_32_S_32_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_128_16_S_32_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_32_S_40_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_128_16_S_40_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_32_S_64_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_128_16_S_64_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_32_S_80_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_128_32_S_80_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_32_S_128_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_128_32_S_128_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm89_cu_cubin_len;

constexpr int32_t S{0};

#if !(defined(MNN_CUDA_ENABLE_SM75) || defined(MNN_CUDA_ENABLE_SM80) || defined(MNN_CUDA_ENABLE_SM86) || defined(MNN_CUDA_ENABLE_SM89))
#error This file can only be included one of sm 75, 80, 86 or 89 are defined.
#endif
static const struct FusedMultiHeadFlashAttentionKernelMetaInfoV2
{
    MHFADataType mDataType;
    int32_t mS;
    int32_t mQStep;
    int32_t mKVStep;
    int32_t mD;
    int32_t mSM;
    unsigned char const* mCubin;
    uint32_t mCubinSize;
    char const* mFuncName;
    int32_t mSharedMemBytes;
    int32_t mThreadsPerCTA;
    int32_t mUnrollStep;
    bool mInterleaved;
} sMhaKernelMetaInfos[] = {
#if defined(MNN_CUDA_ENABLE_SM75)
{ DATA_TYPE_FP16, S, 64, 64, 16, kSM_75,  cubin_fmha_v2_flash_attention_fp16_64_64_S_16_sm75_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_64_S_16_sm75_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_64_S_16_sm75_kernel", 6144, 128, 0, false },
{ DATA_TYPE_FP16, S, 64, 64, 16, kSM_75,  cubin_fmha_v2_flash_attention_fp16_64_64_S_16_sm75_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_64_S_16_sm75_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_64_S_16_sm75_kernel_nl", 6144, 128, 64, false },
{ DATA_TYPE_FP16, S, 64, 64, 32, kSM_75,  cubin_fmha_v2_flash_attention_fp16_64_64_S_32_sm75_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_64_S_32_sm75_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_64_S_32_sm75_kernel", 12288, 128, 0, false },
{ DATA_TYPE_FP16, S, 64, 64, 32, kSM_75,  cubin_fmha_v2_flash_attention_fp16_64_64_S_32_sm75_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_64_S_32_sm75_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_64_S_32_sm75_kernel_nl", 12288, 128, 64, false },
{ DATA_TYPE_FP16, S, 64, 64, 40, kSM_75,  cubin_fmha_v2_flash_attention_fp16_64_64_S_40_sm75_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_64_S_40_sm75_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_64_S_40_sm75_kernel", 24576, 128, 0, false },
{ DATA_TYPE_FP16, S, 64, 64, 40, kSM_75,  cubin_fmha_v2_flash_attention_fp16_64_64_S_40_sm75_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_64_S_40_sm75_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_64_S_40_sm75_kernel_nl", 24576, 128, 64, false },
{ DATA_TYPE_FP16, S, 64, 64, 64, kSM_75,  cubin_fmha_v2_flash_attention_fp16_64_64_S_64_sm75_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_64_S_64_sm75_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_64_S_64_sm75_kernel", 24576, 128, 0, false },
{ DATA_TYPE_FP16, S, 64, 64, 64, kSM_75,  cubin_fmha_v2_flash_attention_fp16_64_64_S_64_sm75_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_64_S_64_sm75_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_64_S_64_sm75_kernel_nl", 24576, 128, 64, false },
{ DATA_TYPE_FP16, S, 64, 32, 80, kSM_75,  cubin_fmha_v2_flash_attention_fp16_64_32_S_80_sm75_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_80_sm75_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_80_sm75_kernel", 32768, 128, 0, false },
{ DATA_TYPE_FP16, S, 64, 32, 80, kSM_75,  cubin_fmha_v2_flash_attention_fp16_64_32_S_80_sm75_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_80_sm75_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_80_sm75_kernel_nl", 32768, 128, 64, false },
{ DATA_TYPE_FP16, S, 64, 32, 128, kSM_75,  cubin_fmha_v2_flash_attention_fp16_64_32_S_128_sm75_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_128_sm75_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_128_sm75_kernel", 32768, 128, 0, false },
{ DATA_TYPE_FP16, S, 64, 32, 128, kSM_75,  cubin_fmha_v2_flash_attention_fp16_64_32_S_128_sm75_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_128_sm75_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_128_sm75_kernel_nl", 32768, 128, 64, false },
{ DATA_TYPE_FP16, S, 64, 16, 160, kSM_75,  cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm75_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm75_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_16_S_160_sm75_kernel", 65536, 128, 0, false },
{ DATA_TYPE_FP16, S, 64, 16, 160, kSM_75,  cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm75_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm75_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_16_S_160_sm75_kernel_nl", 65536, 128, 64, false },
{ DATA_TYPE_FP16, S, 64, 16, 256, kSM_75,  cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm75_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm75_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_16_S_256_sm75_kernel", 65536, 128, 0, false },
{ DATA_TYPE_FP16, S, 64, 16, 256, kSM_75,  cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm75_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm75_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_16_S_256_sm75_kernel_nl", 65536, 128, 64, false },
#endif // defined(MNN_CUDA_ENABLE_SM75)
#if defined(MNN_CUDA_ENABLE_SM80)
{ DATA_TYPE_FP16, S, 64, 32, 16, kSM_80,  cubin_fmha_v2_flash_attention_fp16_64_32_S_16_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_16_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_16_sm80_kernel", 8192, 128, 0, false },
{ DATA_TYPE_FP16, S, 64, 32, 16, kSM_80,  cubin_fmha_v2_flash_attention_fp16_64_32_S_16_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_16_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_16_sm80_kernel_nl", 8192, 128, 64, false },
{ DATA_TYPE_FP16, S, 128, 16, 16, kSM_80,  cubin_fmha_v2_flash_attention_fp16_128_16_S_16_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_16_S_16_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_16_S_16_sm80_kernel", 12288, 128, 0, false },
{ DATA_TYPE_FP16, S, 128, 16, 16, kSM_80,  cubin_fmha_v2_flash_attention_fp16_128_16_S_16_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_16_S_16_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_16_S_16_sm80_kernel_nl", 12288, 128, 128, false },
{ DATA_TYPE_FP16, S, 64, 32, 32, kSM_80,  cubin_fmha_v2_flash_attention_fp16_64_32_S_32_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_32_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_32_sm80_kernel", 12288, 128, 0, false },
{ DATA_TYPE_FP16, S, 64, 32, 32, kSM_80,  cubin_fmha_v2_flash_attention_fp16_64_32_S_32_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_32_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_32_sm80_kernel_nl", 12288, 128, 64, false },
{ DATA_TYPE_FP16, S, 128, 16, 32, kSM_80,  cubin_fmha_v2_flash_attention_fp16_128_16_S_32_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_16_S_32_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_16_S_32_sm80_kernel", 20480, 128, 0, false },
{ DATA_TYPE_FP16, S, 128, 16, 32, kSM_80,  cubin_fmha_v2_flash_attention_fp16_128_16_S_32_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_16_S_32_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_16_S_32_sm80_kernel_nl", 20480, 128, 128, false },
{ DATA_TYPE_FP16, S, 64, 32, 40, kSM_80,  cubin_fmha_v2_flash_attention_fp16_64_32_S_40_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_40_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_40_sm80_kernel", 24576, 128, 0, false },
{ DATA_TYPE_FP16, S, 64, 32, 40, kSM_80,  cubin_fmha_v2_flash_attention_fp16_64_32_S_40_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_40_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_40_sm80_kernel_nl", 24576, 128, 64, false },
{ DATA_TYPE_FP16, S, 128, 16, 40, kSM_80,  cubin_fmha_v2_flash_attention_fp16_128_16_S_40_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_16_S_40_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_16_S_40_sm80_kernel", 40960, 128, 0, false },
{ DATA_TYPE_FP16, S, 128, 16, 40, kSM_80,  cubin_fmha_v2_flash_attention_fp16_128_16_S_40_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_16_S_40_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_16_S_40_sm80_kernel_nl", 40960, 128, 128, false },
{ DATA_TYPE_FP16, S, 64, 32, 64, kSM_80,  cubin_fmha_v2_flash_attention_fp16_64_32_S_64_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_64_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_64_sm80_kernel", 24576, 128, 0, false },
{ DATA_TYPE_FP16, S, 64, 32, 64, kSM_80,  cubin_fmha_v2_flash_attention_fp16_64_32_S_64_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_64_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_64_sm80_kernel_nl", 24576, 128, 64, false },
{ DATA_TYPE_FP16, S, 128, 16, 64, kSM_80,  cubin_fmha_v2_flash_attention_fp16_128_16_S_64_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_16_S_64_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_16_S_64_sm80_kernel", 40960, 128, 0, false },
{ DATA_TYPE_FP16, S, 128, 16, 64, kSM_80,  cubin_fmha_v2_flash_attention_fp16_128_16_S_64_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_16_S_64_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_16_S_64_sm80_kernel_nl", 40960, 128, 128, false },
{ DATA_TYPE_FP16, S, 64, 32, 80, kSM_80,  cubin_fmha_v2_flash_attention_fp16_64_32_S_80_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_80_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_80_sm80_kernel", 49152, 128, 0, false },
{ DATA_TYPE_FP16, S, 64, 32, 80, kSM_80,  cubin_fmha_v2_flash_attention_fp16_64_32_S_80_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_80_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_80_sm80_kernel_nl", 49152, 128, 64, false },
{ DATA_TYPE_FP16, S, 128, 32, 80, kSM_80,  cubin_fmha_v2_flash_attention_fp16_128_32_S_80_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_32_S_80_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_32_S_80_sm80_kernel", 81920, 128, 0, false },
{ DATA_TYPE_FP16, S, 128, 32, 80, kSM_80,  cubin_fmha_v2_flash_attention_fp16_128_32_S_80_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_32_S_80_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_32_S_80_sm80_kernel_nl", 81920, 128, 128, false },
{ DATA_TYPE_FP16, S, 64, 32, 128, kSM_80,  cubin_fmha_v2_flash_attention_fp16_64_32_S_128_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_128_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_128_sm80_kernel", 49152, 128, 0, false },
{ DATA_TYPE_FP16, S, 64, 32, 128, kSM_80,  cubin_fmha_v2_flash_attention_fp16_64_32_S_128_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_128_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_128_sm80_kernel_nl", 49152, 128, 64, false },
{ DATA_TYPE_FP16, S, 128, 32, 128, kSM_80,  cubin_fmha_v2_flash_attention_fp16_128_32_S_128_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_32_S_128_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_32_S_128_sm80_kernel", 81920, 128, 0, false },
{ DATA_TYPE_FP16, S, 128, 32, 128, kSM_80,  cubin_fmha_v2_flash_attention_fp16_128_32_S_128_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_32_S_128_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_32_S_128_sm80_kernel_nl", 81920, 128, 128, false },
{ DATA_TYPE_FP16, S, 64, 16, 160, kSM_80,  cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_16_S_160_sm80_kernel", 98304, 128, 0, false },
{ DATA_TYPE_FP16, S, 64, 16, 160, kSM_80,  cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_16_S_160_sm80_kernel_nl", 98304, 128, 64, false },
{ DATA_TYPE_FP16, S, 64, 16, 256, kSM_80,  cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_16_S_256_sm80_kernel", 98304, 128, 0, false },
{ DATA_TYPE_FP16, S, 64, 16, 256, kSM_80,  cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_16_S_256_sm80_kernel_nl", 98304, 128, 64, false },
#endif // defined(MNN_CUDA_ENABLE_SM80)
#if defined(MNN_CUDA_ENABLE_SM86)
{ DATA_TYPE_FP16, S, 64, 32, 16, kSM_86,  cubin_fmha_v2_flash_attention_fp16_64_32_S_16_sm86_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_16_sm86_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_16_sm86_kernel", 8192, 128, 0, false },
{ DATA_TYPE_FP16, S, 64, 32, 16, kSM_86,  cubin_fmha_v2_flash_attention_fp16_64_32_S_16_sm86_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_16_sm86_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_16_sm86_kernel_nl", 8192, 128, 64, false },
{ DATA_TYPE_FP16, S, 128, 16, 16, kSM_86,  cubin_fmha_v2_flash_attention_fp16_128_16_S_16_sm86_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_16_S_16_sm86_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_16_S_16_sm86_kernel", 12288, 128, 0, false },
{ DATA_TYPE_FP16, S, 128, 16, 16, kSM_86,  cubin_fmha_v2_flash_attention_fp16_128_16_S_16_sm86_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_16_S_16_sm86_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_16_S_16_sm86_kernel_nl", 12288, 128, 128, false },
{ DATA_TYPE_FP16, S, 64, 32, 32, kSM_86,  cubin_fmha_v2_flash_attention_fp16_64_32_S_32_sm86_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_32_sm86_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_32_sm86_kernel", 12288, 128, 0, false },
{ DATA_TYPE_FP16, S, 64, 32, 32, kSM_86,  cubin_fmha_v2_flash_attention_fp16_64_32_S_32_sm86_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_32_sm86_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_32_sm86_kernel_nl", 12288, 128, 64, false },
{ DATA_TYPE_FP16, S, 128, 16, 32, kSM_86,  cubin_fmha_v2_flash_attention_fp16_128_16_S_32_sm86_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_16_S_32_sm86_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_16_S_32_sm86_kernel", 20480, 128, 0, false },
{ DATA_TYPE_FP16, S, 128, 16, 32, kSM_86,  cubin_fmha_v2_flash_attention_fp16_128_16_S_32_sm86_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_16_S_32_sm86_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_16_S_32_sm86_kernel_nl", 20480, 128, 128, false },
{ DATA_TYPE_FP16, S, 64, 32, 40, kSM_86,  cubin_fmha_v2_flash_attention_fp16_64_32_S_40_sm86_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_40_sm86_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_40_sm86_kernel", 24576, 128, 0, false },
{ DATA_TYPE_FP16, S, 64, 32, 40, kSM_86,  cubin_fmha_v2_flash_attention_fp16_64_32_S_40_sm86_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_40_sm86_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_40_sm86_kernel_nl", 24576, 128, 64, false },
{ DATA_TYPE_FP16, S, 128, 16, 40, kSM_86,  cubin_fmha_v2_flash_attention_fp16_128_16_S_40_sm86_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_16_S_40_sm86_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_16_S_40_sm86_kernel", 40960, 128, 0, false },
{ DATA_TYPE_FP16, S, 128, 16, 40, kSM_86,  cubin_fmha_v2_flash_attention_fp16_128_16_S_40_sm86_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_16_S_40_sm86_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_16_S_40_sm86_kernel_nl", 40960, 128, 128, false },
{ DATA_TYPE_FP16, S, 64, 32, 64, kSM_86,  cubin_fmha_v2_flash_attention_fp16_64_32_S_64_sm86_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_64_sm86_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_64_sm86_kernel", 24576, 128, 0, false },
{ DATA_TYPE_FP16, S, 64, 32, 64, kSM_86,  cubin_fmha_v2_flash_attention_fp16_64_32_S_64_sm86_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_64_sm86_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_64_sm86_kernel_nl", 24576, 128, 64, false },
{ DATA_TYPE_FP16, S, 128, 16, 64, kSM_86,  cubin_fmha_v2_flash_attention_fp16_128_16_S_64_sm86_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_16_S_64_sm86_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_16_S_64_sm86_kernel", 40960, 128, 0, false },
{ DATA_TYPE_FP16, S, 128, 16, 64, kSM_86,  cubin_fmha_v2_flash_attention_fp16_128_16_S_64_sm86_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_16_S_64_sm86_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_16_S_64_sm86_kernel_nl", 40960, 128, 128, false },
{ DATA_TYPE_FP16, S, 64, 32, 80, kSM_86,  cubin_fmha_v2_flash_attention_fp16_64_32_S_80_sm86_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_80_sm86_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_80_sm86_kernel", 49152, 128, 0, false },
{ DATA_TYPE_FP16, S, 64, 32, 80, kSM_86,  cubin_fmha_v2_flash_attention_fp16_64_32_S_80_sm86_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_80_sm86_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_80_sm86_kernel_nl", 49152, 128, 64, false },
{ DATA_TYPE_FP16, S, 128, 32, 80, kSM_86,  cubin_fmha_v2_flash_attention_fp16_128_32_S_80_sm86_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_32_S_80_sm86_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_32_S_80_sm86_kernel", 81920, 128, 0, false },
{ DATA_TYPE_FP16, S, 128, 32, 80, kSM_86,  cubin_fmha_v2_flash_attention_fp16_128_32_S_80_sm86_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_32_S_80_sm86_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_32_S_80_sm86_kernel_nl", 81920, 128, 128, false },
{ DATA_TYPE_FP16, S, 64, 32, 128, kSM_86,  cubin_fmha_v2_flash_attention_fp16_64_32_S_128_sm86_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_128_sm86_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_128_sm86_kernel", 49152, 128, 0, false },
{ DATA_TYPE_FP16, S, 64, 32, 128, kSM_86,  cubin_fmha_v2_flash_attention_fp16_64_32_S_128_sm86_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_128_sm86_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_128_sm86_kernel_nl", 49152, 128, 64, false },
{ DATA_TYPE_FP16, S, 128, 32, 128, kSM_86,  cubin_fmha_v2_flash_attention_fp16_128_32_S_128_sm86_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_32_S_128_sm86_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_32_S_128_sm86_kernel", 81920, 128, 0, false },
{ DATA_TYPE_FP16, S, 128, 32, 128, kSM_86,  cubin_fmha_v2_flash_attention_fp16_128_32_S_128_sm86_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_32_S_128_sm86_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_32_S_128_sm86_kernel_nl", 81920, 128, 128, false },
{ DATA_TYPE_FP16, S, 64, 16, 160, kSM_86,  cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm86_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm86_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_16_S_160_sm86_kernel", 98304, 128, 0, false },
{ DATA_TYPE_FP16, S, 64, 16, 160, kSM_86,  cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm86_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm86_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_16_S_160_sm86_kernel_nl", 98304, 128, 64, false },
{ DATA_TYPE_FP16, S, 64, 16, 256, kSM_86,  cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm86_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm86_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_16_S_256_sm86_kernel", 98304, 128, 0, false },
{ DATA_TYPE_FP16, S, 64, 16, 256, kSM_86,  cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm86_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm86_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_16_S_256_sm86_kernel_nl", 98304, 128, 64, false },
#endif // defined(MNN_CUDA_ENABLE_SM86)
#if defined(MNN_CUDA_ENABLE_SM89)
{ DATA_TYPE_FP16, S, 64, 32, 16, kSM_89,  cubin_fmha_v2_flash_attention_fp16_64_32_S_16_sm89_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_16_sm89_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_16_sm89_kernel", 8192, 128, 0, false },
{ DATA_TYPE_FP16, S, 64, 32, 16, kSM_89,  cubin_fmha_v2_flash_attention_fp16_64_32_S_16_sm89_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_16_sm89_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_16_sm89_kernel_nl", 8192, 128, 64, false },
{ DATA_TYPE_FP16, S, 128, 16, 16, kSM_89,  cubin_fmha_v2_flash_attention_fp16_128_16_S_16_sm89_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_16_S_16_sm89_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_16_S_16_sm89_kernel", 12288, 128, 0, false },
{ DATA_TYPE_FP16, S, 128, 16, 16, kSM_89,  cubin_fmha_v2_flash_attention_fp16_128_16_S_16_sm89_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_16_S_16_sm89_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_16_S_16_sm89_kernel_nl", 12288, 128, 128, false },
{ DATA_TYPE_FP16, S, 64, 32, 32, kSM_89,  cubin_fmha_v2_flash_attention_fp16_64_32_S_32_sm89_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_32_sm89_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_32_sm89_kernel", 12288, 128, 0, false },
{ DATA_TYPE_FP16, S, 64, 32, 32, kSM_89,  cubin_fmha_v2_flash_attention_fp16_64_32_S_32_sm89_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_32_sm89_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_32_sm89_kernel_nl", 12288, 128, 64, false },
{ DATA_TYPE_FP16, S, 128, 16, 32, kSM_89,  cubin_fmha_v2_flash_attention_fp16_128_16_S_32_sm89_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_16_S_32_sm89_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_16_S_32_sm89_kernel", 20480, 128, 0, false },
{ DATA_TYPE_FP16, S, 128, 16, 32, kSM_89,  cubin_fmha_v2_flash_attention_fp16_128_16_S_32_sm89_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_16_S_32_sm89_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_16_S_32_sm89_kernel_nl", 20480, 128, 128, false },
{ DATA_TYPE_FP16, S, 64, 32, 40, kSM_89,  cubin_fmha_v2_flash_attention_fp16_64_32_S_40_sm89_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_40_sm89_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_40_sm89_kernel", 24576, 128, 0, false },
{ DATA_TYPE_FP16, S, 64, 32, 40, kSM_89,  cubin_fmha_v2_flash_attention_fp16_64_32_S_40_sm89_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_40_sm89_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_40_sm89_kernel_nl", 24576, 128, 64, false },
{ DATA_TYPE_FP16, S, 128, 16, 40, kSM_89,  cubin_fmha_v2_flash_attention_fp16_128_16_S_40_sm89_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_16_S_40_sm89_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_16_S_40_sm89_kernel", 40960, 128, 0, false },
{ DATA_TYPE_FP16, S, 128, 16, 40, kSM_89,  cubin_fmha_v2_flash_attention_fp16_128_16_S_40_sm89_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_16_S_40_sm89_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_16_S_40_sm89_kernel_nl", 40960, 128, 128, false },
{ DATA_TYPE_FP16, S, 64, 32, 64, kSM_89,  cubin_fmha_v2_flash_attention_fp16_64_32_S_64_sm89_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_64_sm89_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_64_sm89_kernel", 24576, 128, 0, false },
{ DATA_TYPE_FP16, S, 64, 32, 64, kSM_89,  cubin_fmha_v2_flash_attention_fp16_64_32_S_64_sm89_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_64_sm89_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_64_sm89_kernel_nl", 24576, 128, 64, false },
{ DATA_TYPE_FP16, S, 128, 16, 64, kSM_89,  cubin_fmha_v2_flash_attention_fp16_128_16_S_64_sm89_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_16_S_64_sm89_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_16_S_64_sm89_kernel", 40960, 128, 0, false },
{ DATA_TYPE_FP16, S, 128, 16, 64, kSM_89,  cubin_fmha_v2_flash_attention_fp16_128_16_S_64_sm89_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_16_S_64_sm89_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_16_S_64_sm89_kernel_nl", 40960, 128, 128, false },
{ DATA_TYPE_FP16, S, 64, 32, 80, kSM_89,  cubin_fmha_v2_flash_attention_fp16_64_32_S_80_sm89_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_80_sm89_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_80_sm89_kernel", 49152, 128, 0, false },
{ DATA_TYPE_FP16, S, 64, 32, 80, kSM_89,  cubin_fmha_v2_flash_attention_fp16_64_32_S_80_sm89_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_80_sm89_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_80_sm89_kernel_nl", 49152, 128, 64, false },
{ DATA_TYPE_FP16, S, 128, 32, 80, kSM_89,  cubin_fmha_v2_flash_attention_fp16_128_32_S_80_sm89_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_32_S_80_sm89_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_32_S_80_sm89_kernel", 81920, 128, 0, false },
{ DATA_TYPE_FP16, S, 128, 32, 80, kSM_89,  cubin_fmha_v2_flash_attention_fp16_128_32_S_80_sm89_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_32_S_80_sm89_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_32_S_80_sm89_kernel_nl", 81920, 128, 128, false },
{ DATA_TYPE_FP16, S, 64, 32, 128, kSM_89,  cubin_fmha_v2_flash_attention_fp16_64_32_S_128_sm89_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_128_sm89_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_128_sm89_kernel", 49152, 128, 0, false },
{ DATA_TYPE_FP16, S, 64, 32, 128, kSM_89,  cubin_fmha_v2_flash_attention_fp16_64_32_S_128_sm89_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_32_S_128_sm89_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_32_S_128_sm89_kernel_nl", 49152, 128, 64, false },
{ DATA_TYPE_FP16, S, 128, 32, 128, kSM_89,  cubin_fmha_v2_flash_attention_fp16_128_32_S_128_sm89_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_32_S_128_sm89_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_32_S_128_sm89_kernel", 81920, 128, 0, false },
{ DATA_TYPE_FP16, S, 128, 32, 128, kSM_89,  cubin_fmha_v2_flash_attention_fp16_128_32_S_128_sm89_cu_cubin, cubin_fmha_v2_flash_attention_fp16_128_32_S_128_sm89_cu_cubin_len, "fmha_v2_flash_attention_fp16_128_32_S_128_sm89_kernel_nl", 81920, 128, 128, false },
{ DATA_TYPE_FP16, S, 64, 16, 160, kSM_89,  cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm89_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm89_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_16_S_160_sm89_kernel", 98304, 128, 0, false },
{ DATA_TYPE_FP16, S, 64, 16, 160, kSM_89,  cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm89_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm89_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_16_S_160_sm89_kernel_nl", 98304, 128, 64, false },
{ DATA_TYPE_FP16, S, 64, 16, 256, kSM_89,  cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm89_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm89_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_16_S_256_sm89_kernel", 98304, 128, 0, false },
{ DATA_TYPE_FP16, S, 64, 16, 256, kSM_89,  cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm89_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm89_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_16_S_256_sm89_kernel_nl", 98304, 128, 64, false }
#endif // defined(MNN_CUDA_ENABLE_SM89)
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Do not modify this, it is integrated from function set_params, file src/fused_multihead_attention.cpp in
// fmha-flash-attention.
////////////////////////////////////////////////////////////////////////////////////////////////////
static Fused_multihead_flash_attention_params_v2 getMHFAParams(
    // types
    MHFADataType data_type, MHFADataType acc_type,
    // sizes
    int32_t b, int32_t s, int32_t h, int32_t d, int32_t total,
    // device pointers
    void const* qkv_packed_d, void* cu_seqlens_d, void* o_packed_d, void* p_d, void* s_d,
    // scale factors
    float scale_bmm1, float scale_softmax, float scale_bmm2,
    // flags
    bool interleaved, bool ignore_b1opt, bool force_unroll, bool use_int8_scale_max)
{
    Fused_multihead_flash_attention_params_v2 params{};

    // Set the pointers.
    params.qkv_ptr = const_cast<void*>(qkv_packed_d);
    params.qkv_stride_in_bytes = get_size_in_bytes(h * 3 * d, data_type);
    params.o_ptr = o_packed_d;
    params.o_stride_in_bytes = get_size_in_bytes(h * d, data_type);

    if (interleaved)
    {
        params.qkv_stride_in_bytes = total;
        params.o_stride_in_bytes = total;
    }

    params.cu_seqlens = static_cast<int*>(cu_seqlens_d);

#if defined(STORE_P)
    MNN_PRINT("defined STORE_P\n");
    params.p_ptr = p_d;
    params.p_stride_in_bytes = get_size_in_bytes(b * h * s, acc_type);
#endif // defined(STORE_P)

#if defined(STORE_S)
    MNN_PRINT("defined STORE_S\n");
    params.s_ptr = s_d;
    params.s_stride_in_bytes = get_size_in_bytes(b * h * s, data_type);
#endif // defined(STORE_S)

    // Set the dimensions.
    params.b = b;
    params.h = h;
    params.s = s;
    params.d = d;

    // Set the different scale values.
    MHFADataType scale_type1 = data_type == DATA_TYPE_FP16 ? acc_type : DATA_TYPE_FP32;
    MHFADataType scale_type2 = data_type == DATA_TYPE_FP16 ? DATA_TYPE_FP16 : DATA_TYPE_FP32;

    set_alpha(params.scale_bmm1, scale_bmm1, scale_type1);
    set_alpha(params.scale_softmax, scale_softmax, scale_type1);
    set_alpha(params.scale_bmm2, scale_bmm2, scale_type2);

    // Set flags
    params.interleaved = interleaved;
    params.ignore_b1opt = ignore_b1opt;
    params.force_unroll = force_unroll;
    params.use_int8_scale_max = use_int8_scale_max;

    // Do we enable the trick to replace I2F with FP math in the 2nd GEMM?
    if (data_type == DATA_TYPE_INT8)
    {
        params.enable_i2f_trick
            = -double(1 << 22) * double(scale_bmm2) <= -128.F && double(1 << 22) * double(scale_bmm2) >= 127.F;
    }
    return params;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
class FusedMultiHeadFlashAttentionKernel
    : public TSharedCubinKernel<FusedMultiHeadFlashAttentionKernelMetaInfoV2, Fused_multihead_flash_attention_params_v2>
{
public:
    FusedMultiHeadFlashAttentionKernel(FusedMultiHeadFlashAttentionKernelMetaInfoV2 const* pMetaStart,
        int32_t nMetaCount, MHFADataType type, int32_t sm)
        : TSharedCubinKernel<FusedMultiHeadFlashAttentionKernelMetaInfoV2, Fused_multihead_flash_attention_params_v2>(
            pMetaStart, nMetaCount, type, sm)
    {
    }

    uint64_t hashID(int32_t headsize, int32_t qStep, int32_t kvStep, bool interleaved, bool unroll) const
    {
        // we only have 30 bits room for head size
        MNN_ASSERT(headsize <= 0x3FFFFFFF);
        MNN_ASSERT(qStep <= 0xFFFF);
        MNN_ASSERT(kvStep <= 0xFFFF);
        return static_cast<uint64_t>(qStep << 16 | kvStep) << 32 | (headsize << 2) | (interleaved ? 2U : 0U)
            | (unroll ? 1U : 0U);
    }

    uint64_t hashID(Fused_multihead_flash_attention_params_v2 const& param) const
    {
        bool const isSmallBS = param.b * param.h < 64;
        bool const isSM75 = mSM == 75;
        int32_t qStep{64};
        int32_t kvStep{16};
        switch (param.d)
        {
        case 16:
        case 32:
        case 40:
        case 64:
            qStep = isSM75 ? 64 : (isSmallBS ? 64 : 128);
            kvStep = isSM75 ? 64 : (isSmallBS ? 32 : 16);
            break;
        case 80:
        case 128:
            qStep = isSM75 ? 64 : (isSmallBS ? 64 : 128);
            kvStep = isSM75 ? 32 : (isSmallBS ? 32 : 32);
            break;
        default: break;
        }
        return hashID(param.d, qStep, kvStep, param.interleaved, param.force_unroll);
    }

    uint64_t hashID(KernelMeta const& kernelMeta) const
    {
        return hashID(
            kernelMeta.mD, kernelMeta.mQStep, kernelMeta.mKVStep, kernelMeta.mInterleaved, kernelMeta.mUnrollStep > 0);
    }
};

using FusedMHAFlashKernelFactory = TSharedCubinKernelFactory<FusedMultiHeadFlashAttentionKernel>;

inline FusedMultiHeadFlashAttentionKernel const* getFMHAFlashCubinKernels(MHFADataType type, int32_t sm)
{
    return FusedMHAFlashKernelFactory::Get().getCubinKernels(
        sMhaKernelMetaInfos, sizeof(sMhaKernelMetaInfos) / sizeof(sMhaKernelMetaInfos[0]), type, sm);
}

} // namespace MNN
} // namespace CUDA

#endif // FMHA_FLASH_ATTENTION_H
