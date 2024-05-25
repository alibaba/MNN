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

#ifndef FMHA_CROSS_ATTENTION_H
#define FMHA_CROSS_ATTENTION_H

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
// Do not modify this, it is integrated from src/fused_multihead_attention_utils.h in fmha_v2.
////////////////////////////////////////////////////////////////////////////////////////////////////
static void set_alpha(uint32_t& alpha, float norm, MNN::CUDA::MHCADataType dtype)
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

static int64_t get_size_in_bytes(size_t n, MNN::CUDA::MHCADataType dtype)
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
// Do not modify this, it is integrated from src/fused_multihead_attention_demo_bert_params.h in fmha_v2.
////////////////////////////////////////////////////////////////////////////////////////////////////
struct Gmem_params
{
    // The matrix.
    void* ptr;

    // The stride between rows of the Q, K and V matrices.
    int64_t stride_in_bytes;

    // The number of heads
    int32_t h;

    // Hidden dim per head
    int32_t d;

    // array of length b+1 holding prefix sum of actual sequence lenghts.
    int32_t* cu_seqlens;
};

struct Fused_multihead_attention_params_mhca
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

    // Sequence length of Q
    int32_t s_q;
    int32_t d_padded;

    Gmem_params gmem_q_params;
    Gmem_params gmem_kv_params;

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

        s_q = 0;
        d_padded = 0;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Do not modify this, it is integrated from generated/fmha_cubin.h in fmha_v2.
////////////////////////////////////////////////////////////////////////////////////////////////////
extern unsigned char cubin_fmha_mhca_fp16_128_64_sm75_cu_cubin[];
extern unsigned char cubin_fmha_mhca_fp16_128_64_sm80_cu_cubin[];
extern unsigned char cubin_fmha_mhca_fp16_128_64_sm86_cu_cubin[];
extern unsigned char cubin_fmha_mhca_fp16_128_64_sm89_cu_cubin[];

extern unsigned char cubin_fmha_mhca_fp16_128_128_sm75_cu_cubin[];
extern unsigned char cubin_fmha_mhca_fp16_128_128_sm80_cu_cubin[];
extern unsigned char cubin_fmha_mhca_fp16_128_128_sm86_cu_cubin[];
extern unsigned char cubin_fmha_mhca_fp16_128_128_sm89_cu_cubin[];

// No support for D=160 on Turing.
extern unsigned char cubin_fmha_mhca_fp16_128_256_sm80_cu_cubin[];
extern unsigned char cubin_fmha_mhca_fp16_128_256_sm86_cu_cubin[];
extern unsigned char cubin_fmha_mhca_fp16_128_256_sm89_cu_cubin[];

extern uint32_t cubin_fmha_mhca_fp16_128_64_sm75_cu_cubin_len;
extern uint32_t cubin_fmha_mhca_fp16_128_64_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_mhca_fp16_128_64_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_mhca_fp16_128_64_sm89_cu_cubin_len;

extern uint32_t cubin_fmha_mhca_fp16_128_128_sm75_cu_cubin_len;
extern uint32_t cubin_fmha_mhca_fp16_128_128_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_mhca_fp16_128_128_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_mhca_fp16_128_128_sm89_cu_cubin_len;

// No support for D=160 on Turing.
extern uint32_t cubin_fmha_mhca_fp16_128_256_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_mhca_fp16_128_256_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_mhca_fp16_128_256_sm89_cu_cubin_len;

#if !(defined(MNN_CUDA_ENABLE_SM75) || defined(MNN_CUDA_ENABLE_SM80) || defined(MNN_CUDA_ENABLE_SM86) || defined(MNN_CUDA_ENABLE_SM89))
#error This file can only be included one of sm 75, 80, 86 or 89 are defined.
#endif
static const struct FusedMultiHeadCrossAttentionKernelMetaInfoV2
{
    MHCADataType mDataType;
    int32_t mS;
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
    { DATA_TYPE_FP16, 128, 64, kSM_75,  cubin_fmha_mhca_fp16_128_64_sm75_cu_cubin, cubin_fmha_mhca_fp16_128_64_sm75_cu_cubin_len, "fmha_mhca_fp16_128_64_sm75_kernel", 40960, 128, 0, false },
    { DATA_TYPE_FP16, 128, 64, kSM_75,  cubin_fmha_mhca_fp16_128_64_sm75_cu_cubin, cubin_fmha_mhca_fp16_128_64_sm75_cu_cubin_len, "fmha_mhca_fp16_128_64_sm75_kernel_nl", 36864, 128, 32, false },
    { DATA_TYPE_FP16, 128, 128, kSM_75,  cubin_fmha_mhca_fp16_128_128_sm75_cu_cubin, cubin_fmha_mhca_fp16_128_128_sm75_cu_cubin_len, "fmha_mhca_fp16_128_128_sm75_kernel", 81920, 128, 0, false },
    { DATA_TYPE_FP16, 128, 128, kSM_75,  cubin_fmha_mhca_fp16_128_128_sm75_cu_cubin, cubin_fmha_mhca_fp16_128_128_sm75_cu_cubin_len, "fmha_mhca_fp16_128_128_sm75_kernel_nl", 40960, 128, 32, false },

    // No support for D=160 on Turing.
#endif 
#if defined(MNN_CUDA_ENABLE_SM80) 
    { DATA_TYPE_FP16, 128, 64,  kSM_80,  cubin_fmha_mhca_fp16_128_64_sm80_cu_cubin,  cubin_fmha_mhca_fp16_128_64_sm80_cu_cubin_len,  "fmha_mhca_fp16_128_64_sm80_kernel",      49152, 128,  0, false },
    { DATA_TYPE_FP16, 128, 64,  kSM_80,  cubin_fmha_mhca_fp16_128_64_sm80_cu_cubin,  cubin_fmha_mhca_fp16_128_64_sm80_cu_cubin_len,  "fmha_mhca_fp16_128_64_sm80_kernel_nl",   49152, 128, 64, false },
    { DATA_TYPE_FP16, 128, 128, kSM_80,  cubin_fmha_mhca_fp16_128_128_sm80_cu_cubin, cubin_fmha_mhca_fp16_128_128_sm80_cu_cubin_len, "fmha_mhca_fp16_128_128_sm80_kernel",     98304, 128,  0, false },
    { DATA_TYPE_FP16, 128, 128, kSM_80,  cubin_fmha_mhca_fp16_128_128_sm80_cu_cubin, cubin_fmha_mhca_fp16_128_128_sm80_cu_cubin_len, "fmha_mhca_fp16_128_128_sm80_kernel_nl",  81920, 128, 32, false },
    { DATA_TYPE_FP16, 128, 256, kSM_80,  cubin_fmha_mhca_fp16_128_256_sm80_cu_cubin, cubin_fmha_mhca_fp16_128_256_sm80_cu_cubin_len, "fmha_mhca_fp16_128_256_sm80_kernel",    163840, 256,  0, false },
    { DATA_TYPE_FP16, 128, 256, kSM_80,  cubin_fmha_mhca_fp16_128_256_sm80_cu_cubin, cubin_fmha_mhca_fp16_128_256_sm80_cu_cubin_len, "fmha_mhca_fp16_128_256_sm80_kernel_nl", 147456, 256, 16, false },
#endif 
#if defined(MNN_CUDA_ENABLE_SM86)
    { DATA_TYPE_FP16, 128, 64, kSM_86,  cubin_fmha_mhca_fp16_128_64_sm86_cu_cubin, cubin_fmha_mhca_fp16_128_64_sm86_cu_cubin_len, "fmha_mhca_fp16_128_64_sm86_kernel", 49152, 128, 0, false },
    { DATA_TYPE_FP16, 128, 64, kSM_86,  cubin_fmha_mhca_fp16_128_64_sm86_cu_cubin, cubin_fmha_mhca_fp16_128_64_sm86_cu_cubin_len, "fmha_mhca_fp16_128_64_sm86_kernel_nl", 49152, 128, 64, false },
    { DATA_TYPE_FP16, 128, 128, kSM_86,  cubin_fmha_mhca_fp16_128_128_sm86_cu_cubin, cubin_fmha_mhca_fp16_128_128_sm86_cu_cubin_len, "fmha_mhca_fp16_128_128_sm86_kernel", 98304, 128, 0, false },
    { DATA_TYPE_FP16, 128, 128, kSM_86,  cubin_fmha_mhca_fp16_128_128_sm86_cu_cubin, cubin_fmha_mhca_fp16_128_128_sm86_cu_cubin_len, "fmha_mhca_fp16_128_128_sm86_kernel_nl", 98304, 128, 64, false },
    { DATA_TYPE_FP16, 128, 256, kSM_86,  cubin_fmha_mhca_fp16_128_256_sm86_cu_cubin, cubin_fmha_mhca_fp16_128_256_sm86_cu_cubin_len, "fmha_mhca_fp16_128_256_sm86_kernel", 163840, 256, 0, false },
    { DATA_TYPE_FP16, 128, 256, kSM_86,  cubin_fmha_mhca_fp16_128_256_sm86_cu_cubin, cubin_fmha_mhca_fp16_128_256_sm86_cu_cubin_len, "fmha_mhca_fp16_128_256_sm86_kernel_nl", 81920, 256, 16, false },
#endif // defined(MNN_CUDA_ENABLE_SM89)
#if defined(MNN_CUDA_ENABLE_SM89)
    { DATA_TYPE_FP16, 128, 64, kSM_89,  cubin_fmha_mhca_fp16_128_64_sm89_cu_cubin, cubin_fmha_mhca_fp16_128_64_sm89_cu_cubin_len, "fmha_mhca_fp16_128_64_sm89_kernel", 49152, 128, 0, false },
    { DATA_TYPE_FP16, 128, 64, kSM_89,  cubin_fmha_mhca_fp16_128_64_sm89_cu_cubin, cubin_fmha_mhca_fp16_128_64_sm89_cu_cubin_len, "fmha_mhca_fp16_128_64_sm89_kernel_nl", 49152, 128, 64, false },
    { DATA_TYPE_FP16, 128, 128, kSM_89,  cubin_fmha_mhca_fp16_128_128_sm89_cu_cubin, cubin_fmha_mhca_fp16_128_128_sm89_cu_cubin_len, "fmha_mhca_fp16_128_128_sm89_kernel", 98304, 128, 0, false },
    { DATA_TYPE_FP16, 128, 128, kSM_89,  cubin_fmha_mhca_fp16_128_128_sm89_cu_cubin, cubin_fmha_mhca_fp16_128_128_sm89_cu_cubin_len, "fmha_mhca_fp16_128_128_sm89_kernel_nl", 81920, 128, 32, false },
    { DATA_TYPE_FP16, 128, 256, kSM_89,  cubin_fmha_mhca_fp16_128_256_sm89_cu_cubin, cubin_fmha_mhca_fp16_128_256_sm89_cu_cubin_len, "fmha_mhca_fp16_128_256_sm89_kernel", 163840, 256, 0, false },
    { DATA_TYPE_FP16, 128, 256, kSM_89,  cubin_fmha_mhca_fp16_128_256_sm89_cu_cubin, cubin_fmha_mhca_fp16_128_256_sm89_cu_cubin_len, "fmha_mhca_fp16_128_256_sm89_kernel_nl", 81920, 256, 16, false }
#endif // defined(MNN_CUDA_ENABLE_SM89)
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Do not modify this, it is integrated from function set_params, file src/fused_multihead_attention.cpp in fmha_v2.
////////////////////////////////////////////////////////////////////////////////////////////////////
static Fused_multihead_attention_params_mhca getMHCAParams(
    // types
    MHCADataType data_type, MHCADataType acc_type,
    // sizes
    int32_t b, int32_t s_q, int32_t s_kv, int32_t h, int32_t d, int32_t total,
    // device pointers
    void const* q_packed_d, void const* kv_packed_d, void* cu_seqlens_q_d, void* cu_seqlens_kv_d, void* o_packed_d,
    void* p_d, void* s_d,
    // scale factors
    float scale_bmm1, float scale_softmax, float scale_bmm2,
    // flags
    bool interleaved, bool ignore_b1opt, bool force_unroll, bool use_int8_scale_max, bool use_tma)
{
    Fused_multihead_attention_params_mhca params{};

    int32_t const d_padded = std::pow(2, std::ceil(std::log(d) / std::log(2)));

    // Set the pointers.
    params.o_ptr = o_packed_d;
    params.o_stride_in_bytes = get_size_in_bytes(h * d, data_type);

#if defined(STORE_P)
    params.p_ptr = p_d;
    params.p_stride_in_bytes = get_size_in_bytes(b * h * s_kv, acc_type);
#endif // defined(STORE_P)

#if defined(STORE_S)
    params.s_ptr = s_d;
    params.s_stride_in_bytes = get_size_in_bytes(b * h * s_kv, data_type);
#endif // defined(STORE_S)

    // Set the dimensions.
    params.b = b;
    params.h = h;
    params.s_q = s_q;
    params.s = s_kv;
    params.d = d;
    params.d_padded = d_padded;

    // Set the different scale values.
    MHCADataType scale_type1 = data_type == DATA_TYPE_FP16 ? acc_type : DATA_TYPE_FP32;
    MHCADataType scale_type2 = data_type == DATA_TYPE_FP16 ? DATA_TYPE_FP16 : DATA_TYPE_FP32;

    set_alpha(params.scale_bmm1, scale_bmm1, scale_type1);
    set_alpha(params.scale_softmax, scale_softmax, scale_type1);
    set_alpha(params.scale_bmm2, scale_bmm2, scale_type2);

    // Set the pointers.
    params.gmem_q_params.ptr = const_cast<void*>(q_packed_d);
    params.gmem_q_params.stride_in_bytes = get_size_in_bytes(h * d, data_type);
    params.gmem_q_params.h = h;
    params.gmem_q_params.d = d;
    params.gmem_q_params.cu_seqlens = static_cast<int32_t*>(cu_seqlens_q_d);

    params.gmem_kv_params.ptr = const_cast<void*>(kv_packed_d);
    params.gmem_kv_params.stride_in_bytes = get_size_in_bytes(h * 2 * d, data_type);
    params.gmem_kv_params.h = h;
    params.gmem_kv_params.d = d;
    params.gmem_kv_params.cu_seqlens = static_cast<int32_t*>(cu_seqlens_kv_d);

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
class FusedMultiHeadCrossAttentionKernel
    : public TSharedCubinKernel<FusedMultiHeadCrossAttentionKernelMetaInfoV2, Fused_multihead_attention_params_mhca>
{
public:
    FusedMultiHeadCrossAttentionKernel(FusedMultiHeadCrossAttentionKernelMetaInfoV2 const* pMetaStart,
        int32_t nMetaCount, MHCADataType type, int32_t sm)
        : TSharedCubinKernel<FusedMultiHeadCrossAttentionKernelMetaInfoV2, Fused_multihead_attention_params_mhca>(
            pMetaStart, nMetaCount, type, sm)
    {
    }

    uint64_t hashID(int32_t s, int32_t headsize, bool interleaved, bool unroll) const
    {
        // we only have 30 bits room for head size
        MNN_ASSERT(headsize <= 0x3FFFFFFF);
        return static_cast<uint64_t>(s) << 32 | (headsize << 2) | (interleaved ? 2U : 0U) | (unroll ? 1U : 0U);
    }

    uint64_t hashID(Fused_multihead_attention_params_mhca const& param) const
    {
        return hashID(param.s, param.d_padded, param.interleaved, param.force_unroll);
    }

    uint64_t hashID(KernelMeta const& kernelMeta) const
    {
        return hashID(kernelMeta.mS, kernelMeta.mD, kernelMeta.mInterleaved, kernelMeta.mUnrollStep > 0);
    }
};

using FusedMHACrossKernelFactory = TSharedCubinKernelFactory<FusedMultiHeadCrossAttentionKernel>;

inline FusedMultiHeadCrossAttentionKernel const* getFMHCACubinKernels(MHCADataType type, int32_t sm)
{
    return FusedMHACrossKernelFactory::Get().getCubinKernels(
        sMhaKernelMetaInfos, sizeof(sMhaKernelMetaInfos) / sizeof(sMhaKernelMetaInfos[0]), type, sm);
}

} // namespace CUDA
} // namespace MNN

#endif // FMHA_CROSS_ATTENTION_H
