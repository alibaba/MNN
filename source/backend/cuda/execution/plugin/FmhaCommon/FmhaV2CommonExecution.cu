//
//  FmhaCommonExecution.cpp
//  MNN
//
//  Created by MNN on 2024/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#include "FmhaCommonExecution.hpp"

namespace MNN {
namespace CUDA {

template <typename T>
__global__ void SPLIT_FusedQKV(const size_t count, const T* fused_qkv, T* ptr_q,
        T* ptr_k, T* ptr_v,
        int head_size
    ) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
        //[B, S, H, 3, D] -> [B, S, H, D]
        const int bsh = i / head_size;
        const int d = i % head_size;

        ptr_q[i] =  fused_qkv[(bsh * 3 + 0) * head_size + d];
        ptr_k[i] =  fused_qkv[(bsh * 3 + 1) * head_size + d];
        ptr_v[i] =  fused_qkv[(bsh * 3 + 2) * head_size + d];
    }
}

template <typename T>
__global__ void SPLIT_FusedKV(const size_t count, const T* fused_kv,
        T* ptr_k, T* ptr_v,
        int head_size
    ) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
        //[B, S, H, 2, D] -> [B, S, H, D]
        const int bsh = i / head_size;
        const int d = i % head_size;

        ptr_k[i] =  fused_kv[(bsh * 2 + 0) * head_size + d];
        ptr_v[i] =  fused_kv[(bsh * 2 + 1) * head_size + d];
    }
}

template <
int kQueriesPerBlock,
int kKeysPerBlock,
int kMaxK
>
int FmhaCommonExecution::run_attention(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    using Attention = AttentionKernel<
        cutlass::half_t,      // scalar_t
        cutlass::arch::Sm80,  // ArchTag
        true,                 // Memory is aligned
        kQueriesPerBlock,
        kKeysPerBlock,
        kMaxK,
        false,                // Supports dropout
        false                 // Supports bias
    >;

    typename Attention::Params p;
    // set parameters
    {
        // TODO : Split fused qkv [B, S, H, 3, D] --> [B, S, H, D]
        p.query_ptr = (cutlass::half_t *)mQ_Buffer;//inputs[0]->deviceId();
        p.key_ptr = (cutlass::half_t *)mK_Buffer;//inputs[0]->deviceId();
        p.value_ptr = (cutlass::half_t *)mV_Buffer;//inputs[0]->deviceId();
        p.logsumexp_ptr = nullptr; // Only needed for bw
        p.output_accum_ptr = nullptr;
        if (Attention::kNeedsOutputAccumulatorBuffer) {
            p.output_accum_ptr = (float *)mAcc_Buffer;
        }
        p.output_ptr = (cutlass::half_t *)outputs[0]->deviceId();

        // TODO: support arbitrary seq lengths
        // if (cu_seqlens_q.has_value()) {
        //   p.cu_seqlens_q_ptr = (int32_t*)cu_seqlens_q->data_ptr();
        //   p.cu_seqlens_k_ptr = (int32_t*)cu_seqlens_k->data_ptr();
        // }

        p.scale = 1.0f / sqrtf(mHeadSize);

        p.num_heads = mNumHeads;
        p.num_batches = mBatchSize;
        p.head_dim = mHeadSize;
        p.head_dim_value = mHeadSizeV;
        p.num_queries = mSeqLen;
        p.num_keys = mSeqLenKV;
        if (false/*options.causal*/) {
            p.custom_mask_type = Attention::CausalFromTopLeft;
        }

        // All tensors are in BMHK shapes
        p.q_strideH = mHeadSize;
        p.k_strideH = mHeadSize;
        p.v_strideH = mHeadSizeV;
        p.q_strideM = p.q_strideH * mNumHeads;
        p.k_strideM = p.k_strideH * mNumHeads;
        p.v_strideM = p.v_strideH * mNumHeads;
        p.q_strideB = p.q_strideM * mSeqLen;
        p.k_strideB = p.k_strideM * mSeqLenKV;
        p.v_strideB = p.v_strideM * mSeqLenKV;
        p.o_strideM = mHeadSizeV * mNumHeads;
    }
    // launch kernel :)
    constexpr auto kernel_fn = attention_kernel_batched_impl<Attention>;
    int smem_bytes = sizeof(typename Attention::SharedStorage);
    if (smem_bytes > 0xc000) {
        cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    }
    if (!Attention::check_supported(p)) {
        MNN_ERROR("Attention Kernel does not support these inputs\n");
        return -1;
    }
    kernel_fn<<<p.getBlocksGrid(), p.getThreadsGrid(), smem_bytes>>>(p);
    return 0;
}

FmhaCommonExecution::FmhaCommonExecution(const MNN::Op* op, Backend* backend) : Execution(backend) {
    if(op->type() == OpType_FmhaV2) {
        auto fmha_v2_param = op->main_as_FmhaV2Param();
        mNumHeads = fmha_v2_param->heads();
        mType = 0;
    } else if(op->type() == OpType_Fmhca) {
        auto fmhca_param = op->main_as_FmhcaParam();
        mNumHeads = fmhca_param->heads();
        mType = 1;
    }
}
ErrorCode FmhaCommonExecution::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto pool = static_cast<CUDABackend*>(backend())->getBufferPool();
    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();

    auto input = inputs[0];
    auto output = outputs[0];
    MNN_ASSERT(output->dimensions() == 3);

    mBatchSize = output->length(0);
    mSeqLen = output->length(1);
    mHeadSizeV = outputs[0]->length(2)/mNumHeads;
    mHeadSize = mHeadSizeV;
    mSeqLenKV = mSeqLen;
    if(mType == 1) {
        mSeqLenKV = inputs[1]->length(1);
        mHeadSize = inputs[0]->length(2)/mNumHeads;
    }
    
    mSM = runtime->compute_capability();
    MemChunk buffer_q;
    if(mType == 0) {
        buffer_q = pool->alloc(mBatchSize * mSeqLen * mHeadSize * mNumHeads * sizeof(half));
        mQ_Buffer = (void*)((uint8_t*)buffer_q.first + buffer_q.second);
    }
    auto buffer_k = pool->alloc(mBatchSize * mSeqLenKV * mHeadSize * mNumHeads * sizeof(half));
    mK_Buffer = (void*)((uint8_t*)buffer_k.first + buffer_k.second);
    auto buffer_v = pool->alloc(mBatchSize * mSeqLenKV * mHeadSizeV * mNumHeads * sizeof(half));
    mV_Buffer = (void*)((uint8_t*)buffer_v.first + buffer_v.second);
    // output size
    auto buffer_acc = pool->alloc(mBatchSize * mSeqLen * mHeadSizeV * mNumHeads * sizeof(float));
    mAcc_Buffer = (void*)((uint8_t*)buffer_acc.first + buffer_acc.second);
    
    if(mType == 0) {
        pool->free(buffer_q);
    }
    pool->free(buffer_k);
    pool->free(buffer_v);
    pool->free(buffer_acc);
    return NO_ERROR;
}

ErrorCode FmhaCommonExecution::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start FmhaCommonExecution onExecute...");
#endif

    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();

    size_t count = mBatchSize * mSeqLenKV * mHeadSizeV * mNumHeads;
    int block_num = runtime->blocks_num(count);
    int thread_num = runtime->threads_num();
    //printf("type:%d, %p %p %p %p, %d %d %d %d %d\n", mType, inputs[0]->deviceId(), mQ_Buffer, mK_Buffer, mV_Buffer, mHeadSizeV, mNumHeads, mSeqLen, mSeqLenKV, mBatchSize);
    if(mType == 0) {
        SPLIT_FusedQKV<<<block_num, thread_num>>>(count, (const half*)inputs[0]->deviceId(), (half *)mQ_Buffer, (half *)mK_Buffer, (half *)mV_Buffer, mHeadSizeV);
        checkKernelErrors;
    }
    if(mType == 1) {
        mQ_Buffer = (void *)inputs[0]->deviceId();
        SPLIT_FusedKV<<<block_num, thread_num>>>(count, (const half*)inputs[1]->deviceId(), (half *)mK_Buffer, (half *)mV_Buffer, mHeadSizeV);
        checkKernelErrors;
    }
    // Determine kernel configuration based on head size.
    // If head size is less than or equal to 64, each block operates over 64 queries and
    // 64 keys, and partial results can be stored in the register file.
    // If head size is greater than 64, each block operates over 32 queries and 128 keys,
    // and partial results are stored in shared memory.
    int ret = 0;
    if (mHeadSize > 64) {
        static int const kQueriesPerBlock = 32;
        static int const kKeysPerBlock = 128;
        if (mHeadSize <= 128) {
            ret = run_attention<kQueriesPerBlock, kKeysPerBlock, 128>(inputs, outputs);
        } else {
            ret = run_attention<kQueriesPerBlock, kKeysPerBlock, 65536>(inputs, outputs);
        }
    } else {
        static constexpr int kMaxK = 64; // <- Decrease to 32/16 if your problem is smaller
        static int const kQueriesPerBlock = 64;
        static int const kKeysPerBlock = 64;
        ret = run_attention<kQueriesPerBlock, kKeysPerBlock, kMaxK>(inputs, outputs);
    }

    // printf("fmha shape  b:%d s:%d %d h_num:%d h_size:%d, %d\n", mBatchSize, mSeqLen, mSeqLenKV, mNumHeads, mHeadSize, mHeadSizeV);
    checkKernelErrors;
    if(ret != 0) {
        MNN_ERROR("FmhaCommonExecution error\n");
    }
#ifdef LOG_VERBOSE
    MNN_PRINT("end FmhaCommonExecution onExecute...");
#endif
    return NO_ERROR;
}

} // namespace CUDA
} // namespace MNN
