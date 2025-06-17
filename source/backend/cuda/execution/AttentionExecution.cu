#include "AttentionExecution.hpp"
#include "core/TensorUtils.hpp"

namespace MNN {
namespace CUDA {

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

// 从 MNN Tensor 获取 CUDA 设备指针的辅助函数
template<typename T = void> // 默认为 void*
static inline T* getTensorDevicePtr(const Tensor* tensor) {
    if (!tensor || tensor->deviceId() == 0) return nullptr;
    return reinterpret_cast<T*>(tensor->deviceId());
}

// 根据 reserve 列表紧凑化 KV Cache
__global__ void compact_kv_cache_kernel(
    const void* src_key_cache,
    const void* src_value_cache,
    void* dst_key_cache,
    void* dst_value_cache,
    const int* reserve_info, // GPU上的 [begin0, len0, begin1, len1, ...] 数组
    const int* reserve_offsets, // GPU上的 [0, len0, len0+len1, ...] 前缀和数组
    int n_reserve_pairs,
    int past_kv_len_after_remove, // 移除了末尾token后的长度
    int b, int h_kv, int d,
    int kv_cache_max_len,
    size_t element_size
) {
    // 每个线程负责一个 (b, h_kv, d) 组合下，一个 reserve 片段的拷贝
    int bhd_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int reserve_pair_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (bhd_idx >= b * h_kv * d || reserve_pair_idx >= n_reserve_pairs) {
        return;
    }

    int b_idx = bhd_idx / (h_kv * d);
    int h_kv_idx = (bhd_idx % (h_kv * d)) / d;
    int d_idx = bhd_idx % d;

    int copy_src_begin = reserve_info[reserve_pair_idx * 2];
    int copy_len = reserve_info[reserve_pair_idx * 2 + 1];
    int copy_dst_begin_offset = reserve_offsets[reserve_pair_idx]; // 在 reserve 区域内的偏移

    long long src_offset = past_kv_len_after_remove + copy_src_begin;
    long long dst_offset = past_kv_len_after_remove + copy_dst_begin_offset;

    uint8_t* src_k_ptr = (uint8_t*)src_key_cache;
    uint8_t* dst_k_ptr = (uint8_t*)dst_key_cache;
    uint8_t* src_v_ptr = (uint8_t*)src_value_cache;
    uint8_t* dst_v_ptr = (uint8_t*)dst_value_cache;

    // Key Cache: [L, B, H_kv, D] -> 拷贝整个 (B, H_kv, D) 下的 L 片段
    for (int l = 0; l < copy_len; ++l) {
        // Key: [L, B, H, D]
        long long k_src_idx = ((src_offset + l) * b + b_idx) * h_kv * d + h_kv_idx * d + d_idx;
        long long k_dst_idx = ((dst_offset + l) * b + b_idx) * h_kv * d + h_kv_idx * d + d_idx;
        memcpy(dst_k_ptr + k_dst_idx * element_size, src_k_ptr + k_src_idx * element_size, element_size);

        // Value: [B, H, D, L]
        long long v_src_idx = (((long long)b_idx * h_kv + h_kv_idx) * d + d_idx) * kv_cache_max_len + (src_offset + l);
        long long v_dst_idx = (((long long)b_idx * h_kv + h_kv_idx) * d + d_idx) * kv_cache_max_len + (dst_offset + l);
        memcpy(dst_v_ptr + v_dst_idx * element_size, src_v_ptr + v_src_idx * element_size, element_size);
    }
}

// 将新 K 和 V 分别复制到对应 Cache 之后
template<typename T>
__global__ void copy_kv_to_cache_kernel(
    const T* key_input,         // 形状: B, L_k_new, H_kv, D
    const T* value_input,       // 形状: B, L_k_new, H_kv, D
    T* key_cache_output,        // 形状: L_kv_alloc, B, H_kv, D
    T* value_cache_output,      // 形状: B, H_kv, D, L_kv_alloc
    int batch_size,
    int new_kv_seq_len,         // L_k_new
    int kv_num_head,            // H_kv
    int head_dim,               // D
    int past_kv_len,            // L_k_past
    int allocated_kv_len        // L_kv_alloc
) {
    // 每个线程负责拷贝一个元素
    int d_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int l_idx_new = blockIdx.y * blockDim.y + threadIdx.y; 
    int bh_kv_idx = blockIdx.z * blockDim.z + threadIdx.z;

    if (d_idx >= head_dim || l_idx_new >= new_kv_seq_len || bh_kv_idx >= batch_size * kv_num_head) {
        return;
    }

    int b_idx = bh_kv_idx / kv_num_head;
    int h_kv_idx = bh_kv_idx % kv_num_head;

    // 输入 K 和 V 的源索引 (假设输入布局为 [B, L_k_new, H_kv, D])
    long long input_offset = (long long)b_idx * new_kv_seq_len * kv_num_head * head_dim +
                             (long long)l_idx_new * kv_num_head * head_dim +
                             (long long)h_kv_idx * head_dim;
    
    T val_to_copy_k = key_input[input_offset + d_idx];
    T val_to_copy_v = value_input[input_offset + d_idx]; // 从相同偏移读取，因为K和V输入形状相同

    // Cache 中的目标序列索引 (在 past_kv_len 之后追加)
    int dest_seq_idx_cache = past_kv_len + l_idx_new;
    if (dest_seq_idx_cache >= allocated_kv_len) return; // 边界检查

    // Key Cache 输出: [L_kv_alloc, B, H_kv, D]
    long long key_cache_idx = (long long)dest_seq_idx_cache * batch_size * kv_num_head * head_dim +
                              (long long)b_idx * kv_num_head * head_dim +
                              (long long)h_kv_idx * head_dim +
                              d_idx;
    key_cache_output[key_cache_idx] = val_to_copy_k;

    // Value Cache 输出: [B, H_kv, D, L_kv_alloc]
    long long value_cache_idx = (long long)b_idx * kv_num_head * head_dim * allocated_kv_len +
                                (long long)h_kv_idx * head_dim * allocated_kv_len +
                                (long long)d_idx * allocated_kv_len +
                                dest_seq_idx_cache;
    value_cache_output[value_cache_idx] = val_to_copy_v;
}

template<typename T, typename AccT = float> // T 用于存储, AccT 用于累加
__global__ void qk_kernel(
    const T* query_input,       // 形状: B, L_q_full, H_q, D
    const T* key_cache,         // 形状: L_k_total_alloc, B, H_kv, D (使用 param->key_seq_len 获取实际数据长度)
    T* qk_scores_output,        // 形状: B, H_q, L_q_piece, L_k_total
    const void* mask_tensor_data, // 可以是 T* 或 int*
    const AttentionKernelParam* param,
    int q_seq_piece_offset,     // 当前查询片在完整查询序列中的偏移
    bool has_mask_flag,
    bool is_add_mask_flag
) {
    // 每个线程计算 qk_scores_output 中的一个元素
    int k_idx = blockIdx.x * blockDim.x + threadIdx.x;          // 沿 key_sequence_length (L_k_total) 的索引
    int q_idx_in_piece = blockIdx.y * blockDim.y + threadIdx.y; // 当前查询片内的索引 (L_q_piece)
    int bh_q_idx = blockIdx.z * blockDim.z + threadIdx.z;       // 批次和查询头组合索引

    if (k_idx >= param->key_seq_len || q_idx_in_piece >= param->q_seq_piece_len || bh_q_idx >= param->batch * param->head_num) {
        return;
    }

    int b_idx = bh_q_idx / param->head_num;
    int h_q_idx = bh_q_idx % param->head_num;
    int current_full_q_idx = q_seq_piece_offset + q_idx_in_piece; // 在完整查询序列中的实际查询token索引

    if (current_full_q_idx >= param->query_seq_len) return; // 对填充片的边界检查

    AccT score_sum = 0.0f;
    int h_kv_idx = h_q_idx / param->group; // 对应的 KV 头索引

    // Query 元素指针基址 Q[b_idx, current_full_q_idx, h_q_idx, :]
    const T* q_ptr = query_input + (long long)b_idx * param->query_seq_len * param->head_num * param->head_dim +
                                   (long long)current_full_q_idx * param->head_num * param->head_dim +
                                   (long long)h_q_idx * param->head_dim;

    // Key 元素指针基址 K_cache[k_idx, b_idx, h_kv_idx, :]
    // Key Cache: [L_kv_alloc, B, H_kv, D]
    const T* k_ptr = key_cache + (long long)k_idx * param->batch * param->kv_head_num * param->head_dim +
                                 (long long)b_idx * param->kv_head_num * param->head_dim +
                                 (long long)h_kv_idx * param->head_dim;

    for (int d = 0; d < param->head_dim; ++d) {
        score_sum += static_cast<AccT>(q_ptr[d]) * static_cast<AccT>(k_ptr[d]);
    }
    score_sum *= param->scale;

    // 如果存在Mask则应用
    if (has_mask_flag && mask_tensor_data) {
        // current_full_q_idx 是在完整查询序列中的索引 (行索引)
        // k_idx 是在当前有效Key序列中的索引 (列索引)
        // 浮点 Mask 布局为 L_q * L_q (param->query_seq_len * param->query_seq_len)，整数 Mask 布局为 L_q * L_k
        long long mask_idx = (long long)current_full_q_idx * (is_add_mask_flag ? param->query_seq_len : param->key_seq_len) + k_idx - param->key_seq_len + param->query_seq_len;

        if (is_add_mask_flag) {
            // 加性Mask通常是float类型
            if (k_idx >= param->key_seq_len - param->query_seq_len)
                score_sum += k_idx >= param->key_seq_len - param->query_seq_len ?
                    static_cast<const AccT*>(mask_tensor_data)[mask_idx]
                    : 0; // 前 L_k - L_q 个 Mask 均视为 0
        } else {
            // 设置Mask通常是int类型, 0表示mask掉
            if (static_cast<const int*>(mask_tensor_data)[mask_idx] == 0) {
                score_sum = AccT(-1e9f);
            }
        }
    }

    // 输出: qk_scores_output[b_idx, h_q_idx, q_idx_in_piece, k_idx]
    long long out_idx = (long long)b_idx * param->head_num * param->q_seq_piece_len * param->key_seq_len +
                        (long long)h_q_idx * param->q_seq_piece_len * param->key_seq_len +
                        (long long)q_idx_in_piece * param->key_seq_len +
                        k_idx;
    qk_scores_output[out_idx] = static_cast<T>(score_sum);
}

template<typename T, typename AccT = float>
__global__ void softmax_kernel(
    const T* qk_scores,         // 形状: B, H_q, L_q_piece, L_k_total
    T* softmax_result,          // 形状: B, H_q, L_q_piece, L_k_total
    const AttentionKernelParam* param,
    int current_piece_actual_len // 当前分片的实际q_seq长度
) {
    // 每个线程块处理一行进行Softmax (总行数: B * H_q * L_q_piece)
    int row_global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 当前分片中用于softmax的总行数
    int total_rows_for_this_piece = param->batch * param->head_num * current_piece_actual_len;

    if (row_global_idx >= total_rows_for_this_piece) {
        return;
    }
    
    const T* current_row_scores = qk_scores + (long long)row_global_idx * param->key_seq_len;
    T* current_row_result = softmax_result + (long long)row_global_idx * param->key_seq_len;

    // 1. 找到行中的最大值
    AccT max_val = AccT(-1e9f); // 或者 -FLT_MAX
    if (param->key_seq_len > 0) { // 处理空key序列的情况
        max_val = static_cast<AccT>(current_row_scores[0]);
        for (int i = 1; i < param->key_seq_len; ++i) {
            if (static_cast<AccT>(current_row_scores[i]) > max_val) {
                max_val = static_cast<AccT>(current_row_scores[i]);
            }
        }
    }

    // 2. 计算 exp 的和
    AccT sum_exp = 0.0f;
    for (int i = 0; i < param->key_seq_len; ++i) {
        sum_exp += expf(static_cast<AccT>(current_row_scores[i]) - max_val);
    }
    // 避免除以零
    AccT inv_sum_exp = (sum_exp == 0.0f) ? AccT(1e-10f) : (AccT(1.0f) / sum_exp);

    // 3. 计算 Softmax
    for (int i = 0; i < param->key_seq_len; ++i) {
        current_row_result[i] = static_cast<T>(expf(static_cast<AccT>(current_row_scores[i]) - max_val) * inv_sum_exp);
    }
}

template<typename T, typename AccT = float>
__global__ void qkv_kernel(
    const T* softmax_probs,     // 形状: B, H_q, L_q_piece, L_k_total
    const T* value_cache,       // 形状: B, H_kv, D, L_k_alloc_max (使用 param->key_seq_len 获取实际数据, param->max_kv_len 用于 stride)
    T* attention_output,        // 形状: B, L_q_full, H_q, D
    const AttentionKernelParam* param,
    int q_seq_piece_offset      // 当前查询片的偏移
) {
    // 每个线程计算最终输出中D维的一个元素
    int d_idx = blockIdx.x * blockDim.x + threadIdx.x;          // 沿 head_dim 的索引
    int q_idx_in_piece = blockIdx.y * blockDim.y + threadIdx.y; // 当前查询片内的索引
    int bh_q_idx = blockIdx.z * blockDim.z + threadIdx.z;       // 批次和查询头组合索引

    if (d_idx >= param->head_dim || q_idx_in_piece >= param->q_seq_piece_len || bh_q_idx >= param->batch * param->head_num) {
        return;
    }

    int b_idx = bh_q_idx / param->head_num;
    int h_q_idx = bh_q_idx % param->head_num;
    int current_full_q_idx = q_seq_piece_offset + q_idx_in_piece;

    if (current_full_q_idx >= param->query_seq_len) return; // 边界检查

    AccT weighted_sum = 0.0f;
    int h_kv_idx = h_q_idx / param->group; // 对应的 KV 头索引

    // Softmax 概率指针 S[b_idx, h_q_idx, q_idx_in_piece, :]
    const T* prob_ptr = softmax_probs + (long long)b_idx * param->head_num * param->q_seq_piece_len * param->key_seq_len +
                                      (long long)h_q_idx * param->q_seq_piece_len * param->key_seq_len +
                                      (long long)q_idx_in_piece * param->key_seq_len;

    // Value Cache 指针基址 V[b_idx, h_kv_idx, d_idx, :]
    // Value Cache 布局: [B, H_kv, D, L_kv_alloc_max]
    const T* val_ptr_base = value_cache + (long long)b_idx * param->kv_head_num * param->head_dim * param->max_kv_len +
                                        (long long)h_kv_idx * param->head_dim * param->max_kv_len +
                                        (long long)d_idx * param->max_kv_len;

    for (int k_s = 0; k_s < param->key_seq_len; ++k_s) { // 沿 L_k_total (param->key_seq_len) 求和
        weighted_sum += static_cast<AccT>(prob_ptr[k_s]) * static_cast<AccT>(val_ptr_base[k_s]);
    }

    // 最终输出: O[b_idx, current_full_q_idx, h_q_idx, d_idx]
    long long out_idx = (long long)b_idx * param->query_seq_len * param->head_num * param->head_dim +
                        (long long)current_full_q_idx * param->head_num * param->head_dim +
                        (long long)h_q_idx * param->head_dim +
                        d_idx;
    attention_output[out_idx] = static_cast<T>(weighted_sum);
}


// ======= AttentionExecution 类实现 =======

AttentionExecution::AttentionExecution(Backend* backend, bool kv_cache_op_param)
    : Execution(backend), mIsKVCacheEnabled(kv_cache_op_param), mCudaBackend(static_cast<CUDABackend*>(backend)),
      mBatch(0), mQuerySeqLen(0), mNumHead(0), mHeadDim(0), mKvNumHead(0), mNewKvSeqLen(0),
      mQseqSplitNum(1), mHasMask(false), mIsAddMask(false), mParam_gpu(nullptr), mScale(1.0f) {
    mPrecision = halide_type_of<float>(); // 默认精度,可在 onResize 中更改
    if (mIsKVCacheEnabled) {
        mCache.reset(new SharedCache());
        mMeta = (KVMeta*)(mCudaBackend->getRuntime()->pMeta);
        
        // printf("[Attention Constructor] Reading runtime. Address is: %p\n", (void*)(mCudaBackend->getRuntime()));
        // printf("[Attention Constructor] Reading pMeta as address. %p\n", (void*)(mCudaBackend->getRuntime()->pMeta));
    }
}

AttentionExecution::~AttentionExecution() {
    if (mParam_gpu) {
        cudaFree(mParam_gpu); // 如果使用 cudaMalloc 分配则直接释放
        mParam_gpu = nullptr;
    }
    
    // Tensor的shared_ptr将通过CUDABackend的BufferPool处理它们自己的释放
}

// 初始化一个大小为 1 的占位 KVCache
ErrorCode AttentionExecution::init_cache_tensors() {
    if (!mIsKVCacheEnabled || !mCache) return MNN::NO_ERROR; // 如果禁用缓存或缓存未创建，则不执行任何操作
    if (mCache->mPastKey && mCache->mPastValue && mCache->mPastKey->deviceId() != 0) return MNN::NO_ERROR; // 已初始化

    mCache->mPastLength = 0;
    mCache->mMaxLength = 0; 
    
    // 创建占位Tensor，实际分配在 reallocKVCache_gpu 中进行
    mCache->mPastKey.reset(Tensor::createDevice({1, 1, 1, 1}, mPrecision, Tensor::CAFFE)); 
    mCache->mPastValue.reset(Tensor::createDevice({1, 1, 1, 1}, mPrecision, Tensor::CAFFE));
    if (!mCache->mPastKey || !mCache->mPastValue) return MNN::OUT_OF_MEMORY;
    
    bool res = mCudaBackend->onAcquireBuffer(mCache->mPastKey.get(), Backend::STATIC);
    if (!res) return MNN::OUT_OF_MEMORY;
    res = mCudaBackend->onAcquireBuffer(mCache->mPastValue.get(), Backend::STATIC);
    if (!res) return MNN::OUT_OF_MEMORY;

    return MNN::NO_ERROR;
}

// 为新的 KV 大小重新分配 KVCache，并拷贝旧的 KVCache（无 mMeta 版本）
ErrorCode AttentionExecution::reallocKVCache_gpu(int required_total_kv_len, int batch_size, int kv_num_head, int head_dim, cudaStream_t stream) {
    if (!mIsKVCacheEnabled || !mCache) return MNN::NO_ERROR;

    if (required_total_kv_len > mCache->mMaxLength || mCache->mPastKey->deviceId() == 0) {
        int old_max_len = mCache->mMaxLength;
        int old_past_len = mCache->mPastLength; 
        bool needs_copy = old_past_len > 0 && mCache->mPastKey && mCache->mPastKey->deviceId() != 0;

        int new_allocated_max_len = ROUND_UP(required_total_kv_len, mExpandChunk);
        // 如果计算出的新分配长度不大于旧的，并且旧缓存有效，则无需重新分配（除非初始分配）
        if (new_allocated_max_len <= old_max_len && mCache->mPastKey->deviceId() != 0) { 
            return MNN::NO_ERROR;
        }
        
        // Key Cache: [L_kv_max, B, H_kv, D]
        std::shared_ptr<Tensor> new_past_key_tensor(Tensor::createDevice({new_allocated_max_len, batch_size, kv_num_head, head_dim}, mPrecision, Tensor::CAFFE));
        // Value Cache: [B, H_kv, D, L_kv_max]
        std::shared_ptr<Tensor> new_past_value_tensor(Tensor::createDevice({batch_size, kv_num_head, head_dim, new_allocated_max_len}, mPrecision, Tensor::CAFFE));

        if (!new_past_key_tensor || !new_past_value_tensor) return MNN::OUT_OF_MEMORY;

        bool resK = mCudaBackend->onAcquireBuffer(new_past_key_tensor.get(), Backend::STATIC);
        bool resV = mCudaBackend->onAcquireBuffer(new_past_value_tensor.get(), Backend::STATIC);
        if(!resK || !resV) return MNN::OUT_OF_MEMORY;

        if (needs_copy) {
            size_t element_size_bytes = mPrecision.bytes();
            // 拷贝 Key Cache: 从旧的拷贝 [old_past_len, B, H_kv, D] 片段到新的
            size_t key_bytes_to_copy = (size_t)old_past_len * batch_size * kv_num_head * head_dim * element_size_bytes;
            if (key_bytes_to_copy > 0) {
                 cudaMemcpyAsync(getTensorDevicePtr(new_past_key_tensor.get()),
                                 getTensorDevicePtr(mCache->mPastKey.get()),
                                 key_bytes_to_copy, cudaMemcpyDeviceToDevice, stream);
                 checkKernelErrors;
            }
            
            // 拷贝 Value Cache: 从旧的拷贝 [B, H_kv, D, old_past_len] 片段到新的
            // 这需要对 B*H_kv*D 个平面进行2D拷贝
            for (int b = 0; b < batch_size; ++b) {
                for (int h_kv = 0; h_kv < kv_num_head; ++h_kv) {
                    for (int d_s = 0; d_s < head_dim; ++d_s) {
                        uint8_t* dst_ptr = getTensorDevicePtr<uint8_t>(new_past_value_tensor.get()) +
                                           ((((long long)b * kv_num_head + h_kv) * head_dim + d_s) * new_allocated_max_len) * element_size_bytes;
                        uint8_t* src_ptr = getTensorDevicePtr<uint8_t>(mCache->mPastValue.get()) +
                                           ((((long long)b * kv_num_head + h_kv) * head_dim + d_s) * old_max_len) * element_size_bytes;
                        if ((size_t)old_past_len * element_size_bytes > 0) {
                            cudaMemcpyAsync(dst_ptr, src_ptr, (size_t)old_past_len * element_size_bytes, cudaMemcpyDeviceToDevice, stream);
                            checkKernelErrors;
                        }
                    }
                }
            }
        }
        
        if (mCache->mPastKey && mCache->mPastKey->deviceId() != 0) mCudaBackend->onReleaseBuffer(mCache->mPastKey.get(), Backend::STATIC);
        if (mCache->mPastValue && mCache->mPastValue->deviceId() != 0) mCudaBackend->onReleaseBuffer(mCache->mPastValue.get(), Backend::STATIC);
        
        mCache->mPastKey = new_past_key_tensor;
        mCache->mPastValue = new_past_value_tensor;
        mCache->mMaxLength = new_allocated_max_len;
    }
    return MNN::NO_ERROR;
}

// 为新的 KV 大小重新分配 KVCache，并拷贝旧的 KVCache（mMeta 版本）
// required_total_kv_len 为新增 token 后，Cache 需要达到的总长度
ErrorCode AttentionExecution::reallocKVCache_gpu(int required_total_kv_len, const KVMeta* meta, cudaStream_t stream) {
    if (!mIsKVCacheEnabled || !mCache) {
        return MNN::NO_ERROR;
    }
    if (!meta) {
        MNN_ERROR("KVMeta is null in reallocKVCache_gpu, which is required for dynamic cache management.\n");
        return MNN::INVALID_VALUE;
    }

    size_t element_size = mPrecision.bytes();
    bool needs_realloc = required_total_kv_len > mCache->mMaxLength;

    int past_len_after_remove = mCache->mPastLength - meta->remove;
    if (past_len_after_remove < 0) {
        past_len_after_remove = 0;
    }

    // 处理数据紧凑化 (meta->reserve)
    if (meta->n_reserve > 0 && meta->reserve != nullptr) {
        std::vector<int> reserve_offsets_host(meta->n_reserve + 1, 0);
        for (int i = 0; i < meta->n_reserve; ++i) {
            reserve_offsets_host[i+1] = reserve_offsets_host[i] + meta->reserve[2 * i + 1];
        }
        int* reserve_info_gpu = nullptr;
        int* reserve_offsets_gpu = nullptr;
        cudaMallocAsync(&reserve_info_gpu, meta->n_reserve * 2 * sizeof(int), stream);
        cudaMallocAsync(&reserve_offsets_gpu, (meta->n_reserve + 1) * sizeof(int), stream);
        cudaMemcpyAsync(reserve_info_gpu, meta->reserve, meta->n_reserve * 2 * sizeof(int), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(reserve_offsets_gpu, reserve_offsets_host.data(), (meta->n_reserve + 1) * sizeof(int), cudaMemcpyHostToDevice, stream);

        dim3 blockDim(mHeadDim, 1, 1);
        dim3 gridDim(mBatch * mKvNumHead, 1, 1);
        
        // --- 执行紧凑化 ---
        if (needs_realloc) {
            // 优化路径：如果需要扩容，则直接将旧Cache的有效数据紧凑化拷贝到新Cache中
            int new_allocated_max_len = ROUND_UP(required_total_kv_len, mExpandChunk);
            std::shared_ptr<Tensor> new_past_key_tensor(
                Tensor::createDevice({new_allocated_max_len, mBatch, mKvNumHead, mHeadDim},
                mPrecision, Tensor::CAFFE
            ));
            std::shared_ptr<Tensor> new_past_value_tensor(
                Tensor::createDevice({mBatch, mKvNumHead, mHeadDim, new_allocated_max_len},
                mPrecision, Tensor::CAFFE
            ));
            if(!mCudaBackend->onAcquireBuffer(new_past_key_tensor.get(), Backend::STATIC)
            || !mCudaBackend->onAcquireBuffer(new_past_value_tensor.get(), Backend::STATIC)) {
                return MNN::OUT_OF_MEMORY;
            }

            compact_kv_cache_kernel<<<gridDim, blockDim, 0, stream>>>(
                getTensorDevicePtr(mCache->mPastKey.get()), getTensorDevicePtr(mCache->mPastValue.get()),
                getTensorDevicePtr(new_past_key_tensor.get()), getTensorDevicePtr(new_past_value_tensor.get()),
                reserve_info_gpu, reserve_offsets_gpu,
                meta->n_reserve, past_len_after_remove,
                mBatch, mKvNumHead, mHeadDim, new_allocated_max_len, element_size
            );
            
            // 交换指针
            mCudaBackend->onReleaseBuffer(mCache->mPastKey.get(), Backend::STATIC);
            mCudaBackend->onReleaseBuffer(mCache->mPastValue.get(), Backend::STATIC);
            mCache->mPastKey = new_past_key_tensor;
            mCache->mPastValue = new_past_value_tensor;
            mCache->mMaxLength = new_allocated_max_len;
        } else {
            // 如果不需要扩容，则需要一个临时缓冲区来执行原地紧凑化
            std::shared_ptr<Tensor> temp_key(Tensor::createDevice(mCache->mPastKey->shape(), mPrecision, Tensor::CAFFE));
            std::shared_ptr<Tensor> temp_value(Tensor::createDevice(mCache->mPastValue->shape(), mPrecision, Tensor::CAFFE));
            if(!mCudaBackend->onAcquireBuffer(temp_key.get(), Backend::STATIC)
            || !mCudaBackend->onAcquireBuffer(temp_value.get(), Backend::STATIC)) {
                return MNN::OUT_OF_MEMORY;
            }

            // 从旧Cache紧凑化到临时Cache
            compact_kv_cache_kernel<<<gridDim, blockDim, 0, stream>>>(
                getTensorDevicePtr(mCache->mPastKey.get()), getTensorDevicePtr(mCache->mPastValue.get()),
                getTensorDevicePtr(temp_key.get()), getTensorDevicePtr(temp_value.get()),
                reserve_info_gpu, reserve_offsets_gpu,
                meta->n_reserve, past_len_after_remove,
                mBatch, mKvNumHead, mHeadDim, mCache->mMaxLength, element_size
            );
            // 将临时Cache的内容拷贝回旧Cache
            int compacted_len = past_len_after_remove + meta->computeReverseSize();
            cudaMemcpyAsync(getTensorDevicePtr(mCache->mPastKey.get()),
                getTensorDevicePtr(temp_key.get()),
                compacted_len * mBatch * mKvNumHead * mHeadDim * element_size,
                cudaMemcpyDeviceToDevice, stream);
            cudaMemcpy2DAsync(getTensorDevicePtr(mCache->mPastValue.get()),
                mCache->mMaxLength * element_size,
                getTensorDevicePtr(temp_value.get()),
                mCache->mMaxLength * element_size,
                compacted_len * element_size,
                mBatch * mKvNumHead * mHeadDim,
                cudaMemcpyDeviceToDevice, stream);
            mCudaBackend->onReleaseBuffer(temp_key.get(), Backend::STATIC);
            mCudaBackend->onReleaseBuffer(temp_value.get(), Backend::STATIC);
        }
        checkKernelErrors;
        cudaFreeAsync(reserve_info_gpu, stream);
        cudaFreeAsync(reserve_offsets_gpu, stream);
        mCache->mPastLength = past_len_after_remove + meta->computeReverseSize();
    } else { // 无 reserve 指令
        mCache->mPastLength = past_len_after_remove;
        if (needs_realloc) {
            int old_max_len = mCache->mMaxLength;
            int old_past_len_to_copy = mCache->mPastLength;
            int new_allocated_max_len = ROUND_UP(required_total_kv_len, mExpandChunk);

            std::shared_ptr<Tensor> new_past_key_tensor(
                Tensor::createDevice({new_allocated_max_len, mBatch, mKvNumHead, mHeadDim},
                mPrecision, Tensor::CAFFE
            ));
            std::shared_ptr<Tensor> new_past_value_tensor(
                Tensor::createDevice({mBatch, mKvNumHead, mHeadDim, new_allocated_max_len},
                mPrecision, Tensor::CAFFE
            ));
            if(!mCudaBackend->onAcquireBuffer(new_past_key_tensor.get(), Backend::STATIC)
            || !mCudaBackend->onAcquireBuffer(new_past_value_tensor.get(), Backend::STATIC)) {
                return MNN::OUT_OF_MEMORY;
            }

            if (old_past_len_to_copy > 0) {
                cudaMemcpyAsync(getTensorDevicePtr(new_past_key_tensor.get()),
                    getTensorDevicePtr(mCache->mPastKey.get()),
                    (size_t)old_past_len_to_copy * mBatch * mKvNumHead * mHeadDim * element_size,
                    cudaMemcpyDeviceToDevice, stream);
                cudaMemcpy2DAsync(getTensorDevicePtr(new_past_value_tensor.get()),
                    (size_t)new_allocated_max_len * element_size,
                    getTensorDevicePtr(mCache->mPastValue.get()),
                    (size_t)old_max_len * element_size,
                    (size_t)old_past_len_to_copy * element_size,
                    (size_t)mBatch * mKvNumHead * mHeadDim,
                    cudaMemcpyDeviceToDevice, stream);
                checkKernelErrors;
            }

            mCudaBackend->onReleaseBuffer(mCache->mPastKey.get(), Backend::STATIC);
            mCudaBackend->onReleaseBuffer(mCache->mPastValue.get(), Backend::STATIC);
            mCache->mPastKey = new_past_key_tensor;
            mCache->mPastValue = new_past_value_tensor;
            mCache->mMaxLength = new_allocated_max_len;
        }
    }

    return MNN::NO_ERROR;
}

// 为新 KV、QK^T、Softmax 分配 GPU 空间
ErrorCode AttentionExecution::ensureTempBuffers_gpu(int batch, int num_head, int q_seq_piece_len_max, int current_max_total_kv_len, int head_dim) {
    // QK Scores: [B, H_q, Max_L_q_piece, Max_L_k_total]
    std::vector<int> qk_shape = {batch, num_head, q_seq_piece_len_max, current_max_total_kv_len};
    bool qk_realloc = !mTempQK || mTempQK->shape() != qk_shape;
    if (qk_realloc) {
        if(mTempQK && mTempQK->deviceId() != 0) mCudaBackend->onReleaseBuffer(mTempQK.get(), Backend::STATIC);
        mTempQK.reset(Tensor::createDevice(qk_shape, mPrecision, Tensor::CAFFE));
        if(!mTempQK || !mCudaBackend->onAcquireBuffer(mTempQK.get(), Backend::STATIC)) return MNN::OUT_OF_MEMORY;
    }

    // Softmax Probs: 与QK scores形状相同
    bool softmax_realloc = !mTempSoftmax || mTempSoftmax->shape() != qk_shape;
    if (softmax_realloc) {
         if(mTempSoftmax && mTempSoftmax->deviceId() != 0) mCudaBackend->onReleaseBuffer(mTempSoftmax.get(), Backend::STATIC);
        mTempSoftmax.reset(Tensor::createDevice(qk_shape, mPrecision, Tensor::CAFFE));
        if(!mTempSoftmax || !mCudaBackend->onAcquireBuffer(mTempSoftmax.get(), Backend::STATIC)) return MNN::OUT_OF_MEMORY;
    }

    if (!mIsKVCacheEnabled) {
        // Temp K current step: [L_k_new_alloc, B, H_kv, D]
        int temp_k_alloc_len = ROUND_UP(mNewKvSeqLen, 1);
        std::vector<int> temp_k_shape = {temp_k_alloc_len, batch, mKvNumHead, head_dim};
        bool temp_k_realloc = !mTempK_current_step || mTempK_current_step->shape() != temp_k_shape;
        if(temp_k_realloc){
            if(mTempK_current_step && mTempK_current_step->deviceId() !=0) mCudaBackend->onReleaseBuffer(mTempK_current_step.get(), Backend::STATIC);
            mTempK_current_step.reset(Tensor::createDevice(temp_k_shape, mPrecision, Tensor::CAFFE));
            if(!mTempK_current_step || !mCudaBackend->onAcquireBuffer(mTempK_current_step.get(), Backend::STATIC)) return MNN::OUT_OF_MEMORY;
        }

        // Temp V current step: [B, H_kv, D, L_k_new_alloc]
        std::vector<int> temp_v_shape = {batch, mKvNumHead, head_dim, temp_k_alloc_len};
        bool temp_v_realloc = !mTempV_current_step || mTempV_current_step->shape() != temp_v_shape;
         if(temp_v_realloc){
            if(mTempV_current_step && mTempV_current_step->deviceId() !=0) mCudaBackend->onReleaseBuffer(mTempV_current_step.get(), Backend::STATIC);
            mTempV_current_step.reset(Tensor::createDevice(temp_v_shape, mPrecision, Tensor::CAFFE));
            if(!mTempV_current_step || !mCudaBackend->onAcquireBuffer(mTempV_current_step.get(), Backend::STATIC)) return MNN::OUT_OF_MEMORY;
        }
    }
    return MNN::NO_ERROR;
}

bool AttentionExecution::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (nullptr == dst) {
        return true;
    }
    auto exe = new AttentionExecution(bn, mIsKVCacheEnabled);
    exe->mCache = mCache;
    exe->mMeta = mMeta;
    *dst = exe;
    return true;
}

// #define DEBUG_ATTENTION_VERBOSE

static inline float half_to_float_debug(const __half& h_val) {
    return __half2float(h_val);
}

// 打印GPU张量指定切片的辅助函数
void print_gpu_tensor_debug(
    const MNN::Tensor* target_tensor,    // 要打印的张量 (可以是GPU或CPU上的)
    const char* name                     // 张量的名称，用于日志
) {
    if (!target_tensor) {
        printf("\n--- Tensor [%s] is null. ---\n", name);
        return;
    }

    printf("\n--- [%s] ---\n", name);
    target_tensor->print();
}

// 初始化所有参数
ErrorCode AttentionExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    const auto* query_tensor = inputs[0]; // 形状: [B, L_q, H_q, D]
    const auto* key_tensor = inputs[1];   // 形状: [B, L_k_new, H_kv, D]

    mPrecision = query_tensor->getType(); // 获取Tensor的halide_type_t

    mBatch = query_tensor->length(0);
    mQuerySeqLen = query_tensor->length(1);
    mNumHead = query_tensor->length(2);
    mHeadDim = query_tensor->length(3);

    mKvNumHead = key_tensor->length(2);
    mNewKvSeqLen = key_tensor->length(1); // 新K/V段的长度

    if (mHeadDim == 0 || mKvNumHead == 0) return MNN::INVALID_VALUE; // 避免除以零
    if (mNumHead % mKvNumHead != 0) return MNN::INVALID_VALUE; // Group大小必须是整数
    mScale = 1.0f / sqrtf(static_cast<float>(mHeadDim));

    mHasMask = inputs.size() > 3 && inputs[3] != nullptr && inputs[3]->elementSize() > 0;
    if (mHasMask) {
        mIsAddMask = (inputs[3]->getType().code == halide_type_float);
    }
    
    // 简化的 Q 分割逻辑
    mQseqSplitNum = 1;
    if (mQuerySeqLen > 1024) mQseqSplitNum = UP_DIV(mQuerySeqLen, 1024); // 限制每片最多1024
    else if (mQuerySeqLen > 256) mQseqSplitNum = UP_DIV(mQuerySeqLen, 256);
    
    if (mIsKVCacheEnabled) {
        ErrorCode err = init_cache_tensors(); // 确保Cache Tensor结构已初始化
        if (err != MNN::NO_ERROR) return err;
    }
    
    // 如果尚未分配 mParam_gpu 则进行分配
    if (!mParam_gpu) {
        auto cuda_err = cudaMalloc(&mParam_gpu, sizeof(AttentionKernelParam));
        if (cuda_err != cudaSuccess) { MNN_ERROR("cudaMalloc failed for mParam_gpu\n"); return MNN::NOT_SUPPORT; }
    }

    return MNN::NO_ERROR;
}

// 拷贝 KV 到 GPU，对 Q 分片后执行 Attention 流程
ErrorCode AttentionExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    const auto* query_input_tensor = inputs[0];
    const auto* key_input_tensor = inputs[1];
    const auto* value_input_tensor = inputs[2];
    const Tensor* mask_input_tensor = mHasMask ? inputs[3] : nullptr;
    auto final_output_tensor = outputs[0];

    // qk_kernel 默认 Mask 为 [1, 1, L_q, L_k]，但模型会出现 [1, 1, 1, 1] 值为 -0.00 的 Mask 代表无 Mask
    if (mIsKVCacheEnabled && mHasMask && mask_input_tensor && mask_input_tensor->elementSize() == 1) {
        mHasMask = false;
    }

    cudaStream_t stream = 0; // 使用默认 CUDA 流

    #ifdef DEBUG_ATTENTION_VERBOSE // 打印一次全局参数
        static bool mAlreadyPrintedGlobalParams = false;
        if (!mAlreadyPrintedGlobalParams) {
            printf("\n\n================ AttentionExecute Global Initial Params ================\n");
            printf("Global Params: mBatch: %d, mQuerySeqLen: %d, mNumHead: %d, mHeadDim: %d, mKvNumHead: %d, mNewKvSeqLen (from current K input): %d\n",
                mBatch, mQuerySeqLen, mNumHead, mHeadDim, mKvNumHead, mNewKvSeqLen);
            printf("mScale: %f, mHasMask: %d, mIsAddMask: %d, mIsKVCacheEnabled: %d, mQseqSplitNum: %d\n",
                mScale, mHasMask, mIsAddMask, mIsKVCacheEnabled, mQseqSplitNum);
            if (mIsKVCacheEnabled && mCache) {
                printf("[DEBUG] onExecute START: mCache Ptr: 0x%p, mCache->mPastLength: %d\n",
                    (void*)mCache.get(), mCache->mPastLength);
                printf("Initial Cache Status: mPastLength=%d, mMaxLength=%d\n", mCache->mPastLength, mCache->mMaxLength);
            }

            // print_gpu_tensor_debug(query_input_tensor, "Initial Query Input");
            // print_gpu_tensor_debug(key_input_tensor, "Initial Key Input (for current step)"); // new K
            // print_gpu_tensor_debug(value_input_tensor, "Initial Value Input (for current step)"); // new V
            if (mHasMask && mask_input_tensor) {
                // print_gpu_tensor_debug(mask_input_tensor, "Initial Mask Input (Full)");
            }

            if (mMeta) {
                printf("block: %zu previous: %zu remove: %zu add: %zu n_reserve: %d reserve: %p\n",
                    mMeta->block, mMeta->previous, mMeta->remove, mMeta->add, mMeta->n_reserve, mMeta->reserve);
            
                if (mMeta->reserve) {
                    int total = 0;
                    for (int i = 0; i < mMeta->n_reserve; i++) {
                        total += mMeta->reserve[2*i+1];  // 按原始 computeReverseSize 的逻辑累加
                    }
                    printf("computed reverse size: %d\n", total > 0 ? total : -1);
                }
            }
            printf("======================================================================\n");
            // mAlreadyPrintedGlobalParams = true; // 只打印一次
        }
    #endif

    AttentionKernelParam param_cpu;
    param_cpu.batch = mBatch; 
    param_cpu.query_seq_len = mQuerySeqLen;
    param_cpu.head_num = mNumHead;
    param_cpu.kv_head_num = mKvNumHead;
    param_cpu.group = mNumHead / mKvNumHead;
    param_cpu.head_dim = mHeadDim;
    param_cpu.scale = mScale;
    param_cpu.current_kv_seq_len_new = mNewKvSeqLen; // 来自当前输入的K/V长度

    const void* effective_key_cache_ptr;
    const void* effective_value_cache_ptr;
    int current_total_kv_len_for_qk;       // 用于QK和QKV Kernel的 L_k_total
    int allocated_kv_len_for_value_stride; // Value Cache最后一维的Stride (param->max_kv_len)

    if (mIsKVCacheEnabled) {
        if (!mMeta) {
            param_cpu.past_kv_len = mCache->mPastLength;
            current_total_kv_len_for_qk = param_cpu.past_kv_len + mNewKvSeqLen;
            
            ErrorCode err = reallocKVCache_gpu(current_total_kv_len_for_qk, mBatch, mKvNumHead, mHeadDim, stream);
            if (err != MNN::NO_ERROR) return err;
    
            allocated_kv_len_for_value_stride = mCache->mMaxLength;
            param_cpu.key_seq_len = current_total_kv_len_for_qk;
            param_cpu.max_kv_len = mCache->mMaxLength;
        } else { // 使用 mMeta 来计算需要的总长度和过去的长度
            int required_total_kv_len = (mMeta->previous - mMeta->remove + mMeta->computeReverseSize()) + mMeta->add;
            ErrorCode err = reallocKVCache_gpu(required_total_kv_len, mMeta, stream);
            if (err != MNN::NO_ERROR) return err;

            param_cpu.past_kv_len = mCache->mPastLength;
            param_cpu.key_seq_len = mCache->mPastLength + mNewKvSeqLen;
            param_cpu.max_kv_len = mCache->mMaxLength;
            param_cpu.current_kv_seq_len_new = mNewKvSeqLen;
            allocated_kv_len_for_value_stride = mCache->mMaxLength;

            current_total_kv_len_for_qk = param_cpu.key_seq_len;
        }

        // 拷贝新的 K, V 到 Cache
        dim3 copy_blockDim(32, 8, 1); // 暂定 Block大小: 256 线程
        dim3 copy_gridDim(UP_DIV(mHeadDim, (int)copy_blockDim.x),
                            UP_DIV(mNewKvSeqLen, (int)copy_blockDim.y),
                            UP_DIV(mBatch * mKvNumHead, (int)copy_blockDim.z));
        if (mPrecision.bytes() == 4) { // float32
             copy_kv_to_cache_kernel<float><<<copy_gridDim, copy_blockDim, 0, stream>>>(
                getTensorDevicePtr<float>(key_input_tensor), getTensorDevicePtr<float>(value_input_tensor),
                getTensorDevicePtr<float>(mCache->mPastKey.get()), getTensorDevicePtr<float>(mCache->mPastValue.get()),
                mBatch, mNewKvSeqLen, mKvNumHead, mHeadDim, param_cpu.past_kv_len, mCache->mMaxLength);
        } else if (mPrecision.bytes() == 2) { // float16
             copy_kv_to_cache_kernel<__half><<<copy_gridDim, copy_blockDim, 0, stream>>>(
                getTensorDevicePtr<__half>(key_input_tensor), getTensorDevicePtr<__half>(value_input_tensor),
                getTensorDevicePtr<__half>(mCache->mPastKey.get()), getTensorDevicePtr<__half>(mCache->mPastValue.get()),
                mBatch, mNewKvSeqLen, mKvNumHead, mHeadDim, param_cpu.past_kv_len, mCache->mMaxLength);
        } else { return MNN::NOT_SUPPORT; /* 未支持的精度 */ }
        checkKernelErrors;

        effective_key_cache_ptr = getTensorDevicePtr(mCache->mPastKey.get());
        effective_value_cache_ptr = getTensorDevicePtr(mCache->mPastValue.get());

        #ifdef DEBUG_ATTENTION_VERBOSE
            if (current_total_kv_len_for_qk >= 0) {
                // cudaStreamSynchronize(stream); // 确保 copy_kv_to_cache_kernel 完成
                // if (mCache && mCache->mPastKey) print_gpu_tensor_debug(mCache->mPastKey.get(), "Key Cache (After copy_kv_to_cache)"); // L_kv_alloc, B, H_kv, D
                // if (mCache && mCache->mPastValue) print_gpu_tensor_debug(mCache->mPastValue.get(), "Value Cache (After copy_kv_to_cache)"); // B, H_kv, D, L_kv_alloc    
            }
        #endif
    } else { // 没有 KV Cache, 使用当前步骤的临时Tensor存储 K/V
        param_cpu.past_kv_len = 0;
        current_total_kv_len_for_qk = mNewKvSeqLen;
        allocated_kv_len_for_value_stride = ROUND_UP(mNewKvSeqLen,1); // 临时K/V的分配长度
        param_cpu.key_seq_len = mNewKvSeqLen;
        param_cpu.max_kv_len = allocated_kv_len_for_value_stride;

        // 确保在非KV缓存模式下，当前步骤的临时K/V缓冲区已准备好
        ErrorCode temp_err = ensureTempBuffers_gpu(mBatch, mNumHead, UP_DIV(mQuerySeqLen, mQseqSplitNum), current_total_kv_len_for_qk, mHeadDim);
        if(temp_err != MNN::NO_ERROR) return temp_err;
        
        dim3 copy_blockDim(32, 8, 1);
        dim3 copy_gridDim(UP_DIV(mHeadDim, (int)copy_blockDim.x),
                            UP_DIV(mNewKvSeqLen, (int)copy_blockDim.y),
                            UP_DIV(mBatch * mKvNumHead, (int)copy_blockDim.z));
        if (mPrecision.bytes() == 4) {
            copy_kv_to_cache_kernel<float><<<copy_gridDim, copy_blockDim, 0, stream>>>(
                getTensorDevicePtr<float>(key_input_tensor), getTensorDevicePtr<float>(value_input_tensor),
                getTensorDevicePtr<float>(mTempK_current_step.get()), getTensorDevicePtr<float>(mTempV_current_step.get()),
                mBatch, mNewKvSeqLen, mKvNumHead, mHeadDim, 0, allocated_kv_len_for_value_stride);
        } else if (mPrecision.bytes() == 2) {
             copy_kv_to_cache_kernel<__half><<<copy_gridDim, copy_blockDim, 0, stream>>>(
                getTensorDevicePtr<__half>(key_input_tensor), getTensorDevicePtr<__half>(value_input_tensor),
                getTensorDevicePtr<__half>(mTempK_current_step.get()), getTensorDevicePtr<__half>(mTempV_current_step.get()),
                mBatch, mNewKvSeqLen, mKvNumHead, mHeadDim, 0, allocated_kv_len_for_value_stride);
        } else { return MNN::NOT_SUPPORT; }
        checkKernelErrors;
        
        effective_key_cache_ptr = getTensorDevicePtr(mTempK_current_step.get());
        effective_value_cache_ptr = getTensorDevicePtr(mTempV_current_step.get());

        #ifdef DEBUG_ATTENTION_VERBOSE
            // cudaStreamSynchronize(stream); // 确保 copy_kv_to_cache_kernel 完成
            // if (mTempK_current_step) print_gpu_tensor_debug(mTempK_current_step.get(), "Temp K Current Step (Copied from Input K)");
            // if (mTempV_current_step) print_gpu_tensor_debug(mTempV_current_step.get(), "Temp V Current Step (Copied from Input V)");
        #endif
    }

    int max_q_seq_piece_len = UP_DIV(mQuerySeqLen, mQseqSplitNum);
    // param_cpu.q_seq_piece_len 将在循环内针对每个分片实际长度进行更新
    
    // 确保 mTempQK 和 mTempSoftmax 根据最大可能的分片和当前KV长度调整大小
    ErrorCode temp_buf_err = ensureTempBuffers_gpu(mBatch, mNumHead, max_q_seq_piece_len, current_total_kv_len_for_qk, mHeadDim);
    if (temp_buf_err != MNN::NO_ERROR) return temp_buf_err;

    const void* mask_ptr_device = mHasMask ? getTensorDevicePtr(mask_input_tensor) : nullptr;

    for (int i = 0; i < mQseqSplitNum; ++i) {
        int q_seq_offset = i * max_q_seq_piece_len;
        int current_piece_actual_len = std::min(max_q_seq_piece_len, mQuerySeqLen - q_seq_offset);
        if (current_piece_actual_len <= 0) continue;

        // 为当前分片更新param_cpu中的q_seq_piece_len，并拷贝到GPU
        AttentionKernelParam current_iteration_param_cpu = param_cpu;
        current_iteration_param_cpu.q_seq_piece_len = current_piece_actual_len;
        cudaMemcpyAsync(mParam_gpu, &current_iteration_param_cpu, sizeof(AttentionKernelParam), cudaMemcpyHostToDevice, stream);
        checkKernelErrors; // 异步拷贝后检查，确保参数在Kernel启动前已上载

        // QK Kernel
        dim3 qk_blockDim(16, 16, 1); // 暂用 256 线程
        dim3 qk_gridDim(UP_DIV(current_total_kv_len_for_qk, (int)qk_blockDim.x),
                          UP_DIV(current_piece_actual_len, (int)qk_blockDim.y), 
                          UP_DIV(mBatch * mNumHead, (int)qk_blockDim.z));
        
        if (mPrecision.bytes() == 4) {
            qk_kernel<float><<<qk_gridDim, qk_blockDim, 0, stream>>>(
                getTensorDevicePtr<float>(query_input_tensor), static_cast<const float*>(effective_key_cache_ptr),
                getTensorDevicePtr<float>(mTempQK.get()), mask_ptr_device,
                mParam_gpu, q_seq_offset, mHasMask, mIsAddMask);
        } else if (mPrecision.bytes() == 2) {
             qk_kernel<__half><<<qk_gridDim, qk_blockDim, 0, stream>>>(
                getTensorDevicePtr<__half>(query_input_tensor), static_cast<const __half*>(effective_key_cache_ptr),
                getTensorDevicePtr<__half>(mTempQK.get()), mask_ptr_device,
                mParam_gpu, q_seq_offset, mHasMask, mIsAddMask);
        } else { return MNN::NOT_SUPPORT; }
        checkKernelErrors;
        
        // Softmax Kernel
        int softmax_total_rows_for_piece = mBatch * mNumHead * current_piece_actual_len;
        dim3 softmax_blockDim(256, 1, 1); // 每个线程处理一行
        dim3 softmax_gridDim(UP_DIV(softmax_total_rows_for_piece, (int)softmax_blockDim.x), 1, 1);
        if (mPrecision.bytes() == 4) {
            softmax_kernel<float><<<softmax_gridDim, softmax_blockDim, 0, stream>>>(
                getTensorDevicePtr<float>(mTempQK.get()), getTensorDevicePtr<float>(mTempSoftmax.get()), mParam_gpu, current_piece_actual_len);
        } else if (mPrecision.bytes() == 2) {
            softmax_kernel<__half><<<softmax_gridDim, softmax_blockDim, 0, stream>>>(
                getTensorDevicePtr<__half>(mTempQK.get()), getTensorDevicePtr<__half>(mTempSoftmax.get()), mParam_gpu, current_piece_actual_len);
        } else { return MNN::NOT_SUPPORT; }
        checkKernelErrors;
        
        // QKV Kernel
        dim3 qkv_blockDim(32, 8, 1); // 暂用 256 线程
        dim3 qkv_gridDim(UP_DIV(mHeadDim, (int)qkv_blockDim.x),
                         UP_DIV(current_piece_actual_len, (int)qkv_blockDim.y), 
                         UP_DIV(mBatch * mNumHead, (int)qkv_blockDim.z));
        if (mPrecision.bytes() == 4) {
            qkv_kernel<float><<<qkv_gridDim, qkv_blockDim, 0, stream>>>(
                getTensorDevicePtr<float>(mTempSoftmax.get()), static_cast<const float*>(effective_value_cache_ptr),
                getTensorDevicePtr<float>(final_output_tensor), mParam_gpu, q_seq_offset);
        } else if (mPrecision.bytes() == 2) {
            qkv_kernel<__half><<<qkv_gridDim, qkv_blockDim, 0, stream>>>(
                getTensorDevicePtr<__half>(mTempSoftmax.get()), static_cast<const __half*>(effective_value_cache_ptr),
                getTensorDevicePtr<__half>(final_output_tensor), mParam_gpu, q_seq_offset);
        } else { return MNN::NOT_SUPPORT; }
        checkKernelErrors;

        #ifdef DEBUG_ATTENTION_VERBOSE
            cudaStreamSynchronize(stream);

            printf("\n\n DEBUG DUMP FOR ATTENTION PIECE: %d / %d \n", i, mQseqSplitNum - 1);
            printf("====================================================================\n");
            printf("Piece Info: q_seq_offset: %d, current_piece_actual_len: %d, current_total_kv_len_for_qk: %d\n",
                q_seq_offset, current_piece_actual_len, current_total_kv_len_for_qk);

            // 打印当前piece的Kernel参数 (从GPU拷贝回来)
            AttentionKernelParam param_host_for_print;
            cudaError_t err_param_copy = cudaMemcpy( &param_host_for_print, mParam_gpu, sizeof(AttentionKernelParam), cudaMemcpyDeviceToHost);
            if (err_param_copy != cudaSuccess) { printf("ERROR: Failed to copy mParam_gpu to host for printing: %s\n", cudaGetErrorString(err_param_copy)); }
            else {
                printf("AttentionKernelParam (for this piece):\n");
                printf("  Lq_full: %d, Lq_piece: %d, Lk_total: %d, Lk_new: %d, Lk_past: %d\n",
                    param_host_for_print.query_seq_len, param_host_for_print.q_seq_piece_len,
                    param_host_for_print.key_seq_len, param_host_for_print.current_kv_seq_len_new,
                    param_host_for_print.past_kv_len);
                printf("  Hq: %d, Hkv: %d, D: %d, Scale: %.4f, B: %d, Max_Lk_Alloc: %d\n",
                    param_host_for_print.head_num, param_host_for_print.kv_head_num,
                    param_host_for_print.head_dim, param_host_for_print.scale,
                    param_host_for_print.batch, param_host_for_print.max_kv_len);
            }
            printf("--------------------------------------------------------------------\n");

            if (current_total_kv_len_for_qk >= 20) {
                // 打印 Mask (如果存在)
                // Mask 通常是 [1, 1, Lq_full, Lk_total]
                if (mHasMask && mask_input_tensor) {
                    // print_gpu_tensor_debug(mask_input_tensor, "Mask Input Tensor (Relevant Slice view)");
                }

                // 打印 QK^T Scores, mTempQK shape: [B, H_q, L_q_piece, L_k_total]
                // print_gpu_tensor_debug(mTempQK.get(), "QK^T Scores (mTempQK for current piece)");

                // 打印 Softmax Probabilities, mTempSoftmax shape: [B, H_q, L_q_piece, L_k_total]
                // print_gpu_tensor_debug(mTempSoftmax.get(), "Softmax Probs (mTempSoftmax for current piece)");

                // 打印 Attention Output, final_output_tensor shape: [B, L_q_full, H_q, D] or [1, 1, N]
                if (final_output_tensor->dimensions()==3) { // 三维如 [1, 1, N]
                    // print_gpu_tensor_debug(final_output_tensor, "Attention Output Slice (Final)");
                } else { // 假设是 [B, Lq_full, Hq, D]
                    // print_gpu_tensor_debug(final_output_tensor, "Attention Output Slice (Final)");
                }

                printf("========================= END PIECE DEBUG DUMP =======================\n");
            }
        #endif // DEBUG_ATTENTION_VERBOSE
    }

    if (mIsKVCacheEnabled) {
        mCache->mPastLength += mNewKvSeqLen;
    }

    #ifdef DEBUG_ATTENTION_VERBOSE
        if (mIsKVCacheEnabled && mCache) {
            printf("[DEBUG] onExecute END: mCache Ptr: 0x%p, mCache->mPastLength updated to: %d\n",
                (void*)mCache.get(), mCache->mPastLength);
        }
    #endif // DEBUG_ATTENTION_VERBOSE
    
    return MNN::NO_ERROR;
}


// Creator 注册
class AttentionCreator : public CUDABackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        bool op_kv_cache_enabled = false;
        if (op->main_type() == OpParameter_AttentionParam) { 
             auto att_param = op->main_as_AttentionParam();
             if (att_param) {
                 op_kv_cache_enabled = att_param->kv_cache();
             }
        } else {
            MNN_PRINT("[Warning] op->main_type() != OpParameter_AttentionParam\n");
        }
        return new AttentionExecution(backend, op_kv_cache_enabled);
    }
};
static CUDACreatorRegister<AttentionCreator> __init(OpType_Attention);

#endif // MNN_SUPPORT_TRANSFORMER_FUSE

} // namespace CUDA
} // namespace MNN