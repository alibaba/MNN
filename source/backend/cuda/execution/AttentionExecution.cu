#include "AttentionExecution.hpp"
#include "core/TensorUtils.hpp"
#include "SoftmaxExecution.hpp"

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
    int src_kv_cache_max_len, // Value Cache 源的 L 维度步长
    int dst_kv_cache_max_len, // Value Cache 目标的 L 维度步长
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

    int src_offset = past_kv_len_after_remove + copy_src_begin;
    int dst_offset = past_kv_len_after_remove + copy_dst_begin_offset;

    const uint8_t* src_k_ptr = static_cast<const uint8_t*>(src_key_cache);
    uint8_t* dst_k_ptr = static_cast<uint8_t*>(dst_key_cache);
    const uint8_t* src_v_ptr = static_cast<const uint8_t*>(src_value_cache);
    uint8_t* dst_v_ptr = static_cast<uint8_t*>(dst_value_cache);

    for (int l = 0; l < copy_len; ++l) {
        // Key: [L, B, H, D]
        int k_src_idx = (src_offset + l) * b * h_kv * d + b_idx * h_kv * d + h_kv_idx * d + d_idx;
        int k_dst_idx = (dst_offset + l) * b * h_kv * d + b_idx * h_kv * d + h_kv_idx * d + d_idx;
        memcpy(dst_k_ptr + k_dst_idx * element_size, src_k_ptr + k_src_idx * element_size, element_size);

        // Value: [B, H, L, D]
        int v_src_idx = ((b_idx * h_kv + h_kv_idx) * src_kv_cache_max_len + (src_offset + l)) * d + d_idx;
        int v_dst_idx = ((b_idx * h_kv + h_kv_idx) * dst_kv_cache_max_len + (dst_offset + l)) * d + d_idx;
        memcpy(dst_v_ptr + v_dst_idx * element_size, src_v_ptr + v_src_idx * element_size, element_size);
    }
}

// 将新 K 和 V 分别复制到对应 Cache 之后
template<typename T>
__global__ void copy_kv_to_cache_kernel(
    const T* key_input,         // 形状: B, L_k_new, H_kv, D
    const T* value_input,       // 形状: B, L_k_new, H_kv, D
    T* key_cache_output,        // 形状: L_kv_alloc, B, H_kv, D
    T* value_cache_output,      // 形状: B, H_kv, L_kv_alloc, D
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
    int input_offset = b_idx * new_kv_seq_len * kv_num_head * head_dim +
                             l_idx_new * kv_num_head * head_dim +
                             h_kv_idx * head_dim;

    T val_to_copy_k = key_input[input_offset + d_idx];
    T val_to_copy_v = value_input[input_offset + d_idx]; // 从相同偏移读取，因为K和V输入形状相同

    // Cache 中的目标序列索引 (在 past_kv_len 之后追加)
    int dest_seq_idx_cache = past_kv_len + l_idx_new;
    if (dest_seq_idx_cache >= allocated_kv_len) return; // 边界检查

    // Key Cache 输出: [L_kv_alloc, B, H_kv, D]
    int key_cache_idx = dest_seq_idx_cache * batch_size * kv_num_head * head_dim +
                              b_idx * kv_num_head * head_dim +
                              h_kv_idx * head_dim +
                              d_idx;
    key_cache_output[key_cache_idx] = val_to_copy_k;

    // Value Cache 输出: [B, H_kv, L_kv_alloc, D]
    int value_cache_idx = b_idx * kv_num_head * allocated_kv_len * head_dim +
                                h_kv_idx * allocated_kv_len * head_dim +
                                dest_seq_idx_cache * head_dim +
                                d_idx;
    value_cache_output[value_cache_idx] = val_to_copy_v;
}

// =====================================================================
// P0: Flash Decoding Kernel for decode stage (seq_len=1)
// - Each block handles one (batch, q_head) pair
// - Tiles along KV sequence dimension with online softmax
// - Warp-level reductions for QK dot product
// - No O(n^2) intermediate buffers needed
// =====================================================================

// Flash Decoding: each block computes one output head vector
// Supports GQA: multiple Q heads can share one KV head
// Uses online softmax (no intermediate QK buffer)
template<typename T>
__global__ void flash_decode_kernel(
    const T* __restrict__ query_input,    // [B, 1, H_q, D]
    const T* __restrict__ key_cache,      // [L_kv_alloc, B, H_kv, D]
    const T* __restrict__ value_cache,    // [B, H_kv, L_kv_alloc, D]
    T* __restrict__ output,               // [B, 1, H_q, D]
    const int batch,
    const int head_num,         // H_q
    const int kv_head_num,      // H_kv
    const int head_dim,         // D
    const int key_seq_len,      // total KV length
    const int max_kv_len,       // allocated KV length (stride for value cache)
    const float scale
) {
    // Each block: one (batch, head_q) pair
    const int bh_idx = blockIdx.x;
    const int b_idx = bh_idx / head_num;
    const int h_q_idx = bh_idx % head_num;
    const int h_kv_idx = h_q_idx / (head_num / kv_head_num);
    const int tid = threadIdx.x;
    const int WARP_SIZE = 32;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int num_warps = blockDim.x / WARP_SIZE;

    if (b_idx >= batch) return;

    // Load query vector into registers
    // Query: [B, 1, H_q, D] -> q[b_idx, 0, h_q_idx, :]
    const T* q_ptr = query_input + b_idx * head_num * head_dim + h_q_idx * head_dim;

    // Each thread loads multiple D elements
    // For head_dim=128 and 128 threads: each thread loads 1 element
    // For head_dim=128 and 256 threads: use multiple elements per thread across tiles

    // Shared memory for partial results across warps
    extern __shared__ char smem_raw[];
    float* smem_max = reinterpret_cast<float*>(smem_raw);                    // [num_warps]
    float* smem_sum = smem_max + num_warps;                                  // [num_warps]
    float* smem_out = smem_sum + num_warps;                                  // [num_warps * head_dim]

    // Online softmax state: track running max and sum per thread
    float thread_max = -1e20f;
    float thread_sum = 0.0f;

    // Accumulate output per thread: each thread is responsible for a subset of D
    // We accumulate O in registers for the D dims this thread owns
    float thread_out[8]; // max 8 D elements per thread (head_dim=256, 32 threads per warp)
    const int d_per_thread = (head_dim + blockDim.x - 1) / blockDim.x;
    for (int i = 0; i < d_per_thread && i < 8; i++) thread_out[i] = 0.0f;

    // Tile over KV sequence
    const int KV_TILE = 16; // Each iteration processes KV_TILE keys
    for (int kv_start = 0; kv_start < key_seq_len; kv_start += KV_TILE) {
        int kv_end = min(kv_start + KV_TILE, key_seq_len);

        // For each key in this tile, compute QK dot product and update online softmax
        for (int k = kv_start; k < kv_end; k++) {
            // Key Cache: [L_kv_alloc, B, H_kv, D]
            const T* k_ptr = key_cache + k * batch * kv_head_num * head_dim
                            + b_idx * kv_head_num * head_dim
                            + h_kv_idx * head_dim;

            // Compute QK dot product: each thread handles a subset of D
            float qk_partial = 0.0f;
            for (int d = tid; d < head_dim; d += blockDim.x) {
                qk_partial += (float)q_ptr[d] * (float)k_ptr[d];
            }

            // Warp-level reduction for QK dot product
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
                qk_partial += __shfl_xor_sync(0xffffffff, qk_partial, offset);
            }
            // Now lane 0 of each warp has partial sum; reduce across warps via shared memory
            if (lane_id == 0) {
                smem_max[warp_id] = qk_partial;
            }
            __syncthreads();

            float qk_score = 0.0f;
            if (tid == 0) {
                for (int w = 0; w < num_warps; w++) {
                    qk_score += smem_max[w];
                }
                qk_score *= scale;
                smem_max[0] = qk_score; // broadcast
            }
            __syncthreads();
            qk_score = smem_max[0];

            // Online softmax update
            float old_max = thread_max;
            float new_max = fmaxf(old_max, qk_score);
            float exp_diff = expf(old_max - new_max);
            float exp_score = expf(qk_score - new_max);

            // Rescale existing accumulator
            for (int i = 0; i < d_per_thread && i < 8; i++) {
                thread_out[i] *= exp_diff;
            }
            thread_sum = thread_sum * exp_diff + exp_score;
            thread_max = new_max;

            // Accumulate weighted value
            // Value Cache: [B, H_kv, L_kv_alloc, D]
            const T* v_base = value_cache + b_idx * kv_head_num * max_kv_len * head_dim
                            + h_kv_idx * max_kv_len * head_dim;

            for (int i = 0; i < d_per_thread && i < 8; i++) {
                int d = tid + i * blockDim.x;
                if (d < head_dim) {
                    float v_val = (float)v_base[k * head_dim + d];
                    thread_out[i] += exp_score * v_val;
                }
            }
        }
    }

    // Final normalize: divide by sum
    float inv_sum = (thread_sum > 0.0f) ? (1.0f / thread_sum) : 0.0f;

    // Write output: [B, 1, H_q, D]
    T* out_ptr = output + b_idx * head_num * head_dim + h_q_idx * head_dim;
    for (int i = 0; i < d_per_thread && i < 8; i++) {
        int d = tid + i * blockDim.x;
        if (d < head_dim) {
            out_ptr[d] = (T)(thread_out[i] * inv_sum);
        }
    }
}


// =====================================================================
// Flash Decode Kernel with Mask support for speculative decoding
// - Each block handles one (batch, q_head, q_idx) triple
// - Uses online softmax like flash_decode_kernel (single pass, no temp buffers)
// - Supports additive mask for the new KV region (tree attention)
// - Optimal for small seq_len (e.g., 4-32 tokens) with large KV cache
// =====================================================================
template<typename T>
__global__ void flash_decode_kernel_with_mask(
    const T* __restrict__ query_input,    // [B, L_q, H_q, D]
    const T* __restrict__ key_cache,      // [L_kv_alloc, B, H_kv, D]
    const T* __restrict__ value_cache,    // [B, H_kv, L_kv_alloc, D]
    T* __restrict__ output,               // [B, L_q, H_q, D]
    const T* __restrict__ mask,           // [1, 1, L_q, L_q] additive mask for new KV region
    const int batch,
    const int head_num,         // H_q
    const int kv_head_num,      // H_kv
    const int head_dim,         // D
    const int key_seq_len,      // total KV length (past + new)
    const int max_kv_len,       // allocated KV length (stride for value cache)
    const int query_seq_len,    // L_q (number of new/query tokens)
    const float scale
) {
    // Each block: one (batch, head_q, q_idx) triple
    const int idx = blockIdx.x;
    const int q_idx = idx % query_seq_len;
    const int bh_idx = idx / query_seq_len;
    const int b_idx = bh_idx / head_num;
    const int h_q_idx = bh_idx % head_num;
    const int h_kv_idx = h_q_idx / (head_num / kv_head_num);
    const int tid = threadIdx.x;
    const int WARP_SIZE = 32;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int num_warps = blockDim.x / WARP_SIZE;

    if (b_idx >= batch) return;

    // Load query vector: [B, L_q, H_q, D]
    const T* q_ptr = query_input + b_idx * query_seq_len * head_num * head_dim
                    + q_idx * head_num * head_dim + h_q_idx * head_dim;

    // Mask region: keys at index >= (key_seq_len - query_seq_len) are new tokens
    const int past_kv_len = key_seq_len - query_seq_len;

    // Shared memory for warp-level reductions
    extern __shared__ char smem_raw[];
    float* smem = reinterpret_cast<float*>(smem_raw); // [num_warps]

    // Online softmax state
    float thread_max = -1e20f;
    float thread_sum = 0.0f;

    // Per-thread output accumulator
    float thread_out[8];
    const int d_per_thread = (head_dim + blockDim.x - 1) / blockDim.x;
    for (int i = 0; i < d_per_thread && i < 8; i++) thread_out[i] = 0.0f;

    // Value cache base pointer: [B, H_kv, L_kv_alloc, D]
    const T* v_base = value_cache + b_idx * kv_head_num * max_kv_len * head_dim
                    + h_kv_idx * max_kv_len * head_dim;

    // Iterate over all KV positions
    for (int k = 0; k < key_seq_len; k++) {
        // Key Cache: [L_kv_alloc, B, H_kv, D]
        const T* k_ptr = key_cache + k * batch * kv_head_num * head_dim
                        + b_idx * kv_head_num * head_dim
                        + h_kv_idx * head_dim;

        // Compute QK dot product
        float qk_partial = 0.0f;
        for (int d = tid; d < head_dim; d += blockDim.x) {
            qk_partial += (float)q_ptr[d] * (float)k_ptr[d];
        }

        // Warp-level reduction
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            qk_partial += __shfl_xor_sync(0xffffffff, qk_partial, offset);
        }
        if (lane_id == 0) {
            smem[warp_id] = qk_partial;
        }
        __syncthreads();

        float qk_score = 0.0f;
        if (tid == 0) {
            for (int w = 0; w < num_warps; w++) {
                qk_score += smem[w];
            }
            qk_score *= scale;

            // Apply additive mask for new KV positions
            if (k >= past_kv_len) {
                int new_k_idx = k - past_kv_len;
                // mask: [1, 1, L_q, L_q], row=q_idx, col=new_k_idx
                float mask_val = (float)mask[q_idx * query_seq_len + new_k_idx];
                qk_score += mask_val;
            }

            smem[0] = qk_score;
        }
        __syncthreads();
        qk_score = smem[0];

        // Online softmax update
        float old_max = thread_max;
        float new_max = fmaxf(old_max, qk_score);
        float exp_diff = expf(old_max - new_max);
        float exp_score = expf(qk_score - new_max);

        for (int i = 0; i < d_per_thread && i < 8; i++) {
            thread_out[i] *= exp_diff;
        }
        thread_sum = thread_sum * exp_diff + exp_score;
        thread_max = new_max;

        // Accumulate weighted value
        for (int i = 0; i < d_per_thread && i < 8; i++) {
            int d = tid + i * blockDim.x;
            if (d < head_dim) {
                float v_val = (float)v_base[k * head_dim + d];
                thread_out[i] += exp_score * v_val;
            }
        }
    }

    // Final normalize
    float inv_sum = (thread_sum > 0.0f) ? (1.0f / thread_sum) : 0.0f;

    // Write output: [B, L_q, H_q, D]
    T* out_ptr = output + b_idx * query_seq_len * head_num * head_dim
                + q_idx * head_num * head_dim + h_q_idx * head_dim;
    for (int i = 0; i < d_per_thread && i < 8; i++) {
        int d = tid + i * blockDim.x;
        if (d < head_dim) {
            out_ptr[d] = (T)(thread_out[i] * inv_sum);
        }
    }
}

// =====================================================================
// OPT-2: Split-K Flash Decode Kernel
// Multiple blocks cooperate on a single (batch, head) pair,
// each processing a range of KV positions.
// =====================================================================
template<typename T>
__global__ void flash_decode_kernel_splitk(
    const T* __restrict__ query_input,    // [B, 1, H_q, D]
    const T* __restrict__ key_cache,      // [L_kv_alloc, B, H_kv, D]
    const T* __restrict__ value_cache,    // [B, H_kv, L_kv_alloc, D]
    float* __restrict__ partial_output,   // [parallel_blocks, B*H_q, D]
    float* __restrict__ partial_meta,     // [parallel_blocks, B*H_q, 2] (max, sum)
    const int batch,
    const int head_num,
    const int kv_head_num,
    const int head_dim,
    const int key_seq_len,
    const int max_kv_len,
    const float scale_factor,
    const int parallel_blocks
) {
    const int bh_idx = blockIdx.x;         // batch * head_num
    const int split_idx = blockIdx.y;       // which split block
    const int b_idx = bh_idx / head_num;
    const int h_q_idx = bh_idx % head_num;
    const int h_kv_idx = h_q_idx / (head_num / kv_head_num);
    const int tid = threadIdx.x;

    if (b_idx >= batch) return;

    // Compute KV range for this split block
    const int kv_per_block = (key_seq_len + parallel_blocks - 1) / parallel_blocks;
    const int kv_start = split_idx * kv_per_block;
    const int kv_end = min(kv_start + kv_per_block, key_seq_len);
    if (kv_start >= key_seq_len) {
        // This block has no work — write identity metadata
        if (tid == 0) {
            partial_meta[(split_idx * batch * head_num + bh_idx) * 2 + 0] = -1e20f; // max
            partial_meta[(split_idx * batch * head_num + bh_idx) * 2 + 1] = 0.0f;   // sum
        }
        const int d_per_thread = (head_dim + blockDim.x - 1) / blockDim.x;
        float* out_ptr = partial_output + (split_idx * batch * head_num + bh_idx) * head_dim;
        for (int i = 0; i < d_per_thread && i < 8; i++) {
            int d = tid + i * blockDim.x;
            if (d < head_dim) out_ptr[d] = 0.0f;
        }
        return;
    }

    // Load query vector
    const T* q_ptr = query_input + b_idx * head_num * head_dim + h_q_idx * head_dim;

    const int WARP_SIZE_LOCAL = 32;
    const int warp_id = tid / WARP_SIZE_LOCAL;
    const int lane_id = tid % WARP_SIZE_LOCAL;
    const int num_warps = blockDim.x / WARP_SIZE_LOCAL;

    extern __shared__ char smem_raw_splitk[];
    float* smem_qk = reinterpret_cast<float*>(smem_raw_splitk);  // [num_warps]

    float thread_max = -1e20f;
    float thread_sum = 0.0f;

    const int d_per_thread = (head_dim + blockDim.x - 1) / blockDim.x;
    float thread_out[8];
    for (int i = 0; i < d_per_thread && i < 8; i++) thread_out[i] = 0.0f;

    // Process KV positions in this block's range
    for (int k = kv_start; k < kv_end; k++) {
        // Key: [L_kv_alloc, B, H_kv, D]
        const T* k_ptr = key_cache + k * batch * kv_head_num * head_dim
                        + b_idx * kv_head_num * head_dim
                        + h_kv_idx * head_dim;

        // Compute QK dot product
        float qk_partial = 0.0f;
        for (int d = tid; d < head_dim; d += blockDim.x) {
            qk_partial += (float)q_ptr[d] * (float)k_ptr[d];
        }

        // Warp reduction
        for (int offset = WARP_SIZE_LOCAL / 2; offset > 0; offset >>= 1) {
            qk_partial += __shfl_xor_sync(0xffffffff, qk_partial, offset);
        }
        if (lane_id == 0) smem_qk[warp_id] = qk_partial;
        __syncthreads();

        float qk_score = 0.0f;
        if (tid == 0) {
            for (int w = 0; w < num_warps; w++) qk_score += smem_qk[w];
            qk_score *= scale_factor;
            smem_qk[0] = qk_score;
        }
        __syncthreads();
        qk_score = smem_qk[0];

        // Online softmax
        float old_max = thread_max;
        float new_max = fmaxf(old_max, qk_score);
        float exp_diff = expf(old_max - new_max);
        float exp_score = expf(qk_score - new_max);

        for (int i = 0; i < d_per_thread && i < 8; i++) thread_out[i] *= exp_diff;
        thread_sum = thread_sum * exp_diff + exp_score;
        thread_max = new_max;

        // Accumulate weighted value
        const T* v_base = value_cache + b_idx * kv_head_num * max_kv_len * head_dim
                        + h_kv_idx * max_kv_len * head_dim;
        for (int i = 0; i < d_per_thread && i < 8; i++) {
            int d = tid + i * blockDim.x;
            if (d < head_dim) {
                float v_val = (float)v_base[k * head_dim + d];
                thread_out[i] += exp_score * v_val;
            }
        }
    }

    // Write partial results (unnormalized) and metadata
    float* out_ptr = partial_output + (split_idx * batch * head_num + bh_idx) * head_dim;
    for (int i = 0; i < d_per_thread && i < 8; i++) {
        int d = tid + i * blockDim.x;
        if (d < head_dim) {
            out_ptr[d] = thread_out[i]; // unnormalized: sum(exp(qk - local_max) * v)
        }
    }
    if (tid == 0) {
        partial_meta[(split_idx * batch * head_num + bh_idx) * 2 + 0] = thread_max;
        partial_meta[(split_idx * batch * head_num + bh_idx) * 2 + 1] = thread_sum;
    }
}

// =====================================================================
// OPT-2: Combine kernel — merge partial results from split-K blocks
// Grid: (batch * head_num), Block: head_dim threads
// =====================================================================
template<typename T>
__global__ void flash_attn_combine_results(
    const float* __restrict__ partial_output,  // [parallel_blocks, B*H_q, D]
    const float* __restrict__ partial_meta,    // [parallel_blocks, B*H_q, 2]
    T* __restrict__ final_output,              // [B, 1, H_q, D]
    const int batch,
    const int head_num,
    const int head_dim,
    const int parallel_blocks
) {
    const int bh_idx = blockIdx.x;
    const int d = threadIdx.x;
    if (d >= head_dim) return;

    const int b_idx = bh_idx / head_num;
    const int h_q_idx = bh_idx % head_num;

    // Find global max across all blocks
    float global_max = -1e20f;
    for (int s = 0; s < parallel_blocks; s++) {
        float local_max = partial_meta[(s * batch * head_num + bh_idx) * 2 + 0];
        global_max = fmaxf(global_max, local_max);
    }

    // Combine: rescale each block's partial output and sum
    float combined_output = 0.0f;
    float combined_sum = 0.0f;

    for (int s = 0; s < parallel_blocks; s++) {
        float local_max = partial_meta[(s * batch * head_num + bh_idx) * 2 + 0];
        float local_sum = partial_meta[(s * batch * head_num + bh_idx) * 2 + 1];
        float rescale = expf(local_max - global_max);

        combined_output += partial_output[(s * batch * head_num + bh_idx) * head_dim + d] * rescale;
        combined_sum += local_sum * rescale;
    }

    // Normalize
    float inv_sum = (combined_sum > 0.0f) ? (1.0f / combined_sum) : 0.0f;
    T* out_ptr = final_output + b_idx * head_num * head_dim + h_q_idx * head_dim;
    out_ptr[d] = (T)(combined_output * inv_sum);
}

// =====================================================================
// Optimized Prefill QK Kernel: uses shared memory tiling
// =====================================================================
template<typename T, typename AccT = float>
__global__ void qk_kernel_tiled(
    const T* __restrict__ query_input,    // [B, L_q_full, H_q, D]
    const T* __restrict__ key_cache,      // [L_k_total_alloc, B, H_kv, D]
    T* __restrict__ qk_scores_output,     // [B, H_q, L_q_piece, L_k_total]
    const void* mask_tensor_data,
    const AttentionKernelParam* param,
    int q_seq_piece_offset,
    bool has_mask_flag,
    bool is_add_mask_flag
) {
    // Block handles a tile of [QK_TILE_Q x QK_TILE_K] outputs
    const int QK_TILE = 16;
    __shared__ AccT q_tile[QK_TILE][128 + 1]; // +1 to avoid bank conflict, max head_dim=128
    __shared__ AccT k_tile[QK_TILE][128 + 1];

    const int tx = threadIdx.x; // within tile K dim
    const int ty = threadIdx.y; // within tile Q dim
    const int bh_q_idx = blockIdx.z;

    if (bh_q_idx >= param->batch * param->head_num) return;

    const int b_idx = bh_q_idx / param->head_num;
    const int h_q_idx = bh_q_idx % param->head_num;
    const int h_kv_idx = h_q_idx / param->group;

    const int q_idx_in_piece = blockIdx.y * QK_TILE + ty;
    const int k_idx = blockIdx.x * QK_TILE + tx;
    const int current_full_q_idx = q_seq_piece_offset + q_idx_in_piece;

    AccT score_sum = 0.0f;

    // Tile over head_dim
    const int D_TILE = 32;
    for (int d_start = 0; d_start < param->head_dim; d_start += D_TILE) {
        // Load Q tile into shared memory
        if (current_full_q_idx < param->query_seq_len && q_idx_in_piece < param->q_seq_piece_len) {
            for (int dd = tx; dd < D_TILE && (d_start + dd) < param->head_dim; dd += QK_TILE) {
                int q_offset = b_idx * param->query_seq_len * param->head_num * param->head_dim
                             + current_full_q_idx * param->head_num * param->head_dim
                             + h_q_idx * param->head_dim + d_start + dd;
                q_tile[ty][dd] = (AccT)query_input[q_offset];
            }
        } else {
            for (int dd = tx; dd < D_TILE; dd += QK_TILE) {
                q_tile[ty][dd] = AccT(0.0f);
            }
        }

        // Load K tile into shared memory
        if (k_idx < param->key_seq_len) {
            for (int dd = ty; dd < D_TILE && (d_start + dd) < param->head_dim; dd += QK_TILE) {
                int k_offset = k_idx * param->batch * param->kv_head_num * param->head_dim
                             + b_idx * param->kv_head_num * param->head_dim
                             + h_kv_idx * param->head_dim + d_start + dd;
                k_tile[tx][dd] = (AccT)key_cache[k_offset];
            }
        } else {
            for (int dd = ty; dd < D_TILE; dd += QK_TILE) {
                k_tile[tx][dd] = AccT(0.0f);
            }
        }

        __syncthreads();

        // Dot product for this D tile
        int d_end = min(D_TILE, param->head_dim - d_start);
        #pragma unroll 8
        for (int dd = 0; dd < d_end; dd++) {
            score_sum += q_tile[ty][dd] * k_tile[tx][dd];
        }

        __syncthreads();
    }

    if (k_idx >= param->key_seq_len || q_idx_in_piece >= param->q_seq_piece_len || current_full_q_idx >= param->query_seq_len) {
        return;
    }

    score_sum *= param->scale;

    // Apply mask
    if (has_mask_flag && mask_tensor_data) {
        if (is_add_mask_flag) {
            int mask_idx = current_full_q_idx * param->query_seq_len + k_idx - param->key_seq_len + param->query_seq_len;
            if (k_idx >= param->key_seq_len - param->query_seq_len) {
                if (sizeof(T) == sizeof(__half)) {
                    score_sum += __half2float(((const __half*)mask_tensor_data)[mask_idx]);
                } else {
                    score_sum += static_cast<const AccT*>(mask_tensor_data)[mask_idx];
                }
            }
        } else {
            int mask_idx = current_full_q_idx * param->key_seq_len + k_idx;
            if (static_cast<const int*>(mask_tensor_data)[mask_idx] == 0) {
                score_sum = (sizeof(T) == sizeof(__half)) ? AccT(-65504.0f) : AccT(-1e9f);
            }
        }
    }

    if (sizeof(T) == sizeof(__half)) {
        const AccT max_half_val = AccT(65504.0f);
        score_sum = fminf(fmaxf(score_sum, -max_half_val), max_half_val);
    }

    int out_idx = b_idx * param->head_num * param->q_seq_piece_len * param->key_seq_len +
                  h_q_idx * param->q_seq_piece_len * param->key_seq_len +
                  q_idx_in_piece * param->key_seq_len + k_idx;
    qk_scores_output[out_idx] = static_cast<T>(score_sum);
}


// =====================================================================
// Optimized QKV Kernel: shared memory tiling for V accumulation
// =====================================================================
template<typename T, typename AccT = float>
__global__ void qkv_kernel_tiled(
    const T* __restrict__ softmax_probs,  // [B, H_q, L_q_piece, L_k_total]
    const T* __restrict__ value_cache,    // [B, H_kv, L_k_alloc_max, D]
    T* __restrict__ attention_output,     // [B, L_q_full, H_q, D]
    const AttentionKernelParam* param,
    int q_seq_piece_offset
) {
    // Each thread computes one output element
    const int d_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int q_idx_in_piece = blockIdx.y * blockDim.y + threadIdx.y;
    const int bh_q_idx = blockIdx.z;

    if (d_idx >= param->head_dim || q_idx_in_piece >= param->q_seq_piece_len || bh_q_idx >= param->batch * param->head_num) {
        return;
    }

    const int b_idx = bh_q_idx / param->head_num;
    const int h_q_idx = bh_q_idx % param->head_num;
    const int current_full_q_idx = q_seq_piece_offset + q_idx_in_piece;

    if (current_full_q_idx >= param->query_seq_len) return;

    const int h_kv_idx = h_q_idx / param->group;

    AccT weighted_sum = 0.0f;

    const T* prob_ptr = softmax_probs + b_idx * param->head_num * param->q_seq_piece_len * param->key_seq_len +
                        h_q_idx * param->q_seq_piece_len * param->key_seq_len +
                        q_idx_in_piece * param->key_seq_len;

    const T* val_ptr_base = value_cache + b_idx * param->kv_head_num * param->max_kv_len * param->head_dim +
                            h_kv_idx * param->max_kv_len * param->head_dim;

    // Unrolled loop for better ILP
    const int hd = param->head_dim;
    int k_s = 0;
    for (; k_s + 3 < param->key_seq_len; k_s += 4) {
        weighted_sum += (AccT)prob_ptr[k_s]     * (AccT)val_ptr_base[k_s * hd + d_idx];
        weighted_sum += (AccT)prob_ptr[k_s + 1] * (AccT)val_ptr_base[(k_s + 1) * hd + d_idx];
        weighted_sum += (AccT)prob_ptr[k_s + 2] * (AccT)val_ptr_base[(k_s + 2) * hd + d_idx];
        weighted_sum += (AccT)prob_ptr[k_s + 3] * (AccT)val_ptr_base[(k_s + 3) * hd + d_idx];
    }
    for (; k_s < param->key_seq_len; k_s++) {
        weighted_sum += (AccT)prob_ptr[k_s] * (AccT)val_ptr_base[k_s * hd + d_idx];
    }

    int out_idx = b_idx * param->query_seq_len * param->head_num * param->head_dim +
                  current_full_q_idx * param->head_num * param->head_dim +
                  h_q_idx * param->head_dim + d_idx;
    attention_output[out_idx] = static_cast<T>(weighted_sum);
}


// ======= AttentionExecution 类实现 =======

AttentionExecution::AttentionExecution(Backend* backend, bool kv_cache_op_param)
    : Execution(backend), mIsKVCacheEnabled(kv_cache_op_param), mCudaBackend(static_cast<CUDABackend*>(backend)),
      mBatch(0), mQuerySeqLen(0), mNumHead(0), mHeadDim(0), mKvNumHead(0), mNewKvSeqLen(0),
      mQseqSplitNum(1), mHasMask(false), mIsAddMask(false), mParam_gpu(nullptr), mScale(1.0f) {
    mPrecision = 4; // 默认精度,可在 onResize 中更改
    if (mIsKVCacheEnabled) {
        mCache.reset(new SharedCache());
        mMeta = (KVMeta*)(mCudaBackend->getMetaPtr());
    }
}

AttentionExecution::~AttentionExecution() {
    if (mParam_gpu) {
        cudaFree(mParam_gpu);
        mParam_gpu = nullptr;
    }
    if (mSplitKOutputPtr) {
        cudaFree(mSplitKOutputPtr);
        mSplitKOutputPtr = nullptr;
    }
    if (mSplitKMetaPtr) {
        cudaFree(mSplitKMetaPtr);
        mSplitKMetaPtr = nullptr;
    }
}

// 初始化一个大小为 1 的占位 KVCache
ErrorCode AttentionExecution::init_cache_tensors() {
    if (!mIsKVCacheEnabled || !mCache) return MNN::NO_ERROR;
    if (mCache->mPastKey && mCache->mPastValue && mCache->mPastKey->deviceId() != 0) return MNN::NO_ERROR;

    mCache->mPastLength = 0;
    mCache->mMaxLength = 0;

    mCache->mPastKey.reset(mPrecision == 4
        ? Tensor::createDevice<float>({1, 1, 1, 1})
        : Tensor::createDevice<uint16_t>({1, 1, 1, 1}));
    mCache->mPastValue.reset(mPrecision == 4
        ? Tensor::createDevice<float>({1, 1, 1, 1})
        : Tensor::createDevice<uint16_t>({1, 1, 1, 1}));
    if (!mCache->mPastKey || !mCache->mPastValue) return MNN::OUT_OF_MEMORY;

    bool res = mCudaBackend->onAcquireBuffer(mCache->mPastKey.get(), Backend::STATIC);
    if (!res) return MNN::OUT_OF_MEMORY;
    res = mCudaBackend->onAcquireBuffer(mCache->mPastValue.get(), Backend::STATIC);
    if (!res) return MNN::OUT_OF_MEMORY;

    return MNN::NO_ERROR;
}

// P2: Optimized KV Cache realloc with 2x growth strategy and batch memcpy
ErrorCode AttentionExecution::reallocKVCache_gpu(int required_total_kv_len, int batch_size, int kv_num_head, int head_dim, cudaStream_t stream) {
    if (!mIsKVCacheEnabled || !mCache) return MNN::NO_ERROR;

    if (required_total_kv_len > mCache->mMaxLength || mCache->mPastKey->deviceId() == 0) {
        int old_max_len = mCache->mMaxLength;
        int old_past_len = mCache->mPastLength;
        bool needs_copy = old_past_len > 0 && mCache->mPastKey && mCache->mPastKey->deviceId() != 0;

        // P2: 2x growth strategy to reduce realloc frequency
        int new_allocated_max_len = ROUND_UP(required_total_kv_len, mExpandChunk);
        if (old_max_len > 0 && new_allocated_max_len < old_max_len * 2) {
            new_allocated_max_len = std::max(new_allocated_max_len, old_max_len * 2);
        }
        if (new_allocated_max_len <= old_max_len && mCache->mPastKey->deviceId() != 0) {
            return MNN::NO_ERROR;
        }

        std::shared_ptr<Tensor> new_past_key_tensor(mPrecision == 4
            ? Tensor::createDevice<float>({new_allocated_max_len, batch_size, kv_num_head, head_dim})
            : Tensor::createDevice<uint16_t>({new_allocated_max_len, batch_size, kv_num_head, head_dim}));
        std::shared_ptr<Tensor> new_past_value_tensor(mPrecision == 4
            ? Tensor::createDevice<float>({batch_size, kv_num_head, new_allocated_max_len, head_dim})
            : Tensor::createDevice<uint16_t>({batch_size, kv_num_head, new_allocated_max_len, head_dim}));

        if (!new_past_key_tensor || !new_past_value_tensor) return MNN::OUT_OF_MEMORY;

        bool resK = mCudaBackend->onAcquireBuffer(new_past_key_tensor.get(), Backend::STATIC);
        bool resV = mCudaBackend->onAcquireBuffer(new_past_value_tensor.get(), Backend::STATIC);
        if(!resK || !resV) return MNN::OUT_OF_MEMORY;

        if (needs_copy) {
            size_t element_size_bytes = mPrecision;
            // Key Cache: contiguous [old_past_len, B, H_kv, D] -> single memcpy
            size_t key_bytes_to_copy = (size_t)old_past_len * batch_size * kv_num_head * head_dim * element_size_bytes;
            if (key_bytes_to_copy > 0) {
                 cudaMemcpyAsync(getTensorDevicePtr(new_past_key_tensor.get()),
                                 getTensorDevicePtr(mCache->mPastKey.get()),
                                 key_bytes_to_copy, cudaMemcpyDeviceToDevice, stream);
                 checkKernelErrors;
            }

            // P2: Value Cache: use cudaMemcpy2DAsync for batch copy
            // Value layout: [B, H_kv, L_kv_max, D], each (B,H) group has L*D contiguous elements
            if (old_past_len > 0) {
                cudaMemcpy2DAsync(
                    getTensorDevicePtr<uint8_t>(new_past_value_tensor.get()),
                    (size_t)new_allocated_max_len * head_dim * element_size_bytes,  // dst pitch
                    getTensorDevicePtr<uint8_t>(mCache->mPastValue.get()),
                    (size_t)old_max_len * head_dim * element_size_bytes,             // src pitch
                    (size_t)old_past_len * head_dim * element_size_bytes,            // width to copy
                    (size_t)batch_size * kv_num_head,                                // height (num groups)
                    cudaMemcpyDeviceToDevice, stream);
                checkKernelErrors;
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

// P2: Optimized KV Cache realloc with mMeta version
ErrorCode AttentionExecution::reallocKVCache_gpu(int required_total_kv_len, const KVMeta* meta, cudaStream_t stream) {
    if (!mIsKVCacheEnabled || !mCache) {
        return MNN::NO_ERROR;
    }
    if (!meta) {
        MNN_ERROR("KVMeta is null in reallocKVCache_gpu, which is required for dynamic cache management.\n");
        return MNN::INVALID_VALUE;
    }

    size_t element_size = mPrecision;
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
        dim3 gridDim(mBatch * mKvNumHead, meta->n_reserve, 1);

        if (needs_realloc) {
            // P2: 2x growth strategy
            int new_allocated_max_len = ROUND_UP(required_total_kv_len, mExpandChunk);
            if (mCache->mMaxLength > 0) {
                new_allocated_max_len = std::max(new_allocated_max_len, mCache->mMaxLength * 2);
            }
            std::shared_ptr<Tensor> new_past_key_tensor(mPrecision == 4
                ? Tensor::createDevice<float>({new_allocated_max_len, mBatch, mKvNumHead, mHeadDim})
                : Tensor::createDevice<uint16_t>({new_allocated_max_len, mBatch, mKvNumHead, mHeadDim}));
            std::shared_ptr<Tensor> new_past_value_tensor(mPrecision == 4
                ? Tensor::createDevice<float>({mBatch, mKvNumHead, new_allocated_max_len, mHeadDim})
                : Tensor::createDevice<uint16_t>({mBatch, mKvNumHead, new_allocated_max_len, mHeadDim}));
            if(!mCudaBackend->onAcquireBuffer(new_past_key_tensor.get(), Backend::STATIC)
            || !mCudaBackend->onAcquireBuffer(new_past_value_tensor.get(), Backend::STATIC)) {
                return MNN::OUT_OF_MEMORY;
            }

            // Copy preserved data [0, past_len_after_remove) from old to new cache
            if (past_len_after_remove > 0) {
                // Key: [L, B, H_kv, D] — contiguous in first dimension
                cudaMemcpyAsync(getTensorDevicePtr(new_past_key_tensor.get()),
                    getTensorDevicePtr(mCache->mPastKey.get()),
                    (size_t)past_len_after_remove * mBatch * mKvNumHead * mHeadDim * element_size,
                    cudaMemcpyDeviceToDevice, stream);
                // Value: [B, H_kv, L, D] — strided copy
                cudaMemcpy2DAsync(getTensorDevicePtr(new_past_value_tensor.get()),
                    (size_t)new_allocated_max_len * mHeadDim * element_size,
                    getTensorDevicePtr(mCache->mPastValue.get()),
                    (size_t)mCache->mMaxLength * mHeadDim * element_size,
                    (size_t)past_len_after_remove * mHeadDim * element_size,
                    (size_t)mBatch * mKvNumHead,
                    cudaMemcpyDeviceToDevice, stream);
            }

            // Compact reserve data from old cache into new cache
            compact_kv_cache_kernel<<<gridDim, blockDim, 0, stream>>>(
                getTensorDevicePtr(mCache->mPastKey.get()), getTensorDevicePtr(mCache->mPastValue.get()),
                getTensorDevicePtr(new_past_key_tensor.get()), getTensorDevicePtr(new_past_value_tensor.get()),
                reserve_info_gpu, reserve_offsets_gpu,
                meta->n_reserve, past_len_after_remove,
                mBatch, mKvNumHead, mHeadDim, mCache->mMaxLength, new_allocated_max_len, element_size
            );

            mCudaBackend->onReleaseBuffer(mCache->mPastKey.get(), Backend::STATIC);
            mCudaBackend->onReleaseBuffer(mCache->mPastValue.get(), Backend::STATIC);
            mCache->mPastKey = new_past_key_tensor;
            mCache->mPastValue = new_past_value_tensor;
            mCache->mMaxLength = new_allocated_max_len;
        } else {
            // No realloc: compact reserve data in-place directly
            // For speculative decoding, reserve entries are at positions after past_len_after_remove,
            // and destination offsets are always <= source offsets (sorted accept indices),
            // so forward-order in-place copy is safe without temp buffers.
            // [0, past_len_after_remove) is already in place and doesn't need copying.
            compact_kv_cache_kernel<<<gridDim, blockDim, 0, stream>>>(
                getTensorDevicePtr(mCache->mPastKey.get()), getTensorDevicePtr(mCache->mPastValue.get()),
                getTensorDevicePtr(mCache->mPastKey.get()), getTensorDevicePtr(mCache->mPastValue.get()),
                reserve_info_gpu, reserve_offsets_gpu,
                meta->n_reserve, past_len_after_remove,
                mBatch, mKvNumHead, mHeadDim, mCache->mMaxLength, mCache->mMaxLength, element_size
            );
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
            // P2: 2x growth strategy
            int new_allocated_max_len = ROUND_UP(required_total_kv_len, mExpandChunk);
            if (old_max_len > 0) {
                new_allocated_max_len = std::max(new_allocated_max_len, old_max_len * 2);
            }

            std::shared_ptr<Tensor> new_past_key_tensor(mPrecision == 4
                ? Tensor::createDevice<float>({new_allocated_max_len, mBatch, mKvNumHead, mHeadDim})
                : Tensor::createDevice<uint16_t>({new_allocated_max_len, mBatch, mKvNumHead, mHeadDim}));
            std::shared_ptr<Tensor> new_past_value_tensor(mPrecision == 4
                ? Tensor::createDevice<float>({mBatch, mKvNumHead, new_allocated_max_len, mHeadDim})
                : Tensor::createDevice<uint16_t>({mBatch, mKvNumHead, new_allocated_max_len, mHeadDim}));
            if(!mCudaBackend->onAcquireBuffer(new_past_key_tensor.get(), Backend::STATIC)
            || !mCudaBackend->onAcquireBuffer(new_past_value_tensor.get(), Backend::STATIC)) {
                return MNN::OUT_OF_MEMORY;
            }

            if (old_past_len_to_copy > 0) {
                cudaMemcpyAsync(getTensorDevicePtr(new_past_key_tensor.get()),
                    getTensorDevicePtr(mCache->mPastKey.get()),
                    (size_t)old_past_len_to_copy * mBatch * mKvNumHead * mHeadDim * element_size,
                    cudaMemcpyDeviceToDevice, stream);
                // P2: Use cudaMemcpy2DAsync — Value layout: [B, H_kv, L, D]
                cudaMemcpy2DAsync(getTensorDevicePtr(new_past_value_tensor.get()),
                    (size_t)new_allocated_max_len * mHeadDim * element_size,
                    getTensorDevicePtr(mCache->mPastValue.get()),
                    (size_t)old_max_len * mHeadDim * element_size,
                    (size_t)old_past_len_to_copy * mHeadDim * element_size,
                    (size_t)mBatch * mKvNumHead,
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
        mTempQK.reset(mPrecision == 4
            ? Tensor::createDevice<float>(qk_shape)
            : Tensor::createDevice<uint16_t>(qk_shape));
        if(!mTempQK || !mCudaBackend->onAcquireBuffer(mTempQK.get(), Backend::STATIC)) { MNN_ERROR("Attention: mTempQK STATIC alloc failed\n"); return MNN::OUT_OF_MEMORY; }
    }

    // Softmax Probs: 与QK scores形状相同
    bool softmax_realloc = !mTempSoftmax || mTempSoftmax->shape() != qk_shape;
    if (softmax_realloc) {
         if(mTempSoftmax && mTempSoftmax->deviceId() != 0) mCudaBackend->onReleaseBuffer(mTempSoftmax.get(), Backend::STATIC);
        mTempSoftmax.reset(mPrecision == 4
            ? Tensor::createDevice<float>(qk_shape)
            : Tensor::createDevice<uint16_t>(qk_shape));
        if(!mTempSoftmax || !mCudaBackend->onAcquireBuffer(mTempSoftmax.get(), Backend::STATIC)) { MNN_ERROR("Attention: mTempSoftmax STATIC alloc failed\n"); return MNN::OUT_OF_MEMORY; }
    }

    if (!mIsKVCacheEnabled) {
        int temp_k_alloc_len = ROUND_UP(mNewKvSeqLen, 1);
        std::vector<int> temp_k_shape = {temp_k_alloc_len, batch, mKvNumHead, head_dim};
        bool temp_k_realloc = !mTempK_current_step || mTempK_current_step->shape() != temp_k_shape;
        if(temp_k_realloc){
            if(mTempK_current_step && mTempK_current_step->deviceId() !=0) mCudaBackend->onReleaseBuffer(mTempK_current_step.get(), Backend::STATIC);
            mTempK_current_step.reset(mPrecision == 4
                ? Tensor::createDevice<float>(temp_k_shape)
                : Tensor::createDevice<uint16_t>(temp_k_shape) );
            if(!mTempK_current_step || !mCudaBackend->onAcquireBuffer(mTempK_current_step.get(), Backend::STATIC)) return MNN::OUT_OF_MEMORY;
        }

        std::vector<int> temp_v_shape = {batch, mKvNumHead, temp_k_alloc_len, head_dim};
        bool temp_v_realloc = !mTempV_current_step || mTempV_current_step->shape() != temp_v_shape;
         if(temp_v_realloc){
            if(mTempV_current_step && mTempV_current_step->deviceId() !=0) mCudaBackend->onReleaseBuffer(mTempV_current_step.get(), Backend::STATIC);
            mTempV_current_step.reset(mPrecision == 4
                ? Tensor::createDevice<float>(temp_v_shape)
                : Tensor::createDevice<uint16_t>(temp_v_shape));
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

// 初始化所有参数
ErrorCode AttentionExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    const auto* query_tensor = inputs[0]; // 形状: [B, L_q, H_q, D]
    const auto* key_tensor = inputs[1];   // 形状: [B, L_k_new, H_kv, D]

    if (mCudaBackend->useFp16()) {
        mPrecision = 2;
    } else {
        mPrecision = 4;
    }

    mBatch = query_tensor->length(0);
    mQuerySeqLen = query_tensor->length(1);
    mNumHead = query_tensor->length(2);
    mHeadDim = query_tensor->length(3);

    mKvNumHead = key_tensor->length(2);
    mNewKvSeqLen = key_tensor->length(1);

    if (mHeadDim == 0 || mKvNumHead == 0) return MNN::INVALID_VALUE;
    if (mNumHead % mKvNumHead != 0) return MNN::INVALID_VALUE;
    mScale = 1.0f / sqrtf(static_cast<float>(mHeadDim));

    mHasMask = inputs.size() > 3 && inputs[3] != nullptr && inputs[3]->elementSize() > 0;
    if (mHasMask) {
        mIsAddMask = (inputs[3]->getType().code == halide_type_float);
    }

    // Q splitting for prefill
    mQseqSplitNum = 1;
    if (mQuerySeqLen > 1024) mQseqSplitNum = UP_DIV(mQuerySeqLen, 1024);
    else if (mQuerySeqLen > 256) mQseqSplitNum = UP_DIV(mQuerySeqLen, 256);

    if (mIsKVCacheEnabled) {
        ErrorCode err = init_cache_tensors();
        if (err != MNN::NO_ERROR) return err;
    }

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

    if (mIsKVCacheEnabled && mHasMask && mask_input_tensor && mask_input_tensor->elementSize() == 1) {
        mHasMask = false;
    }

    cudaStream_t stream = 0;

    AttentionKernelParam param_cpu;
    param_cpu.batch = mBatch;
    param_cpu.query_seq_len = mQuerySeqLen;
    param_cpu.head_num = mNumHead;
    param_cpu.kv_head_num = mKvNumHead;
    param_cpu.group = mNumHead / mKvNumHead;
    param_cpu.head_dim = mHeadDim;
    param_cpu.scale = mScale;
    param_cpu.current_kv_seq_len_new = mNewKvSeqLen;

    const void* effective_key_cache_ptr;
    const void* effective_value_cache_ptr;
    int current_total_kv_len_for_qk;
    int allocated_kv_len_for_value_stride;

    if (mIsKVCacheEnabled) {
        if (!mMeta) {
            param_cpu.past_kv_len = mCache->mPastLength;
            current_total_kv_len_for_qk = param_cpu.past_kv_len + mNewKvSeqLen;

            ErrorCode err = reallocKVCache_gpu(current_total_kv_len_for_qk, mBatch, mKvNumHead, mHeadDim, stream);
            if (err != MNN::NO_ERROR) return err;

            allocated_kv_len_for_value_stride = mCache->mMaxLength;
            param_cpu.key_seq_len = current_total_kv_len_for_qk;
            param_cpu.max_kv_len = mCache->mMaxLength;
        } else {
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

        // Copy new K, V to cache
        dim3 copy_blockDim(32, 8, 1);
        dim3 copy_gridDim(UP_DIV(mHeadDim, copy_blockDim.x),
                            UP_DIV(mNewKvSeqLen, copy_blockDim.y),
                            UP_DIV(mBatch * mKvNumHead, copy_blockDim.z));
        if (mPrecision == 4) {
             copy_kv_to_cache_kernel<float><<<copy_gridDim, copy_blockDim, 0, stream>>>(
                getTensorDevicePtr<float>(key_input_tensor), getTensorDevicePtr<float>(value_input_tensor),
                getTensorDevicePtr<float>(mCache->mPastKey.get()), getTensorDevicePtr<float>(mCache->mPastValue.get()),
                mBatch, mNewKvSeqLen, mKvNumHead, mHeadDim, param_cpu.past_kv_len, mCache->mMaxLength);
        } else if (mPrecision == 2) {
             copy_kv_to_cache_kernel<__half><<<copy_gridDim, copy_blockDim, 0, stream>>>(
                getTensorDevicePtr<__half>(key_input_tensor), getTensorDevicePtr<__half>(value_input_tensor),
                getTensorDevicePtr<__half>(mCache->mPastKey.get()), getTensorDevicePtr<__half>(mCache->mPastValue.get()),
                mBatch, mNewKvSeqLen, mKvNumHead, mHeadDim, param_cpu.past_kv_len, mCache->mMaxLength);
        } else { return MNN::NOT_SUPPORT; }
        checkKernelErrors;

        effective_key_cache_ptr = getTensorDevicePtr(mCache->mPastKey.get());
        effective_value_cache_ptr = getTensorDevicePtr(mCache->mPastValue.get());
    } else {
        param_cpu.past_kv_len = 0;
        current_total_kv_len_for_qk = mNewKvSeqLen;
        allocated_kv_len_for_value_stride = ROUND_UP(mNewKvSeqLen,1);
        param_cpu.key_seq_len = mNewKvSeqLen;
        param_cpu.max_kv_len = allocated_kv_len_for_value_stride;

        ErrorCode temp_err = ensureTempBuffers_gpu(mBatch, mNumHead, UP_DIV(mQuerySeqLen, mQseqSplitNum), current_total_kv_len_for_qk, mHeadDim);
        if(temp_err != MNN::NO_ERROR) return temp_err;

        dim3 copy_blockDim(32, 8, 1);
        dim3 copy_gridDim(UP_DIV(mHeadDim, copy_blockDim.x),
                            UP_DIV(mNewKvSeqLen, copy_blockDim.y),
                            UP_DIV(mBatch * mKvNumHead, copy_blockDim.z));
        if (mPrecision == 4) {
            copy_kv_to_cache_kernel<float><<<copy_gridDim, copy_blockDim, 0, stream>>>(
                getTensorDevicePtr<float>(key_input_tensor), getTensorDevicePtr<float>(value_input_tensor),
                getTensorDevicePtr<float>(mTempK_current_step.get()), getTensorDevicePtr<float>(mTempV_current_step.get()),
                mBatch, mNewKvSeqLen, mKvNumHead, mHeadDim, 0, allocated_kv_len_for_value_stride);
        } else if (mPrecision == 2) {
             copy_kv_to_cache_kernel<__half><<<copy_gridDim, copy_blockDim, 0, stream>>>(
                getTensorDevicePtr<__half>(key_input_tensor), getTensorDevicePtr<__half>(value_input_tensor),
                getTensorDevicePtr<__half>(mTempK_current_step.get()), getTensorDevicePtr<__half>(mTempV_current_step.get()),
                mBatch, mNewKvSeqLen, mKvNumHead, mHeadDim, 0, allocated_kv_len_for_value_stride);
        } else { return MNN::NOT_SUPPORT; }
        checkKernelErrors;

        effective_key_cache_ptr = getTensorDevicePtr(mTempK_current_step.get());
        effective_value_cache_ptr = getTensorDevicePtr(mTempV_current_step.get());
    }

    // =====================================================================
    // P0/OPT-2: Flash Decoding path for decode stage (seq_len == 1, with KV cache)
    // OPT-2: Use split-K when SM utilization is low
    // =====================================================================
    if (mQuerySeqLen == 1 && mIsKVCacheEnabled && !mHasMask) {
        const int num_threads = 128;
        const int num_warps = num_threads / 32;
        const int smem_size = num_warps * sizeof(float); // just for QK reduction

        const int total_heads = mBatch * mNumHead;
        const int num_sm = mCudaBackend->getCUDARuntime()->prop().multiProcessorCount;

        // Determine parallel_blocks for split-K
        int parallel_blocks = 1;
        if (current_total_kv_len_for_qk > 16) {
            // Heuristic: try to fill all SMs
            int max_blocks_per_sm = 1; // conservative estimate
            parallel_blocks = std::max(1, (max_blocks_per_sm * num_sm) / total_heads);
            parallel_blocks = std::min(parallel_blocks, (current_total_kv_len_for_qk + 31) / 32); // at least 32 KV per block
            parallel_blocks = std::min(parallel_blocks, 32); // cap
        }

        // OPT-2: Try to allocate split-K buffers if needed
        if (parallel_blocks > 1 && mMaxParallelBlocks < parallel_blocks) {
            if (mSplitKOutputPtr) cudaFree(mSplitKOutputPtr);
            if (mSplitKMetaPtr) cudaFree(mSplitKMetaPtr);
            mSplitKOutputPtr = nullptr;
            mSplitKMetaPtr = nullptr;
            size_t out_size = (size_t)parallel_blocks * total_heads * mHeadDim * sizeof(float);
            size_t meta_size = (size_t)parallel_blocks * total_heads * 2 * sizeof(float);
            auto err1 = cudaMalloc(&mSplitKOutputPtr, out_size);
            auto err2 = cudaMalloc(&mSplitKMetaPtr, meta_size);
            if (err1 != cudaSuccess || err2 != cudaSuccess) {
                if (mSplitKOutputPtr) { cudaFree(mSplitKOutputPtr); mSplitKOutputPtr = nullptr; }
                if (mSplitKMetaPtr) { cudaFree(mSplitKMetaPtr); mSplitKMetaPtr = nullptr; }
                mMaxParallelBlocks = 0;
                parallel_blocks = 1; // fallback to single-block
            } else {
                mMaxParallelBlocks = parallel_blocks;
            }
        }

        if (parallel_blocks <= 1) {
            // Single-block flash decode
            const int smem_size_orig = num_warps * (2 * sizeof(float) + mHeadDim * sizeof(float));
            dim3 grid(total_heads);
            dim3 block(num_threads);

            if (mPrecision == 4) {
                flash_decode_kernel<float><<<grid, block, smem_size_orig, stream>>>(
                    getTensorDevicePtr<float>(query_input_tensor),
                    static_cast<const float*>(effective_key_cache_ptr),
                    static_cast<const float*>(effective_value_cache_ptr),
                    getTensorDevicePtr<float>(final_output_tensor),
                    mBatch, mNumHead, mKvNumHead, mHeadDim,
                    current_total_kv_len_for_qk, allocated_kv_len_for_value_stride, mScale);
            } else if (mPrecision == 2) {
                flash_decode_kernel<__half><<<grid, block, smem_size_orig, stream>>>(
                    getTensorDevicePtr<__half>(query_input_tensor),
                    static_cast<const __half*>(effective_key_cache_ptr),
                    static_cast<const __half*>(effective_value_cache_ptr),
                    getTensorDevicePtr<__half>(final_output_tensor),
                    mBatch, mNumHead, mKvNumHead, mHeadDim,
                    current_total_kv_len_for_qk, allocated_kv_len_for_value_stride, mScale);
            } else { return MNN::NOT_SUPPORT; }
        } else {
            // Split-K flash decode
            float* partial_out = mSplitKOutputPtr;
            float* partial_meta = mSplitKMetaPtr;

            dim3 grid_splitk(total_heads, parallel_blocks);
            dim3 block_splitk(num_threads);

            if (mPrecision == 4) {
                flash_decode_kernel_splitk<float><<<grid_splitk, block_splitk, smem_size, stream>>>(
                    getTensorDevicePtr<float>(query_input_tensor),
                    static_cast<const float*>(effective_key_cache_ptr),
                    static_cast<const float*>(effective_value_cache_ptr),
                    partial_out, partial_meta,
                    mBatch, mNumHead, mKvNumHead, mHeadDim,
                    current_total_kv_len_for_qk, allocated_kv_len_for_value_stride, mScale,
                    parallel_blocks);
            } else if (mPrecision == 2) {
                flash_decode_kernel_splitk<__half><<<grid_splitk, block_splitk, smem_size, stream>>>(
                    getTensorDevicePtr<__half>(query_input_tensor),
                    static_cast<const __half*>(effective_key_cache_ptr),
                    static_cast<const __half*>(effective_value_cache_ptr),
                    partial_out, partial_meta,
                    mBatch, mNumHead, mKvNumHead, mHeadDim,
                    current_total_kv_len_for_qk, allocated_kv_len_for_value_stride, mScale,
                    parallel_blocks);
            } else { return MNN::NOT_SUPPORT; }
            checkKernelErrors;

            // Combine results
            dim3 grid_combine(total_heads);
            dim3 block_combine(mHeadDim); // one thread per D dimension

            if (mPrecision == 4) {
                flash_attn_combine_results<float><<<grid_combine, block_combine, 0, stream>>>(
                    partial_out, partial_meta,
                    getTensorDevicePtr<float>(final_output_tensor),
                    mBatch, mNumHead, mHeadDim, parallel_blocks);
            } else if (mPrecision == 2) {
                flash_attn_combine_results<__half><<<grid_combine, block_combine, 0, stream>>>(
                    partial_out, partial_meta,
                    getTensorDevicePtr<__half>(final_output_tensor),
                    mBatch, mNumHead, mHeadDim, parallel_blocks);
            }
        }
        checkKernelErrors;

        mCache->mPastLength += mNewKvSeqLen;
        return MNN::NO_ERROR;
    }

    // =====================================================================
    // Flash Decode with Mask path for speculative/tree decoding
    // Small seq_len (<=32) with KV cache and additive mask
    // Uses per-query flash decode with online softmax (1-pass, no temp buffers)
    // =====================================================================
    if (mQuerySeqLen <= 32 && mIsKVCacheEnabled && mHasMask && mIsAddMask) {
        const int num_threads = 128;
        const int num_warps = num_threads / 32;
        const int smem_size = num_warps * sizeof(float);
        const int total_blocks = mBatch * mNumHead * mQuerySeqLen;

        dim3 grid(total_blocks);
        dim3 block(num_threads);

        if (mPrecision == 4) {
            flash_decode_kernel_with_mask<float><<<grid, block, smem_size, stream>>>(
                getTensorDevicePtr<float>(query_input_tensor),
                static_cast<const float*>(effective_key_cache_ptr),
                static_cast<const float*>(effective_value_cache_ptr),
                getTensorDevicePtr<float>(final_output_tensor),
                static_cast<const float*>(getTensorDevicePtr(mask_input_tensor)),
                mBatch, mNumHead, mKvNumHead, mHeadDim,
                current_total_kv_len_for_qk, allocated_kv_len_for_value_stride,
                mQuerySeqLen, mScale);
        } else if (mPrecision == 2) {
            flash_decode_kernel_with_mask<__half><<<grid, block, smem_size, stream>>>(
                getTensorDevicePtr<__half>(query_input_tensor),
                static_cast<const __half*>(effective_key_cache_ptr),
                static_cast<const __half*>(effective_value_cache_ptr),
                getTensorDevicePtr<__half>(final_output_tensor),
                static_cast<const __half*>(getTensorDevicePtr(mask_input_tensor)),
                mBatch, mNumHead, mKvNumHead, mHeadDim,
                current_total_kv_len_for_qk, allocated_kv_len_for_value_stride,
                mQuerySeqLen, mScale);
        } else { return MNN::NOT_SUPPORT; }
        checkKernelErrors;

        mCache->mPastLength += mNewKvSeqLen;
        return MNN::NO_ERROR;
    }

    // =====================================================================
    // Prefill path (seq_len > 1) or decode with mask
    // =====================================================================
    int max_q_seq_piece_len = UP_DIV(mQuerySeqLen, mQseqSplitNum);

    ErrorCode temp_buf_err = ensureTempBuffers_gpu(mBatch, mNumHead, max_q_seq_piece_len, current_total_kv_len_for_qk, mHeadDim);
    if (temp_buf_err != MNN::NO_ERROR) return temp_buf_err;

    const void* mask_ptr_device = mHasMask ? getTensorDevicePtr(mask_input_tensor) : nullptr;

    for (int i = 0; i < mQseqSplitNum; ++i) {
        int q_seq_offset = i * max_q_seq_piece_len;
        int current_piece_actual_len = std::min(max_q_seq_piece_len, mQuerySeqLen - q_seq_offset);
        if (current_piece_actual_len <= 0) continue;

        AttentionKernelParam current_iteration_param_cpu = param_cpu;
        current_iteration_param_cpu.q_seq_piece_len = current_piece_actual_len;
        cudaMemcpyAsync(mParam_gpu, &current_iteration_param_cpu, sizeof(AttentionKernelParam), cudaMemcpyHostToDevice, stream);
        checkKernelErrors;

        // P0: Use tiled QK kernel with shared memory
        const int QK_TILE = 16;
        dim3 qk_blockDim(QK_TILE, QK_TILE, 1);
        dim3 qk_gridDim(UP_DIV(current_total_kv_len_for_qk, QK_TILE),
                          UP_DIV(current_piece_actual_len, QK_TILE),
                          mBatch * mNumHead);

        if (mPrecision == 4) {
            qk_kernel_tiled<float><<<qk_gridDim, qk_blockDim, 0, stream>>>(
                getTensorDevicePtr<float>(query_input_tensor), static_cast<const float*>(effective_key_cache_ptr),
                getTensorDevicePtr<float>(mTempQK.get()), mask_ptr_device,
                mParam_gpu, q_seq_offset, mHasMask, mIsAddMask);
        } else if (mPrecision == 2) {
             qk_kernel_tiled<__half><<<qk_gridDim, qk_blockDim, 0, stream>>>(
                getTensorDevicePtr<__half>(query_input_tensor), static_cast<const __half*>(effective_key_cache_ptr),
                getTensorDevicePtr<__half>(mTempQK.get()), mask_ptr_device,
                mParam_gpu, q_seq_offset, mHasMask, mIsAddMask);
        } else { return MNN::NOT_SUPPORT; }
        checkKernelErrors;

        // Softmax (reuse existing optimized implementation)
        const int axis = current_total_kv_len_for_qk;
        const int inside = 1;
        const int outside = mBatch * mNumHead * current_piece_actual_len;
        const int count = outside * inside;

        const void* qk_scores_ptr = getTensorDevicePtr(mTempQK.get());
        void* softmax_result_ptr = getTensorDevicePtr(mTempSoftmax.get());

        if (mPrecision == 4) {
            const auto* input_ptr = static_cast<const float*>(qk_scores_ptr);
            auto* output_ptr = static_cast<float*>(softmax_result_ptr);
            if (axis <= 32) {
                SOFTMAX_WARP_32<float><<<count, 32, 0, stream>>>(input_ptr, output_ptr, inside, axis, outside, count);
            } else {
                constexpr int threads_per_block = 256;
                const int calc_multi_num = UP_DIV(axis, threads_per_block);
                SOFTMAX_AXIS_REDUCE<float><<<count, threads_per_block, 0, stream>>>(input_ptr, output_ptr, inside, axis, threads_per_block, calc_multi_num, outside, count);
            }
        } else {
            const auto* input_ptr = static_cast<const __half*>(qk_scores_ptr);
            auto* output_ptr = static_cast<__half*>(softmax_result_ptr);
            if (axis <= 32) {
                SOFTMAX_WARP_32<__half><<<count, 32, 0, stream>>>(input_ptr, output_ptr, inside, axis, outside, count);
            } else {
                constexpr int threads_per_block = 256;
                const int calc_multi_num = UP_DIV(axis, threads_per_block);
                SOFTMAX_AXIS_REDUCE<__half><<<count, threads_per_block, 0, stream>>>(input_ptr, output_ptr, inside, axis, threads_per_block, calc_multi_num, outside, count);
            }
        }
        checkKernelErrors;

        // P0: Use optimized QKV kernel with unrolled loop
        dim3 qkv_blockDim(32, 8, 1);
        dim3 qkv_gridDim(UP_DIV(mHeadDim, qkv_blockDim.x),
                         UP_DIV(current_piece_actual_len, qkv_blockDim.y),
                         mBatch * mNumHead);
        if (mPrecision == 4) {
            qkv_kernel_tiled<float><<<qkv_gridDim, qkv_blockDim, 0, stream>>>(
                getTensorDevicePtr<float>(mTempSoftmax.get()), static_cast<const float*>(effective_value_cache_ptr),
                getTensorDevicePtr<float>(final_output_tensor), mParam_gpu, q_seq_offset);
        } else if (mPrecision == 2) {
            qkv_kernel_tiled<__half><<<qkv_gridDim, qkv_blockDim, 0, stream>>>(
                getTensorDevicePtr<__half>(mTempSoftmax.get()), static_cast<const __half*>(effective_value_cache_ptr),
                getTensorDevicePtr<__half>(final_output_tensor), mParam_gpu, q_seq_offset);
        } else { return MNN::NOT_SUPPORT; }
        checkKernelErrors;
    }

    if (mIsKVCacheEnabled) {
        mCache->mPastLength += mNewKvSeqLen;
    }

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
