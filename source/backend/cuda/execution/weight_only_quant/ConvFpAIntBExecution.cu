//
//  ConvFpAIntBExecution.cpp
//  MNN
//
//  Created by MNN on 2024/03/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

// precision = low/normal 时使用运行时离线反量化，precision = high 时使用在线反量化
#include "ConvFpAIntBExecution.hpp"
#include "../Raster.cuh"
#include "../ConvBaseKernel.cuh"
#include <float.h>

//#define DEBUG

namespace MNN {
namespace CUDA {

const int TILE_DIM = 16; // GEMM_FpAIntB
const int GEMV_TILE = 64; // GEMV_FpAIntB
const int SUB_GEMM_TILE_DIM = 16; // GEMM_Int
const int BLOCK_SIZE = 256; // Quant / SumBq / Bias
const int DEQUANT_TILE_DIM = 32; // Dequant
const int WARP_SIZE = 32;

template<typename T, typename dT>
__global__ void DequantizeInt8Weight(
    const int8_t* quantized_kernel,
    dT* dequantized_kernel,
    const T* scale,
    const T* offset,
    const int oc, const int ic, const int ic_p, const int quanC
) {
    // 每个线程负责反量化一个权重值
    const int col = blockIdx.x * blockDim.x + threadIdx.x; // input channel
    const int row = blockIdx.y * blockDim.y + threadIdx.y; // output channel

    if (row < oc && col < ic) {
        // 计算量化参数的索引
        const int num_quan_groups_per_channel = (quanC > 0) ? (quanC / oc) : 1;
        const int ic_per_group = (num_quan_groups_per_channel > 0) ? (ic / num_quan_groups_per_channel) : ic;
        const int group_idx = col / ic_per_group;
        const int quan_param_index = row * num_quan_groups_per_channel + group_idx;

        const float x_scale  = (float)scale[quan_param_index];
        const float x_offset = (float)offset[quan_param_index];

        // 读取量化的权重
        const float quantized_val = quantized_kernel[row * ic_p + col];

        // 反量化并写入目标 Tensor
        dequantized_kernel[row * ic_p + col] = (dT)(quantized_val * x_scale + x_offset);
    }
}

template<typename T, typename dT>
__global__ void DequantizeInt4Weight(
    const uint8_t* quantized_kernel,
    dT* dequantized_kernel,
    const T* scale,
    const T* offset,
    const int oc, const int ic, const int ic_p, const int quanC
) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x; // ic
    const int row = blockIdx.y * blockDim.y + threadIdx.y; // oc

    if (row < oc && col < ic) {
        const int num_quan_groups_per_channel = (quanC > 0) ? (quanC / oc) : 1;
        const int ic_per_group = max(1, (num_quan_groups_per_channel > 0) ? (ic / num_quan_groups_per_channel) : ic);
        const int group_idx = min(col / ic_per_group, num_quan_groups_per_channel - 1);
        const int quan_param_index = row * num_quan_groups_per_channel + group_idx;

        const float x_scale  = (float)scale[quan_param_index];
        const float x_offset = (float)offset[quan_param_index];

        const uint8_t packed_val = quantized_kernel[row * (ic_p / 2) + (col / 2)];
        
        const int8_t quantized_val = (col % 2 == 0) ?
            ((packed_val >> 4) - 8) :      // 偶数列(ic)，取高4位
            ((packed_val & 0x0F) - 8);     // 奇数列(ic)，取低4位

        dequantized_kernel[row * ic_p + col] = (dT)((float)quantized_val * x_scale + x_offset);
    }
}

// 动态量化 A、计算 sum_A
template<typename T>
__global__ void QuantA(
    const T* A_sub_fp, int8_t* A_sub_q,
    T* scale_A_out, T* offset_A_out, int32_t* sum_A_q_out,
    const int M, const int K_i, const int lda
) {
    // 使用 union 复用共享内存，减少总占用量
    __shared__ union {
        float min_max_vals[2][BLOCK_SIZE / WARP_SIZE]; // 用于 min/max 的 warp 间规约
        int32_t sum_vals[BLOCK_SIZE / WARP_SIZE];      // 用于 sum 的 warp 间规约
        float scale_offset[2];                        // 用于向块内所有线程广播 scale 和 offset
    } smem;

    const int m = blockIdx.x;
    if (m >= M) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    float my_min = FLT_MAX;
    float my_max = -FLT_MAX;

    for (int k = tid; k < K_i; k += BLOCK_SIZE) {
        float val = (float)A_sub_fp[m * lda + k];
        my_min = min(my_min, val);
        my_max = max(my_max, val);
    }

    // Warp 内规约：使用 __shfl_down_sync 在 warp 内无锁计算 min/max
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        my_min = min(my_min, __shfl_down_sync(0xFFFFFFFF, my_min, offset));
        my_max = max(my_max, __shfl_down_sync(0xFFFFFFFF, my_max, offset));
    }

    // Warp 间规约：每个 warp 的 0 号线程将 warp 结果写入共享内存
    if (lane_id == 0) {
        smem.min_max_vals[0][warp_id] = my_min;
        smem.min_max_vals[1][warp_id] = my_max;
    }
    __syncthreads();

    if (warp_id == 0) {
        // 从共享内存加载各 warp 的结果
        if (lane_id < (BLOCK_SIZE / WARP_SIZE)) {
            my_min = smem.min_max_vals[0][lane_id];
            my_max = smem.min_max_vals[1][lane_id];
        } else {
            my_min = FLT_MAX;
            my_max = -FLT_MAX;
        }

        // 在第一个 warp 内部完成最终规约
        for (int offset = (BLOCK_SIZE / WARP_SIZE) / 2; offset > 0; offset >>= 1) {
            my_min = min(my_min, __shfl_down_sync(0xFFFFFFFF, my_min, offset));
            my_max = max(my_max, __shfl_down_sync(0xFFFFFFFF, my_max, offset));
        }
    }
    // 线程 0 计算 scale/offset 并写入共享内存进行广播
    if (tid == 0) {
        float scale = (my_max - my_min) / 255.0f;
        float offset;
        if (abs(scale) > 1e-5) {
            offset = my_max - scale * 127.0f; // my_max 映射到 127, my_min 映射到 -128
        } else {
            scale = 1.0f;
            offset = my_max;
        }

        scale_A_out[m] = (T)scale;
        offset_A_out[m] = (T)offset;
        
        smem.scale_offset[0] = scale;
        smem.scale_offset[1] = offset;
    }
    __syncthreads();

    // 所有线程从共享内存读取 scale/offset, 并进行量化和求和
    const float s_scale_A = smem.scale_offset[0];
    const float s_offset_A = smem.scale_offset[1];

    int32_t my_sum_q = 0;
    for (int k = tid; k < K_i; k += BLOCK_SIZE) {
        float val_fp = (float)A_sub_fp[m * lda + k];
        int32_t val_q = roundf((val_fp - s_offset_A) / s_scale_A);
        int8_t a_q = (int8_t)max(-128, min(127, val_q));
        A_sub_q[m * K_i + k] = a_q;
        my_sum_q += a_q;
    }

    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        my_sum_q += __shfl_down_sync(0xFFFFFFFF, my_sum_q, offset);
    }

    if (lane_id == 0) smem.sum_vals[warp_id] = my_sum_q;
    __syncthreads();

    if (warp_id == 0) {
        my_sum_q = (lane_id < (BLOCK_SIZE / WARP_SIZE)) ? smem.sum_vals[lane_id] : 0;
        for (int offset = (BLOCK_SIZE / WARP_SIZE) / 2; offset > 0; offset >>= 1) {
             my_sum_q += __shfl_down_sync(0xFFFFFFFF, my_sum_q, offset);
        }
    }

    // 线程 0 将最终的和写入全局内存
    if (tid == 0) sum_A_q_out[m] = my_sum_q;
}

__global__ void GEMM_Int8(
    const int8_t* A_q, const int8_t* B_q, int32_t* C_q,
    const int M, const int N, const int K_i,
    const int lda_q, const int ldb, const int ldc
) {
    __shared__ int8_t A_tile_s8[SUB_GEMM_TILE_DIM][SUB_GEMM_TILE_DIM];
    __shared__ int8_t B_tile_s8[SUB_GEMM_TILE_DIM][SUB_GEMM_TILE_DIM];

    const int block_row = blockIdx.y;
    const int block_col = blockIdx.x;
    const int ty = threadIdx.y;
    const int tx = threadIdx.x;

    const int row = block_row * SUB_GEMM_TILE_DIM + ty;
    const int col = block_col * SUB_GEMM_TILE_DIM + tx;

    int32_t acc = 0;
    const int num_k_tiles = UP_DIV(K_i, SUB_GEMM_TILE_DIM);

    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        const int k_tile_base = k_tile * SUB_GEMM_TILE_DIM;

        A_tile_s8[ty][tx] = (row < M && (k_tile_base + tx) < K_i) ? A_q[row * lda_q + k_tile_base + tx] : 0;
        B_tile_s8[ty][tx] = (col < N && (k_tile_base + ty) < K_i) ? B_q[col * ldb + k_tile_base + ty] : 0;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < SUB_GEMM_TILE_DIM; ++k) {
            acc += (int32_t)A_tile_s8[ty][k] * (int32_t)B_tile_s8[k][tx];
        }
        __syncthreads();
    }

    if (row < M && col < N) C_q[row * ldc + col] = acc;
}

template<typename T>
__global__ void DequantAndAcc(
    const int32_t* C_q, T* C_fp_final,
    const T* scale_A_in, const T* offset_A_in,
    const T* base_scale_B, const T* base_offset_B,
    const int32_t* base_sum_B_q,
    const int group_idx, const int num_oc_groups,
    const int32_t* sum_A_q_in,
    const int M, const int N, const int K_i, const int ldc
) {
    // 使用共享内存缓存行和列的量化参数
    __shared__ float smem_scale_A[DEQUANT_TILE_DIM];
    __shared__ float smem_offset_A[DEQUANT_TILE_DIM];
    __shared__ int32_t smem_sum_A_q[DEQUANT_TILE_DIM];

    __shared__ float smem_scale_B[DEQUANT_TILE_DIM];
    __shared__ float smem_offset_B[DEQUANT_TILE_DIM];
    __shared__ int32_t smem_sum_B_q[DEQUANT_TILE_DIM];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int block_row_start = blockIdx.y * DEQUANT_TILE_DIM;
    const int block_col_start = blockIdx.x * DEQUANT_TILE_DIM;

    // 每个线程加载一个 A 的参数
    int m_load_idx = block_row_start + ty;
    if (tx == 0 && m_load_idx < M) {
        smem_scale_A[ty]   = (float)scale_A_in[m_load_idx];
        smem_offset_A[ty]  = (float)offset_A_in[m_load_idx];
        smem_sum_A_q[ty]   = sum_A_q_in[m_load_idx];
    }

    // 每个线程加载一个 B 的参数
    int n_load_idx = block_col_start + tx;
    if (ty == 0 && n_load_idx < N) {
        const size_t b_param_idx = n_load_idx * num_oc_groups + group_idx;
        smem_scale_B[tx]  = (float)base_scale_B[b_param_idx];
        smem_offset_B[tx] = (float)base_offset_B[b_param_idx];
        smem_sum_B_q[tx]  = base_sum_B_q[b_param_idx];
    }

    __syncthreads();

    // 计算每个线程负责的输出点
    const int m = block_row_start + ty;
    const int n = block_col_start + tx;

    if (m < M && n < N) {
        // 从共享内存中读取参数
        const float scale_A = smem_scale_A[ty];
        const float offset_A = smem_offset_A[ty];
        const float sum_A_q = (float)smem_sum_A_q[ty];

        const float scale_B = smem_scale_B[tx];
        const float offset_B = smem_offset_B[tx];
        const float sum_B_q = (float)smem_sum_B_q[tx];
        
        const float c_q_val = (float)C_q[m * ldc + n];

        float term1 = scale_A * (c_q_val * scale_B + sum_A_q * offset_B);
        float term2 = offset_A * (sum_B_q * scale_B + K_i * offset_B);
        float final_val = term1 + term2;

        #if (defined(__CUDACC__) && (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 700))) || defined(_NVHPC_CUDA)
            if constexpr (std::is_same_v<T, float>) {
                atomicAdd(&C_fp_final[m * ldc + n], final_val);
            } else if constexpr (std::is_same_v<T, half>) {
                // printf("[final_val]%.4f %.4f\n", __half2float(C_fp_final[m * ldc + n]), final_val);
                atomicAdd(&C_fp_final[m * ldc + n], __float2half(final_val));
                // printf("[C_fp_final]%.4f\n", __half2float(C_fp_final[m * ldc + n]));
            }
        #else
            C_fp_final[m * ldc + n] += final_val;
        #endif
    }
}

// 预计算权重 B 的修正项 (sum_B_q)
__global__ void Precompute_SumBq(
    const int8_t* B_q, int32_t* sum_B_q_out,
    const int num_groups, const int ic_per_group,
    const int oc, const int ic_p
) {
    // 每个线程处理一个 (output_channel, group) 的修正项
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < (size_t)oc * num_groups; index += blockDim.x * gridDim.x) {
        
        const int i = index % num_groups; // group_idx
        const int n = index / num_groups; // output_channel
        
        const int k_start = i * ic_per_group;
        
        int32_t sum = 0;
        for (int k_offset = 0; k_offset < ic_per_group; ++k_offset) {
            // 访问 B 矩阵的布局是 [oc][ic_p]
            sum += (int32_t)B_q[n * ic_p + k_start + k_offset];
        }
        sum_B_q_out[index] = sum;
    }
}

template<typename T>
__global__ void BiasAndActivation(
    T* data, const T* bias,
    const float minV, const float maxV,
    const int M, const int N, const int ldc
) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < (size_t)M * N; index += blockDim.x * gridDim.x) {
        const int m = index / N;
        const int n = index % N;

        const size_t buffer_idx = m * ldc + n;
        float val = (float)data[buffer_idx];
        val += (float)bias[n];
        val = max(val, minV);
        val = min(val, maxV);
        data[buffer_idx] = (T)val;
    }
}

template <typename T>
__global__ void print_tensor_kernel(const T* d_tensor, int rows, int cols) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("----------- Tensor in Kernel -----------\n");
        printf("Rows: %d, Cols: %d\n", rows, cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                T value = d_tensor[i * cols + j];

                if constexpr (std::is_same_v<T, half>) {
                    printf("%.4f ", __half2float(value));
                } else {
                    printf("%.4f ", value);
                }
            }
            printf("\n");
        }
        printf("---------------------------------------\n");
    }
}

template<typename T>
__global__ void GEMM_FpAInt8B(
    const T* input,
    const int8_t* kernel,
    const T* scale, const T* offset, const T* bias,
    T* output,
    const float maxV, const float minV,
    const int ic, const int ic_p, const int oc, const int oc_p,
    const int batch, const int quanC
) {
    __shared__ T A_tile[TILE_DIM][TILE_DIM]; // [batch, ic]
    __shared__ int8_t B_tile_s8[TILE_DIM][TILE_DIM]; // [ic, oc] kernel 本身是 B^T [oc, ic]
    __shared__ T B_tile_fp[TILE_DIM][TILE_DIM];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int block_row = blockIdx.y; // batch
    const int block_col = blockIdx.x; // output channel

    // 每个线程负责计算输出Tile中的一个元素
    const int out_row = block_row * TILE_DIM + ty; // M
    const int out_col = block_col * TILE_DIM + tx; // N

    float acc = 0.0f;
    // 沿K维度（输入通道ic）分块循环
    const int num_k_tiles = UP_DIV(ic, TILE_DIM);

    const int num_quan_groups_per_channel = (quanC > 0) ? (quanC / oc) : 1;
    const int ic_per_group = (num_quan_groups_per_channel > 0) ? (ic / num_quan_groups_per_channel) : ic;

    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        const int k_tile_base = k_tile * TILE_DIM;

        // 合并访问加载 A_tile (input)，线程 (ty, tx) 加载 A_tile[ty][tx]
        int a_col_idx = k_tile_base + tx;
        A_tile[ty][tx] = (out_row < batch && a_col_idx < ic) ? input[out_row * ic_p + a_col_idx] : (T)0.0f;
        
        // 合并访问加载 B_tile (kernel)，线程 (ty, tx) 加载 B_tile[ty][tx]
        // kernel 布局为 [oc, ic]，需要 B(k,n)，即 kernel(n,k)
        int b_load_row = block_col * TILE_DIM + ty;
        int b_col_idx = k_tile_base + tx;
        B_tile_s8[ty][tx] = (b_load_row < oc && b_col_idx < ic) ? kernel[b_load_row * ic_p + b_col_idx] : 0;

        __syncthreads();

        // 反量化 + 转置
        const int K = ty;
        const int N = tx;

        const int global_k = k_tile_base + K;
        const int global_n = block_col * TILE_DIM + N;

        if (global_n < oc && global_k < ic) {
            const int group_idx = global_k / ic_per_group;
            const int quan_param_index = global_n * num_quan_groups_per_channel + group_idx;

            const float x_scale  = (float)scale[quan_param_index];
            const float x_offset = (float)offset[quan_param_index];
            
            // B_tile_s8(n,k), thread(n,k) -> (tx, ty)
            // So we need to read from B_tile_s8[n_dim_in_tile][k_dim_in_tile] -> B_tile_s8[tx][ty]
            const float b_quant = (float)B_tile_s8[N][K];

            // B_tile_fp[k][n] -> B_tile_fp[ty][tx]
            B_tile_fp[K][N] = (T)(b_quant * x_scale + x_offset);
        }
        
        __syncthreads();

        // 在 SMEM 中进行子矩阵乘法
        if (out_col < oc) {
            #pragma unroll
            for (int k = 0; k < TILE_DIM; ++k) {
                acc += (float)A_tile[ty][k] * (float)B_tile_fp[k][tx];
            }
        }

        __syncthreads();
    }

    // 写回全局内存
    if (out_row < batch && out_col < oc) {
        acc += (float)bias[out_col];
        acc = max(acc, minV);
        acc = min(acc, maxV);
        output[out_row * oc_p + out_col] = (T)acc;
    }
}

template<typename T>
__global__ void GEMV_FpAInt8B(
    const T* input,
    const int8_t* kernel,
    const T* scale, const T* offset, const T* bias,
    T* output,
    const float maxV, const float minV,
    const int batch, const int ic, const int ic_p,
    const int oc, const int oc_p, const int quanC
) {
    __shared__ float partial_sums[GEMV_TILE]; // 每个线程块计算一个输出通道

    const int oz = blockIdx.x;
    const int b  = blockIdx.y;

    // if (oz >= oc || b >= batch) return;

    const int tid = threadIdx.x; // 块内线程ID
    
    // 准备量化参数
    const int num_quan_groups_per_channel = (quanC > 0) ? (quanC / oc) : 1;
    const int ic_per_group = (num_quan_groups_per_channel > 0) ? (ic / num_quan_groups_per_channel) : ic;
    const int quan_param_index_base = oz * num_quan_groups_per_channel;

    // 每个线程计算自己的部分和
    float my_sum = 0.0f;
    for (int k = tid; k < ic; k += blockDim.x) {
        const float inp0 = input[b * ic_p + k];
        const int8_t ker0 = kernel[oz * ic_p + k];
        
        const int group_idx = k / ic_per_group;
        const int quan_param_index = quan_param_index_base + group_idx;
        const float x_scale  = scale[quan_param_index];
        const float x_offset = offset[quan_param_index];
        
        my_sum += inp0 * ((float)ker0 * x_scale + x_offset);
    }

    partial_sums[tid] = my_sum;
    __syncthreads();

    // 在 SMEM 中执行并行规约（蝶式交换）
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) partial_sums[tid] += partial_sums[tid + s];
        __syncthreads();
    }

    if (tid == 0) { // 写回全局内存
        float final_val = partial_sums[0] + (float)bias[oz];
        final_val = max(final_val, minV);
        final_val = min(final_val, maxV);
        output[b * oc_p + oz] = (T)final_val;
    }
}

template<typename T>
__global__ void GEMM_FpAInt4B(
    const T* input,
    const uint8_t* kernel, // kernel 是打包的 int4
    const T* scale, const T* offset, const T* bias,
    T* output,
    const float maxV, const float minV,
    const int ic, const int ic_p, const int oc, const int oc_p,
    const int batch, const int quanC
) {
    __shared__ T A_tile[TILE_DIM][TILE_DIM]; // [batch, ic]
    __shared__ uint8_t B_tile_u8[TILE_DIM][TILE_DIM / 2]; // [ic, oc] kernel 本身是 B^T [oc, ic]
    __shared__ T B_tile_fp[TILE_DIM][TILE_DIM];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int block_row = blockIdx.y; // batch
    const int block_col = blockIdx.x; // output channel

    const int out_row = block_row * TILE_DIM + ty; // M
    const int out_col = block_col * TILE_DIM + tx; // N

    float acc = 0.0f;
    const int num_k_tiles = UP_DIV(ic, TILE_DIM);
    
    const int num_quan_groups_per_channel = (quanC > 0) ? (quanC / oc) : 1;
    const int ic_per_group = (num_quan_groups_per_channel > 0) ? (ic / num_quan_groups_per_channel) : ic;

    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        const int k_tile_base = k_tile * TILE_DIM;

        // 合并访问加载 A_tile (input)
        int a_col_idx = k_tile_base + tx;
        A_tile[ty][tx] = (out_row < batch && a_col_idx < ic) ? input[out_row * ic_p + a_col_idx] : (T)0.0f;
        
        // 合并访问加载 B_tile (kernel)，每个线程加载一个 byte，但 tile 宽度减半
        if (tx < TILE_DIM / 2) {
            int b_load_row = block_col * TILE_DIM + ty;
            // kernel 布局为 [oc, ic/2]，所以列索引要除以2
            int b_col_idx = (k_tile_base / 2) + tx;
            B_tile_u8[ty][tx] = (b_load_row < oc && (k_tile_base + tx*2) < ic) ? kernel[b_load_row * (ic_p / 2) + b_col_idx] : 0;
        }

        __syncthreads();

        // 反量化 + 转置
        const int K = ty;
        const int N = tx;

        const int global_k = k_tile_base + K;
        const int global_n = block_col * TILE_DIM + N;

        if (global_n < oc && global_k < ic) {
            const int group_idx = global_k / ic_per_group;
            const int quan_param_index = global_n * num_quan_groups_per_channel + group_idx;

            const float x_scale  = (float)scale[quan_param_index];
            const float x_offset = (float)offset[quan_param_index];
            
            // B_tile_u8(n,k) -> B_tile_u8[tx][ty/2] (转置后)
            const uint8_t b_packed = B_tile_u8[N][K / 2];
            // 根据 K 的奇偶性解包 int4
            const int8_t b_quant_s8 = (K % 2 == 0) ? ((b_packed >> 4) - 8) : ((b_packed & 0x0F) - 8);
            const float b_quant = (float)b_quant_s8;
            
            // B_tile_fp[k][n] -> B_tile_fp[ty][tx]
            B_tile_fp[K][N] = (T)(b_quant * x_scale + x_offset);
        }
        
        __syncthreads();

        // 在 SMEM 中进行子矩阵乘法
        if (out_col < oc) {
            #pragma unroll
            for (int k = 0; k < TILE_DIM; ++k) {
                acc += (float)A_tile[ty][k] * (float)B_tile_fp[k][tx];
            }
        }

        __syncthreads();
    }

    // 写回全局内存
    if (out_row < batch && out_col < oc) {
        acc += (float)bias[out_col];
        acc = max(acc, minV);
        acc = min(acc, maxV);
        output[out_row * oc_p + out_col] = (T)acc;
    }
}

template<typename T>
__global__ void GEMV_FpAInt4B(
    const T* input,
    const uint8_t* kernel, // kernel 是打包的 int4
    const T* scale, const T* offset, const T* bias,
    T* output,
    const float maxV, const float minV,
    const int batch, const int ic, const int ic_p,
    const int oc, const int oc_p, const int quanC
) {
    extern __shared__ uint8_t smem_buffer[];
    T* smem_input = reinterpret_cast<T*>(smem_buffer);
    float* partial_sums = reinterpret_cast<float*>(smem_buffer + ic_p * sizeof(T));
    
    const int oz = blockIdx.x; // 当前线程块负责计算的输出通道索引
    const int b  = blockIdx.y;

    const int tid = threadIdx.x;
    
    // if (oz >= oc || b >= batch) return;
    
    // 加载 input 到共享内存
    for (int i = tid; i < ic; i += blockDim.x) smem_input[i] = input[b * ic_p + i];
    for (int i = ic + tid; i < ic_p; i += blockDim.x) smem_input[i] = (T)0.0f;
    __syncthreads();

    const int num_quan_groups_per_channel = (quanC > 0) ? (quanC / oc) : 1;
    const int ic_per_group = (num_quan_groups_per_channel > 0) ? (ic / num_quan_groups_per_channel) : ic;
    const int quan_param_index_base = oz * num_quan_groups_per_channel;

    float my_sum = 0.0f;
    for (int k = tid * 2; k < ic; k += blockDim.x * 2) {
        // 加载打包的权重并解包
        const uint8_t ker_packed = kernel[oz * (ic_p / 2) + k / 2];
        const int8_t ker0_s8 = (ker_packed >> 4) - 8;
        const int8_t ker1_s8 = (ker_packed & 0x0F) - 8;

        const int group_idx0 = k / ic_per_group;
        const int quan_param_index0 = quan_param_index_base + group_idx0;
        const float x_scale0  = scale[quan_param_index0];
        const float x_offset0 = offset[quan_param_index0];
        my_sum += (float)smem_input[k] * ((float)ker0_s8 * x_scale0 + x_offset0);
        my_sum += (float)smem_input[k + 1] * ((float)ker1_s8 * x_scale0 + x_offset0);
        
        // if (k + 1 < ic) {
        //     const float inp1 = (float)smem_input[k + 1];
        //     const int group_idx1 = (k + 1) / ic_per_group;

        //     if (group_idx0 == group_idx1) {
        //         my_sum += inp1 * ((float)ker1_s8 * x_scale0 + x_offset0);
        //     } else {
        //         const int quan_param_index1 = quan_param_index_base + group_idx1;
        //         const float x_scale1  = (float)scale[quan_param_index1];
        //         const float x_offset1 = (float)offset[quan_param_index1];
        //         my_sum += inp1 * ((float)ker1_s8 * x_scale1 + x_offset1);
        //     }
        // }
    }

    partial_sums[tid] = my_sum;
    __syncthreads();

    // 并行规约
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) partial_sums[tid] += partial_sums[tid + s];
        __syncthreads();
    }

    if (tid == 0) {
        float final_val = partial_sums[0] + (float)bias[oz];
        final_val = max(final_val, minV);
        final_val = min(final_val, maxV);
        output[b * oc_p + oz] = (T)final_val;
    }
}

template<typename T>
__global__ void CONV_FpAInt8B(const T* input,
    const int8_t* kernel,
    const T* scale, const T* offset, const T* bias,
    T *output,
    const float maxV, const float minV,
    const int ic,  const int ic_p, const int iw, const int ih,
    const int c, const int c_p, const int ow, const int oh,
    const int kw, const int kh, const int dw, const int dh,
    const int sw, const int sh, const int pw, const int ph,
    const int total, const int quanC,
    DivModFast d_oc, DivModFast d_ow, DivModFast d_oh
) {

    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < total; index += blockDim.x * gridDim.x) {
        int oz_2, tmp2, oy, ox, tmp1, ob;
        d_oc.divmod(index, tmp1, oz_2);
        d_ow.divmod(tmp1, tmp2, ox);
        d_oh.divmod(tmp2, ob, oy);

        int oz = oz_2;
        int ix = ox * sw - pw;
        int iy = oy * sh - ph;
        float color0 = bias[oz];
        
        const int num_quan_groups_per_channel = (c > 0 && quanC > 0) ? (quanC / c) : 1;
        const int ic_per_group = (num_quan_groups_per_channel > 0) ? (ic / num_quan_groups_per_channel) : ic;
        const int quan_param_index_base = oz * num_quan_groups_per_channel;

        int fxSta = max(0, (UP_DIV(-ix, dw)));
        int fySta = max(0, (UP_DIV(-iy, dh)));
        int fxEnd = min(kw, UP_DIV(iw - ix, dw));
        int fyEnd = min(kh, UP_DIV(ih - iy, dh));
        int fx, fy, fz;
        for (fy=fySta; fy<fyEnd; ++fy) {
            int sy = fy*dh + iy;
            for (fx=fxSta; fx<fxEnd; ++fx) {
                int sx = fx*dw + ix;
                for (int group_idx = 0; group_idx < num_quan_groups_per_channel; ++group_idx) {
                    const int quan_param_index = quan_param_index_base + group_idx;
                    const float x_scale = scale[quan_param_index];
                    const float x_offset = offset[quan_param_index];

                    const int sz_start = group_idx * ic_per_group;
                    const int sz_end = sz_start + ic_per_group;

                    for (int sz = sz_start; sz < sz_end && sz < ic_p; ++ sz) {
                        int src_offset = ((ob * ih + sy) * iw + sx) * ic_p + sz;
                        float inp0 = input[src_offset];
                        //[Cop, KhKw, Cip]
                        int8_t ker0 = kernel[((oz * kh + fy) * kw + fx) * ic_p + sz];

                        color0 = color0 + inp0 * ((float)ker0 * x_scale + x_offset);
                    }
                }
            }
        }
        color0 = max(color0, minV);
        color0 = min(color0, maxV);

        int dst_offset = ((ob * oh + oy) * ow + ox) * c_p + oz;

        output[dst_offset] = color0;
    }
}

template<typename T>
__global__ void CONV_FpAInt4B(const T* input,
    const uint8_t* kernel,
    const T* scale, const T* offset, const T* bias,
    T *output,
    const float maxV, const float minV,
    const int ic, const int ic_p, const int iw, const int ih,
    const int c, const int c_p, const int ow, const int oh,
    const int kw, const int kh, const int dw, const int dh,
    const int sw, const int sh, const int pw, const int ph,
    const int total, const int quanC,
    DivModFast d_oc, DivModFast d_ow, DivModFast d_oh
) {

    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < total; index += blockDim.x * gridDim.x) {
        int oz_2, tmp2, oy, ox, tmp1, ob;
        d_oc.divmod(index, tmp1, oz_2);
        d_ow.divmod(tmp1, tmp2, ox);
        d_oh.divmod(tmp2, ob, oy);

        int oz = oz_2;
        int ix = ox * sw - pw;
        int iy = oy * sh - ph;
        float color0 = bias[oz];

        const int num_quan_groups_per_channel = (c > 0 && quanC > 0) ? (quanC / c) : 1;
        const int ic_per_group = (num_quan_groups_per_channel > 0) ? (ic / num_quan_groups_per_channel) : ic;
        const int quan_param_index_base = oz * num_quan_groups_per_channel;

        int fxSta = max(0, (UP_DIV(-ix, dw)));
        int fySta = max(0, (UP_DIV(-iy, dh)));
        int fxEnd = min(kw, UP_DIV(iw - ix, dw));
        int fyEnd = min(kh, UP_DIV(ih - iy, dh));
        int fx, fy, fz;
        for (fy=fySta; fy<fyEnd; ++fy) {
            int sy = fy*dh + iy;
            for (fx=fxSta; fx<fxEnd; ++fx) {
                int sx = fx*dw + ix;
                for (int group_idx = 0; group_idx < num_quan_groups_per_channel; ++group_idx) {
                    const int quan_param_index = quan_param_index_base + group_idx;
                    const float x_scale = scale[quan_param_index];
                    const float x_offset = offset[quan_param_index];

                    const int sz_start = group_idx * ic_per_group / 2;
                    const int sz_end = sz_start + ic_per_group / 2;

                    for (int sz = sz_start; sz < sz_end && sz * 2 < ic_p; ++ sz) {
                        int src_offset = ((ob * ih + sy) * iw + sx) * ic_p + 2 * sz;
                        float inp0 = input[src_offset];
                        float inp1 = input[src_offset+1];

                        //[Cop, KhKw, Cip]
                        uint8_t ker = kernel[((oz * kh + fy) * kw + fx) * ic_p / 2 + sz];
                        int8_t ker0 = (ker >> 4) - 8;
                        int8_t ker1 = (ker & 15) - 8;
                        color0 = color0 + inp0 * ((float)ker0 * x_scale + x_offset);
                        color0 = color0 + inp1 * ((float)ker1 * x_scale + x_offset);
                    }
                }
            }
        }
        color0 = max(color0, minV);
        color0 = min(color0, maxV);

        int dst_offset = ((ob * oh + oy) * ow + ox) * c_p + oz;

        output[dst_offset] = color0;
    }
}

__global__ void Rearrange_Packed_Weight_Int4(const uint8_t* param,
    uint8_t* output,
    const int khw,
    const size_t maxCount,
    const int oc,
    const int ic,
    DivModFast d_khw,
    DivModFast d_icp2
) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
        int icp2Index, temp, ocpIndex, khwIndex;
        d_icp2.divmod(index, temp, icp2Index);
        d_khw.divmod(temp, ocpIndex, khwIndex);
        if(ocpIndex >= oc || 2 * icp2Index >= ic) {
            output[index] = 0;
            continue;
        }

        int src_index = (ocpIndex * UP_DIV(ic, 2) + icp2Index) * khw + khwIndex;
        output[index] = param[src_index];
    }
}

__global__ void Rearrange_Weight_Int4(const int8_t* param,
    uint8_t* output,
    const int khw,
    const size_t maxCount,
    const int oc,
    const int ic,
    DivModFast d_khw,
    DivModFast d_icp2
) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
        int icp2Index, temp, ocpIndex, khwIndex;
        d_icp2.divmod(index, temp, icp2Index);
        d_khw.divmod(temp, ocpIndex, khwIndex);
        if(2*icp2Index >= ic || ocpIndex >= oc) {
            output[index] = 0;
            continue;
        }
        // [Co, Ci, KhKw] -> [Cop, KhKw, Cip/2], Ci available for vectorize
        output[index] = ((uint8_t)(param[(ocpIndex * ic + 2*icp2Index+0) * khw + khwIndex] + 8 )) * 16;
        if(2*icp2Index+1 < ic) {
            output[index] += ((uint8_t)(param[(ocpIndex * ic + 2*icp2Index+1) * khw + khwIndex] + 8 ));
        }
    }
}

__global__ void Rearrange_Weight_Int8(const int8_t* param,
    int8_t* output,
    const int khw,
    const size_t maxCount,
    const int oc,
    const int ic,
    DivModFast d_khw,
    DivModFast d_icp
) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
        int icpIndex, temp, ocpIndex, khwIndex;
        d_icp.divmod(index, temp, icpIndex);
        d_khw.divmod(temp, ocpIndex, khwIndex);
        if(icpIndex >= ic || ocpIndex >= oc) {
            output[index] = 0;
            continue;
        }
        // [Co, Ci, KhKw] -> [Cop, KhKw, Cip], Ci available for vectorize
        output[index] = param[(ocpIndex * ic + icpIndex) * khw + khwIndex];
    }
}

bool ConvFpAIntBExecution::isValid(const Convolution2D* conv, Backend* backend) {
    return true;
}


ConvFpAIntBExecution::Resource::Resource(Backend* bn, const MNN::Op* op) {
    mBackend = bn;
    auto runtime = static_cast<CUDABackend*>(bn)->getCUDARuntime();

    auto conv       = op->main_as_Convolution2D();
    auto common     = conv->common();

    //weight host->device
    std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon = ConvolutionCommon::load(op, mBackend, false, true);

    auto oc = common->outputCount();
    if (quanCommon->asymmetric) {
        mQuanC = quanCommon->alpha.size() / 2;
    } else {
        mQuanC = quanCommon->alpha.size();
    }
    auto weightSize = quanCommon->weight.size();
    int l = weightSize / oc;
    int h = oc;
    int ic = common->inputCount();
    if(ic == 0) {
        ic = l / common->kernelX() / common->kernelY();
    }

    int lp = UP_DIV(l, 8) * 8;
    int hp = UP_DIV(h, 8) * 8;
    int quanHp = UP_DIV(mQuanC, 8) * 8;

    // set dequant scale/offset
    {
        float * dequantAlpha = quanCommon->alpha.get();
        std::vector<float> dequantScale(quanHp, 0.0);
        std::vector<float> dequantOffset(quanHp, 0.0);

        for (int o = 0; o < mQuanC; o++) {
            float min = 0.0f;
            float alpha = 0.0f;
            if (quanCommon->asymmetric) {
                min = dequantAlpha[2*o];
                alpha = dequantAlpha[2*o+1];
            } else {
                alpha = dequantAlpha[o];
            }
            dequantScale[o] = alpha;
            dequantOffset[o] = min;
        }

        if(static_cast<CUDABackend*>(bn)->useFp16()) {
            auto tempScaleStorage = static_cast<CUDABackend*>(bn)->getStaticBufferPool()->alloc(quanHp*sizeof(float));
            auto scaleTemp = (float*)((uint8_t*)tempScaleStorage.first + tempScaleStorage.second);
            cuda_check(cudaMemcpy(scaleTemp, dequantScale.data(), quanHp*sizeof(float), cudaMemcpyHostToDevice));

            scaleTensor.reset(Tensor::createDevice<int16_t>({quanHp}));
            bn->onAcquireBuffer(scaleTensor.get(), Backend::STATIC);
            mScale = (void *)scaleTensor.get()->buffer().device;
            callFloat2Half((const void*)scaleTemp, (void*)mScale, quanHp, runtime);

            // Reuse scaleTemp buffer
            cuda_check(cudaMemcpy(scaleTemp, dequantOffset.data(), quanHp*sizeof(float), cudaMemcpyHostToDevice));

            offsetTensor.reset(Tensor::createDevice<int16_t>({quanHp}));
            bn->onAcquireBuffer(offsetTensor.get(), Backend::STATIC);
            mOffset = (void *)offsetTensor.get()->buffer().device;
            callFloat2Half((const void*)scaleTemp, (void*)mOffset, quanHp, runtime);

            static_cast<CUDABackend*>(bn)->getStaticBufferPool()->free(tempScaleStorage);
        } else {
            scaleTensor.reset(Tensor::createDevice<int32_t>({quanHp}));
            bn->onAcquireBuffer(scaleTensor.get(), Backend::STATIC);
            mScale = (void *)scaleTensor.get()->buffer().device;
            cuda_check(cudaMemcpy(mScale, dequantScale.data(), quanHp*sizeof(float), cudaMemcpyHostToDevice));
            
            offsetTensor.reset(Tensor::createDevice<int32_t>({quanHp}));
            bn->onAcquireBuffer(offsetTensor.get(), Backend::STATIC);
            mOffset = (void *)offsetTensor.get()->buffer().device;
            cuda_check(cudaMemcpy(mOffset, dequantOffset.data(), quanHp*sizeof(float), cudaMemcpyHostToDevice));
        }
    }

    // Reorder weight
    {
        int khw = common->kernelX() * common->kernelY();
        int icp = UP_DIV(ic, 8) * 8;
        DivModFast khwD(khw);
        DivModFast icp2D(icp/2);
        DivModFast icpD(icp);

        if(quanCommon->canUseInt4) {
            mIsWeightInt4 = true;

            auto tempCacheBuffer = static_cast<CUDABackend*>(bn)->getStaticBufferPool()->alloc(weightSize * sizeof(int8_t));
            float* cacheWeight = (float*)((uint8_t*)tempCacheBuffer.first + tempCacheBuffer.second);
            runtime->memcpy(cacheWeight, quanCommon->weight.get(), weightSize * sizeof(int8_t), MNNMemcpyHostToDevice);

            weightTensor.reset(Tensor::createDevice<uint8_t>({khw * icp/2 * hp}));
            bn->onAcquireBuffer(weightTensor.get(), Backend::STATIC);
            mFilter = (void *)weightTensor.get()->buffer().device;

            //[Co, Ci, KhKw] -> [Cop, KhKw, Cip/2]
            int block_num = runtime->blocks_num(khw*icp/2*hp);
            int block_size = runtime->threads_num();
            Rearrange_Packed_Weight_Int4<<<block_num, block_size>>>((const uint8_t*)cacheWeight, (uint8_t*)mFilter, khw, khw*icp/2*hp, oc, ic, khwD, icp2D);
            // Rearrange_Weight_Int4<<<block_num, block_size>>>((const int8_t*)cacheWeight, (uint8_t*)mFilter, khw, khw*icp/2*hp, oc, ic, khwD, icp2D);
            checkKernelErrors;

            static_cast<CUDABackend*>(bn)->getStaticBufferPool()->free(tempCacheBuffer);
        } else {
            auto tempCacheBuffer = static_cast<CUDABackend*>(bn)->getStaticBufferPool()->alloc(weightSize * sizeof(int8_t));
            float* cacheWeight = (float*)((uint8_t*)tempCacheBuffer.first + tempCacheBuffer.second);
            runtime->memcpy(cacheWeight, quanCommon->weight.get(), weightSize * sizeof(int8_t), MNNMemcpyHostToDevice);
            
            weightTensor.reset(Tensor::createDevice<int8_t>({khw * icp * hp}));
            bn->onAcquireBuffer(weightTensor.get(), Backend::STATIC);
            mFilter = (void *)weightTensor.get()->buffer().device;

            //[Co, Ci, KhKw] -> [Cop, KhKw, Cip]
            int block_num = runtime->blocks_num(khw*icp*hp);
            int block_size = runtime->threads_num();
            Rearrange_Weight_Int8<<<block_num, block_size>>>((const int8_t*)cacheWeight, (int8_t*)mFilter, khw, khw*icp*hp, oc, ic, khwD, icpD);
            checkKernelErrors;
            static_cast<CUDABackend*>(bn)->getStaticBufferPool()->free(tempCacheBuffer);
        }
    }

    // Copy Bias
    {
        if(static_cast<CUDABackend*>(bn)->useFp16()) {
            int biasSize = conv->bias()->size();
            int hp = UP_DIV(biasSize, 8) * 8;

            auto tempBiasStorage = static_cast<CUDABackend*>(bn)->getStaticBufferPool()->alloc(hp*sizeof(float));
            auto biasTemp = (float*)((uint8_t*)tempBiasStorage.first + tempBiasStorage.second);
            runtime->memset(biasTemp, 0, hp * sizeof(int32_t));
            cuda_check(cudaMemcpy(biasTemp, conv->bias()->data(), conv->bias()->size()*sizeof(float), cudaMemcpyHostToDevice));

            biasTensor.reset(Tensor::createDevice<int16_t>({hp}));
            bn->onAcquireBuffer(biasTensor.get(), Backend::STATIC);
            mBias = (void *)biasTensor.get()->buffer().device;
            callFloat2Half((const void*)biasTemp, (void*)mBias, hp, runtime);

            static_cast<CUDABackend*>(bn)->getStaticBufferPool()->free(tempBiasStorage);
        } else {
            int biasSize = conv->bias()->size();
            int hp = UP_DIV(biasSize, 8) * 8;
            biasTensor.reset(Tensor::createDevice<int32_t>({hp}));
            bn->onAcquireBuffer(biasTensor.get(), Backend::STATIC);
            mBias = (void *)biasTensor.get()->buffer().device;
            runtime->memset(mBias, 0, hp * sizeof(int32_t));
            cuda_check(cudaMemcpy(mBias, conv->bias()->data(), conv->bias()->size()*sizeof(float), cudaMemcpyHostToDevice));
        }
    }

    // High + Int8：预计算 B 的修正项 sum_B_q
    // if(!quanCommon->canUseInt4) {
    //     const int icp = UP_DIV(ic, 8) * 8;
    //     const int num_groups = (mQuanC > 0) ? (mQuanC / oc) : 1;
    //     const int ic_per_group = (num_groups > 0) ? (ic / num_groups) : ic;

    //     const int param_count = oc * num_groups;
    //     mSumBQTensor.reset(Tensor::createDevice<int32_t>({param_count}));
    //     bn->onAcquireBuffer(mSumBQTensor.get(), Backend::STATIC);
    //     mSumBQ = (void*)mSumBQTensor.get()->buffer().device;

    //     const int block_size = BLOCK_SIZE;
    //     const int num_blocks = (param_count + block_size - 1) / block_size;

    //     Precompute_SumBq<<<num_blocks, block_size>>>((const int8_t*)mFilter, (int32_t*)mSumBQ, num_groups, ic_per_group, oc, icp);
    //     checkKernelErrors;
    // }
}

ConvFpAIntBExecution::Resource::~Resource() {
    // Do nothing
}
ConvFpAIntBExecution::ConvFpAIntBExecution(Backend* backend, const MNN::Op* op, std::shared_ptr<Resource> res) : CutlassConvCommonExecution(backend) {
    mOp = op;
    mResource = res;
    auto runtime = static_cast<CUDABackend*>(backend)->getCUDARuntime();
    mPrecisonLevel = static_cast<CUDABackend*>(backend)->getPrecision();
    mFp16Infer = (mPrecisonLevel == 2);
    mFp32Infer = (mPrecisonLevel == 1);
    mFp16Fp32MixInfer = (mPrecisonLevel == 0);
    mBf16Infer = (mPrecisonLevel == 3);
}

ConvFpAIntBExecution::~ConvFpAIntBExecution() {
    if (mDequantFilterTensor) {
        backend()->onReleaseBuffer(mDequantFilterTensor.get(), Backend::STATIC);
    }
}
bool ConvFpAIntBExecution::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (!mValid) {
        return false;
    }
    if (nullptr == dst) {
        return true;
    }
    auto dstExe = new ConvFpAIntBExecution(bn, op, mResource);
    *dst = dstExe;
    return true;
}


ErrorCode ConvFpAIntBExecution::onResize(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) {
    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
    auto input = inputs[0], output = outputs[0];
    const int UNIT = PACK_NUMBER;
    auto convCommon = mOp->main_as_Convolution2D()->common();
    auto pads = ConvolutionCommon::convolutionPadFull(input, output, mOp->main_as_Convolution2D()->common());
    int ic = input->channel();
    const int icp = UP_DIV(ic, 8) * 8;
    auto icDiv = UP_DIV(ic, UNIT);

    mIm2ColParamter.dilateX         = convCommon->dilateX();
    mIm2ColParamter.dilateY         = convCommon->dilateY();
    mIm2ColParamter.strideX         = convCommon->strideX();
    mIm2ColParamter.strideY         = convCommon->strideY();
    mIm2ColParamter.icDiv4          = icDiv;
    mIm2ColParamter.ic              = ic;
    mIm2ColParamter.kernelX         = convCommon->kernelX();
    mIm2ColParamter.kernelY         = convCommon->kernelY();
    mIm2ColParamter.padX = std::get<0>(pads);
    mIm2ColParamter.padY = std::get<1>(pads);

    mIm2ColParamter.ih = input->height();
    mIm2ColParamter.iw = input->width();
    mIm2ColParamter.oh = output->height();
    mIm2ColParamter.ow = output->width();
    mIm2ColParamter.srcZStep = input->height() * input->width() * UNIT * input->batch();
    mIm2ColParamter.srcYStep = input->width() * UNIT;
    mIm2ColParamter.packCUnit = UNIT;

    mActivationType = convCommon->relu() ? 1 : convCommon->relu6() ? 2 : 0;

    //MNN_PRINT("conv size:%d-%d, %d-%d-%d, %d-%d-%d\n", mIm2ColParamter.kernelX, mIm2ColParamter.strideX, input->height(), input->width(), input->channel(), output->height(), output->width(), output->channel());
    int e = output->height() * output->width() * output->batch();
    int l = ic * mIm2ColParamter.kernelX * mIm2ColParamter.kernelY;
    int h = output->channel();
    mGemmInfo.elh[0] = e;
    mGemmInfo.elh[1] = l;
    mGemmInfo.elh[2] = h;
    mGemmInfo.elhPad[0] = UP_DIV(e, 8) * 8;
    mGemmInfo.elhPad[1] = UP_DIV(l, 8) * 8;
    mGemmInfo.elhPad[2] = UP_DIV(h, 8) * 8;
    const int oc = mGemmInfo.elh[2];
    const int ocp = mGemmInfo.elhPad[2];

    mIsConv1x1S1D1P0 = (mIm2ColParamter.kernelX == 1 && mIm2ColParamter.kernelY == 1 &&
        mIm2ColParamter.strideX == 1 && mIm2ColParamter.strideY == 1 &&
        mIm2ColParamter.dilateX == 1 && mIm2ColParamter.dilateY == 1 &&
        mIm2ColParamter.padX == 0 && mIm2ColParamter.padY == 0);

    mNeedIm2Col = !(mIsConv1x1S1D1P0 && (mFp16Infer || mFp32Infer));

    auto pool = static_cast<CUDABackend*>(backend())->getBufferPool();
    if(mNeedIm2Col) {
        size_t im2colBytes = 2;
        // Only when fp32 Im2Col convert to fp32, Fp16Fp32Mix Im2Col convert to fp16
        if(mFp32Infer) {
            im2colBytes = 4;
        }
        auto buffer = pool->alloc(im2colBytes * (size_t)mGemmInfo.elh[0] * (size_t)mGemmInfo.elhPad[1]);
        mIm2ColBuffer = (void*)((uint8_t*)buffer.first + buffer.second);
        pool->free(buffer);
    }

    mDequantFilterTensor = nullptr;

    // 运行时离线反量化
    if (!mFp32Infer) {
        if (mIsConv1x1S1D1P0) {
            std::vector<int> dequantShape = {mGemmInfo.elhPad[2], mGemmInfo.elhPad[1]}; // Shape: [N, K]
            if (mFp16Infer || mFp16Fp32MixInfer) {
                mDequantFilterTensor.reset(Tensor::createDevice<int16_t>(dequantShape));
            } else {
                mDequantFilterTensor.reset(Tensor::createDevice<float>(dequantShape));
            }
            backend()->onAcquireBuffer(mDequantFilterTensor.get(), Backend::DYNAMIC);
            backend()->onReleaseBuffer(mDequantFilterTensor.get(), Backend::DYNAMIC);
            mBiasAddr   = mResource->mBias;
            mBackendPtr = mResource->mBackend;
            mDequantFilter = (void*)mDequantFilterTensor->buffer().device;            
            mFilterAddr = mDequantFilter;
            if (mFp32Infer) {
                return callCutlassGemmCudaCoreFloat32(inputs, outputs);
            }
            mGpuComputeCap = static_cast<CUDABackend*>(backend())->getCUDARuntime()->compute_capability();
            if (mGpuComputeCap < 70) {
                return callCutlassGemmCudaCoreFloat16(inputs, outputs);
            } else if (mGpuComputeCap < 75) {
                return callCutlassGemmTensorCore884(inputs, outputs);
            }
            return callCutlassGemmTensorCore(inputs, outputs);
        }
    }

    return NO_ERROR;
}

ErrorCode ConvFpAIntBExecution::onExecute(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) {
    MNN_ASSERT(inputs.size() == 1);
    MNN_ASSERT(outputs.size() == 1);
    auto input = inputs[0];
    auto output = outputs[0];

    //printf("convcutlass:%p %p\n", input->deviceId(), output->deviceId());
    //MNN_PRINT("cutlass hw:%d-%d\n", input->height(), input->width());
    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
    const void *input_addr = (const void*)inputs[0]->deviceId();
    const void *filter_addr = mResource->mFilter;
    const void *bias_addr = mResource->mBias;
    auto bn = backend();
    void *output_addr = (void*)outputs[0]->deviceId();

    const int sw = mIm2ColParamter.strideX;
    const int sh = mIm2ColParamter.strideY;
    const int kw = mIm2ColParamter.kernelX;
    const int kh = mIm2ColParamter.kernelY;
    const int dw = mIm2ColParamter.dilateX;
    const int dh = mIm2ColParamter.dilateY;
    const int pw = mIm2ColParamter.padX;
    const int ph = mIm2ColParamter.padY;
    const int ic = mIm2ColParamter.ic;
    const int icp = UP_DIV(ic, 8) * 8;
    const int iw = mIm2ColParamter.iw;
    const int ih = mIm2ColParamter.ih;

    const int oc = mGemmInfo.elh[2];
    const int ocp = mGemmInfo.elhPad[2];
    const int ow = mIm2ColParamter.ow;
    const int oh = mIm2ColParamter.oh;

    float maxV = FLT_MAX;
    float minV = -FLT_MAX;
    if (mActivationType == 1) {
        minV = 0.0f;
    }
    if (mActivationType == 2) {
        minV = 0.0f;
        maxV = 6.0f;
    }

    auto total = outputs[0]->batch() * oh * ow * ocp;
    auto& prop = runtime->prop();
    int limitThreads = UP_DIV(total, prop.multiProcessorCount);
    int threadNum = ALIMIN(prop.maxThreadsPerBlock/2, limitThreads);
    int blockNum = prop.multiProcessorCount;

    DivModFast d_oc(ocp);
    DivModFast d_ow(ow);
    DivModFast d_oh(oh);

    const int batch = inputs[0]->batch();
    if (mDequantFilterTensor != nullptr) {
        cuda_check(cudaMemset(mDequantFilter, 0, mDequantFilterTensor->size()));

        if (mResource->mIsWeightInt4) {
            dim3 threads(32, 32);
            dim3 blocks(UP_DIV(ic, threads.x), UP_DIV(oc, threads.y));
            if (mFp16Infer) {
                DequantizeInt4Weight<half, half><<<blocks, threads>>>(
                    (const uint8_t*)mResource->mFilter, (half*)mDequantFilter,
                    (const half*)mResource->mScale, (const half*)mResource->mOffset,
                    oc, ic, icp, mResource->mQuanC
                );
            } else if (mFp16Fp32MixInfer) {
                DequantizeInt4Weight<float, half><<<blocks, threads>>>(
                    (const uint8_t*)mResource->mFilter, (half*)mDequantFilter,
                    (const float*)mResource->mScale, (const float*)mResource->mOffset,
                    oc, ic, icp, mResource->mQuanC
                );
            }
        } else {
            dim3 threads(32, 32);
            dim3 blocks(UP_DIV(ic, threads.x), UP_DIV(oc, threads.y));
            if (mFp16Infer) {
                DequantizeInt8Weight<half, half><<<blocks, threads>>>(
                    (const int8_t*)mResource->mFilter, (half*)mDequantFilter,
                    (const half*)mResource->mScale, (const half*)mResource->mOffset,
                    oc, ic, icp, mResource->mQuanC
                );
            } else if (mFp16Fp32MixInfer) {
                DequantizeInt8Weight<float, half><<<blocks, threads>>>(
                    (const int8_t*)mResource->mFilter, (half*)mDequantFilter,
                    (const float*)mResource->mScale, (const float*)mResource->mOffset,
                    oc, ic, icp, mResource->mQuanC
                );
            }
        }
    }

    
    if(mResource->mIsWeightInt4) {
        if (mIsConv1x1S1D1P0) {
            if (mFp32Infer) { // High + Int4
                if (batch <= 16) {
                    dim3 threads(GEMV_TILE);
                    dim3 blocks(oc, batch);
                    size_t input_smem_size = icp * (mFp16Infer ? sizeof(half) : sizeof(float));
                    size_t reduction_smem_size = GEMV_TILE * sizeof(float);
                    size_t smem_size = input_smem_size + reduction_smem_size;
    
                    GEMV_FpAInt4B<<<blocks, threads, smem_size>>>(
                        (const float*)input_addr, (const uint8_t*)mResource->mFilter,
                        (const float*)mResource->mScale,  (const float*)mResource->mOffset,
                        (const float*)bias_addr, (float*)output_addr,
                        maxV, minV, batch, ic, icp, oc, ocp, mResource->mQuanC
                    );
                } else {
                    dim3 threads(TILE_DIM, TILE_DIM);
                    dim3 blocks(UP_DIV(ocp, TILE_DIM), UP_DIV(batch, TILE_DIM));

                    GEMM_FpAInt4B<<<blocks, threads>>>(
                        (const float*)input_addr, (const uint8_t*)mResource->mFilter,
                        (const float*)mResource->mScale,  (const float*)mResource->mOffset,
                        (const float*)bias_addr, (float*)output_addr,
                        maxV, minV, ic, icp, oc, ocp, batch, mResource->mQuanC
                    );
                }
            } else {
                if (mFp16Fp32MixInfer) {
                    size_t maxCount = mGemmInfo.elh[0] * mGemmInfo.elhPad[1];
                    callFloat2Half(input_addr, mIm2ColBuffer, maxCount, runtime);
                }
                runCutlassGemmFunc();
            }
        } else {
            if(mFp16Infer) {
                CONV_FpAInt4B<<<blockNum, threadNum>>>((const half*)input_addr, (const uint8_t*)mResource->mFilter,
                    (const half*)mResource->mScale,  (const half*)mResource->mOffset, (const half*)bias_addr, (half*)output_addr,
                    maxV, minV, ic, icp, iw, ih, oc, ocp, ow, oh, kw, kh, dw, dh, sw, sh, pw, ph, total, mResource -> mQuanC,
                    d_oc, d_ow, d_oh);
            } else {
                CONV_FpAInt4B<<<blockNum, threadNum>>>((const float*)input_addr, (const uint8_t*)mResource->mFilter,
                    (const float*)mResource->mScale,  (const float*)mResource->mOffset, (const float*)bias_addr, (float*)output_addr,
                    maxV, minV, ic, icp, iw, ih, oc, ocp, ow, oh, kw, kh, dw, dh, sw, sh, pw, ph, total, mResource -> mQuanC,
                    d_oc, d_ow, d_oh);
            }
        }
    } else { // int8
        if (mIsConv1x1S1D1P0) {
            if (mFp32Infer) { // High + Int8
                if (batch <= 16) {
                    dim3 threads(GEMV_TILE);
                    dim3 blocks(oc, batch);
                    
                    if (mFp16Infer) {
                        GEMV_FpAInt8B<<<blocks, threads>>>(
                            (const half*)input_addr, (const int8_t*)mResource->mFilter,
                            (const half*)mResource->mScale,  (const half*)mResource->mOffset,
                            (const half*)bias_addr, (half*)output_addr,
                            maxV, minV, batch, ic, icp, oc, ocp, mResource->mQuanC
                        );
                    } else {
                        GEMV_FpAInt8B<<<blocks, threads>>>(
                            (const float*)input_addr, (const int8_t*)mResource->mFilter,
                            (const float*)mResource->mScale,  (const float*)mResource->mOffset,
                            (const float*)bias_addr, (float*)output_addr,
                            maxV, minV, batch, ic, icp, oc, ocp, mResource->mQuanC
                        );
                    }
                } else {
                    dim3 threads(TILE_DIM, TILE_DIM);
                    dim3 blocks(UP_DIV(ocp, TILE_DIM), UP_DIV(batch, TILE_DIM));

                    GEMM_FpAInt8B<<<blocks, threads>>>(
                        (const float*)input_addr, (const int8_t*)mResource->mFilter,
                        (const float*)mResource->mScale,  (const float*)mResource->mOffset,
                        (const float*)bias_addr, (float*)output_addr,
                        maxV, minV, ic, icp, oc, ocp, batch, mResource->mQuanC
                    );
                }
            } else {
                if (mFp16Fp32MixInfer) {
                    size_t maxCount = mGemmInfo.elh[0] * mGemmInfo.elhPad[1];
                    callFloat2Half(input_addr, mIm2ColBuffer, maxCount, runtime);
                }
                runCutlassGemmFunc();
            }
        } else {
            if(mFp16Infer) {
                CONV_FpAInt8B<<<blockNum, threadNum>>>((const half*)input_addr, (const int8_t*)mResource->mFilter,
                    (const half*)mResource->mScale,  (const half*)mResource->mOffset, (const half*)bias_addr, (half*)output_addr,
                    maxV, minV, ic, icp, iw, ih, oc, ocp, ow, oh, kw, kh, dw, dh, sw, sh, pw, ph, total, mResource -> mQuanC,
                    d_oc, d_ow, d_oh);
            } else {
                CONV_FpAInt8B<<<blockNum, threadNum>>>((const float*)input_addr, (const int8_t*)mResource->mFilter,
                        (const float*)mResource->mScale,  (const float*)mResource->mOffset, (const float*)bias_addr, (float*)output_addr,
                    maxV, minV, ic, icp, iw, ih, oc, ocp, ow, oh, kw, kh, dw, dh, sw, sh, pw, ph, total, mResource -> mQuanC,
                    d_oc, d_ow, d_oh);
            }
        }
    }
    
    checkKernelErrors;
    return NO_ERROR;
}


}// namespace CUDA
}// namespace MNN
