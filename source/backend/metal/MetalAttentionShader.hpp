//
//  MetalAttentionShader.hpp
//  MNN
//
//  Created by MNN on b'2024/12/03'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#if MNN_METAL_ENABLED
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

const char* gMatMulDivMask = R"metal(
#ifdef USE_METAL_TENSOR_OPS
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#endif
#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;
struct Param {
    int query_seq_len;
    int q_seq_piece_len;
    int key_seq_len;
    int head_num;
    int group;
    int head_dim;
    float scale;
    int max_kv_len;
    int batch;
    int kv_align_len;
};

#if MNN_METAL_FLOAT16_STORAGE
typedef simdgroup_half8x8 simdgroup_T8x8;
#else
typedef simdgroup_float8x8 simdgroup_T8x8;
#endif

#define SIMD_GROUP_WIDTH 32

#ifdef USE_METAL_TENSOR_OPS
kernel void prefill_qk_tensor(const device ftype4* input0 [[buffer(0)]],
    device ftype* output [[buffer(1)]],
    device ftype4* past_key [[buffer(2)]],
    constant int &seq_idx [[buffer(3)]],
    constant Param& param [[buffer(4)]],
    constant int& kv_start [[buffer(5)]],
    constant int& output_k_len [[buffer(6)]],
#ifdef ADD_MASK
    const device ftype* mask [[buffer(7)]],
#elif defined(SET_MASK)
    const device int* mask [[buffer(7)]],
#endif
    uint3 gid[[threadgroup_position_in_grid]],
    uint tiitg[[thread_index_in_threadgroup]],
    uint tiisg[[thread_index_in_simdgroup]],
    uint sgitg[[simdgroup_index_in_threadgroup]]
) {
    /*
     // Read:
     ftype 0~1023   ---> input: [M32, K32]
     ftype 1024~2047 ---> input: [N32, K32]
     // Write:
     float 0~1023 ---> input: [M32, N32]
     */
    threadgroup ftype sdata[2048] = {0.f};

    const int K = 32, M = 32, N = 32; 
    const int tb_offset = M * K;
    auto tA = tensor<threadgroup ftype, dextents<int32_t, 2>, tensor_inline>((threadgroup ftype*)sdata, dextents<int32_t, 2>(K, M));//[M, K]
    auto tB = tensor<threadgroup ftype, dextents<int32_t, 2>, tensor_inline>((threadgroup ftype*)sdata + tb_offset, dextents<int32_t, 2>(K, N));//[N, K]

    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(M, N, K, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mmOps;

    auto cT = mmOps.get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();

    // A: [32, 4]
    int ml = tiitg / 4;// 0~31
    int kl = tiitg % 4;// 0~3

    // B: [32, 4]
    int nl = ml;

    // C: [32, 4]
    int mcl = ml;// 0~31
    int ncl = kl;// 0~3

    const int slq = gid.x; // q_seq_len/32 -> M/32
    const int slk = gid.y; // k_seq_len/32 -> N/32
    const int z = gid.z; // head_num * batch

    /** Q:
     threadgroup: [M32, K32] -> [M32, K4, K2, K4]
     index : [ml, kl, K2, K4]
     each thread: K8
     layout: [B0, M, B1, K] -> [B0, M/32, M32, B1, K/32, K4, K2, K4]
     index : [z/head_num, slq, ml, z%head_num, K/32, kl, K2, K4]
     offset: ((z/head_num * q_seq_len + (slq * 32 + ml)) * head_num + z%head_num) * K/4 + (0 * 4 + kl) * 2 + 0
     */
    /** K:
     threadgroup: [N32, K32] -> [M32, K4, K2, K4]
     index : [nl, kl, K2, K4]
     each thread: K8
     layout: [N, B/G, K] -> [N/32, N32, B/G, K/32, K4, K2, K4]
     index : [slk, nl, B/G, K/32, kl, K2, K4]
     offset: ((slk * 32 + nl) * B/G + z/G) * K/4 + (0 * 4 + kl) * 2 + 0
     */
    /** output:
     threadgroup: [M32, N32] -> [M32, N4, N8]
     each thread: N8
     layout: [B, M, N] -> [B, M/32, M32, N/32, N4, N8]
     index : [z, slq, mcl, slk, ncl, N8]
     offset: (z * q_seq_len + slq * 32 + mcl) * N + (slk * 4 + ncl) * 8 + 0
     */

    int group = param.group;
    int q_seq_len = param.query_seq_len;
    int q_seq_piece_len = param.q_seq_piece_len;
    int k_seq_len = param.key_seq_len;
    int head_num = param.head_num;
    int head_dim = param.head_dim;

    const int b = z / head_num;
    const int hn = z % head_num;
    int zin = hn / param.group;

    int idx_slq = seq_idx * q_seq_piece_len + slq * 32 + ml < q_seq_len ? seq_idx * q_seq_piece_len + slq * 32 + ml : q_seq_len - 1;
    int idx_slk_global = kv_start + slk * 32 + nl;
    int idx_slk = idx_slk_global < k_seq_len ? idx_slk_global : k_seq_len - 1;
    // [mBatch, mSeqLen, mNumHead, mHeadDim]
    auto A_offset = input0 + ((b * q_seq_len + idx_slq) * head_num + hn) * head_dim / 4 + (0 * 4 + kl) * 2 + 0;

    // [mKvSeqLen, mBatch, mKvNumHead, mHeadDim]
    auto B_offset = past_key + ((idx_slk * param.batch + b)* head_num / group + zin) * head_dim / 4 + (0 * 4 + kl) * 2 + 0;

    for(int i = 0; i < head_dim/4; i += 8){
        ((threadgroup ftype4*)sdata)[(ml * 4 + kl) * 2 + 0] = A_offset[i + 0];
        ((threadgroup ftype4*)sdata)[(ml * 4 + kl) * 2 + 1] = A_offset[i + 1];

        ((threadgroup ftype4*)sdata)[256 + (nl * 4 + kl) * 2 + 0] = B_offset[i + 0];
        ((threadgroup ftype4*)sdata)[256 + (nl * 4 + kl) * 2 + 1] = B_offset[i + 1];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto sA = tA.slice(0, 0);
        auto sB = tB.slice(0, 0);

        mmOps.run(sA, sB, cT);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    auto tC = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>((threadgroup float*)sdata, dextents<int32_t, 2>(N, M)); // [M , N]
    cT.store(tC);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // [M32, N4, N8]
    auto sindex_base = (mcl * 4 + ncl) * 8 + 0;

    float Vscale = (float)param.scale;

    int base_k_idx =  (slk * 4 + ncl) * 8 + 0;
    auto xy_out = output + (z * q_seq_piece_len + slq * 32 + mcl) * output_k_len + base_k_idx + 0;
    if(slq * 32 + mcl < q_seq_piece_len &&  seq_idx * q_seq_piece_len + slq * 32 + mcl < q_seq_len) {
        int ori_q_idx = seq_idx * q_seq_piece_len + slq * 32 + mcl;
        if(base_k_idx + 0 < output_k_len) {
            auto out0 =  ((threadgroup float*)sdata)[sindex_base + 0] * Vscale;
            #ifdef ADD_MASK
                auto mask_val = (kv_start + base_k_idx + 0) >= k_seq_len - q_seq_len ? mask[(ori_q_idx * q_seq_len + (kv_start + base_k_idx + 0) - k_seq_len + q_seq_len)] : 0.0;
                out0 = mask_val + out0;
            #elif defined(SET_MASK)
                out0 = mask[(ori_q_idx * k_seq_len + (kv_start + base_k_idx + 0))] == 0 ? -FLT_MAX : out0;
            #endif
            xy_out[0] = out0;
        }
        if(base_k_idx + 1 < output_k_len) {
            auto out0 =  ((threadgroup float*)sdata)[sindex_base + 1] * Vscale;
            #ifdef ADD_MASK
                auto mask_val = (kv_start + base_k_idx + 1) >= k_seq_len - q_seq_len ? mask[(ori_q_idx * q_seq_len + (kv_start + base_k_idx + 1) - k_seq_len + q_seq_len)] : 0.0;
                out0 = mask_val + out0;
            #elif defined(SET_MASK)
                out0 = mask[(ori_q_idx * k_seq_len + (kv_start + base_k_idx + 1))] == 0 ? -FLT_MAX : out0;
            #endif
            xy_out[1] = out0;
        }
        if(base_k_idx + 2 < output_k_len) {
            auto out0 =  ((threadgroup float*)sdata)[sindex_base + 2] * Vscale;
            #ifdef ADD_MASK
                auto mask_val = (kv_start + base_k_idx + 2) >= k_seq_len - q_seq_len ? mask[(ori_q_idx * q_seq_len + (kv_start + base_k_idx + 2) - k_seq_len + q_seq_len)] : 0.0;
                out0 = mask_val + out0;
            #elif defined(SET_MASK)
                out0 = mask[(ori_q_idx * k_seq_len + (kv_start + base_k_idx + 2))] == 0 ? -FLT_MAX : out0;
            #endif
            xy_out[2] = out0;
        }
        if(base_k_idx + 3 < output_k_len) {
            auto out0 =  ((threadgroup float*)sdata)[sindex_base + 3] * Vscale;
            #ifdef ADD_MASK
                auto mask_val = (kv_start + base_k_idx + 3) >= k_seq_len - q_seq_len ? mask[(ori_q_idx * q_seq_len + (kv_start + base_k_idx + 3) - k_seq_len + q_seq_len)] : 0.0;
                out0 = mask_val + out0;
            #elif defined(SET_MASK)
                out0 = mask[(ori_q_idx * k_seq_len + (kv_start + base_k_idx + 3))] == 0 ? -FLT_MAX : out0;
            #endif
            xy_out[3] = out0;
        }
        if(base_k_idx + 4 < output_k_len) {
            auto out0 =  ((threadgroup float*)sdata)[sindex_base + 4] * Vscale;
            #ifdef ADD_MASK
                auto mask_val = (kv_start + base_k_idx + 4) >= k_seq_len - q_seq_len ? mask[(ori_q_idx * q_seq_len + (kv_start + base_k_idx + 4) - k_seq_len + q_seq_len)] : 0.0;
                out0 = mask_val + out0;
            #elif defined(SET_MASK)
                out0 = mask[(ori_q_idx * k_seq_len + (kv_start + base_k_idx + 4))] == 0 ? -FLT_MAX : out0;
            #endif
            xy_out[4] = out0;
        }
        if(base_k_idx + 5 < output_k_len) {
            auto out0 =  ((threadgroup float*)sdata)[sindex_base + 5] * Vscale;
            #ifdef ADD_MASK
                auto mask_val = (kv_start + base_k_idx + 5) >= k_seq_len - q_seq_len ? mask[(ori_q_idx * q_seq_len + (kv_start + base_k_idx + 5) - k_seq_len + q_seq_len)] : 0.0;
                out0 = mask_val + out0;
            #elif defined(SET_MASK)
                out0 = mask[(ori_q_idx * k_seq_len + (kv_start + base_k_idx + 5))] == 0 ? -FLT_MAX : out0;
            #endif
            xy_out[5] = out0;
        }
        if(base_k_idx + 6 < output_k_len) {
            auto out0 =  ((threadgroup float*)sdata)[sindex_base + 6] * Vscale;
            #ifdef ADD_MASK
                auto mask_val = (kv_start + base_k_idx + 6) >= k_seq_len - q_seq_len ? mask[(ori_q_idx * q_seq_len + (kv_start + base_k_idx + 6) - k_seq_len + q_seq_len)] : 0.0;
                out0 = mask_val + out0;
            #elif defined(SET_MASK)
                out0 = mask[(ori_q_idx * k_seq_len + (kv_start + base_k_idx + 6))] == 0 ? -FLT_MAX : out0;
            #endif
            xy_out[6] = out0;
        }
        if(base_k_idx + 7 < output_k_len) {
            auto out0 =  ((threadgroup float*)sdata)[sindex_base + 7] * Vscale;
            #ifdef ADD_MASK
                auto mask_val = (kv_start + base_k_idx + 7) >= k_seq_len - q_seq_len ? mask[(ori_q_idx * q_seq_len + (kv_start + base_k_idx + 7) - k_seq_len + q_seq_len)] : 0.0;
                out0 = mask_val + out0;
            #elif defined(SET_MASK)
                out0 = mask[(ori_q_idx * k_seq_len + (kv_start + base_k_idx + 7))] == 0 ? -FLT_MAX : out0;
            #endif
            xy_out[7] = out0;
        }
    }



}
#endif

kernel void prefill_qk(const device ftype* input0 [[buffer(0)]],
    device ftype* output [[buffer(1)]],
    device ftype* past_key [[buffer(2)]],
    constant int &seq_idx [[buffer(3)]],
    constant Param& param [[buffer(4)]],
    constant int& kv_start [[buffer(5)]],
    constant int& output_k_len [[buffer(6)]],
#ifdef ADD_MASK
    const device ftype* mask [[buffer(7)]],
#elif defined(SET_MASK)
    const device int* mask [[buffer(7)]],
#endif
#ifdef SIMD_GROUP_MATRIX
    uint3 gid[[threadgroup_position_in_grid]],
    uint tiitg[[thread_index_in_threadgroup]],
    uint tiisg[[thread_index_in_simdgroup]],
    uint sgitg[[simdgroup_index_in_threadgroup]]
#else
    uint3 gid[[thread_position_in_grid]]
#endif
) {

#ifdef SIMD_GROUP_MATRIX

    /*
     // Read:
     ftype 0~127   ---> input: [M16, K8]
     ftype 128~255 ---> input: [K8, N16]
     // Write:
     float 0~255 ---> input: [N2, M2, M8, N8]
     */
    threadgroup float sdata[256] = {0.f};

#ifdef USE_METAL_TENSOR_OPS

    const int K = 8, M = 16, N = 16; 
    auto tA = tensor<threadgroup ftype, dextents<int32_t, 2>, tensor_inline>((threadgroup ftype*)sdata, dextents<int32_t, 2>(K, M));//[M, K]
    auto tB = tensor<threadgroup ftype, dextents<int32_t, 2>, tensor_inline>((threadgroup ftype*)sdata + 128, dextents<int32_t, 2>(N, K));//[K, N]

    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(M, N, K, false, false, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<1>> mmOps;

    auto cT = mmOps.get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();
#else
    simdgroup_T8x8 sga[2];
    simdgroup_T8x8 sgb[2];
    simdgroup_float8x8 sgd[4];
    for (int i = 0; i < 4; i++){
        sgd[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }
#endif

    int kl = tiitg % 2;// 0~1
    int rcl = tiitg / 2;// 0~15

    const int slq = gid.x; // q_seq_len/16 -> M/16
    const int slk = gid.y; // k_seq_len/16 -> N/16
    const int z = gid.z; // head_num * batch

    /** Q:
     threadgroup: [M16, K8]
     each thread: K4
     layout: [B0, M, B1, K] -> [B0, M/16, M16, B1, K/8, K2, K4]
     index : [z/head_num, slq, rcl, z%head_num, 0, kl, K4]
     offset: ((z/head_num * q_seq_len + (slq * 16 + rcl)) * head_num + z%head_num) * K + (0 * 2 + kl) * 4 + 0
     */
    /** K:
     threadgroup: [K8, N16]
     each thread: N4
     layout: [N, B/G, K] -> [N/16, N16, B/G, K/8, K2, K4]
     index : [slk, rcl, B/G, 0, kl, 0]
     offset: ((slk * 16 + rcl) * B/G + z/G) * K + 0 * 8 + kl * 4 + 0
     */
    /** output:
     threadgroup: [M16, N16]
     each thread: N8
     layout: [B, M, N] -> [B, M/16, M16, N/16, N2, N8]
     index : [z, sl, rcl, kl, 0]
     offset: (z * M + sl * 16 + rcl) * N + slk * 16 + kl * 8 + 0
     */

    int group = param.group;
    int q_seq_len = param.query_seq_len;
    int q_seq_piece_len = param.q_seq_piece_len;
    int k_seq_len = param.key_seq_len;
    int head_num = param.head_num;
    int head_dim = param.head_dim;

    const int b = z / head_num;
    const int hn = z % head_num;
    int zin = hn / param.group;

    int idx_slq = seq_idx * q_seq_piece_len + slq * 16 + rcl < q_seq_len ? seq_idx * q_seq_piece_len + slq * 16 + rcl : q_seq_len - 1;
    int idx_slk_global = kv_start + slk * 16 + rcl;
    int idx_slk = idx_slk_global < k_seq_len ? idx_slk_global : k_seq_len - 1;
    // [mBatch, mSeqLen, mNumHead, mHeadDim]
    auto A_offset = input0 + ((b * q_seq_len + idx_slq) * head_num + hn) * head_dim + (0 * 2 + kl) * 4 + 0;

    // [mKvSeqLen, mBatch, mKvNumHead, mHeadDim]
    auto B_offset = past_key + ((idx_slk * param.batch + b)* head_num / group + zin) * head_dim + 0 * 8 + kl * 4 + 0;

    for(int i = 0; i < head_dim; i += 8){
        ((threadgroup ftype*)sdata)[rcl * 8 + kl * 4 + 0] = A_offset[i + 0];
        ((threadgroup ftype*)sdata)[rcl * 8 + kl * 4 + 1] = A_offset[i + 1];
        ((threadgroup ftype*)sdata)[rcl * 8 + kl * 4 + 2] = A_offset[i + 2];
        ((threadgroup ftype*)sdata)[rcl * 8 + kl * 4 + 3] = A_offset[i + 3];

        ((threadgroup ftype*)sdata)[128 + (kl * 4 + 0) * 16 + rcl] = B_offset[i + 0];
        ((threadgroup ftype*)sdata)[128 + (kl * 4 + 1) * 16 + rcl] = B_offset[i + 1];
        ((threadgroup ftype*)sdata)[128 + (kl * 4 + 2) * 16 + rcl] = B_offset[i + 2];
        ((threadgroup ftype*)sdata)[128 + (kl * 4 + 3) * 16 + rcl] = B_offset[i + 3];
        threadgroup_barrier(mem_flags::mem_threadgroup);

#ifdef USE_METAL_TENSOR_OPS
        auto sA = tA.slice(0, 0);
        auto sB = tB.slice(0, 0);

        mmOps.run(sA, sB, cT);
#else
        simdgroup_load(sga[0], (const threadgroup ftype*)sdata, 8);
        simdgroup_load(sga[1], ((const threadgroup ftype*)sdata) + 64, 8);
        
        simdgroup_load(sgb[0], ((const threadgroup ftype*)sdata) + 128, 16);
        simdgroup_load(sgb[1], ((const threadgroup ftype*)sdata) + 136, 16);
        
        simdgroup_multiply_accumulate(sgd[0], sga[0], sgb[0], sgd[0]);
        simdgroup_multiply_accumulate(sgd[1], sga[1], sgb[0], sgd[1]);
        simdgroup_multiply_accumulate(sgd[2], sga[0], sgb[1], sgd[2]);
        simdgroup_multiply_accumulate(sgd[3], sga[1], sgb[1], sgd[3]);
#endif
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

#ifdef USE_METAL_TENSOR_OPS

    auto tC = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>((threadgroup float*)sdata, dextents<int32_t, 2>(N, M)); // [M , N]
    cT.store(tC);
#else
    simdgroup_store(sgd[0], (threadgroup float*)sdata, 8);
    simdgroup_store(sgd[1], (threadgroup float*)sdata + 64, 8);
    simdgroup_store(sgd[2], (threadgroup float*)sdata + 128, 8);
    simdgroup_store(sgd[3], (threadgroup float*)sdata + 192, 8);
#endif

    threadgroup_barrier(mem_flags::mem_threadgroup);

#ifdef USE_METAL_TENSOR_OPS
    // [M16, N2, N8]
    auto sindex_base = (rcl * 2 + kl) * 8 + 0;
#else
    // [N2, M2, M8, N8]
    auto sindex_base = (kl * 16 + rcl) * 8 + 0;
#endif

    float Vscale = (float)param.scale;

    auto xy_out = output + (z * q_seq_piece_len + slq * 16 + rcl) * output_k_len + slk * 16 + kl * 8 + 0;
    if(slq * 16 + rcl < q_seq_piece_len &&  seq_idx * q_seq_piece_len + slq * 16 + rcl < q_seq_len) {
        int ori_q_idx = seq_idx * q_seq_piece_len + slq * 16 + rcl;
        if(slk * 16 + kl * 8 + 0 < output_k_len) {
            auto out0 =  ((threadgroup float*)sdata)[sindex_base + 0] * Vscale;
            #ifdef ADD_MASK
                auto mask_val = (kv_start + slk * 16 + kl * 8 + 0) >= k_seq_len - q_seq_len ? mask[(ori_q_idx * q_seq_len + (kv_start + slk * 16 + kl * 8 + 0) - k_seq_len + q_seq_len)] : 0.0;
                out0 = mask_val + out0;
            #elif defined(SET_MASK)
                out0 = mask[(ori_q_idx * k_seq_len + (kv_start + slk * 16 + kl * 8 + 0))] == 0 ? -FLT_MAX : out0;
            #endif
            xy_out[0] = out0;
        }
        if(slk * 16 + kl * 8 + 1 < output_k_len) {
            auto out0 =  ((threadgroup float*)sdata)[sindex_base + 1] * Vscale;
            #ifdef ADD_MASK
                auto mask_val = (kv_start + slk * 16 + kl * 8 + 1) >= k_seq_len - q_seq_len ? mask[(ori_q_idx * q_seq_len + (kv_start + slk * 16 + kl * 8 + 1) - k_seq_len + q_seq_len)] : 0.0;
                out0 = mask_val + out0;
            #elif defined(SET_MASK)
                out0 = mask[(ori_q_idx * k_seq_len + (kv_start + slk * 16 + kl * 8 + 1))] == 0 ? -FLT_MAX : out0;
            #endif
            xy_out[1] = out0;
        }
        if(slk * 16 + kl * 8 + 2 < output_k_len) {
            auto out0 =  ((threadgroup float*)sdata)[sindex_base + 2] * Vscale;
            #ifdef ADD_MASK
                auto mask_val = (kv_start + slk * 16 + kl * 8 + 2) >= k_seq_len - q_seq_len ? mask[(ori_q_idx * q_seq_len + (kv_start + slk * 16 + kl * 8 + 2) - k_seq_len + q_seq_len)] : 0.0;
                out0 = mask_val + out0;
            #elif defined(SET_MASK)
                out0 = mask[(ori_q_idx * k_seq_len + (kv_start + slk * 16 + kl * 8 + 2))] == 0 ? -FLT_MAX : out0;
            #endif
            xy_out[2] = out0;
        }
        if(slk * 16 + kl * 8 + 3 < output_k_len) {
            auto out0 =  ((threadgroup float*)sdata)[sindex_base + 3] * Vscale;
            #ifdef ADD_MASK
                auto mask_val = (kv_start + slk * 16 + kl * 8 + 3) >= k_seq_len - q_seq_len ? mask[(ori_q_idx * q_seq_len + (kv_start + slk * 16 + kl * 8 + 3) - k_seq_len + q_seq_len)] : 0.0;
                out0 = mask_val + out0;
            #elif defined(SET_MASK)
                out0 = mask[(ori_q_idx * k_seq_len + (kv_start + slk * 16 + kl * 8 + 3))] == 0 ? -FLT_MAX : out0;
            #endif
            xy_out[3] = out0;
        }
        if(slk * 16 + kl * 8 + 4 < output_k_len) {
            auto out0 =  ((threadgroup float*)sdata)[sindex_base + 4] * Vscale;
            #ifdef ADD_MASK
                auto mask_val = (kv_start + slk * 16 + kl * 8 + 4) >= k_seq_len - q_seq_len ? mask[(ori_q_idx * q_seq_len + (kv_start + slk * 16 + kl * 8 + 4) - k_seq_len + q_seq_len)] : 0.0;
                out0 = mask_val + out0;
            #elif defined(SET_MASK)
                out0 = mask[(ori_q_idx * k_seq_len + (kv_start + slk * 16 + kl * 8 + 4))] == 0 ? -FLT_MAX : out0;
            #endif
            xy_out[4] = out0;
        }
        if(slk * 16 + kl * 8 + 5 < output_k_len) {
            auto out0 =  ((threadgroup float*)sdata)[sindex_base + 5] * Vscale;
            #ifdef ADD_MASK
                auto mask_val = (kv_start + slk * 16 + kl * 8 + 5) >= k_seq_len - q_seq_len ? mask[(ori_q_idx * q_seq_len + (kv_start + slk * 16 + kl * 8 + 5) - k_seq_len + q_seq_len)] : 0.0;
                out0 = mask_val + out0;
            #elif defined(SET_MASK)
                out0 = mask[(ori_q_idx * k_seq_len + (kv_start + slk * 16 + kl * 8 + 5))] == 0 ? -FLT_MAX : out0;
            #endif
            xy_out[5] = out0;
        }
        if(slk * 16 + kl * 8 + 6 < output_k_len) {
            auto out0 =  ((threadgroup float*)sdata)[sindex_base + 6] * Vscale;
            #ifdef ADD_MASK
                auto mask_val = (kv_start + slk * 16 + kl * 8 + 6) >= k_seq_len - q_seq_len ? mask[(ori_q_idx * q_seq_len + (kv_start + slk * 16 + kl * 8 + 6) - k_seq_len + q_seq_len)] : 0.0;
                out0 = mask_val + out0;
            #elif defined(SET_MASK)
                out0 = mask[(ori_q_idx * k_seq_len + (kv_start + slk * 16 + kl * 8 + 6))] == 0 ? -FLT_MAX : out0;
            #endif
            xy_out[6] = out0;
        }
        if(slk * 16 + kl * 8 + 7 < output_k_len) {
            auto out0 =  ((threadgroup float*)sdata)[sindex_base + 7] * Vscale;
            #ifdef ADD_MASK
                auto mask_val = (kv_start + slk * 16 + kl * 8 + 7) >= k_seq_len - q_seq_len ? mask[(ori_q_idx * q_seq_len + (kv_start + slk * 16 + kl * 8 + 7) - k_seq_len + q_seq_len)] : 0.0;
                out0 = mask_val + out0;
            #elif defined(SET_MASK)
                out0 = mask[(ori_q_idx * k_seq_len + (kv_start + slk * 16 + kl * 8 + 7))] == 0 ? -FLT_MAX : out0;
            #endif
            xy_out[7] = out0;
        }
    }

#else
    const int x = gid.x; // query_seq_len
    const int y = gid.y; // head_num * batch
    const int z = gid.z; // key_seq_len
    
    int q_idx = seq_idx * param.q_seq_piece_len + x;
    int z_global = kv_start + z;
    if (x >= param.q_seq_piece_len || q_idx >= param.query_seq_len || y >= param.head_num * param.batch || z_global >= param.key_seq_len) {
        return;
    }
    int group = param.group;
    int query_seq_len = param.query_seq_len;
    int key_seq_len = param.key_seq_len;
    int head_num = param.head_num;
    int head_dim = param.head_dim;
    int b  = y / head_num;
    int hn = y % head_num;
    
    const int offset = head_num * head_dim;
    const int offset_head = y * head_dim;
    const int offset_head_kv = (hn / group) * head_dim;
    // [mBatch, mSeqLen, mNumHead, mHeadDim]
    const device ftype* A_offset = input0 + (b * query_seq_len + q_idx) * offset + offset_head;

    float Vscale = (float)param.scale;
    // [mKvSeqLen, mBatch, mKvNumHead, mHeadDim]
    device const ftype* B_offset = past_key + (z_global * param.batch + b) * offset / group + offset_head_kv;
    const int output_offset = y * param.q_seq_piece_len * output_k_len;
    float out0 = 0.0;
    
    for(int i = 0; i < head_dim; ++i){
        float A = (float)(A_offset[i]);
        float B = (float)(B_offset[i]);
        out0 += B * A;
    }
    
    out0 *= Vscale;
    
 #ifdef ADD_MASK
    auto mask_val = z_global >= key_seq_len - query_seq_len ? mask[((q_idx + 0) * query_seq_len + (z_global - key_seq_len + query_seq_len))] : 0.0;
    out0 = mask_val + out0;
 #elif defined(SET_MASK)
    out0 = mask[((q_idx + 0) * key_seq_len + (z_global + 0))] == 0 ? -FLT_MAX : out0;
 #endif
    output[output_offset + x * output_k_len + z] = (ftype)out0;
#endif
}

kernel void decode_qk(const device ftype* input0 [[buffer(0)]],
    device ftype* output [[buffer(1)]],
    device ftype* past_key [[buffer(2)]],
    // decode actually not compute in block
    constant int &seq_idx [[buffer(3)]],
    constant Param& param [[buffer(4)]],
    constant int& kv_start [[buffer(5)]],
    constant int& output_k_len [[buffer(6)]],
#ifdef ADD_MASK
    const device ftype* mask [[buffer(7)]],
#elif defined(SET_MASK)
    const device int* mask [[buffer(7)]],
#endif
    uint3 gid[[thread_position_in_grid]]
) {
    int x = gid.x; // query_seq_len
    int y = gid.y; // head_num * batch
    int z = gid.z; // key_seq_len
    int group = param.group;
    int kv_head_num = param.head_num / group;
    if (x >= param.query_seq_len || y >= kv_head_num * param.batch || z >= param.key_seq_len) {
        return;
    }

    int key_seq_len = param.key_seq_len;
    int head_num = param.head_num;
    int head_dim = param.head_dim;
    
    int b  = y / kv_head_num;
    int kv_hn = y % kv_head_num;
    const int offset = head_num * head_dim;
    const int offset_head = kv_hn * group * head_dim;
    const int offset_head_kv = kv_hn * head_dim;

    // [mBatch, mSeqLen, mNumHead, mHeadDim]
    const device ftype* A_offset = input0 + (b * param.query_seq_len + x) * offset + offset_head;
    // [mKvSeqLen, mBatch, mKvNumHead, mHeadDim]
    device ftype* Pastkey_offset = past_key + (z * param.batch + b) * offset / group + offset_head_kv;
    float Vscale = (float)param.scale;



    float out[GROUP_SIZE] = {0.0};
    #ifdef HEAD_DIM_UNALIGNED_4
    {
        for(int i = 0; i < head_dim; i++){
            float B = (float)Pastkey_offset[i];
            for(int j = 0; j < group; j++) {
                float A = A_offset[i + head_dim * j];
                out[j] += A * B;
            }
        }
    }
    #else
    {
        for(int i = 0; i < head_dim/4; i++){
            float4 B = float4(((const device ftype4*)Pastkey_offset)[i]);
            for(int j = 0; j < group; j++) {
                float4 A = float4(((const device ftype4*)(A_offset + head_dim * j))[i]);
                out[j] += dot(A, B);
            }
        }
    }
    #endif
    #ifdef ADD_MASK
        float mask_val = z >= key_seq_len - param.query_seq_len ? mask[((x + 0) * param.query_seq_len + (z - key_seq_len + param.query_seq_len))] : 0.0;
    #elif defined(SET_MASK)
        int mask_val = mask[((x + 0) * key_seq_len + (z + 0))];
    #endif
    for(int j = 0; j < group; j++) {
        out[j] *= Vscale;
        #ifdef ADD_MASK
            out[j] += mask_val;
        #elif SET_MASK
            out[j] = mask_val == 0 ? -FLT_MAX : out[j];
        #endif
        output[((y * group + j) * param.query_seq_len + x) * key_seq_len + z] = (ftype)out[j];
    }
}

)metal";

const char* gCopyPastKV = R"metal(
#include <metal_stdlib>
using namespace metal;
struct Param {
    int head_count;
    int kv_seq_len;
    int max_kv_len;
    int dst_k_offset;
    int dst_v_offset;
    int batch;
};
// Key:   [batch, kv_seq_len, head_num / group * head_dim] -> [max_kv_len, batch, head_num / group * head_dim]
// Value: [batch, kv_seq_len, head_num / group * head_dim] -> [batch, head_num / group * head_dim, max_kv_len]
kernel void copy(const device ftype* input0 [[buffer(0)]],
    const device ftype* input1 [[buffer(1)]],
    device ftype* output0 [[buffer(2)]],
    device ftype* output1 [[buffer(3)]],
    constant Param& param [[buffer(4)]],
    uint3 gid[[thread_position_in_grid]]
) {
    const int x = gid.x; // head_num / group * head_dim
    const int y = gid.y; // kv_seq_len
    const int b = gid.z; // batch
    if (x >= param.head_count || y >= param.kv_seq_len || b >= param.batch) {
        return;
    }
    const int in_idx  = (b * param.kv_seq_len + y) * param.head_count + x;
    int out_idx = param.dst_k_offset + (y * param.batch + b) * param.head_count + x;
    output0[out_idx] = input0[in_idx];

    out_idx = param.dst_v_offset + (b * param.head_count + x) * param.max_kv_len + y;
    output1[out_idx] = input1[in_idx];
}
)metal";

const char* gMatMulQKV = R"metal(
#ifdef USE_METAL_TENSOR_OPS
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#endif
#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;
struct Param {
    int query_seq_len;
    int q_seq_piece_len;
    int key_seq_len;
    int head_num;
    int group;
    int head_dim;
    float scale;
    int max_kv_len;
    int batch;
    int kv_align_len;
};
#if MNN_METAL_FLOAT16_STORAGE
typedef simdgroup_half8x8 simdgroup_T8x8;
#else
typedef simdgroup_float8x8 simdgroup_T8x8;
#endif

#ifdef USE_METAL_TENSOR_OPS
kernel void prefill_qkv_tensor(const device ftype* input0 [[buffer(0)]],
    device ftype4* output [[buffer(1)]],
    device ftype4* past_value [[buffer(2)]],
    constant int &seq_idx [[buffer(3)]],
    constant Param& param [[buffer(4)]],
    uint3 gid[[threadgroup_position_in_grid]],
    uint tiitg[[thread_index_in_threadgroup]],
    uint tiisg[[thread_index_in_simdgroup]],
    uint sgitg[[simdgroup_index_in_threadgroup]]
) {
    /*
     // Read:
     ftype 0~1023   ---> input: [M32, K32]
     ftype 1024~2047 ---> input: [N32, K32]
     // Write:
     float 0~1023 ---> input: [M32, N32]
     */

    threadgroup ftype sdata[2048] = {0.f};

    const int K = 32, M = 32, N = 32; 
    const int tb_offset = M * K;
    auto tA = tensor<threadgroup ftype, dextents<int32_t, 2>, tensor_inline>((threadgroup ftype*)sdata, dextents<int32_t, 2>(K, M));//[M, K]
    auto tB = tensor<threadgroup ftype, dextents<int32_t, 2>, tensor_inline>((threadgroup ftype*)sdata + tb_offset, dextents<int32_t, 2>(K, N));//[N, K]

    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(M, N, K, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mmOps;

    auto cT = mmOps.get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();

    // QK:[32, 4] 
    int ml = tiitg / 4;// 0~31
    int kl = tiitg % 4;// 0~3

    // V: [32, 4]
    int nl = ml;// 0~31
    int kvl = kl;// 0~3

    // QKV: [32, 4]
    int mcl = ml;// 0~31
    int ncl = kl;// 0~3

    const int sl = gid.x; // q_seq_len/32 -> M/32
    const int hm = gid.y; // head_dim/32 -> N/32
    const int z = gid.z; // head_num * batch

    /** QK:
     threadgroup: [M32, K32] -> [M32, K4, K8]
     index; [ml, kl, K8]
     each thread: K8
     layout: [B, M, K] -> [B, M/32, M32, K/32, K4, K8]
     index : [z, sl, ml, K/32, kl, K2, K4]
     offset: (z * M + sl * 32 + ml) * K + (0 * 4 + kl) * 8 + 0
     */
    /** V:
     threadgroup: [N32, K32] -> [N32, K4, K8]
     index; [nl, kvl, K8]
     each thread: K8
     layout: [B/G, N, K] -> [B/G, N/32, N32, K/32, K4, K8]
     index : [zin, hm, nl, K/32, kvl, K2, K4]
     offset: ((zin * head_dim + hm * 32 + nl) * param.max_kv_len/4 + (0 * 4 + kvl) * 2 + 0)
     */
    /** output:
     threadgroup: [M32, N32] -> [M32, N4, N8]
     index: [mcl, ncl, N8]
     each thread: N8
     layout: [B0, M, B1, N] -> [B0, M/32, M32, B1, N/32, N4, N8]
     index : [B0, sl, mcl, B1, hm, ncl, N2, N4]
     offset: ((b * q_seq_len + (sl * 32 + mcl)) * head_num + hn) * N/4 + (hm * 4 + ncl) * 2 + 0
     */

    int group = param.group;
    int q_seq_len = param.query_seq_len;
    int q_seq_piece_len = param.q_seq_piece_len;
    int value_seq_len = param.key_seq_len;
    int align_value_len = ((value_seq_len + param.kv_align_len - 1) / param.kv_align_len) * param.kv_align_len;

    int head_num = param.head_num;
    int head_dim = param.head_dim;
    int b = z / head_num;
    int hn = z % head_num;
    int zin = b * (head_num / group) + hn / group;

    int idx_qk_sl = sl * 32 + ml < q_seq_piece_len ? (sl * 32 + ml) : q_seq_piece_len - 1;

    auto A_offset = input0 + (z * q_seq_piece_len + idx_qk_sl) * align_value_len + (0 * 4 + kl) * 8 + 0;
    auto B_offset = past_value + (zin * head_dim + hm * 32 + nl) * param.max_kv_len / 4 + (0 * 4 + kvl) * 2 + 0;
    

    for(int i = 0; i < (value_seq_len+3)/4; i += 8){
        ((threadgroup ftype*)sdata)[(ml * 4 + kl) * 8 + 0] = A_offset[4*i + 0];
        ((threadgroup ftype*)sdata)[(ml * 4 + kl) * 8 + 1] = A_offset[4*i + 1];
        ((threadgroup ftype*)sdata)[(ml * 4 + kl) * 8 + 2] = A_offset[4*i + 2];
        ((threadgroup ftype*)sdata)[(ml * 4 + kl) * 8 + 3] = A_offset[4*i + 3];
        ((threadgroup ftype*)sdata)[(ml * 4 + kl) * 8 + 4] = A_offset[4*i + 4];
        ((threadgroup ftype*)sdata)[(ml * 4 + kl) * 8 + 5] = A_offset[4*i + 5];
        ((threadgroup ftype*)sdata)[(ml * 4 + kl) * 8 + 6] = A_offset[4*i + 6];
        ((threadgroup ftype*)sdata)[(ml * 4 + kl) * 8 + 7] = A_offset[4*i + 7];

        ((threadgroup ftype4*)sdata)[256 + (nl * 4 + kvl) * 2 + 0] = B_offset[i + 0];
        ((threadgroup ftype4*)sdata)[256 + (nl * 4 + kvl) * 2 + 1] = B_offset[i + 1];

        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto sA = tA.slice(0, 0);
        auto sB = tB.slice(0, 0);

        mmOps.run(sA, sB, cT);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    auto tC = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>((threadgroup float*)sdata, dextents<int32_t, 2>(N, M)); // [M , N]
    cT.store(tC);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // [M32, N4, N2, n4]
    auto sindex_base = (mcl * 4 + ncl) * 2 + 0;

    // [M32, N4, N8]
    // [mBatch, mSeqLen, mNumHead, mHeadDim]
    auto xy_out = output + ((b * q_seq_len + seq_idx * q_seq_piece_len + sl * 32 + mcl) * head_num + hn) * head_dim/4 + (hm * 4 + ncl) * 2 + 0;
    if(sl * 32 + mcl < q_seq_piece_len && seq_idx * q_seq_piece_len + sl * 32 + mcl < q_seq_len) {
        if((hm * 4 + ncl) * 2 + 0 < head_dim/4) {
            xy_out[0] =  ftype4(((threadgroup float4*)sdata)[sindex_base + 0]);
        }
        if((hm * 4 + ncl) * 2 + 1 < head_dim/4) {
            xy_out[1] =  ftype4(((threadgroup float4*)sdata)[sindex_base + 1]);
        }
    }

}
#endif

#define SIMD_GROUP_WIDTH 32
kernel void prefill_qkv(const device ftype* input0 [[buffer(0)]],
    device ftype* output [[buffer(1)]],
    device ftype* past_value [[buffer(2)]],
    constant int &seq_idx [[buffer(3)]],
    constant Param& param [[buffer(4)]],
#ifdef SIMD_GROUP_MATRIX
    uint3 gid[[threadgroup_position_in_grid]],
    uint tiitg[[thread_index_in_threadgroup]],
    uint tiisg[[thread_index_in_simdgroup]],
    uint sgitg[[simdgroup_index_in_threadgroup]]
#else
    uint3 gid[[thread_position_in_grid]]
#endif
) {
#ifdef SIMD_GROUP_MATRIX
    /*
     // Read:
     ftype 0~127   ---> input: [M16, K8]
     ftype 128~255 ---> input: [K8, N16]
     // Write:
     ftype 0~255 ---> input: [N2, M2, M8, N8]
     */

    threadgroup float sdata[256] = {0.f};

#ifdef USE_METAL_TENSOR_OPS

    const int K = 8, M = 16, N = 16; 
    auto tA = tensor<threadgroup ftype, dextents<int32_t, 2>, tensor_inline>((threadgroup ftype*)sdata, dextents<int32_t, 2>(K, M));//[M, K]
    auto tB = tensor<threadgroup ftype, dextents<int32_t, 2>, tensor_inline>((threadgroup ftype*)sdata + 128, dextents<int32_t, 2>(N, K));//[K, N]

    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(M, N, K, false, false, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<1>> mmOps;

    auto cT = mmOps.get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();
#else
    simdgroup_T8x8 sga[2];
    simdgroup_T8x8 sgb[2];
    simdgroup_float8x8 sgd[4];
    for (int i = 0; i < 4; i++){
        sgd[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }
#endif

    int rcl = tiitg / 2;// 0~15
    int kl = tiitg % 2;// 0~1

    int nl = tiitg / 8;// 0~3
    int kcl = tiitg % 8;// 0~7

    const int sl = gid.x; // q_seq_len/16 -> M/16
    const int hm = gid.y; // head_dim/16 -> N/16
    const int z = gid.z; // head_num * batch

    /** QK:
     threadgroup: [M16, K8]
     each thread: K4
     layout: [B, M, K] -> [B, M/16, M16, K/8, K2, K4]
     index : [z, sl, rcl, ml, kl, K4]
     offset: (z * M + sl * 16 + rcl) * K + (0 * 2 + kl) * 4 + 0
     */
    /** V:
     threadgroup: [K8, N16]
     each thread: N4
     layout: [B/G, K, N] -> [B/G, K/8, K8, N/16, N4, N4]
     index : [0, kcl, B/G, hm, nl, 0]
     offset: ((0 * 8 + kcl) * B/G + z/G) * N + hm * 16 + nl * 4 + 0
     */
    /** output:
     threadgroup: [M16, N16]
     each thread: N8
     layout: [M, B, N] -> [M/16, M16, B, N/16, N2, N8]
     index : [sl, rcl, B, kl, 0]
     offset: ((sl * 16 + rcl) * B + z) * N + hm * 16 + kl * 8 + 0
     */

    int group = param.group;
    int q_seq_len = param.query_seq_len;
    int q_seq_piece_len = param.q_seq_piece_len;
    int value_seq_len = param.key_seq_len;
    int align_value_len = ((value_seq_len + param.kv_align_len - 1) / param.kv_align_len) * param.kv_align_len;
    int head_num = param.head_num;
    int head_dim = param.head_dim;
    int b = z / head_num;
    int hn = z % head_num;
    int zin = b * (head_num / group) + hn / group;

    int idx_qk_sl = sl * 16 + rcl < q_seq_piece_len ? (sl * 16 + rcl) : q_seq_piece_len - 1;

    auto A_offset = input0 + (z * q_seq_piece_len + idx_qk_sl) * align_value_len + (0 * 2 + kl) * 4 + 0;
    auto B_offset = past_value + (zin * head_dim + hm * 16 + nl * 4 + 0) * param.max_kv_len + (0 * 8 + kcl);
    

    for(int i = 0; i < value_seq_len; i += 8){
        ((threadgroup ftype*)sdata)[rcl * 8 + kl * 4 + 0] = A_offset[i + 0];
        ((threadgroup ftype*)sdata)[rcl * 8 + kl * 4 + 1] = A_offset[i + 1];
        ((threadgroup ftype*)sdata)[rcl * 8 + kl * 4 + 2] = A_offset[i + 2];
        ((threadgroup ftype*)sdata)[rcl * 8 + kl * 4 + 3] = A_offset[i + 3];
        
        ((threadgroup ftype*)sdata)[128 + kcl * 16 + nl * 4 + 0] = B_offset[i + 0 * param.max_kv_len];
        ((threadgroup ftype*)sdata)[128 + kcl * 16 + nl * 4 + 1] = B_offset[i + 1 * param.max_kv_len];
        ((threadgroup ftype*)sdata)[128 + kcl * 16 + nl * 4 + 2] = B_offset[i + 2 * param.max_kv_len];
        ((threadgroup ftype*)sdata)[128 + kcl * 16 + nl * 4 + 3] = B_offset[i + 3 * param.max_kv_len];

        threadgroup_barrier(mem_flags::mem_threadgroup);

#ifdef USE_METAL_TENSOR_OPS
        auto sA = tA.slice(0, 0);
        auto sB = tB.slice(0, 0);

        mmOps.run(sA, sB, cT);
#else
        simdgroup_load(sga[0], (const threadgroup ftype*)sdata, 8);
        simdgroup_load(sga[1], ((const threadgroup ftype*)sdata) + 64, 8);
        
        simdgroup_load(sgb[0], ((const threadgroup ftype*)sdata) + 128, 16);
        simdgroup_load(sgb[1], ((const threadgroup ftype*)sdata) + 136, 16);
        
        simdgroup_multiply_accumulate(sgd[0], sga[0], sgb[0], sgd[0]);
        simdgroup_multiply_accumulate(sgd[1], sga[1], sgb[0], sgd[1]);
        simdgroup_multiply_accumulate(sgd[2], sga[0], sgb[1], sgd[2]);
        simdgroup_multiply_accumulate(sgd[3], sga[1], sgb[1], sgd[3]);
#endif
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

#ifdef USE_METAL_TENSOR_OPS

    auto tC = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>((threadgroup float*)sdata, dextents<int32_t, 2>(N, M)); // [M , N]
    cT.store(tC);
#else
    simdgroup_store(sgd[0], (threadgroup float*)sdata, 8);
    simdgroup_store(sgd[1], (threadgroup float*)sdata + 64, 8);
    simdgroup_store(sgd[2], (threadgroup float*)sdata + 128, 8);
    simdgroup_store(sgd[3], (threadgroup float*)sdata + 192, 8);
#endif

    threadgroup_barrier(mem_flags::mem_threadgroup);

#ifdef USE_METAL_TENSOR_OPS
    // [M16, N2, N8]
    auto sindex_base = (rcl * 2 + kl) * 8 + 0;
#else
    // [N2, M2, M8, N8]
    auto sindex_base = (kl * 16 + rcl) * 8 + 0;
#endif

    // [N2, M2, M8, N8]
    // [mBatch, mSeqLen, mNumHead, mHeadDim]
    auto xy_out = output + ((b * q_seq_len + seq_idx * q_seq_piece_len + sl * 16 + rcl) * head_num + hn) * head_dim + hm * 16 + kl * 8 + 0;
    if(sl * 16 + rcl < q_seq_piece_len && seq_idx * q_seq_piece_len + sl * 16 + rcl < q_seq_len) {
        if(hm * 16 + kl * 8 + 0 < head_dim) {
            xy_out[0] =  ((threadgroup float*)sdata)[sindex_base + 0];
        }
        if(hm * 16 + kl * 8 + 1 < head_dim) {
            xy_out[1] =  ((threadgroup float*)sdata)[sindex_base + 1];
        }
        if(hm * 16 + kl * 8 + 2 < head_dim) {
            xy_out[2] =  ((threadgroup float*)sdata)[sindex_base + 2];
        }
        if(hm * 16 + kl * 8 + 3 < head_dim) {
            xy_out[3] =  ((threadgroup float*)sdata)[sindex_base + 3];
        }
        if(hm * 16 + kl * 8 + 4 < head_dim) {
            xy_out[4] =  ((threadgroup float*)sdata)[sindex_base + 4];
        }
        if(hm * 16 + kl * 8 + 5 < head_dim) {
            xy_out[5] =  ((threadgroup float*)sdata)[sindex_base + 5];
        }
        if(hm * 16 + kl * 8 + 6 < head_dim) {
            xy_out[6] =  ((threadgroup float*)sdata)[sindex_base + 6];
        }
        if(hm * 16 + kl * 8 + 7 < head_dim) {
            xy_out[7] =  ((threadgroup float*)sdata)[sindex_base + 7];
        }
    }

#else
    const int x = gid.x; // q_seq_len
    const int y = gid.y; // head_num * batch
    const int z = gid.z; // head_dim
    int q_idx = seq_idx * param.q_seq_piece_len + x;
    if (x >= param.q_seq_piece_len || q_idx >= param.query_seq_len || y >= param.head_num * param.batch || z >= param.head_dim) {
        return;
    }
    int group = param.group;
    int q_seq_len = param.query_seq_len;
    int q_seq_piece_len = param.q_seq_piece_len;
    int value_seq_len = param.key_seq_len;
    int head_num = param.head_num;
    int head_dim = param.head_dim;
    int align_value_len = ((value_seq_len + param.kv_align_len - 1) / param.kv_align_len) * param.kv_align_len;

    int b = y / head_num;
    int hn = y % head_num;

    int yin = b * (head_num / group) + hn / group;

    const int stride = head_num * head_dim / group;
    const int offset_head = yin * head_dim + z;

    // [mBatch, mNumHead, mSeqLen, mKvSeqLen]
    device const ftype *A_offset = input0 + (y * q_seq_piece_len + x) * align_value_len;
    device const ftype *B_offset = past_value + offset_head * param.max_kv_len;
    float out = 0.0;
    
    for(int i = 0; i < value_seq_len; ++i){
        float A0 = (float)A_offset[i];
        float B = (float)B_offset[i];
        out += A0 * B;
    }
    // [mBatch, mSeqLen, mNumHead, mHeadDim]
    output[(b * q_seq_len + q_idx) * stride * group + (hn * head_dim + z)] = out;
#endif
}

kernel void decode_qkv(const device ftype* input0 [[buffer(0)]],
    device ftype* output [[buffer(1)]],
    device ftype* past_value [[buffer(2)]],
    // docode actually not compute in block
    constant int &seq_idx [[buffer(3)]],
    constant Param& param [[buffer(4)]],
#ifdef SIMD_GROUP_REDUCE
    uint3 gid[[threadgroup_position_in_grid]],
    uint  tiisg[[thread_index_in_simdgroup]],
    uint  sgitg[[simdgroup_index_in_threadgroup]]
#else
    uint3 gid[[thread_position_in_grid]]
#endif
) {
    const int x = gid.x; // query_seq_len
    const int y = gid.y; // head_num * batch
    const int z = gid.z; // head_dim
    if (x >= param.query_seq_len || y >= param.head_num * param.batch || z >= param.head_dim) {
        return;
    }
    int head_dim = param.head_dim;
    int head_num = param.head_num;
    int q_seq_len = param.query_seq_len;
    int group = param.group;
    int b = y / head_num;
    int hn = y % head_num;

    int yin = b * (head_num / group) + hn / group;
    int value_seq_len = param.key_seq_len;
    int align_value_len = ((value_seq_len + param.kv_align_len - 1) / param.kv_align_len) * param.kv_align_len;

    const int offset_head = (yin * head_dim + z) * param.max_kv_len;

    device const ftype *A_offset = input0 + (y * q_seq_len + x) * align_value_len;
    device ftype *Pastvalue_offset = past_value + offset_head;
    float out = 0;
    
#ifdef SIMD_GROUP_REDUCE
    for(int i = tiisg; i < value_seq_len; i+=SIMD_GROUP_WIDTH){
        float A = (float)A_offset[i];
        float B = (float)Pastvalue_offset[i];
        
        out += A * B;
    }
    out = simd_sum(out);
    if(tiisg == 0) {
        // [mBatch, mSeqLen, mNumHead, mHeadDim]
        output[((b * q_seq_len + x) * head_num + hn) * head_dim + z] = (ftype)out;
    }
#else
    for(int i = 0; i < value_seq_len; i++){
        float A = (float)A_offset[i];
        float B = (float)Pastvalue_offset[i];
        
        out += A * B;
    }
    output[((b * q_seq_len + x) * head_num + hn) * head_dim + z] = (ftype)out;
#endif
}
)metal";

const char* gSoftmaxSgReduce = R"metal(
#include <metal_stdlib>
using namespace metal;
struct softmax_shape {
    int inside_size;
    int axis_length;
    int outside_size;
    int axis_align_length;
};
#define SIMD_GROUP_WIDTH 32

kernel void softmax_plane_sg(const device ftype *in     [[buffer(0)]],
                        device ftype *out          [[buffer(1)]],
                        constant softmax_shape& s   [[buffer(2)]],
                        uint2 gid[[threadgroup_position_in_grid]],
                        uint  tiisg[[thread_index_in_simdgroup]],
                        uint  sgitg[[simdgroup_index_in_threadgroup]]
    ) {
    // threadgroup contain one simdgroup
    // simdgroup compute axis data
    if ((int)gid.x >= s.inside_size || (int)gid.y >= s.outside_size) return;
    
    auto in_offset = gid.y * s.axis_length * s.inside_size + gid.x;
    auto out_offset = gid.y * s.axis_align_length * s.inside_size + gid.x;
    auto axis_in  = in + in_offset;
    auto axis_out = out + out_offset;
    
    // get max
    float max1 = -FLT_MAX;
    for (int i = tiisg; i < s.axis_length; i+=SIMD_GROUP_WIDTH) {
        max1 = max(max1, float(axis_in[i * s.inside_size]));
    }
    max1 = simd_max(max1);

    // get sum
    float sum1 = 0;
    for (int i = tiisg; i < s.axis_length; i+=SIMD_GROUP_WIDTH) {
        sum1 += exp(float(axis_in[i * s.inside_size]) - float(max1));
    }
    sum1 = simd_sum(sum1);

    // output
    for (int i = tiisg; i < s.axis_align_length; i+=SIMD_GROUP_WIDTH) {
        axis_out[i * s.inside_size] = i >= s.axis_length ? ftype(0.0) : ftype(exp(float(axis_in[i * s.inside_size]) - float(max1)) / sum1);
    }
}


)metal";

const char* gFlashSoftmax = R"metal(
#include <metal_stdlib>
using namespace metal;

struct Param {
    int query_seq_len;
    int q_seq_piece_len;
    int key_seq_len;
    int head_num;
    int group;
    int head_dim;
    float scale;
    int max_kv_len;
    int batch;
    int kv_align_len;
};

kernel void flash_softmax(
    const device ftype* input [[buffer(0)]],
    device ftype* output [[buffer(1)]],
    device float* runningStats [[buffer(2)]],
    device float* correctionScale [[buffer(3)]],
    constant int& block_len [[buffer(4)]],
    constant Param& param [[buffer(5)]],
    constant int& kv_start [[buffer(6)]],
#ifdef SIMD_GROUP_REDUCE
    uint2 gid [[threadgroup_position_in_grid]],
    uint  tiisg [[thread_index_in_simdgroup]]
#else
    uint3 gid [[thread_position_in_grid]]
#endif
) {
#ifdef SIMD_GROUP_REDUCE
    int s = gid.x;
    int bh = gid.y;
#else
    int s = gid.x;
    int bh = gid.y;
#endif
    
    if (s >= param.query_seq_len || bh >= param.batch * param.head_num) {
        return;
    }
    
    int seq_len = param.query_seq_len;
    int stat_idx = (bh * seq_len + s) * 2;
    int block_offset = (bh * seq_len + s) * block_len;
    
    float prev_max = (float)runningStats[stat_idx];
    float prev_sum = (float)runningStats[stat_idx + 1];
    
    float safe_min = -10000.0;
    if (kv_start == 0) {
        prev_max = safe_min;
        prev_sum = 0;
    }
    
    float block_max = safe_min;
#ifdef SIMD_GROUP_REDUCE
    for (int i = tiisg; i < block_len; i += 32) {
        block_max = max(block_max, float(input[block_offset + i]));
    }
    block_max = simd_max(block_max);
#else
    for (int i = 0; i < block_len; ++i) {
        block_max = max(block_max, float(input[block_offset + i]));
    }
#endif
    
    float new_max = max(prev_max, block_max);
    float scale = exp(prev_max - new_max);
    
    float block_sum = 0;
#ifdef SIMD_GROUP_REDUCE
    for (int i = tiisg; i < block_len; i += 32) {
        float val = exp(float(input[block_offset + i]) - new_max);
        output[block_offset + i] = (ftype)val;
        block_sum += val;
    }
    block_sum = simd_sum(block_sum);
#else
    for (int i = 0; i < block_len; ++i) {
        float val = exp(float(input[block_offset + i]) - new_max);
        output[block_offset + i] = (ftype)val;
        block_sum += val;
    }
#endif
    
    float new_sum = prev_sum * scale + block_sum;
    
#ifdef SIMD_GROUP_REDUCE
    if (tiisg == 0) {
#endif
    runningStats[stat_idx] = (float)new_max;
    runningStats[stat_idx + 1] = (float)new_sum;
    correctionScale[bh * seq_len + s] = (float)scale;
#ifdef SIMD_GROUP_REDUCE
    }
#endif
}
)metal";

const char* gFlashMatMulQKV = R"metal(
#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;
struct Param {
    int query_seq_len;
    int q_seq_piece_len;
    int key_seq_len;
    int head_num;
    int group;
    int head_dim;
    float scale;
    int max_kv_len;
    int batch;
    int kv_align_len;
};

#if MNN_METAL_FLOAT16_STORAGE
typedef simdgroup_half8x8 simdgroup_T8x8;
#else
typedef simdgroup_float8x8 simdgroup_T8x8;
#endif


kernel void flash_matmul_qkv(
    const device ftype* P_block [[buffer(0)]],
    device float* Output [[buffer(1)]],
    const device ftype* V_block [[buffer(2)]],
    const device float* correctionScale [[buffer(3)]],
    constant int& kv_start [[buffer(4)]],
    constant int& block_len [[buffer(5)]],
    constant Param& param [[buffer(6)]],
#if defined(SIMD_GROUP_MATRIX)
    uint3 gid[[threadgroup_position_in_grid]],
    uint tiitg[[thread_index_in_threadgroup]],
    uint tiisg[[thread_index_in_simdgroup]],
    uint sgitg[[simdgroup_index_in_threadgroup]]
#elif defined(SIMD_GROUP_REDUCE)
    uint3 gid[[threadgroup_position_in_grid]],
    uint tiisg[[thread_index_in_simdgroup]]
#else
    uint3 gid [[thread_position_in_grid]]
#endif
) {
#if defined(SIMD_GROUP_MATRIX)
    threadgroup float sdata[256 + 128] = {0.f}; // 128 for A, 128 for B, 128 for C scaling
    simdgroup_float8x8 sgd[4];
    for (int i = 0; i < 4; i++){
        sgd[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }
    
    const int sl = gid.x; // s / 16
    const int hm = gid.y; // d / 16
    const int bh = gid.z;
    
    int b = bh / param.head_num;
    int h = bh % param.head_num;
    int kv_h = h / param.group;
    int yin = b * (param.head_num / param.group) + kv_h;
    
    int rcl = tiitg / 2; // 0~15
    int kl = tiitg % 2;  // 0~1
    int nl = tiitg / 8;  // 0~3
    int kcl = tiitg % 8; // 0~7
    
    int head_dim = param.head_dim;
    int q_seq_len = param.query_seq_len;
    
    if (sl * 16 >= q_seq_len || hm * 16 >= head_dim) return;

    // 0. Load old Output and scale
    if (kv_start > 0) {
        for (int i = 0; i < 8; ++i) {
            int idx = tiitg * 8 + i;
            int local_s = idx / 16;
            int local_d = idx % 16;
            // Map threads to 16x16 block of Output
            int cur_s = sl * 16 + local_s;
            int cur_d = hm * 16 + local_d;
            if (cur_s < q_seq_len && cur_d < head_dim) {
                float scale = correctionScale[bh * q_seq_len + cur_s];
                int out_idx = ((b * q_seq_len + cur_s) * param.head_num + h) * head_dim + cur_d;
                float val = Output[out_idx] * scale;
                // Store to sdata for loading into sgd
                ((threadgroup float*)sdata)[local_s * 16 + local_d] = val;
            } else {
                ((threadgroup float*)sdata)[local_s * 16 + local_d] = 0.f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_load(sgd[0], (threadgroup float*)sdata, 16);
        simdgroup_load(sgd[1], (threadgroup float*)sdata + 128, 16);
        simdgroup_load(sgd[2], (threadgroup float*)sdata + 8, 16);
        simdgroup_load(sgd[3], (threadgroup float*)sdata + 136, 16);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    auto A_offset = P_block + (bh * q_seq_len + sl * 16 + rcl) * block_len + (0 * 2 + kl) * 4 + 0;
    auto B_offset = V_block + (yin * head_dim + hm * 16 + nl * 4 + 0) * param.max_kv_len + (kv_start + (0 * 8 + kcl));

    for (int i = 0; i < block_len; i += 8) {
        ((threadgroup ftype*)sdata)[rcl * 8 + kl * 4 + 0] = A_offset[i + 0];
        ((threadgroup ftype*)sdata)[rcl * 8 + kl * 4 + 1] = A_offset[i + 1];
        ((threadgroup ftype*)sdata)[rcl * 8 + kl * 4 + 2] = A_offset[i + 2];
        ((threadgroup ftype*)sdata)[rcl * 8 + kl * 4 + 3] = A_offset[i + 3];
        
        ((threadgroup ftype*)sdata)[128 + kcl * 16 + nl * 4 + 0] = B_offset[i + 0 * param.max_kv_len];
        ((threadgroup ftype*)sdata)[128 + kcl * 16 + nl * 4 + 1] = B_offset[i + 1 * param.max_kv_len];
        ((threadgroup ftype*)sdata)[128 + kcl * 16 + nl * 4 + 2] = B_offset[i + 2 * param.max_kv_len];
        ((threadgroup ftype*)sdata)[128 + kcl * 16 + nl * 4 + 3] = B_offset[i + 3 * param.max_kv_len];

        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_T8x8 sga[2], sgb[2];
        simdgroup_load(sga[0], (const threadgroup ftype*)sdata, 8);
        simdgroup_load(sga[1], ((const threadgroup ftype*)sdata) + 64, 8);
        simdgroup_load(sgb[0], ((const threadgroup ftype*)sdata) + 128, 16);
        simdgroup_load(sgb[1], ((const threadgroup ftype*)sdata) + 136, 16);
        
        simdgroup_multiply_accumulate(sgd[0], sga[0], sgb[0], sgd[0]);
        simdgroup_multiply_accumulate(sgd[1], sga[1], sgb[0], sgd[1]);
        simdgroup_multiply_accumulate(sgd[2], sga[0], sgb[1], sgd[2]);
        simdgroup_multiply_accumulate(sgd[3], sga[1], sgb[1], sgd[3]);
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_store(sgd[0], (threadgroup float*)sdata, 16);
    simdgroup_store(sgd[1], (threadgroup float*)sdata + 128, 16);
    simdgroup_store(sgd[2], (threadgroup float*)sdata + 8, 16);
    simdgroup_store(sgd[3], (threadgroup float*)sdata + 136, 16);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int i = 0; i < 8; ++i) {
        int idx = tiitg * 8 + i;
        int local_s = idx / 16;
        int local_d = idx % 16;
        int cur_s = sl * 16 + local_s;
        int cur_d = hm * 16 + local_d;
        if (cur_s < q_seq_len && cur_d < head_dim) {
            int out_idx = ((b * q_seq_len + cur_s) * param.head_num + h) * head_dim + cur_d;
            Output[out_idx] = ((threadgroup float*)sdata)[local_s * 16 + local_d];
        }
    }

#elif defined(SIMD_GROUP_REDUCE)
    int d_vec = gid.x;
    int s = gid.y;
    int bh = gid.z;
    
    int head_dim = param.head_dim;
    if (d_vec * 4 >= head_dim || s >= param.query_seq_len || bh >= param.batch * param.head_num) return;
    
    int b = bh / param.head_num;
    int h = bh % param.head_num;
    int kv_h = h / param.group;
    int yin = b * (param.head_num / param.group) + kv_h;
    int v_base_offset = yin * head_dim * param.max_kv_len;
    
    int p_offset = (bh * param.query_seq_len + s) * block_len;
    int out_idx = ((b * param.query_seq_len + s) * param.head_num + h) * head_dim + d_vec * 4;
    
    float4 acc = 0;
    if (kv_start > 0 && tiisg == 0) {
        acc = float4(Output[out_idx], Output[out_idx+1], Output[out_idx+2], Output[out_idx+3]);
        acc *= correctionScale[bh * param.query_seq_len + s];
    }
    
    float4 p_v_acc = 0;
    for (int k = tiisg; k < block_len; k += 32) {
        ftype p_val = P_block[p_offset + k];
        int seq_idx = kv_start + k;
        float4 v;
        v.x = (float)V_block[v_base_offset + (d_vec * 4 + 0) * param.max_kv_len + seq_idx];
        v.y = (float)V_block[v_base_offset + (d_vec * 4 + 1) * param.max_kv_len + seq_idx];
        v.z = (float)V_block[v_base_offset + (d_vec * 4 + 2) * param.max_kv_len + seq_idx];
        v.w = (float)V_block[v_base_offset + (d_vec * 4 + 3) * param.max_kv_len + seq_idx];
        p_v_acc += (float)p_val * v;
    }
    p_v_acc.x = simd_sum(p_v_acc.x);
    p_v_acc.y = simd_sum(p_v_acc.y);
    p_v_acc.z = simd_sum(p_v_acc.z);
    p_v_acc.w = simd_sum(p_v_acc.w);
    
    if (tiisg == 0) {
        acc += p_v_acc;
        Output[out_idx]   = acc.x;
        Output[out_idx+1] = acc.y;
        Output[out_idx+2] = acc.z;
        Output[out_idx+3] = acc.w;
    }
#else
    int d_vec = gid.x;
    int s = gid.y;
    int bh = gid.z;
    
    int head_dim = param.head_dim;
    if (d_vec * 4 >= head_dim || s >= param.query_seq_len || bh >= param.batch * param.head_num) {
        return;
    }
    
    int b = bh / param.head_num;
    int h = bh % param.head_num;
    int kv_h = h / param.group;
    
    int p_offset = (bh * param.query_seq_len + s) * block_len;
    // V layout: [batch, kv_num_head * head_dim, max_kv_len]
    // Same as decode_qkv: offset_head = (yin * head_dim + z) * max_kv_len, where yin = b * kv_num_head + kv_h
    // So for (batch, kv_head, head_dim_idx), base offset = (b * kv_num_head + kv_h) * head_dim * max_kv_len
    int yin = b * (param.head_num / param.group) + kv_h;
    int v_base_offset = yin * head_dim * param.max_kv_len;
    
    int out_idx = ((b * param.query_seq_len + s) * param.head_num + h) * head_dim + d_vec * 4;
    float scale = (float)correctionScale[bh * param.query_seq_len + s];
    
    float4 acc = 0;
    if (kv_start > 0) {
        acc = float4(Output[out_idx], Output[out_idx+1], Output[out_idx+2], Output[out_idx+3]);
        acc *= (float)scale;
    }
    
    for (int k = 0; k < block_len; ++k) {
        ftype p_val = P_block[p_offset + k];
        int seq_idx = kv_start + k;
        
        // V layout: [batch, kv_num_head * head_dim, max_kv_len]
        // For (batch, kv_head, head_dim_idx, seq_idx): offset = ((b * kv_num_head + kv_h) * head_dim + head_dim_idx) * max_kv_len + seq_idx
        float v0 = 0, v1 = 0, v2 = 0, v3 = 0;
        int d0 = d_vec * 4 + 0;
        int d1 = d_vec * 4 + 1;
        int d2 = d_vec * 4 + 2;
        int d3 = d_vec * 4 + 3;
        
        if (d0 < head_dim) {
            int v_idx0 = v_base_offset + d0 * param.max_kv_len + seq_idx;
            v0 = (float)V_block[v_idx0];
        }
        if (d1 < head_dim) {
            int v_idx1 = v_base_offset + d1 * param.max_kv_len + seq_idx;
            v1 = (float)V_block[v_idx1];
        }
        if (d2 < head_dim) {
            int v_idx2 = v_base_offset + d2 * param.max_kv_len + seq_idx;
            v2 = (float)V_block[v_idx2];
        }
        if (d3 < head_dim) {
            int v_idx3 = v_base_offset + d3 * param.max_kv_len + seq_idx;
            v3 = (float)V_block[v_idx3];
        }
        
        acc += (float)p_val * float4(v0, v1, v2, v3);
    }
    
    Output[out_idx]   = (float)acc.x;
    Output[out_idx+1] = (float)acc.y;
    Output[out_idx+2] = (float)acc.z;
    Output[out_idx+3] = (float)acc.w;
#endif
}
)metal";

const char* gFlashScale = R"metal(
#include <metal_stdlib>
using namespace metal;

struct Param {
    int query_seq_len;
    int q_seq_piece_len;
    int key_seq_len;
    int head_num;
    int group;
    int head_dim;
    float scale;
    int max_kv_len;
    int batch;
    int kv_align_len;
};

kernel void flash_scale(
    const device float* Input [[buffer(0)]],
    device ftype* Output [[buffer(1)]],
    const device float* runningStats [[buffer(2)]],
    constant Param& param [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]
) {
    int d_vec = gid.x;
    int s = gid.y;
    int bh = gid.z;
    
    if (d_vec * 4 >= param.head_dim || s >= param.query_seq_len || bh >= param.batch * param.head_num) {
        return;
    }
    
    int stat_idx = (bh * param.query_seq_len + s) * 2;
    float sum = (float)runningStats[stat_idx + 1];
    float inv_sum = 1.0 / sum;
    
    int b = bh / param.head_num;
    int h = bh % param.head_num;
    
    int out_idx = ((b * param.query_seq_len + s) * param.head_num + h) * param.head_dim + d_vec * 4;
    
    Output[out_idx]   = (ftype)(inv_sum * (float)Input[out_idx]  );
    Output[out_idx+1] = (ftype)(inv_sum * (float)Input[out_idx+1]);
    Output[out_idx+2] = (ftype)(inv_sum * (float)Input[out_idx+2]);
    Output[out_idx+3] = (ftype)(inv_sum * (float)Input[out_idx+3]);
}
)metal";


const char* gFlashAttentionFused = R"metal(
#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;

struct Param {
    int query_seq_len;
    int q_seq_piece_len;
    int key_seq_len;
    int head_num;
    int group;
    int head_dim;
    float scale;
    int max_kv_len;
    int batch;
    int kv_align_len;
};

#if MNN_METAL_FLOAT16_STORAGE
typedef simdgroup_half8x8 simdgroup_T8x8;
#else
typedef simdgroup_float8x8 simdgroup_T8x8;
#endif

// 定义支持缓存的最大 Head Dim
#define MAX_HEAD_DIM 128
// Padding 步长，避免 Shared Memory Bank Conflict
// 128 + 8 = 136，错开 Bank 索引
#define Q_SMEM_STRIDE (MAX_HEAD_DIM + 8)

// 优化配置
#define Q_BLOCK 8
#define K_BLOCK_16 16
#define TG_SIZE 128
#define SIMD_GROUPS 4
#define K_BLOCK 64



#define HEAD_DIM 128
typedef uint4 vec_128b; 


// 调整为 5120 (20KB)，M4 缓存充足
#define SMEM_SIZE 4096 

kernel void flash_attention_fused(
    const device ftype* query [[buffer(0)]],
    const device ftype* key [[buffer(1)]],
    const device ftype* value [[buffer(2)]],
    const device ftype* mask [[buffer(3)]],
    device ftype* output [[buffer(4)]],
    constant Param& param [[buffer(5)]],
    uint ltid [[thread_index_in_threadgroup]],      // 0..127 global inside group
#if defined(SIMD_GROUP_REDUCE)
    uint3 gid [[thread_position_in_grid]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]]
#else
    uint3 gid [[thread_position_in_grid]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]]
#endif
) {
#ifdef SIMD_GROUP_MATRIX

    // 8个query threadgroup=128 K_BLOCK=16
    // Shared Memory 布局
    threadgroup ftype sdata[4096]; 
    threadgroup ftype* sdata_q = sdata;
    threadgroup float* sdata_work = (threadgroup float*)(sdata + Q_BLOCK * Q_SMEM_STRIDE);

    threadgroup float* sdata_partials = sdata_work + 128; 
    threadgroup float* sdata_scale = sdata_work + 128;    
    threadgroup float* sdata_final_sum = sdata_work + 136; 
    threadgroup float* sdata_scratch = sdata_work + 512;  

    int sl_blk = tgid.x; 
    int bh = tgid.y;
    int head_dim = param.head_dim;
    int q_seq_len = param.query_seq_len;

    if (sl_blk * Q_BLOCK >= q_seq_len) return;

    int b = bh / param.head_num;
    int h = bh % param.head_num;
    int kv_h = h / param.group;
    int kv_len = param.key_seq_len;
    int max_kv_len = param.max_kv_len;

    // 1. 协作加载 Query
    {
        int q_base_offset = ((b * q_seq_len + sl_blk * Q_BLOCK) * param.head_num + h) * head_dim;
        const device ftype* q_ptr_base = query + q_base_offset;
        int q_global_stride = param.head_num * head_dim;
        
        for (int i = ltid; i < Q_BLOCK * head_dim; i += TG_SIZE) {
            int r = i / head_dim;
            int c = i % head_dim;
            
            // [Fix] 检查 Query 是否越界，越界部分填充 0 避免计算错误
            int global_r = sl_blk * Q_BLOCK + r;
            if (global_r < q_seq_len) {
                sdata_q[r * Q_SMEM_STRIDE + c] = q_ptr_base[r * q_global_stride + c];
            } else {
                sdata_q[r * Q_SMEM_STRIDE + c] = 0.0f;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);


    // 2. 状态初始化
    float row_max = -FLT_MAX;
    float row_sum = 0.0f;

    simdgroup_float8x8 acc_reg[2][2]; 
    for(int i=0; i<2; ++i) for(int j=0; j<2; ++j) acc_reg[i][j] = make_filled_simdgroup_matrix<float, 8>(0.f);

    int k_seq_stride = param.batch * (param.head_num / param.group) * head_dim; 
    int k_base = (b * (param.head_num / param.group) + kv_h) * head_dim;
    int v_base = (b * (param.head_num / param.group) + kv_h) * head_dim * max_kv_len;


    const int global_r = sl_blk * Q_BLOCK + Q_BLOCK - 1 + kv_len - param.query_seq_len;

    // 3. 主循环 K Block
    // ==========================================
    for (int t_blk = 0; t_blk < kv_len; t_blk += K_BLOCK_16) {
        
        bool skip_block = false;

        // Block mask all -inf
        if(t_blk > global_r) {
            skip_block = true;
        }

#if 0
        // ============================================================
        // [Optimization] Fast & Safe Mask Check (Hybrid Version)
        // ============================================================
        {
            // 1. 内存安全：将 scratch 移至 +1024，避开 partials (0~640)
            threadgroup float* sdata_flag = sdata_work + 1024;

            // 2. 每个线程计算局部最大值
            float my_max = -FLT_MAX;

            // 循环 stride 覆盖 (兼容 TG_SIZE < 128)
            for (int i = ltid; i < 128; i += TG_SIZE) {
                int r_local = i / 16; 
                int c_local = i % 16; 
                
                int global_r = sl_blk * Q_BLOCK + r_local;
                int global_c = t_blk + c_local;

                if (global_c < kv_len) {
                    #ifdef ADD_MASK
                        int mask_offset = global_c - kv_len + param.query_seq_len;
                        if (global_r < q_seq_len) {
                             // 如果 mask_offset 有效，读取 Mask
                             if (mask_offset >= 0 && mask_offset < param.query_seq_len) {
                                 float m_val = (float)mask[global_r * param.query_seq_len + mask_offset];
                                 my_max = max(my_max, m_val);
                             } else {
                                 // 越界 (通常是左侧非Mask区) = 有效 (0.0f) -> 不能跳过
                                 my_max = max(my_max, 0.0f);
                             }
                        }
                    #elif defined(SET_MASK)
                        if (global_r < q_seq_len) {
                             float m_val = (float)mask[global_r * kv_len + global_c];
                             // SET_MASK: 0 表示 Mask, 非0 表示 Keep
                             if (m_val != 0.0f) my_max = max(my_max, 0.0f);
                        }
                    #else
                        my_max = 0.0f; // 无 Mask 宏，始终 Active
                    #endif
                }
            }

            // 3. SIMD Group 内快速归约
            float sg_max = simd_max(my_max);

            // 4. 写入 Shared Memory (仅每个SG的第一个线程写)
            if (tiisg == 0) {
                sdata_flag[sgitg] = sg_max;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // 5. Thread 0 汇总 (仅需检查 4 个值)
            if (ltid == 0) {
                float block_max = -FLT_MAX;
                int num_sg = (TG_SIZE + 31) / 32; 
                
                for (int i = 0; i < num_sg; ++i) {
                    block_max = max(block_max, sdata_flag[i]);
                }

                // 阈值判定：只有全为极小值时才跳过
                // 写入 1.0f 表示跳过
                sdata_flag[0] = (block_max <= -10000.0f) ? 1.0f : 0.0f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // 6. 读取跳过标志
            if (sdata_flag[0] > 0.5f) {
                skip_block = true;
            }
        }
#endif
        // ============================================================

        // 使用 if 包裹，而非 continue，确保控制流绝对安全
        if (!skip_block) {
            
            // --- Step A: Q * K^T ---
            simdgroup_float8x8 sg_score[2]; 
            sg_score[0] = make_filled_simdgroup_matrix<float, 8>(0.f);
            sg_score[1] = make_filled_simdgroup_matrix<float, 8>(0.f);
            
            int d_start = sgitg * 32;
            int d_end = min(d_start + 32, head_dim);

            for (int d = d_start; d < d_end; d += 8) {
                simdgroup_T8x8 sgq; 
                simdgroup_T8x8 sgk[2]; 

                simdgroup_load(sgq, sdata_q + d, Q_SMEM_STRIDE, ulong2(0), false);

                const device ftype* k_curr = key + k_base + t_blk * k_seq_stride + d;
                ulong k_stride = ulong(k_seq_stride);
                
                simdgroup_barrier(mem_flags::mem_none);
                simdgroup_load(sgk[0], k_curr, k_stride, ulong2(0), true); 
                simdgroup_load(sgk[1], k_curr + 8 * k_seq_stride, k_stride, ulong2(0), true);
                simdgroup_barrier(mem_flags::mem_none);

                simdgroup_multiply_accumulate(sg_score[0], sgq, sgk[0], sg_score[0]);
                simdgroup_multiply_accumulate(sg_score[1], sgq, sgk[1], sg_score[1]);
            }

            int smem_score_offset = sgitg * 128; 
            simdgroup_store(sg_score[0], sdata_partials + smem_score_offset, 16);
            simdgroup_store(sg_score[1], sdata_partials + smem_score_offset + 8, 16);
            
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // --- Step B: Reduction ---
            if (ltid < 128) {
                float sum = 0.0f;
                #pragma unroll
                for (int i = 0; i < SIMD_GROUPS; ++i) {
                    sum += sdata_partials[i * 128 + ltid];
                }
                sdata_work[ltid] = sum;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // --- Step C: Softmax & P (SG0 Only) ---
            if (ltid < 8) { 
                float m_prev = row_max;
                float s_prev = row_sum;
                float m_curr = -FLT_MAX;
                
                for (int j=0; j<16; ++j) {
                    float val = sdata_work[ltid * 16 + j] * param.scale;
                    int ti = t_blk + j;
                    
                    #ifdef ADD_MASK
                    int mask_offset = ti - kv_len + param.query_seq_len;
                    if (ti < kv_len && mask_offset >= 0 && mask_offset < param.query_seq_len)
                        val += (float)mask[(sl_blk * Q_BLOCK + ltid) * param.query_seq_len + mask_offset];
                    else if (ti >= kv_len) val = -FLT_MAX;
                    #elif defined(SET_MASK)
                    if (ti >= kv_len || mask[(sl_blk * Q_BLOCK + ltid) * kv_len + ti] == 0) val = -FLT_MAX;
                    #endif
                    
                    sdata_work[ltid * 16 + j] = val;
                    m_curr = max(m_curr, val);
                }
                
                float m_new = max(m_prev, m_curr);
                float exp_diff = exp(m_prev - m_new);
                float s_curr = 0.0f;
                
                threadgroup ftype* sdata_p_out = (threadgroup ftype*)sdata_work;
                for (int j=0; j<16; ++j) {
                    float p = exp(sdata_work[ltid * 16 + j] - m_new);
                    sdata_p_out[ltid * 16 + j] = (ftype)p; 
                    s_curr += p;
                }
                
                row_max = m_new;
                row_sum = s_prev * exp_diff + s_curr;
                sdata_scale[ltid] = exp_diff;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // --- Step D: P * V ---
            for (int iter = 0; iter < 2; ++iter) { 
                int d_tile = sgitg * 2 + iter;
                if (d_tile * 16 >= head_dim) continue;
                
                // 注意：这里继续使用 sdata_scratch (offset 512) 没问题，
                // 因为它只在 Block 计算内部使用，不会跨 Block 影响 Skip 逻辑。
                // 且 Skip Flag 已经用完了。
                threadgroup float* my_scratch = sdata_work + 512 + sgitg * 128;
                
                simdgroup_store(acc_reg[iter][0], my_scratch, 16);     
                simdgroup_store(acc_reg[iter][1], my_scratch + 8, 16); 
                threadgroup_barrier(mem_flags::mem_threadgroup);
                
                if (tiisg < 8) {
                    float sc = sdata_scale[tiisg];
                    #pragma unroll
                    for (int j=0; j<16; ++j) my_scratch[tiisg * 16 + j] *= sc;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                
                simdgroup_load(acc_reg[iter][0], my_scratch, 16);
                simdgroup_load(acc_reg[iter][1], my_scratch + 8, 16);

                threadgroup ftype* sdata_p = (threadgroup ftype*)sdata_work;
                simdgroup_T8x8 sgp[2];
                simdgroup_load(sgp[0], sdata_p, 16, ulong2(0), false); 
                simdgroup_load(sgp[1], sdata_p + 8, 16, ulong2(0), false); 

                int d_start = d_tile * 16;
                const device ftype* v_curr = value + v_base + d_start * max_kv_len + t_blk;
                
                simdgroup_T8x8 sgv[4];
                simdgroup_barrier(mem_flags::mem_none);
                simdgroup_load(sgv[0], v_curr, max_kv_len, ulong2(0), true);
                simdgroup_load(sgv[1], v_curr + 8 * max_kv_len, max_kv_len, ulong2(0), true);
                simdgroup_load(sgv[2], v_curr + 8, max_kv_len, ulong2(0), true);
                simdgroup_load(sgv[3], v_curr + 8 * max_kv_len + 8, max_kv_len, ulong2(0), true);
                simdgroup_barrier(mem_flags::mem_none);
                
                simdgroup_multiply_accumulate(acc_reg[iter][0], sgp[0], sgv[0], acc_reg[iter][0]);
                simdgroup_multiply_accumulate(acc_reg[iter][0], sgp[1], sgv[2], acc_reg[iter][0]);
                simdgroup_multiply_accumulate(acc_reg[iter][1], sgp[0], sgv[1], acc_reg[iter][1]);
                simdgroup_multiply_accumulate(acc_reg[iter][1], sgp[1], sgv[3], acc_reg[iter][1]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        } // End of if (!skip_block)
    } // End K Loop

    // 4. Output Finalization (保持不变)
    if (ltid < 8) {
        sdata_final_sum[ltid] = row_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    threadgroup float* my_out_buf = sdata_scratch + sgitg * 128;
    
    for (int iter = 0; iter < 2; ++iter) {
        int d_tile = sgitg * 2 + iter; 
        if (d_tile * 16 >= head_dim) continue;
        
        simdgroup_store(acc_reg[iter][0], my_out_buf, 16);
        simdgroup_store(acc_reg[iter][1], my_out_buf + 8, 16);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        
        if (tiisg < 8) {
            float inv_sum = 1.0f / sdata_final_sum[tiisg];
            int qi = sl_blk * Q_BLOCK + tiisg;
            if (qi < q_seq_len) {
                device ftype* out_ptr = output + ((b * q_seq_len + qi) * param.head_num + h) * head_dim + d_tile * 16;
                #pragma unroll
                for (int j=0; j<16; ++j) {
                    if (d_tile * 16 + j < head_dim) {
                        out_ptr[j] = (ftype)(my_out_buf[tiisg * 16 + j] * inv_sum);
                    }
                }
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

#else
    // ===== Optimized Basic Version: Threadgroup parallel without simd_sum =====
    // Grid: [SeqLen, Batch*Head, 1], Threadgroup: [THREADS_PER_GROUP, 1, 1]
    // Each threadgroup processes one Q token, threads cooperate on dimension reduction
    
    threadgroup float shared_reduce[256]; // For manual reduction, max 256 threads
    
    int s = tgid.x;      // query sequence position
    int bh = tgid.y;     // batch * head_num
    uint tid = tiisg;    // thread index in simdgroup (0-31)
    uint threads_per_group = sgitg * 32 + tiisg; // global thread index in threadgroup
    
    if (s >= param.query_seq_len || bh >= param.batch * param.head_num) return;
    
    int b = bh / param.head_num;
    int h = bh % param.head_num;
    int kv_h = h / param.group;
    
    int head_dim = param.head_dim;
    int kv_len = param.key_seq_len;
    int max_kv_len = param.max_kv_len;
    int group = param.group;
    
    int q_offset = ((b * param.query_seq_len + s) * param.head_num + h) * head_dim;
    
    // Each thread processes multiple dimensions
    int d_per_thread = (head_dim + 31) / 32; // Assume 32 threads per group
    
    float acc[8] = {0.0f}; 
    
    // Load Q values for this thread's dimensions
    float q_local[8] = {0.0f};
    for (int i = 0; i < d_per_thread; ++i) {
        int d = tid + i * 32;
        if (d < head_dim) {
            q_local[i] = (float)query[q_offset + d];
        }
    }
    
    float cur_max = -FLT_MAX; 
    float cur_sum = 0.0f;
    
    // K/V offsets
    int kv_head_num = param.head_num / group;
    int k_seq_stride = param.batch * kv_head_num * head_dim;
    int k_base_offset = (b * kv_head_num + kv_h) * head_dim;
    auto k_ptr_start = key + k_base_offset;
    
    int v_base_offset = (b * kv_head_num + kv_h) * head_dim * max_kv_len;

    for (int t = 0; t < kv_len; ++t) {
        auto k_ptr_t = k_ptr_start + t * k_seq_stride;
        
        // Each thread computes partial dot product for its dimensions
        float partial_dot = 0.0f;
        for (int i = 0; i < d_per_thread; ++i) {
            int d = tid + i * 32;
            if (d < head_dim) {
                partial_dot += q_local[i] * (float)k_ptr_t[d];
            }
        }
        
        // Manual reduction across threads (no simd_sum)
        shared_reduce[tid] = partial_dot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Tree reduction in threadgroup memory
        float score = 0.0f;
        if (tid == 0) {
            for (int i = 0; i < 32; ++i) {
                score += shared_reduce[i];
            }
            score *= param.scale;
            
            #ifdef ADD_MASK
            int mask_offset = t - kv_len + param.query_seq_len;
            if (mask_offset >= 0 && mask_offset < param.query_seq_len) {
                 float m = (float)mask[s * param.query_seq_len + mask_offset];
                 score += m;
            }
            #elif defined(SET_MASK)
            int mask_val = mask[s * kv_len + t];
            if (mask_val == 0) score = -FLT_MAX;
            #endif
            
            shared_reduce[0] = score; // Store score for all threads
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        score = shared_reduce[0]; // All threads read the score
        
        // Online softmax update
        float new_max = max(cur_max, score);
        float exp_score = exp(score - new_max);
        float running_scale = exp(cur_max - new_max);
        
        cur_sum = cur_sum * running_scale + exp_score;
        cur_max = new_max;
        
        // Update accumulator for each dimension
        for (int i = 0; i < d_per_thread; ++i) {
            int d = tid + i * 32;
            if (d < head_dim) {
                float v_val = (float)value[v_base_offset + d * max_kv_len + t];
                acc[i] = acc[i] * running_scale + v_val * exp_score;
            }
        }
    }
    
    float inv_sum = 1.0f / cur_sum;
    
    // Write output
    auto out_ptr = output + ((b * param.query_seq_len + s) * param.head_num + h) * head_dim;
    for (int i = 0; i < d_per_thread; ++i) {
        int d = tid + i * 32;
        if (d < head_dim) {
            out_ptr[d] = (ftype)(acc[i] * inv_sum);
        }
    }
#endif
}
)metal";

#endif/* MNN_SUPPORT_TRANSFORMER_FUSE */
#endif

