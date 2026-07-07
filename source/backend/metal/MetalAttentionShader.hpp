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
    int mask_batch;
    int mask_head_num;
    int mask_q_len;
    int mask_k_len;
    float v_scale;
    float k_scale;
};

static inline bool attention_mask_hit(constant Param& param, int k) {
    if (param.mask_k_len <= 1) {
        return true;
    }
    int mask_k_start = max(param.key_seq_len - param.mask_k_len, 0);
    int local_k = k - mask_k_start;
    return local_k >= 0 && local_k < param.mask_k_len;
}

static inline int attention_mask_offset(constant Param& param, int b, int hn, int q, int k) {
    int mask_b = param.mask_batch <= 1 ? 0 : b;
    int mask_h = param.mask_head_num <= 1 ? 0 : hn;
    int mask_q = param.mask_q_len <= 1 ? 0 : min(q, param.mask_q_len - 1);
    int mask_k_start = max(param.key_seq_len - param.mask_k_len, 0);
    int local_k = param.mask_k_len <= 1 ? 0 : clamp(k - mask_k_start, 0, param.mask_k_len - 1);
    return ((mask_b * param.mask_head_num + mask_h) * param.mask_q_len + mask_q) * param.mask_k_len + local_k;
}

#if MNN_METAL_FLOAT16_STORAGE
typedef simdgroup_half8x8 simdgroup_T8x8;
#else
typedef simdgroup_float8x8 simdgroup_T8x8;
#endif

#define SIMD_GROUP_WIDTH 32
#ifdef QUANT_K
#ifdef DYNAMIC_QUANT_K
#define GETK(v, token_idx) ftype((float(v) * k_scales[(token_idx) * 2] + k_scales[(token_idx) * 2 + 1]))
#define GETK4(v, token_idx) (float4(v) * k_scales[(token_idx) * 2] + k_scales[(token_idx) * 2 + 1])
#else
#define GETK(v, token_idx) ftype((float(v) * param.k_scale))
#define GETK4(v, token_idx) (float4(v) * param.k_scale)
#endif
#else
#define GETK(v, token_idx) v
#define GETK4(v, token_idx) v
#endif
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
    device ftype* k_scales [[buffer(8)]],
    device ftype* v_scales [[buffer(9)]],
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
#ifdef QUANT_K
    auto B_offset = (const device char4*)past_key + ((idx_slk * param.batch + b)* head_num / group + zin) * head_dim / 4 + (0 * 4 + kl) * 2 + 0;
#else
    auto B_offset = past_key + ((idx_slk * param.batch + b)* head_num / group + zin) * head_dim / 4 + (0 * 4 + kl) * 2 + 0;
#endif

    for(int i = 0; i < head_dim/4; i += 8){
        ((threadgroup ftype4*)sdata)[(ml * 4 + kl) * 2 + 0] = A_offset[i + 0];
        ((threadgroup ftype4*)sdata)[(ml * 4 + kl) * 2 + 1] = A_offset[i + 1];

        ((threadgroup ftype4*)sdata)[256 + (nl * 4 + kl) * 2 + 0] = (ftype4)GETK4(B_offset[i + 0], idx_slk * param.batch + b);
        ((threadgroup ftype4*)sdata)[256 + (nl * 4 + kl) * 2 + 1] = (ftype4)GETK4(B_offset[i + 1], idx_slk * param.batch + b);
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

#if defined(DEFAULT_MASK)
    int kv_valid_offset = max(k_seq_len - q_seq_len, 0);
#endif

    int base_k_idx =  (slk * 4 + ncl) * 8 + 0;
    auto xy_out = output + (z * q_seq_piece_len + slq * 32 + mcl) * output_k_len + base_k_idx + 0;
    if(slq * 32 + mcl < q_seq_piece_len &&  seq_idx * q_seq_piece_len + slq * 32 + mcl < q_seq_len) {
        int ori_q_idx = seq_idx * q_seq_piece_len + slq * 32 + mcl;
        if(base_k_idx + 0 < output_k_len) {
            auto out0 =  ((threadgroup float*)sdata)[sindex_base + 0] * Vscale;
            #ifdef ADD_MASK
                if (attention_mask_hit(param, kv_start + base_k_idx + 0)) {
                    auto mask_val = mask[attention_mask_offset(param, b, hn, ori_q_idx, kv_start + base_k_idx + 0)];
                    out0 = mask_val + out0;
                }
            #elif defined(SET_MASK)
                if (attention_mask_hit(param, kv_start + base_k_idx + 0)) {
                    out0 = mask[attention_mask_offset(param, b, hn, ori_q_idx, kv_start + base_k_idx + 0)] == 0 ? -FLT_MAX : out0;
                }
            #elif defined(DEFAULT_MASK)
                int k_global = kv_start + base_k_idx + 0;
                if (k_global > kv_valid_offset + ori_q_idx) {
                    out0 = -FLT_MAX;
                }
            #endif
            xy_out[0] = out0;
        }
        if(base_k_idx + 1 < output_k_len) {
            auto out0 =  ((threadgroup float*)sdata)[sindex_base + 1] * Vscale;
            #ifdef ADD_MASK
                if (attention_mask_hit(param, kv_start + base_k_idx + 1)) {
                    auto mask_val = mask[attention_mask_offset(param, b, hn, ori_q_idx, kv_start + base_k_idx + 1)];
                    out0 = mask_val + out0;
                }
            #elif defined(SET_MASK)
                if (attention_mask_hit(param, kv_start + base_k_idx + 1)) {
                    out0 = mask[attention_mask_offset(param, b, hn, ori_q_idx, kv_start + base_k_idx + 1)] == 0 ? -FLT_MAX : out0;
                }
            #elif defined(DEFAULT_MASK)
                int k_global = kv_start + base_k_idx + 1;
                if (k_global > kv_valid_offset + ori_q_idx) {
                    out0 = -FLT_MAX;
                }
            #endif
            xy_out[1] = out0;
        }
        if(base_k_idx + 2 < output_k_len) {
            auto out0 =  ((threadgroup float*)sdata)[sindex_base + 2] * Vscale;
            #ifdef ADD_MASK
                if (attention_mask_hit(param, kv_start + base_k_idx + 2)) {
                    auto mask_val = mask[attention_mask_offset(param, b, hn, ori_q_idx, kv_start + base_k_idx + 2)];
                    out0 = mask_val + out0;
                }
            #elif defined(SET_MASK)
                if (attention_mask_hit(param, kv_start + base_k_idx + 2)) {
                    out0 = mask[attention_mask_offset(param, b, hn, ori_q_idx, kv_start + base_k_idx + 2)] == 0 ? -FLT_MAX : out0;
                }
            #elif defined(DEFAULT_MASK)
                int k_global = kv_start + base_k_idx + 2;
                if (k_global > kv_valid_offset + ori_q_idx) {
                    out0 = -FLT_MAX;
                }
            #endif
            xy_out[2] = out0;
        }
        if(base_k_idx + 3 < output_k_len) {
            auto out0 =  ((threadgroup float*)sdata)[sindex_base + 3] * Vscale;
            #ifdef ADD_MASK
                if (attention_mask_hit(param, kv_start + base_k_idx + 3)) {
                    auto mask_val = mask[attention_mask_offset(param, b, hn, ori_q_idx, kv_start + base_k_idx + 3)];
                    out0 = mask_val + out0;
                }
            #elif defined(SET_MASK)
                if (attention_mask_hit(param, kv_start + base_k_idx + 3)) {
                    out0 = mask[attention_mask_offset(param, b, hn, ori_q_idx, kv_start + base_k_idx + 3)] == 0 ? -FLT_MAX : out0;
                }
            #elif defined(DEFAULT_MASK)
                int k_global = kv_start + base_k_idx + 3;
                if (k_global > kv_valid_offset + ori_q_idx) {
                    out0 = -FLT_MAX;
                }
            #endif
            xy_out[3] = out0;
        }
        if(base_k_idx + 4 < output_k_len) {
            auto out0 =  ((threadgroup float*)sdata)[sindex_base + 4] * Vscale;
            #ifdef ADD_MASK
                if (attention_mask_hit(param, kv_start + base_k_idx + 4)) {
                    auto mask_val = mask[attention_mask_offset(param, b, hn, ori_q_idx, kv_start + base_k_idx + 4)];
                    out0 = mask_val + out0;
                }
            #elif defined(SET_MASK)
                if (attention_mask_hit(param, kv_start + base_k_idx + 4)) {
                    out0 = mask[attention_mask_offset(param, b, hn, ori_q_idx, kv_start + base_k_idx + 4)] == 0 ? -FLT_MAX : out0;
                }
            #elif defined(DEFAULT_MASK)
                int k_global = kv_start + base_k_idx + 4;
                if (k_global > kv_valid_offset + ori_q_idx) {
                    out0 = -FLT_MAX;
                }
            #endif
            xy_out[4] = out0;
        }
        if(base_k_idx + 5 < output_k_len) {
            auto out0 =  ((threadgroup float*)sdata)[sindex_base + 5] * Vscale;
            #ifdef ADD_MASK
                if (attention_mask_hit(param, kv_start + base_k_idx + 5)) {
                    auto mask_val = mask[attention_mask_offset(param, b, hn, ori_q_idx, kv_start + base_k_idx + 5)];
                    out0 = mask_val + out0;
                }
            #elif defined(SET_MASK)
                if (attention_mask_hit(param, kv_start + base_k_idx + 5)) {
                    out0 = mask[attention_mask_offset(param, b, hn, ori_q_idx, kv_start + base_k_idx + 5)] == 0 ? -FLT_MAX : out0;
                }
            #elif defined(DEFAULT_MASK)
                int k_global = kv_start + base_k_idx + 5;
                if (k_global > kv_valid_offset + ori_q_idx) {
                    out0 = -FLT_MAX;
                }
            #endif
            xy_out[5] = out0;
        }
        if(base_k_idx + 6 < output_k_len) {
            auto out0 =  ((threadgroup float*)sdata)[sindex_base + 6] * Vscale;
            #ifdef ADD_MASK
                if (attention_mask_hit(param, kv_start + base_k_idx + 6)) {
                    auto mask_val = mask[attention_mask_offset(param, b, hn, ori_q_idx, kv_start + base_k_idx + 6)];
                    out0 = mask_val + out0;
                }
            #elif defined(SET_MASK)
                if (attention_mask_hit(param, kv_start + base_k_idx + 6)) {
                    out0 = mask[attention_mask_offset(param, b, hn, ori_q_idx, kv_start + base_k_idx + 6)] == 0 ? -FLT_MAX : out0;
                }
            #elif defined(DEFAULT_MASK)
                int k_global = kv_start + base_k_idx + 6;
                if (k_global > kv_valid_offset + ori_q_idx) {
                    out0 = -FLT_MAX;
                }
            #endif
            xy_out[6] = out0;
        }
        if(base_k_idx + 7 < output_k_len) {
            auto out0 =  ((threadgroup float*)sdata)[sindex_base + 7] * Vscale;
            #ifdef ADD_MASK
                if (attention_mask_hit(param, kv_start + base_k_idx + 7)) {
                    auto mask_val = mask[attention_mask_offset(param, b, hn, ori_q_idx, kv_start + base_k_idx + 7)];
                    out0 = mask_val + out0;
                }
            #elif defined(SET_MASK)
                if (attention_mask_hit(param, kv_start + base_k_idx + 7)) {
                    out0 = mask[attention_mask_offset(param, b, hn, ori_q_idx, kv_start + base_k_idx + 7)] == 0 ? -FLT_MAX : out0;
                }
            #elif defined(DEFAULT_MASK)
                int k_global = kv_start + base_k_idx + 7;
                if (k_global > kv_valid_offset + ori_q_idx) {
                    out0 = -FLT_MAX;
                }
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
    device ftype* k_scales [[buffer(8)]],
    device ftype* v_scales [[buffer(9)]],
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
#ifdef QUANT_K
    auto B_offset = (const device char*)past_key + ((idx_slk * param.batch + b)* head_num / group + zin) * head_dim + 0 * 8 + kl * 4 + 0;
#else
    auto B_offset = past_key + ((idx_slk * param.batch + b)* head_num / group + zin) * head_dim + 0 * 8 + kl * 4 + 0;
#endif

    for(int i = 0; i < head_dim; i += 8){
        // 向量化写入 Q（4 元素一组）
        *((threadgroup ftype4*)(&((threadgroup ftype*)sdata)[rcl * 8 + kl * 4])) = *((const device ftype4*)(&A_offset[i]));

        ((threadgroup ftype*)sdata)[128 + (kl * 4 + 0) * 16 + rcl] = GETK(B_offset[i + 0], idx_slk * param.batch + b);
        ((threadgroup ftype*)sdata)[128 + (kl * 4 + 1) * 16 + rcl] = GETK(B_offset[i + 1], idx_slk * param.batch + b);
        ((threadgroup ftype*)sdata)[128 + (kl * 4 + 2) * 16 + rcl] = GETK(B_offset[i + 2], idx_slk * param.batch + b);
        ((threadgroup ftype*)sdata)[128 + (kl * 4 + 3) * 16 + rcl] = GETK(B_offset[i + 3], idx_slk * param.batch + b);
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

#if defined(DEFAULT_MASK)
    int kv_valid_offset = k_seq_len - q_seq_len;
#endif

    auto xy_out = output + (z * q_seq_piece_len + slq * 16 + rcl) * output_k_len + slk * 16 + kl * 8 + 0;
    if(slq * 16 + rcl < q_seq_piece_len &&  seq_idx * q_seq_piece_len + slq * 16 + rcl < q_seq_len) {
        int ori_q_idx = seq_idx * q_seq_piece_len + slq * 16 + rcl;
        if(slk * 16 + kl * 8 + 0 < output_k_len) {
            auto out0 =  ((threadgroup float*)sdata)[sindex_base + 0] * Vscale;
            #ifdef ADD_MASK
                if (attention_mask_hit(param, kv_start + slk * 16 + kl * 8 + 0)) {
                    auto mask_val = mask[attention_mask_offset(param, b, hn, ori_q_idx, kv_start + slk * 16 + kl * 8 + 0)];
                    out0 = mask_val + out0;
                }
            #elif defined(SET_MASK)
                if (attention_mask_hit(param, kv_start + slk * 16 + kl * 8 + 0)) {
                    out0 = mask[attention_mask_offset(param, b, hn, ori_q_idx, kv_start + slk * 16 + kl * 8 + 0)] == 0 ? -FLT_MAX : out0;
                }
            #elif defined(DEFAULT_MASK)
                int k_global = kv_start + slk * 16 + kl * 8 + 0;
                if (k_global > kv_valid_offset + ori_q_idx) {
                    out0 = -FLT_MAX;
                }
            #endif
            xy_out[0] = out0;
        }
        if(slk * 16 + kl * 8 + 1 < output_k_len) {
            auto out0 =  ((threadgroup float*)sdata)[sindex_base + 1] * Vscale;
            #ifdef ADD_MASK
                if (attention_mask_hit(param, kv_start + slk * 16 + kl * 8 + 1)) {
                    auto mask_val = mask[attention_mask_offset(param, b, hn, ori_q_idx, kv_start + slk * 16 + kl * 8 + 1)];
                    out0 = mask_val + out0;
                }
            #elif defined(SET_MASK)
                if (attention_mask_hit(param, kv_start + slk * 16 + kl * 8 + 1)) {
                    out0 = mask[attention_mask_offset(param, b, hn, ori_q_idx, kv_start + slk * 16 + kl * 8 + 1)] == 0 ? -FLT_MAX : out0;
                }
            #elif defined(DEFAULT_MASK)
                int k_global = kv_start + slk * 16 + kl * 8 + 1;
                if (k_global > kv_valid_offset + ori_q_idx) {
                    out0 = -FLT_MAX;
                }
            #endif
            xy_out[1] = out0;
        }
        if(slk * 16 + kl * 8 + 2 < output_k_len) {
            auto out0 =  ((threadgroup float*)sdata)[sindex_base + 2] * Vscale;
            #ifdef ADD_MASK
                if (attention_mask_hit(param, kv_start + slk * 16 + kl * 8 + 2)) {
                    auto mask_val = mask[attention_mask_offset(param, b, hn, ori_q_idx, kv_start + slk * 16 + kl * 8 + 2)];
                    out0 = mask_val + out0;
                }
            #elif defined(SET_MASK)
                if (attention_mask_hit(param, kv_start + slk * 16 + kl * 8 + 2)) {
                    out0 = mask[attention_mask_offset(param, b, hn, ori_q_idx, kv_start + slk * 16 + kl * 8 + 2)] == 0 ? -FLT_MAX : out0;
                }
            #elif defined(DEFAULT_MASK)
                int k_global = kv_start + slk * 16 + kl * 8 + 2;
                if (k_global > kv_valid_offset + ori_q_idx) {
                    out0 = -FLT_MAX;
                }
            #endif
            xy_out[2] = out0;
        }
        if(slk * 16 + kl * 8 + 3 < output_k_len) {
            auto out0 =  ((threadgroup float*)sdata)[sindex_base + 3] * Vscale;
            #ifdef ADD_MASK
                if (attention_mask_hit(param, kv_start + slk * 16 + kl * 8 + 3)) {
                    auto mask_val = mask[attention_mask_offset(param, b, hn, ori_q_idx, kv_start + slk * 16 + kl * 8 + 3)];
                    out0 = mask_val + out0;
                }
            #elif defined(SET_MASK)
                if (attention_mask_hit(param, kv_start + slk * 16 + kl * 8 + 3)) {
                    out0 = mask[attention_mask_offset(param, b, hn, ori_q_idx, kv_start + slk * 16 + kl * 8 + 3)] == 0 ? -FLT_MAX : out0;
                }
            #elif defined(DEFAULT_MASK)
                int k_global = kv_start + slk * 16 + kl * 8 + 3;
                if (k_global > kv_valid_offset + ori_q_idx) {
                    out0 = -FLT_MAX;
                }
            #endif
            xy_out[3] = out0;
        }
        if(slk * 16 + kl * 8 + 4 < output_k_len) {
            auto out0 =  ((threadgroup float*)sdata)[sindex_base + 4] * Vscale;
            #ifdef ADD_MASK
                if (attention_mask_hit(param, kv_start + slk * 16 + kl * 8 + 4)) {
                    auto mask_val = mask[attention_mask_offset(param, b, hn, ori_q_idx, kv_start + slk * 16 + kl * 8 + 4)];
                    out0 = mask_val + out0;
                }
            #elif defined(SET_MASK)
                if (attention_mask_hit(param, kv_start + slk * 16 + kl * 8 + 4)) {
                    out0 = mask[attention_mask_offset(param, b, hn, ori_q_idx, kv_start + slk * 16 + kl * 8 + 4)] == 0 ? -FLT_MAX : out0;
                }
            #elif defined(DEFAULT_MASK)
                int k_global = kv_start + slk * 16 + kl * 8 + 4;
                if (k_global > kv_valid_offset + ori_q_idx) {
                    out0 = -FLT_MAX;
                }
            #endif
            xy_out[4] = out0;
        }
        if(slk * 16 + kl * 8 + 5 < output_k_len) {
            auto out0 =  ((threadgroup float*)sdata)[sindex_base + 5] * Vscale;
            #ifdef ADD_MASK
                if (attention_mask_hit(param, kv_start + slk * 16 + kl * 8 + 5)) {
                    auto mask_val = mask[attention_mask_offset(param, b, hn, ori_q_idx, kv_start + slk * 16 + kl * 8 + 5)];
                    out0 = mask_val + out0;
                }
            #elif defined(SET_MASK)
                if (attention_mask_hit(param, kv_start + slk * 16 + kl * 8 + 5)) {
                    out0 = mask[attention_mask_offset(param, b, hn, ori_q_idx, kv_start + slk * 16 + kl * 8 + 5)] == 0 ? -FLT_MAX : out0;
                }
            #elif defined(DEFAULT_MASK)
                int k_global = kv_start + slk * 16 + kl * 8 + 5;
                if (k_global > kv_valid_offset + ori_q_idx) {
                    out0 = -FLT_MAX;
                }
            #endif
            xy_out[5] = out0;
        }
        if(slk * 16 + kl * 8 + 6 < output_k_len) {
            auto out0 =  ((threadgroup float*)sdata)[sindex_base + 6] * Vscale;
            #ifdef ADD_MASK
                if (attention_mask_hit(param, kv_start + slk * 16 + kl * 8 + 6)) {
                    auto mask_val = mask[attention_mask_offset(param, b, hn, ori_q_idx, kv_start + slk * 16 + kl * 8 + 6)];
                    out0 = mask_val + out0;
                }
            #elif defined(SET_MASK)
                if (attention_mask_hit(param, kv_start + slk * 16 + kl * 8 + 6)) {
                    out0 = mask[attention_mask_offset(param, b, hn, ori_q_idx, kv_start + slk * 16 + kl * 8 + 6)] == 0 ? -FLT_MAX : out0;
                }
            #elif defined(DEFAULT_MASK)
                int k_global = kv_start + slk * 16 + kl * 8 + 6;
                if (k_global > kv_valid_offset + ori_q_idx) {
                    out0 = -FLT_MAX;
                }
            #endif
            xy_out[6] = out0;
        }
        if(slk * 16 + kl * 8 + 7 < output_k_len) {
            auto out0 =  ((threadgroup float*)sdata)[sindex_base + 7] * Vscale;
            #ifdef ADD_MASK
                if (attention_mask_hit(param, kv_start + slk * 16 + kl * 8 + 7)) {
                    auto mask_val = mask[attention_mask_offset(param, b, hn, ori_q_idx, kv_start + slk * 16 + kl * 8 + 7)];
                    out0 = mask_val + out0;
                }
            #elif defined(SET_MASK)
                if (attention_mask_hit(param, kv_start + slk * 16 + kl * 8 + 7)) {
                    out0 = mask[attention_mask_offset(param, b, hn, ori_q_idx, kv_start + slk * 16 + kl * 8 + 7)] == 0 ? -FLT_MAX : out0;
                }
            #elif defined(DEFAULT_MASK)
                int k_global = kv_start + slk * 16 + kl * 8 + 7;
                if (k_global > kv_valid_offset + ori_q_idx) {
                    out0 = -FLT_MAX;
                }
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
#ifdef QUANT_K
    const device char* B_offset = (const device char*)past_key + ((z_global * param.batch + b) * offset / group + offset_head_kv);
#else
    device const ftype* B_offset = past_key + (z_global * param.batch + b) * offset / group + offset_head_kv;
#endif
    const int output_offset = y * param.q_seq_piece_len * output_k_len;
    float out0 = 0.0;

    // 两路流水：每次处理 8 个标量（两个 float4），减少循环开销
    int itN = head_dim / 8; // head_dim 保证 16 对齐，因此 /8 为整数
    const device ftype4* A4p = (const device ftype4*)A_offset;
#ifdef QUANT_K
    const device char4* B4p_c = (const device char4*)B_offset;
#else
    const device ftype4* B4p = (const device ftype4*)B_offset;
#endif
    for (int i = 0; i < itN; ++i) {
#ifdef QUANT_K
        float4 B0 = GETK4(B4p_c[i * 2 + 0], z_global * param.batch + b);
        float4 B1 = GETK4(B4p_c[i * 2 + 1], z_global * param.batch + b);
#else
        float4 B0 = float4(B4p[i * 2 + 0]);
        float4 B1 = float4(B4p[i * 2 + 1]);
#endif
        float4 A0 = float4(A4p[i * 2 + 0]);
        float4 A1 = float4(A4p[i * 2 + 1]);
        out0 += dot(A0, B0) + dot(A1, B1);
    }

    out0 *= Vscale;

#ifdef ADD_MASK
    if (attention_mask_hit(param, z_global)) {
        auto mask_val = mask[attention_mask_offset(param, b, hn, q_idx, z_global)];
        out0 = mask_val + out0;
    }
#elif defined(SET_MASK)
    if (attention_mask_hit(param, z_global)) {
        out0 = mask[attention_mask_offset(param, b, hn, q_idx, z_global)] == 0 ? -FLT_MAX : out0;
    }
#elif defined(DEFAULT_MASK)
    {
        int kv_valid_offset = max(key_seq_len - query_seq_len, 0);
        int k_global = z_global;
        if (k_global > kv_valid_offset + q_idx) {
            out0 = -FLT_MAX;
        }
    }
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
    device ftype* k_scales [[buffer(8)]],
    device ftype* v_scales [[buffer(9)]],
#ifdef SIMD_GROUP_REDUCE
    uint3 gid[[threadgroup_position_in_grid]],
    uint  tiisg[[thread_index_in_simdgroup]],
    uint  sgitg[[simdgroup_index_in_threadgroup]]
#else
    uint3 gid[[thread_position_in_grid]]
#endif
) {
#ifdef SIMD_GROUP_REDUCE
    int x = gid.x; // query_seq_len
    int y = gid.y; // head_num * batch
    int z = gid.z; // key_seq_len
#else
    int x = gid.x; // query_seq_len
    int y = gid.y; // head_num * batch
    int z = gid.z; // key_seq_len
#endif
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
#ifdef QUANT_K
    const device char* Pastkey_offset = (const device char*)past_key + ((z * param.batch + b) * offset / group + offset_head_kv);
#else
    device ftype* Pastkey_offset = past_key + (z * param.batch + b) * offset / group + offset_head_kv;
#endif
    float Vscale = (float)param.scale;



    // 保持与原 Mask 分支一致的计算路径，避免提前返回带来的数值波动
    float out[GROUP_SIZE] = {0.0};
#if defined(QUANT_K) && defined(DYNAMIC_QUANT_K)
    int k_token_idx = z * param.batch + b;
    float k_scale = k_scales[k_token_idx * 2];
    float k_bias = k_scales[k_token_idx * 2 + 1];
#endif

#ifdef SIMD_GROUP_REDUCE
    {
        int itN = head_dim / 8;
        for (int i = tiisg; i < itN; i+=SIMD_GROUP_WIDTH) {
#ifdef QUANT_K
#ifdef DYNAMIC_QUANT_K
            float4 B0 = float4(((const device char4*)Pastkey_offset)[i * 2 + 0]) * k_scale + k_bias;
            float4 B1 = float4(((const device char4*)Pastkey_offset)[i * 2 + 1]) * k_scale + k_bias;
#else
            float4 B0 = GETK4(((const device char4*)Pastkey_offset)[i * 2 + 0], z * param.batch + b);
            float4 B1 = GETK4(((const device char4*)Pastkey_offset)[i * 2 + 1], z * param.batch + b);
#endif
#else
            float4 B0 = float4(((const device ftype4*)Pastkey_offset)[i * 2 + 0]);
            float4 B1 = float4(((const device ftype4*)Pastkey_offset)[i * 2 + 1]);
#endif
            for (int j = 0; j < group; j++) {
                const device ftype4* Ajp = (const device ftype4*)(A_offset + head_dim * j);
                float4 A0 = float4(Ajp[i * 2 + 0]);
                float4 A1 = float4(Ajp[i * 2 + 1]);
                out[j] += dot(A0, B0) + dot(A1, B1);
            }
        }
    }
    for(int j = 0; j < group; j++) {
        out[j] = simd_sum(out[j]);
    }
#else
    {
        // 统一使用 float4 向量化点积（QUANT_K 走 GETK4）
        int itN = head_dim / 8;
        for (int i = 0; i < itN; ++i) {
#ifdef QUANT_K
#ifdef DYNAMIC_QUANT_K
            float4 B0 = float4(((const device char4*)Pastkey_offset)[i * 2 + 0]) * k_scale + k_bias;
            float4 B1 = float4(((const device char4*)Pastkey_offset)[i * 2 + 1]) * k_scale + k_bias;
#else
            float4 B0 = GETK4(((const device char4*)Pastkey_offset)[i * 2 + 0], z * param.batch + b);
            float4 B1 = GETK4(((const device char4*)Pastkey_offset)[i * 2 + 1], z * param.batch + b);
#endif
#else
            float4 B0 = float4(((const device ftype4*)Pastkey_offset)[i * 2 + 0]);
            float4 B1 = float4(((const device ftype4*)Pastkey_offset)[i * 2 + 1]);
#endif
            for (int j = 0; j < group; j++) {
                const device ftype4* Ajp = (const device ftype4*)(A_offset + head_dim * j);
                float4 A0 = float4(Ajp[i * 2 + 0]);
                float4 A1 = float4(Ajp[i * 2 + 1]);
                out[j] += dot(A0, B0) + dot(A1, B1);
            }
        }
    }
#endif

#ifdef SIMD_GROUP_REDUCE
    if (tiisg == 0) {
#endif

    for(int j = 0; j < group; j++) {
        out[j] *= Vscale;
        #ifdef ADD_MASK
            if (attention_mask_hit(param, z)) {
                float mask_val = mask[attention_mask_offset(param, b, kv_hn * group + j, x, z)];
                out[j] += mask_val;
            }
        #elif defined(SET_MASK)
            if (attention_mask_hit(param, z)) {
                int mask_val = mask[attention_mask_offset(param, b, kv_hn * group + j, x, z)];
                out[j] = mask_val == 0 ? -FLT_MAX : out[j];
            }
        #elif defined(DEFAULT_MASK)
        {
            int kv_valid_offset = max(key_seq_len - param.query_seq_len, 0);
            int k_global = z;
            if (k_global > kv_valid_offset + x) {
                out[j] = -FLT_MAX;
            }
        }
        #endif
        output[((y * group + j) * param.query_seq_len + x) * key_seq_len + z] = (ftype)out[j];
    }
#ifdef SIMD_GROUP_REDUCE
    }
#endif
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
    float v_scale;
    float k_scale;
};
// Key:   [batch, kv_seq_len, head_num / group * head_dim] -> [max_kv_len, batch, head_num / group * head_dim]
// Value: [batch, kv_seq_len, head_num / group * head_dim] -> [batch, head_num / group * head_dim, max_kv_len]

#ifdef KV_QUANT_K
#define KOUT_TYPE char
#else
#define KOUT_TYPE ftype
#endif

#ifdef KV_QUANT_V
#define VOUT_TYPE char
#else
#define VOUT_TYPE ftype
#endif


kernel void copy(const device ftype* input0 [[buffer(0)]],
    const device ftype* input1 [[buffer(1)]],
    device KOUT_TYPE* output0 [[buffer(2)]],
    device VOUT_TYPE* output1 [[buffer(3)]],
    constant Param& param [[buffer(4)]],
    device ftype* k_scales [[buffer(8)]],
    device ftype* v_scales [[buffer(9)]],
#ifdef DYNAMIC_QUANT
    uint3 gid[[threadgroup_position_in_grid]],
    uint tiisg[[thread_index_in_simdgroup]],
    uint titg[[thread_index_in_threadgroup]],
    uint sgitg[[simdgroup_index_in_threadgroup]],
    uint3 tptg_3d[[threads_per_threadgroup]]
#else
    uint3 gid[[thread_position_in_grid]]
#endif
) {
#ifdef DYNAMIC_QUANT
    const int y = gid.y; // kv_seq_len
    const int b = gid.z; // batch
    const uint tptg = tptg_3d.x * tptg_3d.y * tptg_3d.z;
    if (y >= param.kv_seq_len || b >= param.batch) {
        return;
    }

#if defined(KV_QUANT_K) || defined(KV_QUANT_V)
    float k_scale = param.k_scale;
    float k_bias = 0.0f;
    float v_scale = param.v_scale;
    float v_bias = 0.0f;

#ifdef DYNAMIC_QUANT
    // Dynamic quantization scale calculation
    {
#ifdef KV_QUANT_K
        float min_k = 1000000.0f;
        float max_k = -1000000.0f;
#endif
#ifdef KV_QUANT_V
        float min_v = 1000000.0f;
        float max_v = -1000000.0f;
#endif

        int vector_end = (param.head_count / 4) * 4;
        for (int x = int(titg) * 4; x < vector_end; x += int(tptg) * 4) {
            const int in_idx  = (b * param.kv_seq_len + y) * param.head_count + x;
#ifdef KV_QUANT_K
            float4 k4 = float4(((const device ftype4*)(input0 + in_idx))[0]);
            float k_min = metal::min(metal::min(k4.x, k4.y), metal::min(k4.z, k4.w));
            float k_max = metal::max(metal::max(k4.x, k4.y), metal::max(k4.z, k4.w));
            min_k = metal::min(min_k, k_min);
            max_k = metal::max(max_k, k_max);
#endif
#ifdef KV_QUANT_V
            float4 v4 = float4(((const device ftype4*)(input1 + in_idx))[0]);
            float v_min = metal::min(metal::min(v4.x, v4.y), metal::min(v4.z, v4.w));
            float v_max = metal::max(metal::max(v4.x, v4.y), metal::max(v4.z, v4.w));
            min_v = metal::min(min_v, v_min);
            max_v = metal::max(max_v, v_max);
#endif
        }
        for (int x = vector_end + int(titg); x < param.head_count; x += int(tptg)) {
            const int in_idx  = (b * param.kv_seq_len + y) * param.head_count + x;
#ifdef KV_QUANT_K
            float k = (float)input0[in_idx];
            min_k = metal::min(min_k, k);
            max_k = metal::max(max_k, k);
#endif
#ifdef KV_QUANT_V
            float v = (float)input1[in_idx];
            min_v = metal::min(min_v, v);
            max_v = metal::max(max_v, v);
#endif
        }

#ifdef SIMD_GROUP_REDUCE
#ifdef KV_QUANT_K
        min_k = simd_min(min_k);
        max_k = simd_max(max_k);
#endif
#ifdef KV_QUANT_V
        min_v = simd_min(min_v);
        max_v = simd_max(max_v);
#endif
#else
#ifdef KV_QUANT_K
        threadgroup float tg_min_k[256];
        threadgroup float tg_max_k[256];
#endif
#ifdef KV_QUANT_V
        threadgroup float tg_min_v[256];
        threadgroup float tg_max_v[256];
#endif

#ifdef KV_QUANT_K
        tg_min_k[titg] = min_k;
        tg_max_k[titg] = max_k;
#endif
#ifdef KV_QUANT_V
        tg_min_v[titg] = min_v;
        tg_max_v[titg] = max_v;
#endif

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (titg == 0) {
            for (uint i = 1; i < tptg; i++) {
#ifdef KV_QUANT_K
                min_k = metal::min(min_k, tg_min_k[i]);
                max_k = metal::max(max_k, tg_max_k[i]);
#endif
#ifdef KV_QUANT_V
                min_v = metal::min(min_v, tg_min_v[i]);
                max_v = metal::max(max_v, tg_max_v[i]);
#endif
            }
#ifdef KV_QUANT_K
            tg_min_k[0] = min_k;
            tg_max_k[0] = max_k;
#endif
#ifdef KV_QUANT_V
            tg_min_v[0] = min_v;
            tg_max_v[0] = max_v;
#endif
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
#ifdef KV_QUANT_K
        min_k = tg_min_k[0];
        max_k = tg_max_k[0];
#endif
#ifdef KV_QUANT_V
        min_v = tg_min_v[0];
        max_v = tg_max_v[0];
#endif
#endif
#ifdef KV_QUANT_K
        k_scale = (max_k - min_k) / 255.0f;
        if (k_scale < 1e-6f) k_scale = 1e-6f;
        k_bias = min_k + 128.0f * k_scale;
#endif
#ifdef KV_QUANT_V
        v_scale = (max_v - min_v) / 255.0f;
        if (v_scale < 1e-6f) v_scale = 1e-6f;
        v_bias = min_v + 128.0f * v_scale;
#endif

        if (titg == 0) {
#ifdef KV_QUANT_K
            int k_tok_idx = param.dst_k_offset / param.head_count + (y * param.batch + b);
            k_scales[k_tok_idx * 2 + 0] = k_scale;
            k_scales[k_tok_idx * 2 + 1] = k_bias;
#endif
#ifdef KV_QUANT_V
            int v_tok_idx = b * param.max_kv_len + (param.dst_k_offset / param.head_count + y);
            v_scales[v_tok_idx * 2 + 0] = v_scale;
            v_scales[v_tok_idx * 2 + 1] = v_bias;
#endif
        }
    }
#endif // DYNAMIC_QUANT
#endif // KV_QUANT_K || KV_QUANT_V

    int vector_end = (param.head_count / 4) * 4;
    for (int x = int(titg) * 4; x < vector_end; x += int(tptg) * 4) {
        const int in_idx  = (b * param.kv_seq_len + y) * param.head_count + x;

        // Write K
        int out_idx_k = param.dst_k_offset + (y * param.batch + b) * param.head_count + x;
#ifdef KV_QUANT_K
        float4 k = float4(((const device ftype4*)(input0 + in_idx))[0]);
        if (k_scale == 0.0f) {
            ((device char4*)(output0 + out_idx_k))[0] = char4(0);
        } else {
            int4 qi = int4(rint((k - k_bias) / k_scale));
            qi = clamp(qi, int4(-128), int4(127));
            ((device char4*)(output0 + out_idx_k))[0] = char4(qi);
        }
#else
        ((device ftype4*)(output0 + out_idx_k))[0] = ((const device ftype4*)(input0 + in_idx))[0];
#endif

        // Write V
        int out_idx_v = param.dst_v_offset + (b * param.head_count + x) * param.max_kv_len + y;
#ifdef KV_QUANT_V
        float4 v = float4(((const device ftype4*)(input1 + in_idx))[0]);
        if (v_scale == 0.0f) {
            output1[out_idx_v] = (char)0;
            output1[out_idx_v + param.max_kv_len] = (char)0;
            output1[out_idx_v + param.max_kv_len * 2] = (char)0;
            output1[out_idx_v + param.max_kv_len * 3] = (char)0;
        } else {
            int4 qi = int4(rint((v - v_bias) / v_scale));
            qi = clamp(qi, int4(-128), int4(127));
            output1[out_idx_v] = (char)qi.x;
            output1[out_idx_v + param.max_kv_len] = (char)qi.y;
            output1[out_idx_v + param.max_kv_len * 2] = (char)qi.z;
            output1[out_idx_v + param.max_kv_len * 3] = (char)qi.w;
        }
#else
        output1[out_idx_v] = input1[in_idx];
        output1[out_idx_v + param.max_kv_len] = input1[in_idx + 1];
        output1[out_idx_v + param.max_kv_len * 2] = input1[in_idx + 2];
        output1[out_idx_v + param.max_kv_len * 3] = input1[in_idx + 3];
#endif
    }
    for (int x = vector_end + int(titg); x < param.head_count; x += int(tptg)) {
        const int in_idx  = (b * param.kv_seq_len + y) * param.head_count + x;

        int out_idx_k = param.dst_k_offset + (y * param.batch + b) * param.head_count + x;
#ifdef KV_QUANT_K
        float k = (float)input0[in_idx];
        if (k_scale == 0.0f) {
            output0[out_idx_k] = (char)0;
        } else {
            float q = (k - k_bias) / k_scale;
            int qi = (int)rint(q);
            qi = clamp(qi, -128, 127);
            output0[out_idx_k] = (char)qi;
        }
#else
        output0[out_idx_k] = input0[in_idx];
#endif

        int out_idx_v = param.dst_v_offset + (b * param.head_count + x) * param.max_kv_len + y;
#ifdef KV_QUANT_V
        float v = (float)input1[in_idx];
        if (v_scale == 0.0f) {
            output1[out_idx_v] = (char)0;
        } else {
            float q = (v - v_bias) / v_scale;
            int qi = (int)rint(q);
            qi = clamp(qi, -128, 127);
            output1[out_idx_v] = (char)qi;
        }
#else
        output1[out_idx_v] = input1[in_idx];
#endif
    }
#else
    const int x = gid.x; // head_num / group * head_dim
    const int y = gid.y; // kv_seq_len
    const int b = gid.z; // batch
    if (x >= param.head_count || y >= param.kv_seq_len || b >= param.batch) {
        return;
    }
    const int in_idx  = (b * param.kv_seq_len + y) * param.head_count + x;

    int out_idx_k = param.dst_k_offset + (y * param.batch + b) * param.head_count + x;
#ifdef KV_QUANT_K
    float k = (float)input0[in_idx];
    if (param.k_scale == 0.0f) {
        output0[out_idx_k] = (char)0;
    } else {
        float q = k / param.k_scale;
        int qi = (int)rint(q);
        qi = clamp(qi, -128, 127);
        output0[out_idx_k] = (char)qi;
    }
#else
    output0[out_idx_k] = input0[in_idx];
#endif

    int out_idx_v = param.dst_v_offset + (b * param.head_count + x) * param.max_kv_len + y;
#ifdef KV_QUANT_V
    float v = (float)input1[in_idx];
    if (param.v_scale == 0.0f) {
        output1[out_idx_v] = (char)0;
    } else {
        float q = v / param.v_scale;
        int qi = (int)rint(q);
        qi = clamp(qi, -128, 127);
        output1[out_idx_v] = (char)qi;
    }
#else
    output1[out_idx_v] = input1[in_idx];
#endif
#endif
}

#undef KOUT_TYPE
#undef VOUT_TYPE
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
    int mask_batch;
    int mask_head_num;
    int mask_q_len;
    int mask_k_len;
    float v_scale;
    float k_scale;
};
#if MNN_METAL_FLOAT16_STORAGE
typedef simdgroup_half8x8 simdgroup_T8x8;
#else
typedef simdgroup_float8x8 simdgroup_T8x8;
#endif
#ifdef QUANT_V
#ifdef DYNAMIC_QUANT_V
#define GETV(v, tok_idx) ftype((float(v) * v_scales[(tok_idx) * 2] + v_scales[(tok_idx) * 2 + 1]))
#define GETV4(v, tok_idx) (float4(v) * float4(v_scales[(tok_idx) * 2], v_scales[((tok_idx) + 1) * 2], v_scales[((tok_idx) + 2) * 2], v_scales[((tok_idx) + 3) * 2]) + \
     float4(v_scales[(tok_idx) * 2 + 1], v_scales[((tok_idx) + 1) * 2 + 1], v_scales[((tok_idx) + 2) * 2 + 1], v_scales[((tok_idx) + 3) * 2 + 1]))
#else
#define GETV(v, tok_idx) ftype((float(v) * param.v_scale))
#define GETV4(v, tok_idx) (float4(v) * param.v_scale)
#endif
#else
#define GETV(v, tok_idx) v
#define GETV4(v, tok_idx) v
#endif

#ifdef USE_METAL_TENSOR_OPS
kernel void prefill_qkv_tensor(const device ftype* input0 [[buffer(0)]],
    device ftype4* output [[buffer(1)]],
    device ftype4* past_value [[buffer(2)]],
    constant int &seq_idx [[buffer(3)]],
    constant Param& param [[buffer(4)]],
    device ftype* k_scales [[buffer(8)]],
    device ftype* v_scales [[buffer(9)]],
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
#ifdef QUANT_V
    auto B_offset = (const device char4*)past_value + (zin * head_dim + hm * 32 + nl) * param.max_kv_len / 4 + (0 * 4 + kvl) * 2 + 0;
#else
    auto B_offset = past_value + (zin * head_dim + hm * 32 + nl) * param.max_kv_len / 4 + (0 * 4 + kvl) * 2 + 0;
#endif


    for(int i = 0; i < (value_seq_len+3)/4; i += 8){
        // 向量化写入 P（两次 ftype4，覆盖 8 个标量）
        *((threadgroup ftype4*)(&((threadgroup ftype*)sdata)[(ml * 4 + kl) * 8 + 0])) = *((const device ftype4*)(&A_offset[4*i + 0]));
        *((threadgroup ftype4*)(&((threadgroup ftype*)sdata)[(ml * 4 + kl) * 8 + 4])) = *((const device ftype4*)(&A_offset[4*i + 4]));

        ((threadgroup ftype4*)sdata)[256 + (nl * 4 + kvl) * 2 + 0] = (ftype4)GETV4(B_offset[i + 0], b * param.max_kv_len + i * 4 + 0);
        ((threadgroup ftype4*)sdata)[256 + (nl * 4 + kvl) * 2 + 1] = (ftype4)GETV4(B_offset[i + 1], b * param.max_kv_len + i * 4 + 4);

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
    device ftype* k_scales [[buffer(8)]],
    device ftype* v_scales [[buffer(9)]],
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
#ifdef QUANT_V
    auto B_offset = (const device char*)past_value + (zin * head_dim + hm * 16 + nl * 4 + 0) * param.max_kv_len + (0 * 8 + kcl);
#else
    auto B_offset = past_value + (zin * head_dim + hm * 16 + nl * 4 + 0) * param.max_kv_len + (0 * 8 + kcl);
#endif

    for(int i = 0; i < align_value_len; i += 8){
        *((threadgroup ftype4*)(&((threadgroup ftype*)sdata)[rcl * 8 + kl * 4 + 0])) = *((const device ftype4*)(&A_offset[i + 0]));

        ((threadgroup ftype*)sdata)[128 + kcl * 16 + nl * 4 + 0] = GETV(B_offset[i + 0 * param.max_kv_len], b * param.max_kv_len + i);
        ((threadgroup ftype*)sdata)[128 + kcl * 16 + nl * 4 + 1] = GETV(B_offset[i + 1 * param.max_kv_len], b * param.max_kv_len + i);
        ((threadgroup ftype*)sdata)[128 + kcl * 16 + nl * 4 + 2] = GETV(B_offset[i + 2 * param.max_kv_len], b * param.max_kv_len + i);
        ((threadgroup ftype*)sdata)[128 + kcl * 16 + nl * 4 + 3] = GETV(B_offset[i + 3 * param.max_kv_len], b * param.max_kv_len + i);

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
#ifdef ATTENTION_C4
    // [mNumHead * (mHeadDim / 4), mBatch * mSeqLen, 4]
    auto xy_out = output + (b * q_seq_len + seq_idx * q_seq_piece_len + sl * 16 + rcl) * 4 + (hn * head_dim / 4 + hm * 4 + kl * 2) * 4 * param.batch * q_seq_len + 0;
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
            xy_out[q_seq_len * 4 + 0] =  ((threadgroup float*)sdata)[sindex_base + 4];
        }
        if(hm * 16 + kl * 8 + 5 < head_dim) {
            xy_out[q_seq_len * 4 + 1] =  ((threadgroup float*)sdata)[sindex_base + 5];
        }
        if(hm * 16 + kl * 8 + 6 < head_dim) {
            xy_out[q_seq_len * 4 + 2] =  ((threadgroup float*)sdata)[sindex_base + 6];
        }
        if(hm * 16 + kl * 8 + 7 < head_dim) {
            xy_out[q_seq_len * 4 + 3] =  ((threadgroup float*)sdata)[sindex_base + 7];
        }
    }
#else
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
#endif

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
#ifdef QUANT_V
    const device char *B_offset = ((const device char*)past_value) + offset_head * param.max_kv_len;
#else
    device const ftype *B_offset = past_value + offset_head * param.max_kv_len;
#endif
    float4 out4 = 0.0;

    for(int i = 0; i < align_value_len; i += 4){
        float4 A = float4(((const device ftype4*)(A_offset + i))[0]);
#ifdef QUANT_V
        float4 B = GETV4(((const device char4*)(B_offset + i))[0], b * param.max_kv_len + i);
#else
        float4 B = float4(((const device ftype4*)(B_offset + i))[0]);
#endif
        out4 += A * B;
    }
    float out = out4.x + out4.y + out4.z + out4.w;
#ifdef ATTENTION_C4
    // [mNumHead * (mHeadDim / 4), mBatch * mSeqLen, 4]
    {
        int c = hn * head_dim + z;
        int co = c / 4;
        int ci = c % 4;
        output[(b * q_seq_len + x) * 4 + ci + co * param.batch * q_seq_len * 4] = (ftype)out;
    }
#else
    // [mBatch, mSeqLen, mNumHead, mHeadDim]
    output[(b * q_seq_len + q_idx) * stride * group + (hn * head_dim + z)] = out;
#endif
#endif
}

kernel void decode_qkv(const device ftype* input0 [[buffer(0)]],
    device ftype* output [[buffer(1)]],
    device ftype* past_value [[buffer(2)]],
    // docode actually not compute in block
    constant int &seq_idx [[buffer(3)]],
    constant Param& param [[buffer(4)]],
    device ftype* k_scales [[buffer(8)]],
    device ftype* v_scales [[buffer(9)]],
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
#ifdef QUANT_V
    const device char *Pastvalue_offset8 = ((const device char*)past_value) + offset_head;
#else
    device ftype *Pastvalue_offset = past_value + offset_head;
#endif
    float out = 0;

#ifdef SIMD_GROUP_REDUCE
    float4 out4 = 0;
    for(int i = tiisg * 4; i < align_value_len; i+=SIMD_GROUP_WIDTH * 4){
        float4 A = float4(((const device ftype4*)(A_offset + i))[0]);
#ifdef QUANT_V
        float4 B = GETV4(((const device char4*)(Pastvalue_offset8 + i))[0], b * param.max_kv_len + i);
#else
        float4 B = float4(((const device ftype4*)(Pastvalue_offset + i))[0]);
#endif
        out4 += A * B;
    }
    out = out4.x + out4.y + out4.z + out4.w;
    out = simd_sum(out);
    if(tiisg == 0) {
#ifdef ATTENTION_C4
        // [mNumHead * (mHeadDim / 4), mBatch * mSeqLen, 4]
        {
            int c = hn * head_dim + z;
            int co = c / 4;
            int ci = c % 4;
            output[(b * q_seq_len + x) * 4 + ci + co * param.batch * q_seq_len * 4] = (ftype)out;
        }
#else
        // [mBatch, mSeqLen, mNumHead, mHeadDim]
        output[((b * q_seq_len + x) * head_num + hn) * head_dim + z] = (ftype)out;
#endif
    }
#else
    float4 out4 = 0;
    for(int i = 0; i < align_value_len; i += 4){
        float4 A = float4(((const device ftype4*)(A_offset + i))[0]);
#ifdef QUANT_V
        float4 B = GETV4(((const device char4*)(Pastvalue_offset8 + i))[0], b * param.max_kv_len + i);
#else
        float4 B = float4(((const device ftype4*)(Pastvalue_offset + i))[0]);
#endif
        out4 += A * B;
    }
    out = out4.x + out4.y + out4.z + out4.w;
#ifdef ATTENTION_C4
    // [mNumHead * (mHeadDim / 4), mBatch * mSeqLen, 4]
    {
        int c = hn * head_dim + z;
        int co = c / 4;
        int ci = c % 4;
        output[(b * q_seq_len + x) * 4 + ci + co * param.batch * q_seq_len * 4] = (ftype)out;
    }
#else
    output[((b * q_seq_len + x) * head_num + hn) * head_dim + z] = (ftype)out;
#endif
#endif
}

kernel void decode_qkv_c2(const device ftype* input0 [[buffer(0)]],
    device ftype* output [[buffer(1)]],
    device ftype* past_value [[buffer(2)]],
    constant int &seq_idx [[buffer(3)]],
    constant Param& param [[buffer(4)]],
    device ftype* k_scales [[buffer(8)]],
    device ftype* v_scales [[buffer(9)]],
    uint3 gid[[threadgroup_position_in_grid]],
    uint  tiisg[[thread_index_in_simdgroup]]
) {
    const int x = gid.x;
    const int y = gid.y;
    const int z = gid.z * 2;
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

    device const ftype *A_offset = input0 + (y * q_seq_len + x) * align_value_len;
#ifdef QUANT_V
    const device char *B0 = ((const device char*)past_value) + (yin * head_dim + z + 0) * param.max_kv_len;
    const device char *B1 = ((const device char*)past_value) + (yin * head_dim + z + 1) * param.max_kv_len;
#else
    device const ftype *B0 = past_value + (yin * head_dim + z + 0) * param.max_kv_len;
    device const ftype *B1 = past_value + (yin * head_dim + z + 1) * param.max_kv_len;
#endif

    float4 out0 = 0;
    float4 out1 = 0;
    for(int i = tiisg * 4; i < align_value_len; i += SIMD_GROUP_WIDTH * 4){
        float4 A = float4(((const device ftype4*)(A_offset + i))[0]);
#ifdef QUANT_V
#ifdef DYNAMIC_QUANT_V
        int tok_idx = b * param.max_kv_len + i;
        float4 scale4 = float4(v_scales[tok_idx * 2], v_scales[(tok_idx + 1) * 2],
                               v_scales[(tok_idx + 2) * 2], v_scales[(tok_idx + 3) * 2]);
        float4 bias4 = float4(v_scales[tok_idx * 2 + 1], v_scales[(tok_idx + 1) * 2 + 1],
                              v_scales[(tok_idx + 2) * 2 + 1], v_scales[(tok_idx + 3) * 2 + 1]);
        out0 += A * (float4(((const device char4*)(B0 + i))[0]) * scale4 + bias4);
        out1 += A * (float4(((const device char4*)(B1 + i))[0]) * scale4 + bias4);
#else
        out0 += A * GETV4(((const device char4*)(B0 + i))[0], b * param.max_kv_len + i);
        out1 += A * GETV4(((const device char4*)(B1 + i))[0], b * param.max_kv_len + i);
#endif
#else
        out0 += A * float4(((const device ftype4*)(B0 + i))[0]);
        out1 += A * float4(((const device ftype4*)(B1 + i))[0]);
#endif
    }
    float r0 = out0.x + out0.y + out0.z + out0.w;
    float r1 = out1.x + out1.y + out1.z + out1.w;
    r0 = simd_sum(r0);
    r1 = simd_sum(r1);
    if(tiisg == 0) {
        int c0 = hn * head_dim + z;
        int co0 = c0 / 4;
        int ci0 = c0 % 4;
        output[(b * q_seq_len + x) * 4 + ci0 + co0 * param.batch * q_seq_len * 4] = (ftype)r0;
        if (z + 1 < head_dim) {
            int c1 = c0 + 1;
            int co1 = c1 / 4;
            int ci1 = c1 % 4;
            output[(b * q_seq_len + x) * 4 + ci1 + co1 * param.batch * q_seq_len * 4] = (ftype)r1;
        }
    }
}

)metal";

const char* gDecodeQkSoftmax = R"metal(
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
    int mask_batch;
    int mask_head_num;
    int mask_q_len;
    int mask_k_len;
    float v_scale;
    float k_scale;
};
#define SIMD_GROUP_WIDTH 32

kernel void decode_qk_softmax(const device ftype* input0 [[buffer(0)]],
    device ftype* output [[buffer(1)]],
    device ftype* past_key [[buffer(2)]],
    constant int &seq_idx [[buffer(3)]],
    constant Param& param [[buffer(4)]],
#if defined(QUANT_K) && defined(DYNAMIC_QUANT_K)
    device ftype* k_scales [[buffer(8)]],
#endif
    uint3 gid[[threadgroup_position_in_grid]],
    uint tid[[thread_index_in_threadgroup]],
    uint tiisg[[thread_index_in_simdgroup]],
    uint sgitg[[simdgroup_index_in_threadgroup]],
    uint3 tptg_3d[[threads_per_threadgroup]]
) {
    threadgroup float scores0[2048];
    threadgroup float scores1[2048];
    threadgroup float reduce0[32];
    threadgroup float reduce1[32];

    const int tptg = int(tptg_3d.x * tptg_3d.y * tptg_3d.z);
    const int sg_count = tptg / SIMD_GROUP_WIDTH;
    const int kv_head_num = param.head_num / GROUP_SIZE;
    const int b = int(gid.x) / kv_head_num;
    const int kv_hn = int(gid.x) - b * kv_head_num;
#ifdef HEAD_DIM
    const int head_dim = HEAD_DIM;
#else
    const int head_dim = param.head_dim;
#endif
    const int key_seq_len = param.key_seq_len;
    const int align_key_len = ((key_seq_len + param.kv_align_len - 1) / param.kv_align_len) * param.kv_align_len;
    const int x = int(gid.y);
    const int q_idx = seq_idx * param.q_seq_piece_len + x;

    if (b >= param.batch || kv_hn >= kv_head_num || x >= param.q_seq_piece_len || q_idx >= param.query_seq_len) {
        return;
    }

    const int head0 = kv_hn * GROUP_SIZE;
    const int head1 = head0 + 1;
    const int query_offset = (b * param.query_seq_len + q_idx) * param.head_num * head_dim;
    const device ftype* query0 = input0 + query_offset + head0 * head_dim;
    const device ftype* query1 = input0 + query_offset + head1 * head_dim;
    const int key_head_offset = kv_hn * head_dim;
    const int key_stride = kv_head_num * head_dim;

    float local_max0 = -FLT_MAX;
    float local_max1 = -FLT_MAX;
    const int kv_valid_limit = max(key_seq_len - param.query_seq_len, 0) + q_idx;
    for (int k = int(tid); k < key_seq_len; k += tptg) {
#ifdef QUANT_K
        const device char* key = (const device char*)past_key + (k * param.batch + b) * key_stride + key_head_offset;
#else
        const device ftype* key = past_key + (k * param.batch + b) * key_stride + key_head_offset;
#endif
        float s0 = 0.0f;
        float s1 = 0.0f;
        const device ftype4* q04 = (const device ftype4*)query0;
        const device ftype4* q14 = (const device ftype4*)query1;
#ifdef QUANT_K
        const device char4* k4 = (const device char4*)key;
#ifdef DYNAMIC_QUANT_K
        const int k_token_idx = k * param.batch + b;
        const float k_scale = float(k_scales[k_token_idx * 2]);
        const float k_bias = float(k_scales[k_token_idx * 2 + 1]);
#endif
#else
        const device ftype4* k4 = (const device ftype4*)key;
#endif
        for (int d = 0; d < head_dim / 8; ++d) {
#ifdef QUANT_K
#ifdef DYNAMIC_QUANT_K
            float4 k0 = float4(k4[d * 2 + 0]) * k_scale + k_bias;
            float4 k1 = float4(k4[d * 2 + 1]) * k_scale + k_bias;
#else
            float4 k0 = float4(k4[d * 2 + 0]) * param.k_scale;
            float4 k1 = float4(k4[d * 2 + 1]) * param.k_scale;
#endif
#else
            float4 k0 = float4(k4[d * 2 + 0]);
            float4 k1 = float4(k4[d * 2 + 1]);
#endif
            s0 += dot(float4(q04[d * 2 + 0]), k0) + dot(float4(q04[d * 2 + 1]), k1);
            s1 += dot(float4(q14[d * 2 + 0]), k0) + dot(float4(q14[d * 2 + 1]), k1);
        }
        s0 *= param.scale;
        s1 *= param.scale;
        if (k > kv_valid_limit) {
            s0 = -FLT_MAX;
            s1 = -FLT_MAX;
        }
        scores0[k] = s0;
        scores1[k] = s1;
        local_max0 = max(local_max0, s0);
        local_max1 = max(local_max1, s1);
    }

    local_max0 = simd_max(local_max0);
    local_max1 = simd_max(local_max1);
    if (tiisg == 0) {
        reduce0[sgitg] = local_max0;
        reduce1[sgitg] = local_max1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgitg == 0 && tiisg == 0) {
        float max0 = -FLT_MAX;
        float max1 = -FLT_MAX;
        for (int i = 0; i < sg_count; ++i) {
            max0 = max(max0, reduce0[i]);
            max1 = max(max1, reduce1[i]);
        }
        reduce0[0] = max0;
        reduce1[0] = max1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float max0 = reduce0[0];
    const float max1 = reduce1[0];

    float local_sum0 = 0.0f;
    float local_sum1 = 0.0f;
    for (int k = int(tid); k < key_seq_len; k += tptg) {
        float v0 = exp(scores0[k] - max0);
        float v1 = exp(scores1[k] - max1);
        scores0[k] = v0;
        scores1[k] = v1;
        local_sum0 += v0;
        local_sum1 += v1;
    }

    local_sum0 = simd_sum(local_sum0);
    local_sum1 = simd_sum(local_sum1);
    if (tiisg == 0) {
        reduce0[sgitg] = local_sum0;
        reduce1[sgitg] = local_sum1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgitg == 0 && tiisg == 0) {
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        for (int i = 0; i < sg_count; ++i) {
            sum0 += reduce0[i];
            sum1 += reduce1[i];
        }
        reduce0[0] = sum0;
        reduce1[0] = sum1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv_sum0 = 1.0f / reduce0[0];
    const float inv_sum1 = 1.0f / reduce1[0];

    const int base0 = ((b * param.head_num + head0) * param.query_seq_len + q_idx) * align_key_len;
    const int base1 = ((b * param.head_num + head1) * param.query_seq_len + q_idx) * align_key_len;
    for (int k = int(tid); k < key_seq_len; k += tptg) {
        output[base0 + k] = (ftype)(scores0[k] * inv_sum0);
        output[base1 + k] = (ftype)(scores1[k] * inv_sum1);
    }
    for (int k = int(tid) + key_seq_len; k < align_key_len; k += tptg) {
        output[base0 + k] = (ftype)0.0f;
        output[base1 + k] = (ftype)0.0f;
    }
}
)metal";

// softmax sg reduce source moved to MetalSoftmaxShader.cpp

#endif /* MNN_SUPPORT_TRANSFORMER_FUSE */
#endif
