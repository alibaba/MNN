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

static inline long attention_mask_offset(constant Param& param, int b, int hn, int q, int k) {
    int mask_b = param.mask_batch <= 1 ? 0 : b;
    int mask_h = param.mask_head_num <= 1 ? 0 : hn;
    int mask_q = param.mask_q_len <= 1 ? 0 : min(q, param.mask_q_len - 1);
    int mask_k_start = max(param.key_seq_len - param.mask_k_len, 0);
    int local_k = param.mask_k_len <= 1 ? 0 : clamp(k - mask_k_start, 0, param.mask_k_len - 1);
    // Return long: at seq_q=seq_k=500K with head-dim mask fields, the product
    // mask_q_len * mask_k_len already reaches 2.5e11 > INT32_MAX.
    return ((long(mask_b) * param.mask_head_num + mask_h) * param.mask_q_len + mask_q) * (long)param.mask_k_len + local_k;
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

#ifdef CAUSAL_TRI
    // Causal triangular dispatch for 32x32 tiles: mirror the trapezoid tile
    // count in the host (useTensorCausalTri branch, MetalAttention.mm). Decode
    // linear tile id gid.x back to (slq, slk). Same closed-form as the 16-tile
    // variant in prefill_qk with 32 substituted for 16.
    //   base = ceil(D, 32) + 1  is the number of k-tiles the top q-tile row
    //     needs to cover; row lq needs min(kt, lq + base) tiles.
    //   rows 0..r-1 form a triangle (v = lq + base, T1 = r*base + r*(r-1)/2);
    //   rows r..qt-1 each need the full kt tiles.
    int slq, slk;
    {
        int D = (param.key_seq_len - param.query_seq_len) + seq_idx * param.q_seq_piece_len - kv_start;
        int kt = (output_k_len + 31) >> 5;
        int qt = (param.q_seq_piece_len + 31) >> 5;
        int base = ((D + 31) >> 5) + 1;
        int r = clamp(kt - base + 1, 0, qt);
        int T1 = r * base + (r * (r - 1)) / 2;
        int t = (int)gid.x;
        if (t < T1) {
            // Solve S(lq) = lq*base + lq*(lq-1)/2 <= t < S(lq+1); float-precision
            // fix-up below covers the couple of borderline cases.
            float fb = (float)(2 * base - 1);
            int lq = (int)((sqrt(fb * fb + 8.0f * (float)t) - fb) * 0.5f);
            while (lq > 0 && lq * base + (lq * (lq - 1)) / 2 > t) { lq--; }
            while ((lq + 1) * base + ((lq + 1) * lq) / 2 <= t) { lq++; }
            slq = lq;
            slk = t - (lq * base + (lq * (lq - 1)) / 2);
        } else {
            int t2 = t - T1;
            slq = r + t2 / kt;
            slk = t2 % kt;
        }
    }
    const int z = gid.z;
#else
    const int slq = gid.x; // q_seq_len/32 -> M/32
    const int slk = gid.y; // k_seq_len/32 -> N/32
    const int z = gid.z; // head_num * batch
#endif

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

#ifdef CAUSAL_TRI
    // Whole-tile below the causal diagonal: mask is a no-op for every k in
    // this tile (max_k <= min-row-causal-bound), so skip all mask reads and
    // -FLT_MAX branches. Only diagonal-straddling tiles keep the per-element
    // masked path. Matches the prefill_qk shader's three-zone decomposition.
    const bool tileFullValid = (kv_start + slk * 32 + 31) <=
        ((k_seq_len - q_seq_len) + seq_idx * q_seq_piece_len + slq * 32);
#else
    const bool tileFullValid = false;
#endif

    int base_k_idx =  (slk * 4 + ncl) * 8 + 0;
    // Use long for the outer offset: at ~24K prompt, z=B*H up to 16 makes the
    // z*q_seq_piece_len*output_k_len product ~9.6e9 which overflows int32.
    auto xy_out = output + ((long)(z * q_seq_piece_len + slq * 32 + mcl)) * output_k_len + base_k_idx + 0;
    if(slq * 32 + mcl < q_seq_piece_len &&  seq_idx * q_seq_piece_len + slq * 32 + mcl < q_seq_len) {
        int ori_q_idx = seq_idx * q_seq_piece_len + slq * 32 + mcl;
        if(base_k_idx + 0 < output_k_len) {
            auto out0 =  ((threadgroup float*)sdata)[sindex_base + 0] * Vscale;
            #ifdef ADD_MASK
                if (!tileFullValid && attention_mask_hit(param, kv_start + base_k_idx + 0)) {
                    auto mask_val = mask[attention_mask_offset(param, b, hn, ori_q_idx, kv_start + base_k_idx + 0)];
                    out0 = mask_val + out0;
                }
            #elif defined(SET_MASK)
                if (!tileFullValid && attention_mask_hit(param, kv_start + base_k_idx + 0)) {
                    out0 = mask[attention_mask_offset(param, b, hn, ori_q_idx, kv_start + base_k_idx + 0)] == 0 ? -FLT_MAX : out0;
                }
            #elif defined(DEFAULT_MASK)
                int k_global = kv_start + base_k_idx + 0;
                if (!tileFullValid && k_global > kv_valid_offset + ori_q_idx) {
                    out0 = -FLT_MAX;
                }
            #endif
            xy_out[0] = out0;
        }
        if(base_k_idx + 1 < output_k_len) {
            auto out0 =  ((threadgroup float*)sdata)[sindex_base + 1] * Vscale;
            #ifdef ADD_MASK
                if (!tileFullValid && attention_mask_hit(param, kv_start + base_k_idx + 1)) {
                    auto mask_val = mask[attention_mask_offset(param, b, hn, ori_q_idx, kv_start + base_k_idx + 1)];
                    out0 = mask_val + out0;
                }
            #elif defined(SET_MASK)
                if (!tileFullValid && attention_mask_hit(param, kv_start + base_k_idx + 1)) {
                    out0 = mask[attention_mask_offset(param, b, hn, ori_q_idx, kv_start + base_k_idx + 1)] == 0 ? -FLT_MAX : out0;
                }
            #elif defined(DEFAULT_MASK)
                int k_global = kv_start + base_k_idx + 1;
                if (!tileFullValid && k_global > kv_valid_offset + ori_q_idx) {
                    out0 = -FLT_MAX;
                }
            #endif
            xy_out[1] = out0;
        }
        if(base_k_idx + 2 < output_k_len) {
            auto out0 =  ((threadgroup float*)sdata)[sindex_base + 2] * Vscale;
            #ifdef ADD_MASK
                if (!tileFullValid && attention_mask_hit(param, kv_start + base_k_idx + 2)) {
                    auto mask_val = mask[attention_mask_offset(param, b, hn, ori_q_idx, kv_start + base_k_idx + 2)];
                    out0 = mask_val + out0;
                }
            #elif defined(SET_MASK)
                if (!tileFullValid && attention_mask_hit(param, kv_start + base_k_idx + 2)) {
                    out0 = mask[attention_mask_offset(param, b, hn, ori_q_idx, kv_start + base_k_idx + 2)] == 0 ? -FLT_MAX : out0;
                }
            #elif defined(DEFAULT_MASK)
                int k_global = kv_start + base_k_idx + 2;
                if (!tileFullValid && k_global > kv_valid_offset + ori_q_idx) {
                    out0 = -FLT_MAX;
                }
            #endif
            xy_out[2] = out0;
        }
        if(base_k_idx + 3 < output_k_len) {
            auto out0 =  ((threadgroup float*)sdata)[sindex_base + 3] * Vscale;
            #ifdef ADD_MASK
                if (!tileFullValid && attention_mask_hit(param, kv_start + base_k_idx + 3)) {
                    auto mask_val = mask[attention_mask_offset(param, b, hn, ori_q_idx, kv_start + base_k_idx + 3)];
                    out0 = mask_val + out0;
                }
            #elif defined(SET_MASK)
                if (!tileFullValid && attention_mask_hit(param, kv_start + base_k_idx + 3)) {
                    out0 = mask[attention_mask_offset(param, b, hn, ori_q_idx, kv_start + base_k_idx + 3)] == 0 ? -FLT_MAX : out0;
                }
            #elif defined(DEFAULT_MASK)
                int k_global = kv_start + base_k_idx + 3;
                if (!tileFullValid && k_global > kv_valid_offset + ori_q_idx) {
                    out0 = -FLT_MAX;
                }
            #endif
            xy_out[3] = out0;
        }
        if(base_k_idx + 4 < output_k_len) {
            auto out0 =  ((threadgroup float*)sdata)[sindex_base + 4] * Vscale;
            #ifdef ADD_MASK
                if (!tileFullValid && attention_mask_hit(param, kv_start + base_k_idx + 4)) {
                    auto mask_val = mask[attention_mask_offset(param, b, hn, ori_q_idx, kv_start + base_k_idx + 4)];
                    out0 = mask_val + out0;
                }
            #elif defined(SET_MASK)
                if (!tileFullValid && attention_mask_hit(param, kv_start + base_k_idx + 4)) {
                    out0 = mask[attention_mask_offset(param, b, hn, ori_q_idx, kv_start + base_k_idx + 4)] == 0 ? -FLT_MAX : out0;
                }
            #elif defined(DEFAULT_MASK)
                int k_global = kv_start + base_k_idx + 4;
                if (!tileFullValid && k_global > kv_valid_offset + ori_q_idx) {
                    out0 = -FLT_MAX;
                }
            #endif
            xy_out[4] = out0;
        }
        if(base_k_idx + 5 < output_k_len) {
            auto out0 =  ((threadgroup float*)sdata)[sindex_base + 5] * Vscale;
            #ifdef ADD_MASK
                if (!tileFullValid && attention_mask_hit(param, kv_start + base_k_idx + 5)) {
                    auto mask_val = mask[attention_mask_offset(param, b, hn, ori_q_idx, kv_start + base_k_idx + 5)];
                    out0 = mask_val + out0;
                }
            #elif defined(SET_MASK)
                if (!tileFullValid && attention_mask_hit(param, kv_start + base_k_idx + 5)) {
                    out0 = mask[attention_mask_offset(param, b, hn, ori_q_idx, kv_start + base_k_idx + 5)] == 0 ? -FLT_MAX : out0;
                }
            #elif defined(DEFAULT_MASK)
                int k_global = kv_start + base_k_idx + 5;
                if (!tileFullValid && k_global > kv_valid_offset + ori_q_idx) {
                    out0 = -FLT_MAX;
                }
            #endif
            xy_out[5] = out0;
        }
        if(base_k_idx + 6 < output_k_len) {
            auto out0 =  ((threadgroup float*)sdata)[sindex_base + 6] * Vscale;
            #ifdef ADD_MASK
                if (!tileFullValid && attention_mask_hit(param, kv_start + base_k_idx + 6)) {
                    auto mask_val = mask[attention_mask_offset(param, b, hn, ori_q_idx, kv_start + base_k_idx + 6)];
                    out0 = mask_val + out0;
                }
            #elif defined(SET_MASK)
                if (!tileFullValid && attention_mask_hit(param, kv_start + base_k_idx + 6)) {
                    out0 = mask[attention_mask_offset(param, b, hn, ori_q_idx, kv_start + base_k_idx + 6)] == 0 ? -FLT_MAX : out0;
                }
            #elif defined(DEFAULT_MASK)
                int k_global = kv_start + base_k_idx + 6;
                if (!tileFullValid && k_global > kv_valid_offset + ori_q_idx) {
                    out0 = -FLT_MAX;
                }
            #endif
            xy_out[6] = out0;
        }
        if(base_k_idx + 7 < output_k_len) {
            auto out0 =  ((threadgroup float*)sdata)[sindex_base + 7] * Vscale;
            #ifdef ADD_MASK
                if (!tileFullValid && attention_mask_hit(param, kv_start + base_k_idx + 7)) {
                    auto mask_val = mask[attention_mask_offset(param, b, hn, ori_q_idx, kv_start + base_k_idx + 7)];
                    out0 = mask_val + out0;
                }
            #elif defined(SET_MASK)
                if (!tileFullValid && attention_mask_hit(param, kv_start + base_k_idx + 7)) {
                    out0 = mask[attention_mask_offset(param, b, hn, ori_q_idx, kv_start + base_k_idx + 7)] == 0 ? -FLT_MAX : out0;
                }
            #elif defined(DEFAULT_MASK)
                int k_global = kv_start + base_k_idx + 7;
                if (!tileFullValid && k_global > kv_valid_offset + ori_q_idx) {
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

// Tensor prefill uses the dedicated 32x32 kernel above. Keep this legacy
// 16x16x8 implementation disabled because recent Tensor API versions require
// a fixed K dimension to be a multiple of 16.
#ifdef MNN_METAL_TENSOR_OPS_LEGACY_8X8

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

#ifdef CAUSAL_TRI
    // Causal triangular dispatch: the host launches only the trapezoid of tiles
    // at or below the causal diagonal (grid.x = total trapezoid tile count,
    // see the mirrored formula in MetalAttention.mm). Decode the linear tile id
    // back to (slq, slk). Row-tile lq covers v(lq) = min(kt, lq + base) k-tiles:
    //   rows 0..r-1 form a triangle (v = lq + base), rows >= r are full kt rows.
    int slq, slk;
    {
        int D = (param.key_seq_len - param.query_seq_len) + seq_idx * param.q_seq_piece_len - kv_start;
        int kt = (output_k_len + 15) >> 4;
        int qt = (param.q_seq_piece_len + 15) >> 4;
        int base = ((D + 15) >> 4) + 1;
        int r = clamp(kt - base + 1, 0, qt);
        int T1 = r * base + (r * (r - 1)) / 2;
        int t = (int)gid.x;
        if (t < T1) {
            // solve lq: S(lq) = lq*base + lq*(lq-1)/2 <= t < S(lq+1)
            float fb = (float)(2 * base - 1);
            int lq = (int)((sqrt(fb * fb + 8.0f * (float)t) - fb) * 0.5f);
            // float-precision fix-up (at most a couple of steps)
            while (lq > 0 && lq * base + (lq * (lq - 1)) / 2 > t) { lq--; }
            while ((lq + 1) * base + ((lq + 1) * lq) / 2 <= t) { lq++; }
            slq = lq;
            slk = t - (lq * base + (lq * (lq - 1)) / 2);
        } else {
            int t2 = t - T1;
            slq = r + t2 / kt;
            slk = t2 % kt;
        }
    }
#else
    const int slq = gid.x; // q_seq_len/16 -> M/16
    const int slk = gid.y; // k_seq_len/16 -> N/16
#endif
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

#if !defined(CAUSAL_TRI) && (defined(DEFAULT_MASK) || defined(ADD_MASK) || defined(SET_MASK))
    // Causal skip: if this M16xN16 tile lies entirely above the diagonal in the
    // causal-mask sense, the whole tile ends up as -FLT_MAX after masking anyway.
    // Write -FLT_MAX directly and exit to save all the QK matmul work on the upper
    // triangle (~50% of tiles in a square prefill grid).
    // (With CAUSAL_TRI the host never launches these tiles and the CAUSAL_BOUND
    // softmax never reads the upper region, so this check is compiled out.)
    //
    // Assumption: the mask provided by the LLM engine is causal-lower-triangular
    // (which is always the case for standard causal LLM prefill).  For non-causal
    // custom masks this optimization would over-mask.
    {
        int tile_min_k_global = kv_start + slk * 16;
        int tile_max_q_absolute = (k_seq_len - q_seq_len) + seq_idx * q_seq_piece_len + slq * 16 + 15;
        if (tile_min_k_global > tile_max_q_absolute) {
            auto xy_out_skip = output + ((long)(z * q_seq_piece_len + slq * 16 + rcl)) * output_k_len + slk * 16 + kl * 8;
            if (slq * 16 + rcl < q_seq_piece_len && seq_idx * q_seq_piece_len + slq * 16 + rcl < q_seq_len) {
                for (int j = 0; j < 8; ++j) {
                    if (slk * 16 + kl * 8 + j < output_k_len) {
                        xy_out_skip[j] = (ftype)(-FLT_MAX);
                    }
                }
            }
            return;
        }
    }
#endif

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

#ifdef MNN_METAL_TENSOR_OPS_LEGACY_8X8
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

#ifdef MNN_METAL_TENSOR_OPS_LEGACY_8X8

    auto tC = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>((threadgroup float*)sdata, dextents<int32_t, 2>(N, M)); // [M , N]
    cT.store(tC);
#else
    simdgroup_store(sgd[0], (threadgroup float*)sdata, 8);
    simdgroup_store(sgd[1], (threadgroup float*)sdata + 64, 8);
    simdgroup_store(sgd[2], (threadgroup float*)sdata + 128, 8);
    simdgroup_store(sgd[3], (threadgroup float*)sdata + 192, 8);
#endif

    threadgroup_barrier(mem_flags::mem_threadgroup);

#ifdef MNN_METAL_TENSOR_OPS_LEGACY_8X8
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

#ifdef CAUSAL_TRI
    // Three-zone decomposition: a tile entirely at or below the causal diagonal
    // (max k of tile <= causal bound of its min q row) is fully valid — the mask
    // is a no-op there (DEFAULT_MASK never fires; causal ADD/SET masks are
    // 0/pass in the valid region), so skip the per-element mask reads/branches.
    // Only the O(T) diagonal-straddling tiles keep the masked path.
    const bool tileFullValid = (kv_start + slk * 16 + 15) <=
        ((k_seq_len - q_seq_len) + seq_idx * q_seq_piece_len + slq * 16);
#else
    const bool tileFullValid = false;
#endif

    auto xy_out = output + ((long)(z * q_seq_piece_len + slq * 16 + rcl)) * output_k_len + slk * 16 + kl * 8 + 0;
    if(slq * 16 + rcl < q_seq_piece_len &&  seq_idx * q_seq_piece_len + slq * 16 + rcl < q_seq_len) {
        int ori_q_idx = seq_idx * q_seq_piece_len + slq * 16 + rcl;
        if(slk * 16 + kl * 8 + 0 < output_k_len) {
            auto out0 =  ((threadgroup float*)sdata)[sindex_base + 0] * Vscale;
            #ifdef ADD_MASK
                if (!tileFullValid && attention_mask_hit(param, kv_start + slk * 16 + kl * 8 + 0)) {
                    auto mask_val = mask[attention_mask_offset(param, b, hn, ori_q_idx, kv_start + slk * 16 + kl * 8 + 0)];
                    out0 = mask_val + out0;
                }
            #elif defined(SET_MASK)
                if (!tileFullValid && attention_mask_hit(param, kv_start + slk * 16 + kl * 8 + 0)) {
                    out0 = mask[attention_mask_offset(param, b, hn, ori_q_idx, kv_start + slk * 16 + kl * 8 + 0)] == 0 ? -FLT_MAX : out0;
                }
            #elif defined(DEFAULT_MASK)
                int k_global = kv_start + slk * 16 + kl * 8 + 0;
                if (!tileFullValid && k_global > kv_valid_offset + ori_q_idx) {
                    out0 = -FLT_MAX;
                }
            #endif
            xy_out[0] = out0;
        }
        if(slk * 16 + kl * 8 + 1 < output_k_len) {
            auto out0 =  ((threadgroup float*)sdata)[sindex_base + 1] * Vscale;
            #ifdef ADD_MASK
                if (!tileFullValid && attention_mask_hit(param, kv_start + slk * 16 + kl * 8 + 1)) {
                    auto mask_val = mask[attention_mask_offset(param, b, hn, ori_q_idx, kv_start + slk * 16 + kl * 8 + 1)];
                    out0 = mask_val + out0;
                }
            #elif defined(SET_MASK)
                if (!tileFullValid && attention_mask_hit(param, kv_start + slk * 16 + kl * 8 + 1)) {
                    out0 = mask[attention_mask_offset(param, b, hn, ori_q_idx, kv_start + slk * 16 + kl * 8 + 1)] == 0 ? -FLT_MAX : out0;
                }
            #elif defined(DEFAULT_MASK)
                int k_global = kv_start + slk * 16 + kl * 8 + 1;
                if (!tileFullValid && k_global > kv_valid_offset + ori_q_idx) {
                    out0 = -FLT_MAX;
                }
            #endif
            xy_out[1] = out0;
        }
        if(slk * 16 + kl * 8 + 2 < output_k_len) {
            auto out0 =  ((threadgroup float*)sdata)[sindex_base + 2] * Vscale;
            #ifdef ADD_MASK
                if (!tileFullValid && attention_mask_hit(param, kv_start + slk * 16 + kl * 8 + 2)) {
                    auto mask_val = mask[attention_mask_offset(param, b, hn, ori_q_idx, kv_start + slk * 16 + kl * 8 + 2)];
                    out0 = mask_val + out0;
                }
            #elif defined(SET_MASK)
                if (!tileFullValid && attention_mask_hit(param, kv_start + slk * 16 + kl * 8 + 2)) {
                    out0 = mask[attention_mask_offset(param, b, hn, ori_q_idx, kv_start + slk * 16 + kl * 8 + 2)] == 0 ? -FLT_MAX : out0;
                }
            #elif defined(DEFAULT_MASK)
                int k_global = kv_start + slk * 16 + kl * 8 + 2;
                if (!tileFullValid && k_global > kv_valid_offset + ori_q_idx) {
                    out0 = -FLT_MAX;
                }
            #endif
            xy_out[2] = out0;
        }
        if(slk * 16 + kl * 8 + 3 < output_k_len) {
            auto out0 =  ((threadgroup float*)sdata)[sindex_base + 3] * Vscale;
            #ifdef ADD_MASK
                if (!tileFullValid && attention_mask_hit(param, kv_start + slk * 16 + kl * 8 + 3)) {
                    auto mask_val = mask[attention_mask_offset(param, b, hn, ori_q_idx, kv_start + slk * 16 + kl * 8 + 3)];
                    out0 = mask_val + out0;
                }
            #elif defined(SET_MASK)
                if (!tileFullValid && attention_mask_hit(param, kv_start + slk * 16 + kl * 8 + 3)) {
                    out0 = mask[attention_mask_offset(param, b, hn, ori_q_idx, kv_start + slk * 16 + kl * 8 + 3)] == 0 ? -FLT_MAX : out0;
                }
            #elif defined(DEFAULT_MASK)
                int k_global = kv_start + slk * 16 + kl * 8 + 3;
                if (!tileFullValid && k_global > kv_valid_offset + ori_q_idx) {
                    out0 = -FLT_MAX;
                }
            #endif
            xy_out[3] = out0;
        }
        if(slk * 16 + kl * 8 + 4 < output_k_len) {
            auto out0 =  ((threadgroup float*)sdata)[sindex_base + 4] * Vscale;
            #ifdef ADD_MASK
                if (!tileFullValid && attention_mask_hit(param, kv_start + slk * 16 + kl * 8 + 4)) {
                    auto mask_val = mask[attention_mask_offset(param, b, hn, ori_q_idx, kv_start + slk * 16 + kl * 8 + 4)];
                    out0 = mask_val + out0;
                }
            #elif defined(SET_MASK)
                if (!tileFullValid && attention_mask_hit(param, kv_start + slk * 16 + kl * 8 + 4)) {
                    out0 = mask[attention_mask_offset(param, b, hn, ori_q_idx, kv_start + slk * 16 + kl * 8 + 4)] == 0 ? -FLT_MAX : out0;
                }
            #elif defined(DEFAULT_MASK)
                int k_global = kv_start + slk * 16 + kl * 8 + 4;
                if (!tileFullValid && k_global > kv_valid_offset + ori_q_idx) {
                    out0 = -FLT_MAX;
                }
            #endif
            xy_out[4] = out0;
        }
        if(slk * 16 + kl * 8 + 5 < output_k_len) {
            auto out0 =  ((threadgroup float*)sdata)[sindex_base + 5] * Vscale;
            #ifdef ADD_MASK
                if (!tileFullValid && attention_mask_hit(param, kv_start + slk * 16 + kl * 8 + 5)) {
                    auto mask_val = mask[attention_mask_offset(param, b, hn, ori_q_idx, kv_start + slk * 16 + kl * 8 + 5)];
                    out0 = mask_val + out0;
                }
            #elif defined(SET_MASK)
                if (!tileFullValid && attention_mask_hit(param, kv_start + slk * 16 + kl * 8 + 5)) {
                    out0 = mask[attention_mask_offset(param, b, hn, ori_q_idx, kv_start + slk * 16 + kl * 8 + 5)] == 0 ? -FLT_MAX : out0;
                }
            #elif defined(DEFAULT_MASK)
                int k_global = kv_start + slk * 16 + kl * 8 + 5;
                if (!tileFullValid && k_global > kv_valid_offset + ori_q_idx) {
                    out0 = -FLT_MAX;
                }
            #endif
            xy_out[5] = out0;
        }
        if(slk * 16 + kl * 8 + 6 < output_k_len) {
            auto out0 =  ((threadgroup float*)sdata)[sindex_base + 6] * Vscale;
            #ifdef ADD_MASK
                if (!tileFullValid && attention_mask_hit(param, kv_start + slk * 16 + kl * 8 + 6)) {
                    auto mask_val = mask[attention_mask_offset(param, b, hn, ori_q_idx, kv_start + slk * 16 + kl * 8 + 6)];
                    out0 = mask_val + out0;
                }
            #elif defined(SET_MASK)
                if (!tileFullValid && attention_mask_hit(param, kv_start + slk * 16 + kl * 8 + 6)) {
                    out0 = mask[attention_mask_offset(param, b, hn, ori_q_idx, kv_start + slk * 16 + kl * 8 + 6)] == 0 ? -FLT_MAX : out0;
                }
            #elif defined(DEFAULT_MASK)
                int k_global = kv_start + slk * 16 + kl * 8 + 6;
                if (!tileFullValid && k_global > kv_valid_offset + ori_q_idx) {
                    out0 = -FLT_MAX;
                }
            #endif
            xy_out[6] = out0;
        }
        if(slk * 16 + kl * 8 + 7 < output_k_len) {
            auto out0 =  ((threadgroup float*)sdata)[sindex_base + 7] * Vscale;
            #ifdef ADD_MASK
                if (!tileFullValid && attention_mask_hit(param, kv_start + slk * 16 + kl * 8 + 7)) {
                    auto mask_val = mask[attention_mask_offset(param, b, hn, ori_q_idx, kv_start + slk * 16 + kl * 8 + 7)];
                    out0 = mask_val + out0;
                }
            #elif defined(SET_MASK)
                if (!tileFullValid && attention_mask_hit(param, kv_start + slk * 16 + kl * 8 + 7)) {
                    out0 = mask[attention_mask_offset(param, b, hn, ori_q_idx, kv_start + slk * 16 + kl * 8 + 7)] == 0 ? -FLT_MAX : out0;
                }
            #elif defined(DEFAULT_MASK)
                int k_global = kv_start + slk * 16 + kl * 8 + 7;
                if (!tileFullValid && k_global > kv_valid_offset + ori_q_idx) {
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
        // Tail: 4-element remainder (head_dim % 8 == 4).  Only lane 0 accumulates;
        // simd_sum below still returns the correct total.  Covers head_dim=4 case
        // where the main loop above never executes.
        if ((head_dim & 4) != 0 && int(tiisg) == 0) {
            int tail_i4 = itN * 2;  // float4 index for the trailing 4-element chunk
#ifdef QUANT_K
#ifdef DYNAMIC_QUANT_K
            float4 B0 = float4(((const device char4*)Pastkey_offset)[tail_i4]) * k_scale + k_bias;
#else
            float4 B0 = GETK4(((const device char4*)Pastkey_offset)[tail_i4], z * param.batch + b);
#endif
#else
            float4 B0 = float4(((const device ftype4*)Pastkey_offset)[tail_i4]);
#endif
            for (int j = 0; j < group; j++) {
                const device ftype4* Ajp = (const device ftype4*)(A_offset + head_dim * j);
                float4 A0 = float4(Ajp[tail_i4]);
                out[j] += dot(A0, B0);
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
        // Tail: 4-element remainder (head_dim % 8 == 4). Covers head_dim=4 case.
        if ((head_dim & 4) != 0) {
            int tail_i4 = itN * 2;
#ifdef QUANT_K
#ifdef DYNAMIC_QUANT_K
            float4 B0 = float4(((const device char4*)Pastkey_offset)[tail_i4]) * k_scale + k_bias;
#else
            float4 B0 = GETK4(((const device char4*)Pastkey_offset)[tail_i4], z * param.batch + b);
#endif
#else
            float4 B0 = float4(((const device ftype4*)Pastkey_offset)[tail_i4]);
#endif
            for (int j = 0; j < group; j++) {
                const device ftype4* Ajp = (const device ftype4*)(A_offset + head_dim * j);
                float4 A0 = float4(Ajp[tail_i4]);
                out[j] += dot(A0, B0);
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
    int value_c4;
    float v_scale;
    float k_scale;
};
// Key:   [batch, kv_seq_len, head_num / group * head_dim] -> [max_kv_len, batch, head_num / group * head_dim]
// Value: [batch, kv_seq_len, head_num / group * head_dim] -> [max_kv_len, batch, head_num / group * head_dim]

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

static inline int value_c4_offset(int token, int channel, int seq_len) {
    return (channel / 4) * seq_len * 4 + token * 4 + (channel % 4);
}

static inline ftype load_value(const device ftype* input, constant Param& param, int b, int y, int x) {
    if (param.value_c4 != 0) {
        int token = b * param.kv_seq_len + y;
        return input[value_c4_offset(token, x, param.kv_seq_len * param.batch)];
    }
    return input[(b * param.kv_seq_len + y) * param.head_count + x];
}

static inline ftype4 load_value4(const device ftype* input, constant Param& param, int b, int y, int x) {
    if (param.value_c4 != 0) {
        int token = b * param.kv_seq_len + y;
        return ((const device ftype4*)(input + value_c4_offset(token, x, param.kv_seq_len * param.batch)))[0];
    }
    return ((const device ftype4*)(input + (b * param.kv_seq_len + y) * param.head_count + x))[0];
}


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
            float4 v4 = float4(load_value4(input1, param, b, y, x));
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
            float v = (float)load_value(input1, param, b, y, x);
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

        // Write V (row-major, same layout as K)
        int out_idx_v = param.dst_v_offset + (y * param.batch + b) * param.head_count + x;
        ftype4 v4 = load_value4(input1, param, b, y, x);
#ifdef KV_QUANT_V
        float4 v = float4(v4);
        if (v_scale == 0.0f) {
            ((device char4*)(output1 + out_idx_v))[0] = char4(0);
        } else {
            int4 qi = int4(rint((v - v_bias) / v_scale));
            qi = clamp(qi, int4(-128), int4(127));
            ((device char4*)(output1 + out_idx_v))[0] = char4(qi);
        }
#else
        ((device ftype4*)(output1 + out_idx_v))[0] = v4;
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

        int out_idx_v = param.dst_v_offset + (y * param.batch + b) * param.head_count + x;
        ftype value = load_value(input1, param, b, y, x);
#ifdef KV_QUANT_V
        float v = (float)value;
        if (v_scale == 0.0f) {
            output1[out_idx_v] = (char)0;
        } else {
            float q = (v - v_bias) / v_scale;
            int qi = (int)rint(q);
            qi = clamp(qi, -128, 127);
            output1[out_idx_v] = (char)qi;
        }
#else
        output1[out_idx_v] = value;
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

    int out_idx_v = param.dst_v_offset + (y * param.batch + b) * param.head_count + x;
    ftype value = load_value(input1, param, b, y, x);
#ifdef KV_QUANT_V
    float v = (float)value;
    if (param.v_scale == 0.0f) {
        output1[out_idx_v] = (char)0;
    } else {
        float q = v / param.v_scale;
        int qi = (int)rint(q);
        qi = clamp(qi, -128, 127);
        output1[out_idx_v] = (char)qi;
    }
#else
    output1[out_idx_v] = value;
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
#else
#define GETV(v, tok_idx) ftype((float(v) * param.v_scale))
#endif
#else
#define GETV(v, tok_idx) v
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
    auto tB = tensor<threadgroup ftype, dextents<int32_t, 2>, tensor_inline>((threadgroup ftype*)sdata + tb_offset, dextents<int32_t, 2>(K, N));//[K, N]

    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(M, N, K, false, false, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
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
     threadgroup: [K32, N32] -> [K32, N4, N8]
     index; [nl(kv token), kvl(d group), N8]
     each thread: N8 (8 contiguous d of kv token i*4+nl)
     layout: [K(max_kv), B, KVHead, N] row-major, d contiguous
     offset: (((kv * batch + b) * kv_head_num + kh) * head_dim + hm * 32 + kvl * 8)
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
    int kv_head_num = head_num / group;
    int kh = hn / group;

    int idx_qk_sl = sl * 32 + ml < q_seq_piece_len ? (sl * 32 + ml) : q_seq_piece_len - 1;

    auto A_offset = input0 + (long)(z * q_seq_piece_len + idx_qk_sl) * align_value_len + (0 * 4 + kl) * 8 + 0;

    // AV causal early-exit for the tensor-API tile (M=32, K=32). Each iteration
    // loads 8 ftype4 = 32 scalar K, so av_k_upper_v4 must be a multiple of 8.
    // The upper bound is picked as ceil-32(tile_max_valid), which is:
    //   * >= tile_max_valid_len (all live rows in the tile see their full valid K)
    //   * <= softmax's per-row pad_end for every row in the tile (softmax pads
    //     to ceil-32(valid_len + 32); see MetalSoftmaxShader.cpp CAUSAL_BOUND
    //     branch), so the tile never reads memory that softmax did not write.
    // v_k_upper_v4 is 32-scalar-aligned by construction (32 / 4 = 8 v4).
    int av_k_upper_v4 = (value_seq_len + 3) / 4;
#ifdef CAUSAL_BOUND
    {
        int kv_valid_offset = value_seq_len - q_seq_len;
        // tile M=32: the tile-max q_abs (unclipped; the store-side guard at
        // line ~1557 already handles q_abs >= q_seq_len rows).
        int tile_max_q_abs = kv_valid_offset + seq_idx * q_seq_piece_len + sl * 32 + 31;
        int tile_max_valid = tile_max_q_abs + 1;
        if (tile_max_valid < 0) tile_max_valid = 0;
        int k_bound_scalar = ((tile_max_valid + 31) / 32) * 32;   // ceil to 32-scalar
        if (k_bound_scalar > align_value_len) k_bound_scalar = align_value_len;
        av_k_upper_v4 = k_bound_scalar / 4;                         // exact: 32-align / 4 = 8-align v4
    }
#endif

    for(int i = 0; i < av_k_upper_v4; i += 8){
        // 向量化写入 P（两次 ftype4，覆盖 8 个标量）
        *((threadgroup ftype4*)(&((threadgroup ftype*)sdata)[(ml * 4 + kl) * 8 + 0])) = *((const device ftype4*)(&A_offset[4*i + 0]));
        *((threadgroup ftype4*)(&((threadgroup ftype*)sdata)[(ml * 4 + kl) * 8 + 4])) = *((const device ftype4*)(&A_offset[4*i + 4]));

        // row-major V: thread (nl, kvl) loads 8 contiguous d of kv token i*4+nl
        {
            const int kv_tok = i * 4 + nl;
            const long v_off = ((long)(kv_tok * param.batch + b) * kv_head_num + kh) * head_dim + hm * 32 + kvl * 8;
#ifdef QUANT_V
            const int tok_idx = b * param.max_kv_len + kv_tok;
            char4 r0 = ((const device char4*)((const device char*)past_value + v_off))[0];
            char4 r1 = ((const device char4*)((const device char*)past_value + v_off))[1];
            ((threadgroup ftype4*)sdata)[256 + nl * 8 + kvl * 2 + 0] =
                ftype4(GETV(r0.x, tok_idx), GETV(r0.y, tok_idx), GETV(r0.z, tok_idx), GETV(r0.w, tok_idx));
            ((threadgroup ftype4*)sdata)[256 + nl * 8 + kvl * 2 + 1] =
                ftype4(GETV(r1.x, tok_idx), GETV(r1.y, tok_idx), GETV(r1.z, tok_idx), GETV(r1.w, tok_idx));
#else
            ((threadgroup ftype4*)sdata)[256 + nl * 8 + kvl * 2 + 0] = ((const device ftype4*)(past_value + v_off))[0];
            ((threadgroup ftype4*)sdata)[256 + nl * 8 + kvl * 2 + 1] = ((const device ftype4*)(past_value + v_off))[1];
#endif
        }

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

    // Output layout write. Two paths depending on the exported model:
    //   * default: [mBatch, mSeqLen, mNumHead, mHeadDim] as ftype4* stride 1
    //     (each ftype4 packs 4 consecutive d-lane values).
    //   * ATTENTION_C4: [mNumHead * (mHeadDim/4), mBatch * mSeqLen, 4]
    //     — this matches Qwen3 c4-head exports; without this branch the
    //     tensor-API kernel wrote to the default layout while the rest of
    //     the graph assumed C4, producing garbage tokens.
    int d_group0 = (hm * 4 + ncl) * 2 + 0;
    int d_group1 = (hm * 4 + ncl) * 2 + 1;
    int q_abs = seq_idx * q_seq_piece_len + sl * 32 + mcl;
    if(sl * 32 + mcl < q_seq_piece_len && q_abs < q_seq_len) {
#ifdef ATTENTION_C4
        long c4_middle = (long)(b * q_seq_len + q_abs);
        long c4_stride = (long)param.batch * (long)q_seq_len;
        if(d_group0 < head_dim/4) {
            output[(long)(hn * (head_dim/4) + d_group0) * c4_stride + c4_middle] =
                ftype4(((threadgroup float4*)sdata)[sindex_base + 0]);
        }
        if(d_group1 < head_dim/4) {
            output[(long)(hn * (head_dim/4) + d_group1) * c4_stride + c4_middle] =
                ftype4(((threadgroup float4*)sdata)[sindex_base + 1]);
        }
#else
        auto xy_out = output + ((long)((b * q_seq_len + q_abs) * head_num + hn)) * head_dim/4
                    + d_group0;
        if(d_group0 < head_dim/4) {
            xy_out[0] = ftype4(((threadgroup float4*)sdata)[sindex_base + 0]);
        }
        if(d_group1 < head_dim/4) {
            xy_out[1] = ftype4(((threadgroup float4*)sdata)[sindex_base + 1]);
        }
#endif
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

// Tensor prefill uses prefill_qkv_tensor (32x32x32). Do not instantiate the
// legacy 16x16x8 path when compiling that pipeline.
#ifdef MNN_METAL_TENSOR_OPS_LEGACY_8X8

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
     each thread: N4 (4 contiguous d of kv token i+kcl)
     layout: [K(max_kv), B, KVHead, N] row-major, d contiguous
     offset: (((kv * batch + b) * kv_head_num + kh) * head_dim + hm * 16 + nl * 4)
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
    int kv_head_num = head_num / group;
    int kh = hn / group;

    int idx_qk_sl = sl * 16 + rcl < q_seq_piece_len ? (sl * 16 + rcl) : q_seq_piece_len - 1;

    auto A_offset = input0 + (long)(z * q_seq_piece_len + idx_qk_sl) * align_value_len + (0 * 2 + kl) * 4 + 0;

    // Causal skip for AV matmul: because softmax output beyond causal k_max is 0
    // (thanks to -FLT_MAX in QK+softmax), we can stop accumulating early.  Each
    // M16 q-tile has a maximum allowed k = kv_valid_offset + tile_max_q_absolute.
    // Round up to K=8 alignment.  For a square prefill this halves K iterations
    // on average and gives up to 4x speedup for early-q tiles.
    int av_k_upper = align_value_len;
#if defined(DEFAULT_MASK) || defined(ADD_MASK) || defined(SET_MASK) || defined(CAUSAL_BOUND)
    {
        int kv_valid_offset = value_seq_len - q_seq_len;
        int tile_max_q_abs = kv_valid_offset + seq_idx * q_seq_piece_len + sl * 16 + 15;
        int k_bound = tile_max_q_abs + 1;                  // exclusive
        if (k_bound < 0) k_bound = 0;
        int k_bound_aligned = ((k_bound + 7) / 8) * 8;     // round up to K=8 stride
        if (k_bound_aligned < av_k_upper) av_k_upper = k_bound_aligned;
    }
#endif

    for(int i = 0; i < av_k_upper; i += 8){
        *((threadgroup ftype4*)(&((threadgroup ftype*)sdata)[rcl * 8 + kl * 4 + 0])) = *((const device ftype4*)(&A_offset[i + 0]));

        // row-major V: 4 contiguous d of kv token i+kcl
        {
            const int kv_tok = i + kcl;
            const long v_off = ((long)(kv_tok * param.batch + b) * kv_head_num + kh) * head_dim + hm * 16 + nl * 4;
#ifdef QUANT_V
            const int tok_idx = b * param.max_kv_len + kv_tok;
            char4 r = ((const device char4*)((const device char*)past_value + v_off))[0];
            ((threadgroup ftype*)sdata)[128 + kcl * 16 + nl * 4 + 0] = GETV(r.x, tok_idx);
            ((threadgroup ftype*)sdata)[128 + kcl * 16 + nl * 4 + 1] = GETV(r.y, tok_idx);
            ((threadgroup ftype*)sdata)[128 + kcl * 16 + nl * 4 + 2] = GETV(r.z, tok_idx);
            ((threadgroup ftype*)sdata)[128 + kcl * 16 + nl * 4 + 3] = GETV(r.w, tok_idx);
#else
            ((threadgroup ftype4*)((threadgroup ftype*)sdata + 128 + kcl * 16 + nl * 4))[0] =
                ((const device ftype4*)(past_value + v_off))[0];
#endif
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

#ifdef MNN_METAL_TENSOR_OPS_LEGACY_8X8
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

#ifdef MNN_METAL_TENSOR_OPS_LEGACY_8X8

    auto tC = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>((threadgroup float*)sdata, dextents<int32_t, 2>(N, M)); // [M , N]
    cT.store(tC);
#else
    simdgroup_store(sgd[0], (threadgroup float*)sdata, 8);
    simdgroup_store(sgd[1], (threadgroup float*)sdata + 64, 8);
    simdgroup_store(sgd[2], (threadgroup float*)sdata + 128, 8);
    simdgroup_store(sgd[3], (threadgroup float*)sdata + 192, 8);
#endif

    threadgroup_barrier(mem_flags::mem_threadgroup);

#ifdef MNN_METAL_TENSOR_OPS_LEGACY_8X8
    // [M16, N2, N8]
    auto sindex_base = (rcl * 2 + kl) * 8 + 0;
#else
    // [N2, M2, M8, N8]
    auto sindex_base = (kl * 16 + rcl) * 8 + 0;
#endif

    // [N2, M2, M8, N8]
#ifdef ATTENTION_C4
    // [mNumHead * (mHeadDim / 4), mBatch * mSeqLen, 4]
    auto xy_out = output + (long)(b * q_seq_len + seq_idx * q_seq_piece_len + sl * 16 + rcl) * 4 + (long)(hn * head_dim / 4 + hm * 4 + kl * 2) * 4 * param.batch * q_seq_len + 0;
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
    auto xy_out = output + ((long)((b * q_seq_len + seq_idx * q_seq_piece_len + sl * 16 + rcl) * head_num + hn)) * head_dim + hm * 16 + kl * 8 + 0;
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

    int kv_head_num = head_num / group;
    int kh = hn / group;

    const int stride = head_num * head_dim / group;

    // [mBatch, mNumHead, mSeqLen, mKvSeqLen]
    device const ftype *A_offset = input0 + (y * q_seq_piece_len + x) * align_value_len;
    float out = 0.0;

    // row-major V: per-token strided scalar read (d = z fixed)
    for(int i = 0; i < align_value_len; ++i){
#ifdef QUANT_V
        ftype B = GETV(((const device char*)past_value)[((long)i * param.batch + b) * kv_head_num * head_dim + kh * head_dim + z],
                       b * param.max_kv_len + i);
#else
        ftype B = past_value[((long)i * param.batch + b) * kv_head_num * head_dim + kh * head_dim + z];
#endif
        out += float(A_offset[i]) * float(B);
    }
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

    int kv_head_num = head_num / group;
    int kh = hn / group;
    int value_seq_len = param.key_seq_len;
    int align_value_len = ((value_seq_len + param.kv_align_len - 1) / param.kv_align_len) * param.kv_align_len;

    device const ftype *A_offset = input0 + (y * q_seq_len + x) * align_value_len;
    float out = 0;

    // row-major V: per-token strided scalar read (d = z fixed)
#ifdef SIMD_GROUP_REDUCE
    for(int i = tiisg; i < align_value_len; i += SIMD_GROUP_WIDTH){
#ifdef QUANT_V
        ftype B = GETV(((const device char*)past_value)[((long)i * param.batch + b) * kv_head_num * head_dim + kh * head_dim + z],
                       b * param.max_kv_len + i);
#else
        ftype B = past_value[((long)i * param.batch + b) * kv_head_num * head_dim + kh * head_dim + z];
#endif
        out += float(A_offset[i]) * float(B);
    }
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
    for(int i = 0; i < align_value_len; ++i){
#ifdef QUANT_V
        ftype B = GETV(((const device char*)past_value)[((long)i * param.batch + b) * kv_head_num * head_dim + kh * head_dim + z],
                       b * param.max_kv_len + i);
#else
        ftype B = past_value[((long)i * param.batch + b) * kv_head_num * head_dim + kh * head_dim + z];
#endif
        out += float(A_offset[i]) * float(B);
    }
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

    int kv_head_num = head_num / group;
    int kh = hn / group;
    int value_seq_len = param.key_seq_len;
    int align_value_len = ((value_seq_len + param.kv_align_len - 1) / param.kv_align_len) * param.kv_align_len;

    device const ftype *A_offset = input0 + (y * q_seq_len + x) * align_value_len;

    // row-major V: per-token read of the adjacent d pair (z, z+1), contiguous in-row
    float out0 = 0;
    float out1 = 0;
    for(int i = tiisg; i < align_value_len; i += SIMD_GROUP_WIDTH){
        float A = float(A_offset[i]);
        const long v_off = ((long)i * param.batch + b) * kv_head_num * head_dim + kh * head_dim + z;
#ifdef QUANT_V
#ifdef DYNAMIC_QUANT_V
        int tok_idx = b * param.max_kv_len + i;
        float vs = float(v_scales[tok_idx * 2]);
        float vb = float(v_scales[tok_idx * 2 + 1]);
        char2 raw = ((const device char2*)((const device char*)past_value + v_off))[0];
        out0 += A * (float(raw.x) * vs + vb);
        out1 += A * (float(raw.y) * vs + vb);
#else
        char2 raw = ((const device char2*)((const device char*)past_value + v_off))[0];
        out0 += A * (float(raw.x) * param.v_scale);
        out1 += A * (float(raw.y) * param.v_scale);
#endif
#else
        out0 += A * float(past_value[v_off]);
        out1 += A * float(past_value[v_off + 1]);
#endif
    }
    float r0 = out0;
    float r1 = out1;
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

// Determine max KV length based on GROUP_SIZE to stay within 32KB threadgroup memory
// Memory usage: GROUP_SIZE * (MAX_KV + 32) * sizeof(float)
#ifdef SHORT_KV_128
#define DECODE_QK_SOFTMAX_MAX_KV 128
#elif GROUP_SIZE <= 2
#define DECODE_QK_SOFTMAX_MAX_KV 2048
#elif GROUP_SIZE <= 4
#define DECODE_QK_SOFTMAX_MAX_KV 1024
#elif GROUP_SIZE <= 8
#define DECODE_QK_SOFTMAX_MAX_KV 512
#endif

// GROUP_SIZE == 2 specialization: keep the pre-b9a6e60e hard-coded implementation.
// The generic loop-over-GROUP_SIZE version (below) puts scores/reduce state in
// arrays indexed by g, which stops Metal's compiler from fully lifting them into
// registers. For GROUP_SIZE=2 that costs ~15% (measured on Qwen3-0.6B decode).
// The hard-coded pair of scalars gives the compiler two independent instruction
// streams (s0/s1, local_max0/local_max1, etc.) which interleave cleanly.
#if GROUP_SIZE == 2
#ifdef QK_QSPLIT
// Q-head-split variant (host auto gate: group_size==2, non-tensor-API device,
// kv>=512): each
// threadgroup handles ONE query head — gid.z selects the head within the kv
// group. Doubles threadgroup count (8 -> 16 on Qwen3-0.6B) for occupancy, at
// the cost of reading K once per q-head (2x K traffic) and a single dot
// stream per thread (vs the s0/s1 ILP pair below).
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
    threadgroup float scores0[DECODE_QK_SOFTMAX_MAX_KV];
    threadgroup float reduce0[32];

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

    const int head0 = kv_hn * GROUP_SIZE + int(gid.z);
    const int query_offset = (b * param.query_seq_len + q_idx) * param.head_num * head_dim;
    const device ftype* query0 = input0 + query_offset + head0 * head_dim;
    const int key_head_offset = kv_hn * head_dim;
    const int key_stride = kv_head_num * head_dim;

    float local_max0 = -FLT_MAX;
    const int kv_valid_limit = max(key_seq_len - param.query_seq_len, 0) + q_idx;
    for (int k = int(tid); k < key_seq_len; k += tptg) {
#ifdef QUANT_K
        const device char* key = (const device char*)past_key + (k * param.batch + b) * key_stride + key_head_offset;
#else
        const device ftype* key = past_key + (k * param.batch + b) * key_stride + key_head_offset;
#endif
        float s0 = 0.0f;
        const device ftype4* q04 = (const device ftype4*)query0;
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
        }
        s0 *= param.scale;
        if (k > kv_valid_limit) {
            s0 = -FLT_MAX;
        }
        scores0[k] = s0;
        local_max0 = max(local_max0, s0);
    }

    local_max0 = simd_max(local_max0);
    if (tiisg == 0) {
        reduce0[sgitg] = local_max0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgitg == 0 && tiisg == 0) {
        float max0 = -FLT_MAX;
        for (int i = 0; i < sg_count; ++i) {
            max0 = max(max0, reduce0[i]);
        }
        reduce0[0] = max0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float max0 = reduce0[0];

    float local_sum0 = 0.0f;
    for (int k = int(tid); k < key_seq_len; k += tptg) {
        float v0 = exp(scores0[k] - max0);
        scores0[k] = v0;
        local_sum0 += v0;
    }

    local_sum0 = simd_sum(local_sum0);
    if (tiisg == 0) {
        reduce0[sgitg] = local_sum0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgitg == 0 && tiisg == 0) {
        float sum0 = 0.0f;
        for (int i = 0; i < sg_count; ++i) {
            sum0 += reduce0[i];
        }
        reduce0[0] = sum0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv_sum0 = 1.0f / reduce0[0];

    const int base0 = ((b * param.head_num + head0) * param.query_seq_len + q_idx) * align_key_len;
    for (int k = int(tid); k < key_seq_len; k += tptg) {
        output[base0 + k] = (ftype)(scores0[k] * inv_sum0);
    }
    for (int k = int(tid) + key_seq_len; k < align_key_len; k += tptg) {
        output[base0 + k] = (ftype)0.0f;
    }
}
#else  // !QK_QSPLIT: original paired implementation
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
    threadgroup float scores0[DECODE_QK_SOFTMAX_MAX_KV];
    threadgroup float scores1[DECODE_QK_SOFTMAX_MAX_KV];
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
#endif // QK_QSPLIT
#else  // GROUP_SIZE != 2: generic implementation for group_size in {4, 8}
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
    // Threadgroup memory for scores and reduction buffers, indexed as [group][element]
    threadgroup float scores_buf[GROUP_SIZE * DECODE_QK_SOFTMAX_MAX_KV];
    threadgroup float reduce_buf[GROUP_SIZE * 32];

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

    const int head_base = kv_hn * GROUP_SIZE;
    const int query_offset = (b * param.query_seq_len + q_idx) * param.head_num * head_dim;
    const int key_head_offset = kv_hn * head_dim;
    const int key_stride = kv_head_num * head_dim;

    // Pre-compute query pointers for all heads in the group
    const device ftype4* q4_ptrs[GROUP_SIZE];
    for (int g = 0; g < GROUP_SIZE; g++) {
        q4_ptrs[g] = (const device ftype4*)(input0 + query_offset + (head_base + g) * head_dim);
    }

    float local_max[GROUP_SIZE];
    for (int g = 0; g < GROUP_SIZE; g++) {
        local_max[g] = -FLT_MAX;
    }

    const int kv_valid_limit = max(key_seq_len - param.query_seq_len, 0) + q_idx;
    for (int k = int(tid); k < key_seq_len; k += tptg) {
        // Read key data once, shared across all heads in the group
#ifdef QUANT_K
        const device char* key = (const device char*)past_key + (k * param.batch + b) * key_stride + key_head_offset;
        const device char4* k4 = (const device char4*)key;
#ifdef DYNAMIC_QUANT_K
        const int k_token_idx = k * param.batch + b;
        const float k_scale = float(k_scales[k_token_idx * 2]);
        const float k_bias = float(k_scales[k_token_idx * 2 + 1]);
#endif
#else
        const device ftype* key = past_key + (k * param.batch + b) * key_stride + key_head_offset;
        const device ftype4* k4 = (const device ftype4*)key;
#endif

        // Compute dot products for all heads in the group
        float s[GROUP_SIZE];
        for (int g = 0; g < GROUP_SIZE; g++) {
            s[g] = 0.0f;
        }

        for (int d = 0; d < head_dim / 8; ++d) {
#ifdef QUANT_K
#ifdef DYNAMIC_QUANT_K
            float4 kv0 = float4(k4[d * 2 + 0]) * k_scale + k_bias;
            float4 kv1 = float4(k4[d * 2 + 1]) * k_scale + k_bias;
#else
            float4 kv0 = float4(k4[d * 2 + 0]) * param.k_scale;
            float4 kv1 = float4(k4[d * 2 + 1]) * param.k_scale;
#endif
#else
            float4 kv0 = float4(k4[d * 2 + 0]);
            float4 kv1 = float4(k4[d * 2 + 1]);
#endif
            for (int g = 0; g < GROUP_SIZE; g++) {
                s[g] += dot(float4(q4_ptrs[g][d * 2 + 0]), kv0) + dot(float4(q4_ptrs[g][d * 2 + 1]), kv1);
            }
        }

        bool masked = (k > kv_valid_limit);
        for (int g = 0; g < GROUP_SIZE; g++) {
            float sv = masked ? -FLT_MAX : (s[g] * param.scale);
            scores_buf[g * DECODE_QK_SOFTMAX_MAX_KV + k] = sv;
            local_max[g] = max(local_max[g], sv);
        }
    }

    // Max reduction across simdgroups
    for (int g = 0; g < GROUP_SIZE; g++) {
        local_max[g] = simd_max(local_max[g]);
        if (tiisg == 0) {
            reduce_buf[g * 32 + sgitg] = local_max[g];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgitg == 0 && tiisg == 0) {
        for (int g = 0; g < GROUP_SIZE; g++) {
            float m = -FLT_MAX;
            for (int i = 0; i < sg_count; ++i) {
                m = max(m, reduce_buf[g * 32 + i]);
            }
            reduce_buf[g * 32] = m;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float max_val[GROUP_SIZE];
    for (int g = 0; g < GROUP_SIZE; g++) {
        max_val[g] = reduce_buf[g * 32];
    }

    // Exp and sum
    float local_sum[GROUP_SIZE];
    for (int g = 0; g < GROUP_SIZE; g++) {
        local_sum[g] = 0.0f;
    }
    for (int k = int(tid); k < key_seq_len; k += tptg) {
        for (int g = 0; g < GROUP_SIZE; g++) {
            float v = exp(scores_buf[g * DECODE_QK_SOFTMAX_MAX_KV + k] - max_val[g]);
            scores_buf[g * DECODE_QK_SOFTMAX_MAX_KV + k] = v;
            local_sum[g] += v;
        }
    }

    // Sum reduction across simdgroups
    for (int g = 0; g < GROUP_SIZE; g++) {
        local_sum[g] = simd_sum(local_sum[g]);
        if (tiisg == 0) {
            reduce_buf[g * 32 + sgitg] = local_sum[g];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgitg == 0 && tiisg == 0) {
        for (int g = 0; g < GROUP_SIZE; g++) {
            float s = 0.0f;
            for (int i = 0; i < sg_count; ++i) {
                s += reduce_buf[g * 32 + i];
            }
            reduce_buf[g * 32] = s;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Normalize and write output
    float inv_sum[GROUP_SIZE];
    for (int g = 0; g < GROUP_SIZE; g++) {
        inv_sum[g] = 1.0f / reduce_buf[g * 32];
    }

    for (int g = 0; g < GROUP_SIZE; g++) {
        const int head = head_base + g;
        const int base = ((b * param.head_num + head) * param.query_seq_len + q_idx) * align_key_len;
        for (int k = int(tid); k < key_seq_len; k += tptg) {
            output[base + k] = (ftype)(scores_buf[g * DECODE_QK_SOFTMAX_MAX_KV + k] * inv_sum[g]);
        }
        for (int k = int(tid) + key_seq_len; k < align_key_len; k += tptg) {
            output[base + k] = (ftype)0.0f;
        }
    }
}
#endif // GROUP_SIZE == 2
)metal";

// Decode attention in MLX sdpa_vector form: per-token interleaved streaming
// (simdgroup s handles kv tokens s, s+NSG, ...), each lane owns a HEAD_DIM/32
// slice of q/k/v/o kept in registers, and online softmax updates O directly
// from the token's V row -- no score staging, no separate AV phase.
// Compile-time: ftype, GROUP_SIZE, HEAD_DIM, SPLITKV_NSG (+ optional
// ATTENTION_C4, QUANT_K/QUANT_V with per-token DYNAMIC scales).
// Layouts:
//   query : [batch, 1, head_num, head_dim]
//   K     : [max_kv, batch, kv_head_num, head_dim]
//   V     : [max_kv, batch, kv_head_num, head_dim]   (row-major, same as K)
const char* gDecodeSplitKV = R"metal(
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
#ifndef SPLITKV_NSG
#define SPLITKV_NSG 4
#endif
#define DPT (HEAD_DIM / SIMD_GROUP_WIDTH)   // d-values owned per lane (MLX qk_per_thread)
// One threadgroup per q head (grid.y = batch*head_num, MLX sdpa_vector form);
// the GROUP_SIZE loops collapse. The kv-head-grouped variant (GS_LOCAL =
// GROUP_SIZE, SDPA_QSPLIT=0) measured worse and was removed 2026-07-30.
#define GS_LOCAL 1

kernel void decode_splitkv(const device ftype* input0 [[buffer(0)]],
    device ftype* out_final [[buffer(1)]],
    const device ftype* past_key [[buffer(2)]],
    const device ftype* past_value [[buffer(3)]],
    constant Param& param [[buffer(4)]],
    constant int& nwg [[buffer(5)]],
    device ftype* k_scales [[buffer(8)]],
    device ftype* v_scales [[buffer(9)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]]) {

    const int kv_head_num = param.head_num / GROUP_SIZE;
    const int b     = int(gid.y) / param.head_num;
    const int q_head_base = int(gid.y) % param.head_num; // this TG's single q head
    const int kv_hn = q_head_base / GROUP_SIZE;
    const int key_seq_len = param.key_seq_len;
    const int key_stride   = kv_head_num * HEAD_DIM;          // K row: [batch, kv_head_num, HEAD_DIM]
    const int v_seq_stride = param.batch * key_stride;        // V: [kv, batch, kv_head_num, HEAD_DIM]

    // Query slice for this lane, scaled, kept in registers (MLX form).
    thread float q[DPT];
    {
        const device ftype* q_base = input0 + (b * param.head_num + q_head_base) * HEAD_DIM + int(tiisg) * DPT;
        for (int d = 0; d < DPT; ++d) {
            q[d] = float(q_base[d]) * param.scale;
        }
    }

    float S[GS_LOCAL];
    float M[GS_LOCAL];
    float O[GS_LOCAL][DPT];
    for (int g = 0; g < GS_LOCAL; ++g) {
        S[g] = 0.0f;
        M[g] = -FLT_MAX / 2;
        for (int d = 0; d < DPT; ++d) {
            O[g][d] = 0.0f;
        }
    }

    // Cross-simdgroup reduce scratch, filled after the stream loop:
    // (S, M) per simdgroup + transposed O partials. Deliberately independent
    // of HEAD_DIM so threadgroup memory stays small at NSG=32.
    threadgroup float s_sm[SPLITKV_NSG][GS_LOCAL][2];
    threadgroup float s_out[SPLITKV_NSG * SIMD_GROUP_WIDTH];

#ifdef QUANT_K
    const device char* k_cache = (const device char*)past_key;
#else
    const device ftype* k_cache = past_key;
#endif
#ifdef QUANT_V
    const device char* v_cache = (const device char*)past_value;
#else
    const device ftype* v_cache = past_value;
#endif

    // Per-token interleaved streaming: simdgroup s handles tokens
    // s, s+NSG, s+2*NSG, ... (MLX sdpa_vector `i = simd_gid; i < N; i += BN`).
    for (int i = int(sgitg); i < key_seq_len; i += SPLITKV_NSG * nwg) {
        // ---- QK: lane <-> head_dim slice of kv token i ----
#ifdef QUANT_K
        const device char* kp = k_cache + (i * param.batch + b) * key_stride + kv_hn * HEAD_DIM + int(tiisg) * DPT;
#ifdef DYNAMIC_QUANT_K
        const float k_scale = float(k_scales[(i * param.batch + b) * 2 + 0]);
        const float k_bias  = float(k_scales[(i * param.batch + b) * 2 + 1]);
#else
        const float k_scale = param.k_scale;
        const float k_bias  = 0.0f;
#endif
#else
        const device ftype* kp = k_cache + (i * param.batch + b) * key_stride + kv_hn * HEAD_DIM + int(tiisg) * DPT;
#endif
        float score = 0.0f;
        for (int d = 0; d < DPT; ++d) {
#ifdef QUANT_K
            const float k = float(kp[d]) * k_scale + k_bias;
#else
            const float k = float(kp[d]);
#endif
            score += q[d] * k;
        }
        score = simd_sum(score);

        // ---- online softmax + immediate AV from the same token's V row ----
#ifdef QUANT_V
        const device char* vp = v_cache + i * v_seq_stride + (b * kv_head_num + kv_hn) * HEAD_DIM + int(tiisg) * DPT;
#ifdef DYNAMIC_QUANT_V
        const float v_sc = float(v_scales[(b * param.max_kv_len + i) * 2 + 0]);
        const float v_bi = float(v_scales[(b * param.max_kv_len + i) * 2 + 1]);
#else
        const float v_sc = param.v_scale;
        const float v_bi = 0.0f;
#endif
#else
        const device ftype* vp = v_cache + i * v_seq_stride + (b * kv_head_num + kv_hn) * HEAD_DIM + int(tiisg) * DPT;
#endif
        for (int g = 0; g < GS_LOCAL; ++g) {
            const float m_prev = M[g];
            M[g] = max(m_prev, score);
            const float ms = exp(m_prev - M[g]);
            const float vs = exp(score - M[g]);
            S[g] = S[g] * ms + vs;
            for (int d = 0; d < DPT; ++d) {
#ifdef QUANT_V
                const float v = float(vp[d]) * v_sc + v_bi;
#else
                const float v = float(vp[d]);
#endif
                O[g][d] = O[g][d] * ms + vs * v;
            }
        }
    }
    // ---- cross-simdgroup reduce inside the threadgroup ----
    // MLX sdpa_vector-style transposed reduce: (S, M) are combined by one
    // simd_max/simd_sum over lanes indexing simdgroups; O is combined one
    // 32-component group at a time through the shared s_out scratch, so the
    // barrier count is 2 per component group instead of log2(NSG) full-HEAD_DIM
    // sweeps and threadgroup memory stays HEAD_DIM-independent.
    if (tiisg == 0) {
        for (int g = 0; g < GS_LOCAL; ++g) {
            s_sm[sgitg][g][0] = S[g];
            s_sm[sgitg][g][1] = M[g];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int row_base = b * param.head_num + q_head_base;
    const bool lane_is_sg = (int(tiisg) < SPLITKV_NSG);
    for (int g = 0; g < GS_LOCAL; ++g) {
        // lane t holds simdgroup t's (S, M); reduce across lanes
        const float Mi = lane_is_sg ? s_sm[tiisg][g][1] : (-FLT_MAX / 2);
        const float m  = simd_max(Mi);
        const float factor = lane_is_sg ? exp(Mi - m) : 0.0f;
        const float S_tot  = simd_sum(lane_is_sg ? (s_sm[tiisg][g][0] * factor) : 0.0f);

        const float inv_s = (S_tot == 0.0f) ? 0.0f : (1.0f / S_tot);
        const int hn = q_head_base + g;

        for (int dd = 0; dd < DPT; ++dd) {
            // transpose partials: component-lane major so one simdgroup can gather
            // all SPLITKV_NSG partials of a component with a single simd_sum
            s_out[tiisg * SPLITKV_NSG + sgitg] = O[g][dd];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int rep = 0; rep < SIMD_GROUP_WIDTH / SPLITKV_NSG; ++rep) {
                const int lp = int(sgitg) + rep * SPLITKV_NSG; // component-lane owned by this simdgroup
                const float part = lane_is_sg ? (s_out[lp * SPLITKV_NSG + tiisg] * factor) : 0.0f;
                const float acc  = simd_sum(part);
                if (tiisg == 0) {
                    // MLX lane mapping: lane lp owns components lp*DPT .. lp*DPT+DPT-1
                    const int d = lp * DPT + dd;
#ifdef ATTENTION_C4
                    // [mNumHead * (mHeadDim / 4), mBatch * mSeqLen(=1), 4]
                    const int c  = hn * HEAD_DIM + d;
                    out_final[b * 4 + (c % 4) + (c / 4) * param.batch * 4] = ftype(acc * inv_s);
#else
                    // [mBatch, mSeqLen(=1), mNumHead, mHeadDim]
                    out_final[(row_base + g) * HEAD_DIM + d] = ftype(acc * inv_s);
#endif
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}
)metal";

// softmax sg reduce source moved to MetalSoftmaxShader.cpp

#endif /* MNN_SUPPORT_TRANSFORMER_FUSE */
#endif
