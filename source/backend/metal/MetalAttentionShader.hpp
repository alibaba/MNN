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
#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;
struct Param {
    int query_seq_len;
    int key_seq_len;
    int head_num;
    int group;
    int head_dim;
    float scale;
    int max_kv_len;
};
#define SIMD_GROUP_WIDTH 32

kernel void prefill_qk(const device T* input0 [[buffer(0)]],
    device T* output [[buffer(1)]],
    device T* past_key [[buffer(2)]],
#ifdef FLOAT_MASK
    const device T* mask [[buffer(3)]],
#else
    const device int* mask [[buffer(3)]],
#endif
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
    
    simdgroup_float8x8 sga[2];
    simdgroup_float8x8 sgb[2];
    simdgroup_float8x8 sgd[4];
    for (int i = 0; i < 4; i++){
        sgd[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    int kl = tiitg % 2;// 0~1
    int rcl = tiitg / 2;// 0~15

    const int slq = gid.x; // q_seq_len/16 -> M/16
    const int slk = gid.y; // k_seq_len/16 -> N/16
    const int z = gid.z; // head_num

    /** Q:
     threadgroup: [M16, K8]
     each thread: K4
     layout: [M, B, K] -> [M/16, M16, B, K/8, K2, K4]
     index : [slq, rcl, z, 0, kl, K4]
     offset: ((slq * 16 + rcl) * B + z) * K + (0 * 2 + kl) * 4 + 0
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
    int zin = z / param.group;
    int q_seq_len = param.query_seq_len;
    int k_seq_len = param.key_seq_len;
    int head_num = param.head_num;
    int head_dim = param.head_dim;

    threadgroup float sdata[256] = {0.f};

    int idx_slq = slq * 16 + rcl < q_seq_len ? slq * 16 + rcl : q_seq_len - 1;
    int idx_slk = slk * 16 + rcl < k_seq_len ? slk * 16 + rcl : k_seq_len - 1;

    auto A_offset = input0 + (idx_slq * head_num + z) * head_dim + (0 * 2 + kl) * 4 + 0;
    auto B_offset = past_key + (idx_slk * head_num / group + zin) * head_dim + 0 * 8 + kl * 4 + 0;

    for(int i = 0; i < head_dim; i += 8){
        sdata[rcl * 8 + kl * 4 + 0] = A_offset[i + 0];
        sdata[rcl * 8 + kl * 4 + 1] = A_offset[i + 1];
        sdata[rcl * 8 + kl * 4 + 2] = A_offset[i + 2];
        sdata[rcl * 8 + kl * 4 + 3] = A_offset[i + 3];
        
        sdata[128 + (kl * 4 + 0) * 16 + rcl] = B_offset[i + 0];
        sdata[128 + (kl * 4 + 1) * 16 + rcl] = B_offset[i + 1];
        sdata[128 + (kl * 4 + 2) * 16 + rcl] = B_offset[i + 2];
        sdata[128 + (kl * 4 + 3) * 16 + rcl] = B_offset[i + 3];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_load(sga[0], (const threadgroup float*)sdata, 8);
        simdgroup_load(sga[1], ((const threadgroup float*)sdata) + 64, 8);
        
        simdgroup_load(sgb[0], ((const threadgroup float*)sdata) + 128, 16);
        simdgroup_load(sgb[1], ((const threadgroup float*)sdata) + 136, 16);
        
        simdgroup_multiply_accumulate(sgd[0], sga[0], sgb[0], sgd[0]);
        simdgroup_multiply_accumulate(sgd[1], sga[1], sgb[0], sgd[1]);
        simdgroup_multiply_accumulate(sgd[2], sga[0], sgb[1], sgd[2]);
        simdgroup_multiply_accumulate(sgd[3], sga[1], sgb[1], sgd[3]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_store(sgd[0], (threadgroup float*)sdata, 8);
    simdgroup_store(sgd[1], (threadgroup float*)sdata + 64, 8);
    simdgroup_store(sgd[2], (threadgroup float*)sdata + 128, 8);
    simdgroup_store(sgd[3], (threadgroup float*)sdata + 192, 8);
    
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // [N2, M2, M8, N8]
    float Vscale = (float)param.scale;

    auto xy_out = output + (z * q_seq_len + slq * 16 + rcl) * k_seq_len + slk * 16 + kl * 8 + 0;
    if(slq * 16 + rcl < q_seq_len) {
        if(slk * 16 + kl * 8 + 0 < k_seq_len) {
            auto out0 =  sdata[(kl * 16 + rcl) * 8 + 0] * Vscale;
            #ifdef FLOAT_MASK
                out0 = mask[((slq * 16 + rcl) * k_seq_len + (slk * 16 + kl * 8 + 0))] + out0;
            #else
                out0 = mask[((slq * 16 + rcl) * k_seq_len + (slk * 16 + kl * 8 + 0))] == 0 ? -FLT_MAX : out0;
            #endif
            xy_out[0] = out0;
        }
        if(slk * 16 + kl * 8 + 1 < k_seq_len) {
            auto out0 =  sdata[(kl * 16 + rcl) * 8 + 1] * Vscale;
            #ifdef FLOAT_MASK
                out0 = mask[((slq * 16 + rcl) * k_seq_len + (slk * 16 + kl * 8 + 1))] + out0;
            #else
                out0 = mask[((slq * 16 + rcl) * k_seq_len + (slk * 16 + kl * 8 + 1))] == 0 ? -FLT_MAX : out0;
            #endif
            xy_out[1] = out0;
        }
        if(slk * 16 + kl * 8 + 2 < k_seq_len) {
            auto out0 =  sdata[(kl * 16 + rcl) * 8 + 2] * Vscale;
            #ifdef FLOAT_MASK
                out0 = mask[((slq * 16 + rcl) * k_seq_len + (slk * 16 + kl * 8 + 2))] + out0;
            #else
                out0 = mask[((slq * 16 + rcl) * k_seq_len + (slk * 16 + kl * 8 + 2))] == 0 ? -FLT_MAX : out0;
            #endif
            xy_out[2] = out0;
        }
        if(slk * 16 + kl * 8 + 3 < k_seq_len) {
            auto out0 =  sdata[(kl * 16 + rcl) * 8 + 3] * Vscale;
            #ifdef FLOAT_MASK
                out0 = mask[((slq * 16 + rcl) * k_seq_len + (slk * 16 + kl * 8 + 3))] + out0;
            #else
                out0 = mask[((slq * 16 + rcl) * k_seq_len + (slk * 16 + kl * 8 + 3))] == 0 ? -FLT_MAX : out0;
            #endif
            xy_out[3] = out0;
        }
        if(slk * 16 + kl * 8 + 4 < k_seq_len) {
            auto out0 =  sdata[(kl * 16 + rcl) * 8 + 4] * Vscale;
            #ifdef FLOAT_MASK
                out0 = mask[((slq * 16 + rcl) * k_seq_len + (slk * 16 + kl * 8 + 4))] + out0;
            #else
                out0 = mask[((slq * 16 + rcl) * k_seq_len + (slk * 16 + kl * 8 + 4))] == 0 ? -FLT_MAX : out0;
            #endif
            xy_out[4] = out0;
        }
        if(slk * 16 + kl * 8 + 5 < k_seq_len) {
            auto out0 =  sdata[(kl * 16 + rcl) * 8 + 5] * Vscale;
            #ifdef FLOAT_MASK
                out0 = mask[((slq * 16 + rcl) * k_seq_len + (slk * 16 + kl * 8 + 5))] + out0;
            #else
                out0 = mask[((slq * 16 + rcl) * k_seq_len + (slk * 16 + kl * 8 + 5))] == 0 ? -FLT_MAX : out0;
            #endif
            xy_out[5] = out0;
        }
        if(slk * 16 + kl * 8 + 6 < k_seq_len) {
            auto out0 =  sdata[(kl * 16 + rcl) * 8 + 6] * Vscale;
            #ifdef FLOAT_MASK
                out0 = mask[((slq * 16 + rcl) * k_seq_len + (slk * 16 + kl * 8 + 6))] + out0;
            #else
                out0 = mask[((slq * 16 + rcl) * k_seq_len + (slk * 16 + kl * 8 + 6))] == 0 ? -FLT_MAX : out0;
            #endif
            xy_out[6] = out0;
        }
        if(slk * 16 + kl * 8 + 7 < k_seq_len) {
            auto out0 =  sdata[(kl * 16 + rcl) * 8 + 7] * Vscale;
            #ifdef FLOAT_MASK
                out0 = mask[((slq * 16 + rcl) * k_seq_len + (slk * 16 + kl * 8 + 7))] + out0;
            #else
                out0 = mask[((slq * 16 + rcl) * k_seq_len + (slk * 16 + kl * 8 + 7))] == 0 ? -FLT_MAX : out0;
            #endif
            xy_out[7] = out0;
        }
    }

#else
    const int x = gid.x; // query_seq_len
    const int y = gid.y; // head_num
    const int z = gid.z; // key_seq_len

    if (x >= param.query_seq_len || y >= param.head_num || z >= param.key_seq_len) {
        return;
    }
    int group = param.group;
    int query_seq_len = param.query_seq_len;
    int key_seq_len = param.key_seq_len;
    int head_num = param.head_num;
    int head_dim = param.head_dim;
    
    const int offset = head_num * head_dim;
    const int offset_head = y * head_dim;
    const int offset_head_kv = (y / group) * head_dim;
    const device T* A_offset = input0 + x * offset + offset_head;

    float Vscale = (float)param.scale;

    device const T* B_offset = past_key + z * offset / group + offset_head_kv;
    const int output_offset = y * query_seq_len * key_seq_len;
    float out0 = 0.0;
    
    for(int i = 0; i < head_dim; ++i){
        float A = (float)(A_offset[i]);
        float B = (float)(B_offset[i]);
        out0 += B * A;
    }
    
    out0 *= Vscale;
    
#ifdef FLOAT_MASK
    out0 = mask[((x + 0) * key_seq_len + (z + 0))] + out0;
#else
    out0 = mask[((x + 0) * key_seq_len + (z + 0))] == 0 ? -FLT_MAX : out0;
#endif
    output[output_offset + x * key_seq_len + z] = (T)out0;
#endif
}

kernel void decode_qk(const device T* input0 [[buffer(0)]],
    device T* output [[buffer(1)]],
    device T* past_key [[buffer(2)]],
#ifdef FLOAT_MASK
    const device T* mask [[buffer(3)]],
#else
    const device int* mask [[buffer(3)]],
#endif
    constant Param& param [[buffer(4)]],
#ifdef SIMD_GROUP_REDUCE
    uint3 gid[[threadgroup_position_in_grid]],
    uint  tiisg[[thread_index_in_simdgroup]],
    uint  sgitg[[simdgroup_index_in_threadgroup]]
#else
    uint3 gid[[thread_position_in_grid]]
#endif
) {
    int x = gid.x; // query_seq_len
    int y = gid.y; // head_num
    int z = gid.z; // key_seq_len

#ifdef HEAD_NUM_2
    y = y * 2;
#endif
    if (x >= param.query_seq_len || y >= param.head_num || z >= param.key_seq_len) {
        return;
    }
    int group = param.group;

    int key_seq_len = param.key_seq_len;
    int head_num = param.head_num;
    int head_dim = param.head_dim;
    
    const int offset = head_num * head_dim;
    const int offset_head = y * head_dim;
    const int offset_head_kv = (y / param.group) * head_dim;
    const device T* A_offset = input0 + x * offset + offset_head;
    device T* Pastkey_offset = past_key + z * offset / group + offset_head_kv;
    float Vscale = (float)param.scale;
    float out = 0.0;

#ifdef HEAD_NUM_2
    const device T* A_offset_1 = A_offset + head_dim;
    device T* Pastkey_offset_1 = past_key + z * offset / group + ((y+1) / param.group) * head_dim;
    float out_1 = 0.0;
#endif

#ifdef SIMD_GROUP_REDUCE
    for(int i = tiisg; i < head_dim; i+=SIMD_GROUP_WIDTH){
        float A = A_offset[i];
        float B = (float)Pastkey_offset[i];
        
        out += A * B;
    }

#ifdef HEAD_NUM_2
    if(y + 1 < param.head_num) {
        for(int i = tiisg; i < head_dim; i+=SIMD_GROUP_WIDTH){
            float A = A_offset_1[i];
            float B = (float)Pastkey_offset_1[i];
            
            out_1 += A * B;
        }
    }
#endif
    out = simd_sum(out);

#ifdef HEAD_NUM_2
    if(y + 1 < param.head_num) {
        out_1 = simd_sum(out_1);
        if(tiisg == 1) {
            out_1 *= Vscale;
            output[(y+1) * key_seq_len + z] = (T)out_1;
        }
    }
#endif
    if(tiisg == 0) {
        out *= Vscale;
        output[y * key_seq_len + z] = (T)out;
    }

#else
    {
        for(int i = 0; i < head_dim; i++){
            float A = A_offset[i];
            float B = (float)Pastkey_offset[i];
            
            out += A * B;
        }
    }
    out *= Vscale;
    output[y * key_seq_len + z] = (T)out;

#ifdef HEAD_NUM_2
    if(y + 1 < param.head_num) {
        for(int i = 0; i < head_dim; i++){
            float A = A_offset_1[i];
            float B = (float)Pastkey_offset_1[i];
            
            out_1 += A * B;
        }
        out_1 *= Vscale;
        output[(y+1) * key_seq_len + z] = (T)out_1;
    }
#endif

#endif
}

)metal";

const char* gCopyPastKV = R"metal(
#include <metal_stdlib>
using namespace metal;
struct Param {
    int head_count;
    int q_seq_len;
    int max_kv_len;
    int dst_k_offset;
    int dst_v_offset;
};
kernel void copy(const device T* input0 [[buffer(0)]],
    const device T* input1 [[buffer(1)]],
    device T* output0 [[buffer(2)]],
    device T* output1 [[buffer(3)]],
    constant Param& param [[buffer(4)]],
    uint3 gid[[thread_position_in_grid]]
) {
    const int x = gid.x; // head_num / group * head_dim
    const int y = gid.y; // q_seq_len
    if (x >= param.head_count || y >= param.q_seq_len) {
        return;
    }
    const int index = y * param.head_count + x;
    output0[param.dst_k_offset + index] = input0[index];
    output1[param.dst_v_offset + x * param.max_kv_len + y] = input1[index];
}
)metal";

const char* gMatMulQKV = R"metal(

#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;
struct Param {
    int query_seq_len;
    int key_seq_len;
    int head_num;
    int group;
    int head_dim;
    float scale;
    int max_kv_len;
};
#define SIMD_GROUP_WIDTH 32
kernel void prefill_qkv(const device T* input0 [[buffer(0)]],
    device T* output [[buffer(1)]],
    device T* past_value [[buffer(2)]],
    constant Param& param [[buffer(3)]],
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
    
    simdgroup_float8x8 sga[2];
    simdgroup_float8x8 sgb[2];
    simdgroup_float8x8 sgd[4];
    for (int i = 0; i < 4; i++){
        sgd[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    int kl = tiitg % 2;// 0~1
    int rcl = tiitg / 2;// 0~15

    int nl = tiitg % 4;// 0~3
    int kcl = tiitg / 4;// 0~7

    const int sl = gid.x; // q_seq_len/16 -> M/16
    const int hm = gid.y; // head_dim/16 -> N/16
    const int z = gid.z; // head_num

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
     layout: [K, B/G, N] -> [K/8, K8, B/G, N/16, N4, N4]
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
    int zin = z / group;
    int q_seq_len = param.query_seq_len;
    int value_seq_len = param.key_seq_len;
    int head_num = param.head_num;
    int head_dim = param.head_dim;

    threadgroup float sdata[256] = {0.f};

    int idx_qk_sl = sl * 16 + rcl < q_seq_len ? (sl * 16 + rcl) : q_seq_len - 1;

    auto A_offset = input0 + (z * q_seq_len + idx_qk_sl) * value_seq_len + (0 * 2 + kl) * 4 + 0;
    auto B_offset = past_value + (zin * head_dim + hm * 16 + nl * 4 + 0) * param.max_kv_len + (0 * 8 + kcl);
    

    for(int i = 0; i < value_seq_len; i += 8){
        sdata[rcl * 8 + kl * 4 + 0] = (i + kl * 4 + 0 < value_seq_len) ? A_offset[i + 0] : 0.0;
        sdata[rcl * 8 + kl * 4 + 1] = (i + kl * 4 + 1 < value_seq_len) ? A_offset[i + 1] : 0.0;
        sdata[rcl * 8 + kl * 4 + 2] = (i + kl * 4 + 2 < value_seq_len) ? A_offset[i + 2] : 0.0;
        sdata[rcl * 8 + kl * 4 + 3] = (i + kl * 4 + 3 < value_seq_len) ? A_offset[i + 3] : 0.0;
        
        sdata[128 + kcl * 16 + nl * 4 + 0] = (i + kcl < value_seq_len && hm * 16 + nl * 4 + 0 < head_dim) ? B_offset[i + 0 * param.max_kv_len] : 0.0;
        sdata[128 + kcl * 16 + nl * 4 + 1] = (i + kcl < value_seq_len && hm * 16 + nl * 4 + 1 < head_dim) ? B_offset[i + 1 * param.max_kv_len] : 0.0;
        sdata[128 + kcl * 16 + nl * 4 + 2] = (i + kcl < value_seq_len && hm * 16 + nl * 4 + 2 < head_dim) ? B_offset[i + 2 * param.max_kv_len] : 0.0;
        sdata[128 + kcl * 16 + nl * 4 + 3] = (i + kcl < value_seq_len && hm * 16 + nl * 4 + 3 < head_dim) ? B_offset[i + 3 * param.max_kv_len] : 0.0;


        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_load(sga[0], (const threadgroup float*)sdata, 8);
        simdgroup_load(sga[1], ((const threadgroup float*)sdata) + 64, 8);
        
        simdgroup_load(sgb[0], ((const threadgroup float*)sdata) + 128, 16);
        simdgroup_load(sgb[1], ((const threadgroup float*)sdata) + 136, 16);
        
        simdgroup_multiply_accumulate(sgd[0], sga[0], sgb[0], sgd[0]);
        simdgroup_multiply_accumulate(sgd[1], sga[1], sgb[0], sgd[1]);
        simdgroup_multiply_accumulate(sgd[2], sga[0], sgb[1], sgd[2]);
        simdgroup_multiply_accumulate(sgd[3], sga[1], sgb[1], sgd[3]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_store(sgd[0], (threadgroup float*)sdata, 8);
    simdgroup_store(sgd[1], (threadgroup float*)sdata + 64, 8);
    simdgroup_store(sgd[2], (threadgroup float*)sdata + 128, 8);
    simdgroup_store(sgd[3], (threadgroup float*)sdata + 192, 8);
    
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // [N2, M2, M8, N8]
    auto xy_out = output + ((sl * 16 + rcl) * head_num + z) * head_dim + hm * 16 + kl * 8 + 0;
    if(sl * 16 + rcl < q_seq_len) {
        if(hm * 16 + kl * 8 + 0 < head_dim) {
            xy_out[0] =  sdata[(kl * 16 + rcl) * 8 + 0];
        }
        if(hm * 16 + kl * 8 + 1 < head_dim) {
            xy_out[1] =  sdata[(kl * 16 + rcl) * 8 + 1];
        }
        if(hm * 16 + kl * 8 + 2 < head_dim) {
            xy_out[2] =  sdata[(kl * 16 + rcl) * 8 + 2];
        }
        if(hm * 16 + kl * 8 + 3 < head_dim) {
            xy_out[3] =  sdata[(kl * 16 + rcl) * 8 + 3];
        }
        if(hm * 16 + kl * 8 + 4 < head_dim) {
            xy_out[4] =  sdata[(kl * 16 + rcl) * 8 + 4];
        }
        if(hm * 16 + kl * 8 + 5 < head_dim) {
            xy_out[5] =  sdata[(kl * 16 + rcl) * 8 + 5];
        }
        if(hm * 16 + kl * 8 + 6 < head_dim) {
            xy_out[6] =  sdata[(kl * 16 + rcl) * 8 + 6];
        }
        if(hm * 16 + kl * 8 + 7 < head_dim) {
            xy_out[7] =  sdata[(kl * 16 + rcl) * 8 + 7];
        }
    }

#else
    const int x = gid.x; // kv_seq_len
    const int y = gid.y; // head_num
    const int z = gid.z; // head_dim
    if (x >= param.query_seq_len || y >= param.head_num || z >= param.head_dim) {
        return;
    }
    int group = param.group;
    int yin = y / group;
    int q_seq_len = param.query_seq_len;
    int value_seq_len = param.key_seq_len;
    int head_num = param.head_num;
    int head_dim = param.head_dim;
    const int stride = head_num * head_dim / group;
    const int offset_head = yin * head_dim + z;

    device const T *A_offset = input0 + (y * q_seq_len + x) * value_seq_len;
    device const T *B_offset = past_value + offset_head * param.max_kv_len;
    float out = 0.0;
    
    for(int i = 0; i < value_seq_len; ++i){
        float A0 = (float)A_offset[i];
        float B = (float)B_offset[i];
        out += A0 * B;
    }
    output[ x * stride * group + (y * head_dim + z)] = out;
#endif
}

kernel void decode_qkv(const device T* input0 [[buffer(0)]],
    device T* output [[buffer(1)]],
    device T* past_value [[buffer(2)]],
    constant Param& param [[buffer(3)]],
#ifdef SIMD_GROUP_REDUCE
    uint3 gid[[threadgroup_position_in_grid]],
    uint  tiisg[[thread_index_in_simdgroup]],
    uint  sgitg[[simdgroup_index_in_threadgroup]]
#else
    uint3 gid[[thread_position_in_grid]]
#endif
) {
    const int x = gid.x; // query_seq_len
    const int y = gid.y; // head_num
    const int z = gid.z; // head_dim
    if (x >= param.query_seq_len || y >= param.head_num || z >= param.head_dim) {
        return;
    }

    int yin = y / param.group;
    int value_seq_len = param.key_seq_len;

    int head_dim = param.head_dim;

    const int offset_head = (yin * head_dim + z) * param.max_kv_len;

    device const T *A_offset = input0 + y * value_seq_len;
    device T *Pastvalue_offset = past_value + offset_head;
    float out = 0;
    
#ifdef SIMD_GROUP_REDUCE
    for(int i = tiisg; i < value_seq_len; i+=SIMD_GROUP_WIDTH){
        float A = (float)A_offset[i];
        float B = (float)Pastvalue_offset[i];
        
        out += A * B;
    }
    out = simd_sum(out);
    if(tiisg == 0) {
        output[(y * head_dim + z)] = (T)out;
    }
#else
    for(int i = 0; i < value_seq_len; i++){
        float A = (float)A_offset[i];
        float B = (float)Pastvalue_offset[i];
        
        out += A * B;
    }
    output[(y * head_dim + z)] = (T)out;
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
    int flat_length;
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
    
    auto axis_off = gid.y * s.axis_length * s.inside_size + gid.x;
    auto axis_in  = in + axis_off;
    auto axis_out = out + axis_off;
    
    // get max
    float max1 = -INFINITY;
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
    for (int i = tiisg; i < s.axis_length; i+=SIMD_GROUP_WIDTH) {
        axis_out[i * s.inside_size] = ftype(exp(float(axis_in[i * s.inside_size]) - float(max1)) / sum1);
    }
}

)metal";

#endif/* MNN_SUPPORT_TRANSFORMER_FUSE */
#endif

