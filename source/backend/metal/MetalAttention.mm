//
//  MetalAttention.mm
//  MNN
//
//  Created by MNN on b'2024/04/29'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <set>
#import "core/Macro.h"
#import "MetalCast.hpp"
#import "MetalBackend.hpp"
#import "MNNMetalContext.h"
#include "MNN_generated.h"

#if MNN_METAL_ENABLED
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

static const char* gMatMulDivMask = R"metal(
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
};
#define SIMD_GROUP_WIDTH 32

kernel void prefill(const device T* input0 [[buffer(0)]],
    const device T* input1 [[buffer(1)]],
    device T* output [[buffer(2)]],
    device T* past_key [[buffer(3)]],
#ifdef FLOAT_MASK
    const device T* mask [[buffer(4)]],
#else
    const device int* mask [[buffer(4)]],
#endif
    constant Param& param [[buffer(5)]],
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
    const int stride = head_num * head_dim / group;

    threadgroup float sdata[256] = {0.f};

    int idx_slq = slq * 16 + rcl < q_seq_len ? slq * 16 + rcl : q_seq_len - 1;
    int idx_slk = slk * 16 + rcl < k_seq_len ? slk * 16 + rcl : k_seq_len - 1;

    auto A_offset = input0 + (idx_slq * head_num + z) * head_dim + (0 * 2 + kl) * 4 + 0;
    auto B_offset = input1 + (idx_slk * head_num / group + zin) * head_dim + 0 * 8 + kl * 4 + 0;
       
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
                out0 = mask[((slq * 16 + rcl) * key_seq_len + (slk * 16 + kl * 8 + 0))] == 0 ? -FLT_MAX : out0;
            #endif
            xy_out[0] = out0;
        }
        if(slk * 16 + kl * 8 + 1 < k_seq_len) {
            auto out0 =  sdata[(kl * 16 + rcl) * 8 + 1] * Vscale;
            #ifdef FLOAT_MASK
                out0 = mask[((slq * 16 + rcl) * k_seq_len + (slk * 16 + kl * 8 + 1))] + out0;
            #else
                out0 = mask[((slq * 16 + rcl) * key_seq_len + (slk * 16 + kl * 8 + 1))] == 0 ? -FLT_MAX : out0;
            #endif
            xy_out[1] = out0;
        }
        if(slk * 16 + kl * 8 + 2 < k_seq_len) {
            auto out0 =  sdata[(kl * 16 + rcl) * 8 + 2] * Vscale;
            #ifdef FLOAT_MASK
                out0 = mask[((slq * 16 + rcl) * k_seq_len + (slk * 16 + kl * 8 + 2))] + out0;
            #else
                out0 = mask[((slq * 16 + rcl) * key_seq_len + (slk * 16 + kl * 8 + 2))] == 0 ? -FLT_MAX : out0;
            #endif
            xy_out[2] = out0;
        }
        if(slk * 16 + kl * 8 + 3 < k_seq_len) {
            auto out0 =  sdata[(kl * 16 + rcl) * 8 + 3] * Vscale;
            #ifdef FLOAT_MASK
                out0 = mask[((slq * 16 + rcl) * k_seq_len + (slk * 16 + kl * 8 + 3))] + out0;
            #else
                out0 = mask[((slq * 16 + rcl) * key_seq_len + (slk * 16 + kl * 8 + 3))] == 0 ? -FLT_MAX : out0;
            #endif
            xy_out[3] = out0;
        }
        if(slk * 16 + kl * 8 + 4 < k_seq_len) {
            auto out0 =  sdata[(kl * 16 + rcl) * 8 + 4] * Vscale;
            #ifdef FLOAT_MASK
                out0 = mask[((slq * 16 + rcl) * k_seq_len + (slk * 16 + kl * 8 + 4))] + out0;
            #else
                out0 = mask[((slq * 16 + rcl) * key_seq_len + (slk * 16 + kl * 8 + 4))] == 0 ? -FLT_MAX : out0;
            #endif
            xy_out[4] = out0;
        }
        if(slk * 16 + kl * 8 + 5 < k_seq_len) {
            auto out0 =  sdata[(kl * 16 + rcl) * 8 + 5] * Vscale;
            #ifdef FLOAT_MASK
                out0 = mask[((slq * 16 + rcl) * k_seq_len + (slk * 16 + kl * 8 + 5))] + out0;
            #else
                out0 = mask[((slq * 16 + rcl) * key_seq_len + (slk * 16 + kl * 8 + 5))] == 0 ? -FLT_MAX : out0;
            #endif
            xy_out[5] = out0;
        }
        if(slk * 16 + kl * 8 + 6 < k_seq_len) {
            auto out0 =  sdata[(kl * 16 + rcl) * 8 + 6] * Vscale;
            #ifdef FLOAT_MASK
                out0 = mask[((slq * 16 + rcl) * k_seq_len + (slk * 16 + kl * 8 + 6))] + out0;
            #else
                out0 = mask[((slq * 16 + rcl) * key_seq_len + (slk * 16 + kl * 8 + 6))] == 0 ? -FLT_MAX : out0;
            #endif
            xy_out[6] = out0;
        }
        if(slk * 16 + kl * 8 + 7 < k_seq_len) {
            auto out0 =  sdata[(kl * 16 + rcl) * 8 + 7] * Vscale;
            #ifdef FLOAT_MASK
                out0 = mask[((slq * 16 + rcl) * k_seq_len + (slk * 16 + kl * 8 + 7))] + out0;
            #else
                out0 = mask[((slq * 16 + rcl) * key_seq_len + (slk * 16 + kl * 8 + 7))] == 0 ? -FLT_MAX : out0;
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
    const int offset_head_kv = (y / param.group) * head_dim;
    const device T* A_offset = input0 + x * offset + offset_head;

    float Vscale = (float)param.scale;

    device const T* B_offset = input1 + z * offset / group + offset_head_kv;
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

kernel void decode(const device T* input0 [[buffer(0)]],
    const device T* input1 [[buffer(1)]],
    device T* output [[buffer(2)]],
    device T* past_key [[buffer(3)]],
#ifdef FLOAT_MASK
    const device T* mask [[buffer(4)]],
#else
    const device int* mask [[buffer(4)]],
#endif
    constant Param& param [[buffer(5)]],
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
    const int z = gid.z; // key_seq_len
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

    const device T *B_offset = input1 + offset_head_kv;
    float out = 0.0;

#ifdef SIMD_GROUP_REDUCE
    {
        for(int i = tiisg; i < head_dim; i+=SIMD_GROUP_WIDTH){
            float A = A_offset[i];
            float B = (float)Pastkey_offset[i];
            
            out += A * B;
        }
    }
    out = simd_sum(out);
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
#endif
}

)metal";

static const char* gCopyPastKV = R"metal(
#include <metal_stdlib>
using namespace metal;
struct Param {
    int head_count;
    int kv_seq_len;
    int src_offset;
    int dst_offset;
};
kernel void copy(const device T* input0 [[buffer(0)]],
    const device T* input1 [[buffer(1)]],
    device T* output0 [[buffer(2)]],
    device T* output1 [[buffer(3)]],
    constant Param& param [[buffer(4)]],
    uint3 gid[[thread_position_in_grid]]
) {
    const int x = gid.x; // head_num / group * head_dim / 4
    const int y = gid.y; // kv_seq_len
    if (x >= param.head_count || y >= param.kv_seq_len) {
        return;
    }
    const int index = y * param.head_count + x;
    output0[param.dst_offset + index] = input0[param.src_offset + index];
    output1[param.dst_offset + index] = input1[param.src_offset + index];
}
)metal";

static const char* gMatMulQKV = R"metal(

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
};
#define SIMD_GROUP_WIDTH 32
kernel void prefill(const device T* input0 [[buffer(0)]],
    const device T* input1 [[buffer(1)]],
    device T* output [[buffer(2)]],
    device T* past_value [[buffer(3)]],
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
    int zin = z / param.group;
    int qk_seq_len = param.query_seq_len;
    int value_seq_len = param.key_seq_len;
    int head_num = param.head_num;
    int head_dim = param.head_dim;
    const int stride = head_num * head_dim / group;

    threadgroup float sdata[256] = {0.f};

    int idx_qk_sl = sl * 16 + rcl < qk_seq_len ? (sl * 16 + rcl) : qk_seq_len - 1;

    auto A_offset = input0 + (z * qk_seq_len + idx_qk_sl) * value_seq_len + (0 * 2 + kl) * 4 + 0;
    auto B_offset = input1 + ((0 * 8 + kcl) * head_num / group + zin) * head_dim + hm * 16 + nl * 4 + 0;
       
    for(int i = 0; i < value_seq_len; i += 8){
        sdata[rcl * 8 + kl * 4 + 0] = (i + kl * 4 + 0 < value_seq_len) ? A_offset[i + 0] : 0.0;
        sdata[rcl * 8 + kl * 4 + 1] = (i + kl * 4 + 1 < value_seq_len) ? A_offset[i + 1] : 0.0;
        sdata[rcl * 8 + kl * 4 + 2] = (i + kl * 4 + 2 < value_seq_len) ? A_offset[i + 2] : 0.0;
        sdata[rcl * 8 + kl * 4 + 3] = (i + kl * 4 + 3 < value_seq_len) ? A_offset[i + 3] : 0.0;
        
        sdata[128 + kcl * 16 + nl * 4 + 0] = (i + kcl < value_seq_len && hm * 16 + nl * 4 + 0 < head_dim) ? B_offset[i * stride + 0] : 0.0;
        sdata[128 + kcl * 16 + nl * 4 + 1] = (i + kcl < value_seq_len && hm * 16 + nl * 4 + 1 < head_dim) ? B_offset[i * stride + 1] : 0.0;
        sdata[128 + kcl * 16 + nl * 4 + 2] = (i + kcl < value_seq_len && hm * 16 + nl * 4 + 2 < head_dim) ? B_offset[i * stride + 2] : 0.0;
        sdata[128 + kcl * 16 + nl * 4 + 3] = (i + kcl < value_seq_len && hm * 16 + nl * 4 + 3 < head_dim) ? B_offset[i * stride + 3] : 0.0;
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
    if(sl * 16 + rcl < qk_seq_len) {
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
    int yin = y / param.group;
    int qk_seq_len = param.query_seq_len;
    int value_seq_len = param.key_seq_len;
    int head_num = param.head_num;
    int head_dim = param.head_dim;
    const int stride = head_num * head_dim / group;
    const int offset_head = yin * head_dim + z;

    device const T *A_offset = input0 + (y * qk_seq_len + x) * value_seq_len;
    device const T *B_offset = input1 + offset_head;
    float out = 0.0;
    
    for(int i = 0; i < value_seq_len; ++i){
        float A0 = (float)A_offset[i];
        float B = (float)B_offset[i*stride];
        out += A0 * B;
    }
    output[ x * stride * group + (y * head_dim + z)] = out;
#endif
}

kernel void decode(const device T* input0 [[buffer(0)]],
    const device T* input1 [[buffer(1)]],
    device T* output [[buffer(2)]],
    device T* past_value [[buffer(3)]],
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
    const int y = gid.y; // head_num
    const int z = gid.z; // head_dim
    if (x >= param.query_seq_len || y >= param.head_num || z >= param.head_dim) {
        return;
    }
    int group = param.group;
    int yin = y / param.group;

    int value_seq_len = param.key_seq_len;
    int head_num = param.head_num;
    int head_dim = param.head_dim;
    const int stride = head_num * head_dim / group;
    const int offset_head = yin * head_dim + z;

    device const T *A_offset = input0 + y * value_seq_len;
    device T *Pastvalue_offset = past_value + offset_head;
    float out = 0;
    
#ifdef SIMD_GROUP_REDUCE
    for(int i = tiisg; i < value_seq_len; i+=SIMD_GROUP_WIDTH){
        float A = (float)A_offset[i];
        float B = (float)Pastvalue_offset[i * stride];
        
        out += A * B;
    }
    out = simd_sum(out);
    if(tiisg == 0) {
        output[(y * head_dim + z)] = (T)out;
    }
#else
    for(int i = 0; i < value_seq_len; i++){
        float A = (float)A_offset[i];
        float B = (float)Pastvalue_offset[i * stride];
        
        out += A * B;
    }
    output[(y * head_dim + z)] = (T)out;
#endif
}
)metal";

namespace MNN {
class AttentionBufExecution : public MetalExecution {
public:
    struct SharedCache {
        std::shared_ptr<Tensor> mPastKey;
        std::shared_ptr<Tensor> mPastValue;
        int mPastLength = 0, mMaxLength = 0, mKv_seq_len = 0;
    };
    AttentionBufExecution(Backend *backend, bool kv_cache);

    virtual ~AttentionBufExecution() = default;
    virtual void onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, id<MTLComputeCommandEncoder> encoder) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override {
        if (nullptr == dst) {
            return true;
        }
        auto exe = new AttentionBufExecution(bn, mKVCache);
        exe->mCache = mCache;
        *dst = exe;
        return true;
    }

private:
    void _init();
    void reallocKVCache();
    bool mKVCache;
    std::shared_ptr<SharedCache> mCache;
    float mScale;
    const int mExpandChunk = 64;
    bool mIsDecode = false;
    std::shared_ptr<Tensor> mTempQK, mTempSoftMax;
    int mNumHead = 0, mHeadDim = 0, mValueH = 0, mKvNumHead = 0;
    id<MTLComputePipelineState> mKernel_softmax = nil;
    
    id<MTLComputePipelineState> mKernel_qk = nil;
    id<MTLComputePipelineState> mKernel_qkv = nil;
    id<MTLComputePipelineState> mKernel_copy = nil;
    id<MTLComputePipelineState> mKernelPrefill_qk = nil;
    id<MTLComputePipelineState> mKernelPrefill_qkv = nil;
    id<MTLBuffer> mParamQKV;
    id<MTLBuffer> mParamSoftmax;
    id<MTLBuffer> mParamCopy;
};

struct Param {
    int query_seq_len;
    int key_seq_len;
    int head_num;
    int group;
    int head_dim;
    float scale;
};
AttentionBufExecution::AttentionBufExecution(Backend *backend, bool kv_cahce)
    : MetalExecution(backend) , mKVCache(kv_cahce) {
    _init();
}
void AttentionBufExecution::_init() {
    mCache.reset(new SharedCache);
    auto mtbn = static_cast<MetalBackend *>(backend());
    auto context = (__bridge MNNMetalContext *)mtbn->context();
    mParamQKV = [context newDeviceBuffer:sizeof(Param) access:CPUWriteOnly];
    mParamSoftmax = [context newDeviceBuffer:4 * sizeof(int) access:CPUWriteOnly];
    mParamCopy = [context newDeviceBuffer:4 * sizeof(int) access:CPUWriteOnly];
    mTempQK.reset(Tensor::createDevice<float>({0, 0}));
    mTempSoftMax.reset(Tensor::createDevice<float>({0, 0}));
}

void AttentionBufExecution::reallocKVCache() {
    if (!mKVCache || mCache->mPastLength < mCache->mMaxLength) {
        return;
    }

    auto mtbn = static_cast<MetalBackend *>(backend());
    int byte = 4;
    if(mtbn->useFp16InsteadFp32()) {
        byte = 2;
    }
    bool needCopy = mCache->mMaxLength > 0;

    size_t old_size = mKvNumHead * mCache->mMaxLength * mHeadDim * byte;
    mCache->mMaxLength = mCache->mPastLength + mExpandChunk;
    // past_key: [1, numhead, headdim, maxlen]
    auto new_key = Tensor::createDevice<float>({mCache->mMaxLength, mKvNumHead, mHeadDim});
    // past_value: [1, numhead, maxlen, headdim]
    auto new_value = Tensor::createDevice<float>({mCache->mMaxLength, mKvNumHead, mHeadDim});
    size_t size = mKvNumHead * mCache->mMaxLength * mHeadDim * byte;
    backend()->onAcquireBuffer(new_key, Backend::STATIC);
    backend()->onAcquireBuffer(new_value, Backend::STATIC);
    if (needCopy) {
        auto newKeyBuf = MetalBackend::getBuffer(new_key);
        auto new_key_ptr = (uint8_t*)[newKeyBuf.first contents] + newKeyBuf.second;
        auto keyBuf = MetalBackend::getBuffer(mCache->mPastKey.get());
        auto key_ptr = (uint8_t*)[keyBuf.first contents] + keyBuf.second;;
        ::memcpy(new_key_ptr, key_ptr, old_size);
        
        auto newValueBuf = MetalBackend::getBuffer(new_value);
        auto new_value_ptr = (uint8_t*)[newValueBuf.first contents] + newValueBuf.second;
        auto valueBuf = MetalBackend::getBuffer(mCache->mPastValue.get());
        auto value_ptr = (uint8_t*)[valueBuf.first contents] + valueBuf.second;
        ::memcpy(new_value_ptr, value_ptr, old_size);
    }
    mCache->mPastKey.reset(new_key);
    mCache->mPastValue.reset(new_value);
}


void AttentionBufExecution::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, id<MTLComputeCommandEncoder> encoder) {

    auto query = inputs[0];
    auto key = inputs[1];
    auto value = inputs[2];
    auto mask = inputs[3];
    auto mtbn = static_cast<MetalBackend *>(backend());
    auto context = (__bridge MNNMetalContext *)mtbn->context();
    auto shape = query->shape();
    int seq_len = shape[1];
    mNumHead = shape[2];
    mHeadDim = shape[3];
    mScale = 1.0 / sqrt(mHeadDim);
    mIsDecode = seq_len == 1;
    if (mCache->mPastLength == 0 || seq_len > 1) {
        mCache->mPastLength = seq_len;
    }
    mCache->mKv_seq_len = mCache->mPastLength;
    if(mIsDecode){
        mCache->mKv_seq_len = mCache->mPastLength + 1;
    }
    mKvNumHead = key->shape()[2];
    
    auto rt = (MetalRuntime*)mtbn->runtime();
    bool supportSimdReduce = rt->supportSimdGroupReduce();
    bool supportSimdMatrix = rt->supportSimdGroupMatrix();

    // decode and thread number not too large
    bool qkSimdReduce = supportSimdReduce && seq_len == 1 && mCache->mKv_seq_len * mNumHead < mHeadDim * 32;
    // loop_k can divide 8, thus avoid branch
    bool qkSimdMatrix = supportSimdMatrix && seq_len >= 16 && mHeadDim % 8 == 0;

    bool sftmSimdReduce = supportSimdReduce;
    bool qkvSimdReduce = supportSimdReduce && seq_len == 1 && mHeadDim * mNumHead < mCache->mKv_seq_len * 32;
    bool qkvSimdMatrix = supportSimdMatrix && seq_len >= 16;
    
    // Init Kernel
    bool float_mask = (mask->getType() == halide_type_of<float>());
    std::string T = "float";
    std::string T4 = "float4";
    if (mtbn->useFp16InsteadFp32()) {
        T = "half";
        T4 = "half4";
    }
    std::vector<std::string> qkKeys = {
        {"matmul_qk_div_mask", T}
    };
    if(qkSimdReduce) {
        qkKeys.emplace_back("SIMD_GROUP_REDUCE");
    }
    std::vector<std::string> qkvKeys = {
        {"matmul_qkv", T}
    };
    if(qkvSimdReduce) {
        qkvKeys.emplace_back("SIMD_GROUP_REDUCE");
    }
    std::vector<std::string> qkPrefillKeys = {
        {"matmul_qk_div_mask", T, "FOR_PREFILL"}
    };
    if (float_mask) {
        qkPrefillKeys.emplace_back("FLOAT_MASK");
    }
    if(qkSimdMatrix) {
        qkPrefillKeys.emplace_back("SIMD_GROUP_MATRIX");
    }
    std::vector<std::string> qkvPrefillKeys = {
        {"matmul_qkv", T, "FOR_PREFILL"}
    };
    if(qkvSimdMatrix) {
        qkvPrefillKeys.emplace_back("SIMD_GROUP_MATRIX");
    }
    std::vector<std::string> copyPastKeys = {
        {"pastkv_copy", T4}
    };
    std::vector<std::vector<std::string>> keys = {
        qkKeys,
        qkvKeys,
        qkPrefillKeys,
        qkvPrefillKeys,
        copyPastKeys
    };
    std::vector<const char*> sources = {
        gMatMulDivMask,
        gMatMulQKV,
        gMatMulDivMask,
        gMatMulQKV,
        gCopyPastKV
    };
    std::vector<id<MTLComputePipelineState>> pipelines(keys.size());
    for (int i=0; i<keys.size(); ++i) {
        auto pipeline = rt->findPipeline(keys[i]);
        if (nil == pipeline) {
            // Rebuild Pipeline
            MTLCompileOptions *option = [[MTLCompileOptions alloc] init];
            auto dic = [NSMutableDictionary dictionaryWithCapacity:0];
            [dic setValue:@(keys[i][1].c_str()) forKey:@"T"];
            for (int j=2; j<keys[i].size(); ++j) {
                [dic setValue:@"1" forKey:@(keys[i][j].c_str())];;
            }
            option.preprocessorMacros = dic;
            if(std::find(keys[i].begin(), keys[i].end(), "FOR_PREFILL") != keys[i].end()) {
                pipeline = mtbn->makeComputePipelineWithSourceOption(sources[i], "prefill", option);
            } else if(i == 4){
                pipeline = mtbn->makeComputePipelineWithSourceOption(sources[i], "copy", option);

            } else {
                pipeline = mtbn->makeComputePipelineWithSourceOption(sources[i], "decode", option);
            }
            rt->insertPipeline(keys[i], pipeline);
        }
        pipelines[i] = pipeline;
    }
    mKernel_qk = pipelines[0];
    mKernel_qkv = pipelines[1];
    mKernelPrefill_qk = pipelines[2];
    mKernelPrefill_qkv = pipelines[3];
    mKernel_copy = pipelines[4];
    MNN_ASSERT(nil != mKernel_qk);
    MNN_ASSERT(nil != mKernel_qkv);
    MNN_ASSERT(nil != mKernelPrefill_qk);
    MNN_ASSERT(nil != mKernelPrefill_qkv);
    MNN_ASSERT(nil != mKernel_copy);

    if(sftmSimdReduce) {
        mKernel_softmax = [context pipelineWithName:@"softmax_plane_sg" fp16:mtbn->useFp16InsteadFp32()];
    } else {
        mKernel_softmax = [context pipelineWithName:@"softmax_plane" fp16:mtbn->useFp16InsteadFp32()];
    }

    int group_size = mNumHead / mKvNumHead;

    reallocKVCache();
    bool needMalloc = mTempQK->length(0) != mNumHead;
    if (mIsDecode) {
        if (mTempQK->length(1) != mCache->mMaxLength) {
            needMalloc = true;
        }
        mTempQK->setLength(0, mNumHead);
        mTempQK->setLength(1, mCache->mMaxLength);
        mTempSoftMax->setLength(0, mNumHead);
        mTempSoftMax->setLength(1, mCache->mMaxLength);
    } else {
        if (mTempQK->length(1) != mCache->mPastLength * mCache->mPastLength) {
            needMalloc = true;
        }
        mTempQK->setLength(0, mNumHead);
        mTempQK->setLength(1, mCache->mPastLength * mCache->mPastLength);
        mTempSoftMax->setLength(0, mNumHead);
        mTempSoftMax->setLength(1, mCache->mPastLength * mCache->mPastLength);
    }
    if (needMalloc) {
        auto res = backend()->onAcquireBuffer(mTempQK.get(), Backend::STATIC) && backend()->onAcquireBuffer(mTempSoftMax.get(), Backend::STATIC);
        if (!res) {
            MNN_ERROR("MNN::Metal: OUT_OF_MEMORY when execute attention metal\n");
            return;
        }
    }

    // Update Parameters
    {
        auto param = (Param*)mParamQKV.contents;
        param->scale = mScale;
        param->head_dim = mHeadDim;
        param->key_seq_len = mCache->mKv_seq_len;
        param->head_num = mNumHead;
        param->group = group_size;
        param->query_seq_len = seq_len;
    }
    // For softmax parameter
    int inside, outside;
    if (mIsDecode) {
        inside = 1;
        outside = mNumHead;
    } else {
        inside = 1;
        outside = mCache->mKv_seq_len * mNumHead;
    }
    int axis = mCache->mKv_seq_len;
    {
        auto softmax = (int*)mParamSoftmax.contents;
        // Inside, axis, outside, plane(invalid)
        softmax[0] = inside;
        softmax[1] = axis;
        softmax[2] = outside;
        softmax[3] = 0;
    }
    // Run Copy Kernel
    {
        auto copyp = (int*)mParamCopy.contents;
        copyp[0] = mKvNumHead * mHeadDim / 4;
        
        int copy_line;
        if(mIsDecode) {
            copyp[1] = 1;
            copyp[2] = 0;
            copyp[3] = (mCache->mKv_seq_len - 1) * copyp[0];
            copy_line = 1;
        } else {
            copyp[1] = mCache->mKv_seq_len;
            copyp[2] = 0;
            copyp[3] = 0;
            copy_line = mCache->mKv_seq_len;
        }

        id<MTLComputePipelineState> pipeline = mKernel_copy;
        [encoder setComputePipelineState:pipeline];
        MetalBackend::setTensor(key, encoder, 0);
        MetalBackend::setTensor(value, encoder, 1);
        MetalBackend::setTensor(mCache->mPastKey.get(), encoder, 2);
        MetalBackend::setTensor(mCache->mPastValue.get(), encoder, 3);
        [encoder setBuffer:mParamCopy offset:0 atIndex:4];
        
        std::pair<MTLSize, MTLSize> gl;
        gl = [context computeBestGroupAndLocal:pipeline threads:MTLSizeMake(mKvNumHead * mHeadDim / 4, copy_line, 1)];

        [encoder dispatchThreadgroups:gl.first threadsPerThreadgroup:gl.second];

    }
    // Run QK Kernel
    {
        id<MTLComputePipelineState> pipeline;
        if (mIsDecode) {
            pipeline = mKernel_qk;
        } else {
            pipeline = mKernelPrefill_qk;
        }
        [encoder setComputePipelineState:pipeline];
        MetalBackend::setTensor(query, encoder, 0);
        MetalBackend::setTensor(key, encoder, 1);
        MetalBackend::setTensor(mTempQK.get(), encoder, 2);
        MetalBackend::setTensor(mCache->mPastKey.get(), encoder, 3);
        MetalBackend::setTensor(mask, encoder, 4);
        [encoder setBuffer:mParamQKV offset:0 atIndex:5];

        std::pair<MTLSize, MTLSize> gl;
        if(qkSimdReduce) {
            gl = std::make_pair(MTLSizeMake(seq_len, mNumHead, mCache->mKv_seq_len), MTLSizeMake(32, 1, 1));
        } else if(qkSimdMatrix) {
            gl = std::make_pair(MTLSizeMake(UP_DIV(seq_len, 16), UP_DIV(mCache->mKv_seq_len, 16), mNumHead), MTLSizeMake(32, 1, 1));
        } else {
            gl = [context computeBestGroupAndLocal:pipeline threads:MTLSizeMake(seq_len, mNumHead, mCache->mKv_seq_len)];
        }
        [encoder dispatchThreadgroups:gl.first threadsPerThreadgroup:gl.second];
    }
    // Run Softmax Kernel
    {
        [encoder setComputePipelineState:mKernel_softmax];
        MetalBackend::setTensor(mTempQK.get(), encoder, 0);
        MetalBackend::setTensor(mTempSoftMax.get(), encoder, 1);
        [encoder setBuffer:mParamSoftmax offset:0 atIndex:2];

        int thread_group_size = 32;
        std::pair<MTLSize, MTLSize> gl;
        if(sftmSimdReduce) {
            gl = std::make_pair(MTLSizeMake(inside, outside, 1), MTLSizeMake(thread_group_size, 1, 1));
        } else {
            gl = [context computeBestGroupAndLocal: mKernel_softmax threads:MTLSizeMake(inside, outside, 1)];
        }

        [encoder dispatchThreadgroups:gl.first threadsPerThreadgroup:gl.second];
    }
    // Run QKV Kernel
    {
        id<MTLComputePipelineState> pipeline;
        if (mIsDecode) {
            pipeline = mKernel_qkv;
        } else {
            pipeline = mKernelPrefill_qkv;
        }
        [encoder setComputePipelineState:pipeline];
        MetalBackend::setTensor(mTempSoftMax.get(), encoder, 0);
        MetalBackend::setTensor(value, encoder, 1);
        MetalBackend::setTensor(outputs[0], encoder, 2);
        MetalBackend::setTensor(mCache->mPastValue.get(), encoder, 3);
        [encoder setBuffer:mParamQKV offset:0 atIndex:4];
        std::pair<MTLSize, MTLSize> gl;
        if(qkvSimdReduce) {
            gl = std::make_pair(MTLSizeMake(seq_len, mNumHead, mHeadDim), MTLSizeMake(32, 1, 1));
        } else if(qkvSimdMatrix){
            gl = std::make_pair(MTLSizeMake(UP_DIV(seq_len, 16), UP_DIV(mHeadDim, 16), mNumHead), MTLSizeMake(32, 1, 1));
            //printf("qk:%d %d %d, softmax:%d %d %d, qkv:%d %d %d\n", seq_len, mNumHead, mCache->mKv_seq_len, inside, outside, 1, seq_len, mNumHead, mHeadDim);
        } else {
            gl = [context computeBestGroupAndLocal:pipeline threads:MTLSizeMake(seq_len, mNumHead, mHeadDim)];
        }
        [encoder dispatchThreadgroups:gl.first threadsPerThreadgroup:gl.second];
    }
    // Update status
    if(mIsDecode){
        mCache->mPastLength += 1;
        mCache->mKv_seq_len = mCache->mPastLength + 1;
    }
    //printf("qk:%d %d %d, softmax:%d %d %d, qkv:%d %d %d\n", seq_len, mNumHead, mCache->mKv_seq_len, inside, outside, 1, seq_len, mNumHead, mHeadDim);
    return;
}

class AttentionBufCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend, const std::vector<Tensor *> &outputs) const override {
        auto param = op->main_as_AttentionParam();
        return new AttentionBufExecution(backend, param->kv_cache());
    }
};
REGISTER_METAL_OP_TRANSFORMER_CREATOR(AttentionBufCreator, OpType_Attention);

} // namespace MNN
#endif/* MNN_SUPPORT_TRANSFORMER_FUSE */
#endif

