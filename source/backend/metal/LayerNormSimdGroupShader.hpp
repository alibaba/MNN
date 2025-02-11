//
//  layerNormSimdGroupShader.hpp
//  MNN
//
//  Created by MNN on b'2024/12/30'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#if MNN_METAL_ENABLED

const char* gLayerNormSgReduce = R"metal(
#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;
struct layernorm_constants {
    int inside;
    int outside;
    float eps;
    int has_gamma_beta;
};

#define SIMD_GROUP_WIDTH 32
#define CONV_UNROLL (4)
#define CONV_UNROLL_L (8)

kernel void layernorm_x1_sg(const device ftype *in       [[buffer(0)]],
                         device ftype *out            [[buffer(1)]],
                         constant layernorm_constants& cst  [[buffer(2)]],
                         const device float *gamma    [[buffer(3)]],
                         const device float *beta     [[buffer(4)]],
                         uint3  gid  [[threadgroup_position_in_grid]],
                         uint  tiisg[[thread_index_in_simdgroup]],
                         uint  sgitg[[simdgroup_index_in_threadgroup]]) {
    if ((int)gid.x >= cst.inside || (int)gid.y >= cst.outside) {
        return;
    }
    auto in_data = in + gid.y * cst.inside;
    auto out_data = out + gid.y * cst.inside;

    float mean;
    float sum = 0.0f;
    float square_sum = 0.0f;
    
    for(int i = tiisg; i < cst.inside; i+=SIMD_GROUP_WIDTH) {
        sum += in_data[i];
    }
    sum = simd_sum(sum);
    mean = sum / cst.inside;
    
    for(int i = tiisg; i < cst.inside; i+=SIMD_GROUP_WIDTH) {
        float dis = (in_data[i] - mean);
        square_sum += dis * dis;
    }
    square_sum = simd_sum(square_sum);

    if(tiisg == 0) {
        float var = 1.0 / sqrt(square_sum / cst.inside + cst.eps);
        
        float norm = var * ((float)in_data[gid.x] - mean);
        if(cst.has_gamma_beta) {
            out_data[gid.x] = (ftype)(norm * gamma[gid.x] + beta[gid.x]);
        } else {
            out_data[gid.x] = (ftype)(norm);
        }
    }
}

kernel void layernorm_x4_sg(const device ftype4 *in       [[buffer(0)]],
                         device ftype4 *out            [[buffer(1)]],
                         constant layernorm_constants& cst  [[buffer(2)]],
                         const device float4 *gamma    [[buffer(3)]],
                         const device float4 *beta     [[buffer(4)]],
                         uint3  gid  [[threadgroup_position_in_grid]],
                         uint  tiisg[[thread_index_in_simdgroup]],
                         uint  sgitg[[simdgroup_index_in_threadgroup]]) {
    if ((int)gid.x >= cst.inside/4 || (int)gid.y >= cst.outside) {
        return;
    }
    auto in_data = in + gid.y * cst.inside/4;
    auto out_data = out + gid.y * cst.inside/4;

    float mean;
    float sum = 0.0f;
    float square_sum = 0.0f;
    
    for(int i = tiisg; i < cst.inside/4; i+=SIMD_GROUP_WIDTH) {
        sum += in_data[i].x;
        sum += in_data[i].y;
        sum += in_data[i].z;
        sum += in_data[i].w;
    }
    sum = simd_sum(sum);
    mean = sum / cst.inside;
    
    for(int i = tiisg; i < cst.inside/4; i+=SIMD_GROUP_WIDTH) {
        float dis = (in_data[i].x - mean);
        square_sum += dis * dis;
        dis = (in_data[i].y - mean);
        square_sum += dis * dis;
        dis = (in_data[i].z - mean);
        square_sum += dis * dis;
        dis = (in_data[i].w - mean);
        square_sum += dis * dis;
    }
    square_sum = simd_sum(square_sum);

    if(tiisg == 0) {
        float var = 1.0 / sqrt(square_sum / cst.inside + cst.eps);
        
        float4 norm = var * ((float4)in_data[gid.x] - mean);
        if(cst.has_gamma_beta) {
            out_data[gid.x] = (ftype4)(norm * gamma[gid.x] + beta[gid.x]);
        } else {
            out_data[gid.x] = (ftype4)(norm);
        }
    }
}


kernel void layernorm_x1_rms_sg(const device ftype *in       [[buffer(0)]],
                            device ftype *out            [[buffer(1)]],
                            constant layernorm_constants& cst  [[buffer(2)]],
                            const device float *gamma    [[buffer(3)]],
                            const device float *beta     [[buffer(4)]],
                            uint3  gid  [[threadgroup_position_in_grid]],
                            uint  tiisg[[thread_index_in_simdgroup]],
                            uint  sgitg[[simdgroup_index_in_threadgroup]]) {
    if ((int)gid.x >= cst.inside || (int)gid.y >= cst.outside) {
        return;
    }
    auto in_data = in + gid.y * cst.inside;
    auto out_data = out + gid.y * cst.inside;

    float square_sum = 0.0f;
    
    for(int i = tiisg; i < cst.inside; i+=SIMD_GROUP_WIDTH) {
        float dis = in_data[i];
        square_sum += dis * dis;
    }
    
    square_sum = simd_sum(square_sum);
    
    if(tiisg == 0) {
        float var = 1.0 / sqrt(square_sum / cst.inside + cst.eps);
        
        float norm = var * ((float)in_data[gid.x]);
        if(cst.has_gamma_beta) {
            out_data[gid.x] = (ftype)(norm * gamma[gid.x] + beta[gid.x]);
        } else {
            out_data[gid.x] = (ftype)(norm);
        }
    }
}

kernel void layernorm_x4_rms_sg(const device ftype4 *in       [[buffer(0)]],
                             device ftype4 *out            [[buffer(1)]],
                             constant layernorm_constants& cst  [[buffer(2)]],
                             const device float4 *gamma    [[buffer(3)]],
                             const device float4 *beta     [[buffer(4)]],
                             uint3  gid  [[threadgroup_position_in_grid]],
                             uint  tiisg[[thread_index_in_simdgroup]],
                             uint  sgitg[[simdgroup_index_in_threadgroup]]) {
    if ((int)gid.x >= cst.inside/4 || (int)gid.y >= cst.outside) {
        return;
    }

    int in_idx = gid.x;
    int out_idx = gid.y;

    auto in_data = in + out_idx * cst.inside/4;
    auto out_data = out + out_idx * cst.inside/4;

    float4 square_sum = 0.0f;
    float square_sum_all = 0.0f;
    for(int i = tiisg; i < cst.inside/4; i+=SIMD_GROUP_WIDTH) {
        float4 data = float4(in_data[i]);
        square_sum += data * data;
    }
    square_sum_all += (square_sum[0] + square_sum[1] + square_sum[2] + square_sum[3]);
    square_sum_all = simd_sum(square_sum_all);
    
    if(tiisg == 0) {
        float var = 1.0 / sqrt(square_sum_all / cst.inside + cst.eps);
        
        float4 norm = var * ((float4)in_data[in_idx]);
        if(cst.has_gamma_beta) {
            out_data[in_idx] = (ftype4)(norm * gamma[in_idx] + beta[in_idx]);
        } else {
            out_data[in_idx] = (ftype4)(norm);
        }
    }
}

kernel void layernorm_x16_rms_sg(const device ftype4 *in       [[buffer(0)]],
                             device ftype4 *out            [[buffer(1)]],
                             constant layernorm_constants& cst  [[buffer(2)]],
                             const device float4 *gamma    [[buffer(3)]],
                             const device float4 *beta     [[buffer(4)]],
                             uint3  gid  [[threadgroup_position_in_grid]],
                             uint  tiisg[[thread_index_in_simdgroup]],
                             uint  sgitg[[simdgroup_index_in_threadgroup]]) {
    int in_idx = gid.x * 4;
    int out_idx = gid.y;

    auto in_data = in + out_idx * cst.inside/4;
    auto out_data = out + out_idx * cst.inside/4;

    float4 square_sum = 0.0f;
    float square_sum_all = 0.0f;
    for(int i = tiisg; i < cst.inside/4; i+=SIMD_GROUP_WIDTH) {
        float4 data = float4(in_data[i]);
        square_sum += data * data;
    }
    square_sum_all += (square_sum[0] + square_sum[1] + square_sum[2] + square_sum[3]);
    square_sum_all = simd_sum(square_sum_all);
    float var = 1.0 / sqrt(square_sum_all / cst.inside + cst.eps);

    if(tiisg == 0) {
        float4 norm = var * ((float4)in_data[in_idx]);
        if(cst.has_gamma_beta) {
            out_data[in_idx] = (ftype4)(norm * gamma[in_idx] + beta[in_idx]);
        } else {
            out_data[in_idx] = (ftype4)(norm);
        }
    }
    if(tiisg == 1 && in_idx + 1 < cst.inside/4) {
        float4 norm = var * ((float4)in_data[in_idx+1]);
        if(cst.has_gamma_beta) {
            out_data[in_idx+1] = (ftype4)(norm * gamma[in_idx+1] + beta[in_idx+1]);
        } else {
            out_data[in_idx+1] = (ftype4)(norm);
        }
    }
    if(tiisg == 2 && in_idx + 2 < cst.inside/4) {
        float4 norm = var * ((float4)in_data[in_idx+2]);
        if(cst.has_gamma_beta) {
            out_data[in_idx+2] = (ftype4)(norm * gamma[in_idx+2] + beta[in_idx+2]);
        } else {
            out_data[in_idx+2] = (ftype4)(norm);
        }
    }
    if(tiisg == 3 && in_idx + 3 < cst.inside/4) {
        float4 norm = var * ((float4)in_data[in_idx+3]);
        if(cst.has_gamma_beta) {
            out_data[in_idx+3] = (ftype4)(norm * gamma[in_idx+3] + beta[in_idx+3]);
        } else {
            out_data[in_idx+3] = (ftype4)(norm);
        }
    }
}

)metal";

#endif

