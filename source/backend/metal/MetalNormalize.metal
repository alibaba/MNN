//
//  MetalNormalize.metal
//  MNN
//
//  Created by MNN on 2018/08/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <metal_stdlib>
#include "MetalDefine.metal"

using namespace metal;

struct normalize_constants {
    int size;
    int channel;
    int slice;
    
    int channel_shared;
    float eps;
};

static inline ftype4 normalize_filter(ftype4 value, int z, int limit) {
    return select(0, value, z * 4 + int4(0, 1, 2, 3) < limit);
}

kernel void normalize_across_spatial(const device ftype4 *in            [[buffer(0)]],
                                     device ftype4 *out                 [[buffer(1)]],
                                     const device float *scale          [[buffer(2)]],
                                     constant normalize_constants& cst  [[buffer(3)]]) {
    // calc sum
    float4 sum4 = 0;
    for (int z = 0; z < cst.slice; z++) {
        auto z_in = in + z * cst.size;
        for (int i = 0; i < cst.size; i++) {
            float4 value = float4(normalize_filter(z_in[i], z, cst.channel));
            sum4 += value * value;
        }
    }
    float sum = 1.f / sqrt(sum4[0] + sum4[1] + sum4[2] + sum4[3] + cst.eps);
    
    // calc result
    if (cst.channel_shared) {
        auto scaled_sum = scale[0] * sum;
        for (int z = 0; z < cst.slice; z++) {
            auto z_in = in + z * cst.size;
            auto z_out = out + z * cst.size;
            for (int i = 0; i < cst.size; i++) {
                float4 value = float4(normalize_filter(z_in[i], z, cst.channel));
                z_out[i] = ftype4(value * scaled_sum);
            }
        }
    } else {
        for (int z = 0; z < cst.slice; z++) {
            auto z_in = in + z * cst.size;
            auto z_out = out + z * cst.size;
            auto scaled_sum = ((const device float4 *)scale)[z] * sum;
            for (int i = 0; i < cst.size; i++) {
                float4 value = float4(normalize_filter(z_in[i], z, cst.channel));
                z_out[i] = ftype4(value * scaled_sum);
            }
        }
    }
}

kernel void normalize_across_channel(const device ftype4 *in            [[buffer(0)]],
                                     device ftype4 *out                 [[buffer(1)]],
                                     const device float *scale          [[buffer(2)]],
                                     constant normalize_constants& cst  [[buffer(3)]],
                                     uint gid                           [[thread_position_in_grid]]) {
    if ((int)gid >= cst.size) return;
    
    auto xy_in = in + gid;
    auto xy_out = out + gid;
    
    // calc sum
    float4 sum4 = 0;
    for (int z = 0; z < cst.slice; z++) {
        float4 value = float4(normalize_filter(xy_in[z * cst.size], z, cst.channel));
        sum4 += value * value;
    }
    float sum = 1.0 / sqrt(sum4[0] + sum4[1] + sum4[2] + sum4[3] + cst.eps);
    
    // calc result
    if (cst.channel_shared) {
        auto scaled_sum = scale[0] * sum;
        for (int z = 0; z < cst.slice; z++) {
            float4 value = float4(normalize_filter(xy_in[z * cst.size], z, cst.channel));
            xy_out[z * cst.size] = ftype4(value * scaled_sum);
        }
    } else {
        for (int z = 0; z < cst.slice; z++) {
            float4 value = float4(normalize_filter(xy_in[z * cst.size], z, cst.channel));
            float4 result = value * ((const device float4 *)scale)[z] * sum;
            xy_out[z * cst.size] = ftype4(result);
        }
    }
}
