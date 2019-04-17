//
//  MetalLRN.metal
//  MNN
//
//  Created by MNN on 2018/08/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <metal_stdlib>
#include "MetalDefine.metal"

using namespace metal;

struct lrn_constants {
    ftype alpha;
    ftype beta;
    int local_size;
    int channels;
    
    int input_width;
    int input_height;
    int input_size;
    int output_width;
    int output_height;
    int output_size;
};

kernel void lrn_across_channel(const device ftype4 *in      [[buffer(0)]],
                               device ftype4 *out           [[buffer(1)]],
                               constant lrn_constants& cst  [[buffer(2)]],
                               uint3 gid                    [[thread_position_in_grid]]) {
    if ((int)gid.x >= cst.output_width || (int)gid.y >= cst.output_height) return;
    
    auto z_in  = in  + (int)gid.z * cst.input_size  + (int)gid.y * cst.input_width  + (int)gid.x;
    auto z_out = out + (int)gid.z * cst.output_size + (int)gid.y * cst.output_width + (int)gid.x;
    
    int half_size = cst.local_size / 2;
    float4 sum = 0;
    for (int k = -half_size; k <= half_size; k++) {
        int4 j4 = int4(0, 1, 2, 3) + k;
        int4 z4 = int4(floor(float4(j4) / 4));
        int4 r4 = j4 - z4 * 4;
        int4 c4 = gid.z * 4 + j4;
        bool4 v4 = 0 <= c4 && c4 < cst.channels;
        
        if (v4[0]) { float in4 = float(z_in[z4[0] * cst.input_size][r4[0]]); sum[0] += in4 * in4; }
        if (v4[1]) { float in4 = float(z_in[z4[1] * cst.input_size][r4[1]]); sum[1] += in4 * in4; }
        if (v4[2]) { float in4 = float(z_in[z4[2] * cst.input_size][r4[2]]); sum[2] += in4 * in4; }
        if (v4[3]) { float in4 = float(z_in[z4[3] * cst.input_size][r4[3]]); sum[3] += in4 * in4; }
    }
    *z_out = *z_in * ftype4(pow(1.f + cst.alpha * sum, -cst.beta));
}

kernel void lrn_within_channel(const device ftype4 *in      [[buffer(0)]],
                               device ftype4 *out           [[buffer(1)]],
                               constant lrn_constants& cst  [[buffer(2)]],
                               uint3 gid                    [[thread_position_in_grid]]) {
    if ((int)gid.x >= cst.output_width || (int)gid.y >= cst.output_height) return;
    
    auto z_in  = in  + (int)gid.z * cst.input_size;
    auto z_out = out + (int)gid.z * cst.output_size + (int)gid.y * cst.output_width + (int)gid.x;
    int2 offset = int2(gid.xy) - cst.local_size / 2;
    float4 sum = 0;
    for (int w = 0; w < cst.local_size; w++) {
        for (int h = 0; h < cst.local_size; h++) {
            int x = offset.x + w;
            int y = offset.y + h;
            if (x >= 0 && y >= 0 && x < cst.input_width && y < cst.input_height) {
                float4 input = float4(z_in[y * cst.input_width + x]);
                sum += input * input;
            }
        }
    }
    
    auto input = z_in[(int)gid.y * cst.input_width + (int)gid.x];
    *z_out = input * ftype4(pow(1.f + cst.alpha * sum, -cst.beta));
}
