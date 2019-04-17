//
//  MetalScale.metal
//  MNN
//
//  Created by MNN on 2018/08/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <metal_stdlib>
#include "MetalDefine.metal"

using namespace metal;

struct scale_shape {
    int size;
    int steps;
    int batch;
};

kernel void scale_tf(const device ftype *in         [[buffer(0)]],
                     device ftype *out              [[buffer(1)]],
                     constant scale_shape &s        [[buffer(2)]],
                     const device float *scales     [[buffer(3)]],
                     const device float *biasTerms  [[buffer(4)]],
                     uint2 gid                      [[thread_position_in_grid]]) {
    if ((int)gid.x >= s.steps || (int)gid.y >= s.size * s.batch) return;
    
    out[int(gid.y) * s.steps + int(gid.x)] =
    in [int(gid.y) * s.steps + int(gid.x)] * ftype(scales[int(gid.x)]) + ftype(biasTerms[int(gid.x)]);
}

kernel void scale_ca(const device ftype4 *in        [[buffer(0)]],
                     device ftype4 *out             [[buffer(1)]],
                     constant scale_shape &s        [[buffer(2)]],
                     const device float4 *scales    [[buffer(3)]],
                     const device float4 *biasTerms [[buffer(4)]],
                     uint2 gid                      [[thread_position_in_grid]]) {
    if ((int)gid.x >= s.size || (int)gid.y >= s.steps * s.batch) return;

    int z = gid.y % s.steps;
    out[int(gid.y) * s.size + int(gid.x)] =
    in [int(gid.y) * s.size + int(gid.x)] * ftype4(scales[z]) + ftype4(biasTerms[z]);
}
