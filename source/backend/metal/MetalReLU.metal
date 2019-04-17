//
//  MetalReLU.metal
//  MNN
//
//  Created by MNN on 2018/08/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <metal_stdlib>
#include "MetalDefine.metal"

using namespace metal;

kernel void relu_x1(const device ftype *in  [[buffer(0)]],
                    device ftype *out       [[buffer(1)]],
                    constant float &slope   [[buffer(2)]],
                    uint gid                [[thread_position_in_grid]]) {
    auto value = in[int(gid)];
    out[int(gid)] = fmax(value, 0) + fmin(value, 0) * ftype(slope);
}

kernel void relu_x4(const device ftype4 *in [[buffer(0)]],
                    device ftype4 *out      [[buffer(1)]],
                    constant float &slope   [[buffer(2)]],
                    uint gid                [[thread_position_in_grid]]) {
    auto value = in[int(gid)];
    out[int(gid)] = fmax(value, 0) + fmin(value, 0) * ftype(slope);
}
