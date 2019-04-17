//
//  MetalSeLU.metal
//  MNN
//
//  Created by MNN on 2018/10/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <metal_stdlib>
#include "MetalDefine.metal"

using namespace metal;

kernel void selu_x1(const device ftype *in  [[buffer(0)]],
                    device ftype *out       [[buffer(1)]],
                    constant float2 &cst    [[buffer(2)]],
                    uint gid                [[thread_position_in_grid]]) {
    auto value = in[int(gid)];
    auto scale = ftype(cst[0]);
    auto alpha = ftype(cst[1]);
    out[int(gid)] = scale * select(value, alpha * (exp(value) - 1.h), value < 0.h);
}

kernel void selu_x4(const device ftype4 *in [[buffer(0)]],
                    device ftype4 *out      [[buffer(1)]],
                    constant float2 &cst    [[buffer(2)]],
                    uint gid                [[thread_position_in_grid]]) {
    auto value = in[int(gid)];
    auto scale = ftype4(cst[0]);
    auto alpha = ftype4(cst[1]);
    out[int(gid)] = scale * select(value, alpha * (exp(value) - 1.h), value < 0.h);
}
