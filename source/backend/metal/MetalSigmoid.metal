//
//  MetalSigmoid.metal
//  MNN
//
//  Created by MNN on 2018/09/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <metal_stdlib>
#include "MetalDefine.metal"

using namespace metal;

kernel void sigmoid_x1(const device ftype *in   [[buffer(0)]],
                       device ftype *out        [[buffer(1)]],
                       uint gid                 [[thread_position_in_grid]]) {
    out[int(gid)] = 1.0 / (1.0 + exp(-in[int(gid)]));
}

kernel void sigmoid_x4(const device ftype4 *in  [[buffer(0)]],
                       device ftype4 *out       [[buffer(1)]],
                       uint gid                 [[thread_position_in_grid]]) {
    out[int(gid)] = 1.0 / (1.0 + exp(-in[int(gid)]));
}
