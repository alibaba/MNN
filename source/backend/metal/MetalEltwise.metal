//
//  MetalEltwise.metal
//  MNN
//
//  Created by MNN on 2018/08/09.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <metal_stdlib>
#include "MetalDefine.metal"

using namespace metal;

kernel void eltwise_prod(device const ftype4 *in0   [[buffer(0)]],
                         device const ftype4 *in1   [[buffer(1)]],
                         device ftype4 *out         [[buffer(2)]],
                         uint gid                   [[thread_position_in_grid]]) {
    out[(int)gid] = in0[(int)gid] * in1[(int)gid];
}

kernel void eltwise_max(device const ftype4 *in0    [[buffer(0)]],
                        device const ftype4 *in1    [[buffer(1)]],
                        device ftype4 *out          [[buffer(2)]],
                        uint gid                    [[thread_position_in_grid]]) {
    out[(int)gid] = max(in0[(int)gid], in1[(int)gid]);
}

kernel void eltwise_add(device const ftype4 *in0    [[buffer(0)]],
                        device const ftype4 *in1    [[buffer(1)]],
                        device ftype4 *out          [[buffer(2)]],
                        uint gid                    [[thread_position_in_grid]]) {
    out[(int)gid] = in0[(int)gid] + in1[(int)gid];
}
