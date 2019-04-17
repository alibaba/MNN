//
//  MetalFill.metal
//  MNN
//
//  Created by MNN on 2018/10/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <metal_stdlib>

using namespace metal;

kernel void fill_x1(const constant int &in  [[buffer(0)]],
                    device int *out         [[buffer(1)]],
                    uint gid                [[thread_position_in_grid]]) {
    out[int(gid)] = in;
}

kernel void fill_x4(const constant int &in  [[buffer(0)]],
                    device int4 *out        [[buffer(1)]],
                    uint gid                [[thread_position_in_grid]]) {
    out[int(gid)] = int4(in);
}
