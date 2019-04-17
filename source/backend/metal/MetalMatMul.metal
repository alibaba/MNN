//
//  MetalMatMul.metal
//  MNN
//
//  Created by MNN on 2018/09/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <metal_stdlib>
#include "MetalDefine.metal"

using namespace metal;

struct matmul_shape {
    int input0_width;
    int input0_height;
    int input0_size;
    int input1_width;
    int input1_size;
    int output_width;
    int output_height;
    int output_size;
};

kernel void matmul(const device ftype *in0  [[buffer(0)]],
                   const device ftype *in1  [[buffer(1)]],
                   device ftype *out        [[buffer(2)]],
                   constant matmul_shape &s [[buffer(3)]],
                   uint3 gid[[thread_position_in_grid]]) {
    if ((int)gid.x >= s.output_width || (int)gid.y >= s.output_height || s.input0_height != s.input1_width) return;
    
    float value = 0.f;
    for (int i = 0; i < s.input0_height; i++) {
        auto value0 = in0[int(gid.z) * s.input0_size + i * s.input0_width + int(gid.x)];
        auto value1 = in1[i * s.output_height + int(gid.y)];
        value += float(value0) * float(value1);
    }
    out[int(gid.z) * s.output_size + int(gid.y) * s.output_width + int(gid.x)] = ftype(value);
}
