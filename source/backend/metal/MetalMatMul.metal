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
    int M;
    int N;
    int K;
};

kernel void matmul(const device ftype *in0  [[buffer(0)]],
                   const device ftype *in1  [[buffer(1)]],
                   device ftype *out        [[buffer(2)]],
                   constant matmul_shape &s [[buffer(3)]],
                   uint2 gid[[thread_position_in_grid]]) {
    if ((int)gid.x >= s.K || (int)gid.y >= s.M) return;
    
    auto off_in0 = in0 + int(gid.y) * s.N;
    auto off_in1 = in1 + int(gid.x);
    float value = 0.f;
    for (int i = 0; i < s.N; i++, off_in0 += 1, off_in1 += s.K) {
        value += float(*off_in0) * float(*off_in1);
    }
    out[int(gid.y) * s.K + int(gid.x)] = ftype(value);
}
