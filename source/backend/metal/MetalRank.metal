//
//  MetalRank.metal
//  MNN
//
//  Created by MNN on 2018/11/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <metal_stdlib>

using namespace metal;

kernel void rank(constant int &dims [[buffer(0)]],
                 device int *out    [[buffer(1)]]) {
    out[0] = dims;
}
