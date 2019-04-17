//
//  MetalSpatialProduct.metal
//  MNN
//
//  Created by MNN on 2018/08/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <metal_stdlib>
#include "MetalDefine.metal"

using namespace metal;

struct spartial_product_shape {
    int size;
    int slice;
};

kernel void spartial_product(const device ftype4 *in            [[buffer(0)]],
                             const device ftype4 *weight        [[buffer(1)]],
                             device ftype4 *out                 [[buffer(2)]],
                             constant spartial_product_shape& s [[buffer(3)]],
                             uint2 gid                          [[thread_position_in_grid]]) {
    if ((int)gid.x >= s.size || (int)gid.y >= s.slice) return;
    out[int(gid.y) * s.size + int(gid.x)] = in[int(gid.y) * s.size + int(gid.x)] * weight[int(gid.x)][0];
}
