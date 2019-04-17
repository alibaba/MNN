//
//  MetalPack.metal
//  MNN
//
//  Created by MNN on 2018/11/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <metal_stdlib>
#include "MetalDefine.metal"

using namespace metal;

struct pack_constants {
    packed_int3 dimension;
    int src_stride;
    int dst_stride;
};

#define data(tgt, gid, c) tgt[int(gid.z) * c.tgt##_stride + int(gid.y) * c.dimension[0] + int(gid.x)]

kernel void pack_float(const device ftype *src      [[buffer(0)]],
                       device ftype *dst            [[buffer(1)]],
                       constant pack_constants& c   [[buffer(2)]],
                       uint3 gid                    [[thread_position_in_grid]]) {
    if (all(gid < uint3(c.dimension))) data(dst, gid, c) = data(src, gid, c);
}

kernel void pack_int32(const device int *src        [[buffer(0)]],
                       device int *dst              [[buffer(1)]],
                       constant pack_constants& c   [[buffer(2)]],
                       uint3 gid                    [[thread_position_in_grid]]) {
    if (all(gid < uint3(c.dimension))) data(dst, gid, c) = data(src, gid, c);
}
