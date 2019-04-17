//
//  MetalRange.metal
//  MNN
//
//  Created by MNN on 2018/11/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <metal_stdlib>
#include "MetalDefine.metal"

using namespace metal;

template <typename T>
static void range(constant T &start, constant T &delta, device T *flat, uint gid) {
    flat[int(gid)] = start + delta * gid;
}

kernel void range_int32(constant int &start     [[buffer(0)]],
                        constant int &delta     [[buffer(1)]],
                        device int *flat        [[buffer(2)]],
                        uint gid                [[thread_position_in_grid]]) {
    range<int>(start, delta, flat, gid);
}
                        
kernel void range_float(constant ftype &start   [[buffer(0)]],
                        constant ftype &delta   [[buffer(1)]],
                        device ftype *flat      [[buffer(2)]],
                        uint gid                [[thread_position_in_grid]]) {
    range<ftype>(start, delta, flat, gid);
}
