//
//  MetalCast.metal
//  MNN
//
//  Created by MNN on 2018/12/20.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <metal_stdlib>
#include "MetalDefine.metal"

using namespace metal;

kernel void cast_float_to_int32(const device ftype *in  [[buffer(0)]],
                                device int *out         [[buffer(1)]],
                                uint gid                [[thread_position_in_grid]]) {
    out[int(gid)] = int(in[int(gid)]);
}

kernel void cast_int32_to_float(const device int *in    [[buffer(0)]],
                                device ftype *out       [[buffer(1)]],
                                uint gid                [[thread_position_in_grid]]) {
    out[int(gid)] = ftype(in[int(gid)]);
}

kernel void cast_uint8_to_float(const device uchar *in  [[buffer(0)]],
                                device ftype *out       [[buffer(1)]],
                                uint gid                [[thread_position_in_grid]]) {
    out[int(gid)] = ftype(in[int(gid)]);
}

