//
//  MetalSlice.metal
//  MNN
//
//  Created by MNN on 2018/08/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <metal_stdlib>
#include "MetalDefine.metal"

using namespace metal;

struct slice_shape {
    int output_width;
    int output_height;
    int output_outer;
    int output_inner;
    int input_outer;
    int input_inner;
};

kernel void slice_channel(const device ftype4 *in       [[buffer(0)]], // without offset
                          device ftype4 *out            [[buffer(1)]],
                          constant slice_shape& s       [[buffer(2)]],
                          device int2& channel_range    [[buffer(3)]],
                          uint3 gid                     [[thread_position_in_grid]]) {
    if ((int)gid.x >= s.output_width || (int)gid.y >= s.output_height) return;
    
    int xy_off = gid.y * s.input_inner + gid.x;
    
    int2 z_range = channel_range / 4;
    int z = z_range.x + gid.z;
    int r = channel_range.x % 4;
    if (r == 0) {
        out[int(gid.z) * s.output_outer + xy_off] =
        in [z          * s.input_outer  + xy_off];
    } else if (z_range.x == z_range.y) {
        auto v0 = in[(z + 0) * s.input_outer + xy_off];
        auto value = ftype4(        v0[0 + r],
                            r < 3 ? v0[1 + r] : 0,
                            r < 2 ? v0[2 + r] : 0,
                                                0);
        out[int(gid.z) * s.output_outer + xy_off] = value;
    } else {
        auto v0 = in[(z + 0) * s.input_outer + xy_off];
        auto v1 = in[(z + 1) * s.input_outer + xy_off];
        auto value = ftype4(        v0[0 + r],
                            r < 3 ? v0[1 + r] : v1[r - 3],
                            r < 2 ? v0[2 + r] : v1[r - 2],
                                                v1[r - 1]);
        out[int(gid.z) * s.output_outer + xy_off] = value;
    }
}

kernel void slice_width(const device ftype4 *in  [[buffer(0)]], // with offset
                        device ftype4 *out       [[buffer(1)]],
                        constant slice_shape &s [[buffer(2)]],
                        uint3 gid               [[thread_position_in_grid]]) {
    if ((int)gid.x >= s.output_width || (int)gid.y >= s.output_height) return;
    out[int(gid.z) * s.output_outer + int(gid.y) * s.output_inner + int(gid.x)] =
    in [int(gid.z) * s.input_outer  + int(gid.y) * s.input_inner  + int(gid.x)];
}

kernel void slice_tf(const device ftype *in      [[buffer(0)]], // with offset
                     device ftype *out           [[buffer(1)]],
                     constant slice_shape &s    [[buffer(2)]],
                     uint3 gid                  [[thread_position_in_grid]]) {
    if ((int)gid.x >= s.output_width || (int)gid.y >= s.output_height) return;
    out[int(gid.y) * s.output_outer + int(gid.x) * s.output_inner + int(gid.z)] =
    in [int(gid.y) * s.input_outer  + int(gid.x) * s.input_inner  + int(gid.z)];
}
