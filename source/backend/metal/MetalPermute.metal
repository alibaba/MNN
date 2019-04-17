//
//  MetalPermute.metal
//  MNN
//
//  Created by MNN on 2018/08/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <metal_stdlib>
#include "MetalDefine.metal"

using namespace metal;

struct permute_shape {
    int input_width;
    int input_size;
    int input_slice;
    int output_width;
    int output_size;
    int output_slice;
    int output_height;
    int output_batch;
};

kernel void permute_to_cwh(const device ftype4 *in      [[buffer(0)]],
                           device ftype4 *out           [[buffer(1)]],
                           constant permute_shape& s    [[buffer(2)]],
                           uint3 gid                    [[thread_position_in_grid]]) {
    if (any(gid >= uint3(s.output_width, s.output_height, s.output_slice * s.output_batch))) return;
    
    out[(int)gid.z * s.output_size + (int)gid.y * s.output_width + (int)gid.x] =
    in [(int)gid.z * s.input_size  + (int)gid.x * s.input_width  + (int)gid.y];
}

kernel void permute_to_hcw(const device ftype *in       [[buffer(0)]],
                           device ftype4 *out           [[buffer(1)]],
                           constant permute_shape& s    [[buffer(2)]],
                           uint3 gid                    [[thread_position_in_grid]]) {
    if (any(gid >= uint3(s.output_width, s.output_height, s.output_slice * s.output_batch))) return;
    
    int b = gid.z / s.output_slice, oz = gid.z % s.output_slice;
    int iz = b * s.input_slice + gid.y / 4, ih = oz * 4, iw = gid.x, ir = gid.y % 4;
    auto off_in = in + (iz * s.input_size + ih * s.input_width + iw) * 4 + ir;
    auto v = ftype4(off_in[s.input_width * 0 * 4],
                    off_in[s.input_width * 1 * 4],
                    off_in[s.input_width * 2 * 4],
                    off_in[s.input_width * 3 * 4]);
    out[(int)gid.z * s.output_size + (int)gid.y * s.output_width + (int)gid.x] = v;
}

kernel void permute_to_hwc(const device ftype *in       [[buffer(0)]],
                           device ftype4 *out           [[buffer(1)]],
                           constant permute_shape& s    [[buffer(2)]],
                           uint3 gid                    [[thread_position_in_grid]]) {
    if (any(gid >= uint3(s.output_width, s.output_height, s.output_slice * s.output_batch))) return;
    
    int b = gid.z / s.output_slice, oz = gid.z % s.output_slice;
    int iz = b * s.input_slice + gid.x / 4, ih = oz * 4, iw = gid.y, ir = gid.x % 4;
    auto off_in = in + (iz * s.input_size + ih * s.input_width + iw) * 4 + ir;
    auto v = ftype4(off_in[s.input_width * 0 * 4],
                    off_in[s.input_width * 1 * 4],
                    off_in[s.input_width * 2 * 4],
                    off_in[s.input_width * 3 * 4]);
    out[(int)gid.z * s.output_size + (int)gid.y * s.output_width + (int)gid.x] = v;
}

kernel void permute_to_wch(const device ftype *in       [[buffer(0)]],
                           device ftype4 *out           [[buffer(1)]],
                           constant permute_shape& s    [[buffer(2)]],
                           uint3 gid                    [[thread_position_in_grid]]) {
    if (any(gid >= uint3(s.output_width, s.output_height, s.output_slice * s.output_batch))) return;
    
    int b = gid.z / s.output_slice, oz = gid.z % s.output_slice;
    int iz = b * s.input_slice + gid.y / 4, ih = gid.x, iw = oz * 4, ir = gid.y % 4;
    auto off_in = in + (iz * s.input_size + ih * s.input_width + iw) * 4 + ir;
    auto v = ftype4(off_in[0 * 4], off_in[1 * 4], off_in[2 * 4], off_in[3 * 4]);
    out[(int)gid.z * s.output_size + (int)gid.y * s.output_width + (int)gid.x] = v;
}

kernel void permute_to_whc(const device ftype *in       [[buffer(0)]],
                           device ftype4 *out           [[buffer(1)]],
                           constant permute_shape& s    [[buffer(2)]],
                           uint3 gid                    [[thread_position_in_grid]]) {
    if (any(gid >= uint3(s.output_width, s.output_height, s.output_slice * s.output_batch))) return;
    
    int b = gid.z / s.output_slice, oz = gid.z % s.output_slice;
    int iz = b * s.input_slice + gid.x / 4, ih = gid.y, iw = oz * 4, ir = gid.x % 4;
    auto off_in = in + (iz * s.input_size + ih * s.input_width + iw) * 4 + ir;
    auto v = ftype4(off_in[0 * 4], off_in[1 * 4], off_in[2 * 4], off_in[3 * 4]);
    out[(int)gid.z * s.output_size + (int)gid.y * s.output_width + (int)gid.x] = v;
}
