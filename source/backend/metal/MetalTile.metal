//
//  MetalTile.metal
//  MNN
//
//  Created by MNN on 2018/10/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <metal_stdlib>
#include "MetalDefine.metal"

using namespace metal;

struct tile_shape {
    int output_height;
    int output_width;
    int output_channel;
    
    int input_batch;
    int input_height;
    int input_width;
    int input_channel;
};

kernel void tile(const device ftype *in [[buffer(0)]],
                 device ftype *out      [[buffer(1)]],
                 constant tile_shape &s [[buffer(2)]],
                 uint3 gid              [[thread_position_in_grid]]) {
    if ((int)gid.x >= s.output_channel || (int)gid.y >= s.output_width) return;
    
    int ob = gid.z / s.output_height;
    int oh = gid.z % s.output_height;
    int ow = gid.y;
    int oc = gid.x;
    int ib = ob % s.input_batch;
    int ih = oh % s.input_height;
    int z = ib * s.input_height + ih;
    int y = ow % s.input_width;
    int x = oc % s.input_channel;
    
    out[int(gid.z) * s.output_width * s.output_channel + int(gid.y) * s.output_channel + int(gid.x)] =
    in [         z * s.input_width  * s.input_channel  +          y * s.input_channel  + x];
}
