//
//  MetalCrop.metal
//  MNN
//
//  Created by MNN on 2018/10/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <metal_stdlib>
#include "MetalDefine.metal"

using namespace metal;

struct crop_shape {
    int output_width;
    int output_height;
    int output_size;
    int input_width;
    int input_size;
    int offset_x;
    int offset_y;
};

kernel void crop(device const ftype4 *in    [[buffer(0)]],
                 device ftype4 *out         [[buffer(1)]],
                 constant crop_shape &s     [[buffer(2)]],
                 uint3 gid                  [[thread_position_in_grid]]) {
    if ((int)gid.x >= s.output_width || (int)gid.y >= s.output_height) return;
    
    out[int(gid.z) * s.output_size +               int(gid.y)  * s.output_width +              int(gid.x)] =
    in [int(gid.z) * s.input_size  + (s.offset_y + int(gid.y)) * s.input_width  + s.offset_x + int(gid.x)];
}
