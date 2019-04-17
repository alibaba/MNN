//
//  MetalReshape.metal
//  MNN
//
//  Created by MNN on 2018/09/03.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <metal_stdlib>
#include "MetalDefine.metal"

using namespace metal;

struct reshape_dims {
    int input_channel_size;     // input_width * input_height
    int input_batch_size;       // input_width * input_height * input_channel
    int input_batch_slices;     // (input_channel + 3) / 4
    int output_channel_size;    // output_width * output_height
    int output_batch_size;      // output_width * output_height * output_channel
    int output_batch_slices;    // (output_channel + 3) / 4
    int output_slice_size;      // output_width * output_height * 4
    int output_slices;          // (output_channel + 3) / 4 * output_batch
};

kernel void reshape(const device ftype4 *in     [[buffer(0)]],
                    device ftype4 *out          [[buffer(1)]],
                    constant reshape_dims& dims [[buffer(2)]],
                    uint3 gid                   [[thread_position_in_grid]]) {
    if ((int)gid.x >= dims.output_channel_size || (int)gid.z >= dims.output_slices) return; // gid.y == 0
    
    int base = (gid.z / dims.output_batch_slices) * dims.output_batch_size
             + (gid.z % dims.output_batch_slices) * dims.output_slice_size
             +  gid.x;
    int4 index = base + int4(0, 1, 2, 3) * dims.output_channel_size;
    int4 ib = index / dims.input_batch_size;    index = index % dims.input_batch_size;
    int4 ic = index / dims.input_channel_size;  index = index % dims.input_channel_size;
    int4 iz = ib * dims.input_batch_slices + ic / 4;
    int4 ir = ic % 4;
    
    out[int(gid.z) * dims.output_channel_size + int(gid.x)] = {
        in[iz[0] * dims.input_channel_size + index[0]][ir[0]],
        in[iz[1] * dims.input_channel_size + index[1]][ir[1]],
        in[iz[2] * dims.input_channel_size + index[2]][ir[2]],
        in[iz[3] * dims.input_channel_size + index[3]][ir[3]]
    };
}
