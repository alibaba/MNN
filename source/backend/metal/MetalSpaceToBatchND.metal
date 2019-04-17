//
//  MetalSpaceToBatchND.metal
//  MNN
//
//  Created by MNN on 2018/12/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <metal_stdlib>
#include "MetalDefine.metal"

using namespace metal;

struct space_to_batch_nd_shape {
    int block_width;
    int block_height;
    int padding_left;
    int padding_top;
    
    int input_width;
    int input_height;
    int input_slice;
    int input_batch;
    
    int output_width;
    int output_height;
    int output_slice;
    int output_batch;
};

kernel void space_to_batch_nd(const device ftype4 *in               [[buffer(0)]],
                              device ftype4 *out                    [[buffer(1)]],
                              constant space_to_batch_nd_shape &s   [[buffer(2)]],
                              uint3 gid                             [[thread_position_in_grid]]) {
    if (any(int3(gid) >= int3(s.output_width, s.output_height, s.output_slice * s.output_batch))) return;
    
    int ob = gid.z / s.output_slice;
    int oz = gid.z % s.output_slice;
    int oh = gid.y;
    int ow = gid.x;
    
    int ib = ob % s.input_batch;
    int stride = ob / s.input_batch;
    int stride_w = stride % s.block_width;
    int stride_h = stride / s.block_width;
    int iz = oz;
    int ih = oh * s.block_height + stride_h - s.padding_top;
    int iw = ow * s.block_width + stride_w - s.padding_left;
    int zz = ib * s.input_slice + iz;
    
    bool valid = 0 <= iw && iw < s.input_width && 0 <= ih && ih < s.input_height;
    out[(int)gid.z * s.output_height * s.output_width + oh * s.output_width + ow] = valid
        ? in[   zz * s.input_height  * s.input_width  + ih * s.input_width  + iw]
        : 0;
}
