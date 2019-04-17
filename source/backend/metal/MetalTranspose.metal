//
//  MetalTranspose.metal
//  MNN
//
//  Created by MNN on 2018/11/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <metal_stdlib>
#include "MetalDefine.metal"

using namespace metal;

kernel void transpose_prepare(const device int *_dims   [[buffer(0)]],
                              const device int *perm    [[buffer(1)]],
                              device int *perm_strides  [[buffer(2)]],
                              uint gid                  [[thread_position_in_grid]]) {
    int ndims = _dims[0];
    if ((int)gid >= ndims) return;
    
    auto dims = _dims + 1;
    int index = perm[int(gid)];
    int stride = 1;
    while (index < ndims - 1) {
        stride *= dims[index + 1];
        index++;
    }
    perm_strides[int(gid)] = stride;
}

kernel void transpose(const device ftype *in            [[buffer(0)]],
                      const device int *perm_strides    [[buffer(1)]],
                      device ftype *out                 [[buffer(2)]],
                      const device int *_out_strides    [[buffer(3)]],
                      uint gid                          [[thread_position_in_grid]]) {
    int n_out = _out_strides[0];
    auto out_strides = _out_strides + 1;
    
    int off = gid;
    auto off_in = in;
    for (int i = 0; i < n_out; i++) {
        auto dim_index = off / out_strides[i];
        off = off %  out_strides[i];
        off_in += dim_index * perm_strides[i];
    }
    out[int(gid)] = *off_in;
}
