//
//  MetalSliceTF.metal
//  MNN
//
//  Created by MNN on 2018/11/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <metal_stdlib>
#include "MetalDefine.metal"

using namespace metal;

template <typename T>
static void slice_tf(const device T *in,
                     const device int *in_strides,
                     device T *out,
                     const device int *out_strides,
                     const device int *beigins,
                     const int dims,
                     const uint gid) {
    int out_off = int(gid);
    int in_off = 0;
    for (int i = 0; i < dims; i++) {
        auto extent = out_off / out_strides[i] + beigins[i];
        in_off += extent * in_strides[i];
        out_off %= out_strides[i];
    }
    out[int(gid)] = in[in_off];
}

kernel void slice_tf_int32(const device int *in             [[buffer(0)]],
                           const device int *in_strides     [[buffer(1)]],
                           device int *out                  [[buffer(2)]],
                           const device int *out_strides    [[buffer(3)]],
                           const device int *beigins        [[buffer(4)]],
                           constant int &dims               [[buffer(5)]],
                           const uint gid                   [[thread_position_in_grid]]) {
    slice_tf<int>(in, in_strides, out, out_strides, beigins, dims, gid);
}

kernel void slice_tf_float(const device ftype *in           [[buffer(0)]],
                           const device int *in_strides     [[buffer(1)]],
                           device ftype *out                [[buffer(2)]],
                           const device int *out_strides    [[buffer(3)]],
                           const device int *beigins        [[buffer(4)]],
                           constant int &dims               [[buffer(5)]],
                           const uint gid                   [[thread_position_in_grid]]) {
    slice_tf<ftype>(in, in_strides, out, out_strides, beigins, dims, gid);
}
