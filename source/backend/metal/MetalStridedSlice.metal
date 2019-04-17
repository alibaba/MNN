//
//  MetalStridedSlice.metal
//  MNN
//
//  Created by MNN on 2018/11/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <metal_stdlib>
#include "MetalDefine.metal"

using namespace metal;

kernel void strided_slice_prepare(const device int *_dims           [[buffer(0)]],
                                  const device int *_masks          [[buffer(1)]],
                                  const device int *begin           [[buffer(2)]],
                                  const device int *end             [[buffer(3)]],
                                  const device int *strided         [[buffer(4)]],
                                  device packed_int2 *in_slices     [[buffer(5)]],
                                  device packed_int2 *out_strides   [[buffer(6)]]) {
    auto dims_len = _dims[0];
    auto dims = _dims + 1;
    auto masks_len = _masks[0];
    auto masks = (const device packed_int3 *)(_masks + 1); // [begin, end, shrink]
    
    // calc shape
    for (int i = 0; i < masks_len; i++) {
        auto m = masks[i];
        auto begin_shape    = m[0] > 0 ? 0 : min(begin[i], dims[i]);
        auto end_shape      = m[1] > 0 ? dims[i] : min(end[i], dims[i]);
        auto strided_shape  = m[2] > 0 ? 1 : strided[i];
        
        if (begin_shape < 0) begin_shape += dims[i];
        if (end_shape < 0) end_shape += dims[i];
        if (end_shape < begin_shape) {
            int t = begin_shape;
            begin_shape = end_shape;
            end_shape = t;
            
            if (strided_shape < 0) {
                strided_shape = abs(strided_shape);
            } else {
                begin_shape = end_shape; // it's a temp solution according to CPU
            }
        }
        
        in_slices[i] = int2(begin_shape, strided_shape);
        if (m[2] == 0) {
            out_strides[i][0] = (end_shape - begin_shape - 1) / strided_shape + 1;
        } else {
            out_strides[i][0] = 1;
        }
    }
    
    for (int i = masks_len; i < dims_len; i++) {
        out_strides[i][0] = dims[i];
    }
    
    // calc stride
    out_strides[dims_len - 1][1] = 1;
    for (int i = dims_len - 2; i >= 0; i--) {
        out_strides[i][1] = out_strides[i + 1][1] * out_strides[i + 1][0];
    }
}

template <typename T> static inline void strided_slice(const device T *input,
                                                       device T *output,
                                                       const device int *_dims,
                                                       const device packed_int2 *out_strides,
                                                       const device packed_int2 *in_slices,
                                                       uint gid) {
    auto dims_len = _dims[0];
    auto dims = _dims + 1;
    
    int in_off = 0, out_off = gid;
    for (int d = 0; d < dims_len; d++) {
        auto in_ext = dims[d];
        auto out_stride = out_strides[d];
        auto slice = in_slices[d];
        auto out_ext = min(out_off / out_stride[1], out_stride[0] - 1);
        auto index = slice[0] + out_ext * slice[1];
        out_off -= out_ext * out_stride[1];
        in_off = in_off * in_ext + index;
    }
    output[int(gid)] = input[in_off];
}

kernel void strided_slice_int32(const device int *input                 [[buffer(0)]],
                                device int *output                      [[buffer(1)]],
                                const device int *_dims                 [[buffer(2)]],
                                const device packed_int2 *out_stride    [[buffer(3)]],
                                const device packed_int2 *in_slices     [[buffer(4)]],
                                uint gid                                [[thread_position_in_grid]]) {
    strided_slice<int>(input, output, _dims, out_stride, in_slices, gid);
}

kernel void strided_slice_float(const device ftype *input               [[buffer(0)]],
                                device ftype *output                    [[buffer(1)]],
                                const device int *_dims                 [[buffer(2)]],
                                const device packed_int2 *out_stride    [[buffer(3)]],
                                const device packed_int2 *in_slices     [[buffer(4)]],
                                uint gid                                [[thread_position_in_grid]]) {
    strided_slice<ftype>(input, output, _dims, out_stride, in_slices, gid);
}
