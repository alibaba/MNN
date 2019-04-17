//
//  MetalGatherV2.metal
//  MNN
//
//  Created by MNN on 2018/11/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <metal_stdlib>
#include "MetalDefine.metal"

using namespace metal;

template<typename T>
static inline void gatherv2(const device T *in,
                            const device int *_dims,
                            const device int *indices,
                            constant int &_axis,
                            device T *out,
                            uint gid) {
    // prepare
    int n_dims = _dims[0];
    auto dims = _dims + 1;
    int axis = _axis;
    if (axis < 0) axis = n_dims + axis;
    
    // calc
    int fragments = 1, stride = 1;
    for (int i = 0; i <= axis; i++) fragments *= dims[i];
    for (int i = axis + 1; i < n_dims; i++) stride *= dims[i];
    
    // gather
    auto indice = indices[int(gid)];
    if (0 <= indice && indice < fragments) {
        auto s_out = out + gid * stride;
        auto s_in = in + indice * stride;
        for (int i = 0; i < stride; i++) {
            s_out[i] = s_in[i];
        }
    }
}

kernel void gatherv2_float(const device ftype *in       [[buffer(0)]],
                           const device int *_dims      [[buffer(1)]],
                           const device int *indices    [[buffer(2)]],
                           constant int &_axis          [[buffer(3)]],
                           device ftype *out            [[buffer(4)]],
                           uint gid                     [[thread_position_in_grid]]) {
    gatherv2<ftype>(in, _dims, indices, _axis, out, gid);
}

kernel void gatherv2_int32(const device int *in         [[buffer(0)]],
                           const device int *_dims      [[buffer(1)]],
                           const device int *indices    [[buffer(2)]],
                           constant int &_axis          [[buffer(3)]],
                           device int *out              [[buffer(4)]],
                           uint gid                     [[thread_position_in_grid]]) {
    gatherv2<int>(in, _dims, indices, _axis, out, gid);
}
