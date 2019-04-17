//
//  MetalGather.metal
//  MNN
//
//  Created by MNN on 2018/11/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <metal_stdlib>
#include "MetalDefine.metal"

using namespace metal;

struct gather_constants {
    int fragments;
    int stride;
    int extent;
};

kernel void gather(const device ftype *in           [[buffer(0)]],
                   const device int *indices        [[buffer(1)]],
                   device ftype *out                [[buffer(2)]],
                   constant gather_constants &cst   [[buffer(3)]],
                   uint gid                         [[thread_position_in_grid]]) {
    if ((int)gid < cst.extent) {
        auto indice = indices[int(gid)];
        if (0 <= indice && indice < cst.fragments) {
            auto s_out = out +    gid * cst.stride;
            auto s_in  =  in + indice * cst.stride;
            for (int i = 0; i < cst.stride; i++) {
                s_out[i] = s_in[i];
            }
        }
    }
}
