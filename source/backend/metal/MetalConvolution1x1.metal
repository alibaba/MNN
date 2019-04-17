//
//  MetalConvolution1x1.metal
//  MNN
//
//  Created by MNN on 2019/02/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <metal_stdlib>
#include "MetalConvolutionActivation.metal"

using namespace metal;
using namespace MNN;

#define CONV_UNROLL (4)

struct conv1x1_constants {
    int input_size;
    int input_group_slice;
    int input_slice;
    int output_size;
    int output_group_slice;
    int output_slice;
    int batch;
    conv_activation_type activation;
};

kernel void conv1x1(const device ftype4 *in         [[buffer(0)]],
                    device ftype4 *out              [[buffer(1)]],
                    constant conv1x1_constants& cst [[buffer(2)]],
                    const device ftype4x4 *wt       [[buffer(3)]],
                    const device ftype4 *biasTerms  [[buffer(4)]],
                    uint3 gid                       [[thread_position_in_grid]]) {
    if ((int)gid.x >= cst.output_size || (int)gid.y >= cst.output_slice || (int)gid.z >= cst.batch) return;
    
    int g = gid.y / cst.output_group_slice;
    auto xy_wt  = wt                                                    + (int)gid.y * cst.input_group_slice;
    auto xy_in  = in  + (int)gid.z * cst.input_slice  * cst.input_size  +          g * cst.input_size  + (int)gid.x;
    auto xy_out = out + (int)gid.z * cst.output_slice * cst.output_size + (int)gid.y * cst.output_size + (int)gid.x;
    
    float4 result = float4(biasTerms[(short)gid.y]);
    for (auto z = 0; z < cst.input_group_slice; z++, xy_in += cst.input_size) {
        result += float4(*xy_in * xy_wt[z]);
    }
    *xy_out = activate(ftype4(result), cst.activation);
}

kernel void qntconv1x1(const device char4 *in           [[buffer(0)]],
                       device ftype4 *out               [[buffer(1)]],
                       constant conv1x1_constants& cst  [[buffer(2)]],
                       const device char4x4 *wt         [[buffer(3)]],
                       const device ftype4 *biasTerms   [[buffer(4)]],
                       const device float4 *alpha       [[buffer(5)]],
                       uint3 gid                        [[thread_position_in_grid]]) {
    if ((int)gid.x >= cst.output_size || (int)gid.y >= cst.output_slice || (int)gid.z >= cst.batch) return;
    
    int g = gid.y / cst.output_group_slice;
    auto xy_wt  = wt                                                    + (int)gid.y * cst.input_group_slice;
    auto xy_in  = in  + (int)gid.z * cst.input_slice  * cst.input_size  +          g * cst.input_size  + (int)gid.x;
    auto xy_out = out + (int)gid.z * cst.output_slice * cst.output_size + (int)gid.y * cst.output_size + (int)gid.x;
    
    float4 result = 0;
    float4 a4 = alpha[(short)gid.y];
    float4 b4 = float4(biasTerms[(short)gid.y]);
    for (auto z = 0; z < cst.input_group_slice; z++, xy_in += cst.input_size) {
        result += float4(*xy_in) * float4x4(xy_wt[z]);
    }
    *xy_out = activate(ftype4(result * a4 + b4), cst.activation);
}

kernel void conv1x1_g1z4(const device ftype4 *in            [[buffer(0)]],
                         device ftype4 *out                 [[buffer(1)]],
                         constant conv1x1_constants& cst    [[buffer(2)]],
                         const device ftype4x4 *wt          [[buffer(3)]],
                         const device ftype4 *biasTerms     [[buffer(4)]],
                         uint3 gid                          [[thread_position_in_grid]]) {
    if ((int)gid.x >= cst.output_size || (int)gid.y * CONV_UNROLL >= cst.output_slice || (int)gid.z >= cst.batch) return;
    
    int uz = gid.y * CONV_UNROLL;
    auto xy_wt0 = wt + uz * cst.input_slice;
    auto xy_wt1 = uz + 1 < cst.output_slice ? xy_wt0 + cst.input_slice : nullptr;
    auto xy_wt2 = uz + 2 < cst.output_slice ? xy_wt1 + cst.input_slice : nullptr;
    auto xy_wt3 = uz + 3 < cst.output_slice ? xy_wt2 + cst.input_slice : nullptr;
    auto xy_in  = in  + (int)gid.z * cst.input_slice  * cst.input_size                         + (int)gid.x;
    auto xy_out = out + (int)gid.z * cst.output_slice * cst.output_size + uz * cst.output_size + (int)gid.x;
    
    float4 result0 = 0, result1 = 0, result2 = 0, result3 = 0;
    for (auto z = 0; z < cst.input_slice; z++, xy_in += cst.input_size) {
        auto in4 = *xy_in;
        /* true */  result0 += float4(in4 * xy_wt0[z]);
        if (xy_wt1) result1 += float4(in4 * xy_wt1[z]);
        if (xy_wt2) result2 += float4(in4 * xy_wt2[z]);
        if (xy_wt3) result3 += float4(in4 * xy_wt3[z]);
    }
    
    /* true                               */ *xy_out = activate(ftype4(result0 + float4(biasTerms[uz + 0])), cst.activation);
    if (xy_wt1) { xy_out += cst.output_size; *xy_out = activate(ftype4(result1 + float4(biasTerms[uz + 1])), cst.activation); }
    if (xy_wt2) { xy_out += cst.output_size; *xy_out = activate(ftype4(result2 + float4(biasTerms[uz + 2])), cst.activation); }
    if (xy_wt3) { xy_out += cst.output_size; *xy_out = activate(ftype4(result3 + float4(biasTerms[uz + 3])), cst.activation); }
}

kernel void qntconv1x1_g1z4(const device char4 *in          [[buffer(0)]],
                            device ftype4 *out              [[buffer(1)]],
                            constant conv1x1_constants& cst [[buffer(2)]],
                            const device char4x4 *wt        [[buffer(3)]],
                            const device ftype4 *biasTerms  [[buffer(4)]],
                            const device float4 *alpha      [[buffer(5)]],
                            uint3 gid                       [[thread_position_in_grid]]) {
    if ((int)gid.x >= cst.output_size || (int)gid.y * CONV_UNROLL >= cst.output_slice || (int)gid.z >= cst.batch) return;
    
    int uz = gid.y * CONV_UNROLL;
    auto xy_wt0 = wt + uz * cst.input_slice;
    auto xy_wt1 = uz + 1 < cst.output_slice ? xy_wt0 + cst.input_slice : nullptr;
    auto xy_wt2 = uz + 2 < cst.output_slice ? xy_wt1 + cst.input_slice : nullptr;
    auto xy_wt3 = uz + 3 < cst.output_slice ? xy_wt2 + cst.input_slice : nullptr;
    auto xy_in  = in  + (int)gid.z * cst.input_slice  * cst.input_size                         + (int)gid.x;
    auto xy_out = out + (int)gid.z * cst.output_slice * cst.output_size + uz * cst.output_size + (int)gid.x;
    
    float4 result0 = 0, result1 = 0, result2 = 0, result3 = 0;
    for (auto z = 0; z < cst.input_slice; z++, xy_in += cst.input_size) {
        auto in4 = float4(*xy_in);
        /* true */   result0 += in4 * float4x4(xy_wt0[z]);
        if (xy_wt1)  result1 += in4 * float4x4(xy_wt1[z]);
        if (xy_wt2)  result2 += in4 * float4x4(xy_wt2[z]);
        if (xy_wt3)  result3 += in4 * float4x4(xy_wt3[z]);
    }
    
    /* true                         */       *xy_out = activate(ftype4(result0 * alpha[uz + 0] + float4(biasTerms[uz + 0])), cst.activation);
    if (xy_wt1) { xy_out += cst.output_size; *xy_out = activate(ftype4(result1 * alpha[uz + 1] + float4(biasTerms[uz + 1])), cst.activation); }
    if (xy_wt2) { xy_out += cst.output_size; *xy_out = activate(ftype4(result2 * alpha[uz + 2] + float4(biasTerms[uz + 2])), cst.activation); }
    if (xy_wt3) { xy_out += cst.output_size; *xy_out = activate(ftype4(result3 * alpha[uz + 3] + float4(biasTerms[uz + 3])), cst.activation); }
}
