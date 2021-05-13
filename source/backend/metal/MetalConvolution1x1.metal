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
#define CONV_UNROLL_L (8)

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
    
    float4 result = float4(biasTerms[gid.y]);
    for (auto z = 0; z < cst.input_group_slice; z++, xy_in += cst.input_size) {
        result += float4(*xy_in * xy_wt[z]);
    }
    *xy_out = activate(ftype4(result), cst.activation);
}

kernel void conv1x1_g1z4(const device ftype4 *in            [[buffer(0)]],
                         device ftype4 *out                 [[buffer(1)]],
                         constant conv1x1_constants& cst    [[buffer(2)]],
                         const device ftype4x4 *wt          [[buffer(3)]],
                         const device ftype4 *biasTerms     [[buffer(4)]],
                         uint3 gid                          [[thread_position_in_grid]]) {
    if ((int)gid.x * CONV_UNROLL >= cst.output_size || (int)gid.y >= cst.output_slice || (int)gid.z >= cst.batch) return;
    
    int rx = gid.x * CONV_UNROLL;
    int uz = gid.y;
    auto xy_wt = wt + uz * cst.input_slice;
    auto xy_in0  = in  + (int)gid.z * cst.input_slice  * cst.input_size                         + rx + 0;
    auto xy_in1  = in  + (int)gid.z * cst.input_slice  * cst.input_size                         + rx + 1;
    auto xy_in2  = in  + (int)gid.z * cst.input_slice  * cst.input_size                         + rx + 2;
    auto xy_in3  = in  + (int)gid.z * cst.input_slice  * cst.input_size                         + rx + 3;
    auto xy_out = out + (int)gid.z * cst.output_slice * cst.output_size + uz * cst.output_size + rx;
    auto biasValue = float4(biasTerms[uz]);
    float4 result0 = biasValue, result1 = biasValue, result2 = biasValue, result3 = biasValue;
    int computeSize = min(cst.output_size - rx, CONV_UNROLL);
    if (computeSize == CONV_UNROLL) {
        for (auto z = 0; z < cst.input_slice; z++) {
            auto in40 = *xy_in0;
            auto in41 = *xy_in1;
            auto in42 = *xy_in2;
            auto in43 = *xy_in3;
            auto w = xy_wt[z];
            
            result0 += float4(in40 * w);
            result1 += float4(in41 * w);
            result2 += float4(in42 * w);
            result3 += float4(in43 * w);
            xy_in0 += cst.input_size;
            xy_in1 += cst.input_size;
            xy_in2 += cst.input_size;
            xy_in3 += cst.input_size;
        }
    } else if (computeSize == 3) {
        for (auto z = 0; z < cst.input_slice; z++) {
            auto in40 = *xy_in0;
            auto in41 = *xy_in1;
            auto in42 = *xy_in2;
            auto w = xy_wt[z];
            
            result0 += float4(in40 * w);
            result1 += float4(in41 * w);
            result2 += float4(in42 * w);
            xy_in0 += cst.input_size;
            xy_in1 += cst.input_size;
            xy_in2 += cst.input_size;
        }
    } else if (computeSize == 2) {
        for (auto z = 0; z < cst.input_slice; z++) {
            auto in40 = *xy_in0;
            auto in41 = *xy_in1;
            auto w = xy_wt[z];
            
            result0 += float4(in40 * w);
            result1 += float4(in41 * w);
            xy_in0 += cst.input_size;
            xy_in1 += cst.input_size;
        }
    } else {
        for (auto z = 0; z < cst.input_slice; z++) {
            auto in40 = *xy_in0;
            auto w = xy_wt[z];
            
            result0 += float4(in40 * w);
            xy_in0 += cst.input_size;
        }
    }
    
    /* true                               */ *xy_out = activate(ftype4(result0), cst.activation);
    if (computeSize > 1) {xy_out[1] = activate(ftype4(result1), cst.activation); }
    if (computeSize > 2) {xy_out[2] = activate(ftype4(result2), cst.activation); }
    if (computeSize > 3) {xy_out[3] = activate(ftype4(result3), cst.activation); }
}


kernel void conv1x1_g1z8(const device ftype4 *in            [[buffer(0)]],
                         device ftype4 *out                 [[buffer(1)]],
                         constant conv1x1_constants& cst    [[buffer(2)]],
                         const device ftype4x4 *wt          [[buffer(3)]],
                         const device ftype4 *biasTerms     [[buffer(4)]],
                         uint3 gid                          [[thread_position_in_grid]]) {
    if ((int)gid.x * CONV_UNROLL_L >= cst.output_size || (int)gid.y >= cst.output_slice || (int)gid.z >= cst.batch) return;

    int rx = gid.x * CONV_UNROLL_L;
    int uz = gid.y;
    auto xy_wt = wt + uz * cst.input_slice;
    auto xy_in0  = in  + (int)gid.z * cst.input_slice  * cst.input_size + rx + 0;
    auto xy_in1  = in  + (int)gid.z * cst.input_slice  * cst.input_size + rx + 1;
    auto xy_in2  = in  + (int)gid.z * cst.input_slice  * cst.input_size + rx + 2;
    auto xy_in3  = in  + (int)gid.z * cst.input_slice  * cst.input_size + rx + 3;
    auto xy_in4  = in  + (int)gid.z * cst.input_slice  * cst.input_size + rx + 4;
    auto xy_in5  = in  + (int)gid.z * cst.input_slice  * cst.input_size + rx + 5;
    auto xy_in6  = in  + (int)gid.z * cst.input_slice  * cst.input_size + rx + 6;
    auto xy_in7  = in  + (int)gid.z * cst.input_slice  * cst.input_size + rx + 7;

    auto xy_out = out + (int)gid.z * cst.output_slice * cst.output_size + uz * cst.output_size + rx;
    auto biasValue = float4(biasTerms[uz]);
    float4 result0 = biasValue, result1 = biasValue, result2 = biasValue, result3 = biasValue;
    float4 result4 = biasValue, result5 = biasValue, result6 = biasValue, result7 = biasValue;

    int computeSize = min(cst.output_size - rx, CONV_UNROLL_L);
    for (auto z = 0; z < cst.input_slice; z++) {
            auto in40 = *xy_in0;
            auto in41 = *xy_in1;
            auto in42 = *xy_in2;
            auto in43 = *xy_in3;
            auto in44 = *xy_in4;
            auto in45 = *xy_in5;
            auto in46 = *xy_in6;
            auto in47 = *xy_in7;

            auto w = xy_wt[z];

            result0 += float4(in40 * w);
            result1 += float4(in41 * w);
            result2 += float4(in42 * w);
            result3 += float4(in43 * w);
            result4 += float4(in44 * w);
            result5 += float4(in45 * w);
            result6 += float4(in46 * w);
            result7 += float4(in47 * w);
            xy_in0 += cst.input_size;
            xy_in1 += cst.input_size;
            xy_in2 += cst.input_size;
            xy_in3 += cst.input_size;
            xy_in4 += cst.input_size;
            xy_in5 += cst.input_size;
            xy_in6 += cst.input_size;
            xy_in7 += cst.input_size;
    }

    /* true                               */ *xy_out = activate(ftype4(result0), cst.activation);
    if (computeSize > 1) {xy_out[1] = activate(ftype4(result1), cst.activation); }
    if (computeSize > 2) {xy_out[2] = activate(ftype4(result2), cst.activation); }
    if (computeSize > 3) {xy_out[3] = activate(ftype4(result3), cst.activation); }
    if (computeSize > 4) {xy_out[4] = activate(ftype4(result4), cst.activation); }
    if (computeSize > 5) {xy_out[5] = activate(ftype4(result5), cst.activation); }
    if (computeSize > 6) {xy_out[6] = activate(ftype4(result6), cst.activation); }
    if (computeSize > 7) {xy_out[7] = activate(ftype4(result7), cst.activation); }
}

