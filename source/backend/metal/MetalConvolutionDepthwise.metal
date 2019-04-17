//
//  MetalConvolutionDepthwise.metal
//  MNN
//
//  Created by MNN on 2019/02/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <metal_stdlib>
#include "MetalConvolutionActivation.metal"

using namespace metal;
using namespace MNN;

struct conv_dw_cst {
    int input_width;
    int input_height;
    int input_size;
    int output_width;
    int output_height;
    int output_size;
    int slice;
    int batch;
    
    int kernel_x;
    int kernel_y;
    int kernel_size;
    int stride_x;
    int stride_y;
    int pad_x;
    int pad_y;
    int dilation_x;
    int dilation_y;
    conv_activation_type activation;
};

kernel void conv_depthwise(const device ftype4 *in          [[buffer(0)]],
                           device ftype4 *out               [[buffer(1)]],
                           constant conv_dw_cst& cst        [[buffer(2)]],
                           const device ftype4 *wt          [[buffer(3)]],
                           const device ftype4 *biasTerms   [[buffer(4)]],
                           ushort3 gid                      [[thread_position_in_grid]]) {
    if ((int)gid.x >= cst.output_width || (int)gid.y >= cst.output_height || (int)gid.z >= cst.slice * cst.batch) return;
    
    short oz = gid.z % cst.slice;
    short offset_x = (int)gid.x * cst.stride_x - cst.pad_x;
    short offset_y = (int)gid.y * cst.stride_y - cst.pad_y;
    short sx = max(0, (UP_DIV(-offset_x, cst.dilation_x)));
    short ex = min(cst.kernel_x, UP_DIV(cst.input_width - offset_x, cst.dilation_x));
    short sy = max(0, (UP_DIV(-offset_y, cst.dilation_y)));
    short ey = min(cst.kernel_y, UP_DIV(cst.input_height - offset_y, cst.dilation_y));
    offset_x += sx * cst.dilation_x;
    offset_y += sy * cst.dilation_y;

    auto z_wt  = wt  + (int)oz * cst.kernel_size;
    auto z_in  = in  + (int)gid.z * cst.input_size;
    auto z_out = out + (int)gid.z * cst.output_size + (int)gid.y * cst.output_width + (int)gid.x;
    float4 result = float4(biasTerms[(short)oz]);
    for (auto ky = sy, y = offset_y; ky < ey; ky++, y += cst.dilation_y) {
        for (auto kx = sx, x = offset_x; kx < ex; kx++, x += cst.dilation_x) {
            auto wt4 = z_wt[ky * cst.kernel_x   + kx];
            auto in4 = z_in[ y * cst.input_width + x];
            result += float4(in4 * wt4);
        }
    }
    *z_out = activate((ftype4)result, cst.activation);
}

kernel void qntconv_depthwise(const device char4 *in            [[buffer(0)]],
                              device ftype4 *out                [[buffer(1)]],
                              constant conv_dw_cst& cst         [[buffer(2)]],
                              const device char4 *wt            [[buffer(3)]],
                              const device ftype4 *biasTerms    [[buffer(4)]],
                              const device float4 *alpha        [[buffer(5)]],
                              uint3 gid                         [[thread_position_in_grid]]) {
    if ((int)gid.x >= cst.output_width || (int)gid.y >= cst.output_height || (int)gid.z >= cst.slice * cst.batch) return;
    
    short oz = gid.z % cst.slice;
    short offset_x = (int)gid.x * cst.stride_x - cst.pad_x;
    short offset_y = (int)gid.y * cst.stride_y - cst.pad_y;
    short sx = max(0, (UP_DIV(-offset_x, cst.dilation_x)));
    short ex = min(cst.kernel_x, UP_DIV(cst.input_width - offset_x, cst.dilation_x));
    short sy = max(0, (UP_DIV(-offset_y, cst.dilation_y)));
    short ey = min(cst.kernel_y, UP_DIV(cst.input_height - offset_y, cst.dilation_y));
    offset_x += sx * cst.dilation_x;
    offset_y += sy * cst.dilation_y;

    auto z_wt  = wt  + (int)oz * cst.kernel_size;
    auto z_in  = in  + (int)gid.z * cst.input_size;
    auto z_out = out + (int)gid.z * cst.output_size + (int)gid.y * cst.output_width + (int)gid.x;
    float4 result = 0;
    for (auto ky = sy, y = offset_y; ky < ey; ky++, y += cst.dilation_y) {
        for (auto kx = sx, x = offset_x; kx < ex; kx++, x += cst.dilation_x) {
            auto wt4 = float4(z_wt[ky * cst.kernel_x   + kx]);
            auto in4 = float4(z_in[ y * cst.input_width + x]);
            result += in4 * wt4;
        }
    }
    float4 a4 = alpha[(short)oz];
    float4 b4 = float4(biasTerms[(short)oz]);
    *z_out = activate(ftype4(result * a4 + b4), cst.activation);
}
