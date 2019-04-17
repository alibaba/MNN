//
//  MetalTFQuantizedConv2D.metal
//  MNN
//
//  Created by MNN on 2018/11/20.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <metal_stdlib>
#include "MetalDefine.metal"

using namespace metal;
using namespace MNN;

#define CONV_UNROLL (4)

struct tfconv_constants {
    int input_width;
    int input_height;
    int input_size;
    int input_slice;
    int output_width;
    int output_height;
    int output_size;
    int output_slice;
    
    int kernel_x;
    int kernel_y;
    int kernel_size;
    int stride_x;
    int stride_y;
    int pad_x;
    int pad_y;
    int dilation_x;
    int dilation_y;
    
    int input_zero_point;
    int output_zero_point;
    int output_shift_before;
    int output_multiplier;
    int output_shift_after;
    int output_activation_min;
    int output_activation_max;
};

kernel void tfqntconv_z4(const device uchar4 *in        [[buffer(0)]],
                         device uchar4 *out             [[buffer(1)]],
                         constant tfconv_constants& cst [[buffer(2)]],
                         const device short4x4 *weights [[buffer(3)]],
                         const device int4 *biasTerms   [[buffer(4)]],
                         uint3 gid                      [[thread_position_in_grid]]) {
    if ((int)gid.x >= cst.output_width || (int)gid.y >= cst.output_height) return;
    
    auto unroll = int4(0, 1, 2, 3);
    auto unroll_z = unroll + gid.z * CONV_UNROLL;
    auto valids = unroll_z < cst.output_slice;
    auto dilation_w = cst.input_slice * cst.kernel_size;
    auto o_weights = weights + unroll_z[0] * dilation_w;
    
    int offset_x = (int)gid.x * cst.stride_x - cst.pad_x;
    int offset_y = (int)gid.y * cst.stride_y - cst.pad_y;
    int sx = max(0, (UP_DIV(-offset_x, cst.dilation_x)));
    int ex = min(cst.kernel_x, UP_DIV(cst.input_width - offset_x, cst.dilation_x));
    int kw = ex - sx;
    int sy = max(0, (UP_DIV(-offset_y, cst.dilation_y)));
    int ey = min(cst.kernel_y, UP_DIV(cst.input_height - offset_y, cst.dilation_y));
    int kh = ey - sy, dilation_h = cst.input_width * cst.dilation_y;
    offset_x += sx * cst.dilation_x;
    offset_y += sy * cst.dilation_y;
    
    float4x4 fresult = {
        /* true */  float4(biasTerms[unroll_z[0]]),
        valids[1] ? float4(biasTerms[unroll_z[1]]) : 0,
        valids[2] ? float4(biasTerms[unroll_z[2]]) : 0,
        valids[3] ? float4(biasTerms[unroll_z[3]]) : 0,
    };
    auto z_in = in + offset_y * cst.input_width + offset_x;
    auto z_weights = o_weights + sy * cst.kernel_x + sx;
    
    for (int z = 0; z < cst.input_slice; z++, z_in += cst.input_size, z_weights += cst.kernel_size) {
        auto y_in = z_in;
        auto y_weights = z_weights;
        for (int y = 0; y < kh; y++, y_in += dilation_h, y_weights += cst.kernel_x) {
            auto x_in = y_in;
            for (int x = 0; x < kw; x++, x_in += cst.dilation_x) {
                float4 input = float4(int4(*x_in) - cst.input_zero_point);
                
                /* true */     fresult[0] += input * float4x4(y_weights[x + 0 * dilation_w]);
                if (valids[1]) fresult[1] += input * float4x4(y_weights[x + 1 * dilation_w]);
                if (valids[2]) fresult[2] += input * float4x4(y_weights[x + 2 * dilation_w]);
                if (valids[3]) fresult[3] += input * float4x4(y_weights[x + 3 * dilation_w]);
            }
        }
    }
    int4 result[4] = { int4(fresult[0]), int4(fresult[1]), int4(fresult[2]), int4(fresult[3]) };
    
    auto z_out = out + unroll_z[0] * cst.output_size + gid.y * cst.output_width + gid.x;
    /* true */     { result[0] = saturate_round_x2_high_mul(result[0] * (1 << cst.output_shift_before), cst.output_multiplier); }
    if (valids[1]) { result[1] = saturate_round_x2_high_mul(result[1] * (1 << cst.output_shift_before), cst.output_multiplier); }
    if (valids[2]) { result[2] = saturate_round_x2_high_mul(result[2] * (1 << cst.output_shift_before), cst.output_multiplier); }
    if (valids[3]) { result[3] = saturate_round_x2_high_mul(result[3] * (1 << cst.output_shift_before), cst.output_multiplier); }
    /* true */     { result[0] = round_divide_by_pot(result[0], -cst.output_shift_after); }
    if (valids[1]) { result[1] = round_divide_by_pot(result[1], -cst.output_shift_after); }
    if (valids[2]) { result[2] = round_divide_by_pot(result[2], -cst.output_shift_after); }
    if (valids[3]) { result[3] = round_divide_by_pot(result[3], -cst.output_shift_after); }
    /* true */     { result[0] = clamp(result[0] + cst.output_zero_point, cst.output_activation_min, cst.output_activation_max); }
    if (valids[1]) { result[1] = clamp(result[1] + cst.output_zero_point, cst.output_activation_min, cst.output_activation_max); }
    if (valids[2]) { result[2] = clamp(result[2] + cst.output_zero_point, cst.output_activation_min, cst.output_activation_max); }
    if (valids[3]) { result[3] = clamp(result[3] + cst.output_zero_point, cst.output_activation_min, cst.output_activation_max); }
    /* true */     { z_out[0 * cst.output_size] = uchar4(result[0]); }
    if (valids[1]) { z_out[1 * cst.output_size] = uchar4(result[1]); }
    if (valids[2]) { z_out[2 * cst.output_size] = uchar4(result[2]); }
    if (valids[3]) { z_out[3 * cst.output_size] = uchar4(result[3]); }
}

kernel void tfqntconv(const device uchar4 *in         [[buffer(0)]],
                      device uchar4 *out              [[buffer(1)]],
                      constant tfconv_constants& cst  [[buffer(2)]],
                      const device short4x4 *weights  [[buffer(3)]],
                      const device int4 *biasTerms    [[buffer(4)]],
                      uint3 gid                       [[thread_position_in_grid]]) {
    if ((int)gid.x >= cst.output_width || (int)gid.y >= cst.output_height) return;
    
    auto o_weights = weights + gid.z * cst.input_slice * cst.kernel_size;
    
    int offset_x = (int)gid.x * cst.stride_x - cst.pad_x;
    int offset_y = (int)gid.y * cst.stride_y - cst.pad_y;
    int sx = max(0, (UP_DIV(-offset_x, cst.dilation_x)));
    int ex = min(cst.kernel_x, UP_DIV(cst.input_width - offset_x, cst.dilation_x));
    int kw = ex - sx;
    int sy = max(0, (UP_DIV(-offset_y, cst.dilation_y)));
    int ey = min(cst.kernel_y, UP_DIV(cst.input_height - offset_y, cst.dilation_y));
    int kh = ey - sy, dilation_h = cst.input_width * cst.dilation_y;
    offset_x += sx * cst.dilation_x;
    offset_y += sy * cst.dilation_y;
    
    float4 fresult = float4(biasTerms[int(gid.z)]);
    auto z_in = in + offset_y * cst.input_width + offset_x;
    auto z_weights = o_weights + (sy * cst.kernel_x + sx);
    for (int z = 0; z < cst.input_slice; z++, z_in += cst.input_size, z_weights += cst.kernel_size) {
        auto y_in = z_in;
        auto y_weights = z_weights;
        for (int y = 0; y < kh; y++, y_in += dilation_h, y_weights += cst.kernel_x) {
            auto x_in = y_in;
            for (int x = 0; x < kw; x++, x_in += cst.dilation_x) {
                float4 input = float4(int4(*x_in) - cst.input_zero_point);
                fresult += input * float4x4(y_weights[x]);
            }
        }
    }
    
    int4 result = int4(fresult);
    result = saturate_round_x2_high_mul(result * (1 << cst.output_shift_before), cst.output_multiplier);
    result = round_divide_by_pot(result, -cst.output_shift_after);
    result = clamp(result + cst.output_zero_point, cst.output_activation_min, cst.output_activation_max);
    out[int(gid.z) * cst.output_size + int(gid.y) * cst.output_width + int(gid.x)] = uchar4(result);
}

kernel void tfqntconv_depthwise(const device uchar4 *in         [[buffer(0)]],
                                device uchar4 *out              [[buffer(1)]],
                                constant tfconv_constants& cst  [[buffer(2)]],
                                const device short4 *weights    [[buffer(3)]],
                                const device int4 *biasTerms    [[buffer(4)]],
                                uint3 gid                       [[thread_position_in_grid]]) {
    if ((int)gid.x >= cst.output_width || (int)gid.y >= cst.output_height) return;
    
    int offset_x = (int)gid.x * cst.stride_x - cst.pad_x;
    int offset_y = (int)gid.y * cst.stride_y - cst.pad_y;
    int sx = max(0, (UP_DIV(-offset_x, cst.dilation_x)));
    int ex = min(cst.kernel_x, UP_DIV(cst.input_width - offset_x, cst.dilation_x));
    int sy = max(0, (UP_DIV(-offset_y, cst.dilation_y)));
    int ey = min(cst.kernel_y, UP_DIV(cst.input_height - offset_y, cst.dilation_y));
    offset_x += sx * cst.dilation_x;
    offset_y += sy * cst.dilation_y;
    
    float4 fresult = float4(biasTerms[int(gid.z)]);
    auto z_weights = weights + gid.z * cst.kernel_size;
    auto z_in = in + gid.z * cst.input_size;
    
    for (int ky = sy, y = offset_y; ky < ey; ky++, y += cst.dilation_y) {
        auto y_weights = z_weights + ky * cst.kernel_x;
        auto y_in = z_in + y * cst.input_width;
        for (int kx = sx, x = offset_x; kx < ex; kx++, x += cst.dilation_x) {
            int4 input = int4(y_in[x]) - cst.input_zero_point;
            fresult += float4(y_weights[kx]) * float4(input);
        }
    }
    
    int4 result = int4(fresult);
    result = saturate_round_x2_high_mul(result * (1 << cst.output_shift_before), cst.output_multiplier);
    result = round_divide_by_pot(result, -cst.output_shift_after);
    result = clamp(result + cst.output_zero_point, cst.output_activation_min, cst.output_activation_max);
    out[int(gid.z) * cst.output_size + int(gid.y) * cst.output_width + int(gid.x)] = uchar4(result);
}
