//
//  MetalResize.metal
//  MNN
//
//  Created by MNN on 2018/08/07.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <metal_stdlib>
#include "MetalDefine.metal"

using namespace metal;

struct resize_shape {
    int input_width;
    int input_height;
    int input_size;
    int output_width;
    int output_height;
    int output_size;
};

kernel void resize_bilinear(const device ftype4 *in     [[buffer(0)]],
                            device ftype4 *out          [[buffer(1)]],
                            constant resize_shape &c    [[buffer(2)]],
                            constant float2& s          [[buffer(3)]],
                            uint3 gid                   [[thread_position_in_grid]]) {
    if ((int)gid.x >= c.output_width || (int)gid.y >= c.output_height) return;
    
    float srcX = gid.x * s.x, srcY = gid.y * s.y;
    int left = floor(srcX), right = min(left + 1, c.input_width - 1);
    int top = floor(srcY), bottom = min(top + 1, c.input_height - 1);
    
    float x2_factor = srcX - left;
    float y2_factor = srcY - top;
    float x1_factor = 1 - x2_factor;
    float y1_factor = 1 - y2_factor;
    
    auto in_z        = in + gid.z * c.input_size;
    auto in_top      = in_z + top * c.input_width;
    auto in_bottom   = in_z + bottom * c.input_width;
    auto tl = float4(in_top[left])     * x1_factor * y1_factor;
    auto tr = float4(in_top[right])    * x2_factor * y1_factor;
    auto bl = float4(in_bottom[left])  * x1_factor * y2_factor;
    auto br = float4(in_bottom[right]) * x2_factor * y2_factor;
    out[int(gid.z) * c.output_size + int(gid.y) * c.output_width + int(gid.x)] = ftype4(tl + tr + bl + br);
}

static inline float4 resize_cubic_interpolation(float4 A, float4 B, float4 C, float4 D, float factor) {
    float4 a = (B - C) + 0.5f * (B - A) + (D - C) * 0.5f;
    float4 b = C - ((B - A) + (B - C)) - (B + D) * 0.5f;
    float4 c = (C - A) * 0.5f;
    float4 d = B;
    return ((a * factor + b) * factor + c) * factor + d;
}

kernel void resize_cubic(const device ftype4 *in        [[buffer(0)]],
                         device ftype4 *out             [[buffer(1)]],
                         constant resize_shape &c       [[buffer(2)]],
                         uint3 gid                      [[thread_position_in_grid]]) {
    if ((int)gid.x >= c.output_width || (int)gid.y >= c.output_height) return;
    
    float u = float(gid.x) / float(c.output_width - 1);
    float v = float(gid.y) / float(c.output_height - 1);
    float x = u * c.input_width - 0.5f;
    float y = v * c.input_height - 0.5f;
    float x_factor = x - floor(x);
    float y_factor = y - floor(y);
    
    int4 xp = int4(int(x) - 1, int(x) + 0, int(x) + 1, int(x) + 2);
    xp = clamp(xp, 0, c.input_width - 1);
    
    int4 yp = int4(int(y) - 1, int(y) + 0, int(y) + 1, int(y) + 2);
    yp = clamp(yp, 0, c.input_height - 1);
    
    auto in_z = in + gid.z * c.input_size;
    float4x4 ABCD;
    for (int i = 0; i < 4; i++) {
        auto in_y = in_z + yp[i] * c.input_width;
        float4 A = float4(in_y[xp[0]]);
        float4 B = float4(in_y[xp[1]]);
        float4 C = float4(in_y[xp[2]]);
        float4 D = float4(in_y[xp[3]]);
        ABCD[i] = resize_cubic_interpolation(A, B, C, D, x_factor);
    }
    
    auto val = ftype4(resize_cubic_interpolation(ABCD[0], ABCD[1], ABCD[2], ABCD[3], y_factor));
    out[int(gid.z) * c.output_size + int(gid.y) * c.output_width + int(gid.x)] = val;
}
