//
//  MetalQuantizedAvgPool.metal
//  MNN
//
//  Created by MNN on 2018/11/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <metal_stdlib>

using namespace metal;

struct quantized_avg_pool_shape {
    int batch;
    int input_height;
    int input_width;
    int output_height;
    int output_width;
    int channel;
    
    int kernel_width;
    int kernel_height;
    int stride_width;
    int stride_height;
    int pad_width;
    int pad_height;
    int activation_min;
    int activation_max;
};

kernel void quantized_avg_pool(const device uchar *in               [[buffer(0)]],
                               device uchar *out                    [[buffer(1)]],
                               constant quantized_avg_pool_shape& s [[buffer(2)]],
                               uint3 gid                            [[thread_position_in_grid]]) {
    int n = gid.z / s.output_height;
    int h = gid.z % s.output_height, w = gid.y, c = gid.x;
    if (n >= s.batch || h >= s.output_height || w >= s.output_width || c >= s.channel) return;
    
    int off_x = w * s.stride_width - s.pad_width;
    int off_y = h * s.stride_height - s.pad_height;
    int sx = max(0, -off_x);
    int sy = max(0, -off_y);
    int ex = min(s.kernel_width, s.input_width - off_x);
    int ey = min(s.kernel_height, s.input_height - off_y);
    off_x += sx;
    off_y += sy;
    
    int result = 0;
    int count = 0;
    auto c_in = in + n * s.input_height * s.input_width * s.channel + c;
    for (int ky = sy, y = off_y; ky < ey; ky++, y++) {
        for (int kx = sx, x = off_x; kx < ex; kx++, x++) {
            result += c_in[y * s.input_width * s.channel + x * s.channel];
            count += 1;
        }
    }
    if (count > 0) result = round((float)result / count);
    out[int(gid.z) * s.output_width * s.channel + w * s.channel + c] = uchar(clamp(result, s.activation_min, s.activation_max));
}
