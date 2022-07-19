struct resize_shape {
    int input_width;
    int input_height;
    int input_size;
    int output_width;
    int output_height;
    int output_size;
    int sliceMap;
};
kernel void resize_nearest(const device ftype4 *in     [[buffer(0)]],
                            device ftype4 *out          [[buffer(1)]],
                            constant resize_shape &c    [[buffer(2)]],
                            constant float4& s          [[buffer(3)]],
                            uint3 gid                   [[thread_position_in_grid]]) {
    if ((int)gid.x >= c.output_width || (int)gid.y >= c.output_height || (int)gid.z >= c.sliceMap) return;
    
    float srcX = gid.x * s.x + s.y, srcY = gid.y * s.z + s.w;
    int left = floor(srcX);
    int top = floor(srcY);
    
    auto in_z        = in + gid.z * c.input_size;
    auto in_top      = in_z + top * c.input_width;
    out[int(gid.z) * c.output_size + int(gid.y) * c.output_width + int(gid.x)] = in_top[left];
}

kernel void resize_bilinear(const device ftype4 *in     [[buffer(0)]],
                            device ftype4 *out          [[buffer(1)]],
                            constant resize_shape &c    [[buffer(2)]],
                            constant float4& s          [[buffer(3)]],
                            uint3 gid                   [[thread_position_in_grid]]) {
    if ((int)gid.x >= c.output_width || (int)gid.y >= c.output_height || (int)gid.z >= c.sliceMap) return;
    
    float srcX = gid.x * s.x + s.y, srcY = gid.y * s.z + s.w;
    int srcXInt = int(floor(srcX));
    int srcYInt = int(floor(srcY));
    int left = clamp(srcXInt, 0, c.input_width - 1);
    int right = clamp(srcXInt+1, 0, c.input_width - 1);
    int top = clamp(srcYInt, 0, c.input_height - 1);
    int bottom = clamp(srcYInt+1, 0, c.input_height - 1);

    float x2_factor = srcX - float(srcXInt);
    float y2_factor = srcY - float(srcYInt);
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
                         constant float4& s          [[buffer(3)]],
                         uint3 gid                      [[thread_position_in_grid]]) {
    if ((int)gid.x >= c.output_width || (int)gid.y >= c.output_height || (int)gid.z >= c.sliceMap) return;
    float x = gid.x * s.x + s.y, y = gid.y * s.z + s.w;
    
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
