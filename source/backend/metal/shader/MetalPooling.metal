struct pooling_sizes {
    int input_width;
    int input_height;
    int output_width;
    int output_height;
    int slice;
    int kernel_width;
    int kernel_height;
    int stride_width;
    int stride_height;
    int pad_width;
    int pad_height;
};

kernel void pooling_max(const device ftype4 *in     [[buffer(0)]],
                        device ftype4 *out          [[buffer(1)]],
                        constant pooling_sizes& s   [[buffer(2)]],
                        uint3 gid                   [[thread_position_in_grid]]) {
    if (any(gid >= uint3(s.output_width, s.output_height, s.slice))) return;
    
    int off_x = gid.x * s.stride_width - s.pad_width;
    int off_y = gid.y * s.stride_height - s.pad_height;
    int x_max = s.input_width  - 1;
    int y_max = s.input_height - 1;
    int ex = off_x + s.kernel_width;
    int ey = off_y + s.kernel_height;
    
    auto z_in = in + (int)gid.z * s.input_width * s.input_height;
    auto result = ftype4(z_in[clamp(off_y, 0, y_max) * s.input_width + clamp(off_x, 0, x_max)]);
    for (int y = off_y; y < ey; y++) {
        auto y_in = z_in + clamp(y, 0, y_max) * s.input_width;
        for (int x = off_x; x < ex; x++) {
            result = max(result, y_in[clamp(x, 0, x_max)]);
        }
    }
    out[(int)gid.z * s.output_width * s.output_height + (int)gid.y * s.output_width + (int)gid.x] = result;
}

kernel void pooling_avg(const device ftype4 *in     [[buffer(0)]],
                        device ftype4 *out          [[buffer(1)]],
                        constant pooling_sizes& s   [[buffer(2)]],
                        uint3 gid                   [[thread_position_in_grid]]) {
    if (any(gid >= uint3(s.output_width, s.output_height, s.slice))) return;
    
    int off_x = gid.x * s.stride_width - s.pad_width;
    int off_y = gid.y * s.stride_height - s.pad_height;
    int sx = off_x + max(0, -off_x);
    int sy = off_y + max(0, -off_y);
    int ex = off_x + min(s.kernel_width, s.input_width - off_x);
    int ey = off_y + min(s.kernel_height, s.input_height - off_y);
    
    FLOAT4 result = 0;
    auto z_in = in + (int)gid.z * s.input_width * s.input_height;
    for (int y = sy; y < ey; y++) {
        for (int x = sx; x < ex; x++) {
            result += FLOAT4(z_in[y * s.input_width + x]);
        }
    }
    int count = (ey - sy) * (ex - sx);
    FLOAT4 div = count > 0 ? 1.f / count : 1;
    out[(int)gid.z * s.output_width * s.output_height + (int)gid.y * s.output_width + (int)gid.x] = ftype4(result * div);
}
