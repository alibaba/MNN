
struct deconv_constants {
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
    
    int delta_ky;
    int delta_kx;
    int delta_iy;
    int delta_ix;
    int has_bias;
    int batch;
    conv_activation_type activation;
};

kernel void deconv(const device ftype4 *in          [[buffer(0)]],
                   device ftype4 *out               [[buffer(1)]],
                   constant deconv_constants& cst   [[buffer(2)]],
                   const device ftype4x4 *wt        [[buffer(3)]],
                   const device ftype4 *biasTerms   [[buffer(4)]],
                   uint3 gid                        [[thread_position_in_grid]]) {
    if ((int)gid.x >= cst.output_width || (int)gid.y >= cst.output_height || (int)gid.z >= cst.batch * cst.output_slice) return;
    
    int b = gid.z % cst.batch;
    int o = gid.z / cst.batch;
    FLOAT4 result = FLOAT4(biasTerms[o]);

    int oy = (int)gid.y + cst.pad_y;
    int ox = (int)gid.x + cst.pad_x;
    int max_sy = min((cst.input_height - 1) * cst.stride_y, oy / cst.stride_y * cst.stride_y);
    int max_sx = min((cst.input_width - 1) * cst.stride_x, ox / cst.stride_x * cst.stride_x);
    int min_ky = UP_DIV(oy - max_sy, cst.dilation_y);
    int min_kx = UP_DIV(ox - max_sx, cst.dilation_x);
    
    if ((oy - min_ky * cst.dilation_y) % cst.stride_y == 0 && (ox - min_kx * cst.dilation_x) % cst.stride_x == 0) {
        int min_sy = max(0, ROUND_UP(oy + cst.dilation_y - cst.kernel_y * cst.dilation_y, cst.stride_y));
        int min_sx = max(0, ROUND_UP(ox + cst.dilation_x - cst.kernel_x * cst.dilation_x, cst.stride_x));
        int max_ky = (oy - min_sy) / cst.dilation_y;
        int max_kx = (ox - min_sx) / cst.dilation_x;
        int min_iy = (oy - max_ky * cst.dilation_y) / cst.stride_y;
        int min_ix = (ox - max_kx * cst.dilation_x) / cst.stride_x;
        
        auto o_wt = wt + o * cst.input_slice * cst.kernel_size;
        auto b_in = in + b * cst.input_size;
        for (auto z = 0; z < cst.input_slice; z++) {
            for (auto ky = max_ky, iy = min_iy; ky >= min_ky; ky -= cst.delta_ky, iy += cst.delta_iy) {
                for (auto kx = max_kx, ix = min_ix; kx >= min_kx; kx -= cst.delta_kx, ix += cst.delta_ix) {
                    auto wt4 = o_wt[z * cst.kernel_size + ky * cst.kernel_x + kx];
                    auto in4 = b_in[z * cst.input_size * cst.batch + iy * cst.input_width + ix];
                    result += FLOAT4(in4 * wt4);
                }
            }
        }
    }
    out[(int)gid.z * cst.output_size + (int)gid.y * cst.output_width + (int)gid.x] =     activate(ftype4(result), cst.activation);
}

kernel void deconv_depthwise(const device ftype4 *in        [[buffer(0)]],
                             device ftype4 *out             [[buffer(1)]],
                             constant deconv_constants& cst [[buffer(2)]],
                             const device ftype4 *wt        [[buffer(3)]],
                             const device ftype4 *biasTerms [[buffer(4)]],
                             uint3 gid                    [[thread_position_in_grid]]) {
    if ((int)gid.x >= cst.output_width || (int)gid.y >= cst.output_height || (int)gid.z >= cst.batch * cst.output_slice) return;
    
    FLOAT4 result = FLOAT4(biasTerms[(int)(gid.z / cst.batch)]);
    
    int oy = (int)gid.y + cst.pad_y;
    int ox = (int)gid.x + cst.pad_x;
    int max_sy = min((cst.input_height - 1) * cst.stride_y, oy / cst.stride_y * cst.stride_y);
    int max_sx = min((cst.input_width - 1) * cst.stride_x, ox / cst.stride_x * cst.stride_x);
    int min_ky = UP_DIV(oy - max_sy, cst.dilation_y);
    int min_kx = UP_DIV(ox - max_sx, cst.dilation_x);
    
    if ((oy - min_ky * cst.dilation_y) % cst.stride_y == 0 && (ox - min_kx * cst.dilation_x) % cst.stride_x == 0) {
        int min_sy = max(0, ROUND_UP(oy + cst.dilation_y - cst.kernel_y * cst.dilation_y, cst.stride_y));
        int min_sx = max(0, ROUND_UP(ox + cst.dilation_x - cst.kernel_x * cst.dilation_x, cst.stride_x));
        int max_ky = (oy - min_sy) / cst.dilation_y;
        int max_kx = (ox - min_sx) / cst.dilation_x;
        int min_iy = (oy - max_ky * cst.dilation_y) / cst.stride_y;
        int min_ix = (ox - max_kx * cst.dilation_x) / cst.stride_x;
        
        auto z_wt = wt + (int)gid.z * cst.kernel_size;
        auto z_in = in + (int)gid.z * cst.input_size;
        for (auto ky = max_ky, iy = min_iy; ky >= min_ky; ky -= cst.delta_ky, iy += cst.delta_iy) {
            for (auto kx = max_kx, ix = min_ix; kx >= min_kx; kx -= cst.delta_kx, ix += cst.delta_ix) {
                auto wt4 = z_wt[ky * cst.kernel_x + kx];
                auto in4 = z_in[iy * cst.input_width + ix];
                result += FLOAT4(in4 * wt4);
            }
        }
    }
    out[(int)gid.z * cst.output_size + (int)gid.y * cst.output_width + (int)gid.x] =     activate(ftype4(result), cst.activation);
}
