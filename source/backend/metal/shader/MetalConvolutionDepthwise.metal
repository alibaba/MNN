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
                           uint3 gid                      [[thread_position_in_grid]]) {
    if ((int)gid.x >= cst.output_width || (int)gid.y >= cst.output_height || (int)gid.z >= cst.slice * cst.batch) return;
    
    int oz = gid.z / cst.batch;
    int offset_x = (int)gid.x * cst.stride_x - cst.pad_x;
    int offset_y = (int)gid.y * cst.stride_y - cst.pad_y;
    int sx = max(0, (UP_DIV(-offset_x, cst.dilation_x)));
    int ex = min(cst.kernel_x, UP_DIV(cst.input_width - offset_x, cst.dilation_x));
    int sy = max(0, (UP_DIV(-offset_y, cst.dilation_y)));
    int ey = min(cst.kernel_y, UP_DIV(cst.input_height - offset_y, cst.dilation_y));
    offset_x += sx * cst.dilation_x;
    offset_y += sy * cst.dilation_y;

    auto z_wt  = wt  + (int)oz * cst.kernel_size;
    auto z_in  = in  + (int)gid.z * cst.input_size;
    auto z_out = out + (int)gid.z * cst.output_size + (int)gid.y * cst.output_width + (int)gid.x;
    FLOAT4 result = FLOAT4(biasTerms[oz]);
    for (auto ky = sy, y = offset_y; ky < ey; ky++, y += cst.dilation_y) {
        for (auto kx = sx, x = offset_x; kx < ex; kx++, x += cst.dilation_x) {
            auto wt4 = z_wt[ky * cst.kernel_x   + kx];
            auto in4 = z_in[ y * cst.input_width + x];
            result += FLOAT4(in4 * wt4);
        }
    }

    *z_out = activate((ftype4)result, cst.activation);
}
