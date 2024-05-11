struct ROI_shape {
    int input_width;
    int input_height;
    int input_size;
    int input_batch;
    int output_width;
    int output_height;
    int output_size;
    int batch;
    float spatial_scale;
};
kernel void ROI_pooling(const device ftype4 *in [[buffer(0)]],
                        const device ftype *roi [[buffer(1)]],
                        device ftype4 *out      [[buffer(2)]],
                        constant ROI_shape &s   [[buffer(3)]],
                        uint3 gid               [[thread_position_in_grid]]) {
    if ((int)gid.x >= s.output_width || (int)gid.y >= s.output_height) return;
    
    int ob = gid.z % s.batch;
    int iz = gid.z / s.batch;
    
    auto b_roi = roi + ob * 5;
    int ib = int(b_roi[0]);
    int x1 = round(float(b_roi[1]) * s.spatial_scale);
    int y1 = round(float(b_roi[2]) * s.spatial_scale);
    int x2 = round(float(b_roi[3]) * s.spatial_scale);
    int y2 = round(float(b_roi[4]) * s.spatial_scale);
    
    int roi_w = max(x2 - x1 + 1, 1);
    int roi_h = max(y2 - y1 + 1, 1);
    float bin_size_w = (float)roi_w / (float)s.output_width;
    float bin_size_h = (float)roi_h / (float)s.output_height;
    
    int w_start = clamp(x1 + (int)floor(gid.x * bin_size_w)     , 0, s.input_width);
    int w_end   = clamp(x1 + (int)ceil((gid.x + 1) * bin_size_w), 0, s.input_width);
    int h_start = clamp(y1 + (int)floor(gid.y * bin_size_h)     , 0, s.input_height);
    int h_end   = clamp(y1 + (int)ceil((gid.y + 1) * bin_size_h), 0, s.input_height);
    
    int is_empty = (h_end <= h_start) || (w_end <= w_start);
    auto z_in = in + (ib + iz * s.input_batch) * s.input_size;
    auto max4 = is_empty ? 0 : z_in[h_start * s.input_width + w_start];
    for (int y = h_start; y < h_end; y++) {
        auto y_in = z_in + y * s.input_width;
        for (int x = w_start; x < w_end; x++) {
            max4 = max(max4, y_in[x]);
        }
    }
    out[int(gid.z) * s.output_size + int(gid.y) * s.output_width + int(gid.x)] = max4;
}
