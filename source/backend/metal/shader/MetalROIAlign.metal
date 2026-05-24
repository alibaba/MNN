struct roi_align_shape {
    int input_width;
    int input_height;
    int input_size;       // input_width * input_height
    int input_batch;
    int output_width;     // pooled_width
    int output_height;    // pooled_height
    int output_size;      // output_width * output_height
    int num_roi;
    float spatial_scale;
    int sampling_ratio;
    int aligned;
    int has_batch_indices;
};

static inline ftype4 bilinear_interpolate(const device ftype4 *data,
                                           int height, int width,
                                           float y, float x) {
    // Handle boundary
    if (y < -1.0f || y > (float)height || x < -1.0f || x > (float)width) {
        return ftype4(0);
    }
    y = max(y, 0.0f);
    x = max(x, 0.0f);

    int y0 = (int)y;
    int x0 = (int)x;
    int y1, x1;

    if (y0 >= height - 1) {
        y1 = y0 = height - 1;
        y = (float)y0;
    } else {
        y1 = y0 + 1;
    }

    if (x0 >= width - 1) {
        x1 = x0 = width - 1;
        x = (float)x0;
    } else {
        x1 = x0 + 1;
    }

    float dy = y - (float)y0;
    float dx = x - (float)x0;
    float dy1 = 1.0f - dy;
    float dx1 = 1.0f - dx;

    // w1=dy1*dx1 (top-left), w2=dy1*dx (top-right), w3=dy*dx1 (bottom-left), w4=dy*dx (bottom-right)
    ftype4 v1 = data[y0 * width + x0];
    ftype4 v2 = data[y0 * width + x1];
    ftype4 v3 = data[y1 * width + x0];
    ftype4 v4 = data[y1 * width + x1];

    ftype4 result = ftype4((float4)v1 * dy1 * dx1 +
                           (float4)v2 * dy1 * dx +
                           (float4)v3 * dy * dx1 +
                           (float4)v4 * dy * dx);
    return result;
}

kernel void roi_align_avg(const device ftype4 *in       [[buffer(0)]],
                          const device ftype *roi        [[buffer(1)]],
                          device ftype4 *out             [[buffer(2)]],
                          constant roi_align_shape &s    [[buffer(3)]],
                          const device int *batch_idx_buf [[buffer(4)]],
                          uint3 gid                      [[thread_position_in_grid]]) {
    if ((int)gid.x >= s.output_width || (int)gid.y >= s.output_height) return;

    int ob = gid.z % s.num_roi;   // ROI index
    int iz = gid.z / s.num_roi;   // slice index (C/4)

    // Read ROI data
    int ib;
    float x1, y1, x2, y2;
    if (s.has_batch_indices) {
        auto b_roi = roi + ob * 4;
        ib = batch_idx_buf[ob];
        x1 = (float)b_roi[0];
        y1 = (float)b_roi[1];
        x2 = (float)b_roi[2];
        y2 = (float)b_roi[3];
    } else {
        auto b_roi = roi + ob * 5;
        ib = (int)b_roi[0];
        x1 = (float)b_roi[1];
        y1 = (float)b_roi[2];
        x2 = (float)b_roi[3];
        y2 = (float)b_roi[4];
    }

    float offset = s.aligned ? -0.5f : 0.0f;
    x1 = x1 * s.spatial_scale + offset;
    y1 = y1 * s.spatial_scale + offset;
    x2 = x2 * s.spatial_scale + offset;
    y2 = y2 * s.spatial_scale + offset;

    float roi_w = x2 - x1;
    float roi_h = y2 - y1;
    if (!s.aligned) {
        roi_w = max(roi_w, 1.0f);
        roi_h = max(roi_h, 1.0f);
    }

    float bin_size_w = roi_w / (float)s.output_width;
    float bin_size_h = roi_h / (float)s.output_height;

    int sampling_ratio_w = s.sampling_ratio > 0 ? s.sampling_ratio : (int)ceil(roi_w / (float)s.output_width);
    int sampling_ratio_h = s.sampling_ratio > 0 ? s.sampling_ratio : (int)ceil(roi_h / (float)s.output_height);
    sampling_ratio_w = max(sampling_ratio_w, 1);
    sampling_ratio_h = max(sampling_ratio_h, 1);

    float sampling_bin_w = bin_size_w / (float)sampling_ratio_w;
    float sampling_bin_h = bin_size_h / (float)sampling_ratio_h;

    // Pointer to the correct slice of input: [iz * batch + ib] * (H * W)
    auto z_in = in + (ib + iz * s.input_batch) * s.input_size;

    int pw = (int)gid.x;
    int ph = (int)gid.y;

    float sampling_start_w = x1 + pw * bin_size_w;
    float sampling_start_h = y1 + ph * bin_size_h;

    float4 sum = float4(0.0f);
    int count = sampling_ratio_h * sampling_ratio_w;

    for (int i = 0; i < sampling_ratio_h; ++i) {
        float py = sampling_start_h + (0.5f + (float)i) * sampling_bin_h;
        for (int j = 0; j < sampling_ratio_w; ++j) {
            float px = sampling_start_w + (0.5f + (float)j) * sampling_bin_w;
            ftype4 val = bilinear_interpolate(z_in, s.input_height, s.input_width, py, px);
            sum += float4(val);
        }
    }

    sum /= (float)count;
    int out_idx = (int)gid.z * s.output_size + ph * s.output_width + pw;
    out[out_idx] = ftype4(sum);
}

kernel void roi_align_max(const device ftype4 *in       [[buffer(0)]],
                          const device ftype *roi        [[buffer(1)]],
                          device ftype4 *out             [[buffer(2)]],
                          constant roi_align_shape &s    [[buffer(3)]],
                          const device int *batch_idx_buf [[buffer(4)]],
                          uint3 gid                      [[thread_position_in_grid]]) {
    if ((int)gid.x >= s.output_width || (int)gid.y >= s.output_height) return;

    int ob = gid.z % s.num_roi;
    int iz = gid.z / s.num_roi;

    int ib;
    float x1, y1, x2, y2;
    if (s.has_batch_indices) {
        auto b_roi = roi + ob * 4;
        ib = batch_idx_buf[ob];
        x1 = (float)b_roi[0];
        y1 = (float)b_roi[1];
        x2 = (float)b_roi[2];
        y2 = (float)b_roi[3];
    } else {
        auto b_roi = roi + ob * 5;
        ib = (int)b_roi[0];
        x1 = (float)b_roi[1];
        y1 = (float)b_roi[2];
        x2 = (float)b_roi[3];
        y2 = (float)b_roi[4];
    }

    float offset_val = s.aligned ? -0.5f : 0.0f;
    x1 = x1 * s.spatial_scale + offset_val;
    y1 = y1 * s.spatial_scale + offset_val;
    x2 = x2 * s.spatial_scale + offset_val;
    y2 = y2 * s.spatial_scale + offset_val;

    float roi_w = x2 - x1;
    float roi_h = y2 - y1;
    if (!s.aligned) {
        roi_w = max(roi_w, 1.0f);
        roi_h = max(roi_h, 1.0f);
    }

    float bin_size_w = roi_w / (float)s.output_width;
    float bin_size_h = roi_h / (float)s.output_height;

    int sampling_ratio_w = s.sampling_ratio > 0 ? s.sampling_ratio : (int)ceil(roi_w / (float)s.output_width);
    int sampling_ratio_h = s.sampling_ratio > 0 ? s.sampling_ratio : (int)ceil(roi_h / (float)s.output_height);
    sampling_ratio_w = max(sampling_ratio_w, 1);
    sampling_ratio_h = max(sampling_ratio_h, 1);

    float sampling_bin_w = bin_size_w / (float)sampling_ratio_w;
    float sampling_bin_h = bin_size_h / (float)sampling_ratio_h;

    auto z_in = in + (ib + iz * s.input_batch) * s.input_size;

    int pw = (int)gid.x;
    int ph = (int)gid.y;

    float sampling_start_w = x1 + pw * bin_size_w;
    float sampling_start_h = y1 + ph * bin_size_h;

    float4 max_val = float4(-FLT_MAX);
    for (int i = 0; i < sampling_ratio_h; ++i) {
        float py = sampling_start_h + (0.5f + (float)i) * sampling_bin_h;
        for (int j = 0; j < sampling_ratio_w; ++j) {
            float px = sampling_start_w + (0.5f + (float)j) * sampling_bin_w;
            ftype4 val = bilinear_interpolate(z_in, s.input_height, s.input_width, py, px);
            max_val = max(max_val, float4(val));
        }
    }

    int out_idx = (int)gid.z * s.output_size + ph * s.output_width + pw;
    out[out_idx] = ftype4(max_val);
}