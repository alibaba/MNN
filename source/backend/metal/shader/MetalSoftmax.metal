struct softmax_shape {
    int inside_size;
    int axis_length;
    int outside_size;
    int flat_length;
};

static inline float softmax_max4(float4 value) {
    return max(max(value[0], value[1]), max(value[2], value[3]));
}

static inline float softmax_sum4(float4 value) {
    return value[0] + value[1] + value[2] + value[3];
}

static inline float4 softmax_filter(float4 value, int z, int limit) {
    return select(0, value, z * 4 + int4(0, 1, 2, 3) < limit);
}

kernel void softmax_plane(const device ftype *in     [[buffer(0)]],
                       device ftype *out          [[buffer(1)]],
                       constant softmax_shape& s   [[buffer(2)]],
                       uint2 gid                   [[thread_position_in_grid]]) {
    if ((int)gid.x >= s.inside_size || (int)gid.y >= s.outside_size) return;
    
    auto axis_off = gid.y * s.axis_length * s.inside_size + gid.x;
    auto axis_in  = in + axis_off;
    auto axis_out = out + axis_off;
    
    // get max
    auto max1 = axis_in[0];
    for (int i = 1; i < s.axis_length; i++) {
        max1 = max(max1, axis_in[i * s.inside_size]);
    }
    
    // get sum
    float sum1 = 0;
    for (int i = 0; i < s.axis_length; i++) {
        sum1 += float(exp(axis_in[i * s.inside_size] - max1));
    }
    
    // output
    for (int i = 0; i < s.axis_length; i++) {
        axis_out[i * s.inside_size] = ftype(exp(float(axis_in[i * s.inside_size] - max1)) / sum1);
    }
}

kernel void softmax_on_reorder(const device ftype4 *in      [[buffer(0)]],
                               device ftype4 *out           [[buffer(1)]],
                               constant softmax_shape& s    [[buffer(2)]],
                               uint2 gid                    [[thread_position_in_grid]]) {
    if ((int)gid.x >= s.inside_size || (int)gid.y >= s.outside_size) return;
    
    auto axis_off = gid.y * s.axis_length * s.inside_size + gid.x;
    auto axis_in  = in + axis_off;
    auto axis_out = out + axis_off;

    // get max
    auto max4 = softmax_filter(float4(axis_in[0]), 0, s.flat_length);
    for (int i = 1; i < s.axis_length; i++) {
        max4 = max(max4, softmax_filter(float4(axis_in[i * s.inside_size]), i, s.flat_length));
    }
    float max1 = softmax_max4(max4);
    
    // get sum
    float4 sum4 = 0;
    for (int i = 0; i < s.axis_length; i++) {
        sum4 += softmax_filter(exp(float4(axis_in[i * s.inside_size] - max1)), i, s.flat_length);
    }
    float sum1 = softmax_sum4(sum4);
    
    // output
    for (int i = 0; i < s.axis_length; i++) {
        axis_out[i * s.inside_size] = ftype4(exp(float4(axis_in[i * s.inside_size]) - max1) / sum1);
    }
}

kernel void softmax_off_reorder(const device ftype4 *in     [[buffer(0)]],
                                device ftype4 *out          [[buffer(1)]],
                                constant softmax_shape& s   [[buffer(2)]],
                                uint2 gid                   [[thread_position_in_grid]]) {
    if ((int)gid.x >= s.inside_size || (int)gid.y >= s.outside_size) return;

    auto axis_off = gid.y * s.axis_length * s.inside_size + gid.x;
    auto axis_in  = in + axis_off;
    auto axis_out = out + axis_off;

    // get max
    auto max4 = axis_in[0];
    for (int i = 1; i < s.axis_length; i++) {
        max4 = max(max4, axis_in[i * s.inside_size]);
    }

    // get sum
    float4 sum4 = 0;
    for (int i = 0; i < s.axis_length; i++) {
        sum4 += exp(float4(axis_in[i * s.inside_size] - max4));
    }

    // output
    for (int i = 0; i < s.axis_length; i++) {
        axis_out[i * s.inside_size] = ftype4(exp(float4(axis_in[i * s.inside_size] - max4)) / sum4);
    }
}
