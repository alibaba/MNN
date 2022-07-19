struct matmul_shape {
    int4 mat_size;
    int4 in_stride;
};

kernel void matmul(const device ftype *in0  [[buffer(0)]],
                   const device ftype *in1  [[buffer(1)]],
                   device ftype *out        [[buffer(2)]],
                   constant matmul_shape &s [[buffer(3)]],
                   uint2 gid[[thread_position_in_grid]]) {
    if ((int)gid.x < s.mat_size.x && (int)gid.y < s.mat_size.y) {
        auto off_in0 = in0 + int(gid.y) * s.in_stride.x;
        auto off_in1 = in1 + int(gid.x) * s.in_stride.z;
        FLOAT value = 0.f;
        for (int i = 0; i < s.mat_size.z; i++, off_in0 += s.in_stride.y, off_in1 += s.in_stride.w) {
            value += FLOAT(*off_in0) * FLOAT(*off_in1);
        }
        out[int(gid.y) * s.mat_size.x + int(gid.x)] = ftype(value);
    }
}

kernel void matmul_bias(const device ftype *in0  [[buffer(0)]],
                   const device ftype *in1  [[buffer(1)]],
                   const device ftype *biasValue  [[buffer(2)]],
                   device ftype *out        [[buffer(3)]],
                   constant matmul_shape &s [[buffer(4)]],
                   uint2 gid[[thread_position_in_grid]]) {
    if ((int)gid.x < s.mat_size.x && (int)gid.y < s.mat_size.y) {
        auto off_in0 = in0 + int(gid.y) * s.in_stride.x;
        auto off_in1 = in1 + int(gid.x) * s.in_stride.z;
        FLOAT value = 0.f;
        for (int i = 0; i < s.mat_size.z; i++, off_in0 += s.in_stride.y, off_in1 += s.in_stride.w) {
            value += FLOAT(*off_in0) * FLOAT(*off_in1);
        }
        out[int(gid.y) * s.mat_size.x + int(gid.x)] = ftype(value) + biasValue[(int)(gid.x)];
    }
}
