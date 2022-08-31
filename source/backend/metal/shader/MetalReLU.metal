kernel void relu_x1(const device ftype *in  [[buffer(0)]],
                    device ftype *out       [[buffer(1)]],
                    constant float &slope   [[buffer(2)]],
                    uint gid                [[thread_position_in_grid]]) {
    auto value = in[int(gid)];
    out[int(gid)] = fmax(value, (ftype)0) + fmin(value, (ftype)0) * ftype(slope);
}

kernel void relu_x4(const device ftype4 *in [[buffer(0)]],
                    device ftype4 *out      [[buffer(1)]],
                    constant float &slope   [[buffer(2)]],
                    uint gid                [[thread_position_in_grid]]) {
    auto value = in[int(gid)];
    out[int(gid)] = fmax(value, (ftype4)0) + fmin(value, (ftype4)0) * ftype4(slope);
}
