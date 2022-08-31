using namespace metal;
kernel void cast_float_to_int32(const device ftype *in  [[buffer(0)]],
                                device int *out         [[buffer(1)]],
                                uint gid                [[thread_position_in_grid]]) {
    out[int(gid)] = int(in[int(gid)]);
}

kernel void cast_int32_to_float(const device int *in    [[buffer(0)]],
                                device ftype *out       [[buffer(1)]],
                                uint gid                [[thread_position_in_grid]]) {
    out[int(gid)] = ftype(in[int(gid)]);
}

kernel void cast_uint8_to_float(const device uchar *in  [[buffer(0)]],
                                device ftype *out       [[buffer(1)]],
                                uint gid                [[thread_position_in_grid]]) {
    out[int(gid)] = ftype(in[int(gid)]);
}

kernel void cast_uint8_to_int(const device uchar *in  [[buffer(0)]],
                                device int *out       [[buffer(1)]],
                                uint gid                [[thread_position_in_grid]]) {
    out[int(gid)] = ftype(in[int(gid)]);
}

kernel void cast_float_to_uint8(const device ftype *in  [[buffer(0)]],
                                device uchar *out       [[buffer(1)]],
                                uint gid                [[thread_position_in_grid]]) {
    out[int(gid)] = uchar(in[int(gid)]);
}
