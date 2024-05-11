struct prelu_shape {
    int size;
    int slice;
    int batch;
};

kernel void prelu(const device ftype4 *in   [[buffer(0)]],
                  device ftype4 *out        [[buffer(1)]],
                  constant float &slope     [[buffer(2)]],
                  uint gid                  [[thread_position_in_grid]]) {
    auto v4 = in[int(gid)];
    out[int(gid)] = select(v4, ftype4(slope) * v4, signbit(v4));
}

kernel void prelu_slopes(const device ftype4 *in    [[buffer(0)]],
                         device ftype4 *out         [[buffer(1)]],
                         const device float4 *slope [[buffer(2)]],
                         constant prelu_shape& s    [[buffer(3)]],
                         uint3 gid                  [[thread_position_in_grid]]) { // size, slice, batch
    if ((int)gid.x >= s.size || (int)gid.y >= s.slice) return;
    
    int z = gid.z + gid.y * s.batch;
    auto v4 = in[z * s.size + int(gid.x)];
    out[z * s.size + int(gid.x)] = select(v4, ftype4(slope[int(gid.y)]) * v4, signbit(v4));
}
