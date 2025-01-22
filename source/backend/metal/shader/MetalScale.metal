struct scale_shape {
    int size;
    int steps;
    int batch;
    int offset;
};

kernel void scale_ca(const device ftype4 *in        [[buffer(0)]],
                     device ftype4 *out             [[buffer(1)]],
                     constant scale_shape &s        [[buffer(2)]],
                     const device float4 *scalesbias[[buffer(3)]],
                     uint2 gid                      [[thread_position_in_grid]]) {
    if ((int)gid.x >= s.size || (int)gid.y >= s.steps * s.batch) return;

    int z = gid.y / s.batch;
    int offset = s.offset;
    float4 scale = scalesbias[z];
    float4 bias = scalesbias[z+offset];
    out[int(gid.y) * s.size + int(gid.x)] =
    (ftype4)((float4)in[int(gid.y) * s.size + int(gid.x)] * scale + bias);
}
