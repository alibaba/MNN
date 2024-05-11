struct scale_shape {
    int size;
    int steps;
    int batch;
};

kernel void scale_ca(const device ftype4 *in        [[buffer(0)]],
                     device ftype4 *out             [[buffer(1)]],
                     constant scale_shape &s        [[buffer(2)]],
                     const device float4 *scales    [[buffer(3)]],
                     const device float4 *biasTerms [[buffer(4)]],
                     uint2 gid                      [[thread_position_in_grid]]) {
    if ((int)gid.x >= s.size || (int)gid.y >= s.steps * s.batch) return;

    int z = gid.y / s.batch;
    out[int(gid.y) * s.size + int(gid.x)] =
    in [int(gid.y) * s.size + int(gid.x)] * ftype4(scales[z]) + ftype4(biasTerms[z]);
}
