struct Param {
    float minV;
    float maxV;
    int size;
    int remain;
};
kernel void relu6(const device ftype4 *in [[buffer(0)]],
                     device ftype4 *out [[buffer(1)]],
                     constant Param &p [[buffer(2)]],
                     uint3 gid [[thread_position_in_grid]]) {
    if (gid.x < p.size) {
        out[int(gid.x)] = clamp(in[int(gid.x)], (ftype4)p.minV, (ftype4)p.maxV);
    }
}

kernel void relu(const device ftype4 *in [[buffer(0)]],
                    device ftype4 *out [[buffer(1)]],
                    constant Param &p [[buffer(2)]],
                    uint3 gid [[thread_position_in_grid]]) {
    if (gid.x < p.size) {
        auto value = in[int(gid.x)];
        out[int(gid.x)] = fmax(value, (ftype4)0) + fmin(value, (ftype4)0) * ftype4(p.minV);
    }
}