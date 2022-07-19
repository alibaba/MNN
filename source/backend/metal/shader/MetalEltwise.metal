kernel void eltwise_prod(device const ftype *in0   [[buffer(0)]],
                         device const ftype *in1   [[buffer(1)]],
                         device ftype *out         [[buffer(2)]],
                         constant int4& shape        [[buffer(3)]],
                         uint gid                   [[thread_position_in_grid]]) {
    if ((int)gid < shape.x) {
        out[(int)gid] = in0[(int)gid] * in1[(int)gid];
    }
}

kernel void eltwise_max(device const ftype *in0    [[buffer(0)]],
                        device const ftype *in1    [[buffer(1)]],
                        device ftype *out         [[buffer(2)]],
                        constant int4& shape         [[buffer(3)]],
                        uint gid                    [[thread_position_in_grid]]) {
    if ((int)gid < shape.x) {
        out[(int)gid] = max(in0[(int)gid], in1[(int)gid]);
    }
}

kernel void eltwise_add(device const ftype *in0    [[buffer(0)]],
                        device const ftype *in1    [[buffer(1)]],
                        device ftype *out          [[buffer(2)]],
                        constant int4& shape         [[buffer(3)]],
                        uint gid                    [[thread_position_in_grid]]) {
    if ((int)gid < shape.x) {
        out[(int)gid] = in0[(int)gid] + in1[(int)gid];
    }
}
