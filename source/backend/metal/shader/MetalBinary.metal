struct binary_op_shape {
    int i0stride;
    int i1stride;
    int output_data_count;
    int activationType;
};
#define define_op(op) \
kernel void binary_##op##_x1(const device ftype *in0       [[buffer(0)]],\
                          const device ftype *in1       [[buffer(1)]],\
                          device ftype *out             [[buffer(2)]],\
                          constant binary_op_shape& s   [[buffer(3)]],\
                          uint gid                      [[thread_position_in_grid]]) {\
    if ((int)gid >= s.output_data_count) return;\
    auto value0 = in0[s.i0stride * int(gid)];\
    auto value1 = in1[s.i1stride * int(gid)];\
    auto val = op(value0, value1);\
    if(s.activationType == 1) {\
        val = (val < (ftype)0 ? (ftype)0 : val);\
    }\
    out[int(gid)] = val;\
}

static inline ftype add(ftype value1, ftype value2) {
    return value1 + value2;
}

static inline ftype sub(ftype value1, ftype value2) {
    return value1 - value2;
}

static inline ftype mul(ftype value1, ftype value2) {
    return value1 * value2;
}

static inline ftype div(ftype value1, ftype value2) {
    return value2 == 0 ? 0 : value1 / value2;
}

static inline ftype squared_diff(ftype value1, ftype value2) {
    return (value1 - value2) * (value1 - value2);
}
static inline ftype mod(ftype value1, ftype value2) {
    return fmod(value1, value2);
}
static inline ftype floormod(ftype x, ftype y) {
    return x - floor(x / y) * y;
}


define_op(add)
define_op(sub)
define_op(mul)
define_op(div)
define_op(max)
define_op(min)
define_op(pow)
define_op(floormod)
define_op(mod)
define_op(squared_diff)
