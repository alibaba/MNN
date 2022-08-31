struct reduce_shape {
    int outside_size;
    int axis_size;
    int inside_size;
    int outside_step;
};

template <typename M, typename T>
static inline void reduce_mean(const device T *in, device T *out, constant reduce_shape &s, uint2 gid) {
    auto axis_in = in + gid.x * s.outside_step + gid.y;
    M summer = 0;
    for (int i = 0; i < s.axis_size; i++, axis_in += s.inside_size) {
        summer += M(*axis_in);
    }
    out[int(gid.x) * s.inside_size + int(gid.y)] = T(summer / s.axis_size);
}

template <typename M, typename T>
static inline void reduce_sum(const device T *in, device T *out, constant reduce_shape &s, uint2 gid) {
    auto axis_in = in + gid.x * s.outside_step + gid.y;
    M summer = 0;
    for (int i = 0; i < s.axis_size; i++, axis_in += s.inside_size) {
        summer += M(*axis_in);
    }
    out[int(gid.x) * s.inside_size + int(gid.y)] = T(summer);
}

template <typename M, typename T>
static inline void reduce_min(const device T *in, device T *out, constant reduce_shape &s, uint2 gid) {
    auto axis_in = in + gid.x * s.outside_step + gid.y;
    T summer = *axis_in; axis_in += s.inside_size;
    for (int i = 1; i < s.axis_size; i++, axis_in += s.inside_size) {
        summer = min(summer, *axis_in);
    }
    out[int(gid.x) * s.inside_size + int(gid.y)] = summer;
}

template <typename M, typename T>
static inline void reduce_max(const device T *in, device T *out, constant reduce_shape &s, uint2 gid) {
    auto axis_in = in + gid.x * s.outside_step + gid.y;
    T summer = *axis_in; axis_in += s.inside_size;
    for (int i = 1; i < s.axis_size; i++, axis_in += s.inside_size) {
        summer = max(summer, *axis_in);
    }
    out[int(gid.x) * s.inside_size + int(gid.y)] = summer;
}

template <typename M, typename T>
static inline void reduce_prod(const device T *in, device T *out, constant reduce_shape &s, uint2 gid) {
    auto axis_in = in + gid.x * s.outside_step + gid.y;
    M summer = 1;
    for (int i = 0; i < s.axis_size; i++, axis_in += s.inside_size) {
        summer *= M(*axis_in);
    }
    out[int(gid.x) * s.inside_size + int(gid.y)] = T(summer);
}

#define define_reduce(name) \
kernel void reduce_##name##_f(const device ftype *in    [[buffer(0)]], \
                              device ftype *out         [[buffer(1)]], \
                              constant reduce_shape &s  [[buffer(2)]], \
                              uint2 gid                 [[thread_position_in_grid]]) { \
    if (gid.x < (uint)s.outside_size && gid.y < (uint)s.inside_size) reduce_##name<FLOAT, ftype>(in, out, s, gid); \
} \
kernel void reduce_##name##_s(const device int *in      [[buffer(0)]], \
                              device int *out           [[buffer(1)]], \
                              constant reduce_shape &s  [[buffer(2)]], \
                              uint2 gid                 [[thread_position_in_grid]]) { \
    if (gid.x < (uint)s.outside_size && gid.y < (uint)s.inside_size) reduce_##name<int, int>(in, out, s, gid); \
}

define_reduce(mean);
define_reduce(sum);
define_reduce(min);
define_reduce(max);
define_reduce(prod);
