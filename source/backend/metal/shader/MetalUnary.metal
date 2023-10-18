struct unary_shape {
    int width;
    int height;
    int size;
};

static inline float4 MNNEXP(float4 tmp) {
    tmp = clamp(tmp, (float4)-87.0, (float4)87.0);
    return exp(tmp);
}

static inline float4 MNNTANH(float4 value) {
    float4 tmp = MNNEXP((float4)(2.0)*value);
    return (tmp-(float4)1.0)/(tmp+(float4)1.0);
}
static inline float4 neg(float4 value) { return -value; }
static inline float4 square(float4 value) { return value * value; }
static inline float4 expm1(float4 value) {return MNNEXP(value) - 1;}
static inline float4 reciprocal(float4 value) {return 1.0/(value);}
static inline float4 sigmoid(float4 value) {return 1.f / (1.f + MNNEXP(-value));}
static inline float4 log1p(float4 value) {return log(1.f + value);}
static inline float4 hardswish(float4 value) {
    return (float4)(1.0/6.0) * (value * min(max(value+(float4)3, 0), (float4)6));
}
static inline float4 gelu(float4 value) {
    float4 temp = (float4)0.044715 * value * value * value;
    temp = (float4)0.79788458 * (temp + value);
    temp = clamp(temp, (float4)-5.0, (float4)5.0);
    float4 result = ((float4)1.0 + MNNTANH(temp)) * value * (float4)0.5;
    return result;
}

#define define_op(op) \
kernel void unary_##op##_x4(const device ftype4 *in [[buffer(0)]], \
                            device ftype4 *out      [[buffer(1)]], \
                            device unary_shape& s   [[buffer(2)]], \
                            uint3 gid               [[thread_position_in_grid]]) { \
    if (gid.x < (uint)s.width) { \
        int off = gid.z * s.size + gid.y * s.width + gid.x; \
        out[off] = (ftype4)(op((float4)(in[off]))); \
    } \
}

define_op(abs);
define_op(floor);
define_op(ceil);
define_op(expm1);
define_op(square);
define_op(sqrt);
define_op(rsqrt);
define_op(MNNEXP);
define_op(log);
define_op(sin);
define_op(cos);
define_op(tan);
define_op(asin);
define_op(acos);
define_op(atan);
define_op(neg);
define_op(reciprocal)
define_op(MNNTANH);
define_op(sigmoid);
define_op(sign);
define_op(log1p);
define_op(cosh);
define_op(sinh);
define_op(acosh);
define_op(asinh);
define_op(atanh);
define_op(round);
define_op(hardswish);
define_op(gelu);

