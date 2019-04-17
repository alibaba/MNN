//
//  MetalUnary.metal
//  MNN
//
//  Created by MNN on 2018/09/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <metal_stdlib>
#include "MetalDefine.metal"

using namespace metal;

struct unary_shape {
    int width;
    int height;
    int size;
};

static inline ftype4 neg(ftype4 value) { return -value; }
static inline ftype  neg(ftype value)  { return -value; }
static inline ftype4 square(ftype4 value) { return value * value; }
static inline ftype  square(ftype value)  { return value * value; }
//static inline ftype4 reciprocal(ftype4 value) { return 1 / value; }
//static inline ftype  reciprocal(ftype value)  { return 1 / value; }

#define define_op(op) \
kernel void unary_##op##_x4(const device ftype4 *in [[buffer(0)]], \
                            device ftype4 *out      [[buffer(1)]], \
                            device unary_shape& s   [[buffer(2)]], \
                            uint3 gid               [[thread_position_in_grid]]) { \
    if (gid.x < (uint)s.width && gid.y < (uint)s.height) { \
        int off = gid.z * s.size + gid.y * s.width + gid.x; \
        out[off] = op(in[off]); \
    } \
} \
kernel void unary_##op##_x1(const device ftype *in  [[buffer(0)]], \
                            device ftype *out       [[buffer(1)]], \
                            device unary_shape& s   [[buffer(2)]], \
                            uint3 gid               [[thread_position_in_grid]]) { \
    if (gid.x < (uint)s.width && gid.y < (uint)s.height) { \
        int off = gid.z * s.size + gid.y * s.width + gid.x; \
        out[off] = op(in[off]); \
    } \
}

define_op(abs);
define_op(neg);
//define_op(floor);
define_op(ceil);
define_op(square);
define_op(sqrt);
define_op(rsqrt);
define_op(exp);
//define_op(log);
//define_op(sin);
//define_op(cos);
//define_op(tan);
//define_op(asin);
//define_op(acos);
//define_op(atan);
//define_op(reciprocal);
