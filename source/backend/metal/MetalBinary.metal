//
//  MetalBinary.metal
//  MNN
//
//  Created by MNN on 2018/09/03.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <metal_stdlib>
#include "MetalDefine.metal"

using namespace metal;

struct binary_op_shape {
    int in0_data_count;
    int in1_data_count;
    int output_data_count;
    int output_width;
    int output_size;
    int output_dimensions;
};

typedef enum: int {
    binary_op_add       = 0,
    binary_op_sub       = 1,
    binary_op_mul       = 2,
    binary_op_div       = 3,
    binary_op_max       = 4,
    binary_op_min       = 5,
    binary_op_pow       = 6,
    binary_op_real_div  = 7,
    binary_op_minimum   = 8,
    binary_op_maximum   = 9,
    binary_op_greater   = 10,
} binary_op_type;

static inline ftype binary_add(ftype value1, ftype value2) {
    return value1 + value2;
}

static inline ftype binary_sub(ftype value1, ftype value2) {
    return value1 - value2;
}

static inline ftype binary_mul(ftype value1, ftype value2) {
    return value1 * value2;
}

static inline ftype binary_div(ftype value1, ftype value2) {
    return value2 == 0 ? 0 : value1 / value2;
}

static inline ftype binary_max(ftype value1, ftype value2) {
    return max(value1, value2);
}

static inline ftype binary_min(ftype value1, ftype value2) {
    return min(value1, value2);
}

static inline ftype binary_pow(ftype value1, ftype value2) {
    return pow(value1, value2);
}

static inline ftype binary_real_div(ftype value1, ftype value2) {
    return value2 == 0 ? 0 : value1 / value2;
}

static inline ftype binary_activate(ftype value1, ftype value2, binary_op_type type) {
    switch (type) {
        case binary_op_add:
            return binary_add(value1, value2);
        case binary_op_sub:
            return binary_sub(value1, value2);
        case binary_op_mul:
            return binary_mul(value1, value2);
        case binary_op_div:
            return binary_div(value1, value2);
        case binary_op_max:
        case binary_op_maximum:
            return binary_max(value1, value2);
        case binary_op_min:
        case  binary_op_minimum:
            return binary_min(value1, value2);
        case binary_op_pow:
            return binary_pow(value1, value2);
        case binary_op_real_div:
            return binary_real_div(value1, value2);
        default: // not supported
            return 0;
    }
}

kernel void binary_normal(const device ftype *in0       [[buffer(0)]],
                          const device ftype *in1       [[buffer(1)]],
                          device ftype *out             [[buffer(2)]],
                          constant binary_op_shape& s   [[buffer(3)]],
                          constant binary_op_type& type [[buffer(4)]],
                          uint gid                      [[thread_position_in_grid]]) {
    if ((int)gid >= s.output_data_count) return;
    
    // data count == 0 means scalar input or shape(1, 1, 1, ..., 1)
    auto value0 = in0[s.in0_data_count == 1 ? 0 : int(gid)];
    auto value1 = in1[s.in1_data_count == 1 ? 0 : int(gid)];
    out[int(gid)] = binary_activate(value0, value1, type);
}

kernel void binary_notshape(const device ftype *in0         [[buffer(0)]],
                            const device ftype *in1         [[buffer(1)]],
                            device ftype *out               [[buffer(2)]],
                            constant binary_op_shape& s     [[buffer(3)]],
                            constant binary_op_type& type   [[buffer(4)]],
                            constant int *in0_dims          [[buffer(5)]],
                            constant int *in1_dims          [[buffer(6)]],
                            constant int *in0_strides       [[buffer(7)]],
                            constant int *in1_strides       [[buffer(8)]],
                            constant int *out_strides       [[buffer(9)]],
                            uint gid                        [[thread_position_in_grid]]) {
    if ((int)gid >= s.output_data_count) return;
    
    int off0 = 0, off1 = 0, rest = gid;
    for (int i = 0; i < s.output_dimensions; i++){
        int d_off = rest / out_strides[i];
        rest = rest % out_strides[i];
        
        int a_cord = d_off < in0_dims[i] ? d_off : 0;
        int b_cord = d_off < in1_dims[i] ? d_off : 0;
        off0 += a_cord * in0_strides[i];
        off1 += b_cord * in1_strides[i];
    }
    out[int(gid)] = binary_activate(in0[off0], in1[off1], type);
}
