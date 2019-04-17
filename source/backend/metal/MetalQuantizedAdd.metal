//
//  MetalQuantizedAdd.metal
//  MNN
//
//  Created by MNN on 2018/11/12.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <metal_stdlib>
#include "MetalDefine.metal"

using namespace metal;
using namespace MNN;

struct quantized_add_constansts {
    int input1_offset;
    int input2_offset;
    int output_offset;
    int input1_multiplier;
    int input2_multiplier;
    int output_multiplier;
    int right_shift_1;
    int right_shift_2;
    int input1_left_shift;
    int input2_left_shift;
    int output_left_shift;
    int output_right_shift;
    int output_activation_min;
    int output_activation_max;
};

kernel void quantized_add(const device uchar *in1              [[buffer(0)]],
                       const device uchar *in2              [[buffer(1)]],
                       device uchar *out                    [[buffer(2)]],
                       constant quantized_add_constansts& c    [[buffer(3)]],
                       uint gid                             [[thread_position_in_grid]]) {
    int input1_val = c.input1_offset + in1[int(gid)];
    int input2_val = c.input2_offset + in2[int(gid)];
    int shifted_input1_val = input1_val * c.input1_left_shift;
    int shifted_input2_val = input2_val * c.input2_left_shift;
    int scaled_input1_val = round_divide_by_pot(saturate_round_x2_high_mul(shifted_input1_val, c.input1_multiplier), c.right_shift_1);
    int scaled_input2_val = round_divide_by_pot(saturate_round_x2_high_mul(shifted_input2_val, c.input2_multiplier), c.right_shift_2);
    int raw_sum = scaled_input1_val + scaled_input2_val;
    int raw_output = round_divide_by_pot(saturate_round_x2_high_mul(raw_sum * (1 << c.output_left_shift), c.output_multiplier), c.output_right_shift) + c.output_offset;
    int clamped_output = clamp(raw_output, c.output_activation_min, c.output_activation_max);
    out[int(gid)] = uchar(uint(clamped_output));
}
