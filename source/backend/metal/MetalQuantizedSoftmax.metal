//
//  MetalQuantizedSoftmax.metal
//  MNN
//
//  Created by MNN on 2018/11/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <metal_stdlib>
#include "MetalDefine.metal"
#include "MetalFixedPoint.metal"

using namespace metal;
using namespace MNN;

#define kScaledDiffIntegerBits   5
#define kAccumulationIntegerBits 12

using FixedPointScaledDiff  = FixedPoint<int, kScaledDiffIntegerBits>;
using FixedPointAccum       = FixedPoint<int, kAccumulationIntegerBits>;
using FixedPoint0           = FixedPoint<int, 0>;

struct quantized_softmax_constants {
    int outer_size;
    int inner_size;
    int diff_min;
    int input_beta_multiplier;
    int input_beta_left_shift;
};

kernel void quantized_softmax(const device uchar *in                    [[buffer(0)]],
                              device uchar *out                         [[buffer(1)]],
                              constant quantized_softmax_constants& cst [[buffer(2)]],
                              uint gid                                  [[thread_position_in_grid]]) {
    if ((int)gid >= cst.outer_size) return;
    
    auto c_in = in + gid * cst.inner_size;
    auto c_out = out + gid * cst.inner_size;
    
    // get max
    uchar max_in_channel = 0;
    for (int i = 0; i < cst.inner_size; i++) max_in_channel = max(max_in_channel, c_in[i]);
    
    // sum of exp
    FixedPointAccum sum_of_exps = FixedPointAccum::Zero();
    for (int i = 0; i < cst.inner_size; i++) {
        int input_diff = (int)c_in[i] - max_in_channel;
        if (input_diff >= cst.diff_min) {
            const int input_diff_rescaled = MultiplyByQuantizedMultiplierGreaterThanOne(input_diff, cst.input_beta_multiplier, cst.input_beta_left_shift);
            const FixedPointScaledDiff scaled_diff_f8 = FixedPointScaledDiff::FromRaw(input_diff_rescaled);
            sum_of_exps = sum_of_exps + Rescale<kAccumulationIntegerBits>(exp_on_negative_values(scaled_diff_f8));
        }
    }
    
    // scale
    int fixed_sum_of_exps = sum_of_exps.raw();
    int headroom_plus_one = clz(static_cast<uint>(fixed_sum_of_exps));
    int num_bits_over_unit = kAccumulationIntegerBits - headroom_plus_one;
    int shifted_sum_minus_one = static_cast<int>((static_cast<uint>(fixed_sum_of_exps) << headroom_plus_one) - (static_cast<uint>(1) << 31));
    FixedPoint0 shifted_scale = one_over_one_plus_x_for_x_in_0_1(FixedPoint0::FromRaw(shifted_sum_minus_one));

    // write
    for (int i = 0; i < cst.inner_size; i++) {
        int input_diff = static_cast<int>(c_in[i]) - max_in_channel;
        if (input_diff >= cst.diff_min) {
            const int input_diff_rescaled = MultiplyByQuantizedMultiplierGreaterThanOne(input_diff, cst.input_beta_multiplier, cst.input_beta_left_shift);
            const FixedPointScaledDiff scaled_diff_f8 = FixedPointScaledDiff::FromRaw(input_diff_rescaled);
            FixedPoint0 exp_in_0 = exp_on_negative_values(scaled_diff_f8);
            int unsat_output = round_divide_by_pot( (shifted_scale * exp_in_0).raw(), num_bits_over_unit + 31 - 8);
            c_out[i] = max(min(unsat_output, 255), 0);
        } else {
            c_out[i] = 0;
        }
    }
}
