/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "OptimizedComputer.hpp"
#include <string.h>
#include "Macro.h"
#ifdef MNN_USE_NEON
#include <arm_neon.h>
#endif

namespace MNN {
namespace Optimized {
// avgpooling
void AveragePool(const uint8_t* input_data, const std::vector<int>& input_dims, int stride_width, int stride_height,
                 int pad_width, int pad_height, int filter_width, int filter_height, int mOutputActivationMin,
                 int mOutputActivationMax, uint8_t* output_data, const std::vector<int>& output_dims) {
    MNN_ASSERT(mOutputActivationMin < mOutputActivationMax);
    MNN_ASSERT(input_dims.at(0) == output_dims.at(0));
    const int batches = input_dims.at(0);
    MNN_ASSERT(input_dims.at(3) == output_dims.at(3));
    const int depth         = input_dims.at(3);
    const int input_height  = input_dims.at(1);
    const int input_width   = input_dims.at(2);
    const int output_height = output_dims.at(1);
    const int output_width  = output_dims.at(2);

    for (int batch = 0; batch < batches; ++batch) {
        for (int out_y = 0; out_y < output_height; ++out_y) {
            for (int out_x = 0; out_x < output_width; ++out_x) {
                const int in_x_origin    = (out_x * stride_width) - pad_width;
                const int in_y_origin    = (out_y * stride_height) - pad_height;
                const int filter_x_start = std::max(0, -in_x_origin);
                const int filter_x_end   = std::min(filter_width, input_width - in_x_origin);
                const int filter_y_start = std::max(0, -in_y_origin);
                const int filter_y_end   = std::min(filter_height, input_height - in_y_origin);
                const int filter_count   = (filter_x_end - filter_x_start) * (filter_y_end - filter_y_start);
                uint8_t* output_ptr      = output_data + Offset(output_dims, 0, out_x, out_y, batch);
                if (0 == filter_count) {
                    ::memset(output_ptr, mOutputActivationMin, depth * sizeof(uint8_t));
                    continue;
                }
                // 1280 required by Inception v3
                static constexpr int kAccBufferMaxSize = 2048;
                MNN_ASSERT(depth <= kAccBufferMaxSize);
                uint16_t acc[kAccBufferMaxSize];
                memset(acc, 0, depth * sizeof(acc[0]));
                const uint8_t* input_ptr = input_data + input_dims.at(3) * in_x_origin +
                                           input_dims.at(2) * input_dims.at(3) * in_y_origin +
                                           input_dims.at(1) * input_dims.at(2) * input_dims.at(3) * batch;
                for (int fy = filter_y_start; fy < filter_y_end; fy++) {
                    const uint8_t* input_row_ptr =
                        input_ptr + fy * input_dims.at(2) * input_dims.at(3) + filter_x_start * input_dims.at(3);
                    for (int fx = filter_x_start; fx < filter_x_end; fx++) {
                        int channel = 0;
#ifdef MNN_USE_NEON
                        for (; channel <= depth - 16; channel += 16) {
                            uint16x8_t acc_reg[2];
                            for (int i = 0; i < 2; i++) {
                                acc_reg[i] = vld1q_u16(acc + channel + 8 * i);
                            }
                            uint8x16_t input_reg = vld1q_u8(input_row_ptr);
                            input_row_ptr += 16;
                            acc_reg[0] = vaddw_u8(acc_reg[0], vget_low_u8(input_reg));
                            acc_reg[1] = vaddw_u8(acc_reg[1], vget_high_u8(input_reg));
                            for (int i = 0; i < 2; i++) {
                                vst1q_u16(acc + channel + 8 * i, acc_reg[i]);
                            }
                        }
                        for (; channel <= depth - 8; channel += 8) {
                            uint16x8_t acc_reg  = vld1q_u16(acc + channel);
                            uint8x8_t input_reg = vld1_u8(input_row_ptr);
                            input_row_ptr += 8;
                            acc_reg = vaddw_u8(acc_reg, input_reg);
                            vst1q_u16(acc + channel, acc_reg);
                        }
#endif
                        for (; channel < depth; ++channel) {
                            acc[channel] += *input_row_ptr++;
                        }
                    }
                }
                int channel = 0;
#ifdef MNN_USE_NEON
#define AVGPOOL_DIVIDING_BY(FILTER_COUNT)                                      \
    if (filter_count == FILTER_COUNT) {                                        \
        for (; channel <= depth - 8; channel += 8) {                           \
            uint16_t buf[8];                                                   \
            for (int i = 0; i < 8; i++) {                                      \
                buf[i] = (acc[channel + i] + FILTER_COUNT / 2) / FILTER_COUNT; \
            }                                                                  \
            uint8x8_t buf8 = vqmovn_u16(vld1q_u16(buf));                       \
            buf8           = vmin_u8(buf8, vdup_n_u8(mOutputActivationMax));   \
            buf8           = vmax_u8(buf8, vdup_n_u8(mOutputActivationMin));   \
            vst1_u8(output_ptr + channel, buf8);                               \
        }                                                                      \
    }
                AVGPOOL_DIVIDING_BY(9)
                AVGPOOL_DIVIDING_BY(15)
#undef AVGPOOL_DIVIDING_BY
                for (; channel <= depth - 8; channel += 8) {
                    uint16_t buf[8];
                    for (int i = 0; i < 8; i++) {
                        buf[i] = (acc[channel + i] + filter_count / 2) / filter_count;
                    }
                    uint8x8_t buf8 = vqmovn_u16(vld1q_u16(buf));
                    buf8           = vmin_u8(buf8, vdup_n_u8(mOutputActivationMax));
                    buf8           = vmax_u8(buf8, vdup_n_u8(mOutputActivationMin));
                    vst1_u8(output_ptr + channel, buf8);
                }
#endif
                for (; channel < depth; ++channel) {
                    uint16_t a          = (acc[channel] + filter_count / 2) / filter_count;
                    a                   = std::max<uint16_t>(a, mOutputActivationMin);
                    a                   = std::min<uint16_t>(a, mOutputActivationMax);
                    output_ptr[channel] = static_cast<uint8_t>(a);
                }
            }
        }
    }
}

void Logistic(const uint8_t* input_data, const std::vector<int>& input_dims, int32_t inputZeroPoint,
              int32_t input_range_radius, int32_t input_multiplier, int input_left_shift, uint8_t* output_data,
              const std::vector<int>& output_dims) {
    int size = 1;
    for (int i = 0; i < input_dims.size(); i++) {
        size *= input_dims.at(i);
    }

    int c = 0;

#ifdef MNN_USE_NEON
    // Handle 16 values at a time
    for (; c <= size - 16; c += 16) {
        // Read input uint8 values, cast to int16 and subtract inputZeroPoint
        uint8x16_t input_val_u8 = vld1q_u8(input_data + c);
        int16x8_t input_val_centered_0 =
            vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(input_val_u8))), vdupq_n_s16(inputZeroPoint));
        int16x8_t input_val_centered_1 =
            vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(input_val_u8))), vdupq_n_s16(inputZeroPoint));

        // Prepare the bit masks that we will use at the end to implement the logic
        // that was expressed in the scalar code with branching:
        //   if (input_val_centered < -input_range_radius) {
        //     output_val = 0;
        //   } else if (input_val_centered > input_range_radius) {
        //     output_val = 255;
        //   } else {
        //     ...
        uint16x8_t mask_rightclamp_0 = vcgtq_s16(input_val_centered_0, vdupq_n_s16(input_range_radius));
        uint16x8_t mask_rightclamp_1 = vcgtq_s16(input_val_centered_1, vdupq_n_s16(input_range_radius));
        uint16x8_t mask_leftclamp_0  = vcgeq_s16(input_val_centered_0, vdupq_n_s16(-input_range_radius));
        uint16x8_t mask_leftclamp_1  = vcgeq_s16(input_val_centered_1, vdupq_n_s16(-input_range_radius));
        uint8x16_t mask_rightclamp = vcombine_u8(vshrn_n_u16(mask_rightclamp_0, 8), vshrn_n_u16(mask_rightclamp_1, 8));
        uint8x16_t mask_leftclamp  = vcombine_u8(vshrn_n_u16(mask_leftclamp_0, 8), vshrn_n_u16(mask_leftclamp_1, 8));

        // This performs what is expressed in the scalar code as
        // const int32 input_val_rescaled =
        //     MultiplyByQuantizedMultiplierGreaterThanOne(
        //         input_val_centered, input_multiplier, input_left_shift);
        int32x4_t input_val_rescaled_0 =
            vshlq_s32(vmovl_s16(vget_low_s16(input_val_centered_0)), vdupq_n_s32(input_left_shift));
        int32x4_t input_val_rescaled_1 =
            vshlq_s32(vmovl_s16(vget_high_s16(input_val_centered_0)), vdupq_n_s32(input_left_shift));
        int32x4_t input_val_rescaled_2 =
            vshlq_s32(vmovl_s16(vget_low_s16(input_val_centered_1)), vdupq_n_s32(input_left_shift));
        int32x4_t input_val_rescaled_3 =
            vshlq_s32(vmovl_s16(vget_high_s16(input_val_centered_1)), vdupq_n_s32(input_left_shift));
        input_val_rescaled_0 = vqrdmulhq_n_s32(input_val_rescaled_0, input_multiplier);
        input_val_rescaled_1 = vqrdmulhq_n_s32(input_val_rescaled_1, input_multiplier);
        input_val_rescaled_2 = vqrdmulhq_n_s32(input_val_rescaled_2, input_multiplier);
        input_val_rescaled_3 = vqrdmulhq_n_s32(input_val_rescaled_3, input_multiplier);

        // Invoke gemmlowp::logistic on FixedPoint wrapping int32x4_t
        using FixedPoint4                 = FixedPoint<int32x4_t, 4>;
        using FixedPoint0                 = FixedPoint<int32x4_t, 0>;
        const FixedPoint4 input_val_f4_0  = FixedPoint4::FromRaw(input_val_rescaled_0);
        const FixedPoint4 input_val_f4_1  = FixedPoint4::FromRaw(input_val_rescaled_1);
        const FixedPoint4 input_val_f4_2  = FixedPoint4::FromRaw(input_val_rescaled_2);
        const FixedPoint4 input_val_f4_3  = FixedPoint4::FromRaw(input_val_rescaled_3);
        const FixedPoint0 output_val_f0_0 = logistic(input_val_f4_0);
        const FixedPoint0 output_val_f0_1 = logistic(input_val_f4_1);
        const FixedPoint0 output_val_f0_2 = logistic(input_val_f4_2);
        const FixedPoint0 output_val_f0_3 = logistic(input_val_f4_3);

        // Divide by 2^23 as in the scalar code
        int32x4_t output_val_s32_0 = RoundingDivideByPOT(output_val_f0_0.raw(), 23);
        int32x4_t output_val_s32_1 = RoundingDivideByPOT(output_val_f0_1.raw(), 23);
        int32x4_t output_val_s32_2 = RoundingDivideByPOT(output_val_f0_2.raw(), 23);
        int32x4_t output_val_s32_3 = RoundingDivideByPOT(output_val_f0_3.raw(), 23);

        // Cast output values to uint8, saturating
        int16x8_t output_val_s16_0 = vcombine_s16(vqmovn_s32(output_val_s32_0), vqmovn_s32(output_val_s32_1));
        int16x8_t output_val_s16_1 = vcombine_s16(vqmovn_s32(output_val_s32_2), vqmovn_s32(output_val_s32_3));
        uint8x16_t output_val_u8   = vcombine_u8(vqmovun_s16(output_val_s16_0), vqmovun_s16(output_val_s16_1));

        // Perform the bit-masking with the bit masks computed at the beginning,
        // see the comment there.
        output_val_u8 = vorrq_u8(output_val_u8, mask_rightclamp);
        output_val_u8 = vandq_u8(output_val_u8, mask_leftclamp);

        // Store back to memory
        vst1q_u8(output_data + c, output_val_u8);
    }
#endif
    // Leftover loop: handle one value at a time with scalar code.
    for (; c < size; ++c) {
        const uint8_t input_val_u8       = input_data[c];
        const int32_t input_val_centered = static_cast<int32_t>(input_val_u8) - inputZeroPoint;
        uint8_t output_val;
        if (input_val_centered < -input_range_radius) {
            output_val = 0;
        } else if (input_val_centered > input_range_radius) {
            output_val = 255;
        } else {
            const int32_t input_val_rescaled =
                MultiplyByQuantizedMultiplierGreaterThanOne(input_val_centered, input_multiplier, input_left_shift);
            const FixedPoint<int32_t, 4> input_val_f4  = FixedPoint<int32_t, 4>::FromRaw(input_val_rescaled);
            const FixedPoint<int32_t, 0> output_val_f0 = logistic(input_val_f4);
            int32_t output_val_s32                     = RoundingDivideByPOT(output_val_f0.raw(), 23);
            if (output_val_s32 == 256) {
                output_val_s32 = 255;
            }
            MNN_ASSERT(output_val_s32 >= 0);
            MNN_ASSERT(output_val_s32 <= 255);
            output_val = static_cast<uint8_t>(output_val_s32);
        }
        output_data[c] = output_val;
    }
}

} // namespace Optimized
} // namespace MNN
