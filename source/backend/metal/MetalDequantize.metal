//
//  MetalDequantize.metal
//  MNN
//
//  Created by MNN on 2018/11/13.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <metal_stdlib>
#include "MetalDefine.metal"

using namespace metal;

kernel void dequantize_min_combined_uint8(const device uchar *input0    [[buffer(0)]],
                                          const device ftype *input1    [[buffer(1)]],
                                          const device ftype *input2    [[buffer(2)]],
                                          device ftype *output          [[buffer(3)]],
                                          uint gid                      [[thread_position_in_grid]]) {
    float min_range = input1[0];
    float max_range = input2[0];
    float factor = (max_range - min_range) / UINT8_MAX;
    output[int(gid)] = input0[int(gid)] * factor + min_range;
}
kernel void dequantize_min_combined_uint16(const device ushort *input0  [[buffer(0)]],
                                           const device ftype *input1   [[buffer(1)]],
                                           const device ftype *input2   [[buffer(2)]],
                                           device ftype *output         [[buffer(3)]],
                                           uint gid                     [[thread_position_in_grid]]) {
    float min_range = input1[0];
    float max_range = input2[0];
    float factor = (max_range - min_range) / UINT16_MAX;
    output[int(gid)] = input0[int(gid)] * factor + min_range;
}
kernel void dequantize_min_combined_int8(const device char *input0      [[buffer(0)]],
                                         const device ftype *input1     [[buffer(1)]],
                                         const device ftype *input2     [[buffer(2)]],
                                         device ftype *output           [[buffer(3)]],
                                         uint gid                       [[thread_position_in_grid]]) {
    float min_range = input1[0];
    float max_range = input2[0];
    float half_range = (INT8_MAX - INT8_MIN + 1) / 2.f;
    float factor = (max_range - min_range) / (INT8_MAX - INT8_MIN);
    output[int(gid)] = (input0[int(gid)] + half_range) * factor + min_range;
}
kernel void dequantize_min_combined_int16(const device short *input0    [[buffer(0)]],
                                          const device ftype *input1    [[buffer(1)]],
                                          const device ftype *input2    [[buffer(2)]],
                                          device ftype *output          [[buffer(3)]],
                                          uint gid                      [[thread_position_in_grid]]) {
    float min_range = input1[0];
    float max_range = input2[0];
    float half_range = (INT16_MAX - INT16_MIN + 1) / 2.f;
    float factor = (max_range - min_range) / (INT16_MAX - INT16_MIN);
    output[int(gid)] = (input0[int(gid)] + half_range) * factor + min_range;
}
kernel void dequantize_min_combined_int32(const device int *input0      [[buffer(0)]],
                                          const device ftype *input1    [[buffer(1)]],
                                          const device ftype *input2    [[buffer(2)]],
                                          device ftype *output          [[buffer(3)]],
                                          uint gid                      [[thread_position_in_grid]]) {
    float min_range = input1[0];
    float max_range = input2[0];
    float half_range = (INT32_MAX - INT32_MIN + 1) / 2.f;
    float factor = (max_range - min_range) / (INT32_MAX - INT32_MIN);
    output[int(gid)] = (input0[int(gid)] + half_range) * factor + min_range;
}



kernel void dequantize_min_first_uint8(const device uchar *input0   [[buffer(0)]],
                                       const device ftype *input1   [[buffer(1)]],
                                       const device ftype *input2   [[buffer(2)]],
                                       device ftype *output         [[buffer(3)]],
                                       uint gid                     [[thread_position_in_grid]]) {
    float min_range = input1[0];
    float max_range = input2[0];
    float mask = float(uchar(~0));
    float range_scale = (max_range - min_range) / mask;
    float range_min_rounded = max_range == min_range ? min_range : round(min_range / range_scale) * range_scale;
    output[int(gid)] = input0[int(gid)] * range_scale + range_min_rounded;
}
kernel void dequantize_min_first_uint16(const device ushort *input0 [[buffer(0)]],
                                        const device ftype *input1  [[buffer(1)]],
                                        const device ftype *input2  [[buffer(2)]],
                                        device ftype *output        [[buffer(3)]],
                                        uint gid                    [[thread_position_in_grid]]) {
    float min_range = input1[0];
    float max_range = input2[0];
    float mask = float(ushort(~0));
    float range_scale = (max_range - min_range) / mask;
    float range_min_rounded = max_range == min_range ? min_range : round(min_range / range_scale) * range_scale;
    output[int(gid)] = input0[int(gid)] * range_scale + range_min_rounded;
}
kernel void dequantize_min_first_int8(const device char *input0     [[buffer(0)]],
                                      const device ftype *input1    [[buffer(1)]],
                                      const device ftype *input2    [[buffer(2)]],
                                      device ftype *output          [[buffer(3)]],
                                      uint gid                      [[thread_position_in_grid]]) {
    float min_range = input1[0];
    float max_range = input2[0];
    float mask = float(uchar(~0));
    float range_scale = (max_range - min_range) / mask;
    float range_min_rounded = max_range == min_range ? min_range : round(min_range / range_scale) * range_scale;
    float lowest_quantized = float(INT8_MIN);
    float result_add = range_min_rounded - lowest_quantized * range_scale;
    output[int(gid)] = input0[int(gid)] * range_scale + result_add;
}
kernel void dequantize_min_first_int16(const device short *input0   [[buffer(0)]],
                                       const device ftype *input1   [[buffer(1)]],
                                       const device ftype *input2   [[buffer(2)]],
                                       device ftype *output         [[buffer(3)]],
                                       uint gid                     [[thread_position_in_grid]]) {
    float min_range = input1[0];
    float max_range = input2[0];
    float mask = float(ushort(~0));
    float range_scale = (max_range - min_range) / mask;
    float range_min_rounded = max_range == min_range ? min_range : round(min_range / range_scale) * range_scale;
    float lowest_quantized = float(INT16_MIN);
    float result_add = range_min_rounded - lowest_quantized * range_scale;
    output[int(gid)] = input0[int(gid)] * range_scale + result_add;
}
kernel void dequantize_min_first_int32(const device int *input0     [[buffer(0)]],
                                       const device ftype *input1   [[buffer(1)]],
                                       const device ftype *input2   [[buffer(2)]],
                                       device ftype *output         [[buffer(3)]],
                                       uint gid                     [[thread_position_in_grid]]) {
    float min_range = input1[0];
    float max_range = input2[0];
    float mask = float(uint(~0));
    float range_scale = (max_range - min_range) / mask;
    float range_min_rounded = max_range == min_range ? min_range : round(min_range / range_scale) * range_scale;
    float lowest_quantized = float(INT32_MIN);
    float result_add = range_min_rounded - lowest_quantized * range_scale;
    output[int(gid)] = input0[int(gid)] * range_scale + result_add;
}



kernel void dequantize_scaled_uint8(const device uchar *input0      [[buffer(0)]],
                                    const device ftype *input1      [[buffer(1)]],
                                    const device ftype *input2      [[buffer(2)]],
                                    device ftype *output            [[buffer(3)]],
                                    uint gid                        [[thread_position_in_grid]]) {
    float max_range = input2[0];
    float factor = max_range / UINT8_MAX;
    output[int(gid)] = input0[int(gid)] * factor;
}
kernel void dequantize_scaled_uint16(const device ushort *input0    [[buffer(0)]],
                                     const device ftype *input1     [[buffer(1)]],
                                     const device ftype *input2     [[buffer(2)]],
                                     device ftype *output           [[buffer(3)]],
                                     uint gid                       [[thread_position_in_grid]]) {
    float max_range = input2[0];
    float factor = max_range / UINT16_MAX;
    output[int(gid)] = input0[int(gid)] * factor;
}
kernel void dequantize_scaled_int8(const device char *input0        [[buffer(0)]],
                                   const device ftype *input1       [[buffer(1)]],
                                   const device ftype *input2       [[buffer(2)]],
                                   device ftype *output             [[buffer(3)]],
                                   uint gid                         [[thread_position_in_grid]]) {
    float min_range = input1[0];
    float max_range = input2[0];
    float factor = max(min_range / INT8_MIN, max_range / INT8_MAX);
    output[int(gid)] = input0[int(gid)] * factor;
}
kernel void dequantize_scaled_int16(const device short *input0      [[buffer(0)]],
                                    const device ftype *input1      [[buffer(1)]],
                                    const device ftype *input2      [[buffer(2)]],
                                    device ftype *output            [[buffer(3)]],
                                    uint gid                        [[thread_position_in_grid]]) {
    float min_range = input1[0];
    float max_range = input2[0];
    float factor = max(min_range / INT16_MIN, max_range / INT16_MAX);
    output[int(gid)] = input0[int(gid)] * factor;
}
kernel void dequantize_scaled_int32(const device int *input0        [[buffer(0)]],
                                    const device ftype *input1      [[buffer(1)]],
                                    const device ftype *input2      [[buffer(2)]],
                                    device ftype *output            [[buffer(3)]],
                                    uint gid                        [[thread_position_in_grid]]) {
    float min_range = input1[0];
    float max_range = input2[0];
    float factor = max(min_range / INT32_MIN, max_range / INT32_MAX);
    output[int(gid)] = input0[int(gid)] * factor;
}
