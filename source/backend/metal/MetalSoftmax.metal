//
//  MetalSoftmax.metal
//  MNN
//
//  Created by MNN on 2018/08/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <metal_stdlib>
#include "MetalDefine.metal"

using namespace metal;

struct softmax_shape {
    int inside_size;
    int axis_length;
    int outside_size;
    int flat_length;
};

static inline float softmax_max4(float4 value) {
    return max(max(value[0], value[1]), max(value[2], value[3]));
}

static inline float softmax_sum4(float4 value) {
    return value[0] + value[1] + value[2] + value[3];
}

static inline float4 softmax_filter(float4 value, int z, int limit) {
    return select(0, value, z * 4 + int4(0, 1, 2, 3) < limit);
}

kernel void softmax_tf(const device ftype *in     [[buffer(0)]],
                       device ftype *out          [[buffer(1)]],
                       constant softmax_shape& s   [[buffer(2)]],
                       uint2 gid                   [[thread_position_in_grid]]) {
    if ((int)gid.x >= s.inside_size || (int)gid.y >= s.outside_size) return;
    
    auto axis_off = gid.y * s.axis_length * s.inside_size + gid.x;
    auto axis_in  = in + axis_off;
    auto axis_out = out + axis_off;
    
    // get max
    auto max1 = axis_in[0];
    for (int i = 1; i < s.axis_length; i++) {
        max1 = max(max1, axis_in[i * s.inside_size]);
    }
    
    // get sum
    float sum1 = 0;
    for (int i = 0; i < s.axis_length; i++) {
        sum1 += float(exp(axis_in[i * s.inside_size] - max1));
    }
    
    // output
    for (int i = 0; i < s.axis_length; i++) {
        axis_out[i * s.inside_size] = ftype(exp(float(axis_in[i * s.inside_size] - max1)) / sum1);
    }
}

kernel void softmax_on_reorder(const device ftype4 *in      [[buffer(0)]],
                               device ftype4 *out           [[buffer(1)]],
                               constant softmax_shape& s    [[buffer(2)]],
                               uint2 gid                    [[thread_position_in_grid]]) {
    if ((int)gid.x >= s.inside_size || (int)gid.y >= s.outside_size) return;
    
    auto axis_off = gid.y * s.axis_length * s.inside_size + gid.x;
    auto axis_in  = in + axis_off;
    auto axis_out = out + axis_off;

    // get max
    auto max4 = softmax_filter(float4(axis_in[0]), 0, s.flat_length);
    for (int i = 1; i < s.axis_length; i++) {
        max4 = max(max4, softmax_filter(float4(axis_in[i * s.inside_size]), i, s.flat_length));
    }
    float max1 = softmax_max4(max4);
    
    // get sum
    float4 sum4 = 0;
    for (int i = 0; i < s.axis_length; i++) {
        sum4 += softmax_filter(exp(float4(axis_in[i * s.inside_size] - max1)), i, s.flat_length);
    }
    float sum1 = softmax_sum4(sum4);
    
    // output
    for (int i = 0; i < s.axis_length; i++) {
        axis_out[i * s.inside_size] = ftype4(exp(float4(axis_in[i * s.inside_size]) - max1) / sum1);
    }
}

kernel void softmax_off_reorder(const device ftype4 *in     [[buffer(0)]],
                                device ftype4 *out          [[buffer(1)]],
                                constant softmax_shape& s   [[buffer(2)]],
                                uint2 gid                   [[thread_position_in_grid]]) {
    if ((int)gid.x >= s.inside_size || (int)gid.y >= s.outside_size) return;

    auto axis_off = gid.y * s.axis_length * s.inside_size + gid.x;
    auto axis_in  = in + axis_off;
    auto axis_out = out + axis_off;

    // get max
    auto max4 = axis_in[0];
    for (int i = 1; i < s.axis_length; i++) {
        max4 = max(max4, axis_in[i * s.inside_size]);
    }

    // get sum
    float4 sum4 = 0;
    for (int i = 0; i < s.axis_length; i++) {
        sum4 += exp(float4(axis_in[i * s.inside_size] - max4));
    }

    // output
    for (int i = 0; i < s.axis_length; i++) {
        axis_out[i * s.inside_size] = ftype4(exp(float4(axis_in[i * s.inside_size] - max4)) / sum4);
    }
}

kernel void softmax_m_tf(const device ftype *in     [[buffer(0)]],
                         device ftype *out          [[buffer(1)]],
                         constant softmax_shape& s  [[buffer(2)]],
                         threadgroup float *tmp     [[threadgroup(0)]],
                         uint3 threads              [[threads_per_threadgroup]],
                         uint tid                   [[thread_index_in_threadgroup]],
                         uint3 gid                  [[thread_position_in_grid]]) {
    if ((int)gid.y >= s.inside_size || (int)gid.z >= s.outside_size) return;
    
    auto axis_off = gid.z * s.axis_length * s.inside_size + gid.y;
    auto axis_in  = in + axis_off;
    auto axis_out = out + axis_off;
    int axis_step = gid.x == threads.x ? s.axis_length - s.axis_length / threads.x * (threads.x - 1) : s.axis_length / threads.x;
    int axis_stt  = s.axis_length / threads.x * gid.x;
    int axis_end  = axis_stt + axis_step;
    auto axis_tmp = tmp + tid / threads.x * threads.x;
    
    // get max
    auto max1 = axis_in[axis_stt];
    for (int i = 1 + axis_stt; i < axis_end; i++) {
        max1 = max(max1, axis_in[i * s.inside_size]);
    }
    axis_tmp[(int)gid.x] = max1;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto tmp_ftype = (threadgroup ftype *)axis_tmp;
    for (int i = 0; i < (int)threads.x; i++) {
        max1 = max(max1, tmp_ftype[i]);
    }
    
    // get sum
    float sum1 = 0;
    for (int i = axis_stt; i < axis_end; i++) {
        sum1 += float(exp(axis_in[i * s.inside_size] - max1));
    }
    axis_tmp[(int)gid.x] = sum1;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum1 = 0;
    for (int i = 0; i < (int)threads.x; i++) {
        sum1 += axis_tmp[i];
    }
    
    // output
    for (int i = axis_stt; i < axis_end; i++) {
        axis_out[i * s.inside_size] = ftype(exp(float(axis_in[i * s.inside_size] - max1)) / sum1);
    }
}

kernel void softmax_m_on_reorder(const device ftype4 *in    [[buffer(0)]],
                                 device ftype4 *out         [[buffer(1)]],
                                 constant softmax_shape& s  [[buffer(2)]],
                                 threadgroup float *tmp     [[threadgroup(0)]],
                                 uint3 threads              [[threads_per_threadgroup]],
                                 uint tid                   [[thread_index_in_threadgroup]],
                                 uint3 gid                  [[thread_position_in_grid]]) {
    if ((int)gid.y >= s.inside_size || (int)gid.z >= s.outside_size) return;
    
    auto axis_off = gid.z * s.axis_length * s.inside_size + gid.y;
    auto axis_in  = in + axis_off;
    auto axis_out = out + axis_off;
    int axis_step = gid.x == threads.x ? s.axis_length - s.axis_length / threads.x * (threads.x - 1) : s.axis_length / threads.x;
    int axis_stt  = s.axis_length / threads.x * gid.x;
    int axis_end  = axis_stt + axis_step;
    auto axis_tmp = tmp + tid / threads.x * threads.x;
    
    // get max
    auto max4 = softmax_filter(float4(axis_in[axis_stt]), axis_stt, s.flat_length);
    for (int i = 1 + axis_stt; i < axis_end; i++) {
        max4 = max(max4, softmax_filter(float4(axis_in[i * s.inside_size]), i, s.flat_length));
    }
    float max1 = softmax_max4(max4);
    axis_tmp[(int)gid.x] = max1;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = 0; i < (int)threads.x; i++) {
        max1 = max(max1, axis_tmp[i]);
    }
    
    // get sum
    float4 sum4 = 0;
    for (int i = axis_stt; i < axis_end; i++) {
        sum4 += softmax_filter(exp(float4(axis_in[i * s.inside_size] - max1)), i, s.flat_length);
    }
    float sum1 = softmax_sum4(sum4);
    axis_tmp[(int)gid.x] = sum1;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum1 = 0;
    for (int i = 0; i < (int)threads.x; i++) {
        sum1 += axis_tmp[i];
    }
    
    // output
    for (int i = axis_stt; i < axis_end; i++) {
        axis_out[i * s.inside_size] = ftype4(exp(float4(axis_in[i * s.inside_size]) - max1) / sum1);
    }
}

kernel void softmax_m_off_reorder(const device ftype4 *in   [[buffer(0)]],
                                  device ftype4 *out        [[buffer(1)]],
                                  constant softmax_shape& s [[buffer(2)]],
                                  threadgroup float4 *tmp   [[threadgroup(0)]],
                                  uint3 threads              [[threads_per_threadgroup]],
                                  uint tid                  [[thread_index_in_threadgroup]],
                                  uint3 gid                 [[thread_position_in_grid]]) {
    if ((int)gid.y >= s.inside_size || (int)gid.z >= s.outside_size) return;
    
    auto axis_off = gid.z * s.axis_length * s.inside_size + gid.y;
    auto axis_in  = in + axis_off;
    auto axis_out = out + axis_off;
    int axis_step = gid.x == threads.x ? s.axis_length - s.axis_length / threads.x * (threads.x - 1) : s.axis_length / threads.x;
    int axis_stt  = s.axis_length / threads.x * gid.x;
    int axis_end  = axis_stt + axis_step;
    auto axis_tmp = tmp + tid / threads.x * threads.x;
    
    // get max
    auto max4 = axis_in[axis_stt];
    for (int i = 1 + axis_stt; i < axis_end; i++) {
        max4 = max(max4, axis_in[i * s.inside_size]);
    }
    auto tmp_ftype4 = (threadgroup ftype4 *)axis_tmp;
    tmp_ftype4[(int)gid.x] = max4;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = 0; i < (int)threads.x; i++) {
        max4 = max(max4, tmp_ftype4[i]);
    }
    
    // get sum
    float4 sum4 = 0;
    for (int i = axis_stt; i < axis_end; i++) {
        sum4 += exp(float4(axis_in[i * s.inside_size] - max4));
    }
    axis_tmp[(int)gid.x] = sum4;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum4 = 0;
    for (int i = 0; i < (int)threads.x; i++) {
        sum4 += axis_tmp[i];
    }
    
    // output
    for (int i = axis_stt; i < axis_end; i++) {
        axis_out[i * s.inside_size] = ftype4(exp(float4(axis_in[i * s.inside_size] - max4)) / sum4);
    }
}
