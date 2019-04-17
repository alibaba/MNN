//
//  MetalConvolution.metal
//  MNN
//
//  Created by MNN on 2018/08/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <metal_stdlib>
#include "MetalConvolutionActivation.metal"

using namespace metal;
using namespace MNN;

#define CONV_UNROLL (4)

kernel void conv_quantize(const device ftype4 *in   [[buffer(0)]],
                          device char4 *out         [[buffer(1)]],
                          constant float& scale     [[buffer(2)]],
                          constant int2& range      [[buffer(3)]],
                          uint gid                  [[thread_position_in_grid]]) {
    // ftype4 -> int4 -> char4   : right
    // ftype4 -> char4           : wrong
    int4 qnt = int4(round(float4(in[int(gid)]) * scale));
    out[int(gid)] = char4(clamp(qnt, range.x, range.y));
}

struct conv_constants {
    int input_width;
    int input_height;
    int input_size;
    int input_slice;
    int output_width;
    int output_height;
    int output_size;
    int output_slice;
    int threadgroup_input_slice;
    
    int kernel_x;
    int kernel_y;
    int kernel_size;
    int stride_x;
    int stride_y;
    int pad_x;
    int pad_y;
    int dilation_x;
    int dilation_y;
    conv_activation_type activation;    
};

kernel void conv(const device ftype4 *in        [[buffer(0)]],
                 device ftype4 *out             [[buffer(1)]],
                 constant conv_constants& cst   [[buffer(2)]],
                 const device ftype4x4 *wt      [[buffer(3)]],
                 const device ftype4 *biasTerms [[buffer(4)]],
                 uint3 gid                      [[thread_position_in_grid]]) {
    if ((int)gid.x >= cst.output_width || (int)gid.y >= cst.output_height) return;
    
    int offset_x = (int)gid.x * cst.stride_x - cst.pad_x;
    int offset_y = (int)gid.y * cst.stride_y - cst.pad_y;
    int sx = max(0, (UP_DIV(-offset_x, cst.dilation_x)));
    int ex = min(cst.kernel_x, UP_DIV(cst.input_width - offset_x, cst.dilation_x));
    short kw = ex - sx;
    int sy = max(0, (UP_DIV(-offset_y, cst.dilation_y)));
    int ey = min(cst.kernel_y, UP_DIV(cst.input_height - offset_y, cst.dilation_y));
    short kh = ey - sy;
    offset_x += sx * cst.dilation_x;
    offset_y += sy * cst.dilation_y;
    
    auto z_in  = in                                                   + offset_y * cst.input_width    + offset_x;
    auto z_wt  = wt  + (int)gid.z * cst.input_slice * cst.kernel_size + sy * cst.kernel_x             + sx;
    auto z_out = out + (int)gid.z * cst.output_size                   + (int)gid.y * cst.output_width + (int)gid.x;

    int dilation_h = cst.input_width * cst.dilation_y;
    float4 result = float4(biasTerms[(short)gid.z]);
    for (auto z = 0; z < cst.input_slice; z++) {
        for (auto y = 0; y < kh; y++) {
            for (auto x = 0; x < kw; x++) {
                auto wt4 = z_wt[z * cst.kernel_size + y * cst.kernel_x + x];
                auto in4 = z_in[z * cst.input_size  + y * dilation_h   + x * cst.dilation_x];
                result += float4(in4 * wt4);
            }
        }
    }
    *z_out = activate(ftype4(result), cst.activation);
}

kernel void qntconv(const device char4 *in          [[buffer(0)]],
                    device ftype4 *out              [[buffer(1)]],
                    constant conv_constants& cst    [[buffer(2)]],
                    const device char4x4 *wt        [[buffer(3)]],
                    const device ftype4 *biasTerms  [[buffer(4)]],
                    const device float4 *alpha      [[buffer(5)]],
                    uint3 gid                       [[thread_position_in_grid]]) {
    if ((int)gid.x >= cst.output_width || (int)gid.y >= cst.output_height) return;
    
    int offset_x = (int)gid.x * cst.stride_x - cst.pad_x;
    int offset_y = (int)gid.y * cst.stride_y - cst.pad_y;
    int sx = max(0, (UP_DIV(-offset_x, cst.dilation_x)));
    int ex = min(cst.kernel_x, UP_DIV(cst.input_width - offset_x, cst.dilation_x));
    short kw = ex - sx;
    int sy = max(0, (UP_DIV(-offset_y, cst.dilation_y)));
    int ey = min(cst.kernel_y, UP_DIV(cst.input_height - offset_y, cst.dilation_y));
    short kh = ey - sy;
    offset_x += sx * cst.dilation_x;
    offset_y += sy * cst.dilation_y;
    
    auto z_in  = in                                                   + offset_y * cst.input_width    + offset_x;
    auto z_wt  = wt  + (int)gid.z * cst.input_slice * cst.kernel_size + sy * cst.kernel_x             + sx;
    auto z_out = out + (int)gid.z * cst.output_size                   + (int)gid.y * cst.output_width + (int)gid.x;
    
    int dilation_h = cst.input_width * cst.dilation_y;
    float4 result = 0;
    for (auto z = 0; z < cst.input_slice; z++) {
        for (auto y = 0; y < kh; y++) {
            for (auto x = 0; x < kw; x++) {
                auto wt4 = z_wt[z * cst.kernel_size + y * cst.kernel_x + x];
                auto in4 = z_in[z * cst.input_size  + y * dilation_h   + x * cst.dilation_x];
                result += float4(float4(in4) * float4x4(wt4));
            }
        }
    }
    result = result * alpha[(short)gid.z] + float4(biasTerms[(short)gid.z]);
    *z_out = activate(ftype4(result), cst.activation);
}

kernel void conv_z4(const device ftype4 *in         [[buffer(0)]],
                    device ftype4 *out              [[buffer(1)]],
                    constant conv_constants& cst    [[buffer(2)]],
                    const device ftype4x4 *wt       [[buffer(3)]],
                    const device ftype4 *biasTerms  [[buffer(4)]],
                    uint3 gid                       [[thread_position_in_grid]]) {
    if ((int)gid.x >= cst.output_width || (int)gid.y >= cst.output_height) return;
    
    int4 uz = gid.z * CONV_UNROLL + int4(0, 1, 2, 3);
    bool3 valids = uz.yzw < cst.output_slice;
    
    int offset_x = (int)gid.x * cst.stride_x - cst.pad_x;
    int offset_y = (int)gid.y * cst.stride_y - cst.pad_y;
    int sx = max(0, (UP_DIV(-offset_x, cst.dilation_x)));
    int ex = min(cst.kernel_x, UP_DIV(cst.input_width - offset_x, cst.dilation_x));
    short kw = ex - sx;
    int sy = max(0, (UP_DIV(-offset_y, cst.dilation_y)));
    int ey = min(cst.kernel_y, UP_DIV(cst.input_height - offset_y, cst.dilation_y));
    short kh = ey - sy;
    offset_x += sx * cst.dilation_x;
    offset_y += sy * cst.dilation_y;
    
    auto z_in  = in                                              + offset_y * cst.input_width    + offset_x;
    auto z_wt  = wt  + uz[0] * cst.input_slice * cst.kernel_size + sy * cst.kernel_x             + sx;
    auto z_out = out + uz[0] * cst.output_size                   + (int)gid.y * cst.output_width + (int)gid.x;
    
    int ws = cst.input_slice * cst.kernel_size;
    int dilation_h = cst.input_width * cst.dilation_y;
    float4 result0 = 0, result1 = 0, result2 = 0, result3 = 0;
    for (auto z = 0; z < cst.input_slice; z++, z_wt += cst.kernel_size, z_in += cst.input_size) {
        for (auto y = 0; y < kh; y++) {
            for (auto x = 0; x < kw; x++) {
                auto x_wt = z_wt + y * cst.kernel_x + x;
                auto in4  = z_in[  y * dilation_h   + x * cst.dilation_x];
                /* true                   */ result0 += float4(in4 * *x_wt);
                if (valids[0]) { x_wt += ws; result1 += float4(in4 * *x_wt); }
                if (valids[1]) { x_wt += ws; result2 += float4(in4 * *x_wt); }
                if (valids[2]) { x_wt += ws; result3 += float4(in4 * *x_wt); }
            }
        }
    }
    /* true                                 */ *z_out = activate(ftype4(result0 + float4(biasTerms[uz[0]])), cst.activation);
    if (valids[0]) { z_out += cst.output_size; *z_out = activate(ftype4(result1 + float4(biasTerms[uz[1]])), cst.activation); }
    if (valids[1]) { z_out += cst.output_size; *z_out = activate(ftype4(result2 + float4(biasTerms[uz[2]])), cst.activation); }
    if (valids[2]) { z_out += cst.output_size; *z_out = activate(ftype4(result3 + float4(biasTerms[uz[3]])), cst.activation); }
}

kernel void qntconv_z4(const device char4 *in           [[buffer(0)]],
                       device ftype4 *out               [[buffer(1)]],
                       constant conv_constants& cst     [[buffer(2)]],
                       const device char4x4 *wt         [[buffer(3)]],
                       const device ftype4 *biasTerms   [[buffer(4)]],
                       const device float4 *alpha       [[buffer(5)]],
                       ushort3 gid                      [[thread_position_in_grid]]) {
    if ((int)gid.x >= cst.output_width || (int)gid.y >= cst.output_height) return;
    
    short4 uz = gid.z * CONV_UNROLL + short4(0, 1, 2, 3);
    bool3 valids = uz.yzw < cst.output_slice;
    
    short offset_x = (int)gid.x * cst.stride_x - cst.pad_x;
    short offset_y = (int)gid.y * cst.stride_y - cst.pad_y;
    short sx = max(0, (UP_DIV(-offset_x, cst.dilation_x)));
    short ex = min(cst.kernel_x, UP_DIV(cst.input_width - offset_x, cst.dilation_x));
    short kw = ex - sx;
    short sy = max(0, (UP_DIV(-offset_y, cst.dilation_y)));
    short ey = min(cst.kernel_y, UP_DIV(cst.input_height - offset_y, cst.dilation_y));
    short kh = ey - sy;
    offset_y += sy * cst.dilation_y;
    offset_x += sx * cst.dilation_x;
    
    auto z_in  = in                                              + offset_y * cst.input_width      + offset_x;
    auto z_wt  = wt  + uz[0] * cst.input_slice * cst.kernel_size + sy * cst.kernel_x               + sx;
    auto z_out = out + uz[0] * cst.output_size                   + (int)gid.y * cst.output_width + (short)gid.x;
    
    int ws = cst.input_slice * cst.kernel_size;
    int dilation_h = cst.input_width * cst.dilation_y;
    float4 result0 = 0, result1 = 0, result2 = 0, result3 = 0;
    for (auto z = 0; z < cst.input_slice; z++, z_wt += cst.kernel_size, z_in += cst.input_size) {
        for (auto y = 0; y < kh; y++) {
            for (auto x = 0; x < kw; x++) {
                auto x_wt = z_wt + y * cst.kernel_x + x;
                auto x_in = z_in + y * dilation_h   + x * cst.dilation_x;
                auto in4 = float4(*x_in);
                /* true                   */ result0 += in4 * float4x4(*x_wt);
                if (valids[0]) { x_wt += ws; result1 += in4 * float4x4(*x_wt); }
                if (valids[1]) { x_wt += ws; result2 += in4 * float4x4(*x_wt); }
                if (valids[2]) { x_wt += ws; result3 += in4 * float4x4(*x_wt); }
            }
        }
    }
    /* true                                 */ *z_out = activate(ftype4(result0 * alpha[uz[0]] + float4(biasTerms[uz[0]])), cst.activation);
    if (valids[0]) { z_out += cst.output_size; *z_out = activate(ftype4(result1 * alpha[uz[1]] + float4(biasTerms[uz[1]])), cst.activation); }
    if (valids[1]) { z_out += cst.output_size; *z_out = activate(ftype4(result2 * alpha[uz[2]] + float4(biasTerms[uz[2]])), cst.activation); }
    if (valids[2]) { z_out += cst.output_size; *z_out = activate(ftype4(result3 * alpha[uz[3]] + float4(biasTerms[uz[3]])), cst.activation); }
}

kernel void conv_local(const device ftype4 *in          [[buffer(0)]],
                       device ftype4 *out               [[buffer(1)]],
                       constant conv_constants& cst     [[buffer(2)]],
                       const device ftype4x4 *wt        [[buffer(3)]],
                       const device ftype4 *biasTerms   [[buffer(4)]],
                       threadgroup ftype4x4 *cols       [[threadgroup(0)]],
                       ushort3 gid                      [[thread_position_in_grid]],
                       ushort3 tid                      [[thread_position_in_threadgroup]],
                       ushort3 thread_size              [[threads_per_threadgroup]]) {
    short unroll_x = CONV_UNROLL * gid.x;
    short offset_x = unroll_x * cst.stride_x - cst.pad_x;
    short offset_y = gid.y * cst.stride_y - cst.pad_y;
    short sy = max(0, UP_DIV(-offset_y, cst.dilation_y));
    short ey = min(cst.kernel_y, UP_DIV(cst.input_height - offset_y, cst.dilation_y));
    auto o_wt = wt + (int)gid.z * cst.input_slice * cst.kernel_size;

    float4x4 result = float4x4(0);
    short steps = UP_DIV(cst.input_slice, cst.threadgroup_input_slice);
    for (auto s = 0; s < steps; s++)
    {
        int sz_stt = s * cst.threadgroup_input_slice;
        int sz_end = min(sz_stt + cst.threadgroup_input_slice, cst.input_slice);
        int sz_size = sz_end - sz_stt;
        
        // im2col
        int z_step = UP_DIV(sz_size, (int)thread_size.z);
        int z_stt = tid.z * z_step;
        int z_end = min(z_stt + z_step, sz_size);
        
        for (auto z = z_stt; z < z_end; z++) {
            for (auto ky = sy; ky < ey; ky++) {
                for (auto kx = 0; kx < cst.kernel_x; kx++) {
                    auto y_in = in
                        + (z + sz_stt) * cst.input_size
                        + (offset_y + ky * cst.dilation_y) * cst.input_width;
                    int4 x4 = offset_x + kx * cst.dilation_x + cst.stride_x * int4(0, 1, 2, 3);
                    bool4 valids = 0 <= x4 && x4 < cst.input_width;
                    cols[z * cst.kernel_size + ky * cst.kernel_x + kx] = {
                        valids[0] ? y_in[x4[0]] : 0,
                        valids[1] ? y_in[x4[1]] : 0,
                        valids[2] ? y_in[x4[2]] : 0,
                        valids[3] ? y_in[x4[3]] : 0
                    };
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // gemm
        if ((short)gid.z < cst.output_slice) {
            for (auto z = 0; z < sz_size; z++) {
                for (auto ky = sy; ky < ey; ky++) {
                    for (auto kx = 0; kx < cst.kernel_x; kx++) {
                        auto in4 = cols[ z           * cst.kernel_size + ky * cst.kernel_x + kx];
                        auto wt4 = o_wt[(z + sz_stt) * cst.kernel_size + ky * cst.kernel_x + kx];
                        result += {
                            float4(in4[0] * wt4),
                            float4(in4[1] * wt4),
                            float4(in4[2] * wt4),
                            float4(in4[3] * wt4)
                        };
                    }
                }
            }
        }
        
        if (s == steps - 1) break;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    } // end step
    
    // save
    if ((short)gid.z >= cst.output_slice) return;

    float4 b4 = float4(biasTerms[(short)gid.z]);
    auto off_out = out + (int)gid.z * cst.output_size + (int)gid.y * cst.output_width + unroll_x;
    bool3 valids = (unroll_x + int3(1, 2, 3)) < cst.output_width;
    /* true */     off_out[0] = activate((ftype4)(result[0] + b4), cst.activation);
    if (valids[0]) off_out[1] = activate((ftype4)(result[1] + b4), cst.activation);
    if (valids[1]) off_out[2] = activate((ftype4)(result[2] + b4), cst.activation);
    if (valids[2]) off_out[3] = activate((ftype4)(result[3] + b4), cst.activation);
}

kernel void qntconv_local(const device char4 *in            [[buffer(0)]],
                          device ftype4 *out                [[buffer(1)]],
                          constant conv_constants& cst      [[buffer(2)]],
                          const device char4x4 *wt          [[buffer(3)]],
                          const device ftype4 *biasTerms    [[buffer(4)]],
                          const device float4 *alpha        [[buffer(5)]],
                          threadgroup char4x4 *cols         [[threadgroup(0)]],
                          ushort3 gid                       [[thread_position_in_grid]],
                          ushort3 tid                       [[thread_position_in_threadgroup]],
                          ushort3 thread_size               [[threads_per_threadgroup]]) {
    short unroll_x = CONV_UNROLL * gid.x;
    short offset_x = unroll_x * cst.stride_x - cst.pad_x;
    short offset_y = gid.y * cst.stride_y - cst.pad_y;
    short sy = max(0, UP_DIV(-offset_y, cst.dilation_y));
    short ey = min(cst.kernel_y, UP_DIV(cst.input_height - offset_y, cst.dilation_y));
    auto o_wt = wt + (int)gid.z * cst.input_slice * cst.kernel_size;
    
    float4x4 result = {0, 0, 0, 0};
    short steps = UP_DIV(cst.input_slice, cst.threadgroup_input_slice);
    for (auto s = 0; s < steps; s++)
    {
        int sz_stt = s * cst.threadgroup_input_slice;
        int sz_end = min(sz_stt + cst.threadgroup_input_slice, cst.input_slice);
        int sz_size = sz_end - sz_stt;
        
        // im2col
        int z_step = UP_DIV(sz_size, (int)thread_size.z);
        int z_stt = tid.z * z_step;
        int z_end = min(z_stt + z_step, sz_size);
        
        for (auto z = z_stt; z < z_end; z++) {
            for (auto ky = sy; ky < ey; ky++) {
                for (auto kx = 0; kx < cst.kernel_x; kx++) {
                    auto y_in = in
                        + (z + sz_stt) * cst.input_size
                        + (offset_y + ky * cst.dilation_y) * cst.input_width;
                    int4 x4 = offset_x + kx * cst.dilation_x + cst.stride_x * int4(0, 1, 2, 3);
                    bool4 valids = 0 <= x4 && x4 < cst.input_width;
                    cols[z * cst.kernel_size + ky * cst.kernel_x + kx] = {
                        valids[0] ? y_in[x4[0]] : 0,
                        valids[1] ? y_in[x4[1]] : 0,
                        valids[2] ? y_in[x4[2]] : 0,
                        valids[3] ? y_in[x4[3]] : 0
                    };
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // gemm
        if ((short)gid.z < cst.output_slice) {
            for (auto z = 0; z < sz_size; z++) {
                for (auto ky = sy; ky < ey; ky++) {
                    for (auto kx = 0; kx < cst.kernel_x; kx++) {
                        auto in4 = float4x4(cols[ z           * cst.kernel_size + ky * cst.kernel_x + kx]);
                        auto wt4 = float4x4(o_wt[(z + sz_stt) * cst.kernel_size + ky * cst.kernel_x + kx]);
                        result += float4x4(in4[0] * wt4,
                                           in4[1] * wt4,
                                           in4[2] * wt4,
                                           in4[3] * wt4);
                    }
                }
            }
        }
        
        if (s == steps - 1) break;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    } // end step
    
    // save
    if ((short)gid.z >= cst.output_slice) return;
    
    auto off_out = out + (int)gid.z * cst.output_size + (int)gid.y * cst.output_width + unroll_x;
    float4 a4 = alpha[(short)gid.z], b4 = float4(biasTerms[(short)gid.z]);
    bool3 valids = (unroll_x + int3(1, 2, 3)) < cst.output_width;
    /* true */     off_out[0] = activate(ftype4(result[0] * a4 + b4), cst.activation);
    if (valids[0]) off_out[1] = activate(ftype4(result[1] * a4 + b4), cst.activation);
    if (valids[1]) off_out[2] = activate(ftype4(result[2] * a4 + b4), cst.activation);
    if (valids[2]) off_out[3] = activate(ftype4(result[3] * a4 + b4), cst.activation);
}
