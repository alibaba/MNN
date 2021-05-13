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

#define CONV_MUL_PACK_W2(x,y)  \
    x += float4(in00 * k00);\
    y += float4(in01 * k00);\
    x += float4(in01 * k01);\
    y += float4(in02 * k01);\
    x += float4(in02 * k02);\
    y += float4(in03 * k02);\
                            \
    x += float4(in10 * k10);\
    y += float4(in11 * k10);\
    x += float4(in11 * k11);\
    y += float4(in12 * k11);\
    x += float4(in12 * k12);\
    y += float4(in13 * k12);\
                            \
    x += float4(in20 * k20);\
    y += float4(in21 * k20);\
    x += float4(in21 * k21);\
    y += float4(in22 * k21);\
    x += float4(in22 * k22);\
    y += float4(in23 * k22);
                     

#define CONV_NEXT_FLT  \
    z_wt += ws;             \
                            \
    k00 = z_wt[0], k01 = z_wt[1], k02 = z_wt[2];\
    k10 = z_wt[3], k11 = z_wt[4], k12 = z_wt[5];\
    k20 = z_wt[6], k21 = z_wt[7], k22 = z_wt[8];


#define CONV_MUL_PACK_H2(x,y)  \
    x += float4(in10 * k00);\
    y += float4(in11 * k00);\
    x += float4(in11 * k01);\
    y += float4(in12 * k01);\
    x += float4(in12 * k02);\
    y += float4(in13 * k02);\
                            \
    x += float4(in20 * k10);\
    y += float4(in21 * k10);\
    x += float4(in21 * k11);\
    y += float4(in22 * k11);\
    x += float4(in22 * k12);\
    y += float4(in23 * k12);\
                            \
    x += float4(in30 * k20);\
    y += float4(in31 * k20);\
    x += float4(in31 * k21);\
    y += float4(in32 * k21);\
    x += float4(in32 * k22);\
    y += float4(in33 * k22);

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
    int kw = ex - sx;
    int sy = max(0, (UP_DIV(-offset_y, cst.dilation_y)));
    int ey = min(cst.kernel_y, UP_DIV(cst.input_height - offset_y, cst.dilation_y));
    int kh = ey - sy;
    offset_x += sx * cst.dilation_x;
    offset_y += sy * cst.dilation_y;
    
    auto z_in  = in                                                   + offset_y * cst.input_width    + offset_x;
    auto z_wt  = wt  + (int)gid.z * cst.input_slice * cst.kernel_size + sy * cst.kernel_x             + sx;
    auto z_out = out + (int)gid.z * cst.output_size                   + (int)gid.y * cst.output_width + (int)gid.x;

    int dilation_h = cst.input_width * cst.dilation_y;
    float4 result = float4(biasTerms[(int)gid.z]);
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

kernel void convk3s1d1p1_w2z4(const device ftype4 *in         [[buffer(0)]],
                    device ftype4 *out              [[buffer(1)]],
                    constant conv_constants& cst    [[buffer(2)]],
                    const device ftype4x4 *wt       [[buffer(3)]],
                    const device ftype4 *biasTerms  [[buffer(4)]],
                    uint3 gid                       [[thread_position_in_grid]]) {
    if ((int)gid.x * 2 >= cst.output_width || (int)gid.y >= cst.output_height || (int)gid.z * CONV_UNROLL >= cst.output_slice) return;
    
    int4 uz = gid.z * CONV_UNROLL + int4(0, 1, 2, 3);
    bool3 valids = uz.yzw < cst.output_slice;
    
    int offset_x = (int)gid.x * 2 - cst.pad_x;
    int offset_y = (int)gid.y - cst.pad_y;

    auto z_in  = in + offset_y * cst.input_width + offset_x;
    auto z_flt  = wt  + uz[0] * cst.input_slice * cst.kernel_size;
    auto z_out = out + uz[0] * cst.output_size + (int)gid.y * cst.output_width + (int)gid.x * 2;
    
    int ws = cst.input_slice * cst.kernel_size;
    float4 result0 = 0, result1 = 0, result2 = 0, result3 = 0;
    float4 result4 = 0, result5 = 0, result6 = 0, result7 = 0;

    for (auto z = 0; z < cst.input_slice; z++, z_flt += cst.kernel_size, z_in += cst.input_size) {
        auto in00 = (offset_x<0                   || offset_y<0) ? (ftype4)0.f : z_in[0*cst.input_width+0];
        auto in01 = (offset_x+1>=cst.input_width  || offset_y<0) ? (ftype4)0.f : z_in[0*cst.input_width+1];
        auto in02 = (offset_x+2>=cst.input_width  || offset_y<0) ? (ftype4)0.f : z_in[0*cst.input_width+2];
        auto in03 = (offset_x+3>=cst.input_width  || offset_y<0) ? (ftype4)0.f : z_in[0*cst.input_width+3];

        auto in10 = (offset_x<0                   || offset_y+1>=cst.input_height) ? (ftype4)0.f : z_in[1*cst.input_width+0];
        auto in11 = (offset_x+1>=cst.input_width  || offset_y+1>=cst.input_height) ? (ftype4)0.f : z_in[1*cst.input_width+1];
        auto in12 = (offset_x+2>=cst.input_width  || offset_y+1>=cst.input_height) ? (ftype4)0.f : z_in[1*cst.input_width+2];
        auto in13 = (offset_x+3>=cst.input_width  || offset_y+1>=cst.input_height) ? (ftype4)0.f : z_in[1*cst.input_width+3];
        
        auto in20 = (offset_x<0                   || offset_y+2>=cst.input_height) ? (ftype4)0.f : z_in[2*cst.input_width+0];
        auto in21 = (offset_x+1>=cst.input_width  || offset_y+2>=cst.input_height) ? (ftype4)0.f : z_in[2*cst.input_width+1];
        auto in22 = (offset_x+2>=cst.input_width  || offset_y+2>=cst.input_height) ? (ftype4)0.f : z_in[2*cst.input_width+2];
        auto in23 = (offset_x+3>=cst.input_width  || offset_y+2>=cst.input_height) ? (ftype4)0.f : z_in[2*cst.input_width+3];
        
        auto z_wt = z_flt;
        auto k00 = z_wt[0], k01 = z_wt[1], k02 = z_wt[2];
        auto k10 = z_wt[3], k11 = z_wt[4], k12 = z_wt[5];
        auto k20 = z_wt[6], k21 = z_wt[7], k22 = z_wt[8];

        CONV_MUL_PACK_W2(result0,result4);
        CONV_NEXT_FLT;
        CONV_MUL_PACK_W2(result1,result5);
        CONV_NEXT_FLT;
        CONV_MUL_PACK_W2(result2,result6);
        CONV_NEXT_FLT;
        CONV_MUL_PACK_W2(result3,result7);
    }
    /* true */ *z_out = activate(ftype4(result0 + float4(biasTerms[uz[0]])), cst.activation);
    /* true */ *(z_out+1) = activate(ftype4(result4 + float4(biasTerms[uz[0]])), cst.activation);

    if (valids[0]) {
        z_out += cst.output_size;
        *z_out = activate(ftype4(result1 + float4(biasTerms[uz[1]])), cst.activation);
        *(z_out+1) = activate(ftype4(result5 + float4(biasTerms[uz[1]])), cst.activation);
    }
    if (valids[1]) {
        z_out += cst.output_size;
        *z_out = activate(ftype4(result2 + float4(biasTerms[uz[2]])), cst.activation);
        *(z_out+1) = activate(ftype4(result6 + float4(biasTerms[uz[2]])), cst.activation);
    }
    if (valids[2]) {
        z_out += cst.output_size;
        *z_out = activate(ftype4(result3 + float4(biasTerms[uz[3]])), cst.activation);
        *(z_out+1) = activate(ftype4(result7 + float4(biasTerms[uz[3]])), cst.activation);
    }
}


kernel void conv_z4(const device ftype4 *in         [[buffer(0)]],
                    device ftype4 *out              [[buffer(1)]],
                    constant conv_constants& cst    [[buffer(2)]],
                    const device ftype4x4 *wt       [[buffer(3)]],
                    const device ftype4 *biasTerms  [[buffer(4)]],
                    uint3 gid                       [[thread_position_in_grid]]) {
    if ((int)gid.x >= cst.output_width || (int)gid.y >= cst.output_height || (int)gid.z * CONV_UNROLL >= cst.output_slice) return;
    
    int4 uz = gid.z * CONV_UNROLL + int4(0, 1, 2, 3);
    bool3 valids = uz.yzw < cst.output_slice;
    
    int offset_x = (int)gid.x * cst.stride_x - cst.pad_x;
    int offset_y = (int)gid.y * cst.stride_y - cst.pad_y;
    int sx = max(0, (UP_DIV(-offset_x, cst.dilation_x)));
    int ex = min(cst.kernel_x, UP_DIV(cst.input_width - offset_x, cst.dilation_x));
    int kw = ex - sx;
    int sy = max(0, (UP_DIV(-offset_y, cst.dilation_y)));
    int ey = min(cst.kernel_y, UP_DIV(cst.input_height - offset_y, cst.dilation_y));
    int kh = ey - sy;
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

kernel void conv_local(const device ftype4 *in          [[buffer(0)]],
                       device ftype4 *out               [[buffer(1)]],
                       constant conv_constants& cst     [[buffer(2)]],
                       const device ftype4x4 *wt        [[buffer(3)]],
                       const device ftype4 *biasTerms   [[buffer(4)]],
                       threadgroup ftype4x4 *cols       [[threadgroup(0)]],
                       uint3 gid                      [[thread_position_in_grid]],
                       uint3 tid                      [[thread_position_in_threadgroup]],
                       uint3 thread_size              [[threads_per_threadgroup]]) {
    int unroll_x = CONV_UNROLL * gid.x;
    int offset_x = unroll_x * cst.stride_x - cst.pad_x;
    int offset_y = gid.y * cst.stride_y - cst.pad_y;
    int sy = max(0, UP_DIV(-offset_y, cst.dilation_y));
    int ey = min(cst.kernel_y, UP_DIV(cst.input_height - offset_y, cst.dilation_y));
    auto o_wt = wt + (int)gid.z * cst.input_slice * cst.kernel_size;

    float4x4 result = float4x4(0);
    int steps = UP_DIV(cst.input_slice, cst.threadgroup_input_slice);
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
        if ((int)gid.z < cst.output_slice) {
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
    if ((int)gid.z >= cst.output_slice) return;

    float4 b4 = float4(biasTerms[(int)gid.z]);
    auto off_out = out + (int)gid.z * cst.output_size + (int)gid.y * cst.output_width + unroll_x;
    bool3 valids = (unroll_x + int3(1, 2, 3)) < cst.output_width;
    /* true */     off_out[0] = activate((ftype4)(result[0] + b4), cst.activation);
    if (valids[0]) off_out[1] = activate((ftype4)(result[1] + b4), cst.activation);
    if (valids[1]) off_out[2] = activate((ftype4)(result[2] + b4), cst.activation);
    if (valids[2]) off_out[3] = activate((ftype4)(result[3] + b4), cst.activation);
}
