//
//  MetalLSTM.metal
//  MNN
//
//  Created by MNN on 2018/08/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <metal_stdlib>
#include "MetalDefine.metal"

using namespace metal;

struct lstm_shape {
    int output_width;
    int input_width;
    int channel;
    int slice;
};

kernel void lstm_gate(const device ftype4 *in       [[buffer(0)]], // iw * z
                      const device ftype4 *weight   [[buffer(1)]], // iw * ow * 4
                      device ftype4 *gates          [[buffer(2)]], // z * ow * 4
                      constant lstm_shape& shape    [[buffer(3)]],
                      ushort2 gid                   [[thread_position_in_grid]]) {
    const short output_width = shape.output_width;
    const short input_width  = shape.input_width;
    const short ow = gid.x, z = gid.y;
    if (ow >= output_width || z >= shape.slice) return;
    
    auto z_in = in + z * input_width;
    auto z_wt = weight + ow * input_width;

    ftype4 I = 0, F = 0, O = 0, G = 0;
    for (short i = 0; i < input_width; i++) {
        auto in4 = z_in[i];
        auto wt4 = z_wt[i];
        I += in4[0] * wt4;
        F += in4[1] * wt4;
        O += in4[2] * wt4;
        G += in4[3] * wt4;
    }
    auto z_gates = gates + z * 4 * output_width + ow;
    *z_gates = I; z_gates += output_width;
    *z_gates = F; z_gates += output_width;
    *z_gates = O; z_gates += output_width;
    *z_gates = G;
}

static inline float4 sigmoid(float4 IFOG) {
    return float4(1.f / (1.f + exp(-IFOG.xyz)), tanh(IFOG.w));
}

kernel void lstm(const device ftype4 *gates     [[buffer(0)]], // ow * 4 * c4
                 const device ftype4 *weight    [[buffer(1)]], // ow * ow * 4
                 const device ftype4 *bias4     [[buffer(2)]], // ow * 4
                 constant lstm_shape& shape     [[buffer(3)]],
                 device ftype *out              [[buffer(4)]], // ow * c
                 threadgroup ftype *hidden      [[threadgroup(0)]],
                 ushort gid                     [[thread_position_in_grid]]) {
    const short output_width = shape.output_width;
    const short ow           = gid;
    if (ow >= output_width) return;
    
    auto o_weight = weight + ow * output_width;
    auto o_gates  = gates + ow;
    auto o_bias   = float4(bias4[ow]);
    auto o_out    = out + ow * 4;
    auto cell     = 0.h;
    
    for (short c = 0; c < shape.channel; c++, o_gates += output_width) {
        auto IFOG = float4(*o_gates);
        bool cont = c > 0;
        if (cont) {
            for (short i = 0; i < output_width; i++) {
                IFOG += float4(hidden[i] * o_weight[i]);
            }
        }
        IFOG = sigmoid(IFOG + o_bias);
        cell = (cont ? IFOG.y : 0.h) * cell + IFOG.x * IFOG.w;
        
        float H = IFOG.z * tanh(cell);
        o_out[c / 4 * output_width * 4 + c % 4] = H; // H
        hidden[ow] = H;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void lstm_cont(const device ftype4 *gates    [[buffer(0)]], // ow * 4 * c
                      const device ftype4 *weight   [[buffer(1)]], // ow * ow * 4
                      const device ftype4 *bias4    [[buffer(2)]], // ow * 4
                      constant lstm_shape& shape    [[buffer(3)]],
                      device ftype *out             [[buffer(4)]], // ow * c
                      const device ftype *conts     [[buffer(5)]], // c
                      threadgroup ftype *hidden     [[threadgroup(0)]],
                      ushort gid                    [[thread_position_in_grid]]) {
    const short output_width   = shape.output_width;
    const short ow             = gid;
    if (ow >= output_width) return;
    
    auto o_weight = weight + ow * output_width;
    auto o_gates  = gates + ow;
    auto o_bias   = float4(bias4[ow]);
    auto o_out    = out + ow * 4;
    auto cell     = 0.h;
    
    for (short c = 0; c < shape.channel; c++, o_gates += output_width) {
        auto IFOG = float4(*o_gates);
        bool cont = c > 0 && conts[c] > 0;
        if (cont) {
            for (short i = 0; i < output_width; i++) {
                IFOG += float4(hidden[i] * o_weight[i]);
            }
        }
        IFOG = sigmoid(IFOG + o_bias);
        cell = (cont ? IFOG.y : 0.h) * cell + IFOG.x * IFOG.w;
        
        float H = IFOG.z * tanh(cell);
        o_out[c / 4 * output_width * 4 + c % 4] = H; // H
        hidden[ow] = H;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
