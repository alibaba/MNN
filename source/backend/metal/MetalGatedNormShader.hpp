//
//  MetalGatedNormShader.hpp
//  MNN
//
//  Created by MNN on 2026/08/03.
//  Copyright 2018, Alibaba Group Holding Limited
//

#if MNN_METAL_ENABLED
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

// Fused replacement for a linear-attention layer's output gating segment:
//
//   LinearAttention -> Raster -> Cast -> RMSNorm ---\
//                                                    MUL -> Raster -> out_proj
//   in_proj_z ----------------> Raster -> SILU -----/
//
// `la` is the LinearAttention output, NC4HW4 [outside, inside] with the head as
// the batch axis, so its float4 index is `c * outside + h` — the same addressing
// layernorm_c4_rms_sg uses. `z` and `out` are NC4HW4 [1, outside*inside] with
// batch 1, hence contiguous: float4 index `h * CU + c`. Reading z and writing out
// at that index reproduces both Raster repacks, which are exact inverses.
//
// Arithmetic mirrors the ops it replaces: the RMS reduction follows
// layernorm_c4_rms_sg (fp32 float4 accumulation, simd_sum on the float4 before
// the horizontal add, and `1.0f / sqrt(...)` rather than rsqrt), SILU is fp32
// with MNNEXP's +-87 clamp, and the final multiply happens in ftype like
// MetalBinary's half plane pipeline. In fp32 builds the result is bit-identical
// to the unfused chain; under fp16 storage the RMSNorm half rounds differently
// (see skills/metal-optimize/kernel-dev-and-optimize.md §2.4.4).
static const char* gLinearAttnGatedNorm = R"metal(
#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;

#define SIMD_GROUP_WIDTH 32

#ifndef SGS_PER_TG
#define SGS_PER_TG 1
#endif

struct gated_norm_constants {
    int inside;
    int outside;
    float eps;
    int has_gamma_beta;
};

kernel void linear_attn_gated_norm(const device ftype4 *la     [[buffer(0)]],
                                   const device ftype4 *z      [[buffer(1)]],
                                   device ftype4 *out          [[buffer(2)]],
                                   constant gated_norm_constants& cst [[buffer(3)]],
                                   const device float4 *gamma  [[buffer(4)]],
                                   const device float4 *beta   [[buffer(5)]],
                                   uint3 gid   [[threadgroup_position_in_grid]],
                                   uint  tiisg [[thread_index_in_simdgroup]],
                                   uint  sgitg [[simdgroup_index_in_threadgroup]]) {
    int batch = cst.outside;
    int channelUnit = cst.inside / 4;
    int h = (int)gid.y * SGS_PER_TG + (int)sgitg;
    if (h >= batch) {
        return;
    }

    float4 square_sum4 = 0.0f;
    for (int c = tiisg; c < channelUnit; c += SIMD_GROUP_WIDTH) {
        float4 data = float4(la[c * batch + h]);
        square_sum4 += data * data;
    }
    square_sum4 = simd_sum(square_sum4);
    float square_sum = square_sum4[0] + square_sum4[1] + square_sum4[2] + square_sum4[3];
    float var = 1.0f / sqrt(square_sum / (channelUnit * 4) + cst.eps);
    float4 var4 = var;

    for (int c = tiisg; c < channelUnit; c += SIMD_GROUP_WIDTH) {
        int flat = h * channelUnit + c;
        float4 norm = var4 * float4(la[c * batch + h]);
        ftype4 normed;
        if (cst.has_gamma_beta) {
            normed = (ftype4)(norm * gamma[c] + beta[c]);
        } else {
            normed = (ftype4)(norm);
        }
        float4 v = float4(z[flat]);
        ftype4 gate = (ftype4)(v / (1.0f + exp(clamp(-v, (float4)(-87.0f), (float4)(87.0f)))));
        out[flat] = normed * gate;
    }
}
)metal";

#endif /* MNN_SUPPORT_TRANSFORMER_FUSE */
#endif /* MNN_METAL_ENABLED */
