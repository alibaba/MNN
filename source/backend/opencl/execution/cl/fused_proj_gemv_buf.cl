#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

// Decode-time GEMV for a whole FusedLinear group: 2-4 conv1x1 projections that
// share one input activation, computed in a single dispatch.
//
// Two flavours, both int4 weights (QUANT_BIT 4), no channel leaves:
//   FUSE_SILU_MUL: NUM_CONV == 2, gate/up. One workgroup owns the same output
//                  tile of both convs, so the MUL_SILU epilogue happens in
//                  registers and neither projection is ever written to DRAM.
//   otherwise:     NUM_CONV in 2..4 (q/k/v[/w]) writing their own outputs. The
//                  workgroup's tile index selects the member, so the group is
//                  one dispatch instead of one per member.
//
// Mirrors gemv_conv_c8_buf (gemv_conv1x1_buf.cl): 8 output channels per
// workgroup, K split across WGS work-items, tree reduce in local memory. The
// per-member scale/bias/coef stay separate because the dequant scale buffer is
// quant-block-major, so members cannot share one buffer.

__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

#define GLOBAL_SIZE_DIM_2 __private int global_size_dim0, __private int global_size_dim1,

#define UCHAR4_TO_CHAR8(b, scale, offset) \
    wei.s0 = CONVERT_FLOAT((b.s0 >> 4));  \
    wei.s1 = CONVERT_FLOAT((b.s0 & 15));  \
    wei.s2 = CONVERT_FLOAT((b.s1 >> 4));  \
    wei.s3 = CONVERT_FLOAT((b.s1 & 15));  \
    wei.s4 = CONVERT_FLOAT((b.s2 >> 4));  \
    wei.s5 = CONVERT_FLOAT((b.s2 & 15));  \
    wei.s6 = CONVERT_FLOAT((b.s3 >> 4));  \
    wei.s7 = CONVERT_FLOAT((b.s3 & 15));  \
    wei = wei * scale + offset;

#ifdef USE_IMAGE
#define FP_WEIGHT_ARG(name) __read_only image2d_t name
#define FP_LOAD_WEIGHT(w, j, tile) as_uchar16(read_imagei(w, SAMPLER, (int2)((j), (tile))))
#else
#define FP_WEIGHT_ARG(name) __global const uchar* name
#define FP_LOAD_WEIGHT(w, j, tile) vload16((j), w + (tile) * srcChannelC4 * 16)
#endif

// Dequant scale / offset of the 8 output channels at `oc8` for the quant block
// holding input channel k4. Layout is block-major: (block * dstC4 * 4) + oc.
#ifdef ASYMMETRIC
#define FP_LOAD_SCALE(so, dstC4, cf, oc8, k4, scale, offset)                                       \
    {                                                                                              \
        COMPUTE_FLOAT16 scaleOffset = CONVERT_COMPUTE_FLOAT16(                                     \
            convert_float16(vload16(0, so + (oc8) * 2 + ((k4) / blockDim) * (dstC4) * 8)) / (cf)); \
        scale = scaleOffset.s02468ace;                                                             \
        offset = scaleOffset.s13579bdf;                                                            \
    }
#else
#define FP_LOAD_SCALE(so, dstC4, cf, oc8, k4, scale, offset)                                                        \
    {                                                                                                               \
        scale =                                                                                                     \
            CONVERT_COMPUTE_FLOAT8(convert_float8(vload8(0, so + (oc8) + ((k4) / blockDim) * (dstC4) * 4)) / (cf)); \
        offset = (COMPUTE_FLOAT8)(-8) * scale;                                                                      \
    }
#endif

// This work-item's slice of the dot products for output tile `tile` of one member.
#define FP_ACCUM(w, so, dstC4, cf, tile, acc)                                  \
    {                                                                          \
        const int oc8_ = (tile) << 3;                                          \
        for (int j = lid; j < loop; j += WGS) {                                \
            const int k4 = j << 2;                                             \
            COMPUTE_FLOAT8 scale, offset;                                      \
            FP_LOAD_SCALE(so, dstC4, cf, oc8_, k4, scale, offset)              \
            COMPUTE_FLOAT8 wei;                                                \
            COMPUTE_FLOAT4 in = CONVERT_COMPUTE_FLOAT4(vload4(0, input + k4)); \
            uchar16 wq = FP_LOAD_WEIGHT(w, j, tile);                           \
            UCHAR4_TO_CHAR8(wq.s0123, scale, offset)                           \
            acc = mad((COMPUTE_FLOAT8)in.s0, wei, acc);                        \
            UCHAR4_TO_CHAR8(wq.s4567, scale, offset)                           \
            acc = mad((COMPUTE_FLOAT8)in.s1, wei, acc);                        \
            UCHAR4_TO_CHAR8(wq.s89ab, scale, offset)                           \
            acc = mad((COMPUTE_FLOAT8)in.s2, wei, acc);                        \
            UCHAR4_TO_CHAR8(wq.scdef, scale, offset)                           \
            acc = mad((COMPUTE_FLOAT8)in.s3, wei, acc);                        \
        }                                                                      \
    }

// Tree reduce of the workgroup's partial sums. Every work-item leaves with the
// total in `res`, and the trailing barrier lets the scratch be reused.
#define FP_REDUCE(acc, res)                            \
    {                                                  \
        sum0[lid] = acc;                               \
        barrier(CLK_LOCAL_MEM_FENCE);                  \
        for (int i = WGS / 2; i > 0; i /= 2) {         \
            if (lid < i) {                             \
                sum0[lid] = sum0[lid] + sum0[lid + i]; \
            }                                          \
            barrier(CLK_LOCAL_MEM_FENCE);              \
        }                                              \
        res = sum0[0];                                 \
        barrier(CLK_LOCAL_MEM_FENCE);                  \
    }

#if WGS >= 8
__kernel void fused_proj_gemv_buf(GLOBAL_SIZE_DIM_2 __global const FLOAT* input, FP_WEIGHT_ARG(weight0),
                                  FP_WEIGHT_ARG(weight1),
#if NUM_CONV > 2
                                  FP_WEIGHT_ARG(weight2),
#endif
#if NUM_CONV > 3
                                  FP_WEIGHT_ARG(weight3),
#endif
                                  __global const FLOAT* scaleOffset0, __global const FLOAT* scaleOffset1,
#if NUM_CONV > 2
                                  __global const FLOAT* scaleOffset2,
#endif
#if NUM_CONV > 3
                                  __global const FLOAT* scaleOffset3,
#endif
                                  __global const FLOAT* bias0, __global const FLOAT* bias1,
#if NUM_CONV > 2
                                  __global const FLOAT* bias2,
#endif
#if NUM_CONV > 3
                                  __global const FLOAT* bias3,
#endif
                                  __global FLOAT* output0,
#ifndef FUSE_SILU_MUL
                                  __global FLOAT* output1,
#if NUM_CONV > 2
                                  __global FLOAT* output2,
#endif
#if NUM_CONV > 3
                                  __global FLOAT* output3,
#endif
#endif
                                  __private const int srcChannelC4, __private const int blockDim,
                                  __private const int4 dstChannelC4, __private const int4 ocTiles,
                                  __private const float4 coef) {
    const int lid = get_local_id(0);
    const int gid = get_global_id(1);
    const int loop = srcChannelC4;
    __local COMPUTE_FLOAT8 sum0[WGS];

#ifdef FUSE_SILU_MUL
    // gate = convs[0](x), up = convs[1](x), out = up * silu(gate) — the two
    // members share this tile, so both stay in registers.
    COMPUTE_FLOAT8 accGate = 0, accUp = 0;
    FP_ACCUM(weight0, scaleOffset0, dstChannelC4.x, coef.x, gid, accGate)
    FP_ACCUM(weight1, scaleOffset1, dstChannelC4.y, coef.y, gid, accUp)
    COMPUTE_FLOAT8 resGate, resUp;
    FP_REDUCE(accGate, resGate)
    FP_REDUCE(accUp, resUp)
    if (lid == 0) {
        const int oc8 = gid << 3;
        float8 gate = convert_float8(resGate + CONVERT_COMPUTE_FLOAT8(vload8(0, bias0 + oc8)));
        float8 up = convert_float8(resUp + CONVERT_COMPUTE_FLOAT8(vload8(0, bias1 + oc8)));
        // Same expression as BinaryBufExecution's MUL_SILU, so the fused and the
        // unfused graph agree bit for bit.
        float8 out = up * (gate * native_recip((float8)1 + native_exp(-gate)));
        vstore8(CONVERT_FLOAT8(out), 0, output0 + oc8);
    }
#else
    // One member per tile: peel the member index off the flat tile id. It is
    // workgroup-uniform, so the branches below never diverge.
    int tile = gid;
    int ci = 0;
    if (tile >= ocTiles.x) {
        tile -= ocTiles.x;
        ci = 1;
#if NUM_CONV > 2
        if (tile >= ocTiles.y) {
            tile -= ocTiles.y;
            ci = 2;
#if NUM_CONV > 3
            if (tile >= ocTiles.z) {
                tile -= ocTiles.z;
                ci = 3;
            }
#endif
        }
#endif
    }
    COMPUTE_FLOAT8 acc = 0, res;
    if (ci == 0) {
        FP_ACCUM(weight0, scaleOffset0, dstChannelC4.x, coef.x, tile, acc)
    } else if (ci == 1) {
        FP_ACCUM(weight1, scaleOffset1, dstChannelC4.y, coef.y, tile, acc)
    }
#if NUM_CONV > 2
    else if (ci == 2) {
        FP_ACCUM(weight2, scaleOffset2, dstChannelC4.z, coef.z, tile, acc)
    }
#endif
#if NUM_CONV > 3
    else {
        FP_ACCUM(weight3, scaleOffset3, dstChannelC4.w, coef.w, tile, acc)
    }
#endif
    FP_REDUCE(acc, res)
    if (lid == 0) {
        const int oc8 = tile << 3;
        if (ci == 0) {
            vstore8(CONVERT_FLOAT8(res + CONVERT_COMPUTE_FLOAT8(vload8(0, bias0 + oc8))), 0, output0 + oc8);
        } else if (ci == 1) {
            vstore8(CONVERT_FLOAT8(res + CONVERT_COMPUTE_FLOAT8(vload8(0, bias1 + oc8))), 0, output1 + oc8);
        }
#if NUM_CONV > 2
        else if (ci == 2) {
            vstore8(CONVERT_FLOAT8(res + CONVERT_COMPUTE_FLOAT8(vload8(0, bias2 + oc8))), 0, output2 + oc8);
        }
#endif
#if NUM_CONV > 3
        else {
            vstore8(CONVERT_FLOAT8(res + CONVERT_COMPUTE_FLOAT8(vload8(0, bias3 + oc8))), 0, output3 + oc8);
        }
#endif
    }
#endif
}
#endif
