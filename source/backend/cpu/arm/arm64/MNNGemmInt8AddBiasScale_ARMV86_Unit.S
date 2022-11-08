//
//  MNNGemmInt8AddBiasScale_ARMV86_Unit.S
//  MNN
//
//  Created by MNN on 2022/09/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#if defined(__aarch64__)
#include "MNNAsmGlobal.h"

.text
.align 5

.macro SET_BIAS s, d0, d1, d2, d3
    mov \d0\().16b, \s\().16b
    mov \d1\().16b, \s\().16b
    mov \d2\().16b, \s\().16b
    mov \d3\().16b, \s\().16b
.endm
.macro Int32ToFloat z0, z1, z2, z3
    scvtf \z0\().4s, \z0\().4s
    scvtf \z1\().4s, \z1\().4s
    scvtf \z2\().4s, \z2\().4s
    scvtf \z3\().4s, \z3\().4s
.endm
.macro MUL_SCALE s, d0, d1, d2, d3
    fmul \d0\().4s, \d0\().4s, \s\().4s
    fmul \d1\().4s, \d1\().4s, \s\().4s
    fmul \d2\().4s, \d2\().4s, \s\().4s
    fmul \d3\().4s, \d3\().4s, \s\().4s
.endm
.macro FloatToInt32 z0, z1, z2, z3
    fcvtas \z0\().4s, \z0\().4s
    fcvtas \z1\().4s, \z1\().4s
    fcvtas \z2\().4s, \z2\().4s
    fcvtas \z3\().4s, \z3\().4s
.endm
.macro Int32ToInt16 s0, s1, s2, s3, d0, d1
    sqxtn \d0\().4h,  \s0\().4s
    sqxtn2 \d0\().8h, \s1\().4s
    sqxtn \d1\().4h,  \s2\().4s
    sqxtn2 \d1\().8h, \s3\().4s
.endm
.macro Int16ToInt8_ONE s0, s1, d0
    sqxtn \d0\().8b,   \s0\().8h
    sqxtn2 \d0\().16b, \s1\().8h
.endm
.macro Int16ToInt8 s0, s1, s2, s3, d0, d1
    Int16ToInt8_ONE \s0, \s1, \d0
    Int16ToInt8_ONE \s2, \s3, \d1
.endm

asm_function MNNGemmInt8AddBiasScale_ARMV86_Unit

//struct QuanPostTreatParameters {
//    const float* scale;
//    const int32_t* bias;
//    int32_t maxValue;
//    int32_t minValue;
//};

//void MNNGemmInt8AddBiasScale_ARMV86_Unit(int8_t* dst, const int8_t* src,
//    const int8_t* weight, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad,
// const QuanPostTreatParameters* parameters, size_t realDstCount);

//Auto: x0:dst, x1:src, x2:weight, x3:src_depth_quad, x4:dst_step
//x5:dst_depth_quad, x6: parameters, x7: realDstCount

//Load from x7: x8: scale, x9: bias, w12: maxValue, w13: minValue
ldr x8, [x6, #0]
ldr x9, [x6, #8]
ldr w10, [x6, #16]
ldr w14, [x6, #20]

stp d14, d15, [sp, #(-16 * 6)]!
stp d12, d13, [sp, #(16 * 1)]
stp d10, d11, [sp, #(16 * 2)]
stp d8,  d9,  [sp, #(16 * 3)]
stp x21, x22, [sp, #(16 * 4)]
stp x19, x20, [sp, #(16 * 5)]

mov x21, #4 // sizeof(int8_t) * UNIT
mov x22, #160 // GEMM_INT8_DST_XUNIT * GEMM_INT8_SRC_UNIT = 20 * 8 = 160
cbnz x8, Start
mov x21, #16 // sizeof(float) * UNIT

Start:
lsl x15, x3, #5 // x15 = src_depth_quad * UNIT * UNIT_SRC = src_depth_quad * 32 = src_depth_quad << 5

TILE_20:
    cmp x7, #20
    blt TILE_16
LoopDz_TILE_20:
    ld1 {v0.4s}, [x9], #16  // bias
    mov x11, x1 // src
    mov x12, x2 // weight
    mov x13, x3 // src_depth_quad
    mov v1.16b, v0.16b
    uzp1 v12.2d, v0.2d, v1.2d   // bias_0, bias_1, bias_0, bias_1
    uzp2 v13.2d, v0.2d, v1.2d   // bias_2, bias_3, bias_2, bias_3
    mov v14.16b, v12.16b
    mov v15.16b, v13.16b
    SET_BIAS v14, v16, v18, v20, v22
    SET_BIAS v14, v24, v26, v28, v30
    SET_BIAS v15, v17, v19, v21, v23
    SET_BIAS v15, v25, v27, v29, v31
LoopSz_TILE_20:
    // src    : 10 x [2 x 8] : v2-11
    // weight :  2 x [2 x 8] : v0-1
    // dst    : 10 x 2 x [4] : v12-v31
    ld1 {v0.16b, v1.16b}, [x12], #32                    // weight
    ld1 {v2.16b, v3.16b, v4.16b, v5.16b}, [x11], #64    // src
    .inst 0x4e80a44c // smmla v12.4s, v2.16b, v0.16b
    .inst 0x4e81a44d // smmla v13.4s, v2.16b, v1.16b
    .inst 0x4e80a46e // smmla v14.4s, v3.16b, v0.16b
    .inst 0x4e81a46f // smmla v15.4s, v3.16b, v1.16b
    ld1 {v6.16b, v7.16b, v8.16b, v9.16b}, [x11], #64
    .inst 0x4e80a490 // smmla v16.4s, v4.16b, v0.16b
    .inst 0x4e81a491 // smmla v17.4s, v4.16b, v1.16b
    .inst 0x4e80a4b2 // smmla v18.4s, v5.16b, v0.16b
    .inst 0x4e81a4b3 // smmla v19.4s, v5.16b, v1.16b
    ld1 {v10.16b, v11.16b}, [x11], #32
    .inst 0x4e80a4d4 // smmla v20.4s, v6.16b, v0.16b
    .inst 0x4e81a4d5 // smmla v21.4s, v6.16b, v1.16b
    .inst 0x4e80a4f6 // smmla v22.4s, v7.16b, v0.16b
    .inst 0x4e81a4f7 // smmla v23.4s, v7.16b, v1.16b
    .inst 0x4e80a518 // smmla v24.4s, v8.16b, v0.16b
    .inst 0x4e81a519 // smmla v25.4s, v8.16b, v1.16b
    .inst 0x4e80a53a // smmla v26.4s, v9.16b, v0.16b
    .inst 0x4e81a53b // smmla v27.4s, v9.16b, v1.16b
    .inst 0x4e80a55c // smmla v28.4s, v10.16b, v0.16b
    .inst 0x4e81a55d // smmla v29.4s, v10.16b, v1.16b
    subs x13, x13, #1
    .inst 0x4e80a57e // smmla v30.4s, v11.16b, v0.16b
    .inst 0x4e81a57f // smmla v31.4s, v11.16b, v1.16b
    bne LoopSz_TILE_20
LoopSzEnd_TILE_20:
    add x2, x2, x15 // weight += dz * src_depth_quad * (GEMM_INT8_UNIT * GEMM_INT8_SRC_UNIT);
    sub x5, x5, #1  // dz--
    // transpose
    uzp1 v11.2d, v12.2d, v13.2d
    uzp2 v12.2d, v12.2d, v13.2d
    uzp1 v13.2d, v14.2d, v15.2d
    uzp2 v14.2d, v14.2d, v15.2d
    uzp1 v15.2d, v16.2d, v17.2d
    uzp2 v16.2d, v16.2d, v17.2d
    uzp1 v17.2d, v18.2d, v19.2d
    uzp2 v18.2d, v18.2d, v19.2d
    uzp1 v19.2d, v20.2d, v21.2d
    uzp2 v20.2d, v20.2d, v21.2d
    uzp1 v21.2d, v22.2d, v23.2d
    uzp2 v22.2d, v22.2d, v23.2d
    uzp1 v23.2d, v24.2d, v25.2d
    uzp2 v24.2d, v24.2d, v25.2d
    uzp1 v25.2d, v26.2d, v27.2d
    uzp2 v26.2d, v26.2d, v27.2d
    uzp1 v27.2d, v28.2d, v29.2d
    uzp2 v28.2d, v28.2d, v29.2d
    uzp1 v29.2d, v30.2d, v31.2d
    uzp2 v30.2d, v30.2d, v31.2d
    Int32ToFloat v11, v12, v13, v14
    Int32ToFloat v15, v16, v17, v18
    Int32ToFloat v19, v20, v21, v22
    Int32ToFloat v23, v24, v25, v26
    Int32ToFloat v27, v28, v29, v30
    cbnz x8, Tile20Quan
    sub x4, x4, #256
    st1 {v11.4s, v12.4s, v13.4s, v14.4s}, [x0], #64
    st1 {v15.4s, v16.4s, v17.4s, v18.4s}, [x0], #64
    st1 {v19.4s, v20.4s, v21.4s, v22.4s}, [x0], #64
    st1 {v23.4s, v24.4s, v25.4s, v26.4s}, [x0], #64
    st1 {v27.4s, v28.4s, v29.4s, v30.4s}, [x0], x4
    add x4, x4, #256
    b Tile20LoopCheck
Tile20Quan:
    ld1 {v0.4s}, [x8], #16  // scale
    MUL_SCALE v0, v11, v12, v13, v14
    MUL_SCALE v0, v15, v16, v17, v18
    MUL_SCALE v0, v19, v20, v21, v22
    MUL_SCALE v0, v23, v24, v25, v26
    MUL_SCALE v0, v27, v28, v29, v30
    FloatToInt32 v11, v12, v13, v14
    FloatToInt32 v15, v16, v17, v18
    FloatToInt32 v19, v20, v21, v22
    FloatToInt32 v23, v24, v25, v26
    FloatToInt32 v27, v28, v29, v30
    Int32ToInt16 v11, v12, v13, v14, v0, v1
    Int32ToInt16 v15, v16, v17, v18, v2, v3
    Int32ToInt16 v19, v20, v21, v22, v4, v5
    Int32ToInt16 v23, v24, v25, v26, v6, v7
    Int32ToInt16 v27, v28, v29, v30, v8, v9
    Int16ToInt8 v0, v1, v2, v3, v16, v17
    Int16ToInt8 v4, v5, v6, v7, v18, v19
    Int16ToInt8_ONE v8, v9, v20
    dup v11.16b, w10 // max
    dup v10.16b, w14 // min
    smax v16.16b, v10.16b, v16.16b
    smax v17.16b, v10.16b, v17.16b
    smax v18.16b, v10.16b, v18.16b
    smax v19.16b, v10.16b, v19.16b
    smax v20.16b, v10.16b, v20.16b
    smin v16.16b, v11.16b, v16.16b
    smin v17.16b, v11.16b, v17.16b
    smin v18.16b, v11.16b, v18.16b
    smin v19.16b, v11.16b, v19.16b
    smin v20.16b, v11.16b, v20.16b
    sub x4, x4, #64
    st1 {v16.16b, v17.16b, v18.16b, v19.16b}, [x0], #64
    st1 {v20.16b}, [x0], x4  // dst += dz * dst_step;
    add x4, x4, #64
Tile20LoopCheck:
    cmp x5, #1
    bge LoopDz_TILE_20
    b End

TILE_16:
    dup v11.16b, w10 // max
    dup v10.16b, w14 // min
    sub x10, x22, #64
    cmp x7, #16
    blt TILE_8
    mov x16, x5 // dst_depth_quad
    mov x17, x0 // dst
    mov x18, x2 // weight
    mov x19, x8 // scale
    mov x20, x9 // bias
LoopDz_TILE_16:   // while (dz = dst_depth_quad)
    ld1 {v0.4s}, [x20], #16  // bias
    mov x11, x1 // src
    mov x12, x18 // weight
    mov x13, x3 // src_depth_quad
    mov v1.16b, v0.16b
    uzp1 v2.2d, v0.2d, v1.2d   // bias_0, bias_1, bias_0, bias_1
    uzp2 v3.2d, v0.2d, v1.2d   // bias_2, bias_3, bias_2, bias_3
    SET_BIAS v2, v16, v18, v20, v22
    SET_BIAS v2, v24, v26, v28, v30
    SET_BIAS v3, v17, v19, v21, v23
    SET_BIAS v3, v25, v27, v29, v31
LoopSz_TILE_16:
    // src    : 8 x [2 x 8] : v2-9
    // weight : 2 x [2 x 8] : v0-1
    // dst    : 8 x 2 x [4] : v16-v31
    ld1 {v0.16b, v1.16b}, [x12], #32                    // weight
    ld1 {v2.16b, v3.16b, v4.16b, v5.16b}, [x11], #64    // src
    .inst 0x4e80a450 // smmla v16.4s, v2.16b, v0.16b
    .inst 0x4e81a451 // smmla v17.4s, v2.16b, v1.16b
    .inst 0x4e80a472 // smmla v18.4s, v3.16b, v0.16b
    .inst 0x4e81a473 // smmla v19.4s, v3.16b, v1.16b
    ld1 {v6.16b, v7.16b, v8.16b, v9.16b}, [x11], x10
    .inst 0x4e80a494 // smmla v20.4s, v4.16b, v0.16b
    .inst 0x4e81a495 // smmla v21.4s, v4.16b, v1.16b
    .inst 0x4e80a4b6 // smmla v22.4s, v5.16b, v0.16b
    .inst 0x4e81a4b7 // smmla v23.4s, v5.16b, v1.16b
    .inst 0x4e80a4d8 // smmla v24.4s, v6.16b, v0.16b
    .inst 0x4e81a4d9 // smmla v25.4s, v6.16b, v1.16b
    .inst 0x4e80a4fa // smmla v26.4s, v7.16b, v0.16b
    .inst 0x4e81a4fb // smmla v27.4s, v7.16b, v1.16b
    subs x13, x13, #1
    .inst 0x4e80a51c // smmla v28.4s, v8.16b, v0.16b
    .inst 0x4e81a51d // smmla v29.4s, v8.16b, v1.16b
    .inst 0x4e80a53e // smmla v30.4s, v9.16b, v0.16b
    .inst 0x4e81a53f // smmla v31.4s, v9.16b, v1.16b
    bne LoopSz_TILE_16
LoopSzEnd_TILE_16:
    add x18, x18, x15 // weight += dz * src_depth_quad * (GEMM_INT8_UNIT * GEMM_INT8_SRC_UNIT);
    sub x16, x16, #1  // dz--
    // transpose
    uzp1 v15.2d, v16.2d, v17.2d
    uzp2 v16.2d, v16.2d, v17.2d
    uzp1 v17.2d, v18.2d, v19.2d
    uzp2 v18.2d, v18.2d, v19.2d
    uzp1 v19.2d, v20.2d, v21.2d
    uzp2 v20.2d, v20.2d, v21.2d
    uzp1 v21.2d, v22.2d, v23.2d
    uzp2 v22.2d, v22.2d, v23.2d
    uzp1 v23.2d, v24.2d, v25.2d
    uzp2 v24.2d, v24.2d, v25.2d
    uzp1 v25.2d, v26.2d, v27.2d
    uzp2 v26.2d, v26.2d, v27.2d
    uzp1 v27.2d, v28.2d, v29.2d
    uzp2 v28.2d, v28.2d, v29.2d
    uzp1 v29.2d, v30.2d, v31.2d
    uzp2 v30.2d, v30.2d, v31.2d
    Int32ToFloat v15, v16, v17, v18
    Int32ToFloat v19, v20, v21, v22
    Int32ToFloat v23, v24, v25, v26
    Int32ToFloat v27, v28, v29, v30
    cbnz x8, Tile16Quan
    sub x4, x4, #192
    st1 {v15.4s, v16.4s, v17.4s, v18.4s}, [x17], #64
    st1 {v19.4s, v20.4s, v21.4s, v22.4s}, [x17], #64
    st1 {v23.4s, v24.4s, v25.4s, v26.4s}, [x17], #64
    st1 {v27.4s, v28.4s, v29.4s, v30.4s}, [x17], x4
    add x4, x4, #192
    b Tile16LoopCheck
Tile16Quan:
    ld1 {v0.4s}, [x19], #16  // scale
    MUL_SCALE v0, v15, v16, v17, v18
    MUL_SCALE v0, v19, v20, v21, v22
    MUL_SCALE v0, v23, v24, v25, v26
    MUL_SCALE v0, v27, v28, v29, v30
    FloatToInt32 v15, v16, v17, v18
    FloatToInt32 v19, v20, v21, v22
    FloatToInt32 v23, v24, v25, v26
    FloatToInt32 v27, v28, v29, v30
    Int32ToInt16 v15, v16, v17, v18, v0, v1
    Int32ToInt16 v19, v20, v21, v22, v2, v3
    Int32ToInt16 v23, v24, v25, v26, v4, v5
    Int32ToInt16 v27, v28, v29, v30, v6, v7
    Int16ToInt8 v0, v1, v2, v3, v16, v17
    Int16ToInt8 v4, v5, v6, v7, v18, v19
    smax v16.16b, v10.16b, v16.16b
    smax v17.16b, v10.16b, v17.16b
    smax v18.16b, v10.16b, v18.16b
    smax v19.16b, v10.16b, v19.16b
    smin v16.16b, v11.16b, v16.16b
    smin v17.16b, v11.16b, v17.16b
    smin v18.16b, v11.16b, v18.16b
    smin v19.16b, v11.16b, v19.16b
    st1 {v16.16b, v17.16b, v18.16b, v19.16b}, [x17], x4  // dst += dz * dst_step;
Tile16LoopCheck:
    cmp x16, #1
    bge LoopDz_TILE_16
Tile16End:
    sub x7, x7, #16
    add x0, x0, x21, LSL #4
    add x1, x1, #128

TILE_8:
    cmp x7, #8
    blt TILE_4
    mov x16, x5 // dst_depth_quad
    mov x17, x0 // dst
    mov x18, x2 // weight
    mov x19, x8 // scale
    mov x20, x9 // bias
LoopDz_TILE_8:
    ld1 {v0.4s}, [x20], #16  // bias
    mov x11, x1 // src
    mov x12, x18 // weight
    mov x13, x3 // src_depth_quad
    mov v1.16b, v0.16b
    uzp1 v2.2d, v0.2d, v1.2d   // bias_0, bias_1, bias_0, bias_1
    uzp2 v3.2d, v0.2d, v1.2d   // bias_2, bias_3, bias_2, bias_3
    SET_BIAS v2, v24, v26, v28, v30
    SET_BIAS v3, v25, v27, v29, v31
LoopSz_TILE_8:
    // src    : 4 x [2 x 8] : v2-5
    // weight : 2 x [2 x 8] : v0-1
    // dst    : 4 x 2 x [4] : v24-v31
    ld1 {v0.16b, v1.16b}, [x12], #32                   // weight
    ld1 {v2.16b, v3.16b, v4.16b, v5.16b}, [x11], x22   // src
    .inst 0x4e80a458 // smmla v24.4s, v2.16b, v0.16b
    .inst 0x4e81a459 // smmla v25.4s, v2.16b, v1.16b
    .inst 0x4e80a47a // smmla v26.4s, v3.16b, v0.16b
    .inst 0x4e81a47b // smmla v27.4s, v3.16b, v1.16b
    .inst 0x4e80a49c // smmla v28.4s, v4.16b, v0.16b
    .inst 0x4e81a49d // smmla v29.4s, v4.16b, v1.16b
    .inst 0x4e80a4be // smmla v30.4s, v5.16b, v0.16b
    .inst 0x4e81a4bf // smmla v31.4s, v5.16b, v1.16b
    subs x13, x13, #1
    bne LoopSz_TILE_8
LoopSzEnd_TILE_8:
    add x18, x18, x15
    sub x16, x16, #1
    uzp1 v23.2d, v24.2d, v25.2d
    uzp2 v24.2d, v24.2d, v25.2d
    uzp1 v25.2d, v26.2d, v27.2d
    uzp2 v26.2d, v26.2d, v27.2d
    uzp1 v27.2d, v28.2d, v29.2d
    uzp2 v28.2d, v28.2d, v29.2d
    uzp1 v29.2d, v30.2d, v31.2d
    uzp2 v30.2d, v30.2d, v31.2d
    Int32ToFloat v23, v24, v25, v26
    Int32ToFloat v27, v28, v29, v30
    cbnz x8, Tile8Quan
    sub x4, x4, #64
    st1 {v23.4s, v24.4s, v25.4s, v26.4s}, [x17], #64
    st1 {v27.4s, v28.4s, v29.4s, v30.4s}, [x17], x4
    add x4, x4, #64
    b Tile8LoopCheck
Tile8Quan:
    ld1 {v0.4s}, [x19], #16  // scale
    MUL_SCALE v0, v23, v24, v25, v26
    MUL_SCALE v0, v27, v28, v29, v30
    FloatToInt32 v23, v24, v25, v26
    FloatToInt32 v27, v28, v29, v30
    Int32ToInt16 v23, v24, v25, v26, v4, v5
    Int32ToInt16 v27, v28, v29, v30, v6, v7
    Int16ToInt8 v4, v5, v6, v7, v18, v19
    smax v18.16b, v10.16b, v18.16b
    smax v19.16b, v10.16b, v19.16b
    smin v18.16b, v11.16b, v18.16b
    smin v19.16b, v11.16b, v19.16b
    st1 {v18.16b, v19.16b}, [x17], x4  // dst += dz * dst_step
Tile8LoopCheck:
    cmp x16, #1
    bge LoopDz_TILE_8
Tile8End:
    sub x7, x7, #8
    add x0, x0, x21, LSL #3
    add x1, x1, #64

TILE_4:
    cmp x7, #4
    blt TILE_2
    mov x16, x5 // dst_depth_quad
    mov x17, x0 // dst
    mov x18, x2 // weight
    mov x19, x8 // scale
    mov x20, x9 // bias
LoopDz_TILE_4:
    ld1 {v0.4s}, [x20], #16  // bias
    mov x11, x1 // src
    mov x12, x18 // weight
    mov x13, x3 // src_depth_quad
    mov v1.16b, v0.16b
    uzp1 v28.2d, v0.2d, v1.2d   // bias_0, bias_1, bias_0, bias_1
    uzp2 v29.2d, v0.2d, v1.2d   // bias_2, bias_3, bias_2, bias_3
    mov v30.16b, v28.16b
    mov v31.16b, v29.16b
LoopSz_TILE_4:
    // src    : 2 x [2 x 8] : v2-3
    // weight : 2 x [2 x 8] : v0-1
    // dst    : 2 x 2 x [4] : v28-v31
    ld1 {v0.16b, v1.16b}, [x12], #32   // weight
    ld1 {v2.16b, v3.16b}, [x11], x22   // src
    .inst 0x4e80a45c // smmla v28.4s, v2.16b, v0.16b
    .inst 0x4e81a45d // smmla v29.4s, v2.16b, v1.16b
    .inst 0x4e80a47e // smmla v30.4s, v3.16b, v0.16b
    .inst 0x4e81a47f // smmla v31.4s, v3.16b, v1.16b
    subs x13, x13, #1
    bne LoopSz_TILE_4
LoopSzEnd_TILE_4:
    add x18, x18, x15
    sub x16, x16, #1
    uzp1 v27.2d, v28.2d, v29.2d
    uzp2 v28.2d, v28.2d, v29.2d
    uzp1 v29.2d, v30.2d, v31.2d
    uzp2 v30.2d, v30.2d, v31.2d
    Int32ToFloat v27, v28, v29, v30
    cbnz x8, Tile4Quan
    st1 {v27.4s, v28.4s, v29.4s, v30.4s}, [x17], x4
    b Tile4LoopCheck
Tile4Quan:
    ld1 {v0.4s}, [x19], #16  // scale
    MUL_SCALE v0, v27, v28, v29, v30
    FloatToInt32 v27, v28, v29, v30
    Int32ToInt16 v27, v28, v29, v30, v6, v7
    Int16ToInt8_ONE v6, v7, v19
    smax v19.16b, v10.16b, v19.16b
    smin v19.16b, v11.16b, v19.16b
    st1 {v19.16b}, [x17], x4  // dst += dz * dst_step
Tile4LoopCheck:
    cmp x16, #1
    bge LoopDz_TILE_4
Tile4End:
    sub x7, x7, #4
    add x0, x0, x21, LSL #2
    add x1, x1, #32

TILE_2:
    cmp x7, #2
    blt TILE_1
    mov x16, x5 // dst_depth_quad
    mov x17, x0 // dst
    mov x18, x2 // weight
    mov x19, x8 // scale
    mov x20, x9 // bias
LoopDz_TILE_2:
    ld1 {v0.4s}, [x20], #16  // bias
    mov x11, x1 // src
    mov x12, x18 // weight
    mov x13, x3 // src_depth_quad
    mov v1.16b, v0.16b
    uzp1 v30.2d, v0.2d, v1.2d   // bias_0, bias_1, bias_0, bias_1
    uzp2 v31.2d, v0.2d, v1.2d   // bias_2, bias_3, bias_2, bias_3
LoopSz_TILE_2:
    // src    : 1 x [2 x 8] : v2
    // weight : 2 x [2 x 8] : v0-1
    // dst    : 1 x 2 x [4] : v30-v31
    ld1 {v0.16b, v1.16b}, [x12], #32   // weight
    ld1 {v2.16b}, [x11], x22           // src
    .inst 0x4e80a45e // smmla v30.4s, v2.16b, v0.16b
    .inst 0x4e81a45f // smmla v31.4s, v2.16b, v1.16b
    subs x13, x13, #1
    bne LoopSz_TILE_2
LoopSzEnd_TILE_2:
    add x18, x18, x15
    sub x16, x16, #1
    uzp1 v29.2d, v30.2d, v31.2d
    uzp2 v30.2d, v30.2d, v31.2d
    scvtf v29.4s, v29.4s
    scvtf v30.4s, v30.4s
    cbnz x8, Tile2Quan
    st1 {v29.4s, v30.4s}, [x17], x4
    b Tile2LoopCheck
Tile2Quan:
    ld1 {v0.4s}, [x19], #16  // scale
    fmul v29.4s, v29.4s, v0.4s
    fmul v30.4s, v30.4s, v0.4s
    fcvtas v29.4s, v29.4s
    fcvtas v30.4s, v30.4s
    sqxtn v6.4h,  v29.4s
    sqxtn2 v6.8h, v30.4s
    sqxtn v19.8b, v6.8h
    smax v19.16b, v10.16b, v19.16b
    smin v19.16b, v11.16b, v19.16b
    st1 {v19.8b}, [x17], x4  // dst += dz * dst_step

Tile2LoopCheck:
    cmp x16, #1
    bge LoopDz_TILE_2
Tile2End:
    sub x7, x7, #2
    add x0, x0, x21, LSL #1
    add x1, x1, #16

TILE_1:
    cmp x7, #1
    blt End
    mov x16, x5 // dst_depth_quad
    mov x17, x0 // dst
    mov x18, x2 // weight
    mov x19, x8 // scale
    mov x20, x9 // bias
LoopDz_TILE_1:
    ld1 {v0.4s}, [x20], #16  // bias
    mov x11, x1 // src
    mov x12, x18 // weight
    mov x13, x3 // src_depth_quad
    mov v1.16b, v0.16b
    uzp1 v30.2d, v0.2d, v1.2d   // bias_0, bias_1, bias_0, bias_1
    uzp2 v31.2d, v0.2d, v1.2d   // bias_2, bias_3, bias_2, bias_3
LoopSz_TILE_1:
    // src    : 1 x [1 x 8] : v2
    // weight : 2 x [2 x 8] : v0-1
    // dst    : 1 x 2 x [2] : v30-v31
    ld1 {v0.16b, v1.16b}, [x12], #32   // weight
    ld1 {v2.8b}, [x11], x22           // src
    .inst 0x4e80a45e // smmla v30.4s, v2.16b, v0.16b
    .inst 0x4e81a45f // smmla v31.4s, v2.16b, v1.16b
    subs x13, x13, #1
    bne LoopSz_TILE_1
LoopSzEnd_TILE_1:
    add x18, x18, x15
    sub x16, x16, #1
    uzp1 v29.2d, v30.2d, v31.2d
    uzp2 v30.2d, v30.2d, v31.2d
    scvtf v29.4s, v29.4s
    scvtf v30.4s, v30.4s
    cbnz x8, Tile1Quan
    st1 {v29.4s, v30.4s}, [x17], x4
    b Tile1LoopEnd
Tile1Quan:
    ld1 {v0.4s}, [x19], #16  // scale
    fmul v29.4s, v29.4s, v0.4s
    fmul v30.4s, v30.4s, v0.4s
    fcvtas v29.4s, v29.4s
    fcvtas v30.4s, v30.4s
    sqxtn v6.4h,  v29.4s
    sqxtn2 v6.8h, v30.4s
    sqxtn v19.8b, v6.8h
    smax v19.16b, v10.16b, v19.16b
    smin v19.16b, v11.16b, v19.16b
    st1 {v19.s}[0], [x17], x4  // dst += dz * dst_step

Tile1LoopEnd:
    cmp x16, #1
    bge LoopDz_TILE_1

End:
ldp x19, x20, [sp, #(16 * 5)]
ldp x21, x22, [sp, #(16 * 4)]
ldp d8,  d9,  [sp, #(16 * 3)]
ldp d10, d11, [sp, #(16 * 2)]
ldp d12, d13, [sp, #(16 * 1)]
ldp d14, d15, [sp], #(16 * 6)
ret

#endif // __aarch64__
