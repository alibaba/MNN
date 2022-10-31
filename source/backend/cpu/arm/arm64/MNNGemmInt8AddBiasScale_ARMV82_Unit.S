//
//  MNNGemmInt8AddBiasScale_ARMV82_Unit.S
//  MNN
//
//  Created by MNN on 2019/12/17.
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

asm_function MNNGemmInt8AddBiasScale_ARMV82_Unit

//struct QuanPostTreatParameters {
//    const float* scale;
//    const int32_t* bias;
//    int32_t maxValue;
//    int32_t minValue;
//};

//void MNNGemmInt8AddBiasScale_ARMV82_Unit(int8_t* dst, const int8_t* src, 
//    const int8_t* weight, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad,
// const QuanPostTreatParameters* parameters, size_t realDstCount);

//Auto: x0:dst, x1:src, x2:weight, x3:src_depth_quad, x4:dst_step
//x5:dst_depth_quad, x6: parameters, x7: realDstCount

//Load from x7: x8: scale, x9: bias, w12: maxValue, w13: minValue
ldr x8, [x6, #0]
ldr x9, [x6, #8]
ldr w12, [x6, #16]
ldr w13, [x6, #20]
dup v7.16b, w12 // max
dup v6.16b, w13 // min

stp d14, d15, [sp, #(-16 * 6)]!
stp d12, d13, [sp, #(16 * 1)]
stp d10, d11, [sp, #(16 * 2)]
stp d8,  d9,  [sp, #(16 * 3)]
stp x21, x22, [sp, #(16 * 4)]
stp x19, x20, [sp, #(16 * 5)]

mov x21, #4 // sizeof(int8_t) * UNIT
cbnz x8, Start
mov x21, #16 // sizeof(float) * UNIT
Start:
lsl x15, x3, #4 // x15 = src_depth_quad * UNIT * SRC_UNIT
mov x22, #48 // src_steps

TILE_12:
    cmp x7, #12
    blt TILE_8
    cmp x5, #2
    blt L4LoopDz_TILE_12
L8LoopDz_TILE_12:
    ld1 {v0.4s, v1.4s}, [x9], #32 // bias
    mov x11, x1
    mov x13, x3

    SET_BIAS v0, v8, v9, v10, v11
    SET_BIAS v0, v12, v13, v14, v15
    SET_BIAS v0, v16, v17, v18, v19
    SET_BIAS v1, v20, v21, v22, v23
    SET_BIAS v1, v24, v25, v26, v27
    SET_BIAS v1, v28, v29, v30, v31

    L8LoopSz_TILE_12:
        ld1 {v3.16b}, [x2], x15 // weight
        ld1 {v0.16b, v1.16b, v2.16b}, [x11], #48 // src
        .inst 0x4f80e068 // sdot v8.4s, v3.16b, v0.4b[0]
        .inst 0x4fa0e069 // sdot v9.4s, v3.16b, v0.4b[1]
        .inst 0x4f80e86a // sdot v10.4s, v3.16b, v0.4b[2]
        .inst 0x4fa0e86b // sdot v11.4s, v3.16b, v0.4b[3]
        ld1 {v4.16b}, [x2], #16
        .inst 0x4f81e06c // sdot v12.4s, v3.16b, v1.4b[0]
        .inst 0x4fa1e06d // sdot v13.4s, v3.16b, v1.4b[1]
        .inst 0x4f81e86e // sdot v14.4s, v3.16b, v1.4b[2]
        .inst 0x4fa1e86f // sdot v15.4s, v3.16b, v1.4b[3]
        .inst 0x4f82e070 // sdot v16.4s, v3.16b, v2.4b[0]
        .inst 0x4fa2e071 // sdot v17.4s, v3.16b, v2.4b[1]
        .inst 0x4f82e872 // sdot v18.4s, v3.16b, v2.4b[2]
        .inst 0x4fa2e873 // sdot v19.4s, v3.16b, v2.4b[3]
        .inst 0x4f80e094 // sdot v20.4s, v4.16b, v0.4b[0]
        .inst 0x4fa0e095 // sdot v21.4s, v4.16b, v0.4b[1]
        .inst 0x4f80e896 // sdot v22.4s, v4.16b, v0.4b[2]
        .inst 0x4fa0e897 // sdot v23.4s, v4.16b, v0.4b[3]
        sub x2, x2, x15
        .inst 0x4f81e098 // sdot v24.4s, v4.16b, v1.4b[0]
        .inst 0x4fa1e099 // sdot v25.4s, v4.16b, v1.4b[1]
        .inst 0x4f81e89a // sdot v26.4s, v4.16b, v1.4b[2]
        .inst 0x4fa1e89b // sdot v27.4s, v4.16b, v1.4b[3]
        subs x13, x13, #1
        .inst 0x4f82e09c // sdot v28.4s, v4.16b, v2.4b[0]
        .inst 0x4fa2e09d // sdot v29.4s, v4.16b, v2.4b[1]
        .inst 0x4f82e89e // sdot v30.4s, v4.16b, v2.4b[2]
        .inst 0x4fa2e89f // sdot v31.4s, v4.16b, v2.4b[3]
        bne L8LoopSz_TILE_12

    L8LoopSzEnd_TILE_12:
    add x2, x2, x15
    sub x5, x5, #2
    cbnz x8, L8Tile12Quan
    sub x4, x4, #128
    Int32ToFloat v8, v9, v10, v11
    Int32ToFloat v12, v13, v14, v15
    Int32ToFloat v16, v17, v18, v19
    Int32ToFloat v20, v21, v22, v23
    Int32ToFloat v24, v25, v26, v27
    Int32ToFloat v28, v29, v30, v31
    st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x0], #64
    st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [x0], #64
    st1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x0], x4
    st1 {v20.4s, v21.4s, v22.4s, v23.4s}, [x0], #64
    st1 {v24.4s, v25.4s, v26.4s, v27.4s}, [x0], #64
    st1 {v28.4s, v29.4s, v30.4s, v31.4s}, [x0], x4
    add x4, x4, #128
    b L8Tile12LoopCheck

    L8Tile12Quan:
    ld1 {v0.4s, v1.4s}, [x8], #32 // scale
    Int32ToFloat v8, v9, v10, v11
    Int32ToFloat v12, v13, v14, v15
    Int32ToFloat v16, v17, v18, v19
    Int32ToFloat v20, v21, v22, v23
    Int32ToFloat v24, v25, v26, v27
    Int32ToFloat v28, v29, v30, v31
    MUL_SCALE v0, v8, v9, v10, v11
    MUL_SCALE v0, v12, v13, v14, v15
    MUL_SCALE v0, v16, v17, v18, v19
    MUL_SCALE v1, v20, v21, v22, v23
    MUL_SCALE v1, v24, v25, v26, v27
    MUL_SCALE v1, v28, v29, v30, v31
    FloatToInt32 v8, v9, v10, v11
    FloatToInt32 v12, v13, v14, v15
    FloatToInt32 v16, v17, v18, v19
    FloatToInt32 v20, v21, v22, v23
    FloatToInt32 v24, v25, v26, v27
    FloatToInt32 v28, v29, v30, v31
    Int32ToInt16 v8, v9, v10, v11, v0, v1
    Int32ToInt16 v12, v13, v14, v15, v2, v3
    Int32ToInt16 v16, v17, v18, v19, v4, v5
    Int32ToInt16 v20, v21, v22, v23, v8, v9
    Int32ToInt16 v24, v25, v26, v27, v10, v11
    Int32ToInt16 v28, v29, v30, v31, v12, v13
    Int16ToInt8 v0, v1, v2, v3, v16, v17
    Int16ToInt8 v4, v5, v8, v9, v18, v19
    Int16ToInt8 v10, v11, v12, v13, v20, v21
    smax v16.16b, v6.16b, v16.16b
    smax v17.16b, v6.16b, v17.16b
    smax v18.16b, v6.16b, v18.16b
    smax v19.16b, v6.16b, v19.16b
    smax v20.16b, v6.16b, v20.16b
    smax v21.16b, v6.16b, v21.16b
    smin v16.16b, v7.16b, v16.16b
    smin v17.16b, v7.16b, v17.16b
    smin v18.16b, v7.16b, v18.16b
    smin v19.16b, v7.16b, v19.16b
    smin v20.16b, v7.16b, v20.16b
    smin v21.16b, v7.16b, v21.16b
    st1 {v16.16b, v17.16b, v18.16b}, [x0], x4
    st1 {v19.16b, v20.16b, v21.16b}, [x0], x4

    L8Tile12LoopCheck:
    cmp x5, #1
    bgt L8LoopDz_TILE_12
    blt End

L4LoopDz_TILE_12:
    ld1 {v0.4s}, [x9] // bias

    SET_BIAS v0, v8, v9, v10, v11
    SET_BIAS v0, v12, v13, v14, v15
    SET_BIAS v0, v16, v17, v18, v19

    L4LoopSz_TILE_12:
        ld1 {v3.16b}, [x2], #16 // weight
        ld1 {v0.16b, v1.16b, v2.16b}, [x1], #48 // src
        .inst 0x4f80e068 // sdot v8.4s, v3.16b, v0.4b[0]
        .inst 0x4fa0e069 // sdot v9.4s, v3.16b, v0.4b[1]
        .inst 0x4f80e86a // sdot v10.4s, v3.16b, v0.4b[2]
        .inst 0x4fa0e86b // sdot v11.4s, v3.16b, v0.4b[3]
        .inst 0x4f81e06c // sdot v12.4s, v3.16b, v1.4b[0]
        .inst 0x4fa1e06d // sdot v13.4s, v3.16b, v1.4b[1]
        .inst 0x4f81e86e // sdot v14.4s, v3.16b, v1.4b[2]
        .inst 0x4fa1e86f // sdot v15.4s, v3.16b, v1.4b[3]
        subs x3, x3, #1
        .inst 0x4f82e070 // sdot v16.4s, v3.16b, v2.4b[0]
        .inst 0x4fa2e071 // sdot v17.4s, v3.16b, v2.4b[1]
        .inst 0x4f82e872 // sdot v18.4s, v3.16b, v2.4b[2]
        .inst 0x4fa2e873 // sdot v19.4s, v3.16b, v2.4b[3]
        bne L4LoopSz_TILE_12

    L4LoopSzEnd_TILE_12:
    cbnz x8, L4Tile12Quan
    sub x4, x4, #128
    Int32ToFloat v8, v9, v10, v11
    Int32ToFloat v12, v13, v14, v15
    Int32ToFloat v16, v17, v18, v19
    st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x0], #64
    st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [x0], #64
    st1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x0], x4
    add x4, x4, #128
    b End

    L4Tile12Quan:
    ld1 {v0.4s}, [x8] // scale
    Int32ToFloat v8, v9, v10, v11
    Int32ToFloat v12, v13, v14, v15
    Int32ToFloat v16, v17, v18, v19
    MUL_SCALE v0, v8, v9, v10, v11
    MUL_SCALE v0, v12, v13, v14, v15
    MUL_SCALE v0, v16, v17, v18, v19
    FloatToInt32 v8, v9, v10, v11
    FloatToInt32 v12, v13, v14, v15
    FloatToInt32 v16, v17, v18, v19
    Int32ToInt16 v8, v9, v10, v11, v0, v1
    Int32ToInt16 v12, v13, v14, v15, v2, v3
    Int32ToInt16 v16, v17, v18, v19, v4, v5
    Int16ToInt8 v0, v1, v2, v3, v16, v17
    Int16ToInt8_ONE v4, v5, v18
    smax v16.16b, v6.16b, v16.16b
    smax v17.16b, v6.16b, v17.16b
    smax v18.16b, v6.16b, v18.16b
    smin v16.16b, v7.16b, v16.16b
    smin v17.16b, v7.16b, v17.16b
    smin v18.16b, v7.16b, v18.16b
    st1 {v16.16b, v17.16b, v18.16b}, [x0], x4
    b End

TILE_8:
    cmp x7, #8
    blt TILE_4
    mov x10, x0
    mov x12, x2
    mov x14, x5
    mov x19, x8 // scale
    mov x20, x9 // bias
    cmp x5, #2
    blt L4LoopDz_TILE_8
L8LoopDz_TILE_8:
    ld1 {v0.4s, v1.4s}, [x20], #32 // bias
    mov x11, x1
    mov x13, x3

    SET_BIAS v0, v8, v9, v10, v11
    SET_BIAS v0, v12, v13, v14, v15
    SET_BIAS v1, v16, v17, v18, v19
    SET_BIAS v1, v20, v21, v22, v23

    L8LoopSz_TILE_8:
        ld1 {v3.16b}, [x12], x15 // weight
        ld1 {v0.16b, v1.16b}, [x11], x22 // src
        .inst 0x4f80e068 // sdot v8.4s, v3.16b, v0.4b[0]
        .inst 0x4fa0e069 // sdot v9.4s, v3.16b, v0.4b[1]
        .inst 0x4f80e86a // sdot v10.4s, v3.16b, v0.4b[2]
        .inst 0x4fa0e86b // sdot v11.4s, v3.16b, v0.4b[3]
        ld1 {v4.16b}, [x12], #16
        .inst 0x4f81e06c // sdot v12.4s, v3.16b, v1.4b[0]
        .inst 0x4fa1e06d // sdot v13.4s, v3.16b, v1.4b[1]
        .inst 0x4f81e86e // sdot v14.4s, v3.16b, v1.4b[2]
        .inst 0x4fa1e86f // sdot v15.4s, v3.16b, v1.4b[3]
        sub x12, x12, x15
        .inst 0x4f80e090 // sdot v16.4s, v4.16b, v0.4b[0]
        .inst 0x4fa0e091 // sdot v17.4s, v4.16b, v0.4b[1]
        .inst 0x4f80e892 // sdot v18.4s, v4.16b, v0.4b[2]
        .inst 0x4fa0e893 // sdot v19.4s, v4.16b, v0.4b[3]
        subs x13, x13, #1
        .inst 0x4f81e094 // sdot v20.4s, v4.16b, v1.4b[0]
        .inst 0x4fa1e095 // sdot v21.4s, v4.16b, v1.4b[1]
        .inst 0x4f81e896 // sdot v22.4s, v4.16b, v1.4b[2]
        .inst 0x4fa1e897 // sdot v23.4s, v4.16b, v1.4b[3]
        bne L8LoopSz_TILE_8

    L8LoopSzEnd_TILE_8:
    add x12, x12, x15
    sub x14, x14, #2
    cbnz x8, L8Tile8Quan
    sub x4, x4, #64
    Int32ToFloat v8, v9, v10, v11
    Int32ToFloat v12, v13, v14, v15
    Int32ToFloat v16, v17, v18, v19
    Int32ToFloat v20, v21, v22, v23
    st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x10], #64
    st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [x10], x4
    st1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x10], #64
    st1 {v20.4s, v21.4s, v22.4s, v23.4s}, [x10], x4
    add x4, x4, #64
    b L8Tile8LoopCheck

    L8Tile8Quan:
    ld1 {v0.4s, v1.4s}, [x19], #32 // scale
    Int32ToFloat v8, v9, v10, v11
    Int32ToFloat v12, v13, v14, v15
    Int32ToFloat v16, v17, v18, v19
    Int32ToFloat v20, v21, v22, v23
    MUL_SCALE v0, v8, v9, v10, v11
    MUL_SCALE v0, v12, v13, v14, v15
    MUL_SCALE v1, v16, v17, v18, v19
    MUL_SCALE v1, v20, v21, v22, v23
    FloatToInt32 v8, v9, v10, v11
    FloatToInt32 v12, v13, v14, v15
    FloatToInt32 v16, v17, v18, v19
    FloatToInt32 v20, v21, v22, v23
    Int32ToInt16 v8, v9, v10, v11, v0, v1
    Int32ToInt16 v12, v13, v14, v15, v2, v3
    Int32ToInt16 v16, v17, v18, v19, v4, v5
    Int32ToInt16 v20, v21, v22, v23, v8, v9
    Int16ToInt8 v0, v1, v2, v3, v16, v17
    Int16ToInt8 v4, v5, v8, v9, v18, v19
    smax v16.16b, v6.16b, v16.16b
    smax v17.16b, v6.16b, v17.16b
    smax v18.16b, v6.16b, v18.16b
    smax v19.16b, v6.16b, v19.16b
    smin v16.16b, v7.16b, v16.16b
    smin v17.16b, v7.16b, v17.16b
    smin v18.16b, v7.16b, v18.16b
    smin v19.16b, v7.16b, v19.16b
    st1 {v16.16b, v17.16b}, [x10], x4
    st1 {v18.16b, v19.16b}, [x10], x4

    L8Tile8LoopCheck:
    cmp x14, #1
    bgt L8LoopDz_TILE_8
    cbz x14, Tile8End

L4LoopDz_TILE_8:
    ld1 {v0.4s}, [x20], #16 // bias
    mov x11, x1
    mov x13, x3

    SET_BIAS v0, v8, v9, v10, v11
    SET_BIAS v0, v12, v13, v14, v15

    L4LoopSz_TILE_8:
        ld1 {v3.16b}, [x12], #16 // weight
        ld1 {v0.16b, v1.16b}, [x11], x22 // src
        .inst 0x4f80e068 // sdot v8.4s, v3.16b, v0.4b[0]
        .inst 0x4fa0e069 // sdot v9.4s, v3.16b, v0.4b[1]
        .inst 0x4f80e86a // sdot v10.4s, v3.16b, v0.4b[2]
        .inst 0x4fa0e86b // sdot v11.4s, v3.16b, v0.4b[3]
        subs x13, x13, #1
        .inst 0x4f81e06c // sdot v12.4s, v3.16b, v1.4b[0]
        .inst 0x4fa1e06d // sdot v13.4s, v3.16b, v1.4b[1]
        .inst 0x4f81e86e // sdot v14.4s, v3.16b, v1.4b[2]
        .inst 0x4fa1e86f // sdot v15.4s, v3.16b, v1.4b[3]
        bne L4LoopSz_TILE_8

    L4LoopSzEnd_TILE_8:
    cbnz x8, L4Tile8Quan
    sub x4, x4, #64
    Int32ToFloat v8, v9, v10, v11
    Int32ToFloat v12, v13, v14, v15
    st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x10], #64
    st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [x10], x4
    add x4, x4, #64
    b Tile8End

    L4Tile8Quan:
    ld1 {v0.4s, v1.4s}, [x19], #32 // scale
    Int32ToFloat v8, v9, v10, v11
    Int32ToFloat v12, v13, v14, v15
    MUL_SCALE v0, v8, v9, v10, v11
    MUL_SCALE v0, v12, v13, v14, v15
    FloatToInt32 v8, v9, v10, v11
    FloatToInt32 v12, v13, v14, v15
    Int32ToInt16 v8, v9, v10, v11, v0, v1
    Int32ToInt16 v12, v13, v14, v15, v2, v3
    Int16ToInt8 v0, v1, v2, v3, v16, v17
    smax v16.16b, v6.16b, v16.16b
    smax v17.16b, v6.16b, v17.16b
    smin v16.16b, v7.16b, v16.16b
    smin v17.16b, v7.16b, v17.16b
    st1 {v16.16b, v17.16b}, [x10], x4

Tile8End:
    sub x7, x7, #8
    add x0, x0, x21, LSL #3
    add x1, x1, #32

TILE_4:
    cmp x7, #4
    blt TILE_1
    mov x10, x0
    mov x12, x2
    mov x14, x5
    mov x19, x8
    mov x20, x9
    cmp x5, #2
    blt L4LoopDz_TILE_4
L8LoopDz_TILE_4:
    ld1 {v0.4s, v1.4s}, [x20], #32 // bias
    mov x11, x1
    mov x13, x3

    SET_BIAS v0, v8, v9, v10, v11
    SET_BIAS v1, v12, v13, v14, v15

    L8LoopSz_TILE_4:
        ld1 {v3.16b}, [x12], x15 // weight
        ld1 {v0.16b}, [x11], x22 // src
        ld1 {v4.16b}, [x12], #16 // weight
        .inst 0x4f80e068 // sdot v8.4s, v3.16b, v0.4b[0]
        .inst 0x4fa0e069 // sdot v9.4s, v3.16b, v0.4b[1]
        .inst 0x4f80e86a // sdot v10.4s, v3.16b, v0.4b[2]
        .inst 0x4fa0e86b // sdot v11.4s, v3.16b, v0.4b[3]
        subs x13, x13, #1
        sub x12, x12, x15
        .inst 0x4f80e08c // sdot v12.4s, v4.16b, v0.4b[0]
        .inst 0x4fa0e08d // sdot v13.4s, v4.16b, v0.4b[1]
        .inst 0x4f80e88e // sdot v14.4s, v4.16b, v0.4b[2]
        .inst 0x4fa0e88f // sdot v15.4s, v4.16b, v0.4b[3]
        bne L8LoopSz_TILE_4

    L8LoopSzEnd_TILE_4:
    add x12, x12, x15
    sub x14, x14, #2
    cbnz x8, L8Tile4Quan
    Int32ToFloat v8, v9, v10, v11
    Int32ToFloat v12, v13, v14, v15
    st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x10], x4
    st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [x10], x4
    b L8Tile4LoopCheck

    L8Tile4Quan:
    ld1 {v0.4s, v1.4s}, [x19], #32 // scale
    Int32ToFloat v8, v9, v10, v11
    Int32ToFloat v12, v13, v14, v15
    MUL_SCALE v0, v8, v9, v10, v11
    MUL_SCALE v1, v12, v13, v14, v15
    FloatToInt32 v8, v9, v10, v11
    FloatToInt32 v12, v13, v14, v15
    Int32ToInt16 v8, v9, v10, v11, v0, v1
    Int32ToInt16 v12, v13, v14, v15, v2, v3
    Int16ToInt8 v0, v1, v2, v3, v16, v17
    smax v16.16b, v6.16b, v16.16b
    smax v17.16b, v6.16b, v17.16b
    smin v16.16b, v7.16b, v16.16b
    smin v17.16b, v7.16b, v17.16b
    st1 {v16.16b}, [x10], x4
    st1 {v17.16b}, [x10], x4

    L8Tile4LoopCheck:
    cmp x14, #1
    bgt L8LoopDz_TILE_4
    cbz x14, Tile4End

L4LoopDz_TILE_4:
    ld1 {v0.4s}, [x20], #16 // bias
    mov x11, x1
    mov x13, x3
    SET_BIAS v0, v8, v9, v10, v11

    L4LoopSz_TILE_4:
        ld1 {v3.16b}, [x12], #16 // weight
        ld1 {v0.16b}, [x11], x22 // src
        subs x13, x13, #1
        .inst 0x4f80e068 // sdot v8.4s, v3.16b, v0.4b[0]
        .inst 0x4fa0e069 // sdot v9.4s, v3.16b, v0.4b[1]
        .inst 0x4f80e86a // sdot v10.4s, v3.16b, v0.4b[2]
        .inst 0x4fa0e86b // sdot v11.4s, v3.16b, v0.4b[3]
        bne L4LoopSz_TILE_4

    L4LoopSzEnd_TILE_4:
    cbnz x8, L4Tile4Quan
    Int32ToFloat v8, v9, v10, v11
    st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x10], x4
    b Tile4End

    L4Tile4Quan:
    ld1 {v0.4s}, [x19], #16 // scale
    Int32ToFloat v8, v9, v10, v11
    MUL_SCALE v0, v8, v9, v10, v11
    FloatToInt32 v8, v9, v10, v11
    Int32ToInt16 v8, v9, v10, v11, v0, v1
    Int16ToInt8_ONE v0, v1, v16
    smax v16.16b, v6.16b, v16.16b
    smin v16.16b, v7.16b, v16.16b
    st1 {v16.16b}, [x10], x4

Tile4End:
    sub x7, x7, #4
    add x0, x0, x21, LSL #2
    add x1, x1, #16
 
TILE_1:
    cbz x7, End
    mov x10, x0
    mov x12, x2
    mov x14, x5
    mov x19, x8
    mov x20, x9
    cmp x5, #2
    blt L4LoopDz_TILE_1
L8LoopDz_TILE_1:
    ld1 {v0.4s, v1.4s}, [x20], #32 // bias
    mov x11, x1
    mov x13, x3
    mov v8.16b, v0.16b
    mov v9.16b, v1.16b
    L8LoopSz_TILE_1:
        ld1 {v3.16b}, [x12], x15 // weight
        ld1 {v0.s}[0], [x11], x22 // src
        ld1 {v4.16b}, [x12], #16 // weight
        .inst 0x4f80e068 // sdot v8.4s, v3.16b, v0.4b[0]
        subs x13, x13, #1
        sub x12, x12, x15
        .inst 0x4f80e089 // sdot v9.4s, v4.16b, v0.4b[0]
        bne L8LoopSz_TILE_1

    L8LoopSzEnd_TILE_1:
    add x12, x12, x15
    sub x14, x14, #2
    cbnz x8, L8Tile1Quan
    scvtf v8.4s, v8.4s
    scvtf v9.4s, v9.4s
    st1 {v8.4s}, [x10], x4
    st1 {v9.4s}, [x10], x4
    b L8Tile1LoopCheck

    L8Tile1Quan:
    ld1 {v0.4s, v1.4s}, [x19], #32 // scale
    scvtf v8.4s, v8.4s
    scvtf v9.4s, v9.4s
    fmul v8.4s, v8.4s, v0.4s
    fmul v9.4s, v9.4s, v1.4s
    fcvtas v8.4s, v8.4s
    fcvtas v9.4s, v9.4s
    sqxtn v0.4h, v8.4s
    sqxtn2 v0.8h, v9.4s
    sqxtn v16.8b, v0.8h
    smax v16.16b, v6.16b, v16.16b
    smin v16.16b, v7.16b, v16.16b
    st1 {v16.s}[0], [x10], x4
    st1 {v16.s}[1], [x10], x4

    L8Tile1LoopCheck:
    cmp x14, #1
    bgt L8LoopDz_TILE_1
    cbz x14, Tile1End

L4LoopDz_TILE_1:
    ld1 {v0.4s}, [x20], #16 // bias
    mov x11, x1
    mov x13, x3
    mov v8.16b, v0.16b
    L4LoopSz_TILE_1:
        ld1 {v3.16b}, [x12], #16 // weight
        ld1 {v0.s}[0], [x11], x22 // src
        subs x13, x13, #1
        .inst 0x4f80e068 // sdot v8.4s, v3.16b, v0.4b[0]
        bne L4LoopSz_TILE_1

    L4LoopSzEnd_TILE_1:
    cbnz x8, L4Tile1Quan
    scvtf v8.4s, v8.4s
    st1 {v8.4s}, [x10], x4
    b Tile1End

    L4Tile1Quan:
    ld1 {v0.4s}, [x19], #16 // scale
    scvtf v8.4s, v8.4s
    fmul v8.4s, v8.4s, v0.4s
    fcvtas v8.4s, v8.4s
    sqxtn v0.4h, v8.4s
    sqxtn v16.8b, v0.8h
    smax v16.8b, v6.8b, v16.8b
    smin v16.8b, v7.8b, v16.8b
    st1 {v16.s}[0], [x10], x4

Tile1End:
    sub x7, x7, #1
    add x0, x0, x21
    add x1, x1, #4
    b TILE_1

End:
ldp x19, x20, [sp, #(16 * 5)]
ldp x21, x22, [sp, #(16 * 4)]
ldp d8,  d9,  [sp, #(16 * 3)]
ldp d10, d11, [sp, #(16 * 2)]
ldp d12, d13, [sp, #(16 * 1)]
ldp d14, d15, [sp], #(16 * 6)
ret

#endif // __aarch64__
