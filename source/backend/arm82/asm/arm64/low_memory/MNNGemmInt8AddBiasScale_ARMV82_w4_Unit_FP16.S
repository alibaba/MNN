//
//  MNNGemmInt8AddBiasScale_ARMV82_w4_Unit_FP16.S
//  MNN
//
//  Created by MNN on 2019/12/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#if defined(__aarch64__)
#include "MNNAsmGlobal.h"

.text
.align 5

.macro ADD_BIAS_FLOAT d0, d1, d2, d3, z0
    fadd \d0\().4s, \d0\().4s, \z0\().4s
    fadd \d1\().4s, \d1\().4s, \z0\().4s
    fadd \d2\().4s, \d2\().4s, \z0\().4s
    fadd \d3\().4s, \d3\().4s, \z0\().4s
.endm

.macro SET_BIAS d0, d1, d2, d3
    movi \d0\().16b, #0
    movi \d1\().16b, #0
    movi \d2\().16b, #0
    movi \d3\().16b, #0
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
.macro MLA_WEIGHTZERO d0, s0, s1, idx // idx for xKernelSum
    fmla \d0\().4s, \s1\().4s, \s0\().s[\idx]
.endm
.macro MUL_EXTRA_SCALE s, d0, d1, d2, d3
    fmul \d0\().4s, \d0\().4s, \s\().s[0]
    fmul \d1\().4s, \d1\().4s, \s\().s[1]
    fmul \d2\().4s, \d2\().4s, \s\().s[2]
    fmul \d3\().4s, \d3\().4s, \s\().s[3]
.endm
.macro ReLU_FP16 s0, s1, s2, s3, z0, z1 // z0:min z1:max
    fmin \s0\().8h, \s0\().8h, \z1\().8h
    fmin \s1\().8h, \s1\().8h, \z1\().8h
    fmin \s2\().8h, \s2\().8h, \z1\().8h
    fmin \s3\().8h, \s3\().8h, \z1\().8h
    fmax \s0\().8h, \s0\().8h, \z0\().8h
    fmax \s1\().8h, \s1\().8h, \z0\().8h
    fmax \s2\().8h, \s2\().8h, \z0\().8h
    fmax \s3\().8h, \s3\().8h, \z0\().8h
.endm

.macro Float32ToHalf s0, s1, s2, s3, d0, d1
    fcvtn \d0\().4h,  \s0\().4s
    fcvtn2 \d0\().8h, \s1\().4s
    fcvtn \d1\().4h,  \s2\().4s
    fcvtn2 \d1\().8h, \s3\().4s
.endm

asm_function MNNGemmInt8AddBiasScale_ARMV82_w4_Unit_FP16
/* 
struct QuanPostTreatParameters {
    const float* scale;
    const float* biasFloat;
    int32_t maxValue;
    int32_t minValue;
    int32_t useInt8 = 1; // Save result as int8_t dataType; otherwise float32.
    float roundValuePos = 0.5f;
    float roundValueNeg = -0.5f;
    float* srcKernelSum;
    float* weightQuanBias;
    float* fp32minmax;
    ssize_t blockNum = 1;
    const int32_t* bias;
    const float* extraScale = nullptr;
};
*/

//void MNNGemmInt8AddBiasScale_ARMV82_w4_Unit_FP16(int8_t* dst, const int8_t* src, 
//    const int8_t* weight, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad,
// const QuanPostTreatParameters* parameters, size_t realDstCount);

//Auto: x0:dst, x1:src, x2:weight, x3:src_depth_quad, x4:dst_step
//x5:dst_depth_quad, x6: parameters, x7: realDstCount

//Load from x6: x8: scale, x9: bias, x25: xKernelSum, x26: weightQuantBias, x23: fp32minmax, x27: blockNum
ldr x8, [x6, #0]
ldr x9, [x6, #8]
//ldr w12, [x6, #16]

stp d14, d15, [sp, #(-16 * 10)]!
stp d12, d13, [sp, #(16 * 1)]
stp d10, d11, [sp, #(16 * 2)]
stp d8,  d9,  [sp, #(16 * 3)]
stp x21, x22, [sp, #(16 * 4)]
stp x19, x20, [sp, #(16 * 5)]
stp x27, x28, [sp, #(16 * 6)]
stp x25, x26, [sp, #(16 * 7)]
stp x23, x24, [sp, #(16 * 8)]

//ldr w27, [x6, #20]
ldr x25, [x6, #40]  // xKernelSum
ldr x26, [x6, #48]  // weightQuantBias
ldr x23, [x6, #56]  // fp32minmax
ldr x27, [x6, #64]  // blockNum

mov x21, #16 // sizeof(float16_t) * PACK
mul x27, x27, x3
Start:
lsl x15, x27, #3 // x15 = src_depth_quad * UNIT * SRC_UNIT * sizeof(int4_t)
mov x22, #48 // src_steps
add x24, x15, x15
ldr x27, [x6, #80] // extra scale
TILE_12:
    cmp x7, #12
    blt TILE_8

L8LoopDz_TILE_12:
    // ld1 {v0.4s, v1.4s}, [x9], #32 // bias
    mov x11, x1
    mov x13, x3
    movi v7.16b, #15

    // Init 0
    SET_BIAS v8, v9, v10, v11
    SET_BIAS v12, v13, v14, v15
    SET_BIAS v16, v17, v18, v19
    SET_BIAS v20, v21, v22, v23
    SET_BIAS v24, v25, v26, v27
    SET_BIAS v28, v29, v30, v31

    mov x28, x2
    L8LoopSz_TILE_12:
        ld1 {v3.d}[0], [x2], x15 // weight
        ld1 {v4.d}[0], [x2], #8
        ld1 {v0.16b, v1.16b, v2.16b}, [x11], #48 // src
        // int4->int8
        ushr v5.16b, v3.16b, #4
        and v6.16b, v3.16b, v7.16b
        zip1 v3.16b, v5.16b, v6.16b

        .inst 0x4f80e068 // sdot v8.4s, v3.16b, v0.4b[0]
        .inst 0x4fa0e069 // sdot v9.4s, v3.16b, v0.4b[1]
        .inst 0x4f80e86a // sdot v10.4s, v3.16b, v0.4b[2]
        .inst 0x4fa0e86b // sdot v11.4s, v3.16b, v0.4b[3]

        .inst 0x4f81e06c // sdot v12.4s, v3.16b, v1.4b[0]
        .inst 0x4fa1e06d // sdot v13.4s, v3.16b, v1.4b[1]
        .inst 0x4f81e86e // sdot v14.4s, v3.16b, v1.4b[2]
        .inst 0x4fa1e86f // sdot v15.4s, v3.16b, v1.4b[3]
        // int4->int8
        ushr v5.16b, v4.16b, #4
        and v6.16b, v4.16b, v7.16b
        zip1 v4.16b, v5.16b, v6.16b

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
    add x2, x28, x24
    sub x5, x5, #1

    L8Tile12Quan:
    ld1 {v0.4s, v1.4s}, [x8], #32 // scale
    ld1 {v2.4s, v3.4s, v4.4s}, [x25] // x kernel sum
    ld1 {v5.4s, v6.4s}, [x26], #32 // weight quan zeropoint
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

    cbz x27, TILE12_L8_MLA_TERM
    ld1 {v0.4s, v1.4s}, [x27], #32
    ld1 {v7.4s}, [x27]
    MUL_EXTRA_SCALE v0, v8, v9, v10, v11
    MUL_EXTRA_SCALE v1, v12, v13, v14, v15
    MUL_EXTRA_SCALE v7, v16, v17, v18, v19
    MUL_EXTRA_SCALE v0, v20, v21, v22, v23
    MUL_EXTRA_SCALE v1, v24, v25, v26, v27
    MUL_EXTRA_SCALE v7, v28, v29, v30, v31
    sub x27, x27, #32

    TILE12_L8_MLA_TERM:
    MLA_WEIGHTZERO v8,  v2, v5, 0 // tile:0, oc:0-3
    MLA_WEIGHTZERO v9,  v2, v5, 1 // tile:1, oc:0-3
    MLA_WEIGHTZERO v10, v2, v5, 2 // tile:2, oc:0-3
    MLA_WEIGHTZERO v11, v2, v5, 3 // tile:3, oc:0-3
    MLA_WEIGHTZERO v12, v3, v5, 0 // tile:4, oc:0-3
    MLA_WEIGHTZERO v13, v3, v5, 1 // tile:5, oc:0-3
    MLA_WEIGHTZERO v14, v3, v5, 2 // tile:6, oc:0-3
    MLA_WEIGHTZERO v15, v3, v5, 3 // tile:7, oc:0-3
    MLA_WEIGHTZERO v16, v4, v5, 0 // tile:8, oc:0-3
    MLA_WEIGHTZERO v17, v4, v5, 1 // tile:9, oc:0-3
    MLA_WEIGHTZERO v18, v4, v5, 2 // tile:10, oc:0-3 
    MLA_WEIGHTZERO v19, v4, v5, 3 // tile:11, oc:0-3

    //ld1r {v0.4s}, [x23] // f32 min
    //ld1r {v1.4s}, [x24] // f32 max
    MLA_WEIGHTZERO v20, v2, v6, 0 // tile:0, oc:4-7
    MLA_WEIGHTZERO v21, v2, v6, 1 // tile:1, oc:4-7
    MLA_WEIGHTZERO v22, v2, v6, 2 // tile:2, oc:4-7
    MLA_WEIGHTZERO v23, v2, v6, 3 // tile:3, oc:4-7
    MLA_WEIGHTZERO v24, v3, v6, 0 // tile:4, oc:4-7
    MLA_WEIGHTZERO v25, v3, v6, 1 // tile:5, oc:4-7
    MLA_WEIGHTZERO v26, v3, v6, 2 // tile:6, oc:4-7 
    MLA_WEIGHTZERO v27, v3, v6, 3 // tile:7, oc:4-7
    MLA_WEIGHTZERO v28, v4, v6, 0 // tile:8, oc:4-7
    MLA_WEIGHTZERO v29, v4, v6, 1 // tile:9, oc:4-7
    MLA_WEIGHTZERO v30, v4, v6, 2 // tile:10, oc:4-7
    MLA_WEIGHTZERO v31, v4, v6, 3 // tile:11, oc:4-7
    sub x4, x4, #128

    cbz x9, TILE12_ADD_DSTV
    TILE12_ADD_BIAS:
    ld1 {v0.4s, v1.4s}, [x9], #32
    ADD_BIAS_FLOAT v8, v9, v10, v11, v0
    ADD_BIAS_FLOAT v12, v13, v14, v15, v0
    ADD_BIAS_FLOAT v16, v17, v18, v19, v0
    ADD_BIAS_FLOAT v20, v21, v22, v23, v1
    ADD_BIAS_FLOAT v24, v25, v26, v27, v1
    ADD_BIAS_FLOAT v28, v29, v30, v31, v1

    Float32ToHalf v8, v20, v9, v21, v0, v1
    Float32ToHalf v10, v22, v11, v23, v2, v3
    Float32ToHalf v12, v24, v13, v25, v4, v5
    Float32ToHalf v14, v26, v15, v27, v6, v7
    Float32ToHalf v16, v28, v17, v29, v8, v9
    Float32ToHalf v18, v30, v19, v31, v10, v11
    b TILE12_POST

    TILE12_ADD_DSTV:
    Float32ToHalf v8, v20, v9, v21, v0, v1
    Float32ToHalf v10, v22, v11, v23, v2, v3
    Float32ToHalf v12, v24, v13, v25, v4, v5
    Float32ToHalf v14, v26, v15, v27, v6, v7
    Float32ToHalf v16, v28, v17, v29, v8, v9
    Float32ToHalf v18, v30, v19, v31, v10, v11
    ld1 {v20.8h, v21.8h, v22.8h, v23.8h}, [x0], #64
    ld1 {v12.8h, v13.8h, v14.8h, v15.8h}, [x0], #64
    ld1 {v16.8h, v17.8h, v18.8h, v19.8h}, [x0]
    fadd v0.8h, v0.8h, v20.8h
    fadd v1.8h, v1.8h, v21.8h
    fadd v2.8h, v2.8h, v22.8h
    fadd v3.8h, v3.8h, v23.8h
    fadd v4.8h, v4.8h, v12.8h
    fadd v5.8h, v5.8h, v13.8h
    fadd v6.8h, v6.8h, v14.8h
    fadd v7.8h, v7.8h, v15.8h
    fadd v8.8h, v8.8h, v16.8h
    fadd v9.8h, v9.8h, v17.8h
    fadd v10.8h, v10.8h, v18.8h
    fadd v11.8h, v11.8h, v19.8h
    sub x0, x0, #128

    TILE12_POST:
    cbz x23, TILE12_STORE
    ld1r {v24.8h}, [x23], #2 // f32 min
    ld1r {v25.8h}, [x23] // f32 max

    ReLU_FP16 v0, v1, v2, v3, v24, v25
    ReLU_FP16 v4, v5, v6, v7, v24, v25
    ReLU_FP16 v8, v9, v10, v11, v24, v25   
    sub x23, x23, #2

    TILE12_STORE:

    st1 {v0.8h, v1.8h, v2.8h, v3.8h}, [x0], #64
    st1 {v4.8h, v5.8h, v6.8h, v7.8h}, [x0], #64
    st1 {v8.8h, v9.8h, v10.8h, v11.8h}, [x0], x4
    add x4, x4, #128
    L8Tile12LoopCheck:
    cmp x5, #1
    bge L8LoopDz_TILE_12
    blt End

TILE_8:
    cmp x7, #8
    blt TILE_4
    mov x10, x0
    mov x12, x2
    mov x14, x5
    mov x19, x8 // scale
    mov x20, x9 // bias
    mov x6, x26 // weightQuantBias

L8LoopDz_TILE_8:
    mov x11, x1
    mov x13, x3
    movi v7.16b, #15

    SET_BIAS v8, v9, v10, v11
    SET_BIAS v12, v13, v14, v15
    SET_BIAS v16, v17, v18, v19
    SET_BIAS v20, v21, v22, v23
    mov x28, x12
    L8LoopSz_TILE_8:
        ld1 {v3.d}[0], [x12], x15 // weight
        ld1 {v4.d}[0], [x12], #8
        ld1 {v0.16b, v1.16b}, [x11], x22 // src
        // int4->int8
        ushr v5.16b, v3.16b, #4
        and v6.16b, v3.16b, v7.16b
        zip1 v3.16b, v5.16b, v6.16b

        .inst 0x4f80e068 // sdot v8.4s, v3.16b, v0.4b[0]
        .inst 0x4fa0e069 // sdot v9.4s, v3.16b, v0.4b[1]
        .inst 0x4f80e86a // sdot v10.4s, v3.16b, v0.4b[2]
        .inst 0x4fa0e86b // sdot v11.4s, v3.16b, v0.4b[3]
        // int4->int8
        ushr v5.16b, v4.16b, #4
        and v6.16b, v4.16b, v7.16b
        zip1 v4.16b, v5.16b, v6.16b
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
    add x12, x28, x24
    sub x14, x14, #1

    L8Tile8Quan:
    ld1 {v0.4s, v1.4s}, [x19], #32 // scale
    ld1 {v2.4s, v3.4s}, [x25] // x kernel sum
    ld1 {v24.4s, v25.4s}, [x6], #32 // weight quan zeropoint
    Int32ToFloat v8, v9, v10, v11
    Int32ToFloat v12, v13, v14, v15
    Int32ToFloat v16, v17, v18, v19
    Int32ToFloat v20, v21, v22, v23
    MUL_SCALE v0, v8, v9, v10, v11
    MUL_SCALE v0, v12, v13, v14, v15
    MUL_SCALE v1, v16, v17, v18, v19
    MUL_SCALE v1, v20, v21, v22, v23

    cbz x27, TILE8_L8_MLA_TERM
    ld1 {v4.4s, v5.4s}, [x27]
    MUL_EXTRA_SCALE v4, v8, v9, v10, v11
    MUL_EXTRA_SCALE v5, v12, v13, v14, v15
    MUL_EXTRA_SCALE v4, v16, v17, v18, v19
    MUL_EXTRA_SCALE v5, v20, v21, v22, v23

    TILE8_L8_MLA_TERM:
    MLA_WEIGHTZERO v8,  v2, v24, 0 // tile:0, oc:0-3
    MLA_WEIGHTZERO v9,  v2, v24, 1 // tile:1, oc:0-3
    MLA_WEIGHTZERO v10, v2, v24, 2 // tile:2, oc:0-3
    MLA_WEIGHTZERO v11, v2, v24, 3 // tile:3, oc:0-3
    MLA_WEIGHTZERO v12, v3, v24, 0 // tile:4, oc:0-3
    MLA_WEIGHTZERO v13, v3, v24, 1 // tile:5, oc:0-3
    MLA_WEIGHTZERO v14, v3, v24, 2 // tile:6, oc:0-3
    MLA_WEIGHTZERO v15, v3, v24, 3 // tile:7, oc:0-3
    MLA_WEIGHTZERO v16, v2, v25, 0 // tile:0, oc:4-7
    MLA_WEIGHTZERO v17, v2, v25, 1 // tile:1, oc:4-7
    MLA_WEIGHTZERO v18, v2, v25, 2 // tile:2, oc:4-7
    MLA_WEIGHTZERO v19, v2, v25, 3 // tile:3, oc:4-7
    MLA_WEIGHTZERO v20, v3, v25, 0 // tile:4, oc:4-7
    MLA_WEIGHTZERO v21, v3, v25, 1 // tile:5, oc:4-7
    MLA_WEIGHTZERO v22, v3, v25, 2 // tile:6, oc:4-7
    MLA_WEIGHTZERO v23, v3, v25, 3 // tile:7, oc:4-7

    sub x4, x4, #64

    cbz x9, TILE8_ADD_DSTV
    TILE8_ADD_BIAS:
    ld1 {v0.4s, v1.4s}, [x20], #32
    ADD_BIAS_FLOAT v8, v9, v10, v11, v0
    ADD_BIAS_FLOAT v12, v13, v14, v15, v0
    ADD_BIAS_FLOAT v16, v17, v18, v19, v1
    ADD_BIAS_FLOAT v20, v21, v22, v23, v1

    Float32ToHalf v8, v16, v9, v17, v0, v1
    Float32ToHalf v10, v18, v11, v19, v2, v3
    Float32ToHalf v12, v20, v13, v21, v4, v5
    Float32ToHalf v14, v22, v15, v23, v6, v7
    b TILE8_POST

    TILE8_ADD_DSTV:
    Float32ToHalf v8, v16, v9, v17, v0, v1
    Float32ToHalf v10, v18, v11, v19, v2, v3
    Float32ToHalf v12, v20, v13, v21, v4, v5
    Float32ToHalf v14, v22, v15, v23, v6, v7
    ld1 {v24.8h, v25.8h, v26.8h, v27.8h}, [x10], #64
    ld1 {v28.8h, v29.8h, v30.8h, v31.8h}, [x10]
    fadd v0.8h, v0.8h, v24.8h
    fadd v1.8h, v1.8h, v25.8h
    fadd v2.8h, v2.8h, v26.8h
    fadd v3.8h, v3.8h, v27.8h
    fadd v4.8h, v4.8h, v28.8h
    fadd v5.8h, v5.8h, v29.8h
    fadd v6.8h, v6.8h, v30.8h
    fadd v7.8h, v7.8h, v31.8h
    sub x10, x10, #64

    TILE8_POST:
    cbz x23, TILE8_STORE
    ld1r {v24.8h}, [x23], #2 // f16 min
    ld1r {v25.8h}, [x23] // f16 max
    ReLU_FP16 v0, v1, v2, v3, v24, v25
    ReLU_FP16 v4, v5, v6, v7, v24, v25
    sub x23, x23, #2

    TILE8_STORE:
    st1 {v0.8h, v1.8h, v2.8h, v3.8h}, [x10], #64
    st1 {v4.8h, v5.8h, v6.8h, v7.8h}, [x10], x4

    //st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x10], #64
    //st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [x10], x4
    //st1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x10], #64
    //st1 {v20.4s, v21.4s, v22.4s, v23.4s}, [x10], x4
    add x4, x4, #64

    L8Tile8LoopCheck:
    cmp x14, #1
    bge L8LoopDz_TILE_8

cbz x27, Tile8End
add x27, x27, #32
Tile8End:
    sub x7, x7, #8
    add x0, x0, x21, LSL #3
    add x1, x1, #32
    add x25, x25, #32

TILE_4:
    movi v7.16b, #15
    cmp x7, #4
    blt TILE_1
    mov x10, x0
    mov x12, x2
    mov x14, x5
    mov x19, x8
    mov x20, x9
    mov x6, x26 // weightQuantBias
L8LoopDz_TILE_4:
    mov x11, x1
    mov x13, x3

    SET_BIAS v8, v9, v10, v11
    SET_BIAS v12, v13, v14, v15

    mov x28, x12
    L8LoopSz_TILE_4:
        ld1 {v3.d}[0], [x12], x15 // weight
        ld1 {v0.16b}, [x11], x22 // src
        ld1 {v4.d}[0], [x12], #8 // weight
        // int4->int8
        ushr v5.16b, v3.16b, #4
        and v6.16b, v3.16b, v7.16b
        zip1 v3.16b, v5.16b, v6.16b

        .inst 0x4f80e068 // sdot v8.4s, v3.16b, v0.4b[0]
        .inst 0x4fa0e069 // sdot v9.4s, v3.16b, v0.4b[1]
        .inst 0x4f80e86a // sdot v10.4s, v3.16b, v0.4b[2]
        .inst 0x4fa0e86b // sdot v11.4s, v3.16b, v0.4b[3]
        // int4->int8
        ushr v5.16b, v4.16b, #4
        and v6.16b, v4.16b, v7.16b
        zip1 v4.16b, v5.16b, v6.16b
        subs x13, x13, #1
        sub x12, x12, x15
        .inst 0x4f80e08c // sdot v12.4s, v4.16b, v0.4b[0]
        .inst 0x4fa0e08d // sdot v13.4s, v4.16b, v0.4b[1]
        .inst 0x4f80e88e // sdot v14.4s, v4.16b, v0.4b[2]
        .inst 0x4fa0e88f // sdot v15.4s, v4.16b, v0.4b[3]
        bne L8LoopSz_TILE_4

    L8LoopSzEnd_TILE_4:
    add x12, x28, x24
    sub x14, x14, #1

    L8Tile4Quan:
    ld1 {v0.4s, v1.4s}, [x19], #32 // scale
    ld1 {v2.4s}, [x25] // x kernel sum
    ld1 {v24.4s, v25.4s}, [x6], #32 // weight quan zeropoint
    Int32ToFloat v8, v9, v10, v11
    Int32ToFloat v12, v13, v14, v15
    MUL_SCALE v0, v8, v9, v10, v11
    MUL_SCALE v1, v12, v13, v14, v15

    cbz x27, TILE4_L8_MLA_TERM
    ld1 {v4.4s}, [x27]
    MUL_EXTRA_SCALE v4, v8, v9, v10, v11
    MUL_EXTRA_SCALE v4, v12, v13, v14, v15

    TILE4_L8_MLA_TERM:
    MLA_WEIGHTZERO v8, v2, v24, 0 // tile:0, oc:0-3
    MLA_WEIGHTZERO v9, v2, v24, 1 // tile:1, oc:0-3
    MLA_WEIGHTZERO v10, v2, v24, 2 // tile:2, oc:0-3
    MLA_WEIGHTZERO v11, v2, v24, 3 // tile:3, oc:0-3
    MLA_WEIGHTZERO v12, v2, v25, 0 // tile:0, oc:4-7
    MLA_WEIGHTZERO v13, v2, v25, 1 // tile:1, oc:4-7
    MLA_WEIGHTZERO v14, v2, v25, 2 // tile:2, oc:4-7
    MLA_WEIGHTZERO v15, v2, v25, 3 // tile:3, oc:4-7

    cbz x9, TILE4_ADD_DSTV
    TILE4_ADD_BIAS:
    ld1 {v0.4s, v1.4s}, [x20], #32
    ADD_BIAS_FLOAT v8, v9, v10, v11, v0
    ADD_BIAS_FLOAT v12, v13, v14, v15, v1
    Float32ToHalf v8, v12, v9, v13, v0, v1
    Float32ToHalf v10, v14, v11, v15, v2, v3
    b TILE4_POST

    TILE4_ADD_DSTV:
    Float32ToHalf v8, v12, v9, v13, v0, v1
    Float32ToHalf v10, v14, v11, v15, v2, v3
    ld1 {v20.8h, v21.8h, v22.8h, v23.8h}, [x10]
    fadd v0.8h, v0.8h, v20.8h
    fadd v1.8h, v1.8h, v21.8h
    fadd v2.8h, v2.8h, v22.8h
    fadd v3.8h, v3.8h, v23.8h

    TILE4_POST:
    cbz x23, TILE4_STORE
    ld1r {v24.8h}, [x23], #2 // f16 min
    ld1r {v25.8h}, [x23] // f16 max
    sub x23, x23, #2
    ReLU_FP16 v0, v1, v2, v3, v24, v25
    //st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x10], x4
    //st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [x10], x4
    

    TILE4_STORE:
    st1 {v0.8h, v1.8h, v2.8h, v3.8h}, [x10], x4

    L8Tile4LoopCheck:
    cmp x14, #1
    bge L8LoopDz_TILE_4
cbz x27, Tile4End
add x27, x27, #16
Tile4End:
    sub x7, x7, #4
    add x0, x0, x21, LSL #2
    add x1, x1, #16
    add x25, x25, #16
 
TILE_1:
    // Already execute: [movi v7.16b, #15] in TILE_4 
    cbz x7, End
    mov x10, x0
    mov x12, x2
    mov x14, x5
    mov x19, x8
    mov x20, x9
    mov x6, x26 // weightQuantBias

L8LoopDz_TILE_1:
    mov x11, x1
    mov x13, x3

    movi v8.16b, #0
    movi v9.16b, #0
    mov x28, x12
    L8LoopSz_TILE_1:
        ld1 {v3.d}[0], [x12], x15 // weight
        ld1 {v0.s}[0], [x11], x22 // src
        ld1 {v4.d}[0], [x12], #8 // weight
        // int4->int8
        ushr v5.16b, v3.16b, #4
        and v6.16b, v3.16b, v7.16b
        zip1 v3.16b, v5.16b, v6.16b

        .inst 0x4f80e068 // sdot v8.4s, v3.16b, v0.4b[0]
        subs x13, x13, #1
        // int4->int8
        ushr v5.16b, v4.16b, #4
        and v6.16b, v4.16b, v7.16b
        zip1 v4.16b, v5.16b, v6.16b
        sub x12, x12, x15

        .inst 0x4f80e089 // sdot v9.4s, v4.16b, v0.4b[0]
        bne L8LoopSz_TILE_1

    L8LoopSzEnd_TILE_1:
    add x12, x28, x24
    sub x14, x14, #1

    L8Tile1Quan:
    ld1 {v0.4s, v1.4s}, [x19], #32 // scale
    ld1 {v2.s}[0], [x25] // x kernel sum
    ld1 {v24.4s, v25.4s}, [x6], #32 // weight quan zeropoint
    scvtf v8.4s, v8.4s
    scvtf v9.4s, v9.4s
    fmul v8.4s, v8.4s, v0.4s
    fmul v9.4s, v9.4s, v1.4s

    cbz x27, TILE1_L8_MLA_TERM
    ld1 {v4.s}[0], [x27]
    fmul v8.4s, v8.4s, v4.s[0]
    fmul v9.4s, v9.4s, v4.s[0]

    TILE1_L8_MLA_TERM:
    MLA_WEIGHTZERO v8, v2, v24, 0 // tile:0, oc:0-3
    MLA_WEIGHTZERO v9, v2, v25, 0 // tile:0, oc:4-7

    cbz x9, TILE1_ADD_DSTV
    TILE1_ADD_BIAS:
    ld1 {v20.4s, v21.4s}, [x20], #32
    fadd v8.4s, v8.4s, v20.4s
    fadd v9.4s, v9.4s, v21.4s
    fcvtn v0.4h, v8.4s
    fcvtn2 v0.8h, v9.4s
    b TILE1_POST

    TILE1_ADD_DSTV:
    fcvtn v0.4h, v8.4s
    fcvtn2 v0.8h, v9.4s
    ld1 {v3.8h}, [x10]
    fadd v0.8h, v0.8h, v3.8h

    TILE1_POST:
    cbz x23, TILE1_STORE
    ld1r {v24.8h}, [x23], #2 // f16 min
    ld1r {v25.8h}, [x23] // f16 max
    sub x23, x23, #2
    fmax v0.8h, v24.8h, v0.8h
    fmin v0.8h, v25.8h, v0.8h
    // st1 {v8.4s}, [x10], x4
    // st1 {v9.4s}, [x10], x4
    TILE1_STORE:
    st1 {v0.8h}, [x10], x4

    L8Tile1LoopCheck:
    cmp x14, #1
    bge L8LoopDz_TILE_1
cbz x27, Tile1End
add x27, x27, #4
Tile1End:
    sub x7, x7, #1
    add x0, x0, x21
    add x1, x1, #4
    add x25, x25, #4
    b TILE_1

End:
ldp x23, x24, [sp, #(16 * 8)]
ldp x25, x26, [sp, #(16 * 7)]
ldp x27, x28, [sp, #(16 * 6)]
ldp x19, x20, [sp, #(16 * 5)]
ldp x21, x22, [sp, #(16 * 4)]
ldp d8,  d9,  [sp, #(16 * 3)]
ldp d10, d11, [sp, #(16 * 2)]
ldp d12, d13, [sp, #(16 * 1)]
ldp d14, d15, [sp], #(16 * 10)
ret

#endif // __aarch64__
