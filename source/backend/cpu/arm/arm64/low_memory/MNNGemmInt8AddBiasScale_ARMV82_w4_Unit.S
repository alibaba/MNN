//
//  MNNGemmInt8AddBiasScale_ARMV82_w4_Unit.S
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

.macro ADD_FLOAT d0, d1, d2, d3, s0, s1, s2, s3
    fadd \d0\().4s, \d0\().4s, \s0\().4s
    fadd \d1\().4s, \d1\().4s, \s1\().4s
    fadd \d2\().4s, \d2\().4s, \s2\().4s
    fadd \d3\().4s, \d3\().4s, \s3\().4s
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
.macro MUL_EXTRA_SCALE s, d0, d1, d2, d3
    fmul \d0\().4s, \d0\().4s, \s\().s[0]
    fmul \d1\().4s, \d1\().4s, \s\().s[1]
    fmul \d2\().4s, \d2\().4s, \s\().s[2]
    fmul \d3\().4s, \d3\().4s, \s\().s[3]
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
.macro MLA_WEIGHTZERO d0, s0, s1, idx // idx for xKernelSum
    fmla \d0\().4s, \s1\().4s, \s0\().s[\idx]
.endm
.macro ReLU_FP32 s0, s1, s2, s3, z0, z1 // z0:min z1:max
    fmin \s0\().4s, \s0\().4s, \z1\().4s
    fmin \s1\().4s, \s1\().4s, \z1\().4s
    fmin \s2\().4s, \s2\().4s, \z1\().4s
    fmin \s3\().4s, \s3\().4s, \z1\().4s
    fmax \s0\().4s, \s0\().4s, \z0\().4s
    fmax \s1\().4s, \s1\().4s, \z0\().4s
    fmax \s2\().4s, \s2\().4s, \z0\().4s
    fmax \s3\().4s, \s3\().4s, \z0\().4s
.endm

.macro REVERT_INPUT_DEQUANT_BIAS rg0, rg1, rg2, rg3
mul \rg1, \rg2, \rg3
sub \rg0, \rg0, \rg1
.endm

.macro REVERT_WEIGHT_KERNEL_SUM rg0, rg1, rg2, rg3
// y=UP_DIV(ocDiv4,(hp/pack))
add \rg1, \rg3, #1
lsr \rg1, \rg1, #1
// blockNum * y * (hp * sizeof(float))
mul \rg1, \rg2, \rg1
sub \rg0, \rg0, \rg1, LSL #5 // revert weight kernel sum
.endm

asm_function MNNGemmInt8AddBiasScale_ARMV82_w4_Unit
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
    float* weightKernelSum;
    float* fp32minmax;
    ssize_t blockNum = 1;
    const int32_t* bias = nullptr;
    const float* inputScale = nullptr;
    const float* inputBias = nullptr;
    float* accumBuffer = nullptr;
};
*/

//void MNNGemmInt8AddBiasScale_ARMV82_w4_Unit(int8_t* dst, const int8_t* src, 
//    const int8_t* weight, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad,
// const QuanPostTreatParameters* parameters, size_t realDstCount);

//Auto: x0:dst, x1:src, x2:weight, x3:src_depth_quad, x4:dst_step
//x5:dst_depth_quad, x6: parameters, x7: realDstCount

//Load from x6: x9: bias, x8: xKernelSum, x23: fp32minmax
ldr x9, [x6, #8]

stp d14, d15, [sp, #(-16 * 10)]!
stp d12, d13, [sp, #(16 * 1)]
stp d10, d11, [sp, #(16 * 2)]
stp d8,  d9,  [sp, #(16 * 3)]
stp x21, x22, [sp, #(16 * 4)]
stp x19, x20, [sp, #(16 * 5)]
stp x23, x24, [sp, #(16 * 6)]
stp x25, x26, [sp, #(16 * 7)]
stp x27, x28, [sp, #(16 * 8)]

ldr x8, [x6, #40]  // srcKernelSum
ldr x28, [x6, #48] // weightKernelSum
ldr x24, [x6, #80]  // inputScale
ldr x27, [x6, #88]  // inputBias
ldr x10, [x6, #96]  // accumBuffer

ldr x23, [x6, #56]  // fp32minmax
ldr x26, [x6, #64]  // blockNum
lsl x22, x7, #2 // eDest * SRC_UNIT

mov x14, #-32
cbz x27, TILE_12
mov x14, #16

TILE_12:
    cmp x7, #12
    blt TILE_8
    sub x4, x4, #128
    mov x20, x9
    mov x15, x8 // input kernel sum
    mov x6, x27 // input dequant bias
    mov x21, x24 // input dequant scale
    mov x12, #-320

    cmp x5, #2
    blt L4LoopDz_TILE_12

L8LoopDz_TILE_12:
    mov x11, x1
    mov x19, #0
TILE12_BLOCKNUM:
    mov x13, x3
    movi v7.16b, #15

    SET_BIAS v8, v9, v10, v11
    SET_BIAS v12, v13, v14, v15
    SET_BIAS v16, v17, v18, v19
    SET_BIAS v20, v21, v22, v23
    SET_BIAS v24, v25, v26, v27
    SET_BIAS v28, v29, v30, v31

    L8LoopSz_TILE_12:
        ld1 {v5.16b}, [x2], #16 // weight
        ld1 {v0.16b, v1.16b, v2.16b}, [x11], #48 // src
        // int4->int8
        ushr v3.16b, v5.16b, #4
        and v4.16b, v5.16b, v7.16b

        .inst 0x4f80e068 // sdot v8.4s, v3.16b, v0.4b[0]
        .inst 0x4fa0e069 // sdot v9.4s, v3.16b, v0.4b[1]
        .inst 0x4f80e86a // sdot v10.4s, v3.16b, v0.4b[2]
        .inst 0x4fa0e86b // sdot v11.4s, v3.16b, v0.4b[3]

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

    L8Tile12Quan:
    ld1 {v0.4s, v1.4s}, [x2], #32 // scale
    ld1 {v2.4s, v3.4s, v4.4s}, [x8], #48 // input kernel sum
    ld1 {v5.4s, v6.4s}, [x2], #32 // weight quan zeropoint
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

    ld1 {v0.4s, v1.4s}, [x24], #32
    ld1 {v7.4s}, [x24], x14
    MUL_EXTRA_SCALE v0, v8, v9, v10, v11
    MUL_EXTRA_SCALE v1, v12, v13, v14, v15
    MUL_EXTRA_SCALE v7, v16, v17, v18, v19
    MUL_EXTRA_SCALE v0, v20, v21, v22, v23
    MUL_EXTRA_SCALE v1, v24, v25, v26, v27
    MUL_EXTRA_SCALE v7, v28, v29, v30, v31

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

    cbz x27, TILE12_ADD_DSTV
    ld1 {v0.4s, v1.4s, v2.4s}, [x27], #48 // input dequant bias
    ld1 {v3.4s, v4.4s}, [x28], #32 // weight kernel sum
    MLA_WEIGHTZERO v8, v0, v3, 0
    MLA_WEIGHTZERO v9, v0, v3, 1
    MLA_WEIGHTZERO v10, v0, v3, 2
    MLA_WEIGHTZERO v11, v0, v3, 3
    MLA_WEIGHTZERO v12, v1, v3, 0
    MLA_WEIGHTZERO v13, v1, v3, 1
    MLA_WEIGHTZERO v14, v1, v3, 2
    MLA_WEIGHTZERO v15, v1, v3, 3
    MLA_WEIGHTZERO v16, v2, v3, 0
    MLA_WEIGHTZERO v17, v2, v3, 1
    MLA_WEIGHTZERO v18, v2, v3, 2
    MLA_WEIGHTZERO v19, v2, v3, 3

    MLA_WEIGHTZERO v20, v0, v4, 0 // tile:0, oc:4-7
    MLA_WEIGHTZERO v21, v0, v4, 1 // tile:1, oc:4-7
    MLA_WEIGHTZERO v22, v0, v4, 2 // tile:2, oc:4-7
    MLA_WEIGHTZERO v23, v0, v4, 3 // tile:3, oc:4-7
    MLA_WEIGHTZERO v24, v1, v4, 0 // tile:4, oc:4-7
    MLA_WEIGHTZERO v25, v1, v4, 1 // tile:5, oc:4-7
    MLA_WEIGHTZERO v26, v1, v4, 2 // tile:6, oc:4-7 
    MLA_WEIGHTZERO v27, v1, v4, 3 // tile:7, oc:4-7
    MLA_WEIGHTZERO v28, v2, v4, 0 // tile:8, oc:4-7
    MLA_WEIGHTZERO v29, v2, v4, 1 // tile:9, oc:4-7
    MLA_WEIGHTZERO v30, v2, v4, 2 // tile:10, oc:4-7
    MLA_WEIGHTZERO v31, v2, v4, 3 // tile:11, oc:4-7

    TILE12_ADD_DSTV:
    cbz x19, TILE12_L8_ACCUM_BUFFER // x19=0: first block, do not add previous block result
    ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x10], #64
    ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x10], #64
    ADD_FLOAT v8, v9, v10, v11, v0, v1, v2, v3
    ADD_FLOAT v12, v13, v14, v15, v4, v5, v6, v7
    ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x10], #64
    ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x10], #64
    ADD_FLOAT v16, v17, v18, v19, v0, v1, v2, v3
    ADD_FLOAT v20, v21, v22, v23, v4, v5, v6, v7
    ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x10], #64
    ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x10], x12
    ADD_FLOAT v24, v25, v26, v27, v0, v1, v2, v3
    ADD_FLOAT v28, v29, v30, v31, v4, v5, v6, v7

    TILE12_L8_ACCUM_BUFFER:
    add x19, x19, #1
    cmp x19, x26
    beq TILE12_POST
    st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x10], #64
    st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [x10], #64
    st1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x10], #64
    st1 {v20.4s, v21.4s, v22.4s, v23.4s}, [x10], #64
    st1 {v24.4s, v25.4s, v26.4s, v27.4s}, [x10], #64
    st1 {v28.4s, v29.4s, v30.4s, v31.4s}, [x10], x12
    b TILE12_BLOCKNUM

    TILE12_POST:
    sub x5, x5, #2
    cbz x9, TILE12_RELU
    ld1 {v0.4s, v1.4s}, [x20], #32
    ADD_BIAS_FLOAT v8, v9, v10, v11, v0
    ADD_BIAS_FLOAT v12, v13, v14, v15, v0
    ADD_BIAS_FLOAT v16, v17, v18, v19, v0
    ADD_BIAS_FLOAT v20, v21, v22, v23, v1
    ADD_BIAS_FLOAT v24, v25, v26, v27, v1
    ADD_BIAS_FLOAT v28, v29, v30, v31, v1

    TILE12_RELU:
    cbz x23, TILE12_STORE
    ld1r {v0.4s}, [x23], #4 // f32 min
    ld1r {v1.4s}, [x23] // f32 max
    ReLU_FP32 v8, v9, v10, v11, v0, v1
    ReLU_FP32 v12, v13, v14, v15, v0, v1
    ReLU_FP32 v16, v17, v18, v19, v0, v1
    ReLU_FP32 v20, v21, v22, v23, v0, v1
    ReLU_FP32 v24, v25, v26, v27, v0, v1
    ReLU_FP32 v28, v29, v30, v31, v0, v1
    sub x23, x23, #4

    TILE12_STORE:
    st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x0], #64
    st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [x0], #64
    st1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x0], x4
    st1 {v20.4s, v21.4s, v22.4s, v23.4s}, [x0], #64
    st1 {v24.4s, v25.4s, v26.4s, v27.4s}, [x0], #64
    st1 {v28.4s, v29.4s, v30.4s, v31.4s}, [x0], x4

    L8Tile12LoopCheck:
    cbz x5, End
    cmp x5, #2
    mov x8, x15 // revert input kernel sum
    mov x24, x21 // revert input dequant scale
    mov x27, x6 // revert input dequant bias
    bge L8LoopDz_TILE_12

L4LoopDz_TILE_12:
    mov x11, x1
    mov x19, #0
L4_TILE12_BLOCKNUM:
    movi v7.16b, #15
    mov x13, x3

    SET_BIAS v8, v9, v10, v11
    SET_BIAS v12, v13, v14, v15
    SET_BIAS v16, v17, v18, v19

    L4_LoopSz_TILE_12:
        ld1 {v5.16b}, [x2], #16 // weight
        ld1 {v0.16b, v1.16b, v2.16b}, [x1], #48 // src
        // int4->int8
        ushr v3.16b, v5.16b, #4

        .inst 0x4f80e068 // sdot v8.4s, v3.16b, v0.4b[0]
        .inst 0x4fa0e069 // sdot v9.4s, v3.16b, v0.4b[1]
        .inst 0x4f80e86a // sdot v10.4s, v3.16b, v0.4b[2]
        .inst 0x4fa0e86b // sdot v11.4s, v3.16b, v0.4b[3]
        .inst 0x4f81e06c // sdot v12.4s, v3.16b, v1.4b[0]
        .inst 0x4fa1e06d // sdot v13.4s, v3.16b, v1.4b[1]
        .inst 0x4f81e86e // sdot v14.4s, v3.16b, v1.4b[2]
        .inst 0x4fa1e86f // sdot v15.4s, v3.16b, v1.4b[3]
        subs x13, x13, #1
        .inst 0x4f82e070 // sdot v16.4s, v3.16b, v2.4b[0]
        .inst 0x4fa2e071 // sdot v17.4s, v3.16b, v2.4b[1]
        .inst 0x4f82e872 // sdot v18.4s, v3.16b, v2.4b[2]
        .inst 0x4fa2e873 // sdot v19.4s, v3.16b, v2.4b[3]
        bne L4_LoopSz_TILE_12


    L4_Tile12Quan:
    ld1 {v0.4s}, [x2] // scale
    add x2, x2, #32
    ld1 {v2.4s, v3.4s, v4.4s}, [x8], #48 // x kernel sum
    ld1 {v5.4s}, [x2] // weight quan zeropoint
    add x2, x2, #32
    Int32ToFloat v8, v9, v10, v11
    Int32ToFloat v12, v13, v14, v15
    Int32ToFloat v16, v17, v18, v19
    MUL_SCALE v0, v8, v9, v10, v11
    MUL_SCALE v0, v12, v13, v14, v15
    MUL_SCALE v0, v16, v17, v18, v19

    ld1 {v0.4s, v1.4s}, [x24], #32
    ld1 {v7.4s}, [x24], x14
    MUL_EXTRA_SCALE v0, v8, v9, v10, v11
    MUL_EXTRA_SCALE v1, v12, v13, v14, v15
    MUL_EXTRA_SCALE v7, v16, v17, v18, v19

    L4_TILE12_L8_MLA_TERM:
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

    cbz x27, L4_TILE12_ADD_DSTV
    ld1 {v0.4s, v1.4s, v2.4s}, [x27], #48 // input dequant bias
    ld1 {v3.4s}, [x28] // weight kernel sum
    MLA_WEIGHTZERO v8, v0, v3, 0
    MLA_WEIGHTZERO v9, v0, v3, 1
    MLA_WEIGHTZERO v10, v0, v3, 2
    MLA_WEIGHTZERO v11, v0, v3, 3
    MLA_WEIGHTZERO v12, v1, v3, 0
    MLA_WEIGHTZERO v13, v1, v3, 1
    MLA_WEIGHTZERO v14, v1, v3, 2
    MLA_WEIGHTZERO v15, v1, v3, 3
    MLA_WEIGHTZERO v16, v2, v3, 0
    MLA_WEIGHTZERO v17, v2, v3, 1
    MLA_WEIGHTZERO v18, v2, v3, 2
    MLA_WEIGHTZERO v19, v2, v3, 3
    add x28, x28, #32

    L4_TILE12_ADD_DSTV:
    cbz x19, L4_TILE12_ACCUM_BUFFER // x19=0: first block, do not add previous block result
    ld1 {v20.4s, v21.4s, v22.4s, v23.4s}, [x10], #64
    ld1 {v24.4s, v25.4s, v26.4s, v27.4s}, [x10], #64
    ld1 {v28.4s, v29.4s, v30.4s, v31.4s}, [x10]
    ADD_FLOAT v8, v9, v10, v11, v20, v21, v22, v23
    ADD_FLOAT v12, v13, v14, v15, v24, v25, v26, v27
    ADD_FLOAT v16, v17, v18, v19, v28, v29, v30, v31
    sub x10, x10, #128

    L4_TILE12_ACCUM_BUFFER:
    add x19, x19, #1
    cmp x19, x26
    beq L4_TILE12_POST
    st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x10], #64
    st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [x10], #64
    st1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x10]
    sub x10, x10, #128
    b L4_TILE12_BLOCKNUM

    L4_TILE12_POST:
    cbz x9, L4_TILE12_RELU
    ld1 {v0.4s}, [x20], #16
    ADD_BIAS_FLOAT v8, v9, v10, v11, v0
    ADD_BIAS_FLOAT v12, v13, v14, v15, v0
    ADD_BIAS_FLOAT v16, v17, v18, v19, v0

    L4_TILE12_RELU:
    cbz x23, L4_TILE12_STORE
    ld1r {v0.4s}, [x23], #4 // f32 min
    ld1r {v1.4s}, [x23] // f32 max
    ReLU_FP32 v8, v9, v10, v11, v0, v1
    ReLU_FP32 v12, v13, v14, v15, v0, v1
    ReLU_FP32 v16, v17, v18, v19, v0, v1
    sub x23, x23, #4

    L4_TILE12_STORE:
    st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x0], #64
    st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [x0], #64
    st1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x0], x4
    b End

TILE_8:
    mov x25, #0
    cbz x27, TILE_Remain
    mov x25, x22

TILE_Remain:
    cmp x7, #8
    blt TILE_4
    sub x4, x4, #64
    mov x6, x0
    mov x12, x2
    mov x14, x5
    mov x20, x9 // bias
    mov x15, x8 // input kernel sum
    mov x21, x24 // input dequant scale

L8LoopDz_TILE_8:
    cmp x14, #2
    blt L4LoopDz_TILE_8
    mov x11, x1
    mov x19, #0
TILE8_BLOCKNUM:
    mov x13, x3
    movi v7.16b, #15

    SET_BIAS v8, v9, v10, v11
    SET_BIAS v12, v13, v14, v15
    SET_BIAS v16, v17, v18, v19
    SET_BIAS v20, v21, v22, v23

    L8LoopSz_TILE_8:
        ld1 {v5.16b}, [x12], #16 // weight
        ld1 {v0.16b, v1.16b}, [x11], x22 // src
        // int4->int8
        ushr v3.16b, v5.16b, #4
        and v4.16b, v5.16b, v7.16b

        .inst 0x4f80e068 // sdot v8.4s, v3.16b, v0.4b[0]
        .inst 0x4fa0e069 // sdot v9.4s, v3.16b, v0.4b[1]
        .inst 0x4f80e86a // sdot v10.4s, v3.16b, v0.4b[2]
        .inst 0x4fa0e86b // sdot v11.4s, v3.16b, v0.4b[3]

        .inst 0x4f81e06c // sdot v12.4s, v3.16b, v1.4b[0]
        .inst 0x4fa1e06d // sdot v13.4s, v3.16b, v1.4b[1]
        .inst 0x4f81e86e // sdot v14.4s, v3.16b, v1.4b[2]
        .inst 0x4fa1e86f // sdot v15.4s, v3.16b, v1.4b[3]

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

    L8Tile8Quan:
    ld1 {v0.4s, v1.4s}, [x12], #32 // scale
    ld1 {v2.4s, v3.4s}, [x8], x22 // x kernel sum
    ld1 {v24.4s, v25.4s}, [x12], #32 // weight quan zeropoint
    Int32ToFloat v8, v9, v10, v11
    Int32ToFloat v12, v13, v14, v15
    Int32ToFloat v16, v17, v18, v19
    Int32ToFloat v20, v21, v22, v23
    MUL_SCALE v0, v8, v9, v10, v11
    MUL_SCALE v0, v12, v13, v14, v15
    MUL_SCALE v1, v16, v17, v18, v19
    MUL_SCALE v1, v20, v21, v22, v23

    ld1 {v4.4s, v5.4s}, [x24], x25
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

    cbz x27, TILE8_ADD_DSTV
    ld1 {v2.4s, v3.4s}, [x27], x25
    ld1 {v24.4s, v25.4s}, [x28], #32
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

    TILE8_ADD_DSTV:
    cbz x19, TILE8_L8_ACCUM_BUFFER
    ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x10], #64
    ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x10], #64
    ld1 {v24.4s, v25.4s, v26.4s, v27.4s}, [x10], #64
    ld1 {v28.4s, v29.4s, v30.4s, v31.4s}, [x10]
    ADD_FLOAT v8, v9, v10, v11, v0, v1, v2, v3
    ADD_FLOAT v12, v13, v14, v15, v4, v5, v6, v7
    ADD_FLOAT v16, v17, v18, v19, v24, v25, v26, v27
    ADD_FLOAT v20, v21, v22, v23, v28, v29, v30, v31
    sub x10, x10, #192

    TILE8_L8_ACCUM_BUFFER:
    add x19, x19, #1
    cmp x19, x26
    beq TILE8_POST
    st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x10], #64
    st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [x10], #64
    st1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x10], #64
    st1 {v20.4s, v21.4s, v22.4s, v23.4s}, [x10]
    sub x10, x10, #192
    b TILE8_BLOCKNUM

    TILE8_POST:
    sub x14, x14, #2 // oc-2
    cbz x9, TILE8_RELU
    ld1 {v0.4s, v1.4s}, [x20], #32
    ADD_BIAS_FLOAT v8, v9, v10, v11, v0
    ADD_BIAS_FLOAT v12, v13, v14, v15, v0
    ADD_BIAS_FLOAT v16, v17, v18, v19, v1
    ADD_BIAS_FLOAT v20, v21, v22, v23, v1

    TILE8_RELU:
    cbz x23, TILE8_STORE
    ld1r {v0.4s}, [x23], #4 // f32 min
    ld1r {v1.4s}, [x23] // f32 max
    ReLU_FP32 v8, v9, v10, v11, v0, v1
    ReLU_FP32 v12, v13, v14, v15, v0, v1
    ReLU_FP32 v16, v17, v18, v19, v0, v1
    ReLU_FP32 v20, v21, v22, v23, v0, v1
    sub x23, x23, #4

    TILE8_STORE:
    st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x6], #64
    st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [x6], x4
    st1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x6], #64
    st1 {v20.4s, v21.4s, v22.4s, v23.4s}, [x6], x4

    L8Tile8LoopCheck:
    cbz x14, Tile8End
    mov x8, x15 // revert input kernel sum
    mov x24, x21 // revert input dequant scale
    cbz x27, L8LoopDz_TILE_8
    REVERT_INPUT_DEQUANT_BIAS x27, x19, x26, x22
    b L8LoopDz_TILE_8
    
L4LoopDz_TILE_8:
    mov x11, x1
    mov x19, #0
L4_TILE8_BLOCKNUM:
    mov x13, x3
    movi v7.16b, #15

    SET_BIAS v8, v9, v10, v11
    SET_BIAS v12, v13, v14, v15

    L4LoopSz_TILE_8:
        ld1 {v5.16b}, [x12], #16 // weight
        ld1 {v0.16b, v1.16b}, [x11], x22 // src
        // int4->int8
        ushr v3.16b, v5.16b, #4

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

    L4Tile8Quan:
    ld1 {v0.4s}, [x12] // scale
    add x12, x12, #32
    ld1 {v2.4s, v3.4s}, [x8], x22 // x kernel sum
    ld1 {v24.4s}, [x12] // weight quan zeropoint
    add x12, x12, #32
    Int32ToFloat v8, v9, v10, v11
    Int32ToFloat v12, v13, v14, v15
    MUL_SCALE v0, v8, v9, v10, v11
    MUL_SCALE v0, v12, v13, v14, v15

    ld1 {v4.4s, v5.4s}, [x24], x25
    MUL_EXTRA_SCALE v4, v8, v9, v10, v11
    MUL_EXTRA_SCALE v5, v12, v13, v14, v15

    TILE8_L4_MLA_TERM:
    MLA_WEIGHTZERO v8,  v2, v24, 0 // tile:0, oc:0-3
    MLA_WEIGHTZERO v9,  v2, v24, 1 // tile:1, oc:0-3
    MLA_WEIGHTZERO v10, v2, v24, 2 // tile:2, oc:0-3
    MLA_WEIGHTZERO v11, v2, v24, 3 // tile:3, oc:0-3
    MLA_WEIGHTZERO v12, v3, v24, 0 // tile:4, oc:0-3
    MLA_WEIGHTZERO v13, v3, v24, 1 // tile:5, oc:0-3
    MLA_WEIGHTZERO v14, v3, v24, 2 // tile:6, oc:0-3
    MLA_WEIGHTZERO v15, v3, v24, 3 // tile:7, oc:0-3

    cbz x27, L4_TILE8_ADD_DSTV
    ld1 {v2.4s, v3.4s}, [x27], x25
    ld1 {v24.4s}, [x28]
    MLA_WEIGHTZERO v8,  v2, v24, 0 // tile:0, oc:0-3
    MLA_WEIGHTZERO v9,  v2, v24, 1 // tile:1, oc:0-3
    MLA_WEIGHTZERO v10, v2, v24, 2 // tile:2, oc:0-3
    MLA_WEIGHTZERO v11, v2, v24, 3 // tile:3, oc:0-3
    MLA_WEIGHTZERO v12, v3, v24, 0 // tile:4, oc:0-3
    MLA_WEIGHTZERO v13, v3, v24, 1 // tile:5, oc:0-3
    MLA_WEIGHTZERO v14, v3, v24, 2 // tile:6, oc:0-3
    MLA_WEIGHTZERO v15, v3, v24, 3 // tile:7, oc:0-3
    add x28, x28, #32

    L4_TILE8_ADD_DSTV:
    cbz x19, TILE8_L4_ACCUM_BUFFER
    ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x10], #64
    ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x10]
    ADD_FLOAT v8, v9, v10, v11, v0, v1, v2, v3
    ADD_FLOAT v12, v13, v14, v15, v4, v5, v6, v7
    sub x10, x10, #64

    TILE8_L4_ACCUM_BUFFER:
    add x19, x19, #1
    cmp x19, x26
    beq L4_TILE8_POST
    st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x10], #64
    st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [x10]
    sub x10, x10, #64
    b L4_TILE8_BLOCKNUM

    L4_TILE8_POST:
    cbz x9, L4_TILE8_RELU
    ld1 {v0.4s}, [x20], #16
    ADD_BIAS_FLOAT v8, v9, v10, v11, v0
    ADD_BIAS_FLOAT v12, v13, v14, v15, v0

    L4_TILE8_RELU:
    cbz x23, L4_TILE8_STORE
    ld1r {v0.4s}, [x23], #4 // f32 min
    ld1r {v1.4s}, [x23] // f32 max
    ReLU_FP32 v8, v9, v10, v11, v0, v1
    ReLU_FP32 v12, v13, v14, v15, v0, v1
    sub x23, x23, #4

    L4_TILE8_STORE:
    st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x6], #64
    st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [x6], x4

Tile8End:
    add x0, x0, #128
    sub x7, x7, #8
    cbz x7, End
    add x1, x1, #32
    add x8, x15, #32
    add x24, x21, #32
    add x4, x4, #64
    cbz x27, TILE_4
    REVERT_INPUT_DEQUANT_BIAS x27, x19, x26, x22
    REVERT_WEIGHT_KERNEL_SUM x28, x14, x26, x5
    add x27, x27, #32

TILE_4:
    movi v7.16b, #15
    cmp x7, #4
    blt TILE_1_Init
    mov x6, x0
    mov x12, x2
    mov x14, x5
    mov x20, x9
    mov x15, x8 // input kernel sum
    mov x21, x24 // input dequant scale

L8LoopDz_TILE_4:
    cmp x14, #2
    blt L4LoopDz_TILE_4
    mov x11, x1
    mov x19, #0
TILE4_BLOCKNUM:
    mov x13, x3
    SET_BIAS v8, v9, v10, v11
    SET_BIAS v12, v13, v14, v15

    L8LoopSz_TILE_4:
        ld1 {v5.16b}, [x12], #16 // weight
        ld1 {v0.16b}, [x11], x22 // src
        // int4->int8
        ushr v3.16b, v5.16b, #4
        and v4.16b, v5.16b, v7.16b

        .inst 0x4f80e068 // sdot v8.4s, v3.16b, v0.4b[0]
        .inst 0x4fa0e069 // sdot v9.4s, v3.16b, v0.4b[1]
        .inst 0x4f80e86a // sdot v10.4s, v3.16b, v0.4b[2]
        .inst 0x4fa0e86b // sdot v11.4s, v3.16b, v0.4b[3]

        subs x13, x13, #1
        .inst 0x4f80e08c // sdot v12.4s, v4.16b, v0.4b[0]
        .inst 0x4fa0e08d // sdot v13.4s, v4.16b, v0.4b[1]
        .inst 0x4f80e88e // sdot v14.4s, v4.16b, v0.4b[2]
        .inst 0x4fa0e88f // sdot v15.4s, v4.16b, v0.4b[3]
        bne L8LoopSz_TILE_4

    L8Tile4Quan:
    ld1 {v0.4s, v1.4s}, [x12], #32 // scale
    ld1 {v2.4s}, [x8], x22 // x kernel sum
    ld1 {v24.4s, v25.4s}, [x12], #32 // weight quan zeropoint
    Int32ToFloat v8, v9, v10, v11
    Int32ToFloat v12, v13, v14, v15
    MUL_SCALE v0, v8, v9, v10, v11
    MUL_SCALE v1, v12, v13, v14, v15

    ld1 {v4.4s}, [x24], x25
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

    cbz x27, TILE4_ADD_DSTV
    ld1 {v2.4s}, [x27], x25
    ld1 {v24.4s, v25.4s}, [x28], #32
    MLA_WEIGHTZERO v8, v2, v24, 0 // tile:0, oc:0-3
    MLA_WEIGHTZERO v9, v2, v24, 1 // tile:1, oc:0-3
    MLA_WEIGHTZERO v10, v2, v24, 2 // tile:2, oc:0-3
    MLA_WEIGHTZERO v11, v2, v24, 3 // tile:3, oc:0-3
    MLA_WEIGHTZERO v12, v2, v25, 0 // tile:0, oc:4-7
    MLA_WEIGHTZERO v13, v2, v25, 1 // tile:1, oc:4-7
    MLA_WEIGHTZERO v14, v2, v25, 2 // tile:2, oc:4-7
    MLA_WEIGHTZERO v15, v2, v25, 3 // tile:3, oc:4-7

    TILE4_ADD_DSTV:
    cbz x19, TILE4_L8_ACCUM_BUFFER
    ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x10], #64
    ld1 {v20.4s, v21.4s, v22.4s, v23.4s}, [x10]
    ADD_FLOAT v8, v9, v10, v11, v16, v17, v18, v19
    ADD_FLOAT v12, v13, v14, v15, v20, v21, v22, v23
    sub x10, x10, #64

    TILE4_L8_ACCUM_BUFFER:
    add x19, x19, #1
    cmp x19, x26
    beq TILE4_POST
    st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x10], #64
    st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [x10]
    sub x10, x10, #64
    b TILE4_BLOCKNUM

    TILE4_POST:
    sub x14, x14, #2
    cbz x9, TILE4_RELU
    ld1 {v4.4s, v5.4s}, [x20], #32
    ADD_BIAS_FLOAT v8, v9, v10, v11, v4
    ADD_BIAS_FLOAT v12, v13, v14, v15, v5
    TILE4_RELU:
    cbz x23, TILE4_STORE
    ld1r {v26.4s}, [x23], #4 // f32 min
    ld1r {v27.4s}, [x23] // f32 max
    ReLU_FP32 v8, v9, v10, v11, v26, v27
    ReLU_FP32 v12, v13, v14, v15, v26, v27
    sub x23, x23, #4

    TILE4_STORE:
    st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x6], x4
    st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [x6], x4

    L8Tile4LoopCheck:
    cbz x14, Tile4End
    mov x8, x15
    mov x24, x21 // revert input dequant scale
    cbz x27, L8LoopDz_TILE_4
    REVERT_INPUT_DEQUANT_BIAS x27, x19, x26, x22
    b L8LoopDz_TILE_4

L4LoopDz_TILE_4:
    mov x11, x1
    mov x19, #0
L4_TILE4_BLOCKNUM:
    mov x13, x3
    SET_BIAS v8, v9, v10, v11

    L4LoopSz_TILE_4:
        ld1 {v5.16b}, [x12], #16 // weight
        ld1 {v0.16b}, [x11], x22 // src
        // int4->int8
        ushr v3.16b, v5.16b, #4
        subs x13, x13, #1
        .inst 0x4f80e068 // sdot v8.4s, v3.16b, v0.4b[0]
        .inst 0x4fa0e069 // sdot v9.4s, v3.16b, v0.4b[1]
        .inst 0x4f80e86a // sdot v10.4s, v3.16b, v0.4b[2]
        .inst 0x4fa0e86b // sdot v11.4s, v3.16b, v0.4b[3]
        bne L4LoopSz_TILE_4

    L4Tile4Quan:
    ld1 {v0.4s}, [x12] // scale
    add x12, x12, #32
    ld1 {v2.4s}, [x8], x22 // x kernel sum
    ld1 {v24.4s}, [x12] // weight quan zeropoint
    add x12, x12, #32
    Int32ToFloat v8, v9, v10, v11
    MUL_SCALE v0, v8, v9, v10, v11

    ld1 {v4.4s}, [x24], x25
    MUL_EXTRA_SCALE v4, v8, v9, v10, v11

    TILE4_L4_MLA_TERM:
    MLA_WEIGHTZERO v8, v2, v24, 0 // tile:0, oc:0-3
    MLA_WEIGHTZERO v9, v2, v24, 1 // tile:1, oc:0-3
    MLA_WEIGHTZERO v10, v2, v24, 2 // tile:2, oc:0-3
    MLA_WEIGHTZERO v11, v2, v24, 3 // tile:3, oc:0-3

    cbz x27, L4_TILE4_ADD_DSTV
    ld1 {v2.4s}, [x27], x25
    ld1 {v24.4s}, [x28]
    MLA_WEIGHTZERO v8, v2, v24, 0 // tile:0, oc:0-3
    MLA_WEIGHTZERO v9, v2, v24, 1 // tile:1, oc:0-3
    MLA_WEIGHTZERO v10, v2, v24, 2 // tile:2, oc:0-3
    MLA_WEIGHTZERO v11, v2, v24, 3 // tile:3, oc:0-3
    add x28, x28, #32

    L4_TILE4_ADD_DSTV:
    cbz x19, TILE4_L4_ACCUM_BUFFER
    ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x10]
    ADD_FLOAT v8, v9, v10, v11, v16, v17, v18, v19

    TILE4_L4_ACCUM_BUFFER:
    add x19, x19, #1
    cmp x19, x26
    beq L4_TILE4_POST
    st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x10]
    b L4_TILE4_BLOCKNUM

    L4_TILE4_POST:
    cbz x9, L4_TILE4_RELU
    ld1 {v4.4s}, [x20], #16
    ADD_BIAS_FLOAT v8, v9, v10, v11, v4
    L4_TILE4_RELU:
    cbz x23, L4_TILE4_STORE
    ld1r {v26.4s}, [x23], #4 // f32 min
    ld1r {v27.4s}, [x23] // f32 max
    ReLU_FP32 v8, v9, v10, v11, v26, v27
    sub x23, x23, #4

    L4_TILE4_STORE:
    st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x6], x4

Tile4End:
    add x0, x0, #64
    sub x7, x7, #4
    cbz x7, End
    add x1, x1, #16
    add x8, x15, #16
    add x24, x21, #16
    cbz x27, TILE_1_Init
    REVERT_INPUT_DEQUANT_BIAS x27, x19, x26, x22
    REVERT_WEIGHT_KERNEL_SUM x28, x14, x26, x5
    add x27, x27, #16

TILE_1_Init:
    cbz x23, TILE_1
    ld1r {v26.4s}, [x23], #4 // f32 min
    ld1r {v27.4s}, [x23] // f32 max
TILE_1:
    // Already execute: [movi v7.16b, #15] in TILE_4
    mov x6, x0
    mov x12, x2
    mov x14, x5
    mov x20, x9
    mov x15, x8 // input kernel sum
    mov x21, x24 // input dequant scale
L8LoopDz_TILE_1:
    cmp x14, #2
    blt L4LoopDz_TILE_1
    mov x11, x1
    mov x19, #0
TILE1_BLOCKNUM:
    mov x13, x3

    movi v8.16b, #0
    movi v9.16b, #0
    b L8LoopSz_TILE_1_lu1

    L8LoopSz_TILE_1_lu4:
        ld1 {v3.16b, v4.16b, v5.16b, v6.16b}, [x12], #64 // weight: hu=0,1,2,3,pack=0~7
        ld1 {v0.s}[0], [x11], x22 // src
        ld1 {v0.s}[1], [x11], x22
        ld1 {v0.s}[2], [x11], x22
        ld1 {v0.s}[3], [x11], x22

        sub x13, x13, #4
        // int4->int8
        ushr v12.16b, v3.16b, #4
        and v22.16b, v3.16b, v7.16b

        ushr v15.16b, v4.16b, #4
        and v23.16b, v4.16b, v7.16b

        ushr v18.16b, v5.16b, #4
        and v24.16b, v5.16b, v7.16b

        ushr v21.16b, v6.16b, #4
        and v25.16b, v6.16b, v7.16b

        cmp x13, #4
        .inst 0x4f80e188 // sdot v8.4s, v12.16b, v0.4b[0]
        .inst 0x4f80e2c9 // sdot v9.4s, v22.16b, v0.4b[0]
        .inst 0x4fa0e1e8 // sdot v8.4s, v15.16b, v0.4b[1]
        .inst 0x4fa0e2e9 // sdot v9.4s, v23.16b, v0.4b[1]
        .inst 0x4f80ea48 // sdot v8.4s, v18.16b, v0.4b[2]
        .inst 0x4f80eb09 // sdot v9.4s, v24.16b, v0.4b[2]
        .inst 0x4fa0eaa8 // sdot v8.4s, v21.16b, v0.4b[3]
        .inst 0x4fa0eb29 // sdot v9.4s, v25.16b, v0.4b[3]
        bge L8LoopSz_TILE_1_lu4

    cbz x13, L8LoopSzEnd_TILE_1

    L8LoopSz_TILE_1_lu1:
        ld1 {v5.16b}, [x12], #16 // weight
        ld1 {v0.s}[0], [x11], x22 // src
        subs x13, x13, #1
        // int4->int8
        ushr v3.16b, v5.16b, #4
        and v12.16b, v5.16b, v7.16b
        .inst 0x4f80e068 // sdot v8.4s, v3.16b, v0.4b[0]
        .inst 0x4f80e189 // sdot v9.4s, v12.16b, v0.4b[0]
        bne L8LoopSz_TILE_1_lu1

    L8LoopSzEnd_TILE_1:

    L8Tile1Quan:
    ld1 {v0.4s, v1.4s}, [x12], #32 // scale
    ld1 {v2.s}[0], [x8], x22 // x kernel sum
    ld1 {v24.4s, v25.4s}, [x12], #32 // weight quan zeropoint
    scvtf v8.4s, v8.4s
    scvtf v9.4s, v9.4s
    fmul v8.4s, v8.4s, v0.4s
    fmul v9.4s, v9.4s, v1.4s

    ld1 {v4.s}[0], [x24], x25
    fmul v8.4s, v8.4s, v4.s[0]
    fmul v9.4s, v9.4s, v4.s[0]

    TILE1_L8_MLA_TERM:
    MLA_WEIGHTZERO v8, v2, v24, 0 // tile:0, oc:0-3
    MLA_WEIGHTZERO v9, v2, v25, 0 // tile:0, oc:4-7

    cbz x27, TILE1_ADD_DSTV
    ld1 {v2.s}[0], [x27], x25
    ld1 {v24.4s, v25.4s}, [x28], #32
    MLA_WEIGHTZERO v8, v2, v24, 0 // tile:0, oc:0-3
    MLA_WEIGHTZERO v9, v2, v25, 0 // tile:0, oc:4-7

    TILE1_ADD_DSTV:
    cbz x19, TILE1_L8_ACCUM_BUFFER
    ld1 {v10.4s, v11.4s}, [x10]
    fadd v8.4s, v8.4s, v10.4s
    fadd v9.4s, v9.4s, v11.4s

    TILE1_L8_ACCUM_BUFFER:
    add x19, x19, #1
    cmp x19, x26
    beq TILE1_POST
    st1 {v8.4s, v9.4s}, [x10]
    b TILE1_BLOCKNUM

    TILE1_POST:
    sub x14, x14, #2
    cbz x9, TILE1_RELU
    ld1 {v10.4s, v11.4s}, [x20], #32
    fadd v8.4s, v8.4s, v10.4s
    fadd v9.4s, v9.4s, v11.4s
    TILE1_RELU:
    cbz x23, TILE1_STORE
    fmin v8.4s, v8.4s, v27.4s
    fmin v9.4s, v9.4s, v27.4s
    fmax v8.4s, v8.4s, v26.4s
    fmax v9.4s, v9.4s, v26.4s

    TILE1_STORE:
    st1 {v8.4s}, [x6], x4
    st1 {v9.4s}, [x6], x4

    L8Tile1LoopCheck:
    cbz x14, Tile1End
    mov x8, x15
    mov x24, x21 // revert input dequant scale
    cbz x27, L8LoopDz_TILE_1
    REVERT_INPUT_DEQUANT_BIAS x27, x19, x26, x22
    b L8LoopDz_TILE_1

L4LoopDz_TILE_1:
    mov x11, x1
    mov x19, #0
L4_TILE1_BLOCKNUM:
    mov x13, x3
    movi v8.16b, #0

    L4LoopSz_TILE_1_lu1:
        ld1 {v5.16b}, [x12], #16 // weight
        ld1 {v0.s}[0], [x11], x22 // src
        subs x13, x13, #1
        // int4->int8
        ushr v3.16b, v5.16b, #4
        .inst 0x4f80e068 // sdot v8.4s, v3.16b, v0.4b[0]
    bne L4LoopSz_TILE_1_lu1

    L4Tile1Quan:
    ld1 {v0.4s}, [x12] // scale
    add x12, x12, #32
    ld1 {v2.s}[0], [x8], x22 // x kernel sum
    ld1 {v24.4s}, [x12] // weight quan zeropoint
    add x12, x12, #32
    scvtf v8.4s, v8.4s
    fmul v8.4s, v8.4s, v0.4s

    ld1 {v4.s}[0], [x24], x25
    fmul v8.4s, v8.4s, v4.s[0]

    L4_TILE1_L8_MLA_TERM:
    MLA_WEIGHTZERO v8, v2, v24, 0 // tile:0, oc:0-3

    cbz x27, L4_TILE1_ADD_DSTV
    ld1 {v2.s}[0], [x27], x25
    ld1 {v24.4s}, [x28]
    MLA_WEIGHTZERO v8, v2, v24, 0 // tile:0, oc:0-3
    add x28, x28, #32

    L4_TILE1_ADD_DSTV:
    cbz x19, L4_TILE1_L8_ACCUM_BUFFER
    ld1 {v10.4s}, [x10]
    fadd v8.4s, v8.4s, v10.4s

    L4_TILE1_L8_ACCUM_BUFFER:
    add x19, x19, #1
    cmp x19, x26
    beq L4_TILE1_POST
    st1 {v8.4s}, [x10]
    b L4_TILE1_BLOCKNUM

    L4_TILE1_POST:
    cbz x9, L4_TILE1_RELU
    ld1 {v10.4s}, [x20], #16
    fadd v8.4s, v8.4s, v10.4s
    L4_TILE1_RELU:
    cbz x23, L4_TILE1_STORE
    fmin v8.4s, v8.4s, v27.4s
    fmax v8.4s, v8.4s, v26.4s

    L4_TILE1_STORE:
    st1 {v8.4s}, [x6], x4

Tile1End:
    subs x7, x7, #1
    beq End

    add x0, x0, #16
    add x24, x21, #4
    add x1, x1, #4
    add x8, x15, #4
    cbz x27, TILE_1
    REVERT_INPUT_DEQUANT_BIAS x27, x19, x26, x22
    REVERT_WEIGHT_KERNEL_SUM x28, x14, x26, x5
    add x27, x27, #4
    b TILE_1

End:
ldp x27, x28, [sp, #(16 * 8)]
ldp x25, x26, [sp, #(16 * 7)]
ldp x23, x24, [sp, #(16 * 6)]
ldp x19, x20, [sp, #(16 * 5)]
ldp x21, x22, [sp, #(16 * 4)]
ldp d8,  d9,  [sp, #(16 * 3)]
ldp d10, d11, [sp, #(16 * 2)]
ldp d12, d13, [sp, #(16 * 1)]
ldp d14, d15, [sp], #(16 * 10)
ret

#endif // __aarch64__
