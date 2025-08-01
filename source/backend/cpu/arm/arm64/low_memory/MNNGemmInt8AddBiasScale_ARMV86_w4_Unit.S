//
//  MNNGemmInt8AddBiasScale_ARMV86_w4_Unit.S
//  MNN
//
//  Created by MNN on 2022/09/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#if defined(__aarch64__)
#include "MNNAsmGlobal.h"

.text
.align 5

.macro SET_0_5 d0, d1, d2, d3, d4
    movi \d0\().16b, #0
    movi \d1\().16b, #0
    movi \d2\().16b, #0
    movi \d3\().16b, #0
    movi \d4\().16b, #0
.endm
.macro SET_0_4 d0, d1, d2, d3
    movi \d0\().16b, #0
    movi \d1\().16b, #0
    movi \d2\().16b, #0
    movi \d3\().16b, #0
.endm
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
.macro MUL_INPUT_SCALE s, d0, d1, d2, d3
    fmul \d0\().4s, \d0\().4s, \s\().s[0]
    fmul \d1\().4s, \d1\().4s, \s\().s[1]
    fmul \d2\().4s, \d2\().4s, \s\().s[2]
    fmul \d3\().4s, \d3\().4s, \s\().s[3]
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
.macro ReLU_FP32_2 s0, s1, z0, z1 // z0:min z1:max
    fmin \s0\().4s, \s0\().4s, \z1\().4s
    fmin \s1\().4s, \s1\().4s, \z1\().4s
    fmax \s0\().4s, \s0\().4s, \z0\().4s
    fmax \s1\().4s, \s1\().4s, \z0\().4s
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

asm_function MNNGemmInt8AddBiasScale_ARMV86_w4_Unit
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
//void MNNGemmInt8AddBiasScale_ARMV86_w4_Unit(int8_t* dst, const int8_t* src,
//    const int8_t* weight, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad,
// const QuanPostTreatParameters* parameters, size_t realDstCount);

//Auto: x0:dst, x1:src, x2:weight, x3:src_depth_quad, x4:dst_step
//x5:dst_depth_quad, x6: parameters, x7: realDstCount

//Load from x6: x9: bias, x8: srcKernelSum
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
ldr x8, [x6, #40] // srcKernelSum
ldr x28, [x6, #48] // weightKernelSum
ldr x14, [x6, #56]  // fp32minmax
ldr x26, [x6, #64]  // blockNum
lsl x22, x7, #3 // eDest * GEMM_INT8_SRC_UNIT
ldr x23, [x6, #80]  // input scale
ldr x27, [x6, #88]  // input bias
ldr x10, [x6, #96]  // accumBuffer

mov x15, x8      // input kernel sum
mov x21, x23     // input dequant scale
mov x24, #-32       // inputBlockNum=1
cbz x27, TILE_10
mov x24, #8         // inputBlockNum>1

TILE_10:
    cmp x7, #10
    blt TILE_8

    mov x20, x9      // bias
    mov x6, x27      // input bias
    sub x4, x4, #128 // For Tile10
    mov x12, #-256
cmp x5, #2
blt LoopDz4_TILE_10

LoopDz8_TILE_10:
    mov x11, x1 // src
    mov x19, #0

TILE10_BLOCKNUM:
    mov x13, x3 // src_depth_quad
    movi v2.16b, #15

    SET_0_5 v12, v16, v20, v24, v28 // oc:0,1,0,1
    SET_0_5 v13, v17, v21, v25, v29 // oc:2,3,2,3
    SET_0_5 v14, v18, v22, v26, v30 // oc:4,5,4,5
    SET_0_5 v15, v19, v23, v27, v31 // oc:6,7,6,7

LoopSz_TILE_10:
    ld1 {v0.16b, v1.16b}, [x2], #32                    // weight
    ld1 {v3.16b, v4.16b, v5.16b, v6.16b}, [x11], #64    // src: E0-E9
    ld1 {v7.16b}, [x11], #16
    // int4->int8
    ushr v8.16b, v0.16b, #4 // oc:0-1
    ushr v9.16b, v1.16b, #4 // oc:2-3
    and v10.16b, v0.16b, v2.16b // oc:4-5
    and v11.16b, v1.16b, v2.16b // oc:6-7

    subs x13, x13, #1
    .inst 0x4e88a46c // smmla v12.4s, v3.16b, v8.16b // tile0-oc0, tile0-oc1, tile1-oc0, tile1-oc1
    .inst 0x4e89a46d // smmla v13.4s, v3.16b, v9.16b // tile0-oc2, tile0-oc3, tile1-oc2, tile1-oc3
    .inst 0x4e8aa46e // smmla v14.4s, v3.16b, v10.16b // tile0-oc4, tile0-oc5, tile1-oc4, tile1-oc5
    .inst 0x4e8ba46f // smmla v15.4s, v3.16b, v11.16b // tile0-oc6, tile0-oc7, tile1-oc6, tile1-oc7
    
    .inst 0x4e88a490 // smmla v16.4s, v4.16b, v8.16b // tile2-oc0, tile2-oc1, tile3-oc0, tile3-oc1
    .inst 0x4e89a491 // smmla v17.4s, v4.16b, v9.16b // tile2-oc2, tile2-oc3, tile3-oc2, tile3-oc3
    .inst 0x4e8aa492 // smmla v18.4s, v4.16b, v10.16b // tile2-oc4, tile2-oc5, tile3-oc4, tile3-oc5
    .inst 0x4e8ba493 // smmla v19.4s, v4.16b, v11.16b // tile2-oc6, tile2-oc7, tile3-oc6, tile3-oc7
    
    .inst 0x4e88a4b4 // smmla v20.4s, v5.16b, v8.16b // tile4-oc0, tile4-oc1, tile5-oc0, tile5-oc1
    .inst 0x4e89a4b5 // smmla v21.4s, v5.16b, v9.16b // tile4-oc2, tile4-oc3, tile5-oc2, tile5-oc3
    .inst 0x4e8aa4b6 // smmla v22.4s, v5.16b, v10.16b // tile4-oc4, tile4-oc5, tile5-oc4, tile5-oc5
    .inst 0x4e8ba4b7 // smmla v23.4s, v5.16b, v11.16b // tile4-oc6, tile4-oc7, tile5-oc6, tile5-oc7

    .inst 0x4e88a4d8 // smmla v24.4s, v6.16b, v8.16b // tile6-oc0, tile6-oc1, tile7-oc0, tile7-oc1
    .inst 0x4e89a4d9 // smmla v25.4s, v6.16b, v9.16b // tile6-oc2, tile6-oc3, tile7-oc2, tile7-oc3
    .inst 0x4e8aa4da // smmla v26.4s, v6.16b, v10.16b // tile6-oc4, tile6-oc5, tile7-oc4, tile7-oc5
    .inst 0x4e8ba4db // smmla v27.4s, v6.16b, v11.16b // tile6-oc6, tile6-oc7, tile7-oc6, tile7-oc7

    .inst 0x4e88a4fc // smmla v28.4s, v7.16b, v8.16b // tile8-oc0, tile8-oc1, tile9-oc0, tile9-oc1
    .inst 0x4e89a4fd // smmla v29.4s, v7.16b, v9.16b // tile8-oc2, tile8-oc3, tile9-oc2, tile9-oc3
    .inst 0x4e8aa4fe // smmla v30.4s, v7.16b, v10.16b // tile8-oc4, tile8-oc5, tile9-oc4, tile9-oc5
    .inst 0x4e8ba4ff // smmla v31.4s, v7.16b, v11.16b // tile8-oc6, tile8-oc7, tile9-oc6, tile9-oc7
    bne LoopSz_TILE_10
LoopSzEnd_TILE_10:
    // transpose
    uzp1 v0.2d, v12.2d, v13.2d // E0: oc:0-3
    uzp2 v1.2d, v12.2d, v13.2d // E1: oc:0-3
    uzp1 v2.2d, v16.2d, v17.2d
    uzp2 v3.2d, v16.2d, v17.2d
    uzp1 v4.2d, v20.2d, v21.2d
    uzp2 v5.2d, v20.2d, v21.2d
    uzp1 v6.2d, v24.2d, v25.2d
    uzp2 v7.2d, v24.2d, v25.2d
    uzp1 v8.2d, v28.2d, v29.2d
    uzp2 v9.2d, v28.2d, v29.2d

    uzp1 v10.2d, v14.2d, v15.2d // E0: oc:4-7
    uzp2 v11.2d, v14.2d, v15.2d // E1: oc:4-7
    uzp1 v12.2d, v18.2d, v19.2d
    uzp2 v13.2d, v18.2d, v19.2d
    uzp1 v14.2d, v22.2d, v23.2d
    uzp2 v15.2d, v22.2d, v23.2d
    uzp1 v16.2d, v26.2d, v27.2d
    uzp2 v17.2d, v26.2d, v27.2d
    uzp1 v18.2d, v30.2d, v31.2d
    uzp2 v19.2d, v30.2d, v31.2d

    Int32ToFloat v0, v1, v2, v3
    Int32ToFloat v4, v5, v6, v7
    Int32ToFloat v8, v9, v10, v11
    Int32ToFloat v12, v13, v14, v15
    Int32ToFloat v16, v17, v18, v19

Tile10Quan:
    ld1 {v20.4s, v21.4s}, [x2], #32 // weight scale
    ld1 {v22.4s, v23.4s}, [x8], #32 // input kernel sum
    ld1 {v24.d}[0], [x8], #8
    ld1 {v25.4s, v26.4s}, [x2], #32 // weight quan zeropoint
    MUL_SCALE v20, v0, v1, v2, v3
    MUL_SCALE v20, v4, v5, v6, v7
    MUL_SCALE v21, v10, v11, v12, v13
    MUL_SCALE v21, v14, v15, v16, v17
    fmul v8.4s, v8.4s, v20.4s
    fmul v9.4s, v9.4s, v20.4s
    fmul v18.4s, v18.4s, v21.4s
    fmul v19.4s, v19.4s, v21.4s

    ld1 {v27.4s, v28.4s}, [x23], #32 // input scale
    ld1 {v29.d}[0], [x23], x24
    MUL_INPUT_SCALE v27, v0, v1, v2, v3
    MUL_INPUT_SCALE v28, v4, v5, v6, v7
    MUL_INPUT_SCALE v27, v10, v11, v12, v13
    MUL_INPUT_SCALE v28, v14, v15, v16, v17
    fmul v8.4s, v8.4s, v29.s[0]
    fmul v9.4s, v9.4s, v29.s[1]
    fmul v18.4s, v18.4s, v29.s[0]
    fmul v19.4s, v19.4s, v29.s[1]

    TILE10_MLA:
    MLA_WEIGHTZERO v0, v22, v25, 0 // tile:0, oc:0-3
    MLA_WEIGHTZERO v1, v22, v25, 1 // tile:1, oc:0-3
    MLA_WEIGHTZERO v10, v22, v26, 0 // tile:0, oc:4-7
    MLA_WEIGHTZERO v11, v22, v26, 1 // tile:1, oc:4-7

    MLA_WEIGHTZERO v2, v22, v25, 2 // tile:2, oc:0-3
    MLA_WEIGHTZERO v3, v22, v25, 3 // tile:3, oc:0-3
    MLA_WEIGHTZERO v12, v22, v26, 2 // tile:2, oc:4-7
    MLA_WEIGHTZERO v13, v22, v26, 3 // tile:3, oc:4-7

    MLA_WEIGHTZERO v4,  v23, v25, 0 // tile:4, oc:0-3
    MLA_WEIGHTZERO v5,  v23, v25, 1 // tile:5, oc:0-3
    MLA_WEIGHTZERO v14, v23, v26, 0 // tile:4, oc:4-7
    MLA_WEIGHTZERO v15, v23, v26, 1 // tile:5, oc:4-7

    MLA_WEIGHTZERO v6, v23, v25, 2 // tile:6, oc:0-3
    MLA_WEIGHTZERO v7, v23, v25, 3 // tile:7, oc:0-3
    MLA_WEIGHTZERO v16, v23, v26, 2 // tile:6, oc:4-7
    MLA_WEIGHTZERO v17, v23, v26, 3 // tile:7, oc:4-7

    MLA_WEIGHTZERO v8, v24, v25, 0 // tile:8, oc:0-3
    MLA_WEIGHTZERO v9, v24, v25, 1 // tile:9, oc:0-3
    MLA_WEIGHTZERO v18, v24, v26, 0 // tile:8, oc:4-7
    MLA_WEIGHTZERO v19, v24, v26, 1 // tile:9, oc:4-7

    cbz x6, TILE10_ADD_DSTV
    ld1 {v22.4s, v23.4s}, [x27], #32 // input dequant bias
    ld1 {v24.2s}, [x27], #8
    ld1 {v25.4s, v26.4s}, [x28], #32 // weight kernel sum
    MLA_WEIGHTZERO v0, v22, v25, 0 // tile:0, oc:0-3
    MLA_WEIGHTZERO v1, v22, v25, 1 // tile:1, oc:0-3
    MLA_WEIGHTZERO v10, v22, v26, 0 // tile:0, oc:4-7
    MLA_WEIGHTZERO v11, v22, v26, 1 // tile:1, oc:4-7

    MLA_WEIGHTZERO v2, v22, v25, 2 // tile:2, oc:0-3
    MLA_WEIGHTZERO v3, v22, v25, 3 // tile:3, oc:0-3
    MLA_WEIGHTZERO v12, v22, v26, 2 // tile:2, oc:4-7
    MLA_WEIGHTZERO v13, v22, v26, 3 // tile:3, oc:4-7

    MLA_WEIGHTZERO v4,  v23, v25, 0 // tile:4, oc:0-3
    MLA_WEIGHTZERO v5,  v23, v25, 1 // tile:5, oc:0-3
    MLA_WEIGHTZERO v14, v23, v26, 0 // tile:4, oc:4-7
    MLA_WEIGHTZERO v15, v23, v26, 1 // tile:5, oc:4-7

    MLA_WEIGHTZERO v6, v23, v25, 2 // tile:6, oc:0-3
    MLA_WEIGHTZERO v7, v23, v25, 3 // tile:7, oc:0-3
    MLA_WEIGHTZERO v16, v23, v26, 2 // tile:6, oc:4-7
    MLA_WEIGHTZERO v17, v23, v26, 3 // tile:7, oc:4-7

    MLA_WEIGHTZERO v8, v24, v25, 0 // tile:8, oc:0-3
    MLA_WEIGHTZERO v9, v24, v25, 1 // tile:9, oc:0-3
    MLA_WEIGHTZERO v18, v24, v26, 0 // tile:8, oc:4-7
    MLA_WEIGHTZERO v19, v24, v26, 1 // tile:9, oc:4-7

    TILE10_ADD_DSTV:
    cbz x19, TILE10_TEMP_BUFFER
    ld1 {v20.4s, v21.4s, v22.4s, v23.4s}, [x10], #64
    ld1 {v24.4s, v25.4s, v26.4s, v27.4s}, [x10], #64
    ld1 {v28.4s, v29.4s, v30.4s, v31.4s}, [x10], #64
    ADD_FLOAT v0, v1, v2, v3, v20, v21, v22, v23
    ADD_FLOAT v4, v5, v6, v7, v24, v25, v26, v27
    ADD_FLOAT v8, v9, v10, v11, v28, v29, v30, v31

    ld1 {v20.4s, v21.4s, v22.4s, v23.4s}, [x10], #64
    ld1 {v24.4s, v25.4s, v26.4s, v27.4s}, [x10], x12
    ADD_FLOAT v12, v13, v14, v15, v20, v21, v22, v23
    ADD_FLOAT v16, v17, v18, v19, v24, v25, v26, v27

    TILE10_TEMP_BUFFER:
    add x19, x19, #1
    cmp x19, x26
    beq TILE10_POST
    st1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x10], #64
    st1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x10], #64
    st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x10], #64
    st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [x10], #64
    st1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x10], x12
    b TILE10_BLOCKNUM

    TILE10_POST:
    sub x5, x5, #2
    cbz x9, TILE10_RELU
    ld1 {v20.4s, v21.4s}, [x20], #32  // bias
    ADD_BIAS_FLOAT v0, v1, v2, v3, v20
    ADD_BIAS_FLOAT v4, v5, v6, v7, v20
    ADD_BIAS_FLOAT v10, v11, v12, v13, v21
    ADD_BIAS_FLOAT v14, v15, v16, v17, v21
    fadd v8.4s, v8.4s, v20.4s
    fadd v9.4s, v9.4s, v20.4s
    fadd v18.4s, v18.4s, v21.4s
    fadd v19.4s, v19.4s, v21.4s
    TILE10_RELU:
    cbz x14, TILE10_STORE
    ld1r {v30.4s}, [x14], #4 // f32 min
    ld1r {v31.4s}, [x14]     // f32 max
    ReLU_FP32 v0, v1, v2, v3, v30, v31
    ReLU_FP32 v4, v5, v6, v7, v30, v31
    ReLU_FP32 v8, v9, v10, v11, v30, v31
    ReLU_FP32 v12, v13, v14, v15, v30, v31
    ReLU_FP32 v16, v17, v18, v19, v30, v31
    sub x14, x14, #4

    TILE10_STORE:
    st1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x0], #64
    st1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x0], #64
    st1 {v8.4s, v9.4s}, [x0], x4
    st1 {v10.4s, v11.4s, v12.4s, v13.4s}, [x0], #64
    st1 {v14.4s, v15.4s, v16.4s, v17.4s}, [x0], #64
    st1 {v18.4s, v19.4s}, [x0], x4

Tile10LoopCheck:
    cmp x5, #2
    mov x8, x15 // revert input kernel sum
    mov x27, x6 // revert input dequant bias
    mov x23, x21 // revert input dequant scale
    bge LoopDz8_TILE_10
    cbz x5, End

LoopDz4_TILE_10:
    mov x11, x1 // src
    mov x19, #0

L4_TILE10_BLOCKNUM:
    mov x13, x3 // src_depth_quad

    SET_0_5 v12, v16, v20, v24, v28 // oc:0,1,0,1
    SET_0_5 v13, v17, v21, v25, v29 // oc:2,3,2,3

L4_LoopSz_TILE_10:
    ld1 {v0.16b, v1.16b}, [x2], #32                    // weight
    ld1 {v3.16b, v4.16b, v5.16b, v6.16b}, [x11], #64    // src: E0-E9
    ld1 {v7.16b}, [x11], #16
     // int4->int8
    ushr v8.16b, v0.16b, #4 // oc:0-1
    ushr v9.16b, v1.16b, #4 // oc:2-3

    subs x13, x13, #1
    .inst 0x4e88a46c // smmla v12.4s, v3.16b, v8.16b // tile0-oc0, tile0-oc1, tile1-oc0, tile1-oc1
    .inst 0x4e89a46d // smmla v13.4s, v3.16b, v9.16b // tile0-oc2, tile0-oc3, tile1-oc2, tile1-oc3
    
    .inst 0x4e88a490 // smmla v16.4s, v4.16b, v8.16b // tile2-oc0, tile2-oc1, tile3-oc0, tile3-oc1
    .inst 0x4e89a491 // smmla v17.4s, v4.16b, v9.16b // tile2-oc2, tile2-oc3, tile3-oc2, tile3-oc3
    
    .inst 0x4e88a4b4 // smmla v20.4s, v5.16b, v8.16b // tile4-oc0, tile4-oc1, tile5-oc0, tile5-oc1
    .inst 0x4e89a4b5 // smmla v21.4s, v5.16b, v9.16b // tile4-oc2, tile4-oc3, tile5-oc2, tile5-oc3

    .inst 0x4e88a4d8 // smmla v24.4s, v6.16b, v8.16b // tile6-oc0, tile6-oc1, tile7-oc0, tile7-oc1
    .inst 0x4e89a4d9 // smmla v25.4s, v6.16b, v9.16b // tile6-oc2, tile6-oc3, tile7-oc2, tile7-oc3

    .inst 0x4e88a4fc // smmla v28.4s, v7.16b, v8.16b // tile8-oc0, tile8-oc1, tile9-oc0, tile9-oc1
    .inst 0x4e89a4fd // smmla v29.4s, v7.16b, v9.16b // tile8-oc2, tile8-oc3, tile9-oc2, tile9-oc3
    bne L4_LoopSz_TILE_10

    // transpose
    uzp1 v0.2d, v12.2d, v13.2d // E0: oc:0-3
    uzp2 v1.2d, v12.2d, v13.2d // E1: oc:0-3
    uzp1 v2.2d, v16.2d, v17.2d
    uzp2 v3.2d, v16.2d, v17.2d
    uzp1 v4.2d, v20.2d, v21.2d
    uzp2 v5.2d, v20.2d, v21.2d
    uzp1 v6.2d, v24.2d, v25.2d
    uzp2 v7.2d, v24.2d, v25.2d
    uzp1 v8.2d, v28.2d, v29.2d
    uzp2 v9.2d, v28.2d, v29.2d

    Int32ToFloat v0, v1, v2, v3
    Int32ToFloat v4, v5, v6, v7
    scvtf v8.4s, v8.4s
    scvtf v9.4s, v9.4s

L4_Tile10Quan:
    ld1 {v20.4s}, [x2]  // weight scale
    add x2, x2, #32
    ld1 {v22.4s, v23.4s}, [x8], #32 // x kernel sum
    ld1 {v24.d}[0], [x8], #8
    ld1 {v25.4s}, [x2] // weight quan zeropoint
    MUL_SCALE v20, v0, v1, v2, v3
    MUL_SCALE v20, v4, v5, v6, v7
    fmul v8.4s, v8.4s, v20.4s
    fmul v9.4s, v9.4s, v20.4s
    add x2, x2, #32

    ld1 {v27.4s, v28.4s}, [x23], #32 // input dequant scale
    ld1 {v29.d}[0], [x23], x24
    MUL_INPUT_SCALE v27, v0, v1, v2, v3
    MUL_INPUT_SCALE v28, v4, v5, v6, v7
    fmul v8.4s, v8.4s, v29.s[0]
    fmul v9.4s, v9.4s, v29.s[1]

    L4_TILE10_MLA:
    MLA_WEIGHTZERO v0, v22, v25, 0 // tile:0, oc:0-3
    MLA_WEIGHTZERO v1, v22, v25, 1 // tile:1, oc:0-3
    MLA_WEIGHTZERO v2, v22, v25, 2 // tile:2, oc:0-3
    MLA_WEIGHTZERO v3, v22, v25, 3 // tile:3, oc:0-3
    MLA_WEIGHTZERO v4, v23, v25, 0 // tile:4, oc:0-3
    MLA_WEIGHTZERO v5, v23, v25, 1 // tile:5, oc:0-3
    MLA_WEIGHTZERO v6, v23, v25, 2 // tile:6, oc:0-3
    MLA_WEIGHTZERO v7, v23, v25, 3 // tile:7, oc:0-3
    MLA_WEIGHTZERO v8, v24, v25, 0 // tile:8, oc:0-3
    MLA_WEIGHTZERO v9, v24, v25, 1 // tile:9, oc:0-3

    cbz x27, L4_TILE10_ADD_DSTV
    ld1 {v22.4s, v23.4s}, [x27], #32 // input dequant bias
    ld1 {v24.2s}, [x27], #8
    ld1 {v25.4s}, [x28] // weight kernel sum
    MLA_WEIGHTZERO v0, v22, v25, 0 // tile:0, oc:0-3
    MLA_WEIGHTZERO v1, v22, v25, 1 // tile:1, oc:0-3
    MLA_WEIGHTZERO v2, v22, v25, 2 // tile:2, oc:0-3
    MLA_WEIGHTZERO v3, v22, v25, 3 // tile:3, oc:0-3
    MLA_WEIGHTZERO v4, v23, v25, 0 // tile:4, oc:0-3
    MLA_WEIGHTZERO v5, v23, v25, 1 // tile:5, oc:0-3
    MLA_WEIGHTZERO v6, v23, v25, 2 // tile:6, oc:0-3
    MLA_WEIGHTZERO v7, v23, v25, 3 // tile:7, oc:0-3
    MLA_WEIGHTZERO v8, v24, v25, 0 // tile:8, oc:0-3
    MLA_WEIGHTZERO v9, v24, v25, 1 // tile:9, oc:0-3
    add x28, x28, #32

    L4_TILE10_ADD_DSTV:
    cbz x19, L4_TILE10_TEMP_BUFFER
    ld1 {v20.4s, v21.4s, v22.4s, v23.4s}, [x10], #64
    ld1 {v24.4s, v25.4s, v26.4s, v27.4s}, [x10], #64
    ld1 {v28.4s, v29.4s}, [x10]
    ADD_FLOAT v0, v1, v2, v3, v20, v21, v22, v23
    ADD_FLOAT v4, v5, v6, v7, v24, v25, v26, v27
    fadd v8.4s, v8.4s, v28.4s
    fadd v9.4s, v9.4s, v29.4s
    sub x10, x10, #128

    L4_TILE10_TEMP_BUFFER:
    add x19, x19, #1
    cmp x19, x26
    beq L4_TILE10_POST
    st1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x10], #64
    st1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x10], #64
    st1 {v8.4s, v9.4s}, [x10]
    sub x10, x10, #128
    b L4_TILE10_BLOCKNUM

    L4_TILE10_POST:
    cbz x9, L4_TILE10_RELU
    ld1 {v20.4s}, [x20]  // bias
    ADD_BIAS_FLOAT v0, v1, v2, v3, v20
    ADD_BIAS_FLOAT v4, v5, v6, v7, v20
    fadd v8.4s, v8.4s, v20.4s
    fadd v9.4s, v9.4s, v20.4s
    L4_TILE10_RELU:
    cbz x14, L4_TILE10_STORE
    ld1r {v30.4s}, [x14], #4 // f32 min
    ld1r {v31.4s}, [x14]     // f32 max
    ReLU_FP32 v0, v1, v2, v3, v30, v31
    ReLU_FP32 v4, v5, v6, v7, v30, v31
    fmax v8.4s, v8.4s, v30.4s
    fmax v9.4s, v9.4s, v30.4s
    fmin v8.4s, v8.4s, v31.4s
    fmin v9.4s, v9.4s, v31.4s

    L4_TILE10_STORE:
    st1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x0], #64
    st1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x0], #64
    st1 {v8.4s, v9.4s}, [x0], x4
    b End

TILE_8:
    mov x25, #0
    cbz x27, RELU_VECTOR
    lsr x25, x22, #1         // input block quant: realDstCount * sizeof(float)
    RELU_VECTOR:
    cbz x14, TILE_Remain
    ld1r {v30.4s}, [x14], #4 // f32 min
    ld1r {v31.4s}, [x14]     // f32 max

    TILE_Remain:
    movi v28.16b, #15
    cmp x7, #8
    blt TILE_4
    sub x4, x4, #64 // For float32 output
    mov x24, x5 // dst_depth_quad
    mov x6, x0 // dst
    mov x12, x2 // weight
    mov x20, x9 // bias

LoopDz_TILE_8:
    cmp x24, #2
    blt LoopDz4_TILE_8
    mov x11, x1 // src
    mov x19, #0

TILE8_BLOCKNUM:
    mov x13, x3 // src_depth_quad

    SET_0_4 v12, v16, v20, v24 // oc:0,1,0,1
    SET_0_4 v13, v17, v21, v25 // oc:2,3,2,3
    SET_0_4 v14, v18, v22, v26 // oc:4,5,4,5
    SET_0_4 v15, v19, v23, v27 // oc:6,7,6,7
LoopSz_TILE_8:
    ld1 {v3.16b, v4.16b, v5.16b, v6.16b}, [x11], x22    // src: E0-E7

    // int4->int8
    ld1 {v0.16b, v1.16b}, [x12], #32                   // weight
    ushr v8.16b, v0.16b, #4 // oc:0-1
    ushr v9.16b, v1.16b, #4 // oc:2-3
    and v10.16b, v0.16b, v28.16b // oc:4-5
    and v11.16b, v1.16b, v28.16b // oc:6-7

    subs x13, x13, #1
    .inst 0x4e88a46c // smmla v12.4s, v3.16b, v8.16b // tile0-oc0, tile0-oc1, tile1-oc0, tile1-oc1
    .inst 0x4e89a46d // smmla v13.4s, v3.16b, v9.16b // tile0-oc2, tile0-oc3, tile1-oc2, tile1-oc3
    .inst 0x4e8aa46e // smmla v14.4s, v3.16b, v10.16b // tile0-oc4, tile0-oc5, tile1-oc4, tile1-oc5
    .inst 0x4e8ba46f // smmla v15.4s, v3.16b, v11.16b // tile0-oc6, tile0-oc7, tile1-oc6, tile1-oc7
    
    .inst 0x4e88a490 // smmla v16.4s, v4.16b, v8.16b // tile2-oc0, tile2-oc1, tile3-oc0, tile3-oc1
    .inst 0x4e89a491 // smmla v17.4s, v4.16b, v9.16b // tile2-oc2, tile2-oc3, tile3-oc2, tile3-oc3
    .inst 0x4e8aa492 // smmla v18.4s, v4.16b, v10.16b // tile2-oc4, tile2-oc5, tile3-oc4, tile3-oc5
    .inst 0x4e8ba493 // smmla v19.4s, v4.16b, v11.16b // tile2-oc6, tile2-oc7, tile3-oc6, tile3-oc7
    
    .inst 0x4e88a4b4 // smmla v20.4s, v5.16b, v8.16b // tile4-oc0, tile4-oc1, tile5-oc0, tile5-oc1
    .inst 0x4e89a4b5 // smmla v21.4s, v5.16b, v9.16b // tile4-oc2, tile4-oc3, tile5-oc2, tile5-oc3
    .inst 0x4e8aa4b6 // smmla v22.4s, v5.16b, v10.16b // tile4-oc4, tile4-oc5, tile5-oc4, tile5-oc5
    .inst 0x4e8ba4b7 // smmla v23.4s, v5.16b, v11.16b // tile4-oc6, tile4-oc7, tile5-oc6, tile5-oc7

    .inst 0x4e88a4d8 // smmla v24.4s, v6.16b, v8.16b // tile6-oc0, tile6-oc1, tile7-oc0, tile7-oc1
    .inst 0x4e89a4d9 // smmla v25.4s, v6.16b, v9.16b // tile6-oc2, tile6-oc3, tile7-oc2, tile7-oc3
    .inst 0x4e8aa4da // smmla v26.4s, v6.16b, v10.16b // tile6-oc4, tile6-oc5, tile7-oc4, tile7-oc5
    .inst 0x4e8ba4db // smmla v27.4s, v6.16b, v11.16b // tile6-oc6, tile6-oc7, tile7-oc6, tile7-oc7
    bne LoopSz_TILE_8

LoopSzEnd_TILE_8:
    uzp1 v0.2d, v12.2d, v13.2d // E0: oc:0-3
    uzp2 v1.2d, v12.2d, v13.2d // E1: oc:0-3
    uzp1 v8.2d, v14.2d, v15.2d // E0: oc:4-7
    uzp2 v9.2d, v14.2d, v15.2d // E1: oc:4-7

    uzp1 v2.2d, v16.2d, v17.2d // E2: oc:0-3
    uzp2 v3.2d, v16.2d, v17.2d // E3: oc:0-3
    uzp1 v10.2d, v18.2d, v19.2d // E2: oc:4-7
    uzp2 v11.2d, v18.2d, v19.2d // E3: oc:4-7

    uzp1 v4.2d, v20.2d, v21.2d // E4: oc:0-3
    uzp2 v5.2d, v20.2d, v21.2d // E5: oc:0-3
    uzp1 v12.2d, v22.2d, v23.2d // E4: oc:4-7
    uzp2 v13.2d, v22.2d, v23.2d // E5: oc:4-7

    uzp1 v6.2d, v24.2d, v25.2d // E6: oc:0-3
    uzp2 v7.2d, v24.2d, v25.2d // E7: oc:0-3
    uzp1 v14.2d, v26.2d, v27.2d // E6: oc:4-7
    uzp2 v15.2d, v26.2d, v27.2d // E7: oc:4-7
    Int32ToFloat v0, v1, v2, v3
    Int32ToFloat v4, v5, v6, v7
    Int32ToFloat v8, v9, v10, v11
    Int32ToFloat v12, v13, v14, v15

Tile8Quan:
    ld1 {v20.4s, v21.4s}, [x12], #32  // scale
    ld1 {v22.4s, v23.4s}, [x8] // x kernel sum
    ld1 {v25.4s, v26.4s}, [x12], #32 // weight quan zeropoint
    MUL_SCALE v20, v0, v1, v2, v3
    MUL_SCALE v20, v4, v5, v6, v7
    MUL_SCALE v21, v8, v9, v10, v11
    MUL_SCALE v21, v12, v13, v14, v15
    add x8, x8, x22, LSR #1

    ld1 {v18.4s, v19.4s}, [x23], x25 // input dequant scale
    MUL_INPUT_SCALE v18, v0, v1, v2, v3
    MUL_INPUT_SCALE v19, v4, v5, v6, v7
    MUL_INPUT_SCALE v18, v8, v9, v10, v11
    MUL_INPUT_SCALE v19, v12, v13, v14, v15

    TILE8_MLA:
    MLA_WEIGHTZERO v0, v22, v25, 0
    MLA_WEIGHTZERO v1, v22, v25, 1
    MLA_WEIGHTZERO v2, v22, v25, 2
    MLA_WEIGHTZERO v3, v22, v25, 3
    MLA_WEIGHTZERO v4, v23, v25, 0
    MLA_WEIGHTZERO v5, v23, v25, 1
    MLA_WEIGHTZERO v6, v23, v25, 2
    MLA_WEIGHTZERO v7, v23, v25, 3

    MLA_WEIGHTZERO v8, v22, v26, 0
    MLA_WEIGHTZERO v9, v22, v26, 1
    MLA_WEIGHTZERO v10, v22, v26, 2
    MLA_WEIGHTZERO v11, v22, v26, 3
    MLA_WEIGHTZERO v12, v23, v26, 0
    MLA_WEIGHTZERO v13, v23, v26, 1
    MLA_WEIGHTZERO v14, v23, v26, 2
    MLA_WEIGHTZERO v15, v23, v26, 3

    cbz x27, TILE8_ADD_DSTV
    ld1 {v22.4s, v23.4s}, [x27], x25 // input dequant bias
    ld1 {v25.4s, v26.4s}, [x28], #32 // weight kernel sum
    MLA_WEIGHTZERO v0, v22, v25, 0
    MLA_WEIGHTZERO v1, v22, v25, 1
    MLA_WEIGHTZERO v2, v22, v25, 2
    MLA_WEIGHTZERO v3, v22, v25, 3
    MLA_WEIGHTZERO v4, v23, v25, 0
    MLA_WEIGHTZERO v5, v23, v25, 1
    MLA_WEIGHTZERO v6, v23, v25, 2
    MLA_WEIGHTZERO v7, v23, v25, 3

    MLA_WEIGHTZERO v8, v22, v26, 0
    MLA_WEIGHTZERO v9, v22, v26, 1
    MLA_WEIGHTZERO v10, v22, v26, 2
    MLA_WEIGHTZERO v11, v22, v26, 3
    MLA_WEIGHTZERO v12, v23, v26, 0
    MLA_WEIGHTZERO v13, v23, v26, 1
    MLA_WEIGHTZERO v14, v23, v26, 2
    MLA_WEIGHTZERO v15, v23, v26, 3

    TILE8_ADD_DSTV:
    cbz x19, TILE8_TEMP_BUFFER
    ld1 {v20.4s, v21.4s, v22.4s, v23.4s}, [x10], #64
    ld1 {v24.4s, v25.4s, v26.4s, v27.4s}, [x10], #64
    ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x10], #64
    ADD_FLOAT v0, v1, v2, v3, v20, v21, v22, v23
    ADD_FLOAT v4, v5, v6, v7, v24, v25, v26, v27
    ld1 {v20.4s, v21.4s, v22.4s, v23.4s}, [x10]
    ADD_FLOAT v8, v9, v10, v11, v16, v17, v18, v19
    ADD_FLOAT v12, v13, v14, v15, v20, v21, v22, v23
    sub x10, x10, #192

    TILE8_TEMP_BUFFER:
    add x19, x19, #1
    cmp x19, x26
    beq TILE8_POST
    st1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x10], #64
    st1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x10], #64
    st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x10], #64
    st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [x10]
    sub x10, x10, #192
    b TILE8_BLOCKNUM

    TILE8_POST:
    sub x24, x24, #2 // dz-2
    cbz x9, TILE8_RELU
    ld1 {v16.4s, v17.4s}, [x20], #32 // bias
    ADD_BIAS_FLOAT v0, v1, v2, v3, v16
    ADD_BIAS_FLOAT v4, v5, v6, v7, v16
    ADD_BIAS_FLOAT v8, v9, v10, v11, v17
    ADD_BIAS_FLOAT v12, v13, v14, v15, v17

    TILE8_RELU:
    cbz x14, TILE8_STORE
    ReLU_FP32 v0, v1, v2, v3, v30, v31
    ReLU_FP32 v4, v5, v6, v7, v30, v31
    ReLU_FP32 v8, v9, v10, v11, v30, v31
    ReLU_FP32 v12, v13, v14, v15, v30, v31

    TILE8_STORE:
    st1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x6], #64
    st1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x6], x4
    st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x6], #64
    st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [x6], x4

    Tile8LoopCheck:
    cbz x24, Tile8End
    mov x8, x15       // revert input kernel sum
    mov x23, x21      // revert input dequant scale
    cbz x27, LoopDz_TILE_8
    REVERT_INPUT_DEQUANT_BIAS x27, x19, x26, x25
    b LoopDz_TILE_8

LoopDz4_TILE_8:
    mov x11, x1 // src
    mov x19, #0

L4_TILE8_BLOCKNUM:
    mov x13, x3 // src_depth_quad

    SET_0_4 v12, v16, v20, v24 // oc:0,1,0,1
    SET_0_4 v13, v17, v21, v25 // oc:2,3,2,3
L4_LoopSz_TILE_8:
    ld1 {v3.16b, v4.16b, v5.16b, v6.16b}, [x11], x22    // src: E0-E7

    // int4->int8
    ld1 {v0.16b, v1.16b}, [x12], #32                   // weight
    ushr v8.16b, v0.16b, #4 // oc:0-1
    ushr v9.16b, v1.16b, #4 // oc:2-3

    subs x13, x13, #1
    .inst 0x4e88a46c // smmla v12.4s, v3.16b, v8.16b // tile0-oc0, tile0-oc1, tile1-oc0, tile1-oc1
    .inst 0x4e89a46d // smmla v13.4s, v3.16b, v9.16b // tile0-oc2, tile0-oc3, tile1-oc2, tile1-oc3
    
    .inst 0x4e88a490 // smmla v16.4s, v4.16b, v8.16b // tile2-oc0, tile2-oc1, tile3-oc0, tile3-oc1
    .inst 0x4e89a491 // smmla v17.4s, v4.16b, v9.16b // tile2-oc2, tile2-oc3, tile3-oc2, tile3-oc3
    
    .inst 0x4e88a4b4 // smmla v20.4s, v5.16b, v8.16b // tile4-oc0, tile4-oc1, tile5-oc0, tile5-oc1
    .inst 0x4e89a4b5 // smmla v21.4s, v5.16b, v9.16b // tile4-oc2, tile4-oc3, tile5-oc2, tile5-oc3

    .inst 0x4e88a4d8 // smmla v24.4s, v6.16b, v8.16b // tile6-oc0, tile6-oc1, tile7-oc0, tile7-oc1
    .inst 0x4e89a4d9 // smmla v25.4s, v6.16b, v9.16b // tile6-oc2, tile6-oc3, tile7-oc2, tile7-oc3
    bne L4_LoopSz_TILE_8

    uzp1 v0.2d, v12.2d, v13.2d // E0: oc:0-3
    uzp2 v1.2d, v12.2d, v13.2d // E1: oc:0-3

    uzp1 v2.2d, v16.2d, v17.2d // E2: oc:0-3
    uzp2 v3.2d, v16.2d, v17.2d // E3: oc:0-3

    uzp1 v4.2d, v20.2d, v21.2d // E4: oc:0-3
    uzp2 v5.2d, v20.2d, v21.2d // E5: oc:0-3

    uzp1 v6.2d, v24.2d, v25.2d // E6: oc:0-3
    uzp2 v7.2d, v24.2d, v25.2d // E7: oc:0-3
    Int32ToFloat v0, v1, v2, v3
    Int32ToFloat v4, v5, v6, v7

L4_Tile8Quan:
    ld1 {v20.4s}, [x12] // scale
    add x12, x12, #32
    ld1 {v22.4s, v23.4s}, [x8] // x kernel sum
    ld1 {v25.4s}, [x12] // weight quan zeropoint
    add x12, x12, #32
    add x8, x8, x22, LSR #1
    MUL_SCALE v20, v0, v1, v2, v3
    MUL_SCALE v20, v4, v5, v6, v7

    ld1 {v18.4s, v19.4s}, [x23], x25
    MUL_INPUT_SCALE v18, v0, v1, v2, v3
    MUL_INPUT_SCALE v19, v4, v5, v6, v7

    L4_TILE8_MLA:
    MLA_WEIGHTZERO v0, v22, v25, 0
    MLA_WEIGHTZERO v1, v22, v25, 1
    MLA_WEIGHTZERO v2, v22, v25, 2
    MLA_WEIGHTZERO v3, v22, v25, 3
    MLA_WEIGHTZERO v4, v23, v25, 0
    MLA_WEIGHTZERO v5, v23, v25, 1
    MLA_WEIGHTZERO v6, v23, v25, 2
    MLA_WEIGHTZERO v7, v23, v25, 3

    cbz x27, L4_TILE8_ADD_DSTV
    ld1 {v22.4s, v23.4s}, [x27], x25 // input dequant bias
    ld1 {v25.4s}, [x28] // weight kernel sum
    MLA_WEIGHTZERO v0, v22, v25, 0
    MLA_WEIGHTZERO v1, v22, v25, 1
    MLA_WEIGHTZERO v2, v22, v25, 2
    MLA_WEIGHTZERO v3, v22, v25, 3
    MLA_WEIGHTZERO v4, v23, v25, 0
    MLA_WEIGHTZERO v5, v23, v25, 1
    MLA_WEIGHTZERO v6, v23, v25, 2
    MLA_WEIGHTZERO v7, v23, v25, 3
    add x28, x28, #32

    L4_TILE8_ADD_DSTV:
    cbz x19, L4_TILE8_TEMP_BUFFER
    ld1 {v20.4s, v21.4s, v22.4s, v23.4s}, [x10], #64
    ld1 {v24.4s, v25.4s, v26.4s, v27.4s}, [x10]
    ADD_FLOAT v0, v1, v2, v3, v20, v21, v22, v23
    ADD_FLOAT v4, v5, v6, v7, v24, v25, v26, v27
    sub x10, x10, #64

    L4_TILE8_TEMP_BUFFER:
    add x19, x19, #1
    cmp x19, x26
    beq L4_TILE8_POST
    st1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x10], #64
    st1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x10]
    sub x10, x10, #64
    b L4_TILE8_BLOCKNUM

    L4_TILE8_POST:
    cbz x9, L4_TILE8_RELU
    ld1 {v16.4s}, [x20], #16 // bias
    ADD_BIAS_FLOAT v0, v1, v2, v3, v16
    ADD_BIAS_FLOAT v4, v5, v6, v7, v16

    L4_TILE8_RELU:
    cbz x14, L4_TILE8_STORE
    ReLU_FP32 v0, v1, v2, v3, v30, v31
    ReLU_FP32 v4, v5, v6, v7, v30, v31

    L4_TILE8_STORE:
    st1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x6], #64
    st1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x6], x4

Tile8End:
    add x0, x0, #128
    sub x7, x7, #8
    add x1, x1, #64
    add x23, x21, #32 // input dequant scale
    add x8, x15, #32  // input kernel sum
    add x4, x4, #64 // Revert x4 for following tile.
    cbz x27, TILE_4
    REVERT_INPUT_DEQUANT_BIAS x27, x19, x26, x25
    REVERT_WEIGHT_KERNEL_SUM x28, x24, x26, x5
    add x27, x27, #32 // update input dequant bias: + 8 * sizeof(float)

TILE_4:
    cmp x7, #4
    blt TILE_2
    mov x24, x5 // dst_depth_quad
    mov x6, x0 // dst
    mov x12, x2 // weight
    mov x20, x9 // bias
    mov x15, x8
    mov x21, x23

LoopDz_TILE_4:
    cmp x24, #2
    mov x11, x1 // src
    mov x19, #0
    blt LoopDz4_TILE_4
TILE4_BLOCKNUM:
    mov x13, x3 // src_depth_quad
    SET_0_4 v12, v13, v14, v15
    SET_0_4 v16, v17, v18, v19

LoopSz_TILE_4:
    // int4->int8
    ld1 {v0.16b, v1.16b}, [x12], #32                   // weight
    ushr v8.16b, v0.16b, #4 // oc:0-1
    ushr v9.16b, v1.16b, #4 // oc:2-3
    and v10.16b, v0.16b, v28.16b // oc:4-5
    and v11.16b, v1.16b, v28.16b // oc:6-7

    ld1 {v4.16b, v5.16b}, [x11], x22   // src
    subs x13, x13, #1
    .inst 0x4e88a48c // smmla v12.4s, v4.16b, v8.16b // tile0-oc0, tile0-oc1, tile1-oc0, tile1-oc1
    .inst 0x4e89a48d // smmla v13.4s, v4.16b, v9.16b // tile0-oc2, tile0-oc3, tile1-oc2, tile1-oc3
    .inst 0x4e8aa48e // smmla v14.4s, v4.16b, v10.16b // tile0-oc4, tile0-oc5, tile1-oc4, tile1-oc5
    .inst 0x4e8ba48f // smmla v15.4s, v4.16b, v11.16b // tile0-oc6, tile0-oc7, tile1-oc6, tile1-oc7
    
    .inst 0x4e88a4b0 // smmla v16.4s, v5.16b, v8.16b // tile2-oc0, tile2-oc1, tile3-oc0, tile3-oc1
    .inst 0x4e89a4b1 // smmla v17.4s, v5.16b, v9.16b // tile2-oc2, tile2-oc3, tile3-oc2, tile3-oc3
    .inst 0x4e8aa4b2 // smmla v18.4s, v5.16b, v10.16b // tile2-oc4, tile2-oc5, tile3-oc4, tile3-oc5
    .inst 0x4e8ba4b3 // smmla v19.4s, v5.16b, v11.16b // tile2-oc6, tile2-oc7, tile3-oc6, tile3-oc7
    bne LoopSz_TILE_4
LoopSzEnd_TILE_4:
    uzp1 v0.2d, v12.2d, v13.2d // E0: oc:0-3
    uzp2 v1.2d, v12.2d, v13.2d // E1: oc:0-3
    uzp1 v4.2d, v14.2d, v15.2d // E0: oc:4-7
    uzp2 v5.2d, v14.2d, v15.2d // E1: oc:4-7

    uzp1 v2.2d, v16.2d, v17.2d
    uzp2 v3.2d, v16.2d, v17.2d
    uzp1 v6.2d, v18.2d, v19.2d
    uzp2 v7.2d, v18.2d, v19.2d
    Int32ToFloat v0, v1, v2, v3
    Int32ToFloat v4, v5, v6, v7

Tile4Quan:
    ld1 {v20.4s, v21.4s}, [x12], #32  // scale
    ld1 {v22.4s}, [x8] // x kernel sum
    ld1 {v25.4s, v26.4s}, [x12], #32 // weight quan zeropoint
    MUL_SCALE v20, v0, v1, v2, v3
    MUL_SCALE v21, v4, v5, v6, v7
    add x8, x8, x22, LSR #1

    ld1 {v27.4s}, [x23], x25
    MUL_INPUT_SCALE v27, v0, v1, v2, v3
    MUL_INPUT_SCALE v27, v4, v5, v6, v7

    TILE4_MLA:
    MLA_WEIGHTZERO v0, v22, v25, 0 // tile:0, oc:0-3
    MLA_WEIGHTZERO v1, v22, v25, 1 // tile:1, oc:0-3
    MLA_WEIGHTZERO v2, v22, v25, 2 // tile:2, oc:0-3
    MLA_WEIGHTZERO v3, v22, v25, 3 // tile:3, oc:0-3
    MLA_WEIGHTZERO v4, v22, v26, 0 // tile:0, oc:4-7
    MLA_WEIGHTZERO v5, v22, v26, 1 // tile:1, oc:4-7
    MLA_WEIGHTZERO v6, v22, v26, 2 // tile:2, oc:4-7
    MLA_WEIGHTZERO v7, v22, v26, 3 // tile:3, oc:4-7

    cbz x27, TILE4_ADD_DSTV
    ld1 {v22.4s}, [x27], x25 // input dequant bias
    ld1 {v25.4s, v26.4s}, [x28], #32 // weight kernel sum
    MLA_WEIGHTZERO v0, v22, v25, 0 // tile:0, oc:0-3
    MLA_WEIGHTZERO v1, v22, v25, 1 // tile:1, oc:0-3
    MLA_WEIGHTZERO v2, v22, v25, 2 // tile:2, oc:0-3
    MLA_WEIGHTZERO v3, v22, v25, 3 // tile:3, oc:0-3
    MLA_WEIGHTZERO v4, v22, v26, 0 // tile:0, oc:4-7
    MLA_WEIGHTZERO v5, v22, v26, 1 // tile:1, oc:4-7
    MLA_WEIGHTZERO v6, v22, v26, 2 // tile:2, oc:4-7
    MLA_WEIGHTZERO v7, v22, v26, 3 // tile:3, oc:4-7


    TILE4_ADD_DSTV:
    cbz x19, TILE4_L8_ACCUM_BUFFER
    ld1 {v15.4s, v16.4s, v17.4s, v18.4s}, [x10], #64
    ld1 {v19.4s, v20.4s, v21.4s, v22.4s}, [x10]
    ADD_FLOAT v0, v1, v2, v3, v15, v16, v17, v18
    ADD_FLOAT v4, v5, v6, v7, v19, v20, v21, v22
    sub x10, x10, #64

    TILE4_L8_ACCUM_BUFFER:
    add x19, x19, #1
    cmp x19, x26
    beq TILE4_POST
    st1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x10], #64
    st1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x10]
    sub x10, x10, #64
    b TILE4_BLOCKNUM

    TILE4_POST:
    sub x24, x24, #2
    cbz x9, TILE4_RELU
    ld1 {v16.4s, v17.4s}, [x20], #32 // bias
    ADD_BIAS_FLOAT v0, v1, v2, v3, v16
    ADD_BIAS_FLOAT v4, v5, v6, v7, v17

    TILE4_RELU:
    cbz x14, TILE4_STORE
    ReLU_FP32 v0, v1, v2, v3, v30, v31
    ReLU_FP32 v4, v5, v6, v7, v30, v31

    TILE4_STORE:
    st1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x6], x4
    st1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x6], x4

Tile4LoopCheck:
    cbz x24, Tile4End
    mov x8, x15
    mov x23, x21
    cbz x27, LoopDz_TILE_4
    REVERT_INPUT_DEQUANT_BIAS x27, x19, x26, x25
    b LoopDz_TILE_4

LoopDz4_TILE_4:
    mov x11, x1 // src
    mov x19, #0
L4_TILE4_BLOCKNUM:
    mov x13, x3 // src_depth_quad
    SET_0_4 v12, v13, v16, v17

L4_LoopSz_TILE_4:
    // int4->int8
    ld1 {v0.16b, v1.16b}, [x12], #32                   // weight
    ushr v8.16b, v0.16b, #4 // oc:0-1
    ushr v9.16b, v1.16b, #4 // oc:2-3

    ld1 {v4.16b, v5.16b}, [x11], x22   // src
    subs x13, x13, #1
    .inst 0x4e88a48c // smmla v12.4s, v4.16b, v8.16b // tile0-oc0, tile0-oc1, tile1-oc0, tile1-oc1
    .inst 0x4e89a48d // smmla v13.4s, v4.16b, v9.16b // tile0-oc2, tile0-oc3, tile1-oc2, tile1-oc3
    
    .inst 0x4e88a4b0 // smmla v16.4s, v5.16b, v8.16b // tile2-oc0, tile2-oc1, tile3-oc0, tile3-oc1
    .inst 0x4e89a4b1 // smmla v17.4s, v5.16b, v9.16b // tile2-oc2, tile2-oc3, tile3-oc2, tile3-oc3
    bne L4_LoopSz_TILE_4

    uzp1 v0.2d, v12.2d, v13.2d // E0: oc:0-3
    uzp2 v1.2d, v12.2d, v13.2d // E1: oc:0-3
    uzp1 v2.2d, v16.2d, v17.2d
    uzp2 v3.2d, v16.2d, v17.2d
    Int32ToFloat v0, v1, v2, v3

L4_Tile4Quan:
    ld1 {v20.4s}, [x12]  // scale
    add x12, x12, #32
    ld1 {v22.4s}, [x8] // x kernel sum
    ld1 {v25.4s}, [x12] // weight quan zeropoint
    add x12, x12, #32
    MUL_SCALE v20, v0, v1, v2, v3
    add x8, x8, x22, LSR #1

    ld1 {v27.4s}, [x23], x25
    MUL_INPUT_SCALE v27, v0, v1, v2, v3

    L4_TILE4_MLA:
    MLA_WEIGHTZERO v0, v22, v25, 0 // tile:0, oc:0-3
    MLA_WEIGHTZERO v1, v22, v25, 1 // tile:1, oc:0-3
    MLA_WEIGHTZERO v2, v22, v25, 2 // tile:2, oc:0-3
    MLA_WEIGHTZERO v3, v22, v25, 3 // tile:3, oc:0-3

    cbz x27, L4_TILE4_ADD_DSTV
    ld1 {v22.4s}, [x27], x25 // input dequant bias
    ld1 {v25.4s}, [x28] // weight kernel sum
    MLA_WEIGHTZERO v0, v22, v25, 0 // tile:0, oc:0-3
    MLA_WEIGHTZERO v1, v22, v25, 1 // tile:1, oc:0-3
    MLA_WEIGHTZERO v2, v22, v25, 2 // tile:2, oc:0-3
    MLA_WEIGHTZERO v3, v22, v25, 3 // tile:3, oc:0-3
    add x28, x28, #32

    L4_TILE4_ADD_DSTV:
    cbz x19, L4_TILE4_ACCUM_BUFFER
    ld1 {v15.4s, v16.4s, v17.4s, v18.4s}, [x10]
    ADD_FLOAT v0, v1, v2, v3, v15, v16, v17, v18

    L4_TILE4_ACCUM_BUFFER:
    add x19, x19, #1
    cmp x19, x26
    beq L4_TILE4_POST
    st1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x10]
    b L4_TILE4_BLOCKNUM

    L4_TILE4_POST:
    cbz x9, L4_TILE4_RELU
    ld1 {v16.4s}, [x20], #16 // bias
    ADD_BIAS_FLOAT v0, v1, v2, v3, v16

    L4_TILE4_RELU:
    cbz x14, L4_TILE4_STORE
    ReLU_FP32 v0, v1, v2, v3, v30, v31

    L4_TILE4_STORE:
    st1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x6], x4

Tile4End:
    add x0, x0, #64
    sub x7, x7, #4
    add x1, x1, #32
    add x8, x15, #16
    add x23, x21, #16
    cbz x27, TILE_2
    REVERT_INPUT_DEQUANT_BIAS x27, x19, x26, x25
    REVERT_WEIGHT_KERNEL_SUM x28, x24, x26, x5
    add x27, x27, #16

TILE_2:
    cmp x7, #2
    blt TILE_1
    mov x24, x5 // dst_depth_quad
    mov x6, x0 // dst
    mov x12, x2 // weight
    mov x20, x9 // bias
    mov x15, x8
    mov x21, x23

LoopDz_TILE_2:
    cmp x24, #2
    blt LoopDz4_TILE_2
    mov x11, x1 // src
    mov x19, #0
TILE2_BLOCKNUM:
    mov x13, x3 // src_depth_quad
    SET_0_4 v12, v13, v14, v15
LoopSz_TILE_2:
    // int4->int8
    ld1 {v0.16b, v1.16b}, [x12], #32                   // weight
    ushr v8.16b, v0.16b, #4 // oc:0-1
    ushr v9.16b, v1.16b, #4 // oc:2-3
    and v10.16b, v0.16b, v28.16b // oc:4-5
    and v11.16b, v1.16b, v28.16b // oc:6-7

    ld1 {v4.16b}, [x11], x22           // src
    .inst 0x4e88a48c // smmla v12.4s, v4.16b, v8.16b // tile0-oc0, tile0-oc1, tile1-oc0, tile1-oc1
    .inst 0x4e89a48d // smmla v13.4s, v4.16b, v9.16b // tile0-oc2, tile0-oc3, tile1-oc2, tile1-oc3
    .inst 0x4e8aa48e // smmla v14.4s, v4.16b, v10.16b // tile0-oc4, tile0-oc5, tile1-oc4, tile1-oc5
    .inst 0x4e8ba48f // smmla v15.4s, v4.16b, v11.16b // tile0-oc6, tile0-oc7, tile1-oc6, tile1-oc7
    subs x13, x13, #1
    bne LoopSz_TILE_2

    uzp1 v0.2d, v12.2d, v13.2d // E0: oc:0-3
    uzp2 v1.2d, v12.2d, v13.2d // E1: oc:0-3
    uzp1 v2.2d, v14.2d, v15.2d // E0: oc:4-7
    uzp2 v3.2d, v14.2d, v15.2d // E1: oc:4-7
    Int32ToFloat v0, v1, v2, v3

Tile2Quan:
    ld1 {v20.4s, v21.4s}, [x12], #32  // scale
    ld1 {v22.d}[0], [x8] // x kernel sum
    ld1 {v25.4s, v26.4s}, [x12], #32 // weight quan zeropoint
    fmul v0.4s, v0.4s, v20.4s
    fmul v1.4s, v1.4s, v20.4s
    fmul v2.4s, v2.4s, v21.4s
    fmul v3.4s, v3.4s, v21.4s
    add x8, x8, x22, LSR #1

    ld1 {v27.d}[0], [x23], x25
    fmul v0.4s, v0.4s, v27.s[0]
    fmul v1.4s, v1.4s, v27.s[1]
    fmul v2.4s, v2.4s, v27.s[0]
    fmul v3.4s, v3.4s, v27.s[1]

    TILE2_MLA:
    MLA_WEIGHTZERO v0, v22, v25, 0 // tile:0, oc:0-3
    MLA_WEIGHTZERO v1, v22, v25, 1 // tile:1, oc:0-3
    MLA_WEIGHTZERO v2, v22, v26, 0 // tile:0, oc:4-7
    MLA_WEIGHTZERO v3, v22, v26, 1 // tile:1, oc:4-7

    cbz x27, TILE2_ADD_DSTV
    ld1 {v22.2s}, [x27], x25 // input dequant bias
    ld1 {v25.4s, v26.4s}, [x28], #32 // weight kernel sum
    MLA_WEIGHTZERO v0, v22, v25, 0 // tile:0, oc:0-3
    MLA_WEIGHTZERO v1, v22, v25, 1 // tile:1, oc:0-3
    MLA_WEIGHTZERO v2, v22, v26, 0 // tile:0, oc:4-7
    MLA_WEIGHTZERO v3, v22, v26, 1 // tile:1, oc:4-7

    TILE2_ADD_DSTV:
    cbz x19, TILE2_L8_ACCUM_BUFFER
    ld1 {v15.4s, v16.4s, v17.4s, v18.4s}, [x10]
    ADD_FLOAT v0, v1, v2, v3, v15, v16, v17, v18

    TILE2_L8_ACCUM_BUFFER:
    add x19, x19, #1
    cmp x19, x26
    beq TILE2_POST
    st1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x10]
    b TILE2_BLOCKNUM

    TILE2_POST:
    sub x24, x24, #2
    cbz x9, TILE2_RELU
    ld1 {v16.4s, v17.4s}, [x20], #32 // bias
    fadd v0.4s, v0.4s, v16.4s
    fadd v1.4s, v1.4s, v16.4s
    fadd v2.4s, v2.4s, v17.4s
    fadd v3.4s, v3.4s, v17.4s

    TILE2_RELU:
    cbz x14, TILE2_STORE
    ReLU_FP32 v0, v1, v2, v3, v30, v31
    TILE2_STORE:
    st1 {v0.4s, v1.4s}, [x6], x4
    st1 {v2.4s, v3.4s}, [x6], x4

Tile2LoopCheck:
    cbz x24, Tile2End
    mov x8, x15
    mov x23, x21
    cbz x27, LoopDz_TILE_2
    REVERT_INPUT_DEQUANT_BIAS x27, x19, x26, x25
    bge LoopDz_TILE_2
    

LoopDz4_TILE_2:
    mov x11, x1 // src
    mov x19, #0
L4_TILE2_BLOCKNUM:
    mov x13, x3 // src_depth_quad
    movi v12.16b, #0
    movi v13.16b, #0
L4_LoopSz_TILE_2:
    // int4->int8
    ld1 {v0.16b, v1.16b}, [x12], #32                   // weight
    ushr v8.16b, v0.16b, #4 // oc:0-1
    ushr v9.16b, v1.16b, #4 // oc:2-3
    ld1 {v4.16b}, [x11], x22           // src
    .inst 0x4e88a48c // smmla v12.4s, v4.16b, v8.16b // tile0-oc0, tile0-oc1, tile1-oc0, tile1-oc1
    .inst 0x4e89a48d // smmla v13.4s, v4.16b, v9.16b // tile0-oc2, tile0-oc3, tile1-oc2, tile1-oc3
    subs x13, x13, #1
    bne L4_LoopSz_TILE_2

    uzp1 v0.2d, v12.2d, v13.2d // E0: oc:0-3
    uzp2 v1.2d, v12.2d, v13.2d // E1: oc:0-3
    scvtf v0.4s, v0.4s
    scvtf v1.4s, v1.4s

L4_Tile2Quan:
    ld1 {v20.4s}, [x12]  // scale
    ld1 {v22.d}[0], [x8] // x kernel sum
    add x12, x12, #32
    ld1 {v25.4s}, [x12] // weight quan zeropoint
    fmul v0.4s, v0.4s, v20.4s
    fmul v1.4s, v1.4s, v20.4s
    add x8, x8, x22, LSR #1
    ld1 {v27.d}[0], [x23], x25
    fmul v0.4s, v0.4s, v27.s[0]
    fmul v1.4s, v1.4s, v27.s[1]
    add x12, x12, #32

    L4_TILE2_MLA:
    MLA_WEIGHTZERO v0, v22, v25, 0 // tile:0, oc:0-3
    MLA_WEIGHTZERO v1, v22, v25, 1 // tile:1, oc:0-3

    cbz x27, L4_TILE2_ADD_DSTV
    ld1 {v22.2s}, [x27], x25 // input dequant bias
    ld1 {v25.4s}, [x28] // weight kernel sum
    MLA_WEIGHTZERO v0, v22, v25, 0 // tile:0, oc:0-3
    MLA_WEIGHTZERO v1, v22, v25, 1 // tile:1, oc:0-3
    add x28, x28, #32

    L4_TILE2_ADD_DSTV:
    cbz x19, L4_TILE2_ACCUM_BUFFER
    ld1 {v15.4s, v16.4s}, [x10]
    fadd v0.4s, v0.4s, v15.4s
    fadd v1.4s, v1.4s, v16.4s

    L4_TILE2_ACCUM_BUFFER:
    add x19, x19, #1
    cmp x19, x26
    beq L4_TILE2_POST
    st1 {v0.4s, v1.4s}, [x10]
    b L4_TILE2_BLOCKNUM

    L4_TILE2_POST:
    cbz x9, L4_TILE2_RELU
    ld1 {v16.4s}, [x20], #16 // bias
    fadd v0.4s, v0.4s, v16.4s
    fadd v1.4s, v1.4s, v16.4s

    L4_TILE2_RELU:
    cbz x14, L4_TILE2_STORE
    ReLU_FP32 v0, v1, v2, v3, v30, v31
    L4_TILE2_STORE:
    st1 {v0.4s, v1.4s}, [x6], x4


Tile2End:
    add x0, x0, #32
    sub x7, x7, #2
    add x1, x1, #16
    add x8, x15, #8
    add x23, x21, #8
    cbz x27, TILE_1
    REVERT_INPUT_DEQUANT_BIAS x27, x19, x26, x25
    REVERT_WEIGHT_KERNEL_SUM x28, x24, x26, x5
    add x27, x27, #8

TILE_1:
    cmp x7, #1
    blt End
    mov x24, x5 // dst_depth_quad
    mov x6, x0 // dst
    mov x12, x2 // weight
    mov x20, x9 // bias
    mov x15, x8
    mov x21, x23

LoopDz_TILE_1:
    cmp x24, #2
    blt LoopDz4_TILE_1
    mov x11, x1             // src
    mov x19, #0
TILE1_BLOCKNUM:
    mov x13, x3             // src_depth_quad

    movi v16.4s, #0
    movi v17.4s, #0
    movi v18.4s, #0
    movi v19.4s, #0
cmp x13, #4
blt LoopSz_TILE_1_lu1

LoopSz1_TILE_1_lu4:
    ld1 {v5.16b, v6.16b, v7.16b, v8.16b}, [x12], #64     // weight
    ld1 {v9.16b, v10.16b, v11.16b, v12.16b}, [x12], #64
    ld1 {v0.8b}, [x11], x22                              // src
    ld1 {v1.8b}, [x11], x22
    ld1 {v2.8b}, [x11], x22
    ld1 {v3.8b}, [x11], x22

    // int4->int8
    ushr v4.16b, v5.16b, #4
    ushr v14.16b, v6.16b, #4
    and v13.16b, v5.16b, v28.16b
    and v15.16b, v6.16b, v28.16b

    ushr v20.16b, v7.16b, #4
    ushr v21.16b, v8.16b, #4
    and v22.16b, v7.16b, v28.16b
    and v23.16b, v8.16b, v28.16b

    ushr v24.16b, v9.16b, #4
    ushr v25.16b, v10.16b, #4
    and v26.16b, v9.16b, v28.16b
    and v27.16b, v10.16b, v28.16b

    ushr v5.16b, v11.16b, #4
    ushr v6.16b, v12.16b, #4
    and v7.16b, v11.16b, v28.16b
    and v8.16b, v12.16b, v28.16b

    sub x13, x13, #4

    .inst 0x4e84a410 // smmla v16.4s, v0.16b, v4.16b
    .inst 0x4e8ea411 // smmla v17.4s, v0.16b, v14.16b
    .inst 0x4e8da412 // smmla v18.4s, v0.16b, v13.16b
    .inst 0x4e8fa413 // smmla v19.4s, v0.16b, v15.16b

    .inst 0x4e94a430 // smmla v16.4s, v1.16b, v20.16b
    .inst 0x4e95a431 // smmla v17.4s, v1.16b, v21.16b
    .inst 0x4e96a432 // smmla v18.4s, v1.16b, v22.16b
    .inst 0x4e97a433 // smmla v19.4s, v1.16b, v23.16b
    cmp x13, #4
    .inst 0x4e98a450 // smmla v16.4s, v2.16b, v24.16b
    .inst 0x4e99a451 // smmla v17.4s, v2.16b, v25.16b
    .inst 0x4e9aa452 // smmla v18.4s, v2.16b, v26.16b
    .inst 0x4e9ba453 // smmla v19.4s, v2.16b, v27.16b

    .inst 0x4e85a470 // smmla v16.4s, v3.16b, v5.16b
    .inst 0x4e86a471 // smmla v17.4s, v3.16b, v6.16b
    .inst 0x4e87a472 // smmla v18.4s, v3.16b, v7.16b
    .inst 0x4e88a473 // smmla v19.4s, v3.16b, v8.16b

    bge LoopSz1_TILE_1_lu4
    cbz x13, LoopSzEnd_TILE_1

LoopSz_TILE_1_lu1:
    ld1 {v2.8b}, [x11], x22           // src
    // int4->int8
    ld1 {v0.16b, v1.16b}, [x12], #32                   // weight
    ushr v8.16b, v0.16b, #4 // oc:0-1
    ushr v9.16b, v1.16b, #4 // oc:2-3
    and v10.16b, v0.16b, v28.16b // oc:4-5
    and v11.16b, v1.16b, v28.16b // oc:6-7
    subs x13, x13, #1

    .inst 0x4e88a450 // smmla v16.4s, v2.16b, v8.16b
    .inst 0x4e89a451 // smmla v17.4s, v2.16b, v9.16b
    .inst 0x4e8aa452 // smmla v18.4s, v2.16b, v10.16b
    .inst 0x4e8ba453 // smmla v19.4s, v2.16b, v11.16b
    bne LoopSz_TILE_1_lu1
LoopSzEnd_TILE_1:
    uzp1 v25.2d, v16.2d, v17.2d
    uzp1 v26.2d, v18.2d, v19.2d
    scvtf v25.4s, v25.4s
    scvtf v26.4s, v26.4s

Tile1Quan:
    ld1 {v0.4s, v1.4s}, [x12], #32  // scale
    ld1 {v6.s}[0], [x8] // x kernel sum
    ld1 {v8.4s, v9.4s}, [x12], #32 // weight quan zeropoint
    fmul v25.4s, v25.4s, v0.4s
    fmul v26.4s, v26.4s, v1.4s
    add x8, x8, x22, LSR #1

    ld1 {v10.s}[0], [x23], x25
    fmul v25.4s, v25.4s, v10.s[0]
    fmul v26.4s, v26.4s, v10.s[0]

    TILE1_MLA:
    MLA_WEIGHTZERO v25, v6, v8, 0 // tile:0, oc:0-3
    MLA_WEIGHTZERO v26, v6, v9, 0 // tile:0, oc:4-7

    cbz x27, TILE1_ADD_DSTV
    ld1 {v6.s}[0], [x27], x25 // input dequant bias
    ld1 {v8.4s, v9.4s}, [x28], #32 // weight kernel sum
    MLA_WEIGHTZERO v25, v6, v8, 0 // tile:0, oc:0-3
    MLA_WEIGHTZERO v26, v6, v9, 0 // tile:0, oc:4-7

    TILE1_ADD_DSTV:
    cbz x19, TILE1_L8_ACCUM_BUFFER
    ld1 {v15.4s, v16.4s}, [x10]
    fadd v25.4s, v25.4s, v15.4s
    fadd v26.4s, v26.4s, v16.4s

    TILE1_L8_ACCUM_BUFFER:
    add x19, x19, #1
    cmp x19, x26
    beq TILE1_POST
    st1 {v25.4s, v26.4s}, [x10]
    b TILE1_BLOCKNUM

    TILE1_POST:
    sub x24, x24, #2
    cbz x9, TILE1_RELU
    ld1 {v16.4s, v17.4s}, [x20], #32 // bias
    fadd v25.4s, v25.4s, v16.4s
    fadd v26.4s, v26.4s, v17.4s
    TILE1_RELU:
    cbz x14, TILE1_STORE
    fmin v25.4s, v25.4s, v31.4s
    fmax v25.4s, v25.4s, v30.4s
    fmin v26.4s, v26.4s, v31.4s
    fmax v26.4s, v26.4s, v30.4s

    TILE1_STORE:
    st1 {v25.4s}, [x6], x4
    st1 {v26.4s}, [x6], x4

Tile1LoopEnd:
    cbz x24, End
    mov x8, x15
    mov x23, x21
    cbz x27, LoopDz_TILE_1
    REVERT_INPUT_DEQUANT_BIAS x27, x19, x26, x25
    b LoopDz_TILE_1

LoopDz4_TILE_1:
    mov x11, x1             // src
    mov x19, #0
L4_TILE1_BLOCKNUM:
    mov x13, x3             // src_depth_quad

    movi v16.4s, #0
    movi v17.4s, #0
cmp x13, #4
blt L4_LoopSz_TILE_1_lu1

L4_LoopSz1_TILE_1_lu4:
    ld1 {v5.16b, v6.16b, v7.16b, v8.16b}, [x12], #64     // weight
    ld1 {v9.16b, v10.16b, v11.16b, v12.16b}, [x12], #64
    ld1 {v0.8b}, [x11], x22                              // src
    ld1 {v1.8b}, [x11], x22
    ld1 {v2.8b}, [x11], x22
    ld1 {v3.8b}, [x11], x22

    // int4->int8
    ushr v4.16b, v5.16b, #4
    ushr v14.16b, v6.16b, #4
    ushr v20.16b, v7.16b, #4
    ushr v21.16b, v8.16b, #4
    ushr v24.16b, v9.16b, #4
    ushr v25.16b, v10.16b, #4
    ushr v5.16b, v11.16b, #4
    ushr v6.16b, v12.16b, #4

    sub x13, x13, #4

    .inst 0x4e84a410 // smmla v16.4s, v0.16b, v4.16b
    .inst 0x4e8ea411 // smmla v17.4s, v0.16b, v14.16b

    .inst 0x4e94a430 // smmla v16.4s, v1.16b, v20.16b
    .inst 0x4e95a431 // smmla v17.4s, v1.16b, v21.16
    cmp x13, #4
    .inst 0x4e98a450 // smmla v16.4s, v2.16b, v24.16b
    .inst 0x4e99a451 // smmla v17.4s, v2.16b, v25.16b

    .inst 0x4e85a470 // smmla v16.4s, v3.16b, v5.16b
    .inst 0x4e86a471 // smmla v17.4s, v3.16b, v6.16b

    bge L4_LoopSz1_TILE_1_lu4
    cbz x13, L4_LoopSzEnd_TILE_1

L4_LoopSz_TILE_1_lu1:
    ld1 {v2.8b}, [x11], x22           // src
    // int4->int8
    ld1 {v0.16b, v1.16b}, [x12], #32                   // weight
    ushr v8.16b, v0.16b, #4 // oc:0-1
    ushr v9.16b, v1.16b, #4 // oc:2-3
    subs x13, x13, #1

    .inst 0x4e88a450 // smmla v16.4s, v2.16b, v8.16b
    .inst 0x4e89a451 // smmla v17.4s, v2.16b, v9.16b
    bne L4_LoopSz_TILE_1_lu1

L4_LoopSzEnd_TILE_1:
    uzp1 v25.2d, v16.2d, v17.2d
    scvtf v25.4s, v25.4s

L4_Tile1Quan:
    ld1 {v0.4s}, [x12]  // scale
    add x12, x12, #32   // hp*sizeof(float)
    ld1 {v6.s}[0], [x8] // x kernel sum
    ld1 {v8.4s}, [x12] // weight quan zeropoint
    fmul v25.4s, v25.4s, v0.4s
    add x8, x8, x22, LSR #1
    add x12, x12, #32

    ld1 {v10.s}[0], [x23], x25
    fmul v25.4s, v25.4s, v10.s[0]

    L4_TILE1_MLA:
    MLA_WEIGHTZERO v25, v6, v8, 0 // tile:0, oc:0-3

    cbz x27, L4_TILE1_ADD_DSTV
    ld1 {v6.s}[0], [x27], x25 // input dequant bias
    ld1 {v8.4s}, [x28] // weight kernel sum
    MLA_WEIGHTZERO v25, v6, v8, 0 // tile:0, oc:0-3
    add x28, x28, #32

    L4_TILE1_ADD_DSTV:
    cbz x19, L4_TILE1_ACCUM_BUFFER
    ld1 {v15.4s}, [x10]
    fadd v25.4s, v25.4s, v15.4s

    L4_TILE1_ACCUM_BUFFER:
    add x19, x19, #1
    cmp x19, x26
    beq L4_TILE1_POST
    st1 {v25.4s}, [x10]
    b L4_TILE1_BLOCKNUM

    L4_TILE1_POST:
    cbz x9, L4_TILE1_RELU
    ld1 {v16.4s}, [x20], #16 // bias
    fadd v25.4s, v25.4s, v16.4s
    L4_TILE1_RELU:
    cbz x14, L4_TILE1_STORE
    fmin v25.4s, v25.4s, v31.4s
    fmax v25.4s, v25.4s, v30.4s

    L4_TILE1_STORE:
    st1 {v25.4s}, [x6], x4

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
