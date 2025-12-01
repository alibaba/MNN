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
asm_function MNNGemmInt8AddBiasScale_ARMV86_Unit
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
//void MNNGemmInt8AddBiasScale_ARMV86_Unit(int8_t* dst, const int8_t* src,
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
ldr x26, [x6, #64]  // blockNum
lsl x22, x7, #3 // eDest * GEMM_INT8_SRC_UNIT
ldr x23, [x6, #80]  // input scale
ldr x27, [x6, #88]  // input bias
ldr x10, [x6, #96]  // accumBuffer

add x14, x6, #16 // int8 max ptr
cbz x28, Start
ldr x14, [x6, #56]  // float32 maxmin ptr

Start:

mov x15, x8      // input kernel sum
mov x21, x23     // input dequant scale
mov x24, #-32       // inputBlockNum=1
cbz x27, TILE_10
mov x24, #8         // inputBlockNum>1

TILE_10:
    cmp x7, #10
    blt TILE_8
    mov x20, x9 // bias
    sub x4, x4, #32        // For int8 output, x4-32
    mov x6, x27      // input bias
    mov x12, #-256
    cbz x28, LoopDz8_TILE_10
    sub x4, x4, #96 // For int8 output, x4-32-96

LoopDz8_TILE_10:
    cmp x5, #2
    mov x11, x1 // src
    mov x19, #0
    blt LoopDz4_TILE_10

L8_TILE10_BLOCKNUM:
    mov x13, x3 // src_depth_quad

    SET_0_5 v12, v16, v20, v24, v28 // oc:0,1,0,1
    SET_0_5 v13, v17, v21, v25, v29 // oc:2,3,2,3
    SET_0_5 v14, v18, v22, v26, v30 // oc:4,5,4,5
    SET_0_5 v15, v19, v23, v27, v31 // oc:6,7,6,7

LoopSz_TILE_10:
    ld1 {v8.16b, v9.16b, v10.16b, v11.16b}, [x2], #64                    // weight
    ld1 {v3.16b, v4.16b, v5.16b, v6.16b}, [x11], #64    // src: E0-E9
    ld1 {v7.16b}, [x11], #16
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

    cbz x21, TILE10_MLA
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
    cbz x19, TILE10_L8_ACCUM_BUFFER
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

    TILE10_L8_ACCUM_BUFFER:
    add x19, x19, #1
    cmp x19, x26
    beq TILE10_POST
    st1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x10], #64
    st1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x10], #64
    st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x10], #64
    st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [x10], #64
    st1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x10], x12
    b L8_TILE10_BLOCKNUM

    TILE10_POST:
    cbz x28, Tile10QuanUseInt8
    sub x5, x5, #2
    cbz x9, L8_TILE10_RELU
    ld1 {v20.4s, v21.4s}, [x20], #32  // bias
    ADD_BIAS_FLOAT v0, v1, v2, v3, v20
    ADD_BIAS_FLOAT v4, v5, v6, v7, v20
    ADD_BIAS_FLOAT v10, v11, v12, v13, v21
    ADD_BIAS_FLOAT v14, v15, v16, v17, v21
    fadd v8.4s, v8.4s, v20.4s
    fadd v9.4s, v9.4s, v20.4s
    fadd v18.4s, v18.4s, v21.4s
    fadd v19.4s, v19.4s, v21.4s

    L8_TILE10_RELU:
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
    b Tile10LoopCheck

    Tile10QuanUseInt8:
    sub x5, x5, #2
    ld1 {v20.4s, v21.4s}, [x9], #32 // bias
    ld1r {v31.4s}, [x14], #4   // int8 max
    ld1r {v30.4s}, [x14]       // int8 min
    ADD_BIAS_FLOAT v0, v1, v2, v3, v20
    ADD_BIAS_FLOAT v4, v5, v6, v7, v20
    ADD_BIAS_FLOAT v10, v11, v12, v13, v21
    ADD_BIAS_FLOAT v14, v15, v16, v17, v21
    fadd v8.4s, v8.4s, v20.4s
    fadd v9.4s, v9.4s, v20.4s
    fadd v18.4s, v18.4s, v21.4s
    fadd v19.4s, v19.4s, v21.4s

    sub x14, x14, #4
    dup v31.16b, v31.b[0]
    dup v30.16b, v30.b[0]

    FloatToInt32 v0, v1, v2, v3
    FloatToInt32 v4, v5, v6, v7
    FloatToInt32 v10, v11, v12, v13
    FloatToInt32 v14, v15, v16, v17
    FloatToInt32 v8, v9, v18, v19

    Int32ToInt16 v0, v1, v2, v3, v20, v21
    Int32ToInt16 v4, v5, v6, v7, v22, v23
    sqxtn v24.4h, v8.4s
    sqxtn2 v24.8h, v9.4s
    Int32ToInt16 v10, v11, v12, v13, v25, v26
    Int32ToInt16 v14, v15, v16, v17, v27, v28
    sqxtn v29.4h, v18.4s
    sqxtn2 v29.8h, v19.4s

    Int16ToInt8 v20, v21, v22, v23, v0, v1
    sqxtn v2.8b, v24.8h
    Int16ToInt8 v25, v26, v27, v28, v3, v4
    sqxtn v5.8b, v29.8h

    smax v0.16b, v30.16b, v0.16b
    smax v1.16b, v30.16b, v1.16b
    smax v2.8b, v30.8b, v2.8b
    smax v3.16b, v30.16b, v3.16b
    smax v4.16b, v30.16b, v4.16b
    smax v5.8b, v30.8b, v5.8b

    smin v0.16b, v31.16b, v0.16b
    smin v1.16b, v31.16b, v1.16b
    smin v2.8b, v31.8b, v2.8b
    smin v3.16b, v31.16b, v3.16b
    smin v4.16b, v31.16b, v4.16b
    smin v5.8b, v31.8b, v5.8b

    st1 {v0.16b, v1.16b}, [x0], #32
    st1 {v2.8b}, [x0], x4
    st1 {v3.16b, v4.16b}, [x0], #32
    st1 {v5.8b}, [x0], x4

Tile10LoopCheck:
    cbz x5, End
    mov x8, x15 // revert input kernel sum
    mov x27, x6 // revert input dequant bias
    mov x23, x21 // revert input dequant scale
    b LoopDz8_TILE_10

LoopDz4_TILE_10:
    mov x11, x1 // src
    mov x19, #0

L4_TILE10_BLOCKNUM:
    mov x13, x3 // src_depth_quad
    SET_0_5 v12, v13, v16, v17, v20
    SET_0_5 v21, v24, v25, v28, v29

L4_LoopSz_TILE_10:
    ld1 {v8.16b, v9.16b}, [x2]                   // weight
    ld1 {v3.16b, v4.16b, v5.16b, v6.16b}, [x11], #64    // src: E0-E9
    ld1 {v7.16b}, [x11], #16
    subs x13, x13, #1
    add x2, x2, #64  // x2+lp*hp
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
    add x2, x2, #32
    MUL_SCALE v20, v0, v1, v2, v3
    MUL_SCALE v20, v4, v5, v6, v7
    fmul v8.4s, v8.4s, v20.4s
    fmul v9.4s, v9.4s, v20.4s

    cbz x21, L4_TILE10_MLA
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
    cbz x28, Tile10QuanUseInt8_L4
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

    Tile10QuanUseInt8_L4:
    ld1 {v20.4s}, [x9] // bias
    ld1r {v31.4s}, [x14], #4   // int8 max
    ld1r {v30.4s}, [x14]       // int8 min
    ADD_BIAS_FLOAT v0, v1, v2, v3, v20
    ADD_BIAS_FLOAT v4, v5, v6, v7, v20
    fadd v8.4s, v8.4s, v20.4s
    fadd v9.4s, v9.4s, v20.4s

    sub x14, x14, #4
    dup v31.16b, v31.b[0]
    dup v30.16b, v30.b[0]

    FloatToInt32 v0, v1, v2, v3
    FloatToInt32 v4, v5, v6, v7
    fcvtas v8.4s, v8.4s
    fcvtas v9.4s, v9.4s

    Int32ToInt16 v0, v1, v2, v3, v16, v17
    Int32ToInt16 v4, v5, v6, v7, v18, v19
    sqxtn v24.4h, v8.4s
    sqxtn2 v24.8h, v9.4s

    Int16ToInt8 v16, v17, v18, v19, v21, v22
    sqxtn v23.8b, v24.8h

    smax v21.16b, v30.16b, v21.16b
    smax v22.16b, v30.16b, v22.16b
    smax v23.8b, v30.8b, v23.8b

    smin v21.16b, v31.16b, v21.16b
    smin v22.16b, v31.16b, v22.16b
    smin v23.8b, v31.8b, v23.8b

    st1 {v21.16b, v22.16b}, [x0], #32
    st1 {v23.8b}, [x0], x4
    b End

TILE_8:
    // post parameters initilize
    mov x25, #0
    cbz x28, INT8_POST_INIT
    cbz x27, RELU_VECTOR
    lsr x25, x22, #1         // input block quant: realDstCount * sizeof(float)
    RELU_VECTOR:
    cbz x14, TILE_Remain
    ld1r {v30.4s}, [x14], #4 // f32 min
    ld1r {v31.4s}, [x14]     // f32 max
    b TILE_Remain

    INT8_POST_INIT:
    ld1r {v31.4s}, [x14], #4   // int8 max
    ld1r {v30.4s}, [x14]       // int8 min
    dup v31.16b, v31.b[0]
    dup v30.16b, v30.b[0]

    TILE_Remain:
    cmp x7, #8
    blt TILE_4
    mov x24, x5 // dst_depth_quad
    mov x6, x0 // dst
    mov x12, x2 // weight
    mov x20, x9 // bias
    cbz x28, LoopDz_TILE_8
    sub x4, x4, #64 // For float32 output, add #64 when tile8 end.

LoopDz_TILE_8:
    cmp x24, #2
    blt LoopDz4_TILE_8
    mov x11, x1 // src
    mov x19, #0

    L8_TILE8_BLOCKNUM:
    mov x13, x3 // src_depth_quad
    SET_0_4 v12, v16, v20, v24
    SET_0_4 v13, v17, v21, v25
    SET_0_4 v14, v18, v22, v26
    SET_0_4 v15, v19, v23, v27
LoopSz_TILE_8:
    ld1 {v8.16b, v9.16b, v10.16b, v11.16b}, [x12], #64                    // weight
    ld1 {v3.16b, v4.16b, v5.16b, v6.16b}, [x11], x22    // src: E0-E7
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
    cbz x21, TILE8_MLA
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
    b L8_TILE8_BLOCKNUM

    TILE8_POST:
    cbz x28, Tile8QuanUseInt8
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
    cbz x24, TILE8_FLOAT_END
    b Tile8LoopCheck

    Tile8QuanUseInt8:
    sub x24, x24, #2 // dz-2
    ld1 {v16.4s, v17.4s}, [x20], #32
    ADD_BIAS_FLOAT v0, v1, v2, v3, v16
    ADD_BIAS_FLOAT v4, v5, v6, v7, v16
    ADD_BIAS_FLOAT v8, v9, v10, v11, v17
    ADD_BIAS_FLOAT v12, v13, v14, v15, v17

    FloatToInt32 v0, v1, v2, v3
    FloatToInt32 v4, v5, v6, v7
    FloatToInt32 v8, v9, v10, v11
    FloatToInt32 v12, v13, v14, v15

    Int32ToInt16 v0, v1, v2, v3, v20, v21
    Int32ToInt16 v4, v5, v6, v7, v22, v23
    Int32ToInt16 v8, v9, v10, v11, v24, v25
    Int32ToInt16 v12, v13, v14, v15, v26, v27

    Int16ToInt8 v20, v21, v22, v23, v28, v29
    Int16ToInt8 v24, v25, v26, v27, v18, v19
    smax v28.16b, v30.16b, v28.16b
    smax v29.16b, v30.16b, v29.16b
    smax v18.16b, v30.16b, v18.16b
    smax v19.16b, v30.16b, v19.16b
    smin v28.16b, v31.16b, v28.16b
    smin v29.16b, v31.16b, v29.16b
    smin v18.16b, v31.16b, v18.16b
    smin v19.16b, v31.16b, v19.16b
    st1 {v28.16b, v29.16b}, [x6], x4
    st1 {v18.16b, v19.16b}, [x6], x4  // dst += dz * dst_step
    cbz x24, TILE8_INT8_END
Tile8LoopCheck:
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
    ld1 {v8.16b, v9.16b}, [x12]                  // weight
    ld1 {v3.16b, v4.16b, v5.16b, v6.16b}, [x11], x22    // src: E0-E7
    subs x13, x13, #1
    add x12, x12, #64
    .inst 0x4e88a46c // smmla v12.4s, v3.16b, v8.16b // tile0-oc0, tile0-oc1, tile1-oc0, tile1-oc1
    .inst 0x4e89a46d // smmla v13.4s, v3.16b, v9.16b // tile0-oc2, tile0-oc3, tile1-oc2, tile1-oc3

    .inst 0x4e88a490 // smmla v16.4s, v4.16b, v8.16b // tile2-oc0, tile2-oc1, tile3-oc0, tile3-oc1
    .inst 0x4e89a491 // smmla v17.4s, v4.16b, v9.16b // tile2-oc2, tile2-oc3, tile3-oc2, tile3-oc3

    .inst 0x4e88a4b4 // smmla v20.4s, v5.16b, v8.16b // tile4-oc0, tile4-oc1, tile5-oc0, tile5-oc1
    .inst 0x4e89a4b5 // smmla v21.4s, v5.16b, v9.16b // tile4-oc2, tile4-oc3, tile5-oc2, tile5-oc3

    .inst 0x4e88a4d8 // smmla v24.4s, v6.16b, v8.16b // tile6-oc0, tile6-oc1, tile7-oc0, tile7-oc1
    .inst 0x4e89a4d9 // smmla v25.4s, v6.16b, v9.16b // tile6-oc2, tile6-oc3, tile7-oc2, tile7-oc3
    bne L4_LoopSz_TILE_8

LoopSz4End_TILE_8:
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

    cbz x21, L4_TILE8_MLA
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
    cbz x28, Tile8QuanUseInt8_L4
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
    b TILE8_FLOAT_END

    Tile8QuanUseInt8_L4:
    ld1 {v16.4s}, [x20]
    ADD_BIAS_FLOAT v0, v1, v2, v3, v16
    ADD_BIAS_FLOAT v4, v5, v6, v7, v16

    FloatToInt32 v0, v1, v2, v3
    FloatToInt32 v4, v5, v6, v7

    Int32ToInt16 v0, v1, v2, v3, v20, v21
    Int32ToInt16 v4, v5, v6, v7, v22, v23

    Int16ToInt8 v20, v21, v22, v23, v16, v17
    smax v16.16b, v30.16b, v16.16b
    smax v17.16b, v30.16b, v17.16b
    smin v16.16b, v31.16b, v16.16b
    smin v17.16b, v31.16b, v17.16b
    st1 {v16.16b, v17.16b}, [x6], x4
    b TILE8_INT8_END

TILE8_FLOAT_END:
    sub x7, x7, #8
    cbz x7, End
    add x0, x0, #128
    add x1, x1, #64
    add x23, x21, #32 // input dequant scale
    add x8, x15, #32  // input kernel sum
    add x4, x4, #64 // Revert x4 for following tile.
    cbz x27, TILE_4
    REVERT_INPUT_DEQUANT_BIAS x27, x19, x26, x25
    REVERT_WEIGHT_KERNEL_SUM x28, x24, x26, x5
    add x27, x27, #32 // update input dequant bias: + 8 * sizeof(float)
    b TILE_4
TILE8_INT8_END:
    sub x7, x7, #8
    cbz x7, End
    add x1, x1, #64 // source
    add x0, x0, #32 // dst:8*pack*sizeof(int8)
    add x8, x15, #32  // input kernel sum
    cbz x21, TILE_4
    add x23, x21, #32 // input dequant scale


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
    ld1 {v8.16b, v9.16b, v10.16b, v11.16b}, [x12], #64                    // weight
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
    cbz x21, TILE4_MLA
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
    cbz x28, Tile4QuanUseInt8
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
    cbz x24, TILE4_FLOAT_END
    b Tile4LoopCheck

    Tile4QuanUseInt8:
    sub x24, x24, #2
    ld1 {v16.4s, v17.4s}, [x20], #32 // bias
    ADD_BIAS_FLOAT v0, v1, v2, v3, v16
    ADD_BIAS_FLOAT v4, v5, v6, v7, v17
    FloatToInt32 v0, v1, v2, v3
    FloatToInt32 v4, v5, v6, v7
    Int32ToInt16 v0, v1, v2, v3, v8, v9
    Int32ToInt16 v4, v5, v6, v7, v10, v11
    Int16ToInt8_ONE v8, v9, v19
    Int16ToInt8_ONE v10, v11, v20
    smax v19.16b, v30.16b, v19.16b
    smin v19.16b, v31.16b, v19.16b
    smax v20.16b, v30.16b, v20.16b
    smin v20.16b, v31.16b, v20.16b
    st1 {v19.16b}, [x6], x4  // dst += dz * dst_step
    st1 {v20.16b}, [x6], x4
    cbz x24, TILE4_INT8_END
Tile4LoopCheck:
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
    ld1 {v8.16b, v9.16b}, [x12]        // weight
    ld1 {v4.16b, v5.16b}, [x11], x22   // src
    subs x13, x13, #1
    add x12, x12, #64
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

    cbz x21, L4_TILE4_MLA
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
    cbz x28, Tile4QuanUseInt8_L4
    cbz x9, L4_TILE4_RELU
    ld1 {v16.4s}, [x20], #16 // bias
    ADD_BIAS_FLOAT v0, v1, v2, v3, v16

    L4_TILE4_RELU:
    cbz x14, L4_TILE4_STORE
    ReLU_FP32 v0, v1, v2, v3, v30, v31

    L4_TILE4_STORE:
    st1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x6], x4
    b TILE4_FLOAT_END

    Tile4QuanUseInt8_L4:
    ld1 {v16.4s}, [x20] // bias
    ADD_BIAS_FLOAT v0, v1, v2, v3, v16
    FloatToInt32 v0, v1, v2, v3
    Int32ToInt16 v0, v1, v2, v3, v8, v9
    Int16ToInt8_ONE v8, v9, v19
    smax v19.16b, v30.16b, v19.16b
    smin v19.16b, v31.16b, v19.16b
    st1 {v19.16b}, [x6], x4  // dst += dz * dst_step
    b TILE4_INT8_END

TILE4_FLOAT_END:
    add x0, x0, #64
    sub x7, x7, #4
    add x1, x1, #32
    add x8, x15, #16
    add x23, x21, #16
    cbz x27, TILE_2
    REVERT_INPUT_DEQUANT_BIAS x27, x19, x26, x25
    REVERT_WEIGHT_KERNEL_SUM x28, x24, x26, x5
    add x27, x27, #16
    b TILE_2
TILE4_INT8_END:
    add x0, x0, #16
    sub x7, x7, #4
    add x1, x1, #32
    add x8, x15, #16
    cbz x21, TILE_2
    add x23, x21, #16

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
    ld1 {v8.16b, v9.16b, v10.16b, v11.16b}, [x12], #64
    ld1 {v4.16b}, [x11], x22           // src
    .inst 0x4e88a48c // smmla v12.4s, v4.16b, v8.16b // tile0-oc0, tile0-oc1, tile1-oc0, tile1-oc1
    .inst 0x4e89a48d // smmla v13.4s, v4.16b, v9.16b // tile0-oc2, tile0-oc3, tile1-oc2, tile1-oc3
    .inst 0x4e8aa48e // smmla v14.4s, v4.16b, v10.16b // tile0-oc4, tile0-oc5, tile1-oc4, tile1-oc5
    .inst 0x4e8ba48f // smmla v15.4s, v4.16b, v11.16b // tile0-oc6, tile0-oc7, tile1-oc6, tile1-oc7
    subs x13, x13, #1
    bne LoopSz_TILE_2
LoopSzEnd_TILE_2:
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
    cbz x21, TILE2_MLA
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
    cbz x28, Tile2QuanUseInt8
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
    cbz x24, TILE2_FLOAT_END
    b Tile2LoopCheck

    Tile2QuanUseInt8:
    sub x24, x24, #2
    ld1 {v16.4s, v17.4s}, [x20], #32 // bias
    fadd v0.4s, v0.4s, v16.4s
    fadd v1.4s, v1.4s, v16.4s
    fadd v2.4s, v2.4s, v17.4s
    fadd v3.4s, v3.4s, v17.4s
    fcvtas v0.4s, v0.4s
    fcvtas v1.4s, v1.4s
    fcvtas v2.4s, v2.4s
    fcvtas v3.4s, v3.4s
    sqxtn v6.4h,  v0.4s
    sqxtn2 v6.8h, v1.4s
    sqxtn v7.4h, v2.4s
    sqxtn2 v7.8h, v3.4s
    sqxtn v19.8b, v6.8h
    sqxtn v20.8b, v7.8h
    smax v19.8b, v30.8b, v19.8b
    smin v19.8b, v31.8b, v19.8b
    smax v20.8b, v30.8b, v20.8b
    smin v20.8b, v31.8b, v20.8b
    st1 {v19.8b}, [x6], x4  // dst += dz * dst_step
    st1 {v20.8b}, [x6], x4
    cbz x24, TILE2_INT8_END

Tile2LoopCheck:
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
    ld1 {v8.16b, v9.16b}, [x12]
    ld1 {v4.16b}, [x11], x22           // src
    .inst 0x4e88a48c // smmla v12.4s, v4.16b, v8.16b // tile0-oc0, tile0-oc1, tile1-oc0, tile1-oc1
    .inst 0x4e89a48d // smmla v13.4s, v4.16b, v9.16b // tile0-oc2, tile0-oc3, tile1-oc2, tile1-oc3
    subs x13, x13, #1
    add x12, x12, #64
    bne L4_LoopSz_TILE_2

    uzp1 v0.2d, v12.2d, v13.2d // E0: oc:0-3
    uzp2 v1.2d, v12.2d, v13.2d // E1: oc:0-3
    scvtf v0.4s, v0.4s
    scvtf v1.4s, v1.4s

L4_Tile2Quan:
    ld1 {v20.4s}, [x12]  // scale
    add x12, x12, #32
    ld1 {v22.d}[0], [x8] // x kernel sum
    ld1 {v25.4s}, [x12] // weight quan zeropoint
    add x12, x12, #32
    fmul v0.4s, v0.4s, v20.4s
    fmul v1.4s, v1.4s, v20.4s
    add x8, x8, x22, LSR #1

    cbz x21, L4_TILE2_MLA
    ld1 {v27.d}[0], [x23], x25
    fmul v0.4s, v0.4s, v27.s[0]
    fmul v1.4s, v1.4s, v27.s[1]

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
    cbz x28, Tile2QuanUseInt8_L4
    cbz x9, L4_TILE2_RELU
    ld1 {v16.4s}, [x20], #16 // bias
    fadd v0.4s, v0.4s, v16.4s
    fadd v1.4s, v1.4s, v16.4s
    L4_TILE2_RELU:
    cbz x14, TILE2_STORE_L4
    ReLU_FP32_2 v0, v1, v30, v31
    TILE2_STORE_L4:
    st1 {v0.4s, v1.4s}, [x6], x4
    b TILE2_FLOAT_END

    Tile2QuanUseInt8_L4:
    ld1 {v16.4s}, [x20] // bias
    fadd v0.4s, v0.4s, v16.4s
    fadd v1.4s, v1.4s, v16.4s
    fcvtas v0.4s, v0.4s
    fcvtas v1.4s, v1.4s
    sqxtn v6.4h,  v0.4s
    sqxtn2 v6.8h, v1.4s
    sqxtn v19.8b, v6.8h
    smax v19.8b, v30.8b, v19.8b
    smin v19.8b, v31.8b, v19.8b
    st1 {v19.8b}, [x6], x4  // dst += dz * dst_step

TILE2_FLOAT_END:
    add x0, x0, #32 // float
    sub x7, x7, #2
    add x1, x1, #16
    add x8, x15, #8
    add x23, x21, #8
    cbz x27, TILE_1
    REVERT_INPUT_DEQUANT_BIAS x27, x19, x26, x25
    REVERT_WEIGHT_KERNEL_SUM x28, x24, x26, x5
    add x27, x27, #8
    b TILE_1
TILE2_INT8_END:
    add x0, x0, #8 // float
    sub x7, x7, #2
    add x1, x1, #16
    add x8, x15, #8
    cbz x21, TILE_1
    add x23, x21, #8

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
LoopSz_TILE_1:
    ld1 {v8.16b, v9.16b, v10.16b, v11.16b}, [x12], #64   // weight
    ld1 {v2.8b}, [x11], x22           // src
    subs x13, x13, #1

    .inst 0x4e88a450 // smmla v16.4s, v2.16b, v8.16b
    .inst 0x4e89a451 // smmla v17.4s, v2.16b, v9.16b
    .inst 0x4e8aa452 // smmla v18.4s, v2.16b, v10.16b
    .inst 0x4e8ba453 // smmla v19.4s, v2.16b, v11.16b
    bne LoopSz_TILE_1
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
    cbz x21, TILE1_MLA
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
    cbz x28, Tile1QuanUseInt8
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
    cbz x24, End
    b Tile1LoopEnd

    Tile1QuanUseInt8:
    sub x24, x24, #2
    ld1 {v16.4s, v17.4s}, [x20], #32 // bias
    fadd v25.4s, v25.4s, v16.4s
    fadd v26.4s, v26.4s, v17.4s
    fcvtas v25.4s, v25.4s
    fcvtas v26.4s, v26.4s
    sqxtn v6.4h, v25.4s
    sqxtn v7.4h, v26.4s
    sqxtn v6.8b, v6.8h
    sqxtn v7.8b, v7.8h
    smax v6.16b, v30.16b, v6.16b
    smin v6.16b, v31.16b, v6.16b
    smax v7.16b, v30.16b, v7.16b
    smin v7.16b, v31.16b, v7.16b
    st1 {v6.s}[0], [x6], x4  // dst += dz * dst_step
    st1 {v7.s}[0], [x6], x4
    cbz x24, End

Tile1LoopEnd:
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
LoopSz4_TILE_1:
    ld1 {v8.16b, v9.16b}, [x12]   // weight
    ld1 {v2.8b}, [x11], x22           // src
    subs x13, x13, #1
    add x12, x12, #64
    .inst 0x4e88a450 // smmla v16.4s, v2.16b, v8.16b
    .inst 0x4e89a451 // smmla v17.4s, v2.16b, v9.16b
    bne LoopSz4_TILE_1
L4_LoopSzEnd_TILE_1:
    uzp1 v25.2d, v16.2d, v17.2d
    scvtf v25.4s, v25.4s

L4_Tile1Quan:
    ld1 {v0.4s}, [x12]  // scale
    add x12, x12, #32
    ld1 {v6.s}[0], [x8] // x kernel sum
    ld1 {v8.4s}, [x12] // weight quan zeropoint
    add x12, x12, #32
    fmul v25.4s, v25.4s, v0.4s
    add x8, x8, x22, LSR #1
    cbz x21, L4_TILE1_MLA
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
    cbz x28, Tile1QuanUseInt8_L4
    cbz x9, L4_TILE1_RELU
    ld1 {v16.4s}, [x20], #16 // bias
    fadd v25.4s, v25.4s, v16.4s
    L4_TILE1_RELU:
    cbz x14, L4_TILE1_STORE
    fmin v25.4s, v25.4s, v31.4s
    fmax v25.4s, v25.4s, v30.4s

    L4_TILE1_STORE:
    st1 {v25.4s}, [x6], x4
    b End

    Tile1QuanUseInt8_L4:
    ld1 {v16.4s}, [x20] // bias
    fadd v27.4s, v27.4s, v16.4s
    fcvtas v27.4s, v27.4s
    sqxtn v6.4h, v27.4s
    sqxtn v6.8b, v6.8h
    smax v6.8b, v30.8b, v6.8b
    smin v6.8b, v31.8b, v6.8b
    st1 {v6.s}[0], [x6], x4  // dst += dz * dst_step

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
