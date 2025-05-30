//
//  MNNGemmInt8AddBiasScale_16x4_Unit.S
//  MNN
//
//  Created by MNN on 2019/06/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

.macro MLA_WEIGHTZERO d0, s0, s1, idx // idx for xKernelSum
    fmla \d0\().4s, \s1\().4s, \s0\().s[\idx]
.endm
.macro ReLU_FP32_4 s0, s1, s2, s3, z0, z1 // z0:min z1:max
    fmin \s0\().4s, \s0\().4s, \z1\().4s
    fmin \s1\().4s, \s1\().4s, \z1\().4s
    fmin \s2\().4s, \s2\().4s, \z1\().4s
    fmin \s3\().4s, \s3\().4s, \z1\().4s
    fmax \s0\().4s, \s0\().4s, \z0\().4s
    fmax \s1\().4s, \s1\().4s, \z0\().4s
    fmax \s2\().4s, \s2\().4s, \z0\().4s
    fmax \s3\().4s, \s3\().4s, \z0\().4s
.endm
.macro ReLU_FP32_3 s0, s1, s2, z0, z1 // z0:min z1:max
    fmin \s0\().4s, \s0\().4s, \z1\().4s
    fmin \s1\().4s, \s1\().4s, \z1\().4s
    fmin \s2\().4s, \s2\().4s, \z1\().4s
    fmax \s0\().4s, \s0\().4s, \z0\().4s
    fmax \s1\().4s, \s1\().4s, \z0\().4s
    fmax \s2\().4s, \s2\().4s, \z0\().4s
.endm
.macro ReLU_FP32_2 s0, s1, z0, z1 // z0:min z1:max
    fmin \s0\().4s, \s0\().4s, \z1\().4s
    fmin \s1\().4s, \s1\().4s, \z1\().4s
    fmax \s0\().4s, \s0\().4s, \z0\().4s
    fmax \s1\().4s, \s1\().4s, \z0\().4s
.endm
.macro ReLU_FP32_1 s0, z0, z1 // z0:min z1:max
    fmin \s0\().4s, \s0\().4s, \z1\().4s
    fmax \s0\().4s, \s0\().4s, \z0\().4s
.endm
.macro MUL_SCALE4 s, d0, d1, d2, d3
    fmul \d0\().4s, \d0\().4s, \s\().4s
    fmul \d1\().4s, \d1\().4s, \s\().4s
    fmul \d2\().4s, \d2\().4s, \s\().4s
    fmul \d3\().4s, \d3\().4s, \s\().4s
.endm
.macro MUL_SCALE3 s, d0, d1, d2
    fmul \d0\().4s, \d0\().4s, \s\().4s
    fmul \d1\().4s, \d1\().4s, \s\().4s
    fmul \d2\().4s, \d2\().4s, \s\().4s
.endm
.macro MUL_SCALE2 s, d0, d1
    fmul \d0\().4s, \d0\().4s, \s\().4s
    fmul \d1\().4s, \d1\().4s, \s\().4s
.endm
.macro MUL_SCALE1 s, d0
    fmul \d0\().4s, \d0\().4s, \s\().4s
.endm
.macro MUL_EXTRA_SCALE s, d0, d1, d2, d3
    fmul \d0\().4s, \d0\().4s, \s\().s[0]
    fmul \d1\().4s, \d1\().4s, \s\().s[1]
    fmul \d2\().4s, \d2\().4s, \s\().s[2]
    fmul \d3\().4s, \d3\().4s, \s\().s[3]
.endm

asm_function MNNGemmInt8AddBiasScale_16x4_Unit

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
    ssize_t blockNum;
    const int32_t* bias;
    float* extraScale;
};
*/

//void MNNGemmInt8AddBiasScale_16x4_Unit(int8_t* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad, size_t dst_step,
//                                              size_t dst_depth_quad, const QuanPostTreatParameters* post, size_t realSize) {

//Auto: x0: dst*, x1: src*, x2:weight*, x3: src_depth_quad, x4: dst_step, 
// x5: dst_depth_quad, x6: post, x7: realSize

//Load from post:
// x10: bias, w11: maxValue, w6: minValue, x14: srcKernelSum

ldr x10, [x6, #8]
ldr w11, [x6, #16]
ldr x14, [x6, #40] // srcKernelSum

stp d14, d15, [sp, #(-16 * 10)]!
stp d12, d13, [sp, #(16 * 1)]
stp d10, d11, [sp, #(16 * 2)]
stp d8,  d9,  [sp, #(16 * 3)]
stp x19, x20, [sp, #(16 * 4)]
stp x23, x24, [sp, #(16 * 5)]
stp x27, x28, [sp, #(16 * 6)]
stp x25, x26, [sp, #(16 * 7)]
stp x21, x22, [sp, #(16 * 8)]

ldr x28, [x6, #48] // weightKernelSum
ldr x19, [x6, #56] // fp32 min max
ldr x26, [x6, #64] // blocknum
ldr x23, [x6, #80] // input scale
ldr x15, [x6, #96] // accumBuffer
ldr x27, [x6, #88]  // extra bias
ldr w6, [x6, #20]  // minValue
add x20, x19, #4
lsl x24, x7, #4    // eDest * SRC_UNIT

mov x13, x23 // input dequant scale
mov x22, x27 // input dequant bias
mov x21, x14 // input kernel sum

LoopCheck:
cmp x7, #3
beq L3Dz

cmp x7, #2
beq L2Dz

cmp x7, #1
beq L1Dz

mov x12, #0
cbz x27, L4LoopDz
mov x12, #16

L4LoopDz:
    mov x8, x1
    mov x25, #0
    L4_BLOCKNUM:
    
    ld1 {v0.16b, v1.16b, v2.16b, v3.16b}, [x2], #64
    ld1 {v4.16b, v5.16b, v6.16b, v7.16b}, [x1], #64
    
    smull v8.8h, v0.8b, v4.8b
    smull v9.8h, v1.8b, v4.8b
    smull v10.8h, v2.8b, v4.8b
    smull v11.8h, v3.8b, v4.8b
    smull v12.8h, v0.8b, v5.8b
    smull v13.8h, v1.8b, v5.8b
    smull v14.8h, v2.8b, v5.8b
    smull v15.8h, v3.8b, v5.8b

    smlal2 v8.8h, v0.16b, v4.16b
    smlal2 v9.8h, v1.16b, v4.16b
    smlal2 v10.8h, v2.16b, v4.16b
    smlal2 v11.8h, v3.16b, v4.16b
    smlal2 v12.8h, v0.16b, v5.16b
    smlal2 v13.8h, v1.16b, v5.16b
    smlal2 v14.8h, v2.16b, v5.16b
    smlal2 v15.8h, v3.16b, v5.16b

    L4Initialize:
        saddlp v16.4s, v8.8h
        saddlp v17.4s, v9.8h
        saddlp v18.4s, v10.8h
        saddlp v19.4s, v11.8h
        saddlp v20.4s, v12.8h
        saddlp v21.4s, v13.8h
        saddlp v22.4s, v14.8h
        saddlp v23.4s, v15.8h

        smull v8.8h, v0.8b, v6.8b
        smull v9.8h, v1.8b, v6.8b
        smull v10.8h, v2.8b, v6.8b
        smull v11.8h, v3.8b, v6.8b
        smull v12.8h, v0.8b, v7.8b
        smull v13.8h, v1.8b, v7.8b
        smull v14.8h, v2.8b, v7.8b
        smull v15.8h, v3.8b, v7.8b
        subs x9, x3, #1
        smlal2 v8.8h,  v0.16b, v6.16b
        smlal2 v9.8h,  v1.16b, v6.16b
        smlal2 v10.8h, v2.16b, v6.16b
        smlal2 v11.8h, v3.16b, v6.16b
        smlal2 v12.8h, v0.16b, v7.16b
        smlal2 v13.8h, v1.16b, v7.16b
        smlal2 v14.8h, v2.16b, v7.16b
        smlal2 v15.8h, v3.16b, v7.16b

        saddlp v24.4s, v8.8h
        saddlp v25.4s, v9.8h
        saddlp v26.4s, v10.8h
        saddlp v27.4s, v11.8h
        saddlp v28.4s, v12.8h
        saddlp v29.4s, v13.8h
        saddlp v30.4s, v14.8h
        saddlp v31.4s, v15.8h
    L4InitializeEnd:
        beq ComputeSum
    
    L4LoopSz:
        ld1 {v4.16b, v5.16b, v6.16b, v7.16b}, [x1], #64
        ld1 {v0.16b, v1.16b, v2.16b, v3.16b}, [x2], #64

        smull v8.8h, v0.8b, v4.8b
        smull v9.8h, v1.8b, v4.8b
        smull v10.8h, v2.8b, v4.8b
        smull v11.8h, v3.8b, v4.8b
        smull v12.8h, v0.8b, v5.8b
        smull v13.8h, v1.8b, v5.8b
        smull v14.8h, v2.8b, v5.8b
        smull v15.8h, v3.8b, v5.8b

        smlal2 v8.8h, v0.16b, v4.16b
        smlal2 v9.8h, v1.16b, v4.16b
        smlal2 v10.8h, v2.16b, v4.16b
        smlal2 v11.8h, v3.16b, v4.16b
        smlal2 v12.8h, v0.16b, v5.16b
        smlal2 v13.8h, v1.16b, v5.16b
        smlal2 v14.8h, v2.16b, v5.16b
        smlal2 v15.8h, v3.16b, v5.16b

        sadalp v16.4s, v8.8h
        sadalp v17.4s, v9.8h
        sadalp v18.4s, v10.8h
        sadalp v19.4s, v11.8h
        sadalp v20.4s, v12.8h
        sadalp v21.4s, v13.8h
        sadalp v22.4s, v14.8h
        sadalp v23.4s, v15.8h

        smull v8.8h, v0.8b, v6.8b
        smull v9.8h, v1.8b, v6.8b
        smull v10.8h, v2.8b, v6.8b
        smull v11.8h, v3.8b, v6.8b
        smull v12.8h, v0.8b, v7.8b
        smull v13.8h, v1.8b, v7.8b
        smull v14.8h, v2.8b, v7.8b
        smull v15.8h, v3.8b, v7.8b

        subs x9, x9, #1

        smlal2 v8.8h, v0.16b, v6.16b
        smlal2 v9.8h, v1.16b, v6.16b
        smlal2 v10.8h, v2.16b, v6.16b
        smlal2 v11.8h, v3.16b, v6.16b
        smlal2 v12.8h, v0.16b, v7.16b
        smlal2 v13.8h, v1.16b, v7.16b
        smlal2 v14.8h, v2.16b, v7.16b
        smlal2 v15.8h, v3.16b, v7.16b

        sadalp v24.4s, v8.8h
        sadalp v25.4s, v9.8h
        sadalp v26.4s, v10.8h
        sadalp v27.4s, v11.8h
        sadalp v28.4s, v12.8h
        sadalp v29.4s, v13.8h
        sadalp v30.4s, v14.8h
        sadalp v31.4s, v15.8h

        bne L4LoopSz

    ComputeSum:

    addp v4.4s, v16.4s, v17.4s
    addp v5.4s, v18.4s, v19.4s
    addp v6.4s, v20.4s, v21.4s
    addp v7.4s, v22.4s, v23.4s
    addp v8.4s, v24.4s, v25.4s
    addp v9.4s, v26.4s, v27.4s
    addp v10.4s, v28.4s, v29.4s
    addp v11.4s, v30.4s, v31.4s

    addp v12.4s, v4.4s, v5.4s
    addp v13.4s, v6.4s, v7.4s
    addp v14.4s, v8.4s, v9.4s
    addp v15.4s, v10.4s, v11.4s

    L4Quan:
    ld1 {v1.4s}, [x2], #16 // scalefuse
    ld1 {v20.4s}, [x14], #16 // srcKernelSum
    ld1 {v21.4s}, [x2], #16 // weightQuanZero

    scvtf v4.4s, v12.4s
    scvtf v5.4s, v13.4s
    scvtf v6.4s, v14.4s
    scvtf v7.4s, v15.4s
    MUL_SCALE4 v1, v4, v5, v6, v7

    cbz x23, TILE4_MLA
    ld1 {v2.4s}, [x23], x12
    MUL_EXTRA_SCALE v2, v4, v5, v6, v7

    TILE4_MLA:
    MLA_WEIGHTZERO v4, v20, v21, 0
    MLA_WEIGHTZERO v5, v20, v21, 1
    MLA_WEIGHTZERO v6, v20, v21, 2
    MLA_WEIGHTZERO v7, v20, v21, 3

    cbz x28, L4QuantUseInt8

    cbz x27, L4_ADD_DSTV
    ld1 {v20.4s}, [x27], #16 // input dequant bias
    ld1 {v21.4s}, [x28], #16 // weight kernel sum
    MLA_WEIGHTZERO v4, v20, v21, 0
    MLA_WEIGHTZERO v5, v20, v21, 1
    MLA_WEIGHTZERO v6, v20, v21, 2
    MLA_WEIGHTZERO v7, v20, v21, 3

    L4_ADD_DSTV:
    cbz x25, L4_BUFFER
    ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x15]
    fadd v4.4s, v4.4s, v8.4s
    fadd v5.4s, v5.4s, v9.4s
    fadd v6.4s, v6.4s, v10.4s
    fadd v7.4s, v7.4s, v11.4s

    L4_BUFFER:
    add x25, x25, #1
    cmp x25, x26
    beq L4_POST
    st1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x15]
    b L4_BLOCKNUM

    L4_POST:
    cbz x10, L4_RELU
    ld1 {v0.4s}, [x10], #16
    fadd v4.4s, v4.4s, v0.4s
    fadd v5.4s, v5.4s, v0.4s
    fadd v6.4s, v6.4s, v0.4s
    fadd v7.4s, v7.4s, v0.4s
    L4_RELU:
    cbz x19, L4_STORE
    ld1r {v26.4s}, [x19] // f32 min
    ld1r {v27.4s}, [x20] // f32 max
    ReLU_FP32_4 v4, v5, v6, v7, v26, v27

    L4_STORE:
    st1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x0], x4
    b L4LoopCheck

    L4QuantUseInt8:
    ld1 {v0.4s}, [x10], #16
    fadd v4.4s, v4.4s, v0.4s
    fadd v5.4s, v5.4s, v0.4s
    fadd v6.4s, v6.4s, v0.4s
    fadd v7.4s, v7.4s, v0.4s

    dup v31.16b, w6 // Min
    dup v30.16b, w11 // Max
    fcvtas v8.4s, v4.4s
    fcvtas v9.4s, v5.4s
    fcvtas v10.4s, v6.4s
    fcvtas v11.4s, v7.4s

    sqxtn v0.4h, v8.4s
    sqxtn2 v0.8h, v9.4s
    sqxtn v1.4h, v10.4s
    sqxtn2 v1.8h, v11.4s

    sqxtn v2.8b, v0.8h
    sqxtn2 v2.16b, v1.8h

    smin v2.16b, v2.16b, v30.16b
    smax v2.16b, v2.16b, v31.16b
    st1 {v2.16b}, [x0], x4
L4LoopCheck:
    subs x5, x5, #1
    mov x1, x8
    mov x23, x13
    mov x27, x22
    mov x14, x21
    bne L4LoopDz

b End

L3Dz:
mov x12, #-8
sub x4, x4, #8
cbz x28, L3LoopDz
add x4, x4, #8
cbz x27, L3LoopDz
mov x12, #4

L3LoopDz:
    mov x8, x1
    mov x25, #0
    L3_BLOCKNUM:
    ld1 {v0.16b, v1.16b, v2.16b, v3.16b}, [x2], #64
    ld1 {v4.16b, v5.16b, v6.16b}, [x1], x24

    smull v8.8h, v0.8b, v4.8b
    smull v9.8h, v1.8b, v4.8b
    smull v10.8h, v2.8b, v4.8b
    smull v11.8h, v3.8b, v4.8b
    smull v12.8h, v0.8b, v5.8b
    smull v13.8h, v1.8b, v5.8b
    smull v14.8h, v2.8b, v5.8b
    smull v15.8h, v3.8b, v5.8b
    
    smlal2 v8.8h, v0.16b, v4.16b
    smlal2 v9.8h, v1.16b, v4.16b
    smlal2 v10.8h, v2.16b, v4.16b
    smlal2 v11.8h, v3.16b, v4.16b
    smlal2 v12.8h, v0.16b, v5.16b
    smlal2 v13.8h, v1.16b, v5.16b
    smlal2 v14.8h, v2.16b, v5.16b
    smlal2 v15.8h, v3.16b, v5.16b

    L3Initialize:
        saddlp v16.4s, v8.8h
        saddlp v17.4s, v9.8h
        saddlp v18.4s, v10.8h
        saddlp v19.4s, v11.8h
        saddlp v20.4s, v12.8h
        saddlp v21.4s, v13.8h
        saddlp v22.4s, v14.8h
        saddlp v23.4s, v15.8h

        smull v8.8h, v0.8b, v6.8b
        smull v9.8h, v1.8b, v6.8b
        smull v10.8h, v2.8b, v6.8b
        smull v11.8h, v3.8b, v6.8b

        subs x9, x3, #1

        smlal2 v8.8h,  v0.16b, v6.16b
        smlal2 v9.8h,  v1.16b, v6.16b
        smlal2 v10.8h, v2.16b, v6.16b
        smlal2 v11.8h, v3.16b, v6.16b

        saddlp v24.4s, v8.8h
        saddlp v25.4s, v9.8h
        saddlp v26.4s, v10.8h
        saddlp v27.4s, v11.8h
    L3InitializeEnd:
        beq L3ComputeSum

    L3LoopSz:
        ld1 {v4.16b, v5.16b, v6.16b}, [x1], x24
        ld1 {v0.16b, v1.16b, v2.16b, v3.16b}, [x2], #64

        smull v8.8h, v0.8b, v4.8b
        smull v9.8h, v1.8b, v4.8b
        smull v10.8h, v2.8b, v4.8b
        smull v11.8h, v3.8b, v4.8b
        smull v12.8h, v0.8b, v5.8b
        smull v13.8h, v1.8b, v5.8b
        smull v14.8h, v2.8b, v5.8b
        smull v15.8h, v3.8b, v5.8b

        smlal2 v8.8h, v0.16b, v4.16b
        smlal2 v9.8h, v1.16b, v4.16b
        smlal2 v10.8h, v2.16b, v4.16b
        smlal2 v11.8h, v3.16b, v4.16b
        smlal2 v12.8h, v0.16b, v5.16b
        smlal2 v13.8h, v1.16b, v5.16b
        smlal2 v14.8h, v2.16b, v5.16b
        smlal2 v15.8h, v3.16b, v5.16b

        sadalp v16.4s, v8.8h
        sadalp v17.4s, v9.8h
        sadalp v18.4s, v10.8h
        sadalp v19.4s, v11.8h
        sadalp v20.4s, v12.8h
        sadalp v21.4s, v13.8h
        sadalp v22.4s, v14.8h
        sadalp v23.4s, v15.8h

        smull v8.8h, v0.8b, v6.8b
        smull v9.8h, v1.8b, v6.8b
        smull v10.8h, v2.8b, v6.8b
        smull v11.8h, v3.8b, v6.8b

        subs x9, x9, #1

        smlal2 v8.8h,  v0.16b, v6.16b
        smlal2 v9.8h,  v1.16b, v6.16b
        smlal2 v10.8h, v2.16b, v6.16b
        smlal2 v11.8h, v3.16b, v6.16b

        sadalp v24.4s, v8.8h
        sadalp v25.4s, v9.8h
        sadalp v26.4s, v10.8h
        sadalp v27.4s, v11.8h

        bne L3LoopSz

    L3ComputeSum:
    addp v4.4s, v16.4s, v17.4s
    addp v5.4s, v18.4s, v19.4s
    addp v6.4s, v20.4s, v21.4s
    addp v7.4s, v22.4s, v23.4s
    addp v8.4s, v24.4s, v25.4s
    addp v9.4s, v26.4s, v27.4s

    addp v12.4s, v4.4s, v5.4s
    addp v13.4s, v6.4s, v7.4s
    addp v14.4s, v8.4s, v9.4s

    L3Quan:
    ld1 {v1.4s}, [x2], #16
    ld1 {v20.d}[0], [x14], #8 // srcKernelSum
    ld1 {v20.s}[2], [x14], #4
    ld1 {v21.4s}, [x2], #16 // weightQuanZero

    scvtf v4.4s, v12.4s
    scvtf v5.4s, v13.4s
    scvtf v6.4s, v14.4s
    MUL_SCALE3 v1, v4, v5, v6

    cbz x23, TILE3_MLA
    ld1 {v2.d}[0], [x23], #8
    ld1 {v2.s}[2], [x23], x12
    fmul v4.4s, v4.4s, v2.s[0]
    fmul v5.4s, v5.4s, v2.s[1]
    fmul v6.4s, v6.4s, v2.s[2]

    TILE3_MLA:
    MLA_WEIGHTZERO v4, v20, v21, 0
    MLA_WEIGHTZERO v5, v20, v21, 1
    MLA_WEIGHTZERO v6, v20, v21, 2

    cbz x28, L3QuantUseInt8

    cbz x27, L3_ADD_DSTV
    ld1 {v20.2s}, [x27], #8 // input dequant bias
    ld1 {v20.s}[2], [x27], #4
    ld1 {v21.4s}, [x28], #16 // weight kernel sum
    MLA_WEIGHTZERO v4, v20, v21, 0
    MLA_WEIGHTZERO v5, v20, v21, 1
    MLA_WEIGHTZERO v6, v20, v21, 2

    L3_ADD_DSTV:
    cbz x25, L3_BUFFER
    ld1 {v8.4s, v9.4s, v10.4s}, [x15]
    fadd v4.4s, v4.4s, v8.4s
    fadd v5.4s, v5.4s, v9.4s
    fadd v6.4s, v6.4s, v10.4s

    L3_BUFFER:
    add x25, x25, #1
    cmp x25, x26
    beq L3_POST
    st1 {v4.4s, v5.4s, v6.4s}, [x15]
    b L3_BLOCKNUM

    L3_POST:
    cbz x10, L3_RELU
    ld1 {v0.4s}, [x10], #16
    fadd v4.4s, v4.4s, v0.4s
    fadd v5.4s, v5.4s, v0.4s
    fadd v6.4s, v6.4s, v0.4s
    L3_RELU:
    cbz x19, L3_STORE
    ld1r {v26.4s}, [x19] // f32 min
    ld1r {v27.4s}, [x20] // f32 max
    ReLU_FP32_3 v4, v5, v6, v26, v27
    L3_STORE:
    st1 {v4.4s, v5.4s, v6.4s}, [x0], x4
    b L3LoopCheck

    L3QuantUseInt8:
    ld1 {v0.4s}, [x10], #16
    fadd v4.4s, v4.4s, v0.4s
    fadd v5.4s, v5.4s, v0.4s
    fadd v6.4s, v6.4s, v0.4s
    dup v31.16b, w6 // Min
    dup v30.16b, w11 // Max
    fcvtas v8.4s, v4.4s
    fcvtas v9.4s, v5.4s
    fcvtas v10.4s, v6.4s

    sqxtn v0.4h, v8.4s
    sqxtn2 v0.8h, v9.4s
    sqxtn v1.4h, v10.4s

    sqxtn v2.8b, v0.8h
    sqxtn v3.8b, v1.8h

    smin v2.8b, v30.8b, v2.8b
    smin v3.8b, v30.8b, v3.8b

    smax v2.8b, v31.8b, v2.8b
    smax v3.8b, v31.8b, v3.8b

   
    st1 {v2.8b}, [x0], #8
    st1 {v3.s}[0], [x0], x4
L3LoopCheck:
    subs x5, x5, #1
    mov x1, x8
    mov x23, x13
    mov x27, x22
    mov x14, x21
    bne L3LoopDz

b End

L2Dz:
mov x12, #0
cbz x27, L2LoopDz
mov x12, #8

L2LoopDz:
    mov x8, x1
    mov x25, #0
    L2_BLOCKNUM:
    ld1 {v0.16b, v1.16b, v2.16b, v3.16b}, [x2], #64
    ld1 {v4.16b, v5.16b}, [x1], x24
    
    
    smull v8.8h, v0.8b, v4.8b
    smull v9.8h, v1.8b, v4.8b
    smull v10.8h, v2.8b, v4.8b
    smull v11.8h, v3.8b, v4.8b
    smull v12.8h, v0.8b, v5.8b
    smull v13.8h, v1.8b, v5.8b
    smull v14.8h, v2.8b, v5.8b
    smull v15.8h, v3.8b, v5.8b

    smlal2 v8.8h, v0.16b, v4.16b
    smlal2 v9.8h, v1.16b, v4.16b
    smlal2 v10.8h, v2.16b, v4.16b
    smlal2 v11.8h, v3.16b, v4.16b
    smlal2 v12.8h, v0.16b, v5.16b
    smlal2 v13.8h, v1.16b, v5.16b
    smlal2 v14.8h, v2.16b, v5.16b
    smlal2 v15.8h, v3.16b, v5.16b

    L2Initialize:
        saddlp v16.4s, v8.8h
        saddlp v17.4s, v9.8h
        saddlp v18.4s, v10.8h
        saddlp v19.4s, v11.8h
        saddlp v20.4s, v12.8h
        saddlp v21.4s, v13.8h
        saddlp v22.4s, v14.8h
        saddlp v23.4s, v15.8h
        subs x9, x3, #1
    L2InitializeEnd:
        beq L2ComputeSum

    L2LoopSz:
        ld1 {v4.16b, v5.16b}, [x1], x24
        ld1 {v0.16b, v1.16b, v2.16b, v3.16b}, [x2], #64

        smull v8.8h, v0.8b, v4.8b
        smull v9.8h, v1.8b, v4.8b
        smull v10.8h, v2.8b, v4.8b
        smull v11.8h, v3.8b, v4.8b
        smull v12.8h, v0.8b, v5.8b
        smull v13.8h, v1.8b, v5.8b
        smull v14.8h, v2.8b, v5.8b
        smull v15.8h, v3.8b, v5.8b

        smlal2 v8.8h, v0.16b, v4.16b
        smlal2 v9.8h, v1.16b, v4.16b
        smlal2 v10.8h, v2.16b, v4.16b
        smlal2 v11.8h, v3.16b, v4.16b

        subs x9, x9, #1
        smlal2 v12.8h, v0.16b, v5.16b
        smlal2 v13.8h, v1.16b, v5.16b
        smlal2 v14.8h, v2.16b, v5.16b
        smlal2 v15.8h, v3.16b, v5.16b

        sadalp v16.4s, v8.8h
        sadalp v17.4s, v9.8h
        sadalp v18.4s, v10.8h
        sadalp v19.4s, v11.8h
        sadalp v20.4s, v12.8h
        sadalp v21.4s, v13.8h
        sadalp v22.4s, v14.8h
        sadalp v23.4s, v15.8h

        bne L2LoopSz

    L2ComputeSum:

    addp v4.4s, v16.4s, v17.4s
    addp v5.4s, v18.4s, v19.4s
    addp v6.4s, v20.4s, v21.4s
    addp v7.4s, v22.4s, v23.4s

    addp v12.4s, v4.4s, v5.4s
    addp v13.4s, v6.4s, v7.4s

    L2Quan:
    ld1 {v1.4s}, [x2], #16
    ld1 {v20.d}[0], [x14], #8 // srcKernelSum
    ld1 {v21.4s}, [x2], #16 // weightQuanZero

    scvtf v4.4s, v12.4s
    scvtf v5.4s, v13.4s
    MUL_SCALE2 v1, v4, v5

    cbz x23, TILE2_MUL_OHE_SCALE
    ld1 {v2.d}[0], [x23], x12
    fmul v4.4s, v4.4s, v2.s[0]
    fmul v5.4s, v5.4s, v2.s[1]

    TILE2_MUL_OHE_SCALE:
    MLA_WEIGHTZERO v4, v20, v21, 0
    MLA_WEIGHTZERO v5, v20, v21, 1

    cbz x28, L2QuantUseInt8

    cbz x27, L2_ADD_DSTV
    ld1 {v20.2s}, [x27], #8 // input dequant bias
    ld1 {v21.4s}, [x28], #16 // weight kernel sum
    MLA_WEIGHTZERO v4, v20, v21, 0
    MLA_WEIGHTZERO v5, v20, v21, 1

    L2_ADD_DSTV:
    cbz x25, L2_BUFFER
    ld1 {v8.4s, v9.4s}, [x15]
    fadd v4.4s, v4.4s, v8.4s
    fadd v5.4s, v5.4s, v9.4s

    L2_BUFFER:
    add x25, x25, #1
    cmp x25, x26
    beq L2_POST
    st1 {v4.4s, v5.4s}, [x15]
    b L2_BLOCKNUM

    L2_POST:
    cbz x10, L2_RELU
    ld1 {v0.4s}, [x10], #16
    fadd v4.4s, v4.4s, v0.4s
    fadd v5.4s, v5.4s, v0.4s
    L2_RELU:
    cbz x19, L2_STORE
    ld1r {v26.4s}, [x19] // f32 min
    ld1r {v27.4s}, [x20] // f32 max
    ReLU_FP32_2 v4, v5, v26, v27

    L2_STORE:
    st1 {v4.4s, v5.4s}, [x0], x4
    b L2LoopCheck

    L2QuantUseInt8:
    ld1 {v0.4s}, [x10], #16
    fadd v4.4s, v4.4s, v0.4s
    fadd v5.4s, v5.4s, v0.4s
    dup v31.8b, w6 // Min
    dup v30.8b, w11 // Max
    fcvtas v8.4s, v4.4s
    fcvtas v9.4s, v5.4s

    sqxtn v0.4h, v8.4s
    sqxtn2 v0.8h, v9.4s

    sqxtn v2.8b, v0.8h

    smin v8.8b, v30.8b, v8.8b
    smax v9.8b, v31.8b, v9.8b

    
    st1 {v2.8b}, [x0], x4
L2LoopCheck:
    subs x5, x5, #1
    mov x1, x8
    mov x23, x13
    mov x27, x22
    mov x14, x21
    bne L2LoopDz

b End

L1Dz:
mov x12, #0
cbz x27, L1LoopDz
mov x12, #4
L1LoopDz:
    mov x8, x1
    mov x25, #0
    L1_BLOCKNUM:
    ld1 {v0.16b, v1.16b, v2.16b, v3.16b}, [x2], #64
    dup v16.4s, wzr
    dup v17.4s, wzr
    ld1 {v4.16b}, [x1], #16
    
    smull v8.8h, v0.8b, v4.8b
    dup v18.4s, wzr
    smull v9.8h, v1.8b, v4.8b
    dup v19.4s, wzr
    smull v10.8h, v2.8b, v4.8b
    smull v11.8h, v3.8b, v4.8b
    subs x9, x3, #1
    smlal2 v8.8h, v0.16b, v4.16b
    smlal2 v9.8h, v1.16b, v4.16b
    smlal2 v10.8h, v2.16b, v4.16b
    smlal2 v11.8h, v3.16b, v4.16b
    beq L1LoopSzEnd

    L1LoopSz:
        sadalp v16.4s, v8.8h
        ld1 {v4.16b}, [x1], #16
        sadalp v17.4s, v9.8h
        sadalp v18.4s, v10.8h
        sadalp v19.4s, v11.8h
        sadalp v20.4s, v12.8h
        sadalp v21.4s, v13.8h
        sadalp v22.4s, v14.8h
        sadalp v23.4s, v15.8h

        ld1 {v0.16b, v1.16b, v2.16b, v3.16b}, [x2], #64

        smull v8.8h, v0.8b, v4.8b
        smull v9.8h, v1.8b, v4.8b
        smull v10.8h, v2.8b, v4.8b
        smull v11.8h, v3.8b, v4.8b

        smlal2 v8.8h, v0.16b, v4.16b
        smlal2 v9.8h, v1.16b, v4.16b
        smlal2 v10.8h, v2.16b, v4.16b
        smlal2 v11.8h, v3.16b, v4.16b

        subs x9, x9, #1
        bne L1LoopSz

    L1LoopSzEnd:
    sadalp v16.4s, v8.8h
    sadalp v17.4s, v9.8h
    sadalp v18.4s, v10.8h
    sadalp  v19.4s, v11.8h

    addp v4.4s, v16.4s, v17.4s
    addp v5.4s, v18.4s, v19.4s

    addp v12.4s, v4.4s, v5.4s

    L1Quan:
    ld1 {v1.4s}, [x2], #16
    ld1 {v20.s}[0], [x14], #4 // srcKernelSum
    ld1 {v21.4s}, [x2], #16 // weightQuanZero

    scvtf v4.4s, v12.4s
    MUL_SCALE1 v1, v4

    cbz x23, TILE1_MUL_OHE_SCALE
    ld1 {v2.s}[0], [x23], x12
    fmul v4.4s, v4.4s, v2.s[0]

    TILE1_MUL_OHE_SCALE:
    MLA_WEIGHTZERO v4, v20, v21, 0

    cbz x28, L1QuantUseInt8

    cbz x27, L1_ADD_DSTV
    ld1 {v20.s}[0], [x27], #4 // input dequant bias
    ld1 {v21.4s}, [x28], #16 // weight kernel sum
    MLA_WEIGHTZERO v4, v20, v21, 0

    L1_ADD_DSTV:
    cbz x25, L1_BUFFER
    ld1 {v8.4s}, [x15]
    fadd v4.4s, v4.4s, v8.4s

    L1_BUFFER:
    add x25, x25, #1
    cmp x25, x26
    beq L1_POST
    st1 {v4.4s}, [x15]
    b L1_BLOCKNUM

    L1_POST:
    cbz x10, L1_RELU
    ld1 {v0.4s}, [x10], #16
    fadd v4.4s, v4.4s, v0.4s
    L1_RELU:
    cbz x19, L1_STORE
    ld1r {v26.4s}, [x19] // f32 min
    ld1r {v27.4s}, [x20] // f32 max
    ReLU_FP32_1 v4, v26, v27

    L1_STORE:
    st1 {v4.4s}, [x0], x4
    b L1LoopCheck

    L1QuantUseInt8:
    ld1 {v0.4s}, [x10], #16
    fadd v4.4s, v4.4s, v0.4s
    dup v31.4s, w6 // Min
    dup v30.4s, w11 // Max

    fcvtas v8.4s, v4.4s

    smin v8.4s, v30.4s, v8.4s

    smax v8.4s, v31.4s, v8.4s

    sqxtn v0.4h, v8.4s

    sqxtn v2.8b, v0.8h
    st1 {v2.s}[0], [x0], x4
L1LoopCheck:
    subs x5, x5, #1
    mov x1, x8
    mov x23, x13
    mov x27, x22
    mov x14, x21
    bne L1LoopDz

End:
ldp x21, x22, [sp, #(16 * 8)]
ldp x25, x26, [sp, #(16 * 7)]
ldp x27, x28, [sp, #(16 * 6)]
ldp x23, x24, [sp, #(16 * 5)]
ldp x19, x20, [sp, #(16 * 4)]
ldp d8,  d9,  [sp, #(16 * 3)]
ldp d10, d11, [sp, #(16 * 2)]
ldp d12, d13, [sp, #(16 * 1)]
ldp d14, d15, [sp], #(16 * 10)
ret

#endif
