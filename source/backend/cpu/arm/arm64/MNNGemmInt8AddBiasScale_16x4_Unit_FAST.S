//
//  MNNGemmInt8AddBiasScale_16x4_Unit_FAST.S
//  MNN
//
//  Created by MNN on 2020/03/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

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

asm_function MNNGemmInt8AddBiasScale_16x4_Unit_FAST

/* 
struct QuanPostTreatParameters {
    const float* scale;
    const float* bias;
    int32_t maxValue;
    int32_t minValue;
    int32_t useInt8 = 1; // Save result as int8_t dataType; otherwise float32.
    float roundValuePos = 0.5f;
    float roundValueNeg = -0.5f;
    float* srcKernelSum;
    float* weightQuanBias;
    float* fp32minmax;
};
*/

//void MNNGemmInt8AddBiasScale_16x4_Unit_FAST(int8_t* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad, size_t dst_step,
//                                              size_t dst_depth_quad, const QuanPostTreatParameters* post, size_t remain) {

//Auto: x0: dst*, x1: src*, x2:weight*, x3: src_depth_quad, x4: dst_step, 
// x5: dst_depth_quad, x6: post, x7: remain

//Load from post:
// x7: scale, x10: bias, w11: maxValue, w13: minValue, w12: useInt8
// x19: srcKernelSum, x20: weightQuanBias
mov x8, x7
ldr x7, [x6, #0]
ldr x10, [x6, #8]
ldr w11, [x6, #16]
ldr w13, [x6, #20]
ldr w12, [x6, #24]

stp d14, d15, [sp, #(-16 * 6)]!
stp d12, d13, [sp, #(16 * 1)]
stp d10, d11, [sp, #(16 * 2)]
stp d8,  d9,  [sp, #(16 * 3)]
stp x21, x22, [sp, #(16 * 4)]
stp x19, x20, [sp, #(16 * 5)]
ldr x19, [x6, #40]
ldr x20, [x6, #48]

cmp x8, #3
beq L3Dz

cmp x8, #2
beq L2Dz

cmp x8, #1
beq L1Dz

L4Dz:
cmp w12, #1
bne L4LoopDz
sub x4, x4, #8

L4LoopDz:
    mov x8, x1
    // load four weights
    ld1 {v0.16b}, [x2], #16
    ld1 {v1.16b}, [x2], #16
    ld1 {v2.16b}, [x2], #16
    ld1 {v3.16b}, [x2], #16
    // load one tile input
    ld1 {v4.16b}, [x1], #16
    smull v16.8h, v0.8b, v4.8b
    smull v17.8h, v1.8b, v4.8b
    ld1 {v5.16b}, [x1], #16
    smull v18.8h, v2.8b, v4.8b
    mov x9, x3
    smull v19.8h, v3.8b, v4.8b
    smull v20.8h, v0.8b, v5.8b
    smull v21.8h, v1.8b, v5.8b
    ld1 {v6.16b}, [x1], #16
    smull v22.8h, v2.8b, v5.8b
    smull v23.8h, v3.8b, v5.8b
    smull v24.8h, v0.8b, v6.8b
    smull v25.8h, v1.8b, v6.8b
    ld1 {v7.16b}, [x1], #16
    smull v26.8h, v2.8b, v6.8b
    smull v27.8h, v3.8b, v6.8b
    smull v28.8h, v0.8b, v7.8b
    smull v29.8h, v1.8b, v7.8b
    subs x9, x9, #1
    smull v30.8h, v2.8b, v7.8b
    smull v31.8h, v3.8b, v7.8b
    
    beq L4LoopSzEnd

    L4LoopSz:
        smlal2 v16.8h, v0.16b, v4.16b
        smlal2 v17.8h, v1.16b, v4.16b
        smlal2 v18.8h, v2.16b, v4.16b
        smlal2 v19.8h, v3.16b, v4.16b
        smlal2 v20.8h, v0.16b, v5.16b
        ld1 {v4.16b}, [x1], #16
        smlal2 v21.8h, v1.16b, v5.16b
        smlal2 v22.8h, v2.16b, v5.16b
        smlal2 v23.8h, v3.16b, v5.16b
        smlal2 v24.8h, v0.16b, v6.16b
        ld1 {v5.16b}, [x1], #16
        smlal2 v25.8h, v1.16b, v6.16b
        smlal2 v26.8h, v2.16b, v6.16b
        smlal2 v27.8h, v3.16b, v6.16b
        smlal2 v28.8h, v0.16b, v7.16b
        ld1 {v6.16b}, [x1], #16
        smlal2 v29.8h, v1.16b, v7.16b
        ld1 {v0.16b}, [x2], #16
        smlal2 v30.8h, v2.16b, v7.16b
        ld1 {v1.16b}, [x2], #16
        smlal2 v31.8h, v3.16b, v7.16b
        ld1 {v2.16b}, [x2], #16

        smlal v16.8h, v0.8b, v4.8b
        ld1 {v7.16b}, [x1], #16
        smlal v17.8h, v1.8b, v4.8b
        ld1 {v3.16b}, [x2], #16
        smlal v18.8h, v2.8b, v4.8b
        smlal v19.8h, v3.8b, v4.8b
        smlal v20.8h, v0.8b, v5.8b
        smlal v21.8h, v1.8b, v5.8b
        smlal v22.8h, v2.8b, v5.8b
        smlal v23.8h, v3.8b, v5.8b
        smlal v24.8h, v0.8b, v6.8b
        smlal v25.8h, v1.8b, v6.8b
        smlal v26.8h, v2.8b, v6.8b
        smlal v27.8h, v3.8b, v6.8b
        smlal v28.8h, v0.8b, v7.8b
        smlal v29.8h, v1.8b, v7.8b
        smlal v30.8h, v2.8b, v7.8b
        subs x9, x9, #1
        smlal v31.8h, v3.8b, v7.8b
        bne L4LoopSz
    L4LoopSzEnd:

    smlal2 v16.8h, v0.16b, v4.16b
    smlal2 v17.8h, v1.16b, v4.16b
    smlal2 v18.8h, v2.16b, v4.16b
    smlal2 v19.8h, v3.16b, v4.16b
    smlal2 v20.8h, v0.16b, v5.16b
    smlal2 v21.8h, v1.16b, v5.16b
    smlal2 v22.8h, v2.16b, v5.16b
    smlal2 v23.8h, v3.16b, v5.16b
    smlal2 v24.8h, v0.16b, v6.16b
    smlal2 v25.8h, v1.16b, v6.16b
    smlal2 v26.8h, v2.16b, v6.16b
    smlal2 v27.8h, v3.16b, v6.16b
    smlal2 v28.8h, v0.16b, v7.16b
    smlal2 v29.8h, v1.16b, v7.16b
    smlal2 v30.8h, v2.16b, v7.16b
    smlal2 v31.8h, v3.16b, v7.16b

    saddlp v15.4s, v16.8h
    saddlp v14.4s, v17.8h
    saddlp v13.4s, v18.8h
    saddlp v12.4s, v19.8h
    saddlp v11.4s, v20.8h
    saddlp v10.4s, v21.8h
    saddlp v9.4s,  v22.8h
    saddlp v8.4s,  v23.8h
    saddlp v7.4s,  v24.8h
    saddlp v6.4s,  v25.8h
    saddlp v5.4s,  v26.8h
    saddlp v4.4s,  v27.8h
    saddlp v3.4s,  v28.8h
    saddlp v2.4s,  v29.8h
    saddlp v1.4s,  v30.8h
    saddlp v0.4s,  v31.8h
    
    addp v16.4s, v15.4s, v14.4s
    addp v17.4s, v13.4s, v12.4s
    addp v18.4s, v11.4s, v10.4s
    addp v19.4s, v9.4s, v8.4s
    addp v20.4s, v7.4s, v6.4s
    addp v21.4s, v5.4s, v4.4s
    addp v22.4s, v3.4s, v2.4s
    addp v23.4s, v1.4s, v0.4s
    addp v12.4s, v16.4s, v17.4s
    addp v13.4s, v18.4s, v19.4s
    ld1 {v0.4s}, [x10], #16
    addp v14.4s, v20.4s, v21.4s
    addp v15.4s, v22.4s, v23.4s

    //add v16.4s, v12.4s, v0.4s
    //add v17.4s, v13.4s, v0.4s
    //add v18.4s, v14.4s, v0.4s
    //add v19.4s, v15.4s, v0.4s

    L4Quan:
    ld1 {v1.4s}, [x7], #16 // scale
    ld1 {v2.4s}, [x19] // x kernel sum
    ld1 {v24.4s}, [x20], #16 // weight quan zeropoint

    TILE4_INT2FLOAT:
    scvtf v4.4s, v12.4s
    scvtf v5.4s, v13.4s
    scvtf v6.4s, v14.4s
    scvtf v7.4s, v15.4s

    fmul v12.4s, v4.4s, v1.4s
    fmul v13.4s, v5.4s, v1.4s
    fmul v14.4s, v6.4s, v1.4s
    fmul v15.4s, v7.4s, v1.4s

    MLA_WEIGHTZERO v12, v2, v24, 0 // tile:0, oc:0-3
    MLA_WEIGHTZERO v13, v2, v24, 1 // tile:1, oc:0-3
    MLA_WEIGHTZERO v14, v2, v24, 2 // tile:2, oc:0-3
    MLA_WEIGHTZERO v15, v2, v24, 3 // tile:3, oc:0-3


    fadd v12.4s, v12.4s, v0.4s
    fadd v13.4s, v13.4s, v0.4s
    fadd v14.4s, v14.4s, v0.4s
    fadd v15.4s, v15.4s, v0.4s

    cmp w12, #1
    beq L4QuantUseInt8
    ReLU_FP32 v12, v13, v14, v15, v26, v27
    st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [x0], x4
    b L4LoopCheck

    L4QuantUseInt8:
    dup v31.4s, w13 // Min
    dup v30.4s, w11 // Max
    fcvtas v8.4s, v12.4s
    fcvtas v9.4s, v13.4s
    fcvtas v10.4s, v14.4s
    fcvtas v11.4s, v15.4s

    smin v8.4s, v30.4s, v8.4s
    smin v9.4s, v30.4s, v9.4s
    smin v10.4s, v30.4s, v10.4s
    smin v11.4s, v30.4s, v11.4s

    smax v8.4s, v31.4s, v8.4s
    smax v9.4s, v31.4s, v9.4s
    smax v10.4s, v31.4s, v10.4s
    smax v11.4s, v31.4s, v11.4s


    sqxtn v0.4h, v8.4s
    sqxtn2 v0.8h, v9.4s
    sqxtn v1.4h, v10.4s
    sqxtn2 v1.8h, v11.4s

    sqxtn v2.8b, v0.8h
    sqxtn v3.8b, v1.8h
    st1 {v2.8b}, [x0], #8
    st1 {v3.8b}, [x0], x4
L4LoopCheck:
    subs x5, x5, #1
    mov x1, x8
    bne L4LoopDz

b End

L3Dz:
add x3, x19, #8
cmp w12, #1
bne L3LoopDz
sub x4, x4, #8

L3LoopDz:
    mov x8, x1
    // load four weights
    ld1 {v0.16b}, [x2], #16
    ld1 {v1.16b}, [x2], #16
    ld1 {v2.16b}, [x2], #16
    ld1 {v3.16b}, [x2], #16
    // load one tile input
    ld1 {v4.16b}, [x1], #16
    smull v16.8h, v0.8b, v4.8b
    smull v17.8h, v1.8b, v4.8b
    ld1 {v5.16b}, [x1], #16
    smull v18.8h, v2.8b, v4.8b
    mov x9, x3
    smull v19.8h, v3.8b, v4.8b
    smull v20.8h, v0.8b, v5.8b
    smull v21.8h, v1.8b, v5.8b
    ld1 {v6.16b}, [x1], #16
    smull v22.8h, v2.8b, v5.8b
    smull v23.8h, v3.8b, v5.8b
    smull v24.8h, v0.8b, v6.8b
    smull v25.8h, v1.8b, v6.8b
    add x1, x1, #16
    smull v26.8h, v2.8b, v6.8b
    smull v27.8h, v3.8b, v6.8b
    subs x9, x9, #1
    
    beq L3LoopSzEnd

    L3LoopSz:
        smlal2 v16.8h, v0.16b, v4.16b
        smlal2 v17.8h, v1.16b, v4.16b
        smlal2 v18.8h, v2.16b, v4.16b
        smlal2 v19.8h, v3.16b, v4.16b
        smlal2 v20.8h, v0.16b, v5.16b
        ld1 {v4.16b}, [x1], #16
        smlal2 v21.8h, v1.16b, v5.16b
        smlal2 v22.8h, v2.16b, v5.16b
        smlal2 v23.8h, v3.16b, v5.16b
        smlal2 v24.8h, v0.16b, v6.16b
        ld1 {v5.16b}, [x1], #16
        smlal2 v25.8h, v1.16b, v6.16b
        smlal2 v26.8h, v2.16b, v6.16b
        smlal2 v27.8h, v3.16b, v6.16b
        ld1 {v6.16b}, [x1], #16
        ld1 {v0.16b}, [x2], #16
        ld1 {v1.16b}, [x2], #16
        ld1 {v2.16b}, [x2], #16

        smlal v16.8h, v0.8b, v4.8b
        add x1, x1, #16
        smlal v17.8h, v1.8b, v4.8b
        ld1 {v3.16b}, [x2], #16
        smlal v18.8h, v2.8b, v4.8b
        smlal v19.8h, v3.8b, v4.8b
        smlal v20.8h, v0.8b, v5.8b
        smlal v21.8h, v1.8b, v5.8b
        smlal v22.8h, v2.8b, v5.8b
        smlal v23.8h, v3.8b, v5.8b
        smlal v24.8h, v0.8b, v6.8b
        smlal v25.8h, v1.8b, v6.8b
        smlal v26.8h, v2.8b, v6.8b
        smlal v27.8h, v3.8b, v6.8b
        subs x9, x9, #1
        bne L3LoopSz
    L3LoopSzEnd:

    smlal2 v16.8h, v0.16b, v4.16b
    smlal2 v17.8h, v1.16b, v4.16b
    smlal2 v18.8h, v2.16b, v4.16b
    smlal2 v19.8h, v3.16b, v4.16b
    smlal2 v20.8h, v0.16b, v5.16b
    smlal2 v21.8h, v1.16b, v5.16b
    smlal2 v22.8h, v2.16b, v5.16b
    smlal2 v23.8h, v3.16b, v5.16b
    smlal2 v24.8h, v0.16b, v6.16b
    smlal2 v25.8h, v1.16b, v6.16b
    smlal2 v26.8h, v2.16b, v6.16b
    smlal2 v27.8h, v3.16b, v6.16b

    saddlp v15.4s, v16.8h
    saddlp v14.4s, v17.8h
    saddlp v13.4s, v18.8h
    saddlp v12.4s, v19.8h
    saddlp v11.4s, v20.8h
    saddlp v10.4s, v21.8h
    saddlp v9.4s,  v22.8h
    saddlp v8.4s,  v23.8h
    saddlp v7.4s,  v24.8h
    saddlp v6.4s,  v25.8h
    saddlp v5.4s,  v26.8h
    saddlp v4.4s,  v27.8h
    
    addp v16.4s, v15.4s, v14.4s
    addp v17.4s, v13.4s, v12.4s
    addp v18.4s, v11.4s, v10.4s
    addp v19.4s, v9.4s, v8.4s
    addp v20.4s, v7.4s, v6.4s
    addp v21.4s, v5.4s, v4.4s

    addp v12.4s, v16.4s, v17.4s
    addp v13.4s, v18.4s, v19.4s
    addp v14.4s, v20.4s, v21.4s
    ld1 {v0.4s}, [x10], #16

    L3Quan:
    ld1 {v1.4s}, [x7], #16
    ld1 {v2.d}[0], [x19] // x kernel sum
    ld1 {v2.s}[2], [x6]
    ld1 {v24.4s}, [x20], #16 // weight quan zeropoint

    TILE3_INT2FLOAT:
    scvtf v4.4s, v12.4s
    scvtf v5.4s, v13.4s
    scvtf v6.4s, v14.4s

    fmul v12.4s, v4.4s, v1.4s
    fmul v13.4s, v5.4s, v1.4s
    fmul v14.4s, v6.4s, v1.4s
    MLA_WEIGHTZERO v12, v2, v24, 0 // tile:0, oc:0-3
    MLA_WEIGHTZERO v13, v2, v24, 1 // tile:1, oc:0-3
    MLA_WEIGHTZERO v14, v2, v24, 2 // tile:2, oc:0-3


    fadd v12.4s, v12.4s, v0.4s
    fadd v13.4s, v13.4s, v0.4s
    fadd v14.4s, v14.4s, v0.4s
    cmp w12, #1
    beq L3QuantUseInt8
    ReLU_FP32_3 v12, v13, v14, v26, v27
    st1 {v12.4s, v13.4s, v14.4s}, [x0], x4
    b L3LoopCheck

    L3QuantUseInt8:
    dup v31.4s, w13 // Min
    dup v30.4s, w11 // Max
    fcvtas v8.4s, v12.4s
    fcvtas v9.4s, v13.4s
    fcvtas v10.4s, v14.4s

    smin v8.4s, v30.4s, v8.4s
    smin v9.4s, v30.4s, v9.4s
    smin v10.4s, v30.4s, v10.4s

    smax v8.4s, v31.4s, v8.4s
    smax v9.4s, v31.4s, v9.4s
    smax v10.4s, v31.4s, v10.4s

    sqxtn v0.4h, v8.4s
    sqxtn2 v0.8h, v9.4s
    sqxtn v1.4h, v10.4s

    sqxtn v2.8b, v0.8h
    sqxtn v3.8b, v1.8h
    st1 {v2.8b}, [x0], #8
    st1 {v3.s}[0], [x0], x4
L3LoopCheck:
    subs x5, x5, #1
    mov x1, x8
    bne L3LoopDz

b End

L2Dz:
L2LoopDz:
    mov x8, x1
    // load four weights
    ld1 {v0.16b}, [x2], #16
    ld1 {v1.16b}, [x2], #16
    ld1 {v2.16b}, [x2], #16
    ld1 {v3.16b}, [x2], #16
    // load one tile input
    ld1 {v4.16b}, [x1], #16
    smull v16.8h, v0.8b, v4.8b
    smull v17.8h, v1.8b, v4.8b
    ld1 {v5.16b}, [x1], #16
    smull v18.8h, v2.8b, v4.8b
    mov x9, x3
    smull v19.8h, v3.8b, v4.8b
    smull v20.8h, v0.8b, v5.8b
    smull v21.8h, v1.8b, v5.8b
    smull v22.8h, v2.8b, v5.8b
    smull v23.8h, v3.8b, v5.8b
    add x1, x1, #32
    subs x9, x9, #1
    
    beq L2LoopSzEnd

    L2LoopSz:
        smlal2 v16.8h, v0.16b, v4.16b
        smlal2 v17.8h, v1.16b, v4.16b
        smlal2 v18.8h, v2.16b, v4.16b
        smlal2 v19.8h, v3.16b, v4.16b
        smlal2 v20.8h, v0.16b, v5.16b
        ld1 {v4.16b}, [x1], #16
        smlal2 v21.8h, v1.16b, v5.16b
        smlal2 v22.8h, v2.16b, v5.16b
        smlal2 v23.8h, v3.16b, v5.16b
        ld1 {v5.16b}, [x1], #16
        ld1 {v0.16b}, [x2], #16
        ld1 {v1.16b}, [x2], #16
        ld1 {v2.16b}, [x2], #16

        smlal v16.8h, v0.8b, v4.8b
        add x1, x1, #32
        smlal v17.8h, v1.8b, v4.8b
        ld1 {v3.16b}, [x2], #16
        smlal v18.8h, v2.8b, v4.8b
        smlal v19.8h, v3.8b, v4.8b
        smlal v20.8h, v0.8b, v5.8b
        smlal v21.8h, v1.8b, v5.8b
        smlal v22.8h, v2.8b, v5.8b
        smlal v23.8h, v3.8b, v5.8b
        subs x9, x9, #1
        bne L2LoopSz
    L2LoopSzEnd:

    smlal2 v16.8h, v0.16b, v4.16b
    smlal2 v17.8h, v1.16b, v4.16b
    smlal2 v18.8h, v2.16b, v4.16b
    smlal2 v19.8h, v3.16b, v4.16b
    smlal2 v20.8h, v0.16b, v5.16b
    smlal2 v21.8h, v1.16b, v5.16b
    smlal2 v22.8h, v2.16b, v5.16b
    smlal2 v23.8h, v3.16b, v5.16b

    saddlp v15.4s, v16.8h
    saddlp v14.4s, v17.8h
    saddlp v13.4s, v18.8h
    saddlp v12.4s, v19.8h
    saddlp v11.4s, v20.8h
    saddlp v10.4s, v21.8h
    saddlp v9.4s,  v22.8h
    saddlp v8.4s,  v23.8h
    
    addp v16.4s, v15.4s, v14.4s
    addp v17.4s, v13.4s, v12.4s
    addp v18.4s, v11.4s, v10.4s
    addp v19.4s, v9.4s, v8.4s
    addp v12.4s, v16.4s, v17.4s
    addp v13.4s, v18.4s, v19.4s

    L2Quan:
    ld1 {v1.4s}, [x7], #16
    ld1 {v2.d}[0], [x19] // x kernel sum
    ld1 {v24.4s}, [x20], #16 // weight quan zeropoint
    ld1 {v0.4s}, [x10], #16

    TILE2_INT2FLOAT:
    scvtf v4.4s, v12.4s
    scvtf v5.4s, v13.4s

    fmul v12.4s, v4.4s, v1.4s
    fmul v13.4s, v5.4s, v1.4s
    MLA_WEIGHTZERO v12, v2, v24, 0 // tile:0, oc:0-3
    MLA_WEIGHTZERO v13, v2, v24, 1 // tile:1, oc:0-3
    fadd v12.4s, v12.4s, v0.4s
    fadd v13.4s, v13.4s, v0.4s

    cmp w12, #1
    beq L2QuantUseInt8
    ReLU_FP32_2 v12, v13, v26, v27
    st1 {v12.4s, v13.4s}, [x0], x4
    b L2LoopCheck

    L2QuantUseInt8:
    dup v31.4s, w13 // Min
    dup v30.4s, w11 // Max
    fcvtas v8.4s, v12.4s
    fcvtas v9.4s, v13.4s

    smin v8.4s, v30.4s, v8.4s
    smin v9.4s, v30.4s, v9.4s

    smax v8.4s, v31.4s, v8.4s
    smax v9.4s, v31.4s, v9.4s

    sqxtn v0.4h, v8.4s
    sqxtn2 v0.8h, v9.4s

    sqxtn v2.8b, v0.8h
    st1 {v2.8b}, [x0], x4
L2LoopCheck:
    subs x5, x5, #1
    mov x1, x8
    bne L2LoopDz

b End

L1Dz:

L1LoopDz:
    mov x8, x1
    // load four weights
    ld1 {v0.16b}, [x2], #16
    ld1 {v1.16b}, [x2], #16
    ld1 {v2.16b}, [x2], #16
    ld1 {v3.16b}, [x2], #16
    // load one tile input
    ld1 {v4.16b}, [x1], #16
    smull v16.8h, v0.8b, v4.8b
    smull v17.8h, v1.8b, v4.8b
    smull v18.8h, v2.8b, v4.8b
    smull v19.8h, v3.8b, v4.8b
    add x1, x1, #48
    subs x9, x3, #1

    beq L1LoopSzEnd

    L1LoopSz:
        smlal2 v16.8h, v0.16b, v4.16b
        smlal2 v17.8h, v1.16b, v4.16b
        smlal2 v18.8h, v2.16b, v4.16b
        smlal2 v19.8h, v3.16b, v4.16b
        ld1 {v4.16b}, [x1], #16
        ld1 {v0.16b}, [x2], #16
        ld1 {v1.16b}, [x2], #16
        ld1 {v2.16b}, [x2], #16

        smlal v16.8h, v0.8b, v4.8b
        add x1, x1, #48
        smlal v17.8h, v1.8b, v4.8b
        ld1 {v3.16b}, [x2], #16
        smlal v18.8h, v2.8b, v4.8b
        smlal v19.8h, v3.8b, v4.8b
        subs x9, x9, #1
        bne L1LoopSz
    L1LoopSzEnd:

    smlal2 v16.8h, v0.16b, v4.16b
    smlal2 v17.8h, v1.16b, v4.16b
    smlal2 v18.8h, v2.16b, v4.16b
    smlal2 v19.8h, v3.16b, v4.16b

    saddlp v15.4s, v16.8h
    saddlp v14.4s, v17.8h
    saddlp v13.4s, v18.8h
    saddlp v12.4s, v19.8h
    
    addp v16.4s, v15.4s, v14.4s
    addp v17.4s, v13.4s, v12.4s

    addp v12.4s, v16.4s, v17.4s
    ld1 {v0.4s}, [x10], #16


    L1Quan:
    ld1 {v1.4s}, [x7], #16
    ld1 {v2.s}[0], [x19] // x kernel sum
    ld1 {v24.4s}, [x20], #16 // weight quan zeropoint

    TILE1_INT2FLOAT:
    scvtf v4.4s, v12.4s
    fmul v12.4s, v4.4s, v1.4s
    MLA_WEIGHTZERO v12, v2, v24, 0 // tile:0, oc:0-3
    fadd v12.4s, v12.4s, v0.4s

    cmp w12, #1
    beq L1QuantUseInt8
    ReLU_FP32_1 v12, v26, v27
    st1 {v12.4s}, [x0], x4
    b L1LoopCheck

    L1QuantUseInt8:
    dup v31.4s, w13 // Min
    dup v30.4s, w11 // Max
    fcvtas v8.4s, v12.4s

    smin v8.4s, v30.4s, v8.4s

    smax v8.4s, v31.4s, v8.4s

    sqxtn v0.4h, v8.4s

    sqxtn v2.8b, v0.8h
    st1 {v2.s}[0], [x0], x4
L1LoopCheck:
    subs x5, x5, #1
    mov x1, x8
    bne L1LoopDz

End:
ldp x19, x20, [sp, #80]
ldp x21, x22, [sp, #64]
ldp d8,  d9,  [sp, #48]
ldp d10, d11, [sp, #32]
ldp d12, d13, [sp, #16]
ldp d14, d15, [sp], #96
ret

#endif
