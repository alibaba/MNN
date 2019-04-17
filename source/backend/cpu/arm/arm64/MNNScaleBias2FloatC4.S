//
//  MNNScaleBias2FloatC4.S
//  MNN
//
//  Created by MNN on 2018/07/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNScaleBias2FloatC4
//void MNNScaleBias2FloatC4(float* dst, const int16_t* src, const float* alpha, const float* bias, size_t sizeQuad)

//Auto:
//x0:dst, x1:src, x2:alpha, x3:bias
//x4:sizeQuad

mov x12, #16

ld1 {v23.4s}, [x3]//Bias
ld1 {v22.4s}, [x2]//Alpha

L4:
cmp x4, #4
blt L1

L4Loop:
    ld1 {v0.4h}, [x1], x12
    sxtl v2.4s, v0.4h
    ld1 {v4.4h}, [x1], x12
    sxtl v3.4s, v4.4h
    scvtf v2.4s, v2.4s
    ld1 {v1.4h}, [x1], x12
    scvtf v3.4s, v3.4s
    ld1 {v5.4h}, [x1], x12
    sxtl v16.4s, v1.4h
    fmul v2.4s, v2.4s, v22.4s
    fmul v3.4s, v3.4s, v22.4s
    sxtl v17.4s, v5.4h
    fadd v2.4s, v2.4s, v23.4s
    fadd v3.4s, v3.4s, v23.4s
    scvtf v16.4s, v16.4s
    st1 {v2.4s, v3.4s}, [x0], #32
    scvtf v17.4s, v17.4s
    fmul v16.4s, v16.4s, v22.4s
    fmul v17.4s, v17.4s, v22.4s
    fadd v16.4s, v16.4s, v23.4s
    fadd v17.4s, v17.4s, v23.4s
    st1 {v16.4s, v17.4s}, [x0], #32
    sub x4, x4, #4
    cmp x4, #4
    bge L4Loop


L1:
cmp x4, #0
beq EndBasic

Loop1:
    ld1 {v0.4h}, [x1], x12
    sxtl v0.4s, v0.4h
    mov v1.16b, v23.16b
    scvtf v0.4s, v0.4s
    fmla v1.4s, v0.4s, v22.4s

    st1 {v1.4s}, [x0], #16
    subs x4, x4, #1
    bne Loop1


EndBasic:

ret

asm_function MNNScaleBias2FloatC4Relu
//void MNNScaleBias2FloatC4(float* dst, const int16_t* src, const float* alpha, const float* bias, size_t sizeQuad)

//Auto:
//x0:dst, x1:src, x2:alpha, x3:bias
//x4:sizeQuad

mov x12, #16

ld1 {v23.4s}, [x3]//Bias
movi v21.4s, #0
ld1 {v22.4s}, [x2]//Alpha

L4Relu:
cmp x4, #4
blt L1Relu

L4ReluLoop:
    ld1 {v0.4h}, [x1], x12
    sxtl v2.4s, v0.4h
    ld1 {v4.4h}, [x1], x12
    sxtl v3.4s, v4.4h
    scvtf v2.4s, v2.4s
    ld1 {v1.4h}, [x1], x12
    scvtf v3.4s, v3.4s
    ld1 {v5.4h}, [x1], x12
    sxtl v16.4s, v1.4h
    fmul v2.4s, v2.4s, v22.4s
    fmul v3.4s, v3.4s, v22.4s
    sxtl v17.4s, v5.4h

    fadd v2.4s, v2.4s, v23.4s
    fadd v3.4s, v3.4s, v23.4s
    scvtf v16.4s, v16.4s
    fmax v2.4s, v2.4s, v21.4s
    fmax v3.4s, v3.4s, v21.4s
    scvtf v17.4s, v17.4s
    st1 {v2.4s, v3.4s}, [x0], #32
    fmul v16.4s, v16.4s, v22.4s
    fmul v17.4s, v17.4s, v22.4s
    fadd v16.4s, v16.4s, v23.4s
    fadd v17.4s, v17.4s, v23.4s
    fmax v16.4s, v16.4s, v21.4s
    fmax v17.4s, v17.4s, v21.4s
    st1 {v16.4s, v17.4s}, [x0], #32
    sub x4, x4, #4
    cmp x4, #4
    bge L4ReluLoop


L1Relu:
cmp x4, #0
beq EndBasicRelu

Loop1Relu:
    ld1 {v0.4h}, [x1], x12
    sxtl v0.4s, v0.4h
    mov v1.16b, v23.16b
    scvtf v0.4s, v0.4s
    fmla v1.4s, v0.4s, v22.4s
    fmax v1.4s, v1.4s, v21.4s

    st1 {v1.4s}, [x0], #16
    subs x4, x4, #1
    bne Loop1Relu


EndBasicRelu:

ret

#endif
