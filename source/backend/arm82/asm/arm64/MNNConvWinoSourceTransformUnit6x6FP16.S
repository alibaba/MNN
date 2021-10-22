//
//  MNNConvWinoSourceTransformUnit6x6FP16.S
//  MNN
//
//  Created by MNN on 2021/10/18.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNConvWinoSourceTransformUnit6x6FP16
//    void MNNConvWinoSourceTransformUnit6x6FP16(const FLOAT16* srcBlock, FLOAT16* dstStart, size_t srcStep, size_t dstStep);

//Auto:
//x0: srcBlock, x1:dstStart, x2:srcStep, x3:dstStep

lsl x2, x2, #1
lsl x3, x3, #1

ld1 {v16.8h}, [x0], x2          //s0
ld1 {v17.8h}, [x0], x2          //s1
ld1 {v18.8h}, [x0], x2          //s2
ld1 {v19.8h}, [x0], x2          //s3
ld1 {v20.8h}, [x0], x2          //s4
ld1 {v21.8h}, [x0], x2          //s5

movi v26.8h, #0xc4, lsl #8

fsub v23.8h, v19.8h, v17.8h
fsub v22.8h, v20.8h, v18.8h     //b4
fadd v23.8h, v23.8h, v23.8h     //b5

fadd v24.8h, v22.8h, v23.8h     //m3
fsub v25.8h, v22.8h, v23.8h     //m4

fmla v20.8h, v26.8h, v18.8h     //b0
fmla v21.8h, v26.8h, v19.8h     //b1
fmla v18.8h, v26.8h, v16.8h     //b2
fmla v19.8h, v26.8h, v17.8h     //b3

fsub v18.8h, v20.8h, v18.8h     //m0
fsub v21.8h, v21.8h, v19.8h     //m5
fadd v26.8h, v20.8h, v19.8h     //m1
fsub v27.8h, v20.8h, v19.8h     //m2

st1 {v18.8h}, [x1], x3
st1 {v26.8h}, [x1], x3
st1 {v27.8h}, [x1], x3
st1 {v24.8h}, [x1], x3
st1 {v25.8h}, [x1], x3
st1 {v21.8h}, [x1], x3

ret

#endif
