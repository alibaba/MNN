//
//  MNNMatrixCopyUnit.S
//  MNN
//
//  Created by MNN on 2020/01/21.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNMatrixCopyUnit
//void MNNMatrixCopyUnit(float* C, const float* A, size_t cStride, size_t aStride, size_t height)

//Auto: x0: C, x1:A, x2:cStride
//x3:aStride, x4:height

mov x12, #4 //sizeof(float)
mul x2, x12, x2
mul x3, x12, x3

subs x4, x4, #1
mov x8, x0
mov x9, x1
ld1 {v18.4s, v19.4s, v20.4s, v21.4s}, [x1], #64
ld1 {v22.4s, v23.4s, v24.4s, v25.4s}, [x1], #64
ld1 {v26.4s, v27.4s, v28.4s, v29.4s}, [x1], #64

beq LoopYEnd

LoopY:
    // Unit = 14 for arm64a
    st1 {v18.4s, v19.4s, v20.4s, v21.4s}, [x0], #64
    st1 {v22.4s, v23.4s, v24.4s, v25.4s}, [x0], #64
    ld1 {v30.4s, v31.4s}, [x1]
    st1 {v26.4s, v27.4s, v28.4s, v29.4s}, [x0], #64
    st1 {v30.4s, v31.4s}, [x0]
    add x1, x9, x3
    add x0, x8, x2
    mov x8, x0
    mov x9, x1
    ld1 {v18.4s, v19.4s, v20.4s, v21.4s}, [x1], #64
    ld1 {v22.4s, v23.4s, v24.4s, v25.4s}, [x1], #64
    ld1 {v26.4s, v27.4s, v28.4s, v29.4s}, [x1], #64

    subs x4, x4, #1
    bne LoopY

LoopYEnd:

st1 {v18.4s, v19.4s, v20.4s, v21.4s}, [x0], #64
st1 {v22.4s, v23.4s, v24.4s, v25.4s}, [x0], #64
ld1 {v30.4s, v31.4s}, [x1]
st1 {v26.4s, v27.4s, v28.4s, v29.4s}, [x0], #64
st1 {v30.4s, v31.4s}, [x0]


ret

#endif
