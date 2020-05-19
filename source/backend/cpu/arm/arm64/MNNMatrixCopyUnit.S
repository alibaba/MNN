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

sub sp, sp, #128
st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64

mov x12, #4 //sizeof(float)
mul x2, x12, x2
mul x3, x12, x3
mov x12, #4 // cache prefetch param
mul x5, x3, x12

cmp x4, #4
blt L1Loop

L4Loop:
    add x9, x1, x5
    prfm pldl1keep, [x9]
    prfm pldl1keep, [x9, #64]
    add x9, x9, x3
    prfm pldl1keep, [x9]
    prfm pldl1keep, [x9, #64]
    add x9, x9, x3
    prfm pldl1keep, [x9]
    prfm pldl1keep, [x9, #64]
    add x9, x9, x3
    prfm pldl1keep, [x9]
    prfm pldl1keep, [x9, #64]

    mov x9, x1
    ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x1], #64
    ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x1], #64
    add x1, x9, x3
    mov x9, x1
    ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x1], #64
    ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [x1], #64
    add x1, x9, x3
    mov x9, x1
    ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x1], #64
    ld1 {v20.4s, v21.4s, v22.4s, v23.4s}, [x1], #64
    add x1, x9, x3
    mov x9, x1
    ld1 {v24.4s, v25.4s, v26.4s, v27.4s}, [x1], #64
    ld1 {v28.4s, v29.4s, v30.4s, v31.4s}, [x1], #64
    add x1, x9, x3
    mov x8, x0
    st1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x0], #64
    st1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x0], #64
    add x0, x8, x2
    mov x8, x0
    st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x0], #64
    st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [x0], #64
    add x0, x8, x2
    mov x8, x0
    st1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x0], #64
    st1 {v20.4s, v21.4s, v22.4s, v23.4s}, [x0], #64
    add x0, x8, x2
    mov x8, x0
    st1 {v24.4s, v25.4s, v26.4s, v27.4s}, [x0], #64
    st1 {v28.4s, v29.4s, v30.4s, v31.4s}, [x0], #64
    add x0, x8, x2
    subs x4, x4, #4
    cmp x4, #3
    bgt L4Loop

cbz x4, LoopEnd

L1Loop:
    mov x9, x1
    ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x1], #64
    ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x1], #64
    mov x8, x0
    st1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x0], #64
    st1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x0], #64
    add x1, x9, x3
    add x0, x8, x2
    subs x4, x4, #1
    bne L1Loop

LoopEnd:

sub sp, sp, #128
ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64

ret

#endif
