//
//  MNNCopyC4WithStride.S
//  MNN
//
//  Created by MNN on 2018/10/09.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNCopyC4WithStride
//void MNNCopyC4WithStride(const float* source, float* dest, size_t srcStride, size_t dstStride, size_t count);

//Auto
//x0:source, x1:dest, x2: srcStride, x3:dstStride

mov x12, #4
mul x2, x12, x2
mul x3, x12, x3

L8:
cmp x4, #8
blt L1

L8Loop:
    ld1 {v0.4s}, [x0], x2
    st1 {v0.4s}, [x1], x3
    ld1 {v1.4s}, [x0], x2
    st1 {v1.4s}, [x1], x3
    ld1 {v2.4s}, [x0], x2
    st1 {v2.4s}, [x1], x3
    ld1 {v3.4s}, [x0], x2
    st1 {v3.4s}, [x1], x3
    ld1 {v0.4s}, [x0], x2
    st1 {v0.4s}, [x1], x3
    ld1 {v1.4s}, [x0], x2
    st1 {v1.4s}, [x1], x3
    ld1 {v2.4s}, [x0], x2
    st1 {v2.4s}, [x1], x3
    ld1 {v3.4s}, [x0], x2
    st1 {v3.4s}, [x1], x3

    sub x4, x4, #8
    cmp x4, #8
    bge L8Loop

L1:
cmp x4, #0
beq End

L1Loop:
    ld1 {v0.4s}, [x0], x2
    st1 {v0.4s}, [x1], x3

    subs x4, x4, #1
    bne L1Loop

End:

ret

#endif
