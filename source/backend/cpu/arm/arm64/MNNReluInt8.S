//
//  MNNReluInt8.S
//  MNN
//
//  Created by MNN on 2019/06/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNReluInt8
// void MNNReluInt8(int8_t* dst, const int8_t* src, size_t size, size_t zeroPoint)
// Auto
// x0: dst, x1: src, x2: size, x3: zeroPoint

cmp x2, #0
beq ReluEnd

dup v7.16b, w3

L16:
cmp x2, #16
blt L8

L16Loop:
    sub x2, x2, #16
    ld1 {v0.16b}, [x1], #16
    smax v1.16b, v0.16b, v7.16b
    st1 {v1.16b}, [x0], #16

    cmp x2, #16
    bge L16Loop


L8:
cmp x2, #8
blt L4

L8Loop:
    sub x2, x2, #8
    ld1 {v0.8b}, [x1], #8
    smax v1.8b, v0.8b, v7.8b
    st1 {v1.8b}, [x0], #8
    cmp x2, #8
    bge L8Loop

L4:
cmp x2, #4
blt ReluEnd

L4Loop:
    sub x2, x2, #4
    ld1 {v0.s}[0], [x1], #4
    smax v1.8b, v0.8b, v7.8b
    st1 {v1.s}[0], [x0], #4
    cmp x2, #4
    bge L4Loop

ReluEnd:
ret
#endif
