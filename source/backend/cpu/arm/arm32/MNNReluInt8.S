//
//  MNNReluInt8.S
//  MNN
//
//  Created by MNN on 2019/06/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __arm__
#ifndef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNReluInt8
// void MNNReluInt8(int8_t* dst, const int8_t* src, size_t size, size_t zeroPoint)
// Auto
// r0: dst, r1: src, r2: size, r3: zeroPoint

push {lr}

cmp r2, #0
beq ReluEnd

vdup.i8 q15, r3

L16:
cmp r2, #16
blt L8

L16Loop:
    vld1.32 {q0}, [r1]!
    vmaxq.s8 q1, q0, q15
    sub r2, r2, #16
    vst1.32 {q1}, [r0]!
    cmp r2, #16
    bge L16Loop

L8:
cmp r2, #8
blt L4

L8Loop:
    vld1.32 {d0}, [r1]!
    vmax.s8 d1, d0, d30
    sub r2, r2, #8
    vst1.32 {d1}, [r0]!
    cmp r2, #8
    bge L8Loop

L4:
cmp r2, #4
blt ReluEnd

L4Loop:
    vld1.32 {d0[0]}, [r1]!
    vmax.s8 d1, d0, d30
    sub r2, r2, #4
    vst1.32 {d1[0]}, [r0]!
    cmp r2, #4
    bge L4Loop

ReluEnd:
pop {pc}

#endif
#endif
