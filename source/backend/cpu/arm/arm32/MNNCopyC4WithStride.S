//
//  MNNCopyC4WithStride.S
//  MNN
//
//  Created by MNN on 2018/10/09.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __arm__
#ifndef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNCopyC4WithStride
//void MNNCopyC4WithStride(const float* source, float* dest, size_t srcStride, size_t dstStride, size_t count);

//Auto
//r0:source, r1:dest, r2: srcStride, r3:dstStride

//Load from sp: r4:count
push {r4, lr}
ldr r4, [sp, #8]

mov r12, #4
mul r2, r12, r2
mul r3, r12, r3

L8:
cmp r4, #8
blt L1

L8Loop:
    mov r12, r1
    vld1.32 {q0}, [r0], r2
    vst1.32 {q0}, [r1], r3
    vld1.32 {q1}, [r0], r2
    vst1.32 {q1}, [r1], r3
    vld1.32 {q2}, [r0], r2
    vst1.32 {q2}, [r1], r3
    vld1.32 {q3}, [r0], r2
    vst1.32 {q3}, [r1], r3
    vld1.32 {q0}, [r0], r2
    vst1.32 {q0}, [r1], r3
    vld1.32 {q1}, [r0], r2
    vst1.32 {q1}, [r1], r3
    vld1.32 {q2}, [r0], r2
    vst1.32 {q2}, [r1], r3
    vld1.32 {q3}, [r0], r2
    vst1.32 {q3}, [r1], r3

    sub r4, r4, #8
    cmp r4, #8
    bge L8Loop

L1:
cmp r4, #0
beq End

L1Loop:
    vld1.32 {q0}, [r0], r2
    vst1.32 {q0}, [r1], r3

    subs r4, r4, #1
    bne L1Loop

End:

pop {r4, pc}


#endif
#endif
