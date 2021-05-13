//
//  MNNPackedMatMulFP16.S
//  MNN
//
//  Created by MNN on 2020/06/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef __arm__
#ifndef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5
// 12 * 4 MatMul
asm_function MNNPackedMatMulFP16
//void MNNPackedMatMulFP16(FLOAT16* C, const FLOAT16* A, const FLOAT16* B, const size_t* parameter, const FLOAT16* postParameters, const FLOAT16* bias);
// Auto: r0: C, r1:A, r2:B, r3:parameter
// Load from sp: r5: postParameters, r6:bias

push {r4-r11, lr}
ldr r5, [sp, #36]
ldr r6, [sp, #40]

ldr r4, [r3, #8] // h
ldr r7, [r3, #4] // l
add r4, r4, #7
ldr r8, [r3, #12]//cStride
ldr r3, [r3, #20]//bExtraStride
lsr r4, r4, #3

sub r8, r8, #192

vpush {q4-q7}
// q0, q1: src
// q3: weight
// q4 - q15: dst

cmp r5, #0
beq LoopH
vld1.32 {q0}, [r5]
vcvt.f16.f32 d0, q0

.macro COMPUTE op, s0, s1, d0, d1, d2, d3
    \op \d0, \s0, \s1[0]
    \op \d1, \s0, \s1[1]
    \op \d2, \s0, \s1[2]
    \op \d3, \s0, \s1[3]
.endm

.macro CLIP op, s0, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11
    \op \d0, \d0, \s0
    \op \d1, \d1, \s0
    \op \d2, \d2, \s0
    \op \d3, \d3, \s0
    \op \d4, \d4, \s0
    \op \d5, \d5, \s0
    \op \d6, \d6, \s0
    \op \d7, \d7, \s0
    \op \d8, \d8, \s0
    \op \d9, \d9, \s0
    \op \d10, \d10, \s0
    \op \d11, \d11, \s0
.endm

LoopH:
    subs r12, r7, #1
    mov r11, r1
    vld1.16 {q3}, [r2]!
    vld1.16 {d1, d2, d3}, [r11]!
    COMPUTE vmul.f16, q3, d1, q4, q5, q6, q7
    COMPUTE vmul.f16, q3, d2, q8, q9, q10, q11
    COMPUTE vmul.f16, q3, d3, q12, q13, q14, q15
    beq LoopLEnd
    LoopL:
        vld1.16 {q3}, [r2]!
        vld1.16 {d1, d2, d3}, [r11]!
        COMPUTE vmla.f16, q3, d1, q4, q5, q6, q7
        COMPUTE vmla.f16, q3, d2, q8, q9, q10, q11
        COMPUTE vmla.f16, q3, d3, q12, q13, q14, q15

        subs r12, r12, #1
        bne LoopL
    LoopLEnd:
    cmp r5, #0
    beq Store
    vld1.16 {q3}, [r6]!
    vmla.f16 q4,  q3, d0[1]
    vmla.f16 q5,  q3, d0[1]
    vmla.f16 q6,  q3, d0[1]
    vmla.f16 q7,  q3, d0[1]
    vmla.f16 q8,  q3, d0[1]
    vmla.f16 q9,  q3, d0[1]
    vmla.f16 q10, q3, d0[1]
    vmla.f16 q11, q3, d0[1]
    vmla.f16 q12, q3, d0[1]
    vmla.f16 q13, q3, d0[1]
    vmla.f16 q14, q3, d0[1]
    vmla.f16 q15, q3, d0[1]

    b PostTreat

    LoadOrigin:
    mov r11, r0
    vld1.16 {q1, q2}, [r11]!
    vmla.f16 q4, q1, d0[1]
    vmla.f16 q5, q2, d0[1]

    vld1.16 {q1, q2}, [r11]!
    vmla.f16 q6, q1, d0[1]
    vmla.f16 q7, q2, d0[1]

    vld1.16 {q1, q2}, [r11]!
    vmla.f16 q8, q1, d0[1]
    vmla.f16 q9, q2, d0[1]

    vld1.16 {q1, q2}, [r11]!
    vmla.f16 q10, q1, d0[1]
    vmla.f16 q11, q2, d0[1]

    vld1.16 {q1, q2}, [r11]!
    vmla.f16 q12, q1, d0[1]
    vmla.f16 q13, q2, d0[1]

    vld1.16 {q1, q2}, [r11]!
    vmla.f16 q14, q1, d0[1]
    vmla.f16 q15, q2, d0[1]

    PostTreat:
    vdup.16 q2, d0[2] // min
    vdup.16 q1, d0[3] // max

    CLIP vmax.f16, q2, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, q15
    CLIP vmin.f16, q1, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, q15

    Store:
    vst1.16 {q4, q5}, [r0]!
    vst1.16 {q6, q7}, [r0]!
    vst1.16 {q8, q9}, [r0]!
    vst1.16 {q10, q11}, [r0]!
    vst1.16 {q12, q13}, [r0]!
    vst1.16 {q14, q15}, [r0]!

    add r0, r0, r8
    add r2, r2, r3

    subs r4, r4, #1
    bne LoopH

End:
vpop {q4-q7}
pop {r4-r11, pc}

#endif
#endif
