//
//  MNNPackedMatMul.S
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
asm_function MNNPackedMatMul
//void MNNPackedMatMul(float* C, const float* A, const float* B, const size_t* parameter, const float* postParameters, const float* bias);
// Auto: r0: C, r1:A, r2:B, r3:parameter
// Load from sp: r5: postParameters, r6:bias

push {r4-r8, r10, r11, lr} // avoid to touch platform-register r-9
ldr r5, [sp, #32]
ldr r6, [sp, #36]

ldr r4, [r3, #8] // h
ldr r7, [r3, #4] // l
add r4, r4, #3
ldr r8, [r3, #12]//cStride
ldr r3, [r3, #20]//bExtraStride
lsr r4, r4, #2

sub r8, r8, #192

vpush {q4-q7}
// q0, q1, q2: src
// q3: weight
// q4 - q15: dst

LoopH:
    subs r12, r7, #1
    mov r11, r1
    vld1.32 {q3}, [r2]!
    vld1.32 {q0, q1}, [r11]!
    vmul.f32 q4, q3, d0[0]
    vmul.f32 q5, q3, d0[1]
    vmul.f32 q6, q3, d1[0]
    vld1.32 {q2}, [r11]!
    vmul.f32 q7, q3, d1[1]

    vmul.f32 q8, q3, d2[0]
    vmul.f32 q9, q3, d2[1]
    vmul.f32 q10, q3, d3[0]
    vmul.f32 q11, q3, d3[1]

    vmul.f32 q12, q3, d4[0]
    vmul.f32 q13, q3, d4[1]
    vmul.f32 q14, q3, d5[0]
    vmul.f32 q15, q3, d5[1]
    beq LoopLEnd
    LoopL:
        vld1.32 {q3}, [r2]!
        vld1.32 {q0, q1}, [r11]!
        vmla.f32 q4, q3, d0[0]
        vmla.f32 q5, q3, d0[1]
        vmla.f32 q6, q3, d1[0]
        vld1.32 {q2}, [r11]!
        vmla.f32 q7, q3, d1[1]

        vmla.f32 q8, q3, d2[0]
        vmla.f32 q9, q3, d2[1]
        vmla.f32 q10, q3, d3[0]
        vmla.f32 q11, q3, d3[1]

        vmla.f32 q12, q3, d4[0]
        vmla.f32 q13, q3, d4[1]
        vmla.f32 q14, q3, d5[0]
        vmla.f32 q15, q3, d5[1]

        subs r12, r12, #1
        bne LoopL
    LoopLEnd:
    cmp r5, #0
    beq Store
    vld1.32 {q0}, [r5]
    cmp r6, #0
    beq LoadOrigin
    vld1.32 {q3}, [r6]!
    vmla.f32 q4,  q3, d0[1]
    vmla.f32 q5,  q3, d0[1]
    vmla.f32 q6,  q3, d0[1]
    vmla.f32 q7,  q3, d0[1]
    vmla.f32 q8,  q3, d0[1]
    vmla.f32 q9,  q3, d0[1]
    vmla.f32 q10, q3, d0[1]
    vmla.f32 q11, q3, d0[1]
    vmla.f32 q12, q3, d0[1]
    vmla.f32 q13, q3, d0[1]
    vmla.f32 q14, q3, d0[1]
    vmla.f32 q15, q3, d0[1]

    b PostTreat

    LoadOrigin:
    mov r11, r0
    vld1.32 {q1, q2}, [r11]!
    vmla.f32 q4, q1, d0[1]
    vmla.f32 q5, q2, d0[1]

    vld1.32 {q1, q2}, [r11]!
    vmla.f32 q6, q1, d0[1]
    vmla.f32 q7, q2, d0[1]

    vld1.32 {q1, q2}, [r11]!
    vmla.f32 q8, q1, d0[1]
    vmla.f32 q9, q2, d0[1]

    vld1.32 {q1, q2}, [r11]!
    vmla.f32 q10, q1, d0[1]
    vmla.f32 q11, q2, d0[1]

    vld1.32 {q1, q2}, [r11]!
    vmla.f32 q12, q1, d0[1]
    vmla.f32 q13, q2, d0[1]

    vld1.32 {q1, q2}, [r11]!
    vmla.f32 q14, q1, d0[1]
    vmla.f32 q15, q2, d0[1]

    PostTreat:
    vdup.f32 q2, d1[0] // min
    vdup.f32 q1, d1[1] // max

    vmax.f32 q4, q4, q2
    vmax.f32 q5, q5, q2
    vmax.f32 q6, q6, q2
    vmax.f32 q7, q7, q2
    vmax.f32 q8, q8, q2
    vmax.f32 q9, q9, q2
    vmax.f32 q10, q10, q2
    vmax.f32 q11, q11, q2
    vmax.f32 q12, q12, q2
    vmax.f32 q13, q13, q2
    vmax.f32 q14, q14, q2
    vmax.f32 q15, q15, q2

    vmin.f32 q4, q4, q1
    vmin.f32 q5, q5, q1
    vmin.f32 q6, q6, q1
    vmin.f32 q7, q7, q1
    vmin.f32 q8, q8, q1
    vmin.f32 q9, q9, q1
    vmin.f32 q10, q10, q1
    vmin.f32 q11, q11, q1
    vmin.f32 q12, q12, q1
    vmin.f32 q13, q13, q1
    vmin.f32 q14, q14, q1
    vmin.f32 q15, q15, q1

    Store:
    vst1.32 {q4, q5}, [r0]!
    vst1.32 {q6, q7}, [r0]!
    vst1.32 {q8, q9}, [r0]!
    vst1.32 {q10, q11}, [r0]!
    vst1.32 {q12, q13}, [r0]!
    vst1.32 {q14, q15}, [r0]!

    add r0, r0, r8
    add r2, r2, r3

    subs r4, r4, #1
    bne LoopH

vpop {q4-q7}
pop {r4-r8, r10, r11, pc}

#endif
#endif
