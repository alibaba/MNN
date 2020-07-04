//
//  MNNPackC4ForMatMul_A.S
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
asm_function MNNPackC4ForMatMul_A
//void MNNPackC4ForMatMul_A(float* dest, const float* source, size_t e, size_t l, size_t eReal)
//Auto: r0: dest, r1:source, r2: e, r3:l, r4: eReal
// eReal -> eReal * 4 * sizeof(float) - 192
push {r4-r11, lr}
ldr r4, [sp, #36]

mov r9, #4
mov r12, #16
mul r4, r12, r4
mul r8, r9, r2

sub r4, r4, #192

// Set r9 as l * 12 * sizeof(float)
mov r12, #48
mul r9, r3, r12

Body:
cmp r2, #12
blt Right

LoopE12:
    mov r6, r0
    mov r7, r1
    mov r5, r3
    cmp r5, #4
    blt LoopEL3
    LoopL4:
.macro MAIN_TRANSPOSE
        vld1.32 {q0, q1}, [r1]!
        vld1.32 {q2, q3}, [r1]!

        vld1.32 {q8, q9}, [r1]!
        vld1.32 {q10, q11}, [r1]!

        vld1.32 {q12, q13}, [r1]!
        vld1.32 {q14, q15}, [r1]!

        vtrn.32 d0, d2
        vtrn.32 d1, d3
        vtrn.32 d4, d6
        vtrn.32 d5, d7

        vswp d1, d4
        vswp d3, d6

        vtrn.32 d16, d18
        vtrn.32 d17, d19
        vtrn.32 d20, d22
        vtrn.32 d21, d23

        vswp d17, d20
        vswp d19, d22

        vtrn.32 d24, d26
        vtrn.32 d25, d27
        vtrn.32 d28, d30
        vtrn.32 d29, d31

        vswp d25, d28
        vswp d27, d30
.endm
        MAIN_TRANSPOSE

        vst1.32 {q0}, [r0]!
        vst1.32 {q8}, [r0]!
        vst1.32 {q12}, [r0]!

        vst1.32 {q1}, [r0]!
        vst1.32 {q9}, [r0]!
        vst1.32 {q13}, [r0]!

        vst1.32 {q2}, [r0]!
        vst1.32 {q10}, [r0]!
        vst1.32 {q14}, [r0]!

        vst1.32 {q3}, [r0]!
        vst1.32 {q11}, [r0]!
        vst1.32 {q15}, [r0]!

        add r1, r1, r4
        sub r5, r5, #4
        cmp r5, #4
        bge LoopL4

    LoopEL3:
    cmp r5, #3
    blt LoopEL2
        MAIN_TRANSPOSE

        vst1.32 {q0}, [r0]!
        vst1.32 {q8}, [r0]!
        vst1.32 {q12}, [r0]!

        vst1.32 {q1}, [r0]!
        vst1.32 {q9}, [r0]!
        vst1.32 {q13}, [r0]!

        vst1.32 {q2}, [r0]!
        vst1.32 {q10}, [r0]!
        vst1.32 {q14}, [r0]!


        sub r5, r5, #3

    LoopEL2:
    cmp r5, #2
    blt LoopEL1
        MAIN_TRANSPOSE
        vst1.32 {q0}, [r0]!
        vst1.32 {q8}, [r0]!
        vst1.32 {q12}, [r0]!

        vst1.32 {q1}, [r0]!
        vst1.32 {q9}, [r0]!
        vst1.32 {q13}, [r0]!
        sub r5, r5, #2

    LoopEL1:
    cmp r5, #1
    blt LoopEEnd
        MAIN_TRANSPOSE
        vst1.32 {q0}, [r0]!
        vst1.32 {q8}, [r0]!
        vst1.32 {q12}, [r0]!
    LoopEEnd:

    sub r2, r2, #12
    cmp r2, #12
    add r0, r6, r9
    add r1, r7, #192 // 12 * 4 * sizeof(float)
    bge LoopE12

cmp r2, #0
beq End

Right:
add r4, r4, #192

LoopE1:
    mov r6, r0
    mov r7, r1
    mov r5, r3
    cmp r5, #4
    blt LoopE1L3
    LoopE1L4:
        vld1.32 {q0}, [r1], r4
        vst1.32 {d0[0]}, [r0], r8
        vst1.32 {d0[1]}, [r0], r8
        vst1.32 {d1[0]}, [r0], r8
        vst1.32 {d1[1]}, [r0], r8
        sub r5, r5, #4
        cmp r5, #4
        bge LoopE1L4

    LoopE1L3:
    cmp r5, #3
    blt LoopE1L2
        vld1.32 {q0}, [r1], r4
        vst1.32 {d0[0]}, [r0], r8
        vst1.32 {d0[1]}, [r0], r8
        vst1.32 {d1[0]}, [r0], r8

        sub r5, r5, #3

    LoopE1L2:
    cmp r5, #2
    blt LoopE1L1
        vld1.32 {d0}, [r1], r4
        vst1.32 {d0[0]}, [r0], r8
        vst1.32 {d0[1]}, [r0], r8
        sub r5, r5, #2

    LoopE1L1:
    cmp r5, #1
    blt LoopE1End
        vld1.32 {d0[0]}, [r1], r4
        vst1.32 {d0[0]}, [r0], r8

    LoopE1End:

    subs r2, r2, #1
    add r0, r6, #4
    add r1, r7, #16 // 4 * sizeof(float)
    bne LoopE1

End:

pop {r4-r11, pc}

#endif
#endif
