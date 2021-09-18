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
//void MNNPackC4ForMatMul_A(float* destOrigin, float const** sourceGroup, const int32_t* info, const int32_t* el)
//Auto: r0: dest, r1:sourceGroup, r2: info, r3:el
push {r4-r8, r10, r11, lr} // avoid to touch platform-register r-9
ldr r10, [r2, #0] // number
ldr r4, [r2, #4] // eReal
ldr r11, [r2, #8] // eDest
ldr r6, [r2, #12] // xOffset
// xOffset -> xOffset * 4 * sizeof(float)
// eReal -> eReal * 4 * sizeof(float)
// eDest -> eDest * sizeof(float)
mov r12, #4 // sizeof(float).  kept as a const
mov lr, #16
mul r4, lr, r4
mul r11, r12, r11
mul r6, lr, r6

LoopNumber:
ldr r5, [r3, #4] // l
ldr r8, [r3, #8] // eOffset
ldr r7, [r3, #12] // lOffset

push {r0, r1}
ldr r1, [r1, #0]

// Compute dest ptr: r0 = r0 + eOffset * sizeof(float) + lOffset * eDest * sizeof(float)
mul r7, r11, r7
mul r8, r12, r8
add r0, r0, r7
add r0, r0, r8

mov r2, #12         // the fast-pack-eSize
mul r2, r12, r2     // fast-pack-eSize * sizeof(dataType)
cmp r2, r11         // check eP==fast-pack-eSize

ldr r2, [r3, #0] // e
bne Right

Body:
cmp r2, #12
bne Right
    cmp r5, #4
    blt LoopEL3
    LoopL4:
        mov r2, r1
.macro MAIN_TRANSPOSE
        vld1.32 {q0}, [r1], r6
        vld1.32 {q1}, [r1], r6
        vld1.32 {q2}, [r1], r6
        vld1.32 {q3}, [r1], r6
        vld1.32 {q8}, [r1], r6
        vld1.32 {q9}, [r1], r6
        vld1.32 {q10}, [r1], r6
        vld1.32 {q11}, [r1], r6
        vld1.32 {q12}, [r1], r6
        vld1.32 {q13}, [r1], r6
        vld1.32 {q14}, [r1], r6
        vld1.32 {q15}, [r1], r6

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

        add r1, r2, r4
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

        b LoopEEnd

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
        b LoopEEnd

    LoopEL1:
    cmp r5, #0
    beq LoopEEnd
        MAIN_TRANSPOSE
        vst1.32 {q0}, [r0]!
        vst1.32 {q8}, [r0]!
        vst1.32 {q12}, [r0]!
    LoopEEnd:

b End


Right:

LoopE1:
    mov lr, r5
    mov r7, r1
    mov r8, r0
    cmp r5, #4
    blt LoopE1L3
    LoopE1L4:
        vld1.32 {q0}, [r1], r4
        vst1.32 {d0[0]}, [r0], r11
        vst1.32 {d0[1]}, [r0], r11
        vst1.32 {d1[0]}, [r0], r11
        vst1.32 {d1[1]}, [r0], r11
        sub r5, r5, #4
        cmp r5, #4
        bge LoopE1L4

    LoopE1L3:
    cmp r5, #3
    blt LoopE1L2
        vld1.32 {q0}, [r1], r4
        vst1.32 {d0[0]}, [r0], r11
        vst1.32 {d0[1]}, [r0], r11
        vst1.32 {d1[0]}, [r0], r11

        sub r5, r5, #3

    LoopE1L2:
    cmp r5, #2
    blt LoopE1L1
        vld1.32 {d0}, [r1], r4
        vst1.32 {d0[0]}, [r0], r11
        vst1.32 {d0[1]}, [r0], r11
        sub r5, r5, #2

    LoopE1L1:
    cmp r5, #1
    blt LoopE1End
        vld1.32 {d0[0]}, [r1], r4
        vst1.32 {d0[0]}, [r0], r11

    LoopE1End:

    subs r2, r2, #1
    add r0, r8, r12
    add r1, r7, r6
    mov r5, lr
    bne LoopE1

End:

pop {r0, r1}
subs r10, r10, #1
add r3, r3, #16
add r1, r1, #4

bne LoopNumber


pop {r4-r8, r10, r11, pc}

#endif
#endif