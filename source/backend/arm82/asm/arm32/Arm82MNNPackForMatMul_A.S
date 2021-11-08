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
asm_function Arm82MNNPackForMatMul_A
//void Arm82MNNPackForMatMul_A(FLOAT16* destOrigin, FLOAT16 const** sourceGroup, const int32_t* info, const int32_t* el)
//Auto: r0: dest, r1:sourceGroup, r2: info, r3:el
push {r4-r11, lr}
vpush {q4-q7}
ldr r10, [r2, #0] // number
ldr r4, [r2, #4] // eReal
ldr r11, [r2, #8] // eDest
ldr r6, [r2, #12] // xOffset
// xOffset -> xOffset * 8 * sizeof(FLOAT16)
// eReal -> eReal * 8 * sizeof(FLOAT16)
// eDest -> eDest * sizeof(FLOAT16)
mov r12, #2 // sizeof(FLOAT16)
mov r9, #16 // 8 * sizeof(FLOAT16)
mul r4, r9, r4
mul r11, r12, r11
mul r6, r9, r6

LoopNumber:
ldr r5, [r3, #4] // l
ldr r8, [r3, #8] // eOffset
ldr r7, [r3, #12] // lOffset

push {r0, r1}
ldr r1, [r1, #0]

// Compute dest ptr: r0 = r0 + eOffset * sizeof(float) + lOffset * eDest * sizeof(float)
mul r7, r11, r7
lsl r8, r8, #1 // r8 = r8 * sizeof(FLOAT16)
add r0, r0, r7
add r0, r0, r8

ldr r2, [r3, #0] // e

E12:
cmp r2, #12
blt E8
    cmp r5, #8
    blt E12_LoopLExtra
    E12_LoopL8:
        mov r12, r1
// {q0-q11} => [[d0, d8, d16],
//              [d2, d10, d18],
//              [d4, d12, d20],
//              [d6, d14, d22],
//              [d1, d9, d17],
//              [d3, d11, d19],
//              [d5, d13, d21],
//              [d7, d15, d23]]
.macro TRANSPOSE_8X12
    vld1.16 {q0}, [r1], r6
    vld1.16 {q1}, [r1], r6
    vld1.16 {q2}, [r1], r6
    vld1.16 {q3}, [r1], r6
    vld1.16 {q4}, [r1], r6
    vld1.16 {q5}, [r1], r6
    vld1.16 {q6}, [r1], r6
    vld1.16 {q7}, [r1], r6
    vld1.16 {q8}, [r1], r6
    vld1.16 {q9}, [r1], r6
    vld1.16 {q10}, [r1], r6
    vld1.16 {q11}, [r1], r6
    vtrn.16 d0, d2
    vtrn.16 d4, d6
    vtrn.16 d1, d3
    vtrn.16 d5, d7
    vtrn.16 d8, d10
    vtrn.16 d12, d14
    vtrn.16 d9, d11
    vtrn.16 d13, d15
    vtrn.16 d16, d18
    vtrn.16 d20, d22
    vtrn.16 d17, d19
    vtrn.16 d21, d23
    vtrn.32 d0, d4
    vtrn.32 d2, d6
    vtrn.32 d1, d5
    vtrn.32 d3, d7
    vtrn.32 d8, d12
    vtrn.32 d10, d14
    vtrn.32 d9, d13
    vtrn.32 d11, d15
    vtrn.32 d16, d20
    vtrn.32 d18, d22
    vtrn.32 d17, d21
    vtrn.32 d19, d23
.endm
.macro STORE_LINE_12 addr, v0, v1, v2
    vst1.16 {\v0}, [\addr]!
    vst1.16 {\v1}, [\addr]!
    vst1.16 {\v2}, [\addr]!
.endm
        TRANSPOSE_8X12
        STORE_LINE_12 r0, d0, d8, d16
        STORE_LINE_12 r0, d2, d10, d18
        STORE_LINE_12 r0, d4, d12, d20
        STORE_LINE_12 r0, d6, d14, d22
        STORE_LINE_12 r0, d1, d9, d17
        STORE_LINE_12 r0, d3, d11, d19
        STORE_LINE_12 r0, d5, d13, d21
        STORE_LINE_12 r0, d7, d15, d23

        add r1, r12, r4
        sub r5, r5, #8
        cmp r5, #8
        bge E12_LoopL8

    cmp r5, #0
    beq E12_LoopLEnd
    E12_LoopLExtra:
        TRANSPOSE_8X12
    E12_LoopL7:
        cmp r5, #7 // if r5 < 7
        blt E12_LoopL6 // jump to E12_LoopL6
        STORE_LINE_12 r0, d0, d8, d16
        STORE_LINE_12 r0, d2, d10, d18
        STORE_LINE_12 r0, d4, d12, d20
        STORE_LINE_12 r0, d6, d14, d22
        STORE_LINE_12 r0, d1, d9, d17
        STORE_LINE_12 r0, d3, d11, d19
        STORE_LINE_12 r0, d5, d13, d21
        b E12_LoopLEnd
    E12_LoopL6:
        cmp r5, #6 // if r5 < 6
        blt E12_LoopL5 // jump to E12_LoopL5
        STORE_LINE_12 r0, d0, d8, d16
        STORE_LINE_12 r0, d2, d10, d18
        STORE_LINE_12 r0, d4, d12, d20
        STORE_LINE_12 r0, d6, d14, d22
        STORE_LINE_12 r0, d1, d9, d17
        STORE_LINE_12 r0, d3, d11, d19
        b E12_LoopLEnd
    E12_LoopL5:
        cmp r5, #5 // if r5 < 5
        blt E12_LoopL4 // jump to E12_LoopL4
        STORE_LINE_12 r0, d0, d8, d16
        STORE_LINE_12 r0, d2, d10, d18
        STORE_LINE_12 r0, d4, d12, d20
        STORE_LINE_12 r0, d6, d14, d22
        STORE_LINE_12 r0, d1, d9, d17
        b E12_LoopLEnd
    E12_LoopL4:
        cmp r5, #4 // if r5 < 4
        blt E12_LoopL3 // jump to E12_LoopL3
        STORE_LINE_12 r0, d0, d8, d16
        STORE_LINE_12 r0, d2, d10, d18
        STORE_LINE_12 r0, d4, d12, d20
        STORE_LINE_12 r0, d6, d14, d22
        b E12_LoopLEnd
    E12_LoopL3:
        cmp r5, #3 // if r5 < 3
        blt E12_LoopL2 // jump to E12_LoopL2
        STORE_LINE_12 r0, d0, d8, d16
        STORE_LINE_12 r0, d2, d10, d18
        STORE_LINE_12 r0, d4, d12, d20
        b E12_LoopLEnd
    E12_LoopL2:
        cmp r5, #2 // if r5 < 2
        blt E12_LoopL1 // jump to E12_LoopL1
        STORE_LINE_12 r0, d0, d8, d16
        STORE_LINE_12 r0, d2, d10, d18
        b E12_LoopLEnd
    E12_LoopL1:
        STORE_LINE_12 r0, d0, d8, d16
    E12_LoopLEnd:
        b End

E8:
cmp r2, #8
blt E4
    sub r11, r11, #8
    mov r9, r5
    mov r7, r1
    mov r8, r0
    cmp r5, #8
    blt E8_LoopLExtra
    E8_LoopL8:
        mov r12, r1
// {q0-q7} => [[d0, d8],
//             [d2, d10],
//             [d4, d12],
//             [d6, d14],
//             [d1, d9],
//             [d3, d11],
//             [d5, d13],
//             [d7, d15]]
.macro TRANSPOSE_8X8
    vld1.16 {q0}, [r1], r6
    vld1.16 {q1}, [r1], r6
    vld1.16 {q2}, [r1], r6
    vld1.16 {q3}, [r1], r6
    vld1.16 {q4}, [r1], r6
    vld1.16 {q5}, [r1], r6
    vld1.16 {q6}, [r1], r6
    vld1.16 {q7}, [r1], r6
    vtrn.16 d0, d2
    vtrn.16 d4, d6
    vtrn.16 d1, d3
    vtrn.16 d5, d7
    vtrn.16 d8, d10
    vtrn.16 d12, d14
    vtrn.16 d9, d11
    vtrn.16 d13, d15
    vtrn.32 d0, d4
    vtrn.32 d2, d6
    vtrn.32 d1, d5
    vtrn.32 d3, d7
    vtrn.32 d8, d12
    vtrn.32 d10, d14
    vtrn.32 d9, d13
    vtrn.32 d11, d15
.endm
.macro STORE_LINE_8 addr, offset, v0, v1
    vst1.16 {\v0}, [\addr]!
    vst1.16 {\v1}, [\addr], \offset
.endm
        TRANSPOSE_8X8
        STORE_LINE_8 r0, r11, d0, d8
        STORE_LINE_8 r0, r11, d2, d10
        STORE_LINE_8 r0, r11, d4, d12
        STORE_LINE_8 r0, r11, d6, d14
        STORE_LINE_8 r0, r11, d1, d9
        STORE_LINE_8 r0, r11, d3, d11
        STORE_LINE_8 r0, r11, d5, d13
        STORE_LINE_8 r0, r11, d7, d15
        add r1, r12, r4
        sub r5, r5, #8
        cmp r5, #8
        bge E8_LoopL8

    cmp r5, #0
    beq E8_LoopLEnd
    E8_LoopLExtra:
        TRANSPOSE_8X8
    E8_LoopL7:
        cmp r5, #7 // if r5 < 7
        blt E8_LoopL6 // jump to E8_LoopL6
        STORE_LINE_8 r0, r11, d0, d8
        STORE_LINE_8 r0, r11, d2, d10
        STORE_LINE_8 r0, r11, d4, d12
        STORE_LINE_8 r0, r11, d6, d14
        STORE_LINE_8 r0, r11, d1, d9
        STORE_LINE_8 r0, r11, d3, d11
        STORE_LINE_8 r0, r11, d5, d13
        b E8_LoopLEnd
    E8_LoopL6:
        cmp r5, #6 // if r5 < 6
        blt E8_LoopL5 // jump to E8_LoopL5
        STORE_LINE_8 r0, r11, d0, d8
        STORE_LINE_8 r0, r11, d2, d10
        STORE_LINE_8 r0, r11, d4, d12
        STORE_LINE_8 r0, r11, d6, d14
        STORE_LINE_8 r0, r11, d1, d9
        STORE_LINE_8 r0, r11, d3, d11
        b E8_LoopLEnd
    E8_LoopL5:
        cmp r5, #5 // if r5 < 5
        blt E8_LoopL4 // jump to E8_LoopL4
        STORE_LINE_8 r0, r11, d0, d8
        STORE_LINE_8 r0, r11, d2, d10
        STORE_LINE_8 r0, r11, d4, d12
        STORE_LINE_8 r0, r11, d6, d14
        STORE_LINE_8 r0, r11, d1, d9
        b E8_LoopLEnd
    E8_LoopL4:
        cmp r5, #4 // if r5 < 4
        blt E8_LoopL3 // jump to E8_LoopL3
        STORE_LINE_8 r0, r11, d0, d8
        STORE_LINE_8 r0, r11, d2, d10
        STORE_LINE_8 r0, r11, d4, d12
        STORE_LINE_8 r0, r11, d6, d14
        b E8_LoopLEnd
    E8_LoopL3:
        cmp r5, #3 // if r5 < 3
        blt E8_LoopL2 // jump to E8_LoopL2
        STORE_LINE_8 r0, r11, d0, d8
        STORE_LINE_8 r0, r11, d2, d10
        STORE_LINE_8 r0, r11, d4, d12
        b E8_LoopLEnd
    E8_LoopL2:
        cmp r5, #2 // if r5 < 2
        blt E8_LoopL1 // jump to E8_LoopL1
        STORE_LINE_8 r0, r11, d0, d8
        STORE_LINE_8 r0, r11, d2, d10
        b E8_LoopLEnd
    E8_LoopL1:
        STORE_LINE_8 r0, r11, d0, d8
    E8_LoopLEnd:
        add r11, r11, #8
        lsl r1, r6, #3
        add r1, r7, r1
        sub r2, r2, #8
        add r0, r8, #16 // 8 * sizeof(FLOAT16)
        mov r5, r9

E4:
cmp r2, #4
blt E1
    mov r9, r5
    mov r7, r1
    mov r8, r0
    cmp r5, #8
    blt E4_LoopLExtra
    E4_LoopL8:
        mov r12, r1
// {q0-q3} => [[d0],
//             [d2],
//             [d4],
//             [d6],
//             [d1],
//             [d3],
//             [d5],
//             [d7]]
.macro TRANSPOSE_8X4
    vld1.16 {q0}, [r1], r6
    vld1.16 {q1}, [r1], r6
    vld1.16 {q2}, [r1], r6
    vld1.16 {q3}, [r1], r6
    vtrn.16 d0, d2
    vtrn.16 d4, d6
    vtrn.16 d1, d3
    vtrn.16 d5, d7
    vtrn.32 d0, d4
    vtrn.32 d2, d6
    vtrn.32 d1, d5
    vtrn.32 d3, d7
.endm
        TRANSPOSE_8X4
        vst1.16 {d0}, [r0], r11
        vst1.16 {d2}, [r0], r11
        vst1.16 {d4}, [r0], r11
        vst1.16 {d6}, [r0], r11
        vst1.16 {d1}, [r0], r11
        vst1.16 {d3}, [r0], r11
        vst1.16 {d5}, [r0], r11
        vst1.16 {d7}, [r0], r11

        add r1, r12, r4
        sub r5, r5, #8
        cmp r5, #8
        bge E4_LoopL8

    cmp r5, #0
    beq E4_LoopLEnd
    E4_LoopLExtra:
        TRANSPOSE_8X4
    E4_LoopL7:
        cmp r5, #7 // if r5 < 7
        blt E4_LoopL6 // jump to E4_LoopL6
        vst1.16 {d0}, [r0], r11
        vst1.16 {d2}, [r0], r11
        vst1.16 {d4}, [r0], r11
        vst1.16 {d6}, [r0], r11
        vst1.16 {d1}, [r0], r11
        vst1.16 {d3}, [r0], r11
        vst1.16 {d5}, [r0], r11
        b E4_LoopLEnd
    E4_LoopL6:
        cmp r5, #6 // if r5 < 6
        blt E4_LoopL5 // jump to E4_LoopL5
        vst1.16 {d0}, [r0], r11
        vst1.16 {d2}, [r0], r11
        vst1.16 {d4}, [r0], r11
        vst1.16 {d6}, [r0], r11
        vst1.16 {d1}, [r0], r11
        vst1.16 {d3}, [r0], r11
        b E4_LoopLEnd
    E4_LoopL5:
        cmp r5, #5 // if r5 < 5
        blt E4_LoopL4 // jump to E4_LoopL4
        vst1.16 {d0}, [r0], r11
        vst1.16 {d2}, [r0], r11
        vst1.16 {d4}, [r0], r11
        vst1.16 {d6}, [r0], r11
        vst1.16 {d1}, [r0], r11
        b E4_LoopLEnd
    E4_LoopL4:
        cmp r5, #4 // if r5 < 4
        blt E4_LoopL3 // jump to E4_LoopL3
        vst1.16 {d0}, [r0], r11
        vst1.16 {d2}, [r0], r11
        vst1.16 {d4}, [r0], r11
        vst1.16 {d6}, [r0], r11
        b E4_LoopLEnd
    E4_LoopL3:
        cmp r5, #3 // if r5 < 3
        blt E4_LoopL2 // jump to E4_LoopL2
        vst1.16 {d0}, [r0], r11
        vst1.16 {d2}, [r0], r11
        vst1.16 {d4}, [r0], r11
        b E4_LoopLEnd
    E4_LoopL2:
        cmp r5, #2 // if r5 < 2
        blt E4_LoopL1 // jump to E4_LoopL1
        vst1.16 {d0}, [r0], r11
        vst1.16 {d2}, [r0], r11
        b E4_LoopLEnd
    E4_LoopL1:
        vst1.16 {d0}, [r0], r11
    E4_LoopLEnd:
        lsl r1, r6, #2
        add r1, r7, r1
        sub r2, r2, #4
        add r0, r8, #8 // 4 * sizeof(FLOAT16)
        mov r5, r9

E1:
cmp r2, #0
beq End
LoopE1:
    mov r9, r5
    mov r7, r1
    mov r8, r0
    cmp r5, #8
    blt E1_LoopL7
    E1_LoopL8:
        vld1.16 {q0}, [r1], r4
        vst1.16 {d0[0]}, [r0], r11
        vst1.16 {d0[1]}, [r0], r11
        vst1.16 {d0[2]}, [r0], r11
        vst1.16 {d0[3]}, [r0], r11
        vst1.16 {d1[0]}, [r0], r11
        vst1.16 {d1[1]}, [r0], r11
        vst1.16 {d1[2]}, [r0], r11
        vst1.16 {d1[3]}, [r0], r11
        sub r5, r5, #8
        cmp r5, #8
        bge E1_LoopL8

    E1_LoopL7:
        cmp r5, #7 // if r5 < 7
        blt E1_LoopL6 // jump to E1_LoopL6
        vld1.16 {q0}, [r1], r4
        vst1.16 {d0[0]}, [r0], r11
        vst1.16 {d0[1]}, [r0], r11
        vst1.16 {d0[2]}, [r0], r11
        vst1.16 {d0[3]}, [r0], r11
        vst1.16 {d1[0]}, [r0], r11
        vst1.16 {d1[1]}, [r0], r11
        vst1.16 {d1[2]}, [r0], r11
        b E1_LoopLEnd
    E1_LoopL6:
        cmp r5, #6 // if r5 < 6
        blt E1_LoopL5 // jump to E1_LoopL5
        vld1.16 {q0}, [r1], r4
        vst1.16 {d0[0]}, [r0], r11
        vst1.16 {d0[1]}, [r0], r11
        vst1.16 {d0[2]}, [r0], r11
        vst1.16 {d0[3]}, [r0], r11
        vst1.16 {d1[0]}, [r0], r11
        vst1.16 {d1[1]}, [r0], r11
        b E1_LoopLEnd
    E1_LoopL5:
        cmp r5, #5 // if r5 < 5
        blt E1_LoopL4 // jump to E1_LoopL4
        vld1.16 {q0}, [r1], r4
        vst1.16 {d0[0]}, [r0], r11
        vst1.16 {d0[1]}, [r0], r11
        vst1.16 {d0[2]}, [r0], r11
        vst1.16 {d0[3]}, [r0], r11
        vst1.16 {d1[0]}, [r0], r11
        b E1_LoopLEnd
    E1_LoopL4:
        cmp r5, #4 // if r5 < 4
        blt E1_LoopL3 // jump to E1_LoopL3
        vld1.16 {d0}, [r1], r4
        vst1.16 {d0[0]}, [r0], r11
        vst1.16 {d0[1]}, [r0], r11
        vst1.16 {d0[2]}, [r0], r11
        vst1.16 {d0[3]}, [r0], r11
        b E1_LoopLEnd
    E1_LoopL3:
        cmp r5, #3 // if r5 < 3
        blt E1_LoopL2 // jump to E1_LoopL2
        vld1.16 {d0}, [r1], r4
        vst1.16 {d0[0]}, [r0], r11
        vst1.16 {d0[1]}, [r0], r11
        vst1.16 {d0[2]}, [r0], r11
        b E1_LoopLEnd
    E1_LoopL2:
        cmp r5, #2 // if r5 < 2
        blt E1_LoopL1 // jump to E1_LoopL1
        vld1.16 {d0}, [r1], r4
        vst1.16 {d0[0]}, [r0], r11
        vst1.16 {d0[1]}, [r0], r11
        b E1_LoopLEnd
    E1_LoopL1:
        cmp r5, #1 // if r5 < 1
        blt E1_LoopLEnd
        vld1.16 {d0}, [r1], r4
        vst1.16 {d0[0]}, [r0], r11
    E1_LoopLEnd:
        subs r2, r2, #1
        add r0, r8, #2
        add r1, r7, r6
        mov r5, r9
        bne LoopE1

End:

pop {r0, r1}
subs r10, r10, #1
add r3, r3, #16
add r1, r1, #4

bne LoopNumber
vpop {q4-q7}
pop {r4-r11, pc}

#endif
#endif
