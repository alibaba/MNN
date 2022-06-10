//
//  MNNPackC4ForMatMul_A.S
//  MNN
//
//  Created by MNN on 2020/06/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef __aarch64__
#include "MNNAsmGlobal.h"

// [x0, x1, x2, x3] => [x0, x6, x2, x3] =mov=> [x0, x1, x2, x3]
.macro transpose_4x4 x0, x1, x2, x3, x5, x6
// x0: [00,01,02,03]    \   x5:[00,10,02,12]    \   x0:[00,10,20,30]
// x1: [10,11,12,13]  ===\  x1:[01,11,03,13]  ===\  x6:[01,11,21,31]
// x2: [20,21,22,23]  ===/  x6:[20,30,22,32]  ===/  x2:[02,12,22,32]
// x3: [30,31,32,33]    /   x3:[21,31,23,33]    /   x3:[03,13,23,33]
    trn1 \x5\().4s,  \x0\().4s, \x1\().4s
    trn2 \x1\().4s,  \x0\().4s, \x1\().4s
    trn1 \x6\().4s,  \x2\().4s, \x3\().4s
    trn2 \x3\().4s,  \x2\().4s, \x3\().4s
    trn1 \x0\().2d,  \x5\().2d, \x6\().2d
    trn2 \x2\().2d,  \x5\().2d, \x6\().2d
    trn1 \x6\().2d,  \x1\().2d, \x3\().2d
    trn2 \x3\().2d,  \x1\().2d, \x3\().2d
    mov \x1\().16b, \x6\().16b
.endm

.text
.align 5
asm_function MNNPackC4ForMatMul_A
//void MNNPackC4ForMatMul_A(float* destOrigin, float const** sourceGroup, const int32_t* info, const int32_t* el)
//Auto: x0: dest, x1:sourceGroup, x2: info, x3:el

ldr w10, [x2, #0] // number
mov x4, #0
mov x11, #0
mov x6, #0
ldr w4, [x2, #4] // eReal
ldr w11, [x2, #8] // eDest
mov x12, x11      // eP
ldr w6, [x2, #12] // xOffset
// xOffset -> xOffset * 4 * sizeof(float)
// eReal -> eReal * 4 * sizeof(float)
// eDest -> eDest * sizeof(float)
mov x9, #16
mul x4, x9, x4 // eReal * 4 * sizeof(float)
lsl x11, x11, #2 // eDest * sizeof(float)
mul x6, x9, x6 // xOffset * 4 * sizeof(float)

LoopNumber:
mov x8, #0
mov x7, #0
ldr w5, [x3, #4] // l
ldr w8, [x3, #8] // eOffset
ldr w7, [x3, #12] // lOffset

mov x13, x0
mov x14, x1
ldr x1, [x1, #0] // sourceGroup

ldr w2, [x3, #0] // e

// Compute dest ptr: x0 = x0 + eOffset * sizeof(float) + lOffset * eDest * sizeof(float)
mul x7, x11, x7
lsl x8, x8, #2
add x0, x0, x7
add x0, x0, x8

cmp w12, #16
bne E12Body

Body:

cmp w2, #16
bne Right
    cmp w5, #4
    blt LoopE16L3
    LoopE16L4:
        mov x15, x1 // sourceGroup
.macro MAIN_TRANSPOSE_16x4
        ld1 {v0.4s}, [x1], x6 // x6 = xOffset * 4 * sizeof(float)
        ld1 {v4.4s}, [x1], x6
        ld1 {v16.4s}, [x1], x6
        ld1 {v20.4s}, [x1], x6
        ld1 {v1.4s}, [x1], x6
        ld1 {v5.4s}, [x1], x6
        ld1 {v17.4s}, [x1], x6
        ld1 {v21.4s}, [x1], x6
        ld1 {v2.4s}, [x1], x6
        ld1 {v6.4s}, [x1], x6
        ld1 {v18.4s}, [x1], x6
        ld1 {v22.4s}, [x1], x6
        ld1 {v3.4s}, [x1], x6
        ld1 {v7.4s}, [x1], x6
        ld1 {v19.4s}, [x1], x6
        ld1 {v23.4s}, [x1], x6

        transpose_4x4 v0, v4, v16, v20, v24, v25
        transpose_4x4 v1, v5, v17, v21, v26, v27
        transpose_4x4 v2, v6, v18, v22, v28, v29
        transpose_4x4 v3, v7, v19, v23, v30, v31
.endm
        MAIN_TRANSPOSE_16x4

        stp q0,  q1,  [x0]
        stp q2,  q3,  [x0, #(32 * 1)]
        stp q4,  q5,  [x0, #(32 * 2)]
        stp q6,  q7,  [x0, #(32 * 3)]
        stp q16, q17, [x0, #(32 * 4)]
        stp q18, q19, [x0, #(32 * 5)]
        stp q20, q21, [x0, #(32 * 6)]
        stp q22, q23, [x0, #(32 * 7)]
        add x0, x0, #(32 * 8)

        add x1, x15, x4 // x1 = x1 + (eReal*4)block
        sub x5, x5, #4
        cmp w5, #4
        bge LoopE16L4

    LoopE16L3:
    cmp w5, #3
    blt LoopE16L2
        MAIN_TRANSPOSE_16x4

        stp q0,  q1,  [x0]
        stp q2,  q3,  [x0, #(32 * 1)]
        stp q4,  q5,  [x0, #(32 * 2)]
        stp q6,  q7,  [x0, #(32 * 3)]
        stp q16, q17, [x0, #(32 * 4)]
        stp q18, q19, [x0, #(32 * 5)]
        add x0, x0, #(32 * 6)

        b LoopE16End

    LoopE16L2:
    cmp w5, #2
    blt LoopE16L1
        MAIN_TRANSPOSE_16x4
        stp q0,  q1,  [x0]
        stp q2,  q3,  [x0, #(32 * 1)]
        stp q4,  q5,  [x0, #(32 * 2)]
        stp q6,  q7,  [x0, #(32 * 3)]
        add x0, x0, #(32 * 4)
        b LoopE16End

    LoopE16L1:
    cmp w5, #1
    blt LoopE16End
        MAIN_TRANSPOSE_16x4

        stp q0,  q1,  [x0]
        stp q2,  q3,  [x0, #(32 * 1)]
        add x0, x0, #(32 * 2)
    LoopE16End:
b End

E12Body:
cmp w12, #12
bne Right
cmp w2, #12
bne Right
    cmp w5, #4
    blt LoopE12L3
    LoopE12L4:
        mov x15, x1 // sourceGroup
.macro MAIN_TRANSPOSE_12x4
        ld1 {v0.4s}, [x1], x6 // x6 = xOffset * 4 * sizeof(float)
        ld1 {v3.4s}, [x1], x6
        ld1 {v6.4s}, [x1], x6
        ld1 {v17.4s}, [x1], x6
        ld1 {v1.4s}, [x1], x6
        ld1 {v4.4s}, [x1], x6
        ld1 {v7.4s}, [x1], x6
        ld1 {v18.4s}, [x1], x6
        ld1 {v2.4s}, [x1], x6
        ld1 {v5.4s}, [x1], x6
        ld1 {v16.4s}, [x1], x6
        ld1 {v19.4s}, [x1], x6

        transpose_4x4 v0, v3, v6, v17, v23, v24
        transpose_4x4 v1, v4, v7, v18, v25, v26
        transpose_4x4 v2, v5, v16, v19, v27, v28
.endm
        MAIN_TRANSPOSE_12x4

        stp q0,  q1,  [x0]
        stp q2,  q3,  [x0, #(32 * 1)]
        stp q4,  q5,  [x0, #(32 * 2)]
        stp q6,  q7,  [x0, #(32 * 3)]
        stp q16, q17, [x0, #(32 * 4)]
        stp q18, q19, [x0, #(32 * 5)]
        add x0, x0, #(32 * 6)

        // st1 {v0.4s}, [x0], #16
        // st1 {v4.4s}, [x0], #16
        // st1 {v16.4s}, [x0], #16
        // st1 {v1.4s}, [x0], #16
        // st1 {v5.4s}, [x0], #16
        // st1 {v17.4s}, [x0], #16
        // st1 {v2.4s}, [x0], #16
        // st1 {v6.4s}, [x0], #16
        // st1 {v18.4s}, [x0], #16
        // st1 {v3.4s}, [x0], #16
        // st1 {v7.4s}, [x0], #16
        // st1 {v19.4s}, [x0], #16

        add x1, x15, x4 // x1 = x1 + (eReal*4)block
        sub x5, x5, #4
        cmp w5, #4
        bge LoopE12L4

    LoopE12L3:
    cmp w5, #3
    blt LoopE12L2
        MAIN_TRANSPOSE_12x4

        stp q0,  q1,  [x0]
        stp q2,  q3,  [x0, #(32 * 1)]
        stp q4,  q5,  [x0, #(32 * 2)]
        stp q6,  q7,  [x0, #(32 * 3)]
        str q16, [x0, #(32 * 4)]
        add x0, x0, #(32 * 4 + 16)

        // st1 {v0.4s}, [x0], #16
        // st1 {v4.4s}, [x0], #16
        // st1 {v16.4s}, [x0], #16
//
        // st1 {v1.4s}, [x0], #16
        // st1 {v5.4s}, [x0], #16
        // st1 {v17.4s}, [x0], #16
//
        // st1 {v2.4s}, [x0], #16
        // st1 {v6.4s}, [x0], #16
        // st1 {v18.4s}, [x0], #16

        b LoopE12End

    LoopE12L2:
    cmp w5, #2
    blt LoopE12L1
        MAIN_TRANSPOSE_12x4
        stp q0,  q1,  [x0]
        stp q2,  q3,  [x0, #(32 * 1)]
        stp q4,  q5,  [x0, #(32 * 2)]
        add x0, x0, #(32 * 3)

        // st1 {v0.4s}, [x0], #16
        // st1 {v4.4s}, [x0], #16
        // st1 {v16.4s}, [x0], #16
//
        // st1 {v1.4s}, [x0], #16
        // st1 {v5.4s}, [x0], #16
        // st1 {v17.4s}, [x0], #16
        b LoopE12End

    LoopE12L1:
    cmp w5, #1
    blt LoopE12End
        MAIN_TRANSPOSE_12x4

        stp q0,  q1,  [x0]
        str q2,  [x0, #32]
        add x0, x0, #(32 + 16)

        // st1 {v0.4s}, [x0], #16
        // st1 {v4.4s}, [x0], #16
        // st1 {v16.4s}, [x0], #16
    LoopE12End:
b End

Right:

LoopE1:
    mov w9, w5
    mov x7, x1
    mov x8, x0
    cmp w5, #4
    blt LoopE1L3
    LoopE1L4:
        ld1 {v0.4s}, [x1], x4
        st1 {v0.s}[0], [x0], x11
        st1 {v0.s}[1], [x0], x11
        st1 {v0.s}[2], [x0], x11
        st1 {v0.s}[3], [x0], x11
        sub w5, w5, #4
        cmp w5, #4
        bge LoopE1L4

    LoopE1L3:
    cmp w5, #3
    blt LoopE1L2
        ld1 {v0.4s}, [x1], x4
        st1 {v0.s}[0], [x0], x11
        st1 {v0.s}[1], [x0], x11
        st1 {v0.s}[2], [x0], x11

        sub w5, w5, #3

    LoopE1L2:
    cmp w5, #2
    blt LoopE1L1
        ld1 {v0.2s}, [x1], x4
        st1 {v0.s}[0], [x0], x11
        st1 {v0.s}[1], [x0], x11
        sub w5, w5, #2

    LoopE1L1:
    cmp w5, #1
    blt LoopE1End
        ld1 {v0.s}[0], [x1], x4
        st1 {v0.s}[0], [x0], x11

    LoopE1End:

    subs w2, w2, #1
    add x0, x8, #4
    add x1, x7, x6
    mov w5, w9
    bne LoopE1

End:

mov x0, x13
mov x1, x14
subs w10, w10, #1
add x3, x3, #16
add x1, x1, #8

bne LoopNumber

ret

#endif
