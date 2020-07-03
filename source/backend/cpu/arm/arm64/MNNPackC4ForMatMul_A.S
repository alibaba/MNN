//
//  MNNPackC4ForMatMul_A.S
//  MNN
//
//  Created by MNN on 2020/06/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5
asm_function MNNPackC4ForMatMul_A
//void MNNPackC4ForMatMul_A(float* dest, const float* source, size_t e, size_t l, size_t eReal)
//Auto: x0: dest, x1:source, x2: e, x3:l, x4: eReal
// eReal -> eReal * 4 * sizeof(float) - 192
mov x13, #4
mov x12, #16
mul x4, x12, x4
mul x8, x13, x2

sub x4, x4, #192

// Set x13 as l * 12 * sizeof(float)
mov x12, #48
mul x13, x3, x12

Body:
cmp x2, #12
blt Right
.macro transpose_4x4 x0, x1, x2, x3, x5, x6
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

LoopE12:
    mov x6, x0
    mov x7, x1
    mov x5, x3
    cmp x5, #4
    blt LoopEL3
    LoopL4:
.macro MAIN_TRANSPOSE
        ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x1], #64
        ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x1], #64

        ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x1], #64

        transpose_4x4 v0, v1, v2, v3, v21, v22
        transpose_4x4 v4, v5, v6, v7, v23, v24
        transpose_4x4 v16, v17, v18, v19, v25, v26
.endm
        MAIN_TRANSPOSE
        stp q0, q4, [x0], #32
        stp q16, q1, [x0], #32
        stp q5, q17, [x0], #32
        stp q2, q6, [x0], #32
        stp q18, q3, [x0], #32
        stp q7, q19, [x0], #32

        add x1, x1, x4
        sub x5, x5, #4
        cmp x5, #4
        bge LoopL4

    LoopEL3:
    cmp x5, #3
    blt LoopEL2
        MAIN_TRANSPOSE

        st1 {v0.4s}, [x0], #16
        st1 {v4.4s}, [x0], #16
        st1 {v16.4s}, [x0], #16

        st1 {v1.4s}, [x0], #16
        st1 {v5.4s}, [x0], #16
        st1 {v17.4s}, [x0], #16

        st1 {v2.4s}, [x0], #16
        st1 {v6.4s}, [x0], #16
        st1 {v18.4s}, [x0], #16

        sub x5, x5, #3

    LoopEL2:
    cmp x5, #2
    blt LoopEL1
        MAIN_TRANSPOSE
        st1 {v0.4s}, [x0], #16
        st1 {v4.4s}, [x0], #16
        st1 {v16.4s}, [x0], #16

        st1 {v1.4s}, [x0], #16
        st1 {v5.4s}, [x0], #16
        st1 {v17.4s}, [x0], #16
        sub x5, x5, #2

    LoopEL1:
    cmp x5, #1
    blt LoopEEnd
        MAIN_TRANSPOSE
        st1 {v0.4s}, [x0], #16
        st1 {v4.4s}, [x0], #16
        st1 {v16.4s}, [x0], #16
    LoopEEnd:


    sub x2, x2, #12
    cmp x2, #12
    add x0, x6, x13
    add x1, x7, #192 // 12 * 4 * sizeof(float)
    bge LoopE12

cmp x2, #0
beq End

Right:
add x4, x4, #192

LoopE1:
    mov x6, x0
    mov x7, x1
    mov x5, x3
    cmp x5, #4
    blt LoopE1L3
    LoopE1L4:
        ld1 {v0.4s}, [x1], x4
        st1 {v0.s}[0], [x0], x8
        st1 {v0.s}[1], [x0], x8
        st1 {v0.s}[2], [x0], x8
        st1 {v0.s}[3], [x0], x8
        sub x5, x5, #4
        cmp x5, #4
        bge LoopE1L4

    LoopE1L3:
    cmp x5, #3
    blt LoopE1L2
        ld1 {v0.4s}, [x1], x4
        st1 {v0.s}[0], [x0], x8
        st1 {v0.s}[1], [x0], x8
        st1 {v0.s}[2], [x0], x8
        sub x5, x5, #3

    LoopE1L2:
    cmp x5, #2
    blt LoopE1L1
        ld1 {v0.d}[0], [x1], x4
        st1 {v0.s}[0], [x0], x8
        st1 {v0.s}[1], [x0], x8
        sub x5, x5, #2

    LoopE1L1:
    cmp x5, #1
    blt LoopE1End
        ld1 {v0.s}[0], [x1], x4
        st1 {v0.s}[0], [x0], x8

    LoopE1End:

    subs x2, x2, #1
    add x0, x6, #4
    add x1, x7, #16 // 4 * sizeof(float)
    bne LoopE1

End:



ret

#endif
