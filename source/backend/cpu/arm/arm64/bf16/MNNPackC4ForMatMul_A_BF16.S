
//
//  NEON_MNNPackC4ForMatMul_A_BF16.S
//  MNN
//
//  Created by MNN on 2021/02/26.
//  Copyright © 2018-2021 Alibaba Group Holding Limited
//
#ifdef __aarch64__
#include "MNNAsmGlobal.h"

.macro transpose_4x4 x0, x1, x2, x3, x5, x6 // transpose 4x4 of sizeof(int16_t), only low half simd vector is valid.
    trn1 \x5\().4h,  \x0\().4h, \x1\().4h
    trn2 \x1\().4h,  \x0\().4h, \x1\().4h
    trn1 \x6\().4h,  \x2\().4h, \x3\().4h
    trn2 \x3\().4h,  \x2\().4h, \x3\().4h
    trn1 \x0\().2s,  \x5\().2s, \x6\().2s
    trn2 \x2\().2s,  \x5\().2s, \x6\().2s
    trn1 \x6\().2s,  \x1\().2s, \x3\().2s
    trn2 \x3\().2s,  \x1\().2s, \x3\().2s
    mov \x1\().8b, \x6\().8b
.endm

.text
.align 5
asm_function NEON_MNNPackC4ForMatMul_A_BF16
// treate float pointer as int16_t*
//void NEON_MNNPackC4ForMatMul_A_BF16(float* destOrigin, float const** sourceGroup, const int32_t* info, const int32_t* el)
//Auto: x0: dest, x1:sourceGroup, x2: info, x3:el
ldr w10, [x2, #0] // number
mov x4, #0
mov x11, #0
mov x6, #0
ldr w4, [x2, #4] // eReal
ldr w11, [x2, #8] // eDest
ldr w6, [x2, #12] // xOffset
// xOffset -> xOffset * 4 * sizeof(float)
// eReal -> eReal * 4 * sizeof(int16_t)
// eDest -> eDest * sizeof(int16_t)
mov x12, #2 // sizeof(int16_t).  kept as a const
mov x9, #8
mov x15, #16 // sizeof(float)
mul x4, x15, x4
mul x11, x12, x11

mul x6, x15, x6

LoopNumber:
mov x2, #0
mov x5, #0
mov x8, #0
mov x7, #0
ldr w5, [x3, #4] // l
ldr w8, [x3, #8] // eOffset
ldr w7, [x3, #12] // lOffset

mov x13, x0
mov x14, x1
ldr x1, [x1, #0]

// Compute dest ptr: x0 = x0 + eOffset * sizeof(int16_t) + lOffset * eDest * sizeof(int16_t)
mul x7, x11, x7
mul x8, x12, x8
add x0, x0, x7
add x0, x0, x8

ldr w2, [x3, #0] // e

Body:
cmp w2, #12 // original eDest
bne Right
    cmp w5, #4
    blt LoopEL3
    LoopL4:
        mov x2, x1
.macro MAIN_TRANSPOSE
        ld1 {v0.4s}, [x1], x6 // load size: 4 * sizeof(int16_t), jump one stride line as x6
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

        shrn v0.4h, v0.4s, #16
        shrn v3.4h, v3.4s, #16
        shrn v6.4h, v6.4s, #16
        shrn v17.4h, v17.4s, #16

        shrn v1.4h, v1.4s, #16
        shrn v4.4h, v4.4s, #16
        shrn v7.4h, v7.4s, #16
        shrn v18.4h, v18.4s, #16

        shrn v2.4h, v2.4s, #16
        shrn v5.4h, v5.4s, #16
        shrn v16.4h, v16.4s, #16
        shrn v19.4h, v19.4s, #16

        transpose_4x4 v0, v3, v6, v17, v23, v24
        transpose_4x4 v1, v4, v7, v18, v25, v26
        transpose_4x4 v2, v5, v16, v19, v27, v28
.endm
        MAIN_TRANSPOSE

        stp d0,  d1,  [x0]             // store size: 2 * 4 * sizeof(int16_t)
        stp d2,  d3,  [x0, #(16 * 1)]
        stp d4,  d5,  [x0, #(16 * 2)]
        stp d6,  d7,  [x0, #(16 * 3)]
        stp d16, d17, [x0, #(16 * 4)]
        stp d18, d19, [x0, #(16 * 5)]
        add x0, x0, #(16 * 6)

        add x1, x2, x4
        sub x5, x5, #4
        cmp w5, #4
        bge LoopL4

    LoopEL3:
    cmp w5, #3
    blt LoopEL2
        MAIN_TRANSPOSE

        stp d0,  d1,  [x0]              // store size: 2 * 4 * sizeof(int16_t)
        stp d2,  d3,  [x0, #(16 * 1)]
        stp d4,  d5,  [x0, #(16 * 2)]
        stp d6,  d7,  [x0, #(16 * 3)]
        str d16, [x0, #(16 * 4)]
        add x0, x0, #(16 * 4 + 8)

        b LoopEEnd

    LoopEL2:
    cmp w5, #2
    blt LoopEL1
        MAIN_TRANSPOSE
        stp d0,  d1,  [x0] // store size: 2 * 4 * sizeof(int16_t)
        stp d2,  d3,  [x0, #(16 * 1)]
        stp d4,  d5,  [x0, #(16 * 2)]
        add x0, x0, #(16 * 3)

        b LoopEEnd

    LoopEL1:
    cmp w5, #1
    blt LoopEEnd
        MAIN_TRANSPOSE
        stp d0, d1, [x0]
        str d2, [x0, #16]
        add x0, x0, #(16 + 8)

    LoopEEnd:

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
        shrn v0.4h, v0.4s, #16
        st1 {v0.h}[0], [x0], x11
        st1 {v0.h}[1], [x0], x11
        st1 {v0.h}[2], [x0], x11
        st1 {v0.h}[3], [x0], x11
        sub w5, w5, #4
        cmp w5, #4
        bge LoopE1L4

    LoopE1L3:
    cmp w5, #3
    blt LoopE1L2
        ld1 {v0.4s}, [x1], x4
        shrn v0.4h, v0.4s, #16
        st1 {v0.h}[0], [x0], x11
        st1 {v0.h}[1], [x0], x11
        st1 {v0.h}[2], [x0], x11

        sub w5, w5, #3

    LoopE1L2:
    cmp w5, #2
    blt LoopE1L1
        ld1 {v0.4s}, [x1], x4
        shrn v0.4h, v0.4s, #16
        st1 {v0.h}[0], [x0], x11
        st1 {v0.h}[1], [x0], x11
        sub w5, w5, #2

    LoopE1L1:
    cmp w5, #1
    blt LoopE1End
        ld1 {v0.s}[0], [x1], x4
        shrn v0.4h, v0.4s, #16
        st1 {v0.h}[0], [x0], x11

    LoopE1End:

    subs w2, w2, #1
    add x0, x8, x12 // !!!! caution : sizeof(int16_t)
    add x1, x7, x6
    mov w5, w9
    bne LoopE1

End:

mov x0, x13
mov x1, x14
subs w10, w10, #1

// x3 is (const int32_t* el), this array size of 4. as a result for next struct element,
// address added by 4 * sizeof(int32_t)
add x3, x3, #16

// x1 is (const int16_t** sourceGroup), even though data content is int16_t,
// the element in sourceGroup in 'int16_t*', as a result for next struct element,
// value added by sizeof(void*)
add x1, x1, #8
bne LoopNumber

ret

#endif
