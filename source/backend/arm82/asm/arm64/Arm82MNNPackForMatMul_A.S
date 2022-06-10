//
//  Arm82MNNPackForMatMul_A.S
//  MNN
//
//  Created by MNN on 2020/06/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef __aarch64__

#include "MNNAsmGlobal.h"
// (l/8,e,8) -> (e/12,l,12)
// trans 8x12 == trans 8x8 + trans 4x4 + trans 4x4

.text
.align 5
asm_function Arm82MNNPackForMatMul_A
//void Arm82MNNPackForMatMul_A(FLOAT16* destOrigin, FLOAT16 const** sourceGroup, const int32_t* info, const int32_t* el)
//Auto: x0: dest, x1:sourceGroup, x2: info, x3:el
stp d14, d15, [sp, #-64]!
stp d12, d13, [sp, #16]
stp d10, d11, [sp, #32]
stp d8,  d9,  [sp, #48]

ldr w10, [x2, #0] // number
mov x4, #0
mov x11, #0
mov x6, #0
ldr w4, [x2, #4] // eReal
ldr w11, [x2, #8] // eDest
ldr w6, [x2, #12] // xOffset
// xOffset -> xOffset * 8 * sizeof(FLOAT16)
// eReal -> eReal * 8 * sizeof(FLOAT16)
// eDest -> eDest * sizeof(FLOAT16)
mov x9, #2   // sizeof(FLOAT16)
mov x12, #16 // 8 * sizeof(FLOAT16)
mul x4, x12, x4
mul x11, x9, x11
mul x6, x12, x6

LoopNumber:
mov x5, #0
mov x8, #0
mov x7, #0
ldr w5, [x3, #4] // l
ldr w8, [x3, #8] // eOffset
ldr w7, [x3, #12] // lOffset

mov x13, x0
mov x14, x1
ldr x1, [x1, #0]

// Compute dest ptr: x0 = x0 + eOffset * sizeof(FLOAT16) + lOffset * eDest * sizeof(FLOAT16)
mov x9, #2   // sizeof(FLOAT16)
mul x7, x11, x7
mul x8, x9, x8
add x0, x0, x7
add x0, x0, x8
mov x2, #0
ldr w2, [x3, #0] // e

Body:
    cmp w2, #12 // eP
    blt E8
    cmp x5, #8
    blt Body_LoopLExtra
    Body_LoopL8:
        mov x2, x1

.macro TRANSPOSE_8x8 d0, d1, d2, d3, d4, d5, d6, d7, t0, t1, t2, t3, t4, t5, t6, t7
    zip1 \t0\().8h, v0.8h, v1.8h
    zip2 \t1\().8h, v0.8h, v1.8h
    zip1 \t2\().8h, v2.8h, v3.8h
    zip2 \t3\().8h, v2.8h, v3.8h
    zip1 \t4\().8h, v4.8h, v5.8h
    zip2 \t5\().8h, v4.8h, v5.8h
    zip1 \t6\().8h, v6.8h, v7.8h
    zip2 \t7\().8h, v6.8h, v7.8h
    zip1 v0.4s, \t0\().4s, \t2\().4s
    zip2 v1.4s, \t0\().4s, \t2\().4s
    zip1 v2.4s, \t1\().4s, \t3\().4s
    zip2 v3.4s, \t1\().4s, \t3\().4s
    zip1 v4.4s, \t4\().4s, \t6\().4s
    zip2 v5.4s, \t4\().4s, \t6\().4s
    zip1 v6.4s, \t5\().4s, \t7\().4s
    zip2 v7.4s, \t5\().4s, \t7\().4s
    zip1 \d0\().2d, v0.2d, v4.2d
    zip2 \d1\().2d, v0.2d, v4.2d
    zip1 \d2\().2d, v1.2d, v5.2d
    zip2 \d3\().2d, v1.2d, v5.2d
    zip1 \d4\().2d, v2.2d, v6.2d
    zip2 \d5\().2d, v2.2d, v6.2d
    zip1 \d6\().2d, v3.2d, v7.2d
    zip2 \d7\().2d, v3.2d, v7.2d
.endm

.macro TRANSPOSE_8x4 s0, s1, s2, s3, d0, d1, d2, d3, t0, t1, t2, t3
    zip1 \t0\().8h, \s0\().8h, \s1\().8h
    zip2 \t1\().8h, \s0\().8h, \s1\().8h
    zip1 \t2\().8h, \s2\().8h, \s3\().8h
    zip2 \t3\().8h, \s2\().8h, \s3\().8h
    zip1 \d0\().4s, \t0\().4s, \t2\().4s
    zip2 \d1\().4s, \t0\().4s, \t2\().4s
    zip1 \d2\().4s, \t1\().4s, \t3\().4s
    zip2 \d3\().4s, \t1\().4s, \t3\().4s
.endm

.macro MAIN_TRANSPOSE_E12
// src:[v0-v11]
    ld1 {v0.8h}, [x1], x6
    ld1 {v1.8h}, [x1], x6
    ld1 {v2.8h}, [x1], x6
    ld1 {v3.8h}, [x1], x6
    ld1 {v4.8h}, [x1], x6
    ld1 {v5.8h}, [x1], x6
    ld1 {v6.8h}, [x1], x6
    ld1 {v7.8h}, [x1], x6
    ld1 {v8.8h}, [x1], x6
    ld1 {v9.8h}, [x1], x6
    ld1 {v10.8h}, [x1], x6
    ld1 {v11.8h}, [x1], x6
// [v0, v1, v2, v3, v4, v5, v6, v7] => [v20, v12, v23, v13, v26, v14, v29, v15]
// tmp: [21, 22, 24, 25, 27, 28, 30, 31]
    TRANSPOSE_8x8 v20, v12, v23, v13, v26, v14, v29, v15, v21, v22, v24, v25, v27, v28, v30, v31
// [v8, v9, v10, v11] => [v16, v17, v18, v19]
// tmp can be used: [0, 1, 2, 3, 4, 5, 6, 7, 21, 22, 24, 25, 27, 28, 30, 31]
    TRANSPOSE_8x4 v8, v9, v10, v11, v16, v17, v18, v19, v0, v1, v2, v3
// merge: [(v12, v16), (v13, v17), (v14, v18), (v15, v19)] => [(v21, v22), (v24, v25), (v27, v28), (v30, v31)]
    trn1 v21.2d, v16.2d, v12.2d
    trn2 v22.2d, v12.2d, v16.2d
    trn1 v24.2d, v17.2d, v13.2d
    trn2 v25.2d, v13.2d, v17.2d
    trn1 v27.2d, v18.2d, v14.2d
    trn2 v28.2d, v14.2d, v18.2d
    trn1 v30.2d, v19.2d, v15.2d
    trn2 v31.2d, v15.2d, v19.2d
// dst:[v20-v31]
.endm
        MAIN_TRANSPOSE_E12
        st1 {v20.8h, v21.8h, v22.8h, v23.8h}, [x0], #64
        st1 {v24.8h, v25.8h, v26.8h, v27.8h}, [x0], #64
        st1 {v28.8h, v29.8h, v30.8h, v31.8h}, [x0], #64

        add x1, x2, x4
        sub x5, x5, #8
        cmp x5, #8
        bge Body_LoopL8

    cbz x5, Body_LoopLEnd
    Body_LoopLExtra:
        MAIN_TRANSPOSE_E12
        cmp x5, #7 // if x5 < 7
        blt Body_LoopLEx6 // jump to Body_LoopLEx6
    Body_LoopLEx7:
        st1 {v20.8h, v21.8h, v22.8h, v23.8h}, [x0], #64
        st1 {v24.8h, v25.8h, v26.8h, v27.8h}, [x0], #64
        st1 {v28.8h, v29.8h}, [x0], #32
        st1 {v30.4h}, [x0], #8
        b Body_LoopLEnd
    Body_LoopLEx6:
        cmp x5, #6 // if x5 < 6
        blt Body_LoopLEx5 // jump to Body_LoopLEx5
        st1 {v20.8h, v21.8h, v22.8h, v23.8h}, [x0], #64
        st1 {v24.8h, v25.8h, v26.8h, v27.8h}, [x0], #64
        st1 {v28.8h}, [x0], #16
        b Body_LoopLEnd
    Body_LoopLEx5:
        cmp x5, #5 // if x5 < 5
        blt Body_LoopLEx4 // jump to Body_LoopLEx4
        st1 {v20.8h, v21.8h, v22.8h, v23.8h}, [x0], #64
        st1 {v24.8h, v25.8h, v26.8h}, [x0], #48
        st1 {v27.4h}, [x0], #8
        b Body_LoopLEnd
    Body_LoopLEx4:
        cmp x5, #4 // if x5 < 4
        blt Body_LoopLEx3 // jump to Body_LoopLEx3
        st1 {v20.8h, v21.8h, v22.8h, v23.8h}, [x0], #64
        st1 {v24.8h, v25.8h}, [x0], #32
        b Body_LoopLEnd
    Body_LoopLEx3:
        cmp x5, #3 // if x5 < 3
        blt Body_LoopLEx2 // jump to Body_LoopLEx2
        st1 {v20.8h, v21.8h, v22.8h, v23.8h}, [x0], #64
        st1 {v24.4h}, [x0], #8
        b Body_LoopLEnd
    Body_LoopLEx2:
        cmp x5, #2 // if x5 < 2
        blt Body_LoopLEx1 // jump to Body_LoopLEx1
        st1 {v20.8h, v21.8h, v22.8h}, [x0], #48
        b Body_LoopLEnd
    Body_LoopLEx1:
        cmp x5, #1 // if x5 < 1
        blt Body_LoopLEnd
        st1 {v20.8h}, [x0], #16
        st1 {v21.4h}, [x0], #8
    Body_LoopLEnd:
        b End

E8:
    cmp w2, #8
    blt E4

    mov x9, x5
    mov x7, x1
    mov x8, x0
    cmp x5, #8
    blt E8_LoopLExtra
    E8_LoopL8:
        mov x12, x1
    .macro MAIN_TRANSPOSE_E8
    // src:[v0-v7]
        ld1 {v0.8h}, [x1], x6
        ld1 {v1.8h}, [x1], x6
        ld1 {v2.8h}, [x1], x6
        ld1 {v3.8h}, [x1], x6
        ld1 {v4.8h}, [x1], x6
        ld1 {v5.8h}, [x1], x6
        ld1 {v6.8h}, [x1], x6
        ld1 {v7.8h}, [x1], x6
        TRANSPOSE_8x8 v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23
    .endm

        MAIN_TRANSPOSE_E8
        st1 {v8.8h}, [x0], x11
        st1 {v9.8h}, [x0], x11
        st1 {v10.8h}, [x0], x11
        st1 {v11.8h}, [x0], x11
        st1 {v12.8h}, [x0], x11
        st1 {v13.8h}, [x0], x11
        st1 {v14.8h}, [x0], x11
        st1 {v15.8h}, [x0], x11

        add x1, x12, x4
        sub x5, x5, #8
        cmp x5, #8
    bge E8_LoopL8

    cbz x5, E8_LoopLEnd
    E8_LoopLExtra:
        MAIN_TRANSPOSE_E8
        cmp x5, #7 // if x5 < 7
        blt E8_LoopLEx6 // jump to E8_LoopLEx6
    E8_LoopLEx7:
        st1 {v8.8h}, [x0], x11
        st1 {v9.8h}, [x0], x11
        st1 {v10.8h}, [x0], x11
        st1 {v11.8h}, [x0], x11
        st1 {v12.8h}, [x0], x11
        st1 {v13.8h}, [x0], x11
        st1 {v14.8h}, [x0], x11
        b E8_LoopLEnd
    E8_LoopLEx6:
        cmp x5, #6 // if x5 < 6
        blt E8_LoopLEx5 // jump to E8_LoopLEx5
        st1 {v8.8h}, [x0], x11
        st1 {v9.8h}, [x0], x11
        st1 {v10.8h}, [x0], x11
        st1 {v11.8h}, [x0], x11
        st1 {v12.8h}, [x0], x11
        st1 {v13.8h}, [x0], x11
        b E8_LoopLEnd
    E8_LoopLEx5:
        cmp x5, #5 // if x5 < 5
        blt E8_LoopLEx4 // jump to E8_LoopLEx4
        st1 {v8.8h}, [x0], x11
        st1 {v9.8h}, [x0], x11
        st1 {v10.8h}, [x0], x11
        st1 {v11.8h}, [x0], x11
        st1 {v12.8h}, [x0], x11
        b E8_LoopLEnd
    E8_LoopLEx4:
        cmp x5, #4 // if x5 < 4
        blt E8_LoopLEx3 // jump to E8_LoopLEx3
        st1 {v8.8h}, [x0], x11
        st1 {v9.8h}, [x0], x11
        st1 {v10.8h}, [x0], x11
        st1 {v11.8h}, [x0], x11
        b E8_LoopLEnd
    E8_LoopLEx3:
        cmp x5, #3 // if x5 < 3
        blt E8_LoopLEx2 // jump to E8_LoopLEx2
        st1 {v8.8h}, [x0], x11
        st1 {v9.8h}, [x0], x11
        st1 {v10.8h}, [x0], x11
        b E8_LoopLEnd
    E8_LoopLEx2:
        cmp x5, #2 // if x5 < 2
        blt E8_LoopLEx1 // jump to E8_LoopLEx1
        st1 {v8.8h}, [x0], x11
        st1 {v9.8h}, [x0], x11
        b E8_LoopLEnd
    E8_LoopLEx1:
        cmp x5, #1 // if x5 < 1
        blt E8_LoopLEnd
        st1 {v8.8h}, [x0], x11
    E8_LoopLEnd:
        sub w2, w2, #8
        add x0, x8, #16 // 8 * sizeof(FLOAT16)
        add x1, x7, x6, LSL #3
        mov w5, w9
        cbz w2, End

E4:
    cmp w2, #4
    blt E1

    mov x9, x5
    mov x7, x1
    mov x8, x0
    cmp x5, #8
    blt E4_LoopLExtra
    E4_LoopL8:
        mov x12, x1
    .macro MAIN_TRANSPOSE_E4
    // src:[v0-v7]
        ld1 {v0.8h}, [x1], x6
        ld1 {v1.8h}, [x1], x6
        ld1 {v2.8h}, [x1], x6
        ld1 {v3.8h}, [x1], x6
        TRANSPOSE_8x4 v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11
    .endm

        MAIN_TRANSPOSE_E4
        st1 {v4.d}[0], [x0], x11
        st1 {v4.d}[1], [x0], x11
        st1 {v5.d}[0], [x0], x11
        st1 {v5.d}[1], [x0], x11
        st1 {v6.d}[0], [x0], x11
        st1 {v6.d}[1], [x0], x11
        st1 {v7.d}[0], [x0], x11
        st1 {v7.d}[1], [x0], x11

        add x1, x12, x4
        sub x5, x5, #8
        cmp x5, #8
    bge E4_LoopL8

    cbz x5, E4_LoopLEnd
    E4_LoopLExtra:
        MAIN_TRANSPOSE_E4
        cmp x5, #7 // if x5 < 7
        blt E4_LoopLEx6 // jump to E4_LoopLEx6
    E4_LoopLEx7:
        st1 {v4.d}[0], [x0], x11
        st1 {v4.d}[1], [x0], x11
        st1 {v5.d}[0], [x0], x11
        st1 {v5.d}[1], [x0], x11
        st1 {v6.d}[0], [x0], x11
        st1 {v6.d}[1], [x0], x11
        st1 {v7.d}[0], [x0], x11
        b E4_LoopLEnd
    E4_LoopLEx6:
        cmp x5, #6 // if x5 < 6
        blt E4_LoopLEx5 // jump to E4_LoopLEx5
        st1 {v4.d}[0], [x0], x11
        st1 {v4.d}[1], [x0], x11
        st1 {v5.d}[0], [x0], x11
        st1 {v5.d}[1], [x0], x11
        st1 {v6.d}[0], [x0], x11
        st1 {v6.d}[1], [x0], x11
        b E4_LoopLEnd
    E4_LoopLEx5:
        cmp x5, #5 // if x5 < 5
        blt E4_LoopLEx4 // jump to E4_LoopLEx4
        st1 {v4.d}[0], [x0], x11
        st1 {v4.d}[1], [x0], x11
        st1 {v5.d}[0], [x0], x11
        st1 {v5.d}[1], [x0], x11
        st1 {v6.d}[0], [x0], x11
        b E4_LoopLEnd
    E4_LoopLEx4:
        cmp x5, #4 // if x5 < 4
        blt E4_LoopLEx3 // jump to E4_LoopLEx3
        st1 {v4.d}[0], [x0], x11
        st1 {v4.d}[1], [x0], x11
        st1 {v5.d}[0], [x0], x11
        st1 {v5.d}[1], [x0], x11
        b E4_LoopLEnd
    E4_LoopLEx3:
        cmp x5, #3 // if x5 < 3
        blt E4_LoopLEx2 // jump to E4_LoopLEx2
        st1 {v4.d}[0], [x0], x11
        st1 {v4.d}[1], [x0], x11
        st1 {v5.d}[0], [x0], x11
        b E4_LoopLEnd
    E4_LoopLEx2:
        cmp x5, #2 // if x5 < 2
        blt E4_LoopLEx1 // jump to E4_LoopLEx1
        st1 {v4.d}[0], [x0], x11
        st1 {v4.d}[1], [x0], x11
        b E4_LoopLEnd
    E4_LoopLEx1:
        cmp x5, #1 // if x5 < 1
        blt E4_LoopLEnd
        st1 {v4.d}[0], [x0], x11
    E4_LoopLEnd:
        sub w2, w2, #4
        add x0, x8, #8 // 4 * sizeof(FLOAT16)
        add x1, x7, x6, LSL #2
        mov w5, w9
        cbz w2, End

E1:
LoopE1:
    mov x9, x5
    mov x7, x1
    mov x8, x0
    cmp x5, #8
    blt E1_LoopLEx7

    E1_LoopL8:
        ld1 {v0.8h}, [x1], x4
        st1 {v0.h}[0], [x0], x11
        st1 {v0.h}[1], [x0], x11
        st1 {v0.h}[2], [x0], x11
        st1 {v0.h}[3], [x0], x11
        st1 {v0.h}[4], [x0], x11
        st1 {v0.h}[5], [x0], x11
        st1 {v0.h}[6], [x0], x11
        st1 {v0.h}[7], [x0], x11
        sub x5, x5, #8
        cmp x5, #8
        bge E1_LoopL8

    E1_LoopLEx7:
        cmp x5, #7 // if x5 < 7
        blt E1_LoopLEx6 // jump to E1_LoopLEx6
        ld1 {v0.8h}, [x1], x4
        st1 {v0.h}[0], [x0], x11
        st1 {v0.h}[1], [x0], x11
        st1 {v0.h}[2], [x0], x11
        st1 {v0.h}[3], [x0], x11
        st1 {v0.h}[4], [x0], x11
        st1 {v0.h}[5], [x0], x11
        st1 {v0.h}[6], [x0], x11
        b E1_LoopLEnd
    E1_LoopLEx6:
        cmp x5, #6 // if x5 < 6
        blt E1_LoopLEx5 // jump to E1_LoopLEx5
        ld1 {v0.8h}, [x1], x4
        st1 {v0.h}[0], [x0], x11
        st1 {v0.h}[1], [x0], x11
        st1 {v0.h}[2], [x0], x11
        st1 {v0.h}[3], [x0], x11
        st1 {v0.h}[4], [x0], x11
        st1 {v0.h}[5], [x0], x11
        b E1_LoopLEnd
    E1_LoopLEx5:
        cmp x5, #5 // if x5 < 5
        blt E1_LoopLEx4 // jump to E1_LoopLEx4
        ld1 {v0.8h}, [x1], x4
        st1 {v0.h}[0], [x0], x11
        st1 {v0.h}[1], [x0], x11
        st1 {v0.h}[2], [x0], x11
        st1 {v0.h}[3], [x0], x11
        st1 {v0.h}[4], [x0], x11
        b E1_LoopLEnd
    E1_LoopLEx4:
        cmp x5, #4 // if x5 < 4
        blt E1_LoopLEx3 // jump to E1_LoopLEx3
        ld1 {v0.d}[0], [x1], x4
        st1 {v0.h}[0], [x0], x11
        st1 {v0.h}[1], [x0], x11
        st1 {v0.h}[2], [x0], x11
        st1 {v0.h}[3], [x0], x11
        b E1_LoopLEnd
    E1_LoopLEx3:
        cmp x5, #3 // if x5 < 3
        blt E1_LoopLEx2 // jump to E1_LoopLEx2
        ld1 {v0.d}[0], [x1], x4
        st1 {v0.h}[0], [x0], x11
        st1 {v0.h}[1], [x0], x11
        st1 {v0.h}[2], [x0], x11
        b E1_LoopLEnd
    E1_LoopLEx2:
        cmp x5, #2 // if x5 < 2
        blt E1_LoopLEx1 // jump to E1_LoopLEx1
        ld1 {v0.s}[0], [x1], x4
        st1 {v0.h}[0], [x0], x11
        st1 {v0.h}[1], [x0], x11
        b E1_LoopLEnd
    E1_LoopLEx1:
        cmp x5, #1 // if x5 < 1
        blt E1_LoopLEnd
        ld1 {v0.h}[0], [x1], x4
        st1 {v0.h}[0], [x0], x11
    E1_LoopLEnd:
        subs w2, w2, #1
        add x0, x8, #2 // sizeof(FLOAT16)
        add x1, x7, x6
        mov w5, w9
        bne LoopE1

End:

mov x0, x13
mov x1, x14
subs w10, w10, #1
add x3, x3, #16 // 4 * sizeof(int32_t)
add x1, x1, #8  // sizeof(FLOAT16*)

bne LoopNumber

ldp d8,  d9,  [sp, #48]
ldp d10, d11, [sp, #32]
ldp d12, d13, [sp, #16]
ldp d14, d15, [sp], #64

ret

#endif
