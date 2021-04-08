//
//  MNNTranspose32Bit4x4.S
//  MNN
//
//  Created by MNN on 2020/09/27.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5
asm_function MNNTranspose32Bit4x4
//void MNNTranspose32Bit4x4(int32_t* dstO, const int32_t* srcO, int* dim)
//Auto: x0: dstO, x1:srcO, x2: dim

mov x4, #0
mov x5, #0
mov x6, #0
mov x7, #0
ldr w4, [x2, #0]
ldr w5, [x2, #4]
ldr w6, [x2, #8]
ldr w7, [x2, #12]

// x4, x5 -> wC4, hC4
lsr x4, x4, #2
lsr x5, x5, #2

// x6, x7 -> srcStride * sizeof(float), dstStride * sizeof(float)
lsl x6, x6, #2
lsl x7, x7, #2

// [x0, x1, x2, x3] => [x0, x6, x2, x3]
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
.endm

LoopY:
    mov x2, x4
    mov x8, x0
    mov x9, x1
    LoopX:
        ld1 {v0.4s}, [x1], x6
        ld1 {v1.4s}, [x1], x6
        ld1 {v2.4s}, [x1], x6
        ld1 {v3.4s}, [x1], x6

        transpose_4x4 v0, v1, v2, v3, v5, v6

        mov x12, x0

        st1 {v0.4s}, [x12], x7
        st1 {v6.4s}, [x12], x7
        st1 {v2.4s}, [x12], x7
        st1 {v3.4s}, [x12], x7

        add x0, x0, #16 // 4 * sizeof(float)

        subs x2, x2, #1
        bne LoopX


    lsl x12, x7, #2
    subs x5, x5, #1
    add x1, x9, #16 // 4 * sizeof(float)
    add x0, x8, x12
    bne LoopY

End:

ret

#endif
