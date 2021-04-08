//
//  MNNPackC8FP16.S
//  MNN
//
//  Created by MNN on 2020/6/30.
//  Copyright © 2020 Alibaba. All rights reserved.
//
#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5
asm_function MNNPackC8FP16
//void MNNPackC8FP16(FLOAT16* dest, const FLOAT16* source, size_t area, size_t depth);
// depth, area ->  depthC8, area, 8
// Auto: x0:dest, x1:source, x2: area, x3: depth
// x4: areaC8, x5:depthC8, x6: sourceStride, x7: destStride

lsr x4, x2, #3
lsr x5, x3, #3
mov x12, #2  // sizeof(FLOAT16)
mov x13, #16 // 8 * sizeof(FLOAT16)
mul x6, x12, x2
mul x7, x13, x2
mov x12, #32
mul x15, x12, x2

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

LoopH:
mov x8, x0
mov x9, x1
mov x12, x4

LoopL:
mov x10, x9
ld1 {v16.4s, v17.4s}, [x9], x6
ld1 {v18.4s, v19.4s}, [x9], x6
ld1 {v20.4s, v21.4s}, [x9], x6
ld1 {v22.4s, v23.4s}, [x9], x6

ld1 {v24.4s, v25.4s}, [x9], x6
ld1 {v26.4s, v27.4s}, [x9], x6
ld1 {v28.4s, v29.4s}, [x9], x6
ld1 {v30.4s, v31.4s}, [x9], x6

transpose_4x4 v16, v18, v20, v22, v0, v1
transpose_4x4 v17, v19, v21, v23, v2, v3
transpose_4x4 v24, v26, v28, v30, v4, v5
transpose_4x4 v25, v27, v29, v31, v6, v7

stp q16, q24, [x8], #32
stp q18, q26, [x8], #32
stp q20, q28, [x8], #32
stp q22, q30, [x8], #32

stp q17, q25, [x8], #32
stp q19, q27, [x8], #32
stp q21, q29, [x8], #32
stp q23, q31, [x8], #32

add x9, x10, #32

subs x12, x12, #1
bne LoopL


subs x5, x5, #1
add x0, x0, x7
add x1, x1, x15
bne LoopH


ret

#endif
