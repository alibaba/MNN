//
//  MNNMatrixMax.S
//  MNN
//
//  Created by MNN on 2019/02/12.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNMatrixMax
//void MNNMatrixMax(float* C, const float* A, const float* B, size_t widthC4, size_t cStride, size_t aStride, size_t bStride, size_t height)

//Auto: x0: C, x1:A, x2:B, x3:widthC4
//x4:cStride, x5:aStride, x6:bStride, x7:height

mov x12, #4 //sizeof(float)
mul x4, x12, x4
mul x5, x12, x5
mul x6, x12, x6

LoopY:
mov x8, x0
mov x9, x1
mov x10, x2

mov x11, x3

L4:
cmp x11, #4
blt L1
sub x11, x11, #4
ld1 {v0.4s, v1.4s}, [x1], #32
ld1 {v2.4s, v3.4s}, [x2], #32

fmax v0.4s, v0.4s, v2.4s
ld1 {v16.4s, v17.4s}, [x1], #32
fmax v1.4s, v1.4s, v3.4s

cmp x11, #4
blt L4LoopEnd

L4Loop:
ld1 {v18.4s, v19.4s}, [x2], #32
st1 {v0.4s, v1.4s}, [x0], #32
fmax v16.4s, v16.4s, v18.4s
fmax v17.4s, v17.4s, v19.4s

ld1 {v0.4s, v1.4s}, [x1], #32
st1 {v16.4s, v17.4s}, [x0], #32
ld1 {v2.4s, v3.4s}, [x2], #32
fmax v0.4s, v0.4s, v2.4s
ld1 {v16.4s, v17.4s}, [x1], #32
fmax v1.4s, v1.4s, v3.4s

sub x11, x11, #4
cmp x11, #4
bge L4Loop

L4LoopEnd:
ld1 {v18.4s, v19.4s}, [x2], #32
st1 {v0.4s, v1.4s}, [x0], #32
fmax v16.4s, v16.4s, v18.4s
fmax v17.4s, v17.4s, v19.4s
st1 {v16.4s, v17.4s}, [x0], #32

L1:
cmp x11, #0
beq EndLine

L1Loop:
ld1 {v0.4s}, [x1], #16
ld1 {v1.4s}, [x2], #16
fmax v0.4s, v0.4s, v1.4s
st1 {v0.4s}, [x0], #16
subs x11, x11, #1
bne L1Loop

EndLine:
add x0, x8, x4
add x1, x9, x5
add x2, x10, x6

subs x7, x7, #1
bne LoopY

ret
#endif
