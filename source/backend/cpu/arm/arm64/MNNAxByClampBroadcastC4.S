//
//  MNNMatrixSub.S
//  MNN
//
//  Created by MNN on 2020/06/20.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNAxByClampBroadcastC4
//void MNNAxByClampBroadcastC4(float* C, const float* A, const float* B, size_t width, size_t cStride, size_t aStride, size_t height, const float* parameters)
//Auto: x0: C, x1:A, x2:B, x3:width
//x4:cStride, x5:aStride, x6:height, x7:parameters
ld1 {v7.4s}, [x7]
dup v30.4s, v7.s[2]
dup v31.4s, v7.s[3]
mov x12, #4 //sizeof(float)
mul x4, x12, x4
mul x5, x12, x5

LoopY:
mov x8, x0
mov x9, x1
ld1 {v6.4s}, [x2], #16

mov x11, x3

L8:
cmp x11, #8
blt L1

sub x11, x11, #8
cmp x11, #8

ldp q16, q17, [x1], #32
ldp q18, q19, [x1], #32
fmla v16.4s, v6.4s, v7.s[1]
fmla v17.4s, v6.4s, v7.s[1]
ldp q20, q21, [x1], #32
fmla v18.4s, v6.4s, v7.s[1]
fmla v19.4s, v6.4s, v7.s[1]
ldp q22, q23, [x1], #32

fmla v20.4s, v6.4s, v7.s[1]
fmla v21.4s, v6.4s, v7.s[1]
fmla v22.4s, v6.4s, v7.s[1]
fmla v23.4s, v6.4s, v7.s[1]

blt L8ComputeEnd

L8Loop:

fmax v16.4s, v16.4s, v30.4s
fmax v17.4s, v17.4s, v30.4s
fmax v18.4s, v18.4s, v30.4s
fmax v19.4s, v19.4s, v30.4s
fmax v20.4s, v20.4s, v30.4s
fmax v21.4s, v21.4s, v30.4s
fmax v22.4s, v22.4s, v30.4s
fmax v23.4s, v23.4s, v30.4s

fmin v16.4s, v16.4s, v31.4s
fmin v17.4s, v17.4s, v31.4s
fmin v18.4s, v18.4s, v31.4s
fmin v19.4s, v19.4s, v31.4s
stp q16, q17, [x0], #32
fmin v20.4s, v20.4s, v31.4s
fmin v21.4s, v21.4s, v31.4s
stp q18, q19, [x0], #32
fmin v22.4s, v22.4s, v31.4s
ldp q16, q17, [x1], #32
fmin v23.4s, v23.4s, v31.4s
ldp q18, q19, [x1], #32
fmla v16.4s, v6.4s, v7.s[1]
fmla v17.4s, v6.4s, v7.s[1]

stp q20, q21, [x0], #32
fmla v18.4s, v6.4s, v7.s[1]
stp q22, q23, [x0], #32
fmla v19.4s, v6.4s, v7.s[1]
ldp q20, q21, [x1], #32
ldp q22, q23, [x1], #32

fmla v20.4s, v6.4s, v7.s[1]
fmla v21.4s, v6.4s, v7.s[1]
fmla v22.4s, v6.4s, v7.s[1]
fmla v23.4s, v6.4s, v7.s[1]


sub x11, x11, #8
cmp x11, #8
bge L8Loop

L8ComputeEnd:

fmax v16.4s, v16.4s, v30.4s
fmax v17.4s, v17.4s, v30.4s
fmax v18.4s, v18.4s, v30.4s
fmax v19.4s, v19.4s, v30.4s
fmax v20.4s, v20.4s, v30.4s
fmax v21.4s, v21.4s, v30.4s
fmax v22.4s, v22.4s, v30.4s
fmax v23.4s, v23.4s, v30.4s

fmin v16.4s, v16.4s, v31.4s
fmin v17.4s, v17.4s, v31.4s
fmin v18.4s, v18.4s, v31.4s
fmin v19.4s, v19.4s, v31.4s
fmin v20.4s, v20.4s, v31.4s
fmin v21.4s, v21.4s, v31.4s
fmin v22.4s, v22.4s, v31.4s
fmin v23.4s, v23.4s, v31.4s
stp q16, q17, [x0], #32
stp q18, q19, [x0], #32

stp q20, q21, [x0], #32
stp q22, q23, [x0], #32

L1:
cmp x11, #0
beq EndLine

L1Loop:
ld1 {v0.4s}, [x1], #16
fmla v0.4s, v6.4s, v7.s[1]
fmax v0.4s, v0.4s, v30.4s
fmin v0.4s, v0.4s, v31.4s
st1 {v0.4s}, [x0], #16
subs x11, x11, #1
bne L1Loop

EndLine:
add x0, x8, x4
add x1, x9, x5

subs x6, x6, #1
bne LoopY

ret

#endif
