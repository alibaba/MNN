//
//  MNNExpC8.S
//  MNN
//
//  Created by MNN on 2019/01/18.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"
.text
.align 5

//void MNNExpC8(float* dest, const float* source, const float* offset, const float* parameters, size_t countC8)
asm_function MNNExpC8

//x0: dest, x1:source, x2: offset, x3:parameters, x4:countC8
ldr w5, [x2, #0]
ldr w6, [x2, #4]
ldr w7, [x2, #8]

ld1 {v0.4s, v1.4s}, [x3]
movi v2.4s, #23
movi v3.4s, #87
scvtf v3.4s, v3.4s
fneg v4.4s, v3.4s
dup v30.4s, w5
dup v31.4s, w6
dup v29.4s, w7

// Summer
movi v28.4s, #0

Loop:

ld1 {v16.4s, v17.4s}, [x1], #32
fmul v16.4s, v16.4s, v30.4s
fmul v17.4s, v17.4s, v30.4s
fadd v16.4s, v16.4s, v29.4s
fadd v17.4s, v17.4s, v29.4s
fmin v16.4s, v16.4s, v3.4s
fmin v17.4s, v17.4s, v3.4s
fmax v18.4s, v16.4s, v4.4s
fmax v19.4s, v17.4s, v4.4s

fmul v16.4s, v18.4s, v0.s[1]
fmul v17.4s, v19.4s, v0.s[1]
fcvtzs v16.4s, v16.4s
fcvtzs v17.4s, v17.4s
scvtf v20.4s, v16.4s
scvtf v21.4s, v17.4s

//v18.4s, v19.4s: t
fmls v18.4s, v20.4s, v0.s[0]
fmls v19.4s, v21.4s, v0.s[0]

.macro MLA_TWO z0 z1 z2 z3
dup \z1, \z0
fmla \z1, \z2, \z3
.endm

MLA_TWO v1.s[2], v20.4s, v18.4s, v1.s[3]
MLA_TWO v1.s[2], v21.4s, v19.4s, v1.s[3]
MLA_TWO v1.s[1], v22.4s, v18.4s, v20.4s
MLA_TWO v1.s[1], v23.4s, v19.4s, v21.4s
MLA_TWO v1.s[0], v20.4s, v18.4s, v22.4s
MLA_TWO v1.s[0], v21.4s, v19.4s, v23.4s
MLA_TWO v0.s[3], v22.4s, v18.4s, v20.4s
MLA_TWO v0.s[3], v23.4s, v19.4s, v21.4s
MLA_TWO v0.s[2], v20.4s, v18.4s, v22.4s
MLA_TWO v0.s[2], v21.4s, v19.4s, v23.4s

//v20.4s, v21.4s is expRemain

ushl v16.4s, v16.4s, v2.4s
ushl v17.4s, v17.4s, v2.4s
add v20.4s, v20.4s, v16.4s
add v21.4s, v21.4s, v17.4s

fadd v20.4s, v20.4s, v31.4s
fadd v21.4s, v21.4s, v31.4s

st1 {v20.4s, v21.4s}, [x0], #32
fadd v28.4s, v28.4s, v20.4s
fadd v28.4s, v28.4s, v21.4s

subs x4, x4, #1
bne Loop

// Bias
add x7, x2, #12
ld1 {v27.s}[0], [x7]
faddp v28.4s, v28.4s, v28.4s
faddp v28.2s, v28.2s, v28.2s
fadd v27.2s, v28.2s, v27.2s
st1 {v27.s}[0], [x7]

ret

#endif

