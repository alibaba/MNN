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

//void MNNExpC8(float* dest, const float* source, const float* parameters, size_t countC8)
asm_function MNNExpC8

//x0: dest, x1:source, x2:parameters, x3:countC8

ld1 {v0.4s, v1.4s}, [x2]
movi v2.4s, #23
movi v3.4s, #87
scvtf v3.4s, v3.4s
fneg v4.4s, v3.4s

Loop:

ld1 {v16.4s, v17.4s}, [x1], #32

fmin v16.4s, v16.4s, v3.4s
fmin v17.4s, v17.4s, v3.4s
fmax v16.4s, v16.4s, v4.4s
fmax v17.4s, v17.4s, v4.4s

fneg v18.4s, v16.4s
fneg v19.4s, v17.4s

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

st1 {v20.4s, v21.4s}, [x0], #32

subs x3, x3, #1
bne Loop

ret

#endif

