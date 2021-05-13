//
//  MNNExpFP16.S
//  MNN
//
//  Created by MNN on 2019/01/18.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"
.text
.align 5

//void MNNExpFP16(FLOAT16* dest, const FLOAT16* source, const FLOAT16* parameters, size_t block)
asm_function MNNExpFP16

//x0: dest, x1:source, x2:parameters, x3:block

ld1 {v0.8h}, [x2]
movi v2.8h, #10
movi v3.8h, #11
scvtf v3.8h, v3.8h
fneg v4.8h, v3.8h

Loop:

ld1 {v16.8h, v17.8h}, [x1], #32

fmin v16.8h, v16.8h, v3.8h
fmin v17.8h, v17.8h, v3.8h
fmax v16.8h, v16.8h, v4.8h
fmax v17.8h, v17.8h, v4.8h

fneg v18.8h, v16.8h
fneg v19.8h, v17.8h

fmul v16.8h, v18.8h, v0.h[1]
fmul v17.8h, v19.8h, v0.h[1]
fcvtzs v16.8h, v16.8h
fcvtzs v17.8h, v17.8h
scvtf v20.8h, v16.8h
scvtf v21.8h, v17.8h

//v18.8h, v19.8h: t
fmls v18.8h, v20.8h, v0.h[0]
fmls v19.8h, v21.8h, v0.h[0]

.macro MLA_TWO z0 z1 z2 z3
dup \z1, \z0
fmla \z1, \z2, \z3
.endm

MLA_TWO v0.h[6], v20.8h, v18.8h, v0.h[7]
MLA_TWO v0.h[6], v21.8h, v19.8h, v0.h[7]
MLA_TWO v0.h[5], v22.8h, v18.8h, v20.8h
MLA_TWO v0.h[5], v23.8h, v19.8h, v21.8h
MLA_TWO v0.h[4], v20.8h, v18.8h, v22.8h
MLA_TWO v0.h[4], v21.8h, v19.8h, v23.8h
MLA_TWO v0.h[3], v22.8h, v18.8h, v20.8h
MLA_TWO v0.h[3], v23.8h, v19.8h, v21.8h
MLA_TWO v0.h[2], v20.8h, v18.8h, v22.8h
MLA_TWO v0.h[2], v21.8h, v19.8h, v23.8h

//v20.8h, v21.8h is expRemain

ushl v16.8h, v16.8h, v2.8h
ushl v17.8h, v17.8h, v2.8h
add v20.8h, v20.8h, v16.8h
add v21.8h, v21.8h, v17.8h

st1 {v20.8h, v21.8h}, [x0], #32

subs x3, x3, #1
bne Loop

ret

#endif

