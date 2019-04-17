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
movi v2.4s, #1
scvtf v3.4s, v2.4s

movi v5.4s, #24
neg v6.4s, v5.4s

Loop:

ld1 {v16.4s, v17.4s}, [x1], #32

fneg v18.4s, v16.4s
fneg v19.4s, v17.4s

fmul v16.4s, v18.4s, v0.s[1]
fmul v17.4s, v19.4s, v0.s[1]
fcvtzs v16.4s, v16.4s
fcvtzs v17.4s, v17.4s
scvtf v20.4s, v16.4s
scvtf v21.4s, v17.4s
smax v16.4s, v16.4s, v6.4s
smax v17.4s, v17.4s, v6.4s
smin v16.4s, v16.4s, v5.4s
smin v17.4s, v17.4s, v5.4s

//v18.4s, v19.4s: t
fmls v18.4s, v20.4s, v0.s[0]
fmls v19.4s, v21.4s, v0.s[0]

fmul v20.4s, v18.4s, v1.s[3]
dup v22.4s, v1.s[2]
fmul v21.4s, v19.4s, v1.s[3]

fadd v20.4s, v20.4s, v22.4s
fadd v21.4s, v21.4s, v22.4s

.macro MLA_TWO z0
fmul v20.4s, v20.4s, v18.4s
dup v22.4s, \z0
fmul v21.4s, v21.4s, v19.4s
fadd v20.4s, v20.4s, v22.4s
fadd v21.4s, v21.4s, v22.4s
.endm

MLA_TWO v1.s[1]
MLA_TWO v1.s[0]
MLA_TWO v0.s[3]
MLA_TWO v0.s[2]

//v20.4s, v21.4s is expRemain

//Compute expBasic
cmle v18.4s, v16.4s, #0
cmle v19.4s, v17.4s, #0

neg v22.4s, v16.4s
neg v23.4s, v17.4s

sshl v22.4s, v2.4s, v22.4s
sshl v23.4s, v2.4s, v23.4s

sshl v16.4s, v2.4s, v16.4s
sshl v17.4s, v2.4s, v17.4s

scvtf v22.4s, v22.4s
scvtf v23.4s, v23.4s
scvtf v16.4s, v16.4s
scvtf v17.4s, v17.4s
frecpe v22.4s, v22.4s
frecpe v23.4s, v23.4s

bit v16.16b, v22.16b, v18.16b
bit v17.16b, v23.16b, v19.16b

fmul v16.4s, v16.4s, v20.4s
fmul v17.4s, v17.4s, v21.4s

st1 {v16.4s, v17.4s}, [x0], #32


subs x3, x3, #1
bne Loop

ret

#endif
