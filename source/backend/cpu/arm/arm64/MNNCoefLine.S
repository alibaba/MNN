//
//  MNNCoefLine.S
//  MNN
//
//  Created by MNN on 2019/02/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNCoefLine
//void MNNCoefLine(float* dst, const float* prepare, float* t, size_t number)
//Auto: x0:dst, x1:prepare, x2:t, x3:number

ld1 {v31.s}[0], [x2]
dup v31.4s, v31.s[0]

cmp x3, #16
blt MNNCoefLineL1

MNNCoefLineL16Loop:
ld4 {v0.4s, v1.4s, v2.4s, v3.4s}, [x1], #64

fmla v1.4s, v0.4s, v31.4s

ld4 {v4.4s, v5.4s, v6.4s, v7.4s}, [x1], #64
fmla v2.4s, v1.4s, v31.4s
fmla v5.4s, v4.4s, v31.4s
ld4 {v16.4s, v17.4s, v18.4s, v19.4s}, [x1], #64
fmla v3.4s, v2.4s, v31.4s
fmla v17.4s, v16.4s, v31.4s

fmla v6.4s, v5.4s, v31.4s
ld4 {v20.4s, v21.4s, v22.4s, v23.4s}, [x1], #64
fmla v7.4s, v6.4s, v31.4s
fmla v18.4s, v17.4s, v31.4s
st1 {v3.4s}, [x0], #16
fmla v21.4s, v20.4s, v31.4s
st1 {v7.4s}, [x0], #16
fmla v19.4s, v18.4s, v31.4s

fmla v22.4s, v21.4s, v31.4s
st1 {v19.4s}, [x0], #16
fmla v23.4s, v22.4s, v31.4s

st1 {v23.4s}, [x0], #16

sub x3, x3, #16
cmp x3, #16
bge MNNCoefLineL16Loop


MNNCoefLineL1:
cmp x3, #0
beq MNNCoefLineEnd

MNNCoefLineL1Loop:
ld1 {v4.4s}, [x1], #16

dup v0.2s, v4.s[0]
dup v1.2s, v4.s[1]
dup v2.2s, v4.s[2]
dup v3.2s, v4.s[3]

fmla v1.2s, v0.2s, v31.2s
fmla v2.2s, v1.2s, v31.2s
fmla v3.2s, v2.2s, v31.2s


st1 {v3.s}[0], [x0], #4


subs x3, x3, #1
bne MNNCoefLineL1Loop
MNNCoefLineEnd:
ret
#endif
