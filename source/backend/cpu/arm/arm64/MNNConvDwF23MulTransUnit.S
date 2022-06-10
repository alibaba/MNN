//
//  MNNConvDwF23MulTransUnit.S
//  MNN
//
//  Created by MNN on 2019/4/4.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNConvDwF23MulTransUnit
//void MNNConvDwF23MulTransUnit(float **cacheLine, const float *weigth, float *dest, size_t ow, const float* bias, const float* parameters);
//Auto: x0:cacheLine, x1:weight, x2:dest, x3:ow, x4: bias, x5: parameters
stp d10, d11, [sp, #-32]!
stp d8,  d9,  [sp, #16]

ld1 {v8.4s}, [x4] // bias
ldr w9, [x5, #8]
ldr w10, [x5, #12]
dup v9.4s, w9 // min
dup v10.4s, w10 // max

ldr x4, [x0, #0]
ldr x5, [x0, #8]
ldr x6, [x0, #16]

ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x1], #64
ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x1], #64
ld1 {v28.4s, v29.4s, v30.4s, v31.4s}, [x1]


L2:
cmp x3, #2
blt L1

LoopL2:

ld1 {v20.4s, v21.4s}, [x4], #32
fmul v0.4s, v4.4s, v20.4s
ld1 {v22.4s, v23.4s}, [x4], #32
fmul v1.4s, v5.4s, v21.4s
fmul v2.4s, v6.4s, v22.4s
ld1 {v20.4s, v21.4s}, [x5], #32
fmul v3.4s, v7.4s, v23.4s

fmla v0.4s, v16.4s, v20.4s
ld1 {v22.4s, v23.4s}, [x5], #32
fmla v1.4s, v17.4s, v21.4s
fmla v2.4s, v18.4s, v22.4s
fmla v3.4s, v19.4s, v23.4s

ld1 {v20.4s, v21.4s}, [x6], #32
fmla v0.4s, v28.4s, v20.4s
fmla v1.4s, v29.4s, v21.4s
fadd v0.4s, v1.4s, v0.4s
ld1 {v22.4s, v23.4s}, [x6], #32

fmla v2.4s, v30.4s, v22.4s
fmla v3.4s, v31.4s, v23.4s
fadd v0.4s, v0.4s, v2.4s

fadd v3.4s, v3.4s, v1.4s
fsub v1.4s, v3.4s, v2.4s

fadd v0.4s, v0.4s, v8.4s
fadd v1.4s, v1.4s, v8.4s

fmin v0.4s, v0.4s, v10.4s
fmin v1.4s, v1.4s, v10.4s

fmax v0.4s, v0.4s, v9.4s
fmax v1.4s, v1.4s, v9.4s

st1 {v0.4s, v1.4s}, [x2], #32

sub x3, x3, #2
cmp x3, #2
bge LoopL2


L1:
cmp x3, #0
beq End
ld1 {v20.4s, v21.4s, v22.4s}, [x4]
fmul v0.4s, v4.4s, v20.4s
fmul v1.4s, v5.4s, v21.4s
fmul v2.4s, v6.4s, v22.4s
ld1 {v20.4s, v21.4s, v22.4s}, [x5]

fmla v0.4s, v16.4s, v20.4s
fmla v1.4s, v17.4s, v21.4s
fmla v2.4s, v18.4s, v22.4s

ld1 {v20.4s, v21.4s, v22.4s}, [x6]
fmla v0.4s, v28.4s, v20.4s
fmla v1.4s, v29.4s, v21.4s
fadd v0.4s, v1.4s, v0.4s

fmla v2.4s, v30.4s, v22.4s
fadd v0.4s, v0.4s, v2.4s

fadd v0.4s, v0.4s, v8.4s
fmin v0.4s, v0.4s, v10.4s
fmax v0.4s, v0.4s, v9.4s

st1 {v0.4s}, [x2]
End:

ldp d8,  d9,  [sp, #16]
ldp d10, d11, [sp], #32

ret
#endif
