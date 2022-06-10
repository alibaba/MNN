//
//  MNNConvDwF23MulTransUnitFP16.S
//  MNN
//
//  Created by MNN on 2019/4/4.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNConvDwF23MulTransUnitFP16
//void MNNConvDwF23MulTransUnitFP16(FLOAT16 **cacheLine, const FLOAT16 *weigth, FLOAT16 *dest, size_t ow);
//Auto: x0:cacheLine, x1:weight, x2:dest, x3:ow, x4: bias, x5: parameters

stp d10, d11, [sp, #-32]!
stp d8,  d9,  [sp, #16]

ld1 {v8.8h}, [x4] // bias
ldr w9, [x5, #8]
ldr w10, [x5, #12]
dup v9.4s, w9 // min
dup v10.4s, w10 // max
fcvtn v9.4h, v9.4s
fcvtn v10.4h, v10.4s
dup v9.8h, v9.h[0]
dup v10.8h, v10.h[0]

ldr x4, [x0, #0]
ldr x5, [x0, #8]
ldr x6, [x0, #16]

ld1 {v4.8h, v5.8h, v6.8h, v7.8h}, [x1], #64
ld1 {v16.8h, v17.8h, v18.8h, v19.8h}, [x1], #64
ld1 {v28.8h, v29.8h, v30.8h, v31.8h}, [x1]

L2:
cmp x3, #2
blt L1

LoopL2:

ld1 {v20.8h, v21.8h}, [x4], #32
fmul v0.8h, v4.8h, v20.8h
ld1 {v22.8h, v23.8h}, [x4], #32
fmul v1.8h, v5.8h, v21.8h
fmul v2.8h, v6.8h, v22.8h
ld1 {v20.8h, v21.8h}, [x5], #32
fmul v3.8h, v7.8h, v23.8h

fmla v0.8h, v16.8h, v20.8h
ld1 {v22.8h, v23.8h}, [x5], #32
fmla v1.8h, v17.8h, v21.8h
fmla v2.8h, v18.8h, v22.8h
fmla v3.8h, v19.8h, v23.8h

ld1 {v20.8h, v21.8h}, [x6], #32
fmla v0.8h, v28.8h, v20.8h
fmla v1.8h, v29.8h, v21.8h
fadd v0.8h, v1.8h, v0.8h
ld1 {v22.8h, v23.8h}, [x6], #32

fmla v2.8h, v30.8h, v22.8h
fmla v3.8h, v31.8h, v23.8h
fadd v0.8h, v0.8h, v2.8h

fadd v3.8h, v3.8h, v1.8h
fsub v1.8h, v3.8h, v2.8h

fadd v0.8h, v0.8h, v8.8h
fadd v1.8h, v1.8h, v8.8h

fmin v0.8h, v0.8h, v10.8h
fmin v1.8h, v1.8h, v10.8h

fmax v0.8h, v0.8h, v9.8h
fmax v1.8h, v1.8h, v9.8h

st1 {v0.8h, v1.8h}, [x2], #32

sub x3, x3, #2
cmp x3, #2
bge LoopL2


L1:
cmp x3, #0
beq End
ld1 {v20.8h, v21.8h, v22.8h}, [x4]
fmul v0.8h, v4.8h, v20.8h
fmul v1.8h, v5.8h, v21.8h
fmul v2.8h, v6.8h, v22.8h
ld1 {v20.8h, v21.8h, v22.8h}, [x5]

fmla v0.8h, v16.8h, v20.8h
fmla v1.8h, v17.8h, v21.8h
fmla v2.8h, v18.8h, v22.8h

ld1 {v20.8h, v21.8h, v22.8h}, [x6]
fmla v0.8h, v28.8h, v20.8h
fmla v1.8h, v29.8h, v21.8h
fadd v0.8h, v1.8h, v0.8h

fmla v2.8h, v30.8h, v22.8h
fadd v0.8h, v0.8h, v2.8h

fadd v0.8h, v0.8h, v8.8h

fmin v0.8h, v0.8h, v10.8h

fmax v0.8h, v0.8h, v9.8h
st1 {v0.8h}, [x2]
End:

ldp d8,  d9,  [sp, #16]
ldp d10, d11, [sp], #32

ret
#endif
