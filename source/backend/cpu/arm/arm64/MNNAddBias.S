//
//  MNNAddBias.S
//  MNN
//
//  Created by MNN on 2019/02/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__
#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNAddBias
//void MNNAddBias(float* dst, const float* bias, int planeNumber, int biasNumber)
//x0:dst, x1:bias, x2:planeNumber, x3:biasNumber

cmp x3, #0
beq End

cmp x2, #0
beq End

LoopBias:
ld1 {v31.4s}, [x1], #16

mov x4, x2

L4:
cmp x4, #3
ble L1
Loop4:
mov x5, x0
ld1 {v0.4s, v1.4s}, [x5], #32
fadd v0.4s, v0.4s, v31.4s
ld1 {v2.4s, v3.4s}, [x5]
fadd v1.4s, v1.4s, v31.4s
fadd v2.4s, v2.4s, v31.4s
st1 {v0.4s, v1.4s}, [x0], #32
fadd v3.4s, v3.4s, v31.4s
st1 {v2.4s, v3.4s}, [x0], #32
sub x4, x4, #4
cmp x4, #4
bge Loop4

L1:
cmp x4, #0
beq EndLoopPlane
Loop1:
ld1 {v0.4s}, [x0]
fadd v0.4s, v0.4s, v31.4s
subs x4, x4, #1
st1 {v0.4s}, [x0], #16
bne Loop1

EndLoopPlane:

subs x3, x3, #1
bne LoopBias


End:

ret

#endif
