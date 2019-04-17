//
//  MNNReluWithSlope.S
//  MNN
//
//  Created by MNN on 2019/02/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__
#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNReluWithSlope
//void MNNReluWithSlope(float* dst, const float* src, size_t sizeQuad, float slope)

//Auto Load:
//x0:dst, x1:src, x2:sizeQuad, s0:slope

dup v23.4s, v0.s[0]

ReluL4:
cmp x2, #3
ble ReluL1

ReluL4Loop:
ld1 {v0.4s, v1.4s}, [x1], #32

fmul v16.4s, v23.4s, v0.4s
fmul v17.4s, v23.4s, v1.4s
fcmle v20.4s, v0.4s, #0

ld1 {v2.4s, v3.4s}, [x1], #32

fcmle v21.4s, v1.4s, #0

bit v0.16b, v16.16b, v20.16b
bit v1.16b, v17.16b, v21.16b
fmul v18.4s, v2.4s, v23.4s

st1 {v0.4s, v1.4s}, [x0], #32

fcmle v20.4s, v2.4s, #0
fcmle v21.4s, v3.4s, #0
fmul v19.4s, v3.4s, v23.4s
bit v2.16b, v18.16b, v20.16b
bit v3.16b, v19.16b, v21.16b

st1 {v2.4s, v3.4s}, [x0], #32

sub x2, x2, #4
cmp x2, #4
bge ReluL4Loop

ReluL1:
cmp x2, #0
beq ReluEnd

ReluL1Loop:
ld1 {v0.4s}, [x1], #16
fcmle v2.4s, v0.4s, #0
fmul v1.4s, v0.4s, v23.4s
bit v0.16b, v1.16b, v2.16b
st1 {v0.4s}, [x0], #16
subs x2, x2, #1
bne ReluL1Loop


ReluEnd:


ret

#endif
