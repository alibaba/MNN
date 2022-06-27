//
//  MNNReluWithSlopeChannelBF16.S
//  MNN
//
//  Created by MNN on 2022/06/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__
#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNReluWithSlopeChannelBF16
//void MNNReluWithSlopeChannel(float* dst, const float* src, const float* slope, size_t sizeQuad, size_t depthQuad)

//Auto Load:
//x0:dst, x1:src, x2:slope, x3:sizeQuad, x4:depthQuad


cmp x4, #0
beq PReluEnd
cmp x3, #0
beq PReluEnd


PReluZLoop:
ld1 {v23.4h}, [x2], #8
shll v23.4s, v23.4h, #16
mov x5, x3
cmp x5, #3
ble PReluL1

PReluL4Loop:
ld1 {v0.4h, v1.4h}, [x1], #16
shll v0.4s, v0.4h, #16
shll v1.4s, v1.4h, #16

fcmle v20.4s, v0.4s, #0
fcmle v21.4s, v1.4s, #0

ld1 {v2.4h, v3.4h}, [x1], #16
shll v2.4s, v2.4h, #16
shll v3.4s, v3.4h, #16

fmul v16.4s, v0.4s, v23.4s
fmul v17.4s, v1.4s, v23.4s
bit v0.16b, v16.16b, v20.16b
bit v1.16b, v17.16b, v21.16b

fmul v16.4s, v2.4s, v23.4s
fmul v17.4s, v3.4s, v23.4s
shrn v0.4h, v0.4s, #16
shrn v1.4h, v1.4s, #16


st1 {v0.4h, v1.4h}, [x0], #16

fcmle v20.4s, v2.4s, #0
fcmle v21.4s, v3.4s, #0
bit v2.16b, v16.16b, v20.16b
bit v3.16b, v17.16b, v21.16b
shrn v2.4h, v2.4s, #16
shrn v3.4h, v3.4s, #16

st1 {v2.4h, v3.4h}, [x0], #16
sub x5, x5, #4
cmp x5, #4
bge PReluL4Loop

PReluL1:
cmp x5, #0

beq PReluL1End

PReluL1Loop:
ld1 {v0.4h}, [x1], #8
shll v0.4s, v0.4h, #16
fcmle v2.4s, v0.4s, #0
fmul v1.4s, v0.4s, v23.4s
bit v0.16b, v1.16b, v2.16b
shrn v0.4h, v0.4s, #16
st1 {v0.4h}, [x0], #8
subs x5, x5, #1
bne PReluL1Loop

PReluL1End:

subs x4, x4, #1
bne PReluZLoop


PReluEnd:

ret
#endif
