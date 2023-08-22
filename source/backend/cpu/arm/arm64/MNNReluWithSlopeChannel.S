//
//  MNNReluWithSlopeChannel.S
//  MNN
//
//  Created by MNN on 2019/02/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__
#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNReluWithSlopeChannel
//void MNNReluWithSlopeChannel(float* dst, const float* src, const float* slope, size_t sizeQuad, size_t depthQuad)

//Auto Load:
//x0:dst, x1:src, x2:slope, x3:sizeQuad, x4:depthQuad
stp d14, d15, [sp, #-64]!
stp d12, d13, [sp, #16]
stp d10, d11, [sp, #32]
stp d8,  d9,  [sp, #48]

cmp x4, #0
beq PReluEnd
cmp x3, #0
beq PReluEnd


PReluZLoop:
ld1 {v31.4s}, [x2], #16
mov x5, x3

PReluL16:
cmp x5, #15
ble PReluL8

PReluL16Loop:
ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x1], #64
ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x1], #64
ld1 {v24.4s, v25.4s, v26.4s, v27.4s}, [x1], #64

fcmle v16.4s, v0.4s, #0
fcmle v17.4s, v1.4s, #0
fcmle v18.4s, v2.4s, #0
fcmle v19.4s, v3.4s, #0
fcmle v20.4s, v4.4s, #0
fcmle v21.4s, v5.4s, #0
fcmle v22.4s, v6.4s, #0
fcmle v23.4s, v7.4s, #0

fmul v8.4s,  v0.4s, v31.4s
fmul v9.4s,  v1.4s, v31.4s
fmul v10.4s, v2.4s, v31.4s
fmul v11.4s, v3.4s, v31.4s
fmul v12.4s, v4.4s, v31.4s
fmul v13.4s, v5.4s, v31.4s
fmul v14.4s, v6.4s, v31.4s
fmul v15.4s, v7.4s, v31.4s

fcmle v28.4s, v24.4s, #0
fcmle v29.4s, v25.4s, #0
fcmle v30.4s, v26.4s, #0

bit v0.16b, v8.16b, v16.16b
bit v1.16b, v9.16b, v17.16b
bit v2.16b, v10.16b, v18.16b
bit v3.16b, v11.16b, v19.16b
bit v4.16b, v12.16b, v20.16b
bit v5.16b, v13.16b, v21.16b
bit v6.16b, v14.16b, v22.16b
bit v7.16b, v15.16b, v23.16b

st1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x0], #64
st1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x0], #64
fcmle v8.4s, v27.4s, #0
fmul v9.4s,  v24.4s, v31.4s
fmul v10.4s, v25.4s, v31.4s
fmul v11.4s, v26.4s, v31.4s
fmul v12.4s, v27.4s, v31.4s
ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x1], #64

fcmle v13.4s, v16.4s, #0
fcmle v14.4s, v17.4s, #0
fcmle v15.4s, v18.4s, #0
fcmle v0.4s, v19.4s, #0

fmul v20.4s, v16.4s, v31.4s
fmul v21.4s, v17.4s, v31.4s
fmul v22.4s, v18.4s, v31.4s
fmul v23.4s, v19.4s, v31.4s


bit v24.16b, v9.16b, v28.16b
bit v25.16b, v10.16b, v29.16b
bit v26.16b, v11.16b, v30.16b
bit v27.16b, v12.16b, v8.16b
bit v16.16b, v20.16b, v13.16b
bit v17.16b, v21.16b, v14.16b
bit v18.16b, v22.16b, v15.16b
bit v19.16b, v23.16b, v0.16b

st1 {v24.4s, v25.4s, v26.4s, v27.4s}, [x0], #64
st1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x0], #64

sub x5, x5, #16
cmp x5, #16
bge PReluL16Loop

PReluL8:
cmp x5, #7
ble PReluL4

PReluL8Loop:
ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x1], #64
ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x1], #64

fcmle v16.4s, v0.4s, #0
fcmle v17.4s, v1.4s, #0
fcmle v18.4s, v2.4s, #0
fcmle v19.4s, v3.4s, #0
fcmle v20.4s, v4.4s, #0
fcmle v21.4s, v5.4s, #0
fcmle v22.4s, v6.4s, #0
fcmle v23.4s, v7.4s, #0

fmul v8.4s,  v0.4s, v31.4s
fmul v9.4s,  v1.4s, v31.4s
fmul v10.4s, v2.4s, v31.4s
fmul v11.4s, v3.4s, v31.4s
fmul v12.4s, v4.4s, v31.4s
fmul v13.4s, v5.4s, v31.4s
fmul v14.4s, v6.4s, v31.4s
fmul v15.4s, v7.4s, v31.4s


bit v0.16b, v8.16b, v16.16b
bit v1.16b, v9.16b, v17.16b
bit v2.16b, v10.16b, v18.16b
bit v3.16b, v11.16b, v19.16b
bit v4.16b, v12.16b, v20.16b
bit v5.16b, v13.16b, v21.16b
bit v6.16b, v14.16b, v22.16b
bit v7.16b, v15.16b, v23.16b

st1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x0], #64
st1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x0], #64

sub x5, x5, #8
cmp x5, #8
bge PReluL8Loop

PReluL4:
cmp x5, #3
ble PReluL1

PReluL4Loop:
ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x1], #64

fcmle v8.4s, v0.4s, #0
fcmle v9.4s, v1.4s, #0
fcmle v10.4s, v2.4s, #0
fcmle v11.4s, v3.4s, #0

fmul v4.4s, v0.4s, v31.4s
fmul v5.4s, v1.4s, v31.4s
fmul v6.4s, v2.4s, v31.4s
fmul v7.4s, v3.4s, v31.4s

bit v0.16b, v4.16b, v8.16b
bit v1.16b, v5.16b, v9.16b
bit v2.16b, v6.16b, v10.16b
bit v3.16b, v7.16b, v11.16b

st1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x0], #64

sub x5, x5, #4
cmp x5, #4
bge PReluL4Loop

PReluL1:
cmp x5, #0
beq PReluL1End

PReluL1Loop:
ld1 {v0.4s}, [x1], #16
fcmle v2.4s, v0.4s, #0
fmul v1.4s, v0.4s, v31.4s
bit v0.16b, v1.16b, v2.16b
st1 {v0.4s}, [x0], #16
subs x5, x5, #1
bne PReluL1Loop

PReluL1End:

subs x4, x4, #1
bne PReluZLoop


PReluEnd:
ldp d8,  d9,  [sp, #48]
ldp d10, d11, [sp, #32]
ldp d12, d13, [sp, #16]
ldp d14, d15, [sp], #64

ret
#endif
