//
//  MNNFloat2Int8.S
//  MNN
//
//  Created by MNN on 2019/01/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNFloat2Int8
//void MNNFloat2Int8(const float* src, int8_t* dst, size_t sizeQuad, float* scale, size_t aMin, size_t aMax, size_t zeroPoint);
//x0:src, x1:dst, x2:sizeQuad, x3:scale, x4:aMin, x5:aMax, x6:zeroPoint
stp d14, d15, [sp, #-64]!
stp d12, d13, [sp, #16]
stp d10, d11, [sp, #32]
stp d8,  d9,  [sp, #48]

ld1 {v31.4s}, [x3]

dup v30.16b, w4
dup v29.16b, w5

// copy zero point
dup v28.4s, w6
scvtf v28.4s, v28.4s

FL32:
cmp x2, #32
ble FL16

FLLoop32:
ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x0], #64
ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x0], #64
ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x0], #64
ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [x0], #64
ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x0], #64
ld1 {v20.4s, v21.4s, v22.4s, v23.4s}, [x0], #64
// ld1 {v24.4s, v25.4s, v26.4s, v27.4s}, [x0], #64
// ld1 {v28.4s, v29.4s, v30.4s, v31.4s}, [x0], #64
fmul v0.4s, v0.4s, v31.4s
fadd v0.4s, v0.4s, v28.4s
fmul v1.4s, v1.4s, v31.4s
fadd v1.4s, v1.4s, v28.4s
fmul v2.4s, v2.4s, v31.4s
fadd v2.4s, v2.4s, v28.4s
fmul v3.4s, v3.4s, v31.4s
fadd v3.4s, v3.4s, v28.4s

fmul v4.4s, v4.4s, v31.4s
fadd v4.4s, v4.4s, v28.4s
fmul v5.4s, v5.4s, v31.4s
fadd v5.4s, v5.4s, v28.4s
fmul v6.4s, v6.4s, v31.4s
fadd v6.4s, v6.4s, v28.4s
fmul v7.4s, v7.4s, v31.4s
fadd v7.4s, v7.4s, v28.4s

fmul v8.4s, v8.4s, v31.4s
fadd v8.4s, v8.4s, v28.4s
fmul v9.4s, v9.4s, v31.4s
fadd v9.4s, v9.4s, v28.4s
fmul v10.4s, v10.4s, v31.4s
fadd v10.4s, v10.4s, v28.4s
fmul v11.4s, v11.4s, v31.4s
fadd v11.4s, v11.4s, v28.4s

fmul v12.4s, v12.4s, v31.4s
fadd v12.4s, v12.4s, v28.4s
fmul v13.4s, v13.4s, v31.4s
fadd v13.4s, v13.4s, v28.4s
fmul v14.4s, v14.4s, v31.4s
fadd v14.4s, v14.4s, v28.4s
fmul v15.4s, v15.4s, v31.4s
fadd v15.4s, v15.4s, v28.4s


fmul v16.4s, v16.4s, v31.4s
fadd v16.4s, v16.4s, v28.4s
fmul v17.4s, v17.4s, v31.4s
fadd v17.4s, v17.4s, v28.4s
fmul v18.4s, v18.4s, v31.4s
fadd v18.4s, v18.4s, v28.4s
fmul v19.4s, v19.4s, v31.4s
fadd v19.4s, v19.4s, v28.4s

fmul v20.4s, v20.4s, v31.4s
fadd v20.4s, v20.4s, v28.4s
fmul v21.4s, v21.4s, v31.4s
fadd v21.4s, v21.4s, v28.4s
fmul v22.4s, v22.4s, v31.4s
fadd v22.4s, v22.4s, v28.4s
fmul v23.4s, v23.4s, v31.4s
fadd v23.4s, v23.4s, v28.4s

fcvtas v0.4s, v0.4s
fcvtas v1.4s, v1.4s
fcvtas v2.4s, v2.4s
fcvtas v3.4s, v3.4s
fcvtas v4.4s, v4.4s
fcvtas v5.4s, v5.4s
fcvtas v6.4s, v6.4s
fcvtas v7.4s, v7.4s

fcvtas v8.4s, v8.4s
fcvtas v9.4s, v9.4s
fcvtas v10.4s, v10.4s
fcvtas v11.4s, v11.4s
fcvtas v12.4s, v12.4s
fcvtas v13.4s, v13.4s
fcvtas v14.4s, v14.4s
fcvtas v15.4s, v15.4s

fcvtas v16.4s, v16.4s
fcvtas v17.4s, v17.4s
fcvtas v18.4s, v18.4s
fcvtas v19.4s, v19.4s
fcvtas v20.4s, v20.4s
fcvtas v21.4s, v21.4s
fcvtas v22.4s, v22.4s
fcvtas v23.4s, v23.4s


sqxtn v24.4h, v0.4s
sqxtn2 v24.8h, v1.4s
sqxtn v25.4h, v2.4s
sqxtn2 v25.8h, v3.4s
sqxtn v26.4h, v4.4s
sqxtn2 v26.8h, v5.4s
sqxtn v27.4h, v6.4s
sqxtn2 v27.8h, v7.4s

sqxtn v0.4h, v8.4s
sqxtn2 v0.8h, v9.4s
sqxtn v1.4h, v10.4s
sqxtn2 v1.8h, v11.4s
sqxtn v2.4h, v12.4s
sqxtn2 v2.8h, v13.4s
sqxtn v3.4h, v14.4s
sqxtn2 v3.8h, v15.4s

sqxtn v4.4h, v16.4s
sqxtn2 v4.8h, v17.4s
sqxtn v5.4h, v18.4s
sqxtn2 v5.8h, v19.4s
sqxtn v6.4h, v20.4s
sqxtn2 v6.8h, v21.4s
sqxtn v7.4h, v22.4s
sqxtn2 v7.8h, v23.4s

ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x0], #64
ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [x0], #64

sqxtn v24.8b, v24.8h
sqxtn2 v24.16b, v25.8h
sqxtn v26.8b, v26.8h
sqxtn2 v26.16b, v27.8h
sqxtn v0.8b, v0.8h
sqxtn2 v0.16b, v1.8h
sqxtn v2.8b, v2.8h
sqxtn2 v2.16b, v3.8h

sqxtn v4.8b, v4.8h
sqxtn v6.8b, v6.8h
sqxtn2 v4.16b, v5.8h
sqxtn2 v6.16b, v7.8h

fmul v8.4s, v8.4s, v31.4s
fadd v8.4s, v8.4s, v28.4s
fmul v9.4s, v9.4s, v31.4s
fadd v9.4s, v9.4s, v28.4s
fmul v10.4s, v10.4s, v31.4s
fadd v10.4s, v10.4s, v28.4s
fmul v11.4s, v11.4s, v31.4s
fadd v11.4s, v11.4s, v28.4s

fmul v12.4s, v12.4s, v31.4s
fadd v12.4s, v12.4s, v28.4s
fmul v13.4s, v13.4s, v31.4s
fadd v13.4s, v13.4s, v28.4s
fmul v14.4s, v14.4s, v31.4s
fadd v14.4s, v14.4s, v28.4s
fmul v15.4s, v15.4s, v31.4s
fadd v15.4s, v15.4s, v28.4s

fcvtas v8.4s, v8.4s
fcvtas v9.4s, v9.4s
fcvtas v10.4s, v10.4s
fcvtas v11.4s, v11.4s
fcvtas v12.4s, v12.4s
fcvtas v13.4s, v13.4s
fcvtas v14.4s, v14.4s
fcvtas v15.4s, v15.4s

sqxtn v16.4h, v8.4s
sqxtn2 v16.8h, v9.4s
sqxtn v17.4h, v10.4s
sqxtn2 v17.8h, v11.4s
sqxtn v18.4h, v12.4s
sqxtn2 v18.8h, v13.4s
sqxtn v19.4h, v14.4s
sqxtn2 v19.8h, v15.4s

smin v24.16b, v24.16b, v29.16b
smax v24.16b, v24.16b, v30.16b
smin v25.16b, v26.16b, v29.16b
smax v25.16b, v25.16b, v30.16b

sqxtn v20.8b, v16.8h
sqxtn2 v20.16b, v17.8h
sqxtn v21.8b, v18.8h
sqxtn2 v21.16b, v19.8h

smin v26.16b, v0.16b, v29.16b
smax v26.16b, v26.16b, v30.16b
smin v27.16b, v2.16b, v29.16b
smax v27.16b, v27.16b, v30.16b

smin v12.16b, v4.16b, v29.16b
smax v12.16b, v12.16b, v30.16b
smin v13.16b, v6.16b, v29.16b
smax v13.16b, v13.16b, v30.16b

smin v14.16b, v20.16b, v29.16b
smax v14.16b, v14.16b, v30.16b
smin v15.16b, v21.16b, v29.16b
smax v15.16b, v15.16b, v30.16b

st1 {v24.4s, v25.4s, v26.4s, v27.4s}, [x1], #64
st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [x1], #64

sub x2, x2, #32
cmp x2, #32
bge FLLoop32

FL16:
cmp x2, #16
ble FL8

FLLoop16:
ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x0], #64
ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x0], #64
ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x0], #64
ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [x0], #64
fmul v0.4s, v0.4s, v31.4s
fadd v0.4s, v0.4s, v28.4s
fmul v1.4s, v1.4s, v31.4s
fadd v1.4s, v1.4s, v28.4s
fmul v2.4s, v2.4s, v31.4s
fadd v2.4s, v2.4s, v28.4s
fmul v3.4s, v3.4s, v31.4s
fadd v3.4s, v3.4s, v28.4s

fmul v4.4s, v4.4s, v31.4s
fadd v4.4s, v4.4s, v28.4s
fmul v5.4s, v5.4s, v31.4s
fadd v5.4s, v5.4s, v28.4s
fmul v6.4s, v6.4s, v31.4s
fadd v6.4s, v6.4s, v28.4s
fmul v7.4s, v7.4s, v31.4s
fadd v7.4s, v7.4s, v28.4s

fmul v8.4s, v8.4s, v31.4s
fadd v8.4s, v8.4s, v28.4s
fmul v9.4s, v9.4s, v31.4s
fadd v9.4s, v9.4s, v28.4s
fmul v10.4s, v10.4s, v31.4s
fadd v10.4s, v10.4s, v28.4s
fmul v11.4s, v11.4s, v31.4s
fadd v11.4s, v11.4s, v28.4s

fmul v12.4s, v12.4s, v31.4s
fadd v12.4s, v12.4s, v28.4s
fmul v13.4s, v13.4s, v31.4s
fadd v13.4s, v13.4s, v28.4s
fmul v14.4s, v14.4s, v31.4s
fadd v14.4s, v14.4s, v28.4s
fmul v15.4s, v15.4s, v31.4s
fadd v15.4s, v15.4s, v28.4s

fcvtas v0.4s, v0.4s
fcvtas v1.4s, v1.4s
fcvtas v2.4s, v2.4s
fcvtas v3.4s, v3.4s
fcvtas v4.4s, v4.4s
fcvtas v5.4s, v5.4s
fcvtas v6.4s, v6.4s
fcvtas v7.4s, v7.4s

fcvtas v8.4s, v8.4s
fcvtas v9.4s, v9.4s
fcvtas v10.4s, v10.4s
fcvtas v11.4s, v11.4s
fcvtas v12.4s, v12.4s
fcvtas v13.4s, v13.4s
fcvtas v14.4s, v14.4s
fcvtas v15.4s, v15.4s

sqxtn v16.4h, v0.4s
sqxtn2 v16.8h, v1.4s
sqxtn v17.4h, v2.4s
sqxtn2 v17.8h, v3.4s
sqxtn v18.4h, v4.4s
sqxtn2 v18.8h, v5.4s
sqxtn v19.4h, v6.4s
sqxtn2 v19.8h, v7.4s

sqxtn v20.4h, v8.4s
sqxtn2 v20.8h, v9.4s
sqxtn v21.4h, v10.4s
sqxtn2 v21.8h, v11.4s
sqxtn v22.4h, v12.4s
sqxtn2 v22.8h, v13.4s
sqxtn v23.4h, v14.4s
sqxtn2 v23.8h, v15.4s

sqxtn v24.8b, v16.8h
sqxtn2 v24.16b, v17.8h
sqxtn v25.8b, v18.8h
sqxtn2 v25.16b, v19.8h
sqxtn v26.8b, v20.8h
sqxtn2 v26.16b, v21.8h
sqxtn v27.8b, v22.8h
sqxtn2 v27.16b, v23.8h
smin v24.16b, v24.16b, v29.16b
smax v24.16b, v24.16b, v30.16b
smin v25.16b, v25.16b, v29.16b
smax v25.16b, v25.16b, v30.16b
smin v26.16b, v26.16b, v29.16b
smax v26.16b, v26.16b, v30.16b
smin v27.16b, v27.16b, v29.16b
smax v27.16b, v27.16b, v30.16b

st1 {v24.4s, v25.4s, v26.4s, v27.4s}, [x1], #64

sub x2, x2, #16
cmp x2, #16
bge FLLoop16

FL8:
cmp x2, #8
ble FL4

FLLoop8:
ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x0], #64
ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x0], #64
fmul v0.4s, v0.4s, v31.4s
fadd v0.4s, v0.4s, v28.4s
fmul v1.4s, v1.4s, v31.4s
fadd v1.4s, v1.4s, v28.4s
fmul v2.4s, v2.4s, v31.4s
fadd v2.4s, v2.4s, v28.4s
fmul v3.4s, v3.4s, v31.4s
fadd v3.4s, v3.4s, v28.4s

fmul v4.4s, v4.4s, v31.4s
fadd v4.4s, v4.4s, v28.4s
fmul v5.4s, v5.4s, v31.4s
fadd v5.4s, v5.4s, v28.4s
fmul v6.4s, v6.4s, v31.4s
fadd v6.4s, v6.4s, v28.4s
fmul v7.4s, v7.4s, v31.4s
fadd v7.4s, v7.4s, v28.4s

fcvtas v0.4s, v0.4s
fcvtas v1.4s, v1.4s
fcvtas v2.4s, v2.4s
fcvtas v3.4s, v3.4s
fcvtas v4.4s, v4.4s
fcvtas v5.4s, v5.4s
fcvtas v6.4s, v6.4s
fcvtas v7.4s, v7.4s

sqxtn v8.4h, v0.4s
sqxtn2 v8.8h, v1.4s
sqxtn v9.4h, v2.4s
sqxtn2 v9.8h, v3.4s
sqxtn v10.4h, v4.4s
sqxtn2 v10.8h, v5.4s
sqxtn v11.4h, v6.4s
sqxtn2 v11.8h, v7.4s

sqxtn v12.8b, v8.8h
sqxtn2 v12.16b, v9.8h
sqxtn v13.8b, v10.8h
sqxtn2 v13.16b, v11.8h
smin v12.16b, v12.16b, v29.16b
smax v12.16b, v12.16b, v30.16b
smin v13.16b, v13.16b, v29.16b
smax v13.16b, v13.16b, v30.16b

st1 {v12.4s, v13.4s}, [x1], #32

sub x2, x2, #8
cmp x2, #8
bge FLLoop8

FL4:
cmp x2, #3
ble FL1

FLLoop4:
ld1 {v0.4s, v1.4s}, [x0], #32
fmul v0.4s, v0.4s, v31.4s
fadd v0.4s, v0.4s, v28.4s
ld1 {v2.4s, v3.4s}, [x0], #32
fmul v1.4s, v1.4s, v31.4s
fadd v1.4s, v1.4s, v28.4s
fmul v2.4s, v2.4s, v31.4s
fadd v2.4s, v2.4s, v28.4s
fmul v3.4s, v3.4s, v31.4s
fadd v3.4s, v3.4s, v28.4s

fcvtas v0.4s, v0.4s
fcvtas v4.4s, v2.4s
fcvtas v6.4s, v3.4s
fcvtas v2.4s, v1.4s

sqxtn v0.4h, v0.4s
sqxtn2 v0.8h, v2.4s
sqxtn v1.4h, v4.4s
sqxtn2 v1.8h, v6.4s

sqxtn v0.8b, v0.8h
sqxtn2 v0.16b, v1.8h
smin v0.16b, v0.16b, v29.16b
smax v0.16b, v0.16b, v30.16b

st1 {v0.4s}, [x1], #16

sub x2, x2, #4
cmp x2, #4
bge FLLoop4


FL1:
cmp x2, #0
beq FLEnd

FLLoop1:
ld1 {v0.4s}, [x0], #16
fmul v0.4s, v0.4s, v31.4s
fadd v0.4s, v0.4s, v28.4s

//st1 {v31.4s}, [x0], #16
fcvtas v0.4s, v0.4s
sqxtn v0.4h, v0.4s
sqxtn v0.8b, v0.8h

smin v0.8b, v0.8b, v29.8b
smax v0.8b, v0.8b, v30.8b

st1 {v0.s}[0], [x1], #4

subs x2, x2, #1
bne FLLoop1

FLEnd:
ldp d8,  d9,  [sp, #48]
ldp d10, d11, [sp, #32]
ldp d12, d13, [sp, #16]
ldp d14, d15, [sp], #64
ret
#endif
