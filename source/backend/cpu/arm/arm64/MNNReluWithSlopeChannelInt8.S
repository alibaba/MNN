//
//  MNNReluWithSlopeChannel.S
//  MNN
//
//  Created by MNN on 2023/07/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//
/*
    struct QuanPrePostParameters{
    float* inputScale;
    float* outputScale;
    ssize_t* inputZeroPoint;
    ssize_t* outputZeroPoint;
    ssize_t minValue;
    ssize_t maxValue;
};
 */
#ifdef __aarch64__
#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNReluWithSlopeChannelInt8
// MNNReluWithSlopeChannelInt8(int8_t* dst, const int8_t* src, const float* slope, size_t planeNumber, size_t depthQuad, QuanPrePostParameters *params)
// Auto load:
// x0: dst, x1: src, x2: slope, x3: planeNumber, x4: depthQuad, x5: params
// Load from x5:  x8: inputZeroPoint, x9: outputZeroPoint, x10: minValue, x11: maxValue

ldr x8, [x5, #16]
ldr x9, [x5, #24]
ldr x10, [x5, #32]
ldr x11, [x5, #40]


stp d14, d15, [sp, #-64]!
stp d12, d13, [sp, #16]
stp d10, d11, [sp, #32]
stp d8,  d9,  [sp, #48]

cmp x3, #0
beq End
cmp x4, #0
beq End

ld1r {v29.8b}, [x8] // inputZeroPoint
ld1r {v28.8b}, [x9] // outputZeroPoint
dup v26.16b, w10
dup v27.16b, w11
/*
Quant parameters
*/


PReluZLoop:
ld1 {v31.4s}, [x2], #16 // slope
mov x5, x3
cmp x5, #3
ble PReluL1

PReluL4Loop:
ld1 {v0.16b}, [x1], #16
cmlt v30.16b, v0.16b, #0 // mask0: x<0

sxtl v1.8h, v0.8b
sxtl2 v2.8h, v0.16b
ssubw v1.8h, v1.8h, v29.8b
ssubw v2.8h, v2.8h, v29.8b
sxtl v3.4s, v1.4h
sxtl2 v4.4s, v1.8h
sxtl v5.4s, v2.4h
sxtl2 v6.4s, v2.8h

scvtf v3.4s, v3.4s
scvtf v4.4s, v4.4s
scvtf v5.4s, v5.4s
scvtf v6.4s, v6.4s

fmul v3.4s, v3.4s, v31.4s
fmul v4.4s, v4.4s, v31.4s
fmul v5.4s, v5.4s, v31.4s
fmul v6.4s, v6.4s, v31.4s

fcvtas v3.4s, v3.4s
fcvtas v4.4s, v4.4s
fcvtas v5.4s, v5.4s
fcvtas v6.4s, v6.4s

sqxtn  v7.4h, v3.4s
sqxtn2 v7.8h, v4.4s
sqxtn  v8.4h, v5.4s
sqxtn2 v8.8h, v6.4s

saddw v7.8h, v7.8h, v28.8b
saddw v8.8h, v8.8h, v28.8b

sqxtn v9.8b, v7.8h
sqxtn2 v9.16b, v8.8h
smax v9.16b, v9.16b, v26.16b
smin v9.16b, v9.16b, v27.16b

bit v0.16b, v9.16b, v30.16b
st1 {v0.16b}, [x0], #16

sub x5, x5, #4
cmp x5, #4
bge PReluL4Loop

PReluL1:
cmp x5, #0

beq PReluL1End

PReluL1Loop:
ld1 {v0.s}[0], [x1], #4
cmlt v30.8b, v0.8b, #0

sxtl v1.8h, v0.8b
ssubw v1.8h, v1.8h, v29.8b
sxtl v1.4s, v1.4h
scvtf v1.4s, v1.4s
fmul v1.4s, v1.4s, v31.4s
fcvtas v1.4s, v1.4s
sqxtn v1.4h, v1.4s
saddw v1.8h, v1.8h, v28.8b
sqxtn v1.8b, v1.8h
smax v1.8b, v1.8b, v26.8b
smin v1.8b, v1.8b, v27.8b

bit v0.8b, v1.8b, v30.8b
st1 {v0.s}[0], [x0], #4
subs x5, x5, #1
bne PReluL1Loop

PReluL1End:

subs x4, x4, #1
bne PReluZLoop

End:
    ldp d8,  d9,  [sp, #48]
    ldp d10, d11, [sp, #32]
    ldp d12, d13, [sp, #16]
    ldp d14, d15, [sp], #64
    ret

#endif