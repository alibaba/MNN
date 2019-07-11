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
//void MNNFloat2Int8(const float* src, int8_t* dst, size_t sizeQuad, float* scale, size_t aMin, size_t aMax);
//x0:src, x1:dst, x2:sizeQuad, x3:scale, x4:aMin, x5:aMax

ld1 {v31.4s}, [x3]

dup v30.16b, w4
dup v29.16b, w5

cmp x2, #3
ble FLLoop1

FLLoop4:
ld1 {v0.4s, v1.4s}, [x0], #32
fmul v0.4s, v0.4s, v31.4s
ld1 {v2.4s, v3.4s}, [x0], #32
fmul v1.4s, v1.4s, v31.4s
fmul v2.4s, v2.4s, v31.4s
fmul v3.4s, v3.4s, v31.4s

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

ret
#endif
