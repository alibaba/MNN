//
//  MNNUInt8ToInt16WithOffsetC4Common.S
//  MNN
//
//  Created by MNN on 2018/10/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNUInt8ToInt16WithOffsetC4Common

//void MNNUInt8ToInt16WithOffsetC4Common(int16_t* dst, const uint8_t* src, size_t zeroPoint, size_t sizeQuad, size_t dstStride, size_t srcStride)

//Auto: x0:dst, x1:src, x2: zeroPoint, x3: sizeQuad
//x4: dstStride, x5:srcStride


dup v23.8h, w2

L8:
cmp x3, #8
blt L1


LoopL8:

ld1 {v0.s}[0], [x1], x5
ld1 {v0.s}[1], [x1], x5
ld1 {v0.s}[2], [x1], x5

uxtl v16.8h, v0.8b

ld1 {v0.s}[3], [x1], x5
sub v16.8h, v16.8h, v23.8h
uxtl2 v17.8h, v0.16b
st1 {v16.d}[0], [x0], x4
ld1 {v1.s}[0], [x1], x5
st1 {v16.d}[1], [x0], x4
sub v17.8h, v17.8h, v23.8h
ld1 {v1.s}[1], [x1], x5
st1 {v17.d}[0], [x0], x4
ld1 {v1.s}[2], [x1], x5
uxtl v16.8h, v1.8b
st1 {v17.d}[1], [x0], x4
sub v16.8h, v16.8h, v23.8h
ld1 {v1.s}[3], [x1], x5
st1 {v16.d}[0], [x0], x4
uxtl2 v17.8h, v1.16b
st1 {v16.d}[1], [x0], x4
sub v17.8h, v17.8h, v23.8h
st1 {v17.d}[0], [x0], x4
st1 {v17.d}[1], [x0], x4

sub x3, x3, #8
cmp x3, #8
bge LoopL8


L1:
cmp x3, #0
beq End

LoopL1:
ld1 {v0.s}[0], [x1], x5
uxtl v0.8h, v0.8b
sub v0.4h, v0.4h, v23.4h

st1 {v0.4h}, [x0], x4

subs x3, x3, #1
bne LoopL1



End:

ret

#endif
