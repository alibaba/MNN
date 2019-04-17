//
//  MNNPackC4.S
//  MNN
//
//  Created by MNN on 2019/02/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__
#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNPackC4
//void MNNPackC4(float* dst, const float* src, size_t area, size_t depth)
//Auto load:
//x0:dst, x1:src, x2:area, x3:depth
mul x4, x2, x3
cmp x4, #0
beq UpEnd


//x4: srcDepthOffset:area*sizeof(float)
mov x4, #4
mul x4, x2, x4

UpL4:
cmp x3, #3
ble UpL3

UpL4Loop:
add x5, x1, x4
add x6, x4, x5
add x7, x4, x6
mov x8, x2
cmp x8, #3
ble UpL4AreaRemain
UpL4AreaLoop:
ld1 {v0.4s}, [x1], #16
ld1 {v1.4s}, [x5], #16
ld1 {v2.4s}, [x6], #16
ld1 {v3.4s}, [x7], #16

st4 {v0.4s, v1.4s, v2.4s, v3.4s}, [x0], #64
sub x8, x8, #4
cmp x8, #4
bge UpL4AreaLoop

UpL4AreaRemain:
cmp x8, #0
beq UpL4AreaRemainEnd
UpL4AreaRemainLoop:
ld1 {v0.s}[0], [x1], #4
ld1 {v0.s}[1], [x5], #4
ld1 {v0.s}[2], [x6], #4
ld1 {v0.s}[3], [x7], #4

st1 {v0.4s}, [x0], #16

subs x8, x8, #1
bne UpL4AreaRemainLoop
UpL4AreaRemainEnd:
sub x3, x3, #4
mov x1, x7
cmp x3, #4
bge UpL4Loop

UpL3:
cmp x3, #2
ble UpL2
add x5, x1, x4
add x6, x4, x5
mov x8, x2
cmp x8, #3
ble UpL3AreaRemain
UpL3AreaLoop:
ld1 {v0.4s}, [x1], #16
movi v3.4s, #0
ld1 {v1.4s}, [x5], #16
ld1 {v2.4s}, [x6], #16

st4 {v0.4s, v1.4s, v2.4s, v3.4s}, [x0], #64
sub x8, x8, #4
cmp x8, #4
bge UpL3AreaLoop

cmp x8, #0
beq UpL3AreaRemainEnd
UpL3AreaRemain:
movi v0.4s, #0
ld1 {v0.s}[0], [x1], #4
ld1 {v0.s}[1], [x5], #4
ld1 {v0.s}[2], [x6], #4

st1 {v0.4s}, [x0], #16

subs x8, x8, #1
bne UpL3AreaRemain

UpL3AreaRemainEnd:
sub x3, x3, #3


UpL2:
cmp x3, #1
ble UpL1
add x5, x1, x4
mov x8, x2
cmp x8, #3
ble UpL2AreaRemain
UpL2AreaLoop:
ld1 {v0.4s}, [x1], #16
movi v3.4s, #0
ld1 {v1.4s}, [x5], #16
movi v2.4s, #0

st4 {v0.4s, v1.4s, v2.4s, v3.4s}, [x0], #64
sub x8, x8, #4
cmp x8, #4
bge UpL2AreaLoop

cmp x8, #0
beq UpL2AreaRemainEnd
UpL2AreaRemain:
movi v0.4s, #0
ld1 {v0.s}[0], [x1], #4
ld1 {v0.s}[1], [x5], #4

st1 {v0.4s}, [x0], #16

subs x8, x8, #1
bne UpL2AreaRemain

UpL2AreaRemainEnd:
sub x3, x3, #2

UpL1:
cmp x3, #0
beq UpEnd
mov x8, x2
cmp x8, #3
ble UpL1AreaRemain
UpL1AreaLoop:
ld1 {v0.4s}, [x1], #16
movi v3.4s, #0
movi v1.4s, #0
movi v2.4s, #0

st4 {v0.4s, v1.4s, v2.4s, v3.4s}, [x0], #64
sub x8, x8, #4
cmp x8, #4
bge UpL1AreaLoop

cmp x8, #0
beq UpL1AreaRemainEnd
UpL1AreaRemain:
movi v0.4s, #0
ld1 {v0.s}[0], [x1], #4

st1 {v0.4s}, [x0], #16

subs x8, x8, #1
bne UpL1AreaRemain

UpL1AreaRemainEnd:

UpEnd:

ret

#endif
