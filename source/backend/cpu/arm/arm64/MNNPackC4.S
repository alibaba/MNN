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
//void MNNPackC4(float* dst, const float* src, size_t area, size_t depth, int32_t* areaOffset)
//Auto load:
//x0:dst, x1:src, x2:area, x3:depth, x4: areaOffset, x5: areaOffset
mul x12, x2, x3
cmp x12, #0
beq UpEnd

ldr w10, [x4, #4] // dstDepthOffset
ldr w9, [x4, #0] // srcDepthOffset
uxtw x10, w10
uxtw x9, w9

//x12: srcDepthOffset:area*sizeof(float)
mov x12, #4
mul x12, x9, x12

//r10 -> 4 * (dstArea * sizeof(float) - area * sizeof(float))
mov x5, #16
sub x10, x10, x2
mul x10, x5, x10

//r9 -> (srcArea * sizeof(float) - area * sizeof(float))
mov x6, #4
sub x9, x9, x2
mul x9, x6, x9

UpL4:
cmp x3, #3
ble UpL3

UpL4Loop:
add x5, x1, x12
add x6, x12, x5
add x7, x12, x6
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
add x1, x7, x9
cmp x3, #4
add x0, x10, x0
bge UpL4Loop

UpL3:
cmp x3, #2
ble UpL2
add x5, x1, x12
add x6, x12, x5
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
add x5, x1, x12
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
