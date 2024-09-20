//
//  MNNPackInt8C2.S
//  MNN
//
//  Created by MNN on 2019/02/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__
#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNPackInt8C2
//void MNNPackInt8C2(float* dst, const float* src, size_t area, size_t depth, int32_t* areaOffset)
//Auto load:
//x0:dst, x1:src, x2:area, x3:depth, x4: areaOffset, x5: areaOffset

ldr w10, [x4, #4] // dstDepthOffset
ldr w9, [x4, #0] // srcDepthOffset
uxtw x10, w10
uxtw x9, w9

//x12: srcDepthOffset:area*sizeof(float)
mov x12, #4
mul x12, x9, x12

//r10 -> 2 * (dstArea * sizeof(float) - area * sizeof(float))
mov x5, #8
sub x10, x10, x2
mul x10, x5, x10

//r9 -> (srcArea * sizeof(float) - area * sizeof(float))
mov x6, #4
sub x9, x9, x2
mul x9, x6, x9

UpL2:
cmp x3, #1
ble UpL1

UpL2Loop:
add x5, x1, x12
mov x8, x2
cmp x8, #3
ble UpL2AreaRemain
UpL2AreaLoop:
ld1 {v0.4s}, [x1], #16
ld1 {v1.4s}, [x5], #16

st2 {v0.4s, v1.4s}, [x0], #32
sub x8, x8, #4
cmp x8, #4
bge UpL2AreaLoop

cmp x8, #0
beq UpL2AreaRemainEnd
UpL2AreaRemain:
ld1 {v0.s}[0], [x1], #4
ld1 {v0.s}[1], [x5], #4

st1 {v0.d}[0], [x0], #8

subs x8, x8, #1
bne UpL2AreaRemain

UpL2AreaRemainEnd:
sub x3, x3, #2
add x1, x5, x9
cmp x3, #2
add x0, x10, x0
bge UpL2Loop

UpL1:
cmp x3, #0
beq UpEnd
mov x8, x2
cmp x8, #3
ble UpL1AreaRemain
UpL1AreaLoop:
ld1 {v0.4s}, [x1], #16
movi v1.4s, #0

st2 {v0.4s, v1.4s}, [x0], #32
sub x8, x8, #4
cmp x8, #4
bge UpL1AreaLoop

cmp x8, #0
beq UpL1AreaRemainEnd
UpL1AreaRemain:
movi v0.4s, #0
ld1 {v0.s}[0], [x1], #4

st1 {v0.d}[0], [x0], #8

subs x8, x8, #1
bne UpL1AreaRemain

UpL1AreaRemainEnd:

UpEnd:

ret

#endif
