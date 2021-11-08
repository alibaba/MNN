//
//  MNNPackC4_BF16.S
//  MNN
//
//  Created by MNN on 2021/02/24.
//  Copyright © 2018-2021 Alibaba Group Holding Limited.
//

#ifdef __aarch64__
#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNPackC4_BF16
//void MNNPackC4_BF16(float* dst, const float* src, size_t area, size_t depth)
//Auto load:
//x0:dst, x1:src, x2:area, x3:depth
mul x12, x2, x3
cmp x12, #0
beq UpEnd

ldr w10, [x4, #4] // dstDepthOffset
ldr w9, [x4, #0] // srcDepthOffset
uxtw x10, w10
uxtw x9, w9

//x12: srcDepthOffset:area*sizeof(int16_t)
mov x12, #2
mul x12, x9, x12

//r10 -> 4 * (dstArea * sizeof(int16_t) - area * sizeof(int16_t))
mov x5, #8
sub x10, x10, x2
mul x10, x5, x10

//r9 -> (srcArea * sizeof(int16_t) - area * sizeof(int16_t))
mov x6, #2
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
ld1 {v0.4h}, [x1], #8  // 4 * sizeof(int16_t)
ld1 {v1.4h}, [x5], #8
ld1 {v2.4h}, [x6], #8
ld1 {v3.4h}, [x7], #8

st4 {v0.4h, v1.4h, v2.4h, v3.4h}, [x0], #32  // 16 * sizeof(int16_t)
sub x8, x8, #4
cmp x8, #4
bge UpL4AreaLoop

UpL4AreaRemain:
cmp x8, #0
beq UpL4AreaRemainEnd
UpL4AreaRemainLoop:
ld1 {v0.h}[0], [x1], #2 // sizeof(int16_t)
ld1 {v0.h}[1], [x5], #2
ld1 {v0.h}[2], [x6], #2
ld1 {v0.h}[3], [x7], #2

st1 {v0.4h}, [x0], #8  // 4 * sizeof(int16_t)

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
ld1 {v0.4h}, [x1], #8  // 4 * sizeof(int16_t)
movi v3.4h, #0
ld1 {v1.4h}, [x5], #8
ld1 {v2.4h}, [x6], #8

st4 {v0.4h, v1.4h, v2.4h, v3.4h}, [x0], #32 // 16 * sizeof(int16_t)
sub x8, x8, #4
cmp x8, #4
bge UpL3AreaLoop

cmp x8, #0
beq UpL3AreaRemainEnd
UpL3AreaRemain:
movi v0.4h, #0
ld1 {v0.h}[0], [x1], #2 // sizeof(int16_t)
ld1 {v0.h}[1], [x5], #2
ld1 {v0.h}[2], [x6], #2

st1 {v0.4h}, [x0], #8 // 4 * sizeof(int16_t)

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
ld1 {v0.4h}, [x1], #8  // 4 * sizeof(int16_t)
movi v3.4h, #0
ld1 {v1.4h}, [x5], #8
movi v2.4h, #0

st4 {v0.4h, v1.4h, v2.4h, v3.4h}, [x0], #32 // 16 * sizeof(int16_t)
sub x8, x8, #4
cmp x8, #4
bge UpL2AreaLoop

cmp x8, #0
beq UpL2AreaRemainEnd
UpL2AreaRemain:
movi v0.4s, #0
ld1 {v0.h}[0], [x1], #2  // 2 * sizeof(int16_t)
ld1 {v0.h}[1], [x5], #2

st1 {v0.4h}, [x0], #8  // 4 * sizeof(int16_t)

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
ld1 {v0.4h}, [x1], #8  // 4 * sizeof(int16_t)
movi v3.4h, #0
movi v1.4h, #0
movi v2.4h, #0

st4 {v0.4h, v1.4h, v2.4h, v3.4h}, [x0], #32  // 16 * sizeof(int16_t)
sub x8, x8, #4
cmp x8, #4
bge UpL1AreaLoop

cmp x8, #0
beq UpL1AreaRemainEnd
UpL1AreaRemain:
movi v0.4h, #0
ld1 {v0.h}[0], [x1], #2  // sizeof(int16_t)

st1 {v0.4h}, [x0], #8  //4 * sizeof(int16_t)

subs x8, x8, #1
bne UpL1AreaRemain

UpL1AreaRemainEnd:

UpEnd:

ret

#endif
