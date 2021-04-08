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
mul x4, x2, x3
cmp x4, #0
beq UpEnd


//x4: srcDepthOffset:area*sizeof(float)
mov x4, #2 // sizeof(int16_t)
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
add x5, x1, x4
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
