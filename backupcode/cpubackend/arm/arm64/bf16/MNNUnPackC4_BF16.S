//
//  NEON_MNNUnPackC4_BF16.S
//  MNN
//
//  Created by MNN on 2019/02/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__
#include "MNNAsmGlobal.h"

.text
.align 5

asm_function NEON_MNNUnpackC4_BF16
// treate float pointer as int16_t*
//void NEON_MNNUnpackC4_BF16(float* dst, const float* src, size_t area, size_t depth)
//Auto load:
//x0:dst, x1:src, x2:area, x3:depth
mul x12, x2, x3
cmp x12, #0
beq DownEnd
ldr w10, [x4, #4] // dstDepthOffset
ldr w9, [x4, #0] // srcDepthOffset
uxtw x10, w10
uxtw x9, w9

//Swap x0, x1
mov x12, x0
mov x0, x1
mov x1, x12

//x12: dstDepthOffset:dstArea*sizeof(int16)
mov x12, #2
mul x12, x10, x12

//r10 -> (dstArea * sizeof(int16) - area * sizeof(int16))
mov x5, #2
sub x10, x10, x2
mul x10, x5, x10

//r9 -> 4 * (srcArea * sizeof(int16) - area * sizeof(int16))
mov x6, #8
sub x9, x9, x2
mul x9, x6, x9

DownL4:
cmp x3, #3
ble DownL3

DownL4Loop:
add x5, x1, x12
add x6, x12, x5
add x7, x12, x6
mov x8, x2
cmp x8, #3
ble DownL4AreaRemain
DownL4AreaLoop:
ld4 {v0.4h, v1.4h, v2.4h, v3.4h}, [x0], #32 // 16 * sizeof(int16_t)
st1 {v0.4h}, [x1], #8 // 4 * sizeof(int16_t)
st1 {v1.4h}, [x5], #8 // 4 * sizeof(int16_t)
st1 {v2.4h}, [x6], #8 // 4 * sizeof(int16_t)
st1 {v3.4h}, [x7], #8 // 4 * sizeof(int16_t)
sub x8, x8, #4
cmp x8, #4
bge DownL4AreaLoop

DownL4AreaRemain:
cmp x8, #0
beq DownL4AreaRemainEnd
DownL4AreaRemainLoop:
ld1 {v0.4h}, [x0], #8
st1 {v0.h}[0], [x1], #2
st1 {v0.h}[1], [x5], #2
st1 {v0.h}[2], [x6], #2
st1 {v0.h}[3], [x7], #2


subs x8, x8, #1
bne DownL4AreaRemainLoop
DownL4AreaRemainEnd:
sub x3, x3, #4
add x1, x7, x10
cmp x3, #4
add x0, x9, x0
bge DownL4Loop

DownL3:
cmp x3, #2
ble DownL2
add x5, x1, x12
add x6, x12, x5
mov x8, x2
cmp x8, #3
ble DownL3AreaRemain
DownL3AreaLoop:
ld4 {v0.4h, v1.4h, v2.4h, v3.4h}, [x0], #32  // 16 * sizeof(int16_t)
st1 {v0.4h}, [x1], #8 // 4 * sizeof(int16_t)
st1 {v1.4h}, [x5], #8 // 4 * sizeof(int16_t)
st1 {v2.4h}, [x6], #8 // 4 * sizeof(int16_t)
sub x8, x8, #4
cmp x8, #4
bge DownL3AreaLoop

cmp x8, #0
beq DownL3AreaRemainEnd
DownL3AreaRemain:
ld1 {v0.4h}, [x0], #8   // 4 * sizeof(int16_t)
st1 {v0.h}[0], [x1], #2 // sizeof(int16_t)
st1 {v0.h}[1], [x5], #2 // sizeof(int16_t)
st1 {v0.h}[2], [x6], #2 // sizeof(int16_t)

subs x8, x8, #1
bne DownL3AreaRemain

DownL3AreaRemainEnd:
sub x3, x3, #3


DownL2:
cmp x3, #1
ble DownL1
add x5, x1, x12
mov x8, x2
cmp x8, #3
ble DownL2AreaRemain
DownL2AreaLoop:
ld4 {v0.4h, v1.4h, v2.4h, v3.4h}, [x0], #32 // 16 * sizeof(int16_t)
st1 {v0.4h}, [x1], #8
st1 {v1.4h}, [x5], #8

sub x8, x8, #4
cmp x8, #4
bge DownL2AreaLoop

cmp x8, #0
beq DownL2AreaRemainEnd
DownL2AreaRemain:
ld1 {v0.4h}, [x0], #8 // 4 * sizeof(int16_t)
st1 {v0.h}[0], [x1], #2
st1 {v0.h}[1], [x5], #2

subs x8, x8, #1
bne DownL2AreaRemain

DownL2AreaRemainEnd:
sub x3, x3, #2

DownL1:
cmp x3, #0
beq DownEnd
mov x8, x2
cmp x8, #3
ble DownL1AreaRemain
DownL1AreaLoop:
ld4 {v0.4h, v1.4h, v2.4h, v3.4h}, [x0], #32 // 16 * sizeof(int16_t)
st1 {v0.4h}, [x1], #8

sub x8, x8, #4
cmp x8, #4
bge DownL1AreaLoop

cmp x8, #0
beq DownL1AreaRemainEnd
DownL1AreaRemain:
movi v0.4h, #0
ld1 {v0.4h}, [x0], #8  // 4 * sizeof(int16_t)
st1 {v0.h}[0], [x1], #2


subs x8, x8, #1
bne DownL1AreaRemain

DownL1AreaRemainEnd:

DownEnd:

ret


#endif

