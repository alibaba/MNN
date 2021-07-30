//
//  NEON_MNNUnPackC4_BF16.S
//  MNN
//
//  Created by MNN on 2021/02/24.
//  Copyright © 2018-2021 Alibaba Group Holding Limited.
//

#ifdef __arm__
#ifndef __aarch64__

#include "MNNAsmGlobal.h"
.text
.align 5

// .macro transpose
// vtrn.16 d0, d1
// vtrn.16 d2, d3
// vswp d0[2-3], d1[2-3] // should swap high half of d-vector, the half length is 32-bit. there is no instruction, we use vld4.16 instead
// vswp d2[2-3], d3[2-3]
// .endm


asm_function NEON_MNNUnpackC4_BF16
// treate float pointer as int16_t*
//void NEON_MNNUnpackC4_BF16(float* dst, const float* src, size_t area, size_t depth);
//Auto load:
//r0:dst, r1:src, r2:area, r3:depth


push {r4-r8, r10, lr} // avoid to touch platform-register r-9
ldr lr, [sp, #28]
ldr r10, [lr, #4]
ldr lr, [lr, #0]

mul r4, r2, r3
cmp r4, #0
beq DownEnd

//Swap r0 and r1 for conviniense
mov r4, r0
mov r0, r1
mov r1, r4

//r4: dstDepthOffset:dstArea*sizeof(int16_t)
mov r4, #2
mul r4, r10, r4

//lr -> 4 * (srcArea * sizeof(int16_t) - area * sizeof(int16_t))
mov r12, #8
sub lr, lr, r2
mul lr, r12, lr

//r10 -> (dstArea * sizeof(int16_t) - area * sizeof(int16_t))
mov r12, #2
sub r10, r10, r2
mul r10, r12, r10

DownL4:
cmp r3, #3
ble DownL3

DownL4Loop:
add r5, r1, r4
add r6, r4, r5
add r7, r4, r6
mov r8, r2
cmp r8, #3
ble DownL4AreaRemain
DownL4AreaLoop:
vld4.16 {d0, d1, d2, d3}, [r0]! // load and transpose 4x4 matrix of int16_t
// transpose // no suitable instruction to transpose int16_t type
sub r8, r8, #4
vst1.16 {d0}, [r1]!
vst1.16 {d1}, [r5]!
vst1.16 {d2}, [r6]!
vst1.16 {d3}, [r7]!
cmp r8, #4
bge DownL4AreaLoop

DownL4AreaRemain:
cmp r8, #0
beq DownL4AreaRemainEnd
DownL4AreaRemainLoop:

vld1.16 {d0}, [r0]!

vst1.16 {d0[0]}, [r1]!
vst1.16 {d0[1]}, [r5]!
vst1.16 {d0[2]}, [r6]!
vst1.16 {d0[3]}, [r7]!

subs r8, r8, #1
bne DownL4AreaRemainLoop
DownL4AreaRemainEnd:
sub r3, r3, #4
add r1, r7, r10
cmp r3, #4
add r0, lr, r0
bge DownL4Loop

DownL3:
cmp r3, #2
ble DownL2
add r5, r1, r4
add r6, r4, r5
mov r8, r2
cmp r8, #3
ble DownL3AreaRemain
DownL3AreaLoop:
vld4.16 {d0, d1, d2, d3}, [r0]! // load and transpose 4x4 matrix of int16_t
// transpose
sub r8, r8, #4
vst1.16 {d0}, [r1]!
vst1.16 {d1}, [r5]!
vst1.16 {d2}, [r6]!
cmp r8, #4
bge DownL3AreaLoop

cmp r8, #0
beq DownL3AreaRemainEnd
DownL3AreaRemain:
vld1.16 {d0}, [r0]!

vst1.16 {d0[0]}, [r1]!
vst1.16 {d0[1]}, [r5]!
vst1.16 {d0[2]}, [r6]!

subs r8, r8, #1
bne DownL3AreaRemain

DownL3AreaRemainEnd:
sub r3, r3, #3


DownL2:
cmp r3, #1
ble DownL1
add r5, r1, r4
mov r8, r2
cmp r8, #3
ble DownL2AreaRemain
DownL2AreaLoop:
vld4.16 {d0, d1, d2, d3}, [r0]! // load and transpose 4x4 matrix of int16_t
// transpose
vst1.16 {d0}, [r1]!
vst1.16 {d1}, [r5]!
sub r8, r8, #4
cmp r8, #4
bge DownL2AreaLoop

cmp r8, #0
beq DownL2AreaRemainEnd
DownL2AreaRemain:
vld1.16 {d0}, [r0]!
vst1.16 {d0[0]}, [r1]!
vst1.16 {d0[1]}, [r5]!

subs r8, r8, #1
bne DownL2AreaRemain

DownL2AreaRemainEnd:
sub r3, r3, #2

DownL1:
cmp r3, #0
beq DownEnd
mov r8, r2
cmp r8, #3
ble DownL1AreaRemain
DownL1AreaLoop:
vld4.16 {d0, d1, d2, d3}, [r0]! // load and transpose 4x4 matrix of int16_t
// transpose
vst1.16 {d0}, [r1]!
sub r8, r8, #4
cmp r8, #4
bge DownL1AreaLoop

cmp r8, #0
beq DownL1AreaRemainEnd
DownL1AreaRemain:
vld1.16 {d0}, [r0]!

vst1.16 {d0[0]}, [r1]!
subs r8, r8, #1
bne DownL1AreaRemain

DownL1AreaRemainEnd:

DownEnd:



pop {r4-r8, r10, pc}



#endif
#endif
