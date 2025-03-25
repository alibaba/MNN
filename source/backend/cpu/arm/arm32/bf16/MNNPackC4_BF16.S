//
//  MNNPackC4_BF16.S
//  MNN
//
//  Created by MNN on 2021/02/26.
//  Copyright © 2018-2021 Alibaba Group Holding Limited
//




#ifdef __arm__
#ifndef __aarch64__

#include "MNNAsmGlobal.h"
.text
.align 5

// .macro transpose
// vtrn.16 d0, d1
// vtrn.16 d2, d3
// vswp d0[2-3], d1[2-3] // should swap high half of d-vector, the half is 32-bit. there is no instruction, we use vst4.16 instead
// vswp d2[2-3], d3[2-3]
// .endm

asm_function MNNPackC4_BF16
// treate float pointer as int16_t*
//void MNNPackC4_BF16(float* dst, const float* src, size_t area, size_t depth, int32_t* areaOffset)
//Auto load:
//r0:dst, r1:src, r2:area, r3:depth

push {r4-r8, r10, lr} // avoid to touch platform-register r-9

ldr lr, [sp, #28]
ldr r10, [lr, #4]
ldr lr, [lr, #0]

mul r4, r2, r3
cmp r4, #0
beq UpEnd

//r4: srcDepthOffset:srcArea*sizeof(float)
mov r4, #4
mul r4, lr, r4

//r10 -> 4 * (dstArea * sizeof(int16_t) - area * sizeof(int16_t))
mov r12, #8
sub r10, r10, r2
mul r10, r12, r10

//lr -> (srcArea * sizeof(float) - area * sizeof(float))
mov r12, #4
sub lr, lr, r2
mul lr, r12, lr

UpL4:
cmp r3, #3
ble UpL3

UpL4Loop:
add r5, r1, r4
add r6, r4, r5
add r7, r4, r6
mov r8, r2
cmp r8, #3
ble UpL4AreaRemain
UpL4AreaLoop:
vld1.32 {q0}, [r1]! // load 4 elements of 16-bit into 64bit vector register d0
vld1.32 {q1}, [r5]!
vld1.32 {q2}, [r6]!
vld1.32 {q3}, [r7]!
vshrn.i32 d0, q0, #16
vshrn.i32 d1, q1, #16
vshrn.i32 d2, q2, #16
vshrn.i32 d3, q3, #16

// transpose // no suitable instruction to transpose int16_t type
vst4.16 {d0, d1, d2, d3}, [r0]!
sub r8, r8, #4
cmp r8, #4
bge UpL4AreaLoop

UpL4AreaRemain:
cmp r8, #0
beq UpL4AreaRemainEnd
UpL4AreaRemainLoop:
vld1.32 {d0[0]}, [r1]!
vld1.32 {d0[1]}, [r5]!
vld1.32 {d1[0]}, [r6]!
vld1.32 {d1[1]}, [r7]!
vshrn.i32 d0, q0, #16

vst1.16 {d0}, [r0]!

subs r8, r8, #1
bne UpL4AreaRemainLoop
UpL4AreaRemainEnd:
sub r3, r3, #4
add r1, r7, lr
cmp r3, #4
add r0, r10, r0
bge UpL4Loop

UpL3:
cmp r3, #2
ble UpL2
add r5, r1, r4
add r6, r4, r5
mov r8, r2
cmp r8, #3
ble UpL3AreaRemain
UpL3AreaLoop:
vld1.32 {q0}, [r1]!
vld1.32 {q1}, [r5]!
vld1.32 {q2}, [r6]!
vshrn.i32 d0, q0, #16
vshrn.i32 d1, q1, #16
vshrn.i32 d2, q2, #16
vmov.i16 d3, #0

// transpose // no suitable instruction to transpose int16_t type
vst4.16 {d0, d1, d2, d3}, [r0]!
sub r8, r8, #4
cmp r8, #4
bge UpL3AreaLoop

cmp r8, #0
beq UpL3AreaRemainEnd
UpL3AreaRemain:
vmov.i32 q0, #0
vld1.32 {d0[0]}, [r1]!
vld1.32 {d0[1]}, [r5]!
vld1.32 {d1[0]}, [r6]!
vshrn.i32 d0, q0, #16

vst1.16 {d0}, [r0]!

subs r8, r8, #1
bne UpL3AreaRemain

UpL3AreaRemainEnd:
sub r3, r3, #3


UpL2:
cmp r3, #1
ble UpL1
add r5, r1, r4
mov r8, r2
cmp r8, #3
ble UpL2AreaRemain
UpL2AreaLoop:
vld1.32 {q0}, [r1]!
vld1.32 {q1}, [r5]!
vshrn.i32 d0, q0, #16
vshrn.i32 d1, q1, #16

vmov.i16 d3, #0
vmov.i16 d2, #0
vst4.16 {d0, d1, d2, d3}, [r0]!
sub r8, r8, #4
cmp r8, #4
bge UpL2AreaLoop

cmp r8, #0
beq UpL2AreaRemainEnd
UpL2AreaRemain:
vmov.i32 q0, #0
vld1.32 {d0[0]}, [r1]!
vld1.32 {d0[1]}, [r5]!

vshrn.i32 d0, q0, #16

vst1.16 {d0}, [r0]!

subs r8, r8, #1
bne UpL2AreaRemain

UpL2AreaRemainEnd:
sub r3, r3, #2

UpL1:
cmp r3, #0
beq UpEnd
mov r8, r2
cmp r8, #3
ble UpL1AreaRemain
UpL1AreaLoop:
vld1.32 {q0}, [r1]!
vshrn.i32 d0, q0, #16
vmov.i16 d3, #0
vmov.i16 d1, #0
vmov.i16 d2, #0
// transpose // no suitable instruction to transpose int16_t type
vst4.16 {d0, d1, d2, d3}, [r0]!
sub r8, r8, #4
cmp r8, #4
bge UpL1AreaLoop

cmp r8, #0
beq UpL1AreaRemainEnd
UpL1AreaRemain:
vmov.i16 q0, #0
vld1.32 {d0[0]}, [r1]!
vshrn.i32 d0, q0, #16

vst1.16 {d0}, [r0]!

subs r8, r8, #1
bne UpL1AreaRemain

UpL1AreaRemainEnd:

UpEnd:

pop {r4-r8, r10, pc}

#endif
#endif
