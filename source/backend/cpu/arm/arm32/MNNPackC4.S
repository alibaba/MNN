//
//  MNNPackC4.S
//  MNN
//
//  Created by MNN on 2019/02/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __arm__
#ifndef __aarch64__

#include "MNNAsmGlobal.h"
.text
.align 5

.macro transpose
vtrn.32 d0, d2
vtrn.32 d1, d3
vtrn.32 d4, d6
vtrn.32 d5, d7
vswp d1, d4
vswp d3, d6
.endm

asm_function MNNPackC4
//void MNNPackC4(float* dst, const float* src, size_t area, size_t depth)
//Auto load:
//r0:dst, r1:src, r2:area, r3:depth


push {r4, r5, r6, r7, r8, lr}

mul r4, r2, r3
cmp r4, #0
beq UpEnd

//r4: srcDepthOffset:area*sizeof(float)
mov r4, #4
mul r4, r2, r4

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
vld1.32 {q0}, [r1]!
vld1.32 {q1}, [r5]!
vld1.32 {q2}, [r6]!
vld1.32 {q3}, [r7]!
transpose
vst1.32 {q0, q1}, [r0]!
sub r8, r8, #4
vst1.32 {q2, q3}, [r0]!
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

vst1.32 {q0}, [r0]!

subs r8, r8, #1
bne UpL4AreaRemainLoop
UpL4AreaRemainEnd:
sub r3, r3, #4
mov r1, r7
cmp r3, #4
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
vmov.i32 q3, #0
vld1.32 {q1}, [r5]!
vld1.32 {q2}, [r6]!
transpose
vst1.32 {q0, q1}, [r0]!
sub r8, r8, #4
vst1.32 {q2, q3}, [r0]!
cmp r8, #4
bge UpL3AreaLoop

cmp r8, #0
beq UpL3AreaRemainEnd
UpL3AreaRemain:
vmov.i32 q0, #0
vld1.32 {d0[0]}, [r1]!
vld1.32 {d0[1]}, [r5]!
vld1.32 {d1[0]}, [r6]!

vst1.32 {q0}, [r0]!

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
vmov.i32 q3, #0
vld1.32 {q1}, [r5]!
vmov.i32 q2, #0
transpose
vst1.32 {q0, q1}, [r0]!
vst1.32 {q2, q3}, [r0]!
sub r8, r8, #4
cmp r8, #4
bge UpL2AreaLoop

cmp r8, #0
beq UpL2AreaRemainEnd
UpL2AreaRemain:
vmov.i32 q0, #0
vld1.32 {d0[0]}, [r1]!
vld1.32 {d0[1]}, [r5]!

vst1.32 {q0}, [r0]!

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
vmov.i32 q3, #0
vmov.i32 q1, #0
vmov.i32 q2, #0
transpose
vst1.32 {q0, q1}, [r0]!
vst1.32 {q2, q3}, [r0]!
sub r8, r8, #4
cmp r8, #4
bge UpL1AreaLoop

cmp r8, #0
beq UpL1AreaRemainEnd
UpL1AreaRemain:
vmov.i32 q0, #0
vld1.32 {d0[0]}, [r1]!

vst1.32 {q0}, [r0]!

subs r8, r8, #1
bne UpL1AreaRemain

UpL1AreaRemainEnd:

UpEnd:

pop {r4, r5, r6, r7, r8, pc}

#endif
#endif
