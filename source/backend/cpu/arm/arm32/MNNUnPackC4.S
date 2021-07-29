//
//  MNNUnPackC4.S
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


asm_function MNNUnpackC4
//void MNNUnpackC4(float* dst, const float* src, size_t area, size_t depth, int32_t* areaOffset);
//Auto load:
//lr: srcArea, r10: dstArea

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

//r4: dstDepthOffset:dstArea*sizeof(float)
mov r4, #4
mul r4, r10, r4

//lr -> 4 * (srcArea * sizeof(float) - area * sizeof(float))
mov r12, #16
sub lr, lr, r2
mul lr, r12, lr

//r10 -> (dstArea * sizeof(float) - area * sizeof(float))
mov r12, #4
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
vld1.32 {q0, q1}, [r0]!
vld1.32 {q2, q3}, [r0]!
transpose
sub r8, r8, #4
vst1.32 {q0}, [r1]!
vst1.32 {q1}, [r5]!
vst1.32 {q2}, [r6]!
vst1.32 {q3}, [r7]!
cmp r8, #4
bge DownL4AreaLoop

DownL4AreaRemain:
cmp r8, #0
beq DownL4AreaRemainEnd
DownL4AreaRemainLoop:

vld1.32 {q0}, [r0]!

vst1.32 {d0[0]}, [r1]!
vst1.32 {d0[1]}, [r5]!
vst1.32 {d1[0]}, [r6]!
vst1.32 {d1[1]}, [r7]!

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
vld1.32 {q0, q1}, [r0]!
vld1.32 {q2, q3}, [r0]!
transpose
sub r8, r8, #4
vst1.32 {q0}, [r1]!
vst1.32 {q1}, [r5]!
vst1.32 {q2}, [r6]!
cmp r8, #4
bge DownL3AreaLoop

cmp r8, #0
beq DownL3AreaRemainEnd
DownL3AreaRemain:
vld1.32 {q0}, [r0]!

vst1.32 {d0[0]}, [r1]!
vst1.32 {d0[1]}, [r5]!
vst1.32 {d1[0]}, [r6]!

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
vld1.32 {q0, q1}, [r0]!
vld1.32 {q2, q3}, [r0]!
transpose
vst1.32 {q0}, [r1]!
vst1.32 {q1}, [r5]!
sub r8, r8, #4
cmp r8, #4
bge DownL2AreaLoop

cmp r8, #0
beq DownL2AreaRemainEnd
DownL2AreaRemain:
vld1.32 {q0}, [r0]!
vst1.32 {d0[0]}, [r1]!
vst1.32 {d0[1]}, [r5]!

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
vld1.32 {q0, q1}, [r0]!
vld1.32 {q2, q3}, [r0]!
transpose
vst1.32 {q0}, [r1]!
sub r8, r8, #4
cmp r8, #4
bge DownL1AreaLoop

cmp r8, #0
beq DownL1AreaRemainEnd
DownL1AreaRemain:
vld1.32 {q0}, [r0]!

vst1.32 {d0[0]}, [r1]!
subs r8, r8, #1
bne DownL1AreaRemain

DownL1AreaRemainEnd:

DownEnd:



pop {r4-r8, r10, pc}



#endif
#endif
