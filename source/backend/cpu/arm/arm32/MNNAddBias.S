//
//  MNNAddBias.S
//  MNN
//
//  Created by MNN on 2019/02/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __arm__
#ifndef __aarch64__
#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNAddBias
//void MNNAddBias(float* dst, const float* bias, int planeNumber, int biasNumber)
//r0:dst, r1:bias, r2:planeNumber, r3:biasNumber
push {r4, r5, lr}

cmp r3, #0
beq End

cmp r2, #0
beq End

LoopBias:
vld1.32 {q15}, [r1]!

mov r4, r2

L4:
cmp r4, #3
ble L1
Loop4:
mov r5, r0
vld1.32 {q0, q1}, [r5]!
vadd.f32 q0, q0, q15
vld1.32 {q2, q3}, [r5]
vadd.f32 q1, q1, q15
vadd.f32 q2, q2, q15
vst1.32 {q0, q1}, [r0]!
vadd.f32 q3, q3, q15
vst1.32 {q2, q3}, [r0]!
sub r4, r4, #4
cmp r4, #4
bge Loop4

L1:
cmp r4, #0
beq EndLoopPlane
Loop1:
vld1.32 {q0}, [r0]
vadd.f32 q0, q0, q15
subs r4, r4, #1
vst1.32 {q0}, [r0]!
bne Loop1

EndLoopPlane:

subs r3, r3, #1
bne LoopBias


End:


pop {r4, r5, pc}

#endif
#endif
