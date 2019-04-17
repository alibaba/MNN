//
//  MNNAddBiasRelu.S
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

asm_function MNNAddBiasRelu
//void MNNAddBiasRelu(float* dst, const float* bias, int planeNumber, int biasNumber)
//r0:dst, r1:bias, r2:planeNumber, r3:biasNumber
push {r4, r5, lr}

cmp r3, #0
beq BiasReluEnd

cmp r2, #0
beq BiasReluEnd

vmov.i32 q14, #0
ReluLoopBias:
vld1.32 {q15}, [r1]!

mov r4, r2

ReluBiasReluL4:
cmp r4, #3
ble BiasReluL1
ReluLoop4:
mov r5, r0
vld1.32 {q0, q1}, [r5]!
vadd.f32 q0, q0, q15
vadd.f32 q1, q1, q15
vld1.32 {q2, q3}, [r5]
vmax.f32 q0, q0, q14
vmax.f32 q1, q1, q14
vadd.f32 q2, q2, q15
vst1.32 {q0, q1}, [r0]!
vmax.f32 q2, q2, q14
vadd.f32 q3, q3, q15
vmax.f32 q3, q3, q14
vst1.32 {q2, q3}, [r0]!
sub r4, r4, #4
cmp r4, #4
bge ReluLoop4

BiasReluL1:
cmp r4, #0
beq EndReluLoopPlane
ReluLoop1:
vld1.32 {q0}, [r0]
vadd.f32 q0, q0, q15
vmax.f32 q0, q0, q14
subs r4, r4, #1
vst1.32 {q0}, [r0]!
bne ReluLoop1

EndReluLoopPlane:

subs r3, r3, #1
bne ReluLoopBias


BiasReluEnd:


pop {r4, r5, pc}

#endif
#endif
