//
//  MNNReluWithSlope.S
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

asm_function MNNReluWithSlope
//void MNNReluWithSlope(float* dst, const float* src, size_t sizeQuad, float slope)

//Auto Load:
//r0:dst, r1:src, r2:sizeQuad, r3:slope

push {lr}

vdup.f32 q15, r3
vmov.i32 q14, #0

ReluL4:
cmp r2, #3
ble ReluL1

ReluL4Loop:
vld1.32 {q0, q1}, [r1]!

vmul.f32 q8, q15, q0
vmul.f32 q9, q15, q1
vcle.f32 q12, q0, q14

vld1.f32 {q2, q3}, [r1]!

vcle.f32 q13, q1, q14

vbit.32 q0, q8, q12
vbit.32 q1, q9, q13
vmul.f32 q10, q2, q15

vst1.32 {q0, q1}, [r0]!

vcle.f32 q12, q2, q14
vcle.f32 q13, q3, q14
vmul.f32 q11, q3, q15
vbit.32 q2, q10, q12
vbit.32 q3, q11, q13

vst1.32 {q2, q3}, [r0]!

sub r2, r2, #4
cmp r2, #4
bge ReluL4Loop

ReluL1:
cmp r2, #0
beq ReluEnd

ReluL1Loop:
vld1.32 {q0}, [r1]!
vcle.f32 q2, q0, q14
vmul.f32 q1, q0, q15
vbit.32 q0, q1, q2
vst1.32 {q0}, [r0]!
subs r2, r2, #1
bne ReluL1Loop


ReluEnd:


pop {pc}


#endif
#endif
