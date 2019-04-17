//
//  MNNReluWithSlopeChannel.S
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

asm_function MNNReluWithSlopeChannel
//void MNNReluWithSlopeChannel(float* dst, const float* src, const float* slope, size_t sizeQuad, size_t depthQuad)

//Auto Load:
//r0:dst, r1:src, r2:slope, r3:sizeQuad

//Load from sp
//r4:depthQuad

push {r4,r5, lr}
ldr r4, [sp, #12]

cmp r4, #0
beq PReluEnd
cmp r3, #0
beq PReluEnd


PReluZLoop:
vld1.32 {q15}, [r2]!
mov r5, r3
cmp r5, #3
ble PReluL1

PReluL4Loop:
vld1.32 {q0, q1}, [r1]!

vcle.f32 q12, q0, #0
vcle.f32 q13, q1, #0

vld1.32 {q2, q3}, [r1]!

vmul.f32 q8, q0, q15
vmul.f32 q9, q1, q15
vbit.32 q0, q8, q12
vbit.32 q1, q9, q13

vmul.f32 q8, q2, q15
vmul.f32 q9, q3, q15

vst1.32 {q0, q1}, [r0]!

vcle.f32 q12, q2, #0
vcle.f32 q13, q3, #0
vbit.32 q2, q8, q12
vbit.32 q3, q9, q13

vst1.32 {q2, q3}, [r0]!
sub r5, r5, #4
cmp r5, #4
bge PReluL4Loop

PReluL1:
cmp r5, #0

beq PReluL1End

PReluL1Loop:
vld1.32 {q0}, [r1]!
vcle.f32 q2, q0, #0
vmul.f32 q1, q0, q15
vbit.32 q0, q1, q2
vst1.32 {q0}, [r0]!
subs r5, r5, #1
bne PReluL1Loop

PReluL1End:

subs r4, r4, #1
bne PReluZLoop


PReluEnd:

pop {r4, r5, pc}

#endif
#endif
