//
//  MNNAxByClampBroadcastUnit.S
//  MNN
//
//  Created by MNN on 2020/06/20.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __arm__
#ifndef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNAxByClampBroadcastUnit
//void MNNAxByClampBroadcastUnit(float* C, const float* A, const float* B, size_t width, size_t cStride, size_t aStride, size_t height, const float* parameters)
//Auto: r0: C, r1:A, r2:B, r3:width
//r4:cStride, r5:aStride, r6:height, r7:parameters
push {r4-r8, r10, r11, lr}
ldr r4, [sp, #32]
ldr r5, [sp, #36]
ldr r6, [sp, #40]
ldr r7, [sp, #44]


vld1.32 {q3}, [r7]
vdup.f32 q14, d7[0]
vdup.f32 q15, d7[1]
mov r12, #4 //sizeof(float)
mul r4, r12, r4
mul r5, r12, r5

LoopY:
mov r8, r0
mov lr, r1
vld1.32 {q13}, [r2]!

mov r11, r3

L1:
cmp r11, #0
beq EndLine

L1Loop:
vld1.32 {q0}, [r1]!
vmla.f32 q0, q13, d6[1]
vmax.f32 q0, q0, q14
vmin.f32 q0, q0, q15
vst1.32 {q0}, [r0]!
subs r11, r11, #1
bne L1Loop

EndLine:
add r0, r8, r4
add r1, lr, r5

subs r6, r6, #1
bne LoopY

pop {r4-r8, r10, r11, pc}

#endif
#endif
