//
//  MNNMatrixProd.S
//  MNN
//
//  Created by MNN on 2019/02/12.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __arm__
#ifndef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNMatrixProd
//void MNNMatrixProd(float* C, const float* A, const float* B, size_t widthC4, size_t cStride, size_t aStride, size_t bStride, size_t height)

push {r4-r8, r10, r11, lr} // avoid to touch platform-register r-9

//Auto: r0: C, r1:A, r2:B, r3:widthC4

//Load from sp:
//r4:cStride, r5:aStride, r6:bStride, r7:height

ldr r4, [sp, #32]
ldr r5, [sp, #36]
ldr r6, [sp, #40]
ldr r7, [sp, #44]

mov r12, #4 //sizeof(float)
mul r4, r12, r4
mul r5, r12, r5
mul r6, r12, r6

LoopY:
mov r8, r0
mov lr, r1
mov r10, r2

mov r11, r3

L4:
cmp r11, #4
blt L1
sub r11, r11, #4
vld1.32 {q0, q1}, [r1]!
cmp r11, #4
vld1.32 {q2, q3}, [r2]!

vmul.f32 q0, q0, q2
vld1.32 {q8, q9}, [r1]!

blt L4LoopEnd

L4Loop:
vmul.f32 q1, q1, q3
vld1.32 {q10, q11}, [r2]!
vst1.32 {q0, q1}, [r0]!
vmul.f32 q8, q8, q10
vmul.f32 q9, q9, q11

vld1.32 {q0, q1}, [r1]!
vst1.32 {q8, q9}, [r0]!

vld1.32 {q2, q3}, [r2]!

vmul.f32 q0, q0, q2
vld1.32 {q8, q9}, [r1]!

sub r11, r11, #4
cmp r11, #4
bge L4Loop

L4LoopEnd:
vmul.f32 q1, q1, q3
vld1.32 {q10, q11}, [r2]!
vst1.32 {q0, q1}, [r0]!
vmul.f32 q8, q8, q10
vmul.f32 q9, q9, q11
vst1.32 {q8, q9}, [r0]!

L1:
cmp r11, #0
beq EndLine

L1Loop:
vld1.32 {q0}, [r1]!
vld1.32 {q1}, [r2]!
vmul.f32 q0, q0, q1
vst1.32 {q0}, [r0]!
subs r11, r11, #1
bne L1Loop

EndLine:
add r0, r8, r4
add r1, lr, r5
add r2, r10, r6

subs r7, r7, #1
bne LoopY


pop {r4-r8, r10, r11, pc}

#endif
#endif
