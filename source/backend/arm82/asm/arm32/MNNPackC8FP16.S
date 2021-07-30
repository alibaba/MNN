//
//  MNNPackC8FP16.S
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
vtrn.16 q0, q1
vtrn.16 q2, q3
vtrn.16 q8, q9
vtrn.16 q10, q11
vtrn.32 q0, q2
vtrn.32 q1, q3
vtrn.32 q8, q10
vtrn.32 q9, q11

vswp d1, d16
vswp d3, d18
vswp d5, d20
vswp d7, d22

.endm

asm_function MNNPackC8FP16_C8
//void MNNPackC8FP16_C8(float* dst, const float* src, size_t area, size_t depth, int32_t* areaOffset)
//Auto load:
//r0:dst, r1:src, r2:area, r3:depth
//r9: srcArea, r10: dstArea

push {r4, r5, r6, r7, r8, r9, r10, r11, lr}
ldr r9, [sp, #36]
ldr r10, [r9, #4]
ldr r9, [r9, #0]

mul r4, r2, r3
cmp r4, #0
beq UpEnd

//r4: srcDepthOffset:srcArea*sizeof(int16)
mov r4, #2
mul r4, r9, r4

//r10 -> 8 * (dstArea * sizeof(int16) - area * sizeof(int16))
mov r12, #16
sub r10, r10, r2
mul r10, r12, r10

//r9 -> (srcArea * sizeof(int16) - area * sizeof(int16))
mov r12, #2
sub r9, r9, r2
mul r9, r12, r9

// 8 * sizeof(int16)
vmov.i32 q14, #16
vmov.i32 q13, #2

UpL8:
//cmp r3, #8
//ble UpL7

UpL8Loop:
add r5, r1, r4
add r6, r4, r5
add r7, r4, r6
add r12, r4, r7
vmov.i32 d30[0], r12
add r11, r4, r12
vmov.i32 d30[1], r11
add r12, r4, r11
vmov.i32 d31[0], r12
add r11, r4, r12
vmov.i32 d31[1], r11

mov r8, r2
cmp r8, #8
ble UpL8AreaRemain
UpL8AreaLoop:
vld1.32 {q0}, [r1]!
vld1.32 {q1}, [r5]!
vld1.32 {q2}, [r6]!
vld1.32 {q3}, [r7]!
vmov.i32 r11, d30[0]
vmov.i32 r12, d30[1]
vld1.32 {q8}, [r11]
vld1.32 {q9}, [r12]
vmov.i32 r11, d31[0]
vmov.i32 r12, d31[1]
vld1.32 {q10}, [r11]
vld1.32 {q11}, [r12]
vadd.i32 q15, q14, q15

transpose
vst1.32 {q0, q1}, [r0]!
sub r8, r8, #8
vst1.32 {q2, q3}, [r0]!
vst1.32 {q8, q9}, [r0]!
vst1.32 {q10, q11}, [r0]!
cmp r8, #8
bge UpL8AreaLoop

UpL8AreaRemain:
cmp r8, #0
beq UpL8AreaRemainEnd
UpL8AreaRemainLoop:
vld1.16 {d0[0]}, [r1]!
vld1.16 {d0[1]}, [r5]!
vld1.16 {d0[2]}, [r6]!
vld1.16 {d0[3]}, [r7]!
vmov.i32 r11, d30[0]
vmov.i32 r12, d30[1]
vld1.16 {d1[0]}, [r11]
vld1.16 {d1[1]}, [r12]
vmov.i32 r11, d31[0]
vmov.i32 r12, d31[1]
vld1.16 {d1[2]}, [r11]
vld1.16 {d1[3]}, [r12]
vadd.i32 q15, q13, q15

vst1.32 {q0}, [r0]!

subs r8, r8, #1
bne UpL8AreaRemainLoop
UpL8AreaRemainEnd:
vmov.i32 r7, d31[1]
sub r3, r3, #8
add r1, r7, r9
cmp r3, #8
add r0, r10, r0
bge UpL8Loop

UpEnd:
pop {r4, r5, r6, r7, r8, r9, r10, r11, pc}

#endif
#endif
