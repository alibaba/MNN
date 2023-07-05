//
//  MNNCubicLineC16.s
//  ALL_BUILD
//
//  Created by MNN on 2023/4/12.
//

#ifdef __arm__
#ifndef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

.macro _vroundq_f32 plus minus x
vcgt.f32 q12, \x, #0
vbsl.f32 q12, \plus, \minus
vadd.f32 q13, q12, \x
vcvt.s32.f32 \x, q13
.endm

asm_function MNNCubicLineC16
// void MNNCubicLineC16(int8_t* dst, const float* A, const float* B, const float* C, const float* D, float* t, int8_t* zeroPoint,
//                     size_t number, ssize_t minValue, ssize_t maxValue);
// Auto load: r0: dst, r1: A, r2: B, r3: C
// r4:  D, r11: t, r5: zeroPoint, lr: number, r6:minValue, r7:maxValue

push {r4-r8, r10-r11, lr}
ldr r4, [sp, #32]
ldr r11, [sp, #36]
ldr r5, [sp, #40]
ldr lr, [sp, #44]
ldr r6, [sp, #48]
ldr r7, [sp, #52]
vpush {q4-q7}

cmp lr, #0
beq END
ldr r10, [r11, #0]

L1Loop:
//B
vld1.32 {q3, q4}, [r2]!
vld1.32 {q5, q6}, [r2]!
//C
vld1.32 {q10, q11}, [r3]!
vld1.32 {q12, q13}, [r3]!

// Caculate b0,c0
vmov.f32 s0, #-2.25
vmov.f32 s1, #1.25
vmov.f32 s5, #1.0   
vmov.f32 d1[0], r10   // s2: t


vmul.f32 s3, s2, s2 // t*t
vmul.f32 s4, s3, s2 // t*t*t
vmul.f32 s3, s3, s0 // -2.25*t^2
vmla.f32 s3, s4, s1 // 1.25*t^3
vadd.f32 s3, s5, s3 // s3: b0

vsub.f32 s6, s5, s2 // s6: 1-t
vmul.f32 s7, s6, s6 // (1-t)^2
vmul.f32 s8, s7, s6 // (1-t)^3
vmul.f32 s8, s8, s1
vmla.f32 s8, s7, s0
vadd.f32 s8, s5, s8 //s8: c0

vmul.f32 q10, q10, d4[0]
vmul.f32 q11, q11, d4[0]
vmul.f32 q12, q12, d4[0]
vmul.f32 q13, q13, d4[0]
vmla.f32 q10, q3, d1[1]
vmla.f32 q11, q4, d1[1]
vmla.f32 q12, q5, d1[1]
vmla.f32 q13, q6, d1[1]

//A
vld1.32{q3, q4}, [r1]!
vld1.32{q5, q6}, [r1]!

// Caculate a0, d0
vmov.f32 d1[0], r10    // s2: t
vmov.f32 s5, #1.0
vsub.f32 s6, s5, s2

vmov.f32 s0, #-0.75
vmov.f32 s1, #3.75
vmov.f32 s3, #3.0
vadd.f32 s2, s2, s5 // s2: 1+t
vadd.f32 s6, s6, s5 // s6: 2-t

vmov.f32 s5, #-6.0   
vmul.f32 s4, s2, s2 // s4: (1+t)^2
vmul.f32 s7, s2, s4 // s7: (1+t)^3
vmul.f32 s7, s7, s0
vmla.f32 s7, s4, s1
vmla.f32 s7, s2, s5
vadd.f32 s7, s7, s3 // s7: a0

vmul.f32 s8, s6, s6 // s8: (2-t)^2
vmul.f32 s9, s8, s6 // s9: (2-t)^3
vmul.f32 s9, s9, s0
vmla.f32 s9, s8, s1
vmla.f32 s9, s6, s5
vadd.f32 s9, s9, s3 // s9: d0

vmla.f32 q10, q3, d3[1]
vmla.f32 q11, q4, d3[1]
vmla.f32 q12, q5, d3[1]
vmla.f32 q13, q6, d3[1]

// D
vld1.32 {q3, q4}, [r4]!
vld1.32{q5, q6}, [r4]!

vmla.f32 q10, q3, d4[1]
vmla.f32 q11, q4, d4[1]
vmla.f32 q12, q5, d4[1]
vmla.f32 q13, q6, d4[1]

vmov.f32 q1, #0.5
vmov.f32 q2, #-0.5
vdup.s8  q7, r7
vdup.s8  q8, r6
vld1.8 {d0[0]}, [r5]
vdup.8 d0, d0[0]

_vroundq_f32 q1, q2, q10
_vroundq_f32 q1, q2, q11
_vroundq_f32 q1, q2, q12
_vroundq_f32 q1, q2, q13

vqmovn.s32 d20, q10
vqmovn.s32 d21, q11
vqmovn.s32 d22, q12
vqmovn.s32 d23, q13

vaddw.s8 q10, q10, d0
vaddw.s8 q11, q11, d0

vqmovn.s16 d20, q10
vqmovn.s16 d21, q11

vmax.s8 q10, q10, q8
vmin.s8 q10, q10, q7

vst1.8 {q10}, [r0]!

sub lr, lr, #1
cmp lr, #1
bge L1Loop

END:
vpop {q4-q7}
pop {r4-r8, r10-r11, pc}

#endif
#endif