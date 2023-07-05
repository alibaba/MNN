//
//  MNNCubicSampleC16.s
//  ALL_BUILD
//
//  Created by MNN on 2023/4/12.
//

#ifdef __arm__
#ifndef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNCubicSampleC16
// void MNNCubicSampleC16(const int8_t* src, float* dst, const int32_t* position, const float* factor, int8_t* zeroPoint, size_t number);
// Auto load: r0: src, r1: dst, r2: position, r3: factor
// r10, zeroPoint, r4:  number

push {r4-r8, r10, lr}
ldr r10, [sp, #28]
ldr r4, [sp, #32]
mov lr, #16
vpush {q4-q7}

cmp r4, #0
beq END

vld1.8 {d30[0]}, [r10]
vdup.8 d30, d30[0]  // zeroPoint

L1Loop:
ldr r5, [r2, #0]
ldr r6, [r2, #4]
ldr r7, [r2, #8]
ldr r8, [r2, #12]
add r2, r2, #16

mul r5, lr, r5
mul r6, lr, r6
mul r7, lr, r7
mul r8, lr, r8

add r5, r5, r0
add r6, r6, r0
add r7, r7, r0
add r8, r8, r0
//B
vld1.8 {q0}, [r6]
vmovl.s8 q1, d0
vmovl.s8 q2, d1
vsubw.s8 q1, q1, d30
vsubw.s8 q2, q2, d30
vmovl.s16 q3, d2
vmovl.s16 q4, d3
vmovl.s16 q5, d4
vmovl.s16 q6, d5
//C
vld1.8 {q7}, [r7]
vmovl.s8 q8, d14
vmovl.s8 q9, d15
vsubw.s8 q8, q8, d30
vsubw.s8 q9, q9, d30
vmovl.s16 q10, d16
vmovl.s16 q11, d17
vmovl.s16 q12, d18
vmovl.s16 q13, d19

vcvt.f32.s32 q3, q3
vcvt.f32.s32 q4, q4
vcvt.f32.s32 q5, q5
vcvt.f32.s32 q6, q6

vcvt.f32.s32 q10, q10
vcvt.f32.s32 q11, q11
vcvt.f32.s32 q12, q12
vcvt.f32.s32 q13, q13
// Caculate b0,c0
ldr r10, [r3] // factor
vmov.f32 s0, #-2.25
vmov.f32 s1, #1.25
vmov.f32 s5, #1.0   
vmov.f32 d1[0], r10    // s2: t

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
vld1.8 {q0}, [r5]
vmovl.s8 q1, d0
vmovl.s8 q2, d1
vsubw.s8 q1, q1, d30
vsubw.s8 q2, q2, d30
vmovl.s16 q3, d2
vmovl.s16 q4, d3
vmovl.s16 q5, d4
vmovl.s16 q6, d5
vcvt.f32.s32 q3, q3
vcvt.f32.s32 q4, q4
vcvt.f32.s32 q5, q5
vcvt.f32.s32 q6, q6

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
vld1.8 {q7}, [r8]
vmovl.s8 q8, d14
vmovl.s8 q9, d15
vsubw.s8 q8, q8, d30
vsubw.s8 q9, q9, d30
vmovl.s16 q3, d16
vmovl.s16 q4, d17
vmovl.s16 q5, d18
vmovl.s16 q6, d19
vcvt.f32.s32 q3, q3
vcvt.f32.s32 q4, q4
vcvt.f32.s32 q5, q5
vcvt.f32.s32 q6, q6

vmla.f32 q10, q3, d4[1]
vmla.f32 q11, q4, d4[1]
vmla.f32 q12, q5, d4[1]
vmla.f32 q13, q6, d4[1]
vst1.32 {q10, q11}, [r1]!
vst1.32 {q12, q13}, [r1]!

sub r4, r4, #1
add r3, r3, #4
cmp r4, #1
bge L1Loop
cmp r4, #0
beq END

END:
vpop {q4-q7}
pop {r4-r8, r10, pc}

#endif
#endif