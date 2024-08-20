//
//  MNNExpC8.S
//  MNN
//
//  Created by MNN on 2019/01/18.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __arm__
#ifndef __aarch64__

#include "MNNAsmGlobal.h"
.text
.align 5
//void MNNExpC8(float* dest, const float* source, const float* offset, const float* parameters, size_t countC8)
asm_function MNNExpC8

//r0: dest, r1:source, r2: offset, r3:parameters, r4:countC8
push {r4, r5, lr}
ldr r4, [sp, #12]
vpush {q4, q5, q6, q7}
vmov.i32 q7, #0
ldr r5, [r2, #0]
vdup.32 q4, r5 // Alpha
ldr r5, [r2, #4]
vdup.32 q5, r5 // Beta
ldr r5, [r2, #8]
vdup.32 q6, r5 // Bias


vld1.32 {q0, q1}, [r3]

vmov.i32 q2, #87
vcvt.f32.s32 q2, q2
vneg.f32 q3, q2

Loop:

vld1.32 {q8, q9}, [r1]!
vmul.f32 q8, q8, q4
vmul.f32 q9, q9, q4
vadd.f32 q8, q8, q6
vadd.f32 q9, q9, q6

vmin.f32 q8, q8, q2
vmin.f32 q9, q9, q2
vmax.f32 q10, q8, q3
vmax.f32 q11, q9, q3

vmul.f32 q8, q10, d0[1]
vmul.f32 q9, q11, d0[1]
vcvt.s32.f32 q8, q8
vcvt.s32.f32 q9, q9

vcvt.f32.s32 q12, q8
vcvt.f32.s32 q13, q9

//q10, q11: t
vmls.f32 q10, q12, d0[0]
vmls.f32 q11, q13, d0[0]

.macro MLA_TWO z0 z1 z2 z3
vdup.32 \z1, \z0
vmla.f32 \z1, \z2, \z3
.endm

MLA_TWO d3[0], q12, q10, d3[1]
MLA_TWO d3[0], q13, q11, d3[1]
MLA_TWO d2[1], q14, q10, q12
MLA_TWO d2[1], q15, q11, q13
MLA_TWO d2[0], q12, q10, q14
MLA_TWO d2[0], q13, q11, q15
MLA_TWO d1[1], q14, q10, q12
MLA_TWO d1[1], q15, q11, q13
MLA_TWO d1[0], q12, q10, q14
MLA_TWO d1[0], q13, q11, q15

//q12, q13 is expRemain

vshl.i32 q8, q8, #23
vshl.i32 q9, q9, #23
vadd.i32 q12, q12, q8
vadd.i32 q13, q13, q9

vadd.f32 q12, q12, q5
vadd.f32 q13, q13, q5
vadd.f32 q7, q12, q7

vst1.32 {q12, q13}, [r0]!
vadd.f32 q7, q13, q7

subs r4, r4, #1
bne Loop
add r5, r2, #12
vld1.32 {d0[0]}, [r5]
vadd.f32 d14, d14, d15
vtrn.32 d14, d15
vadd.f32 d14, d14, d15
vadd.f32 d0, d14, d0
vst1.32 {d0[0]}, [r5]

vpop {q4, q5, q6, q7}
pop {r4, r5, pc}


#endif
#endif
