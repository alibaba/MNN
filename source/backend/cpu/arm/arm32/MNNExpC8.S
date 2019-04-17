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
//void MNNExpC8(float* dest, const float* source, const float* parameters, size_t countC8)
asm_function MNNExpC8

//r0: dest, r1:source, r2:parameters, r3:countC8
push {r4, lr}
vpush {q5, q6}

vld1.32 {q0, q1}, [r2]
vmov.i32 q2, #1
vcvt.f32.s32 q3, q2

vmov.i32 q5, #16
vneg.s32 q6, q5

Loop:

vld1.32 {q8, q9}, [r1]!

vneg.f32 q10, q8
vneg.f32 q11, q9


vmul.f32 q8, q10, d0[1]
vmul.f32 q9, q11, d0[1]
vcvt.s32.f32 q8, q8
vcvt.s32.f32 q9, q9

vcvt.f32.s32 q12, q8
vcvt.f32.s32 q13, q9

vmin.s32 q8, q8, q5
vmin.s32 q9, q9, q5
vmax.s32 q8, q8, q6
vmax.s32 q9, q9, q6

//q10, q11: t
vmls.f32 q10, q12, d0[0]
vmls.f32 q11, q13, d0[0]

vmul.f32 q12, q10, d3[1]
vdup.32 q14, d3[0]
vmul.f32 q13, q11, d3[1]

vadd.f32 q12, q12, q14
vadd.f32 q13, q13, q14

.macro MLA_TWO z0
vmul.f32 q12, q12, q10
vdup.32 q14, \z0
vmul.f32 q13, q13, q11
vadd.f32 q12, q12, q14
vadd.f32 q13, q13, q14
.endm

MLA_TWO d2[1]
MLA_TWO d2[0]
MLA_TWO d1[1]
MLA_TWO d1[0]

//q12, q13 is expRemain

//Compute expBasic
vcle.s32 q10, q8, #0
vcle.s32 q11, q9, #0

vneg.s32 q14, q8
vneg.s32 q15, q9

vshl.s32 q14, q2, q14
vshl.s32 q15, q2, q15

vshl.s32 q8, q2, q8
vshl.s32 q9, q2, q9

vcvt.f32.s32 q14, q14
vcvt.f32.s32 q15, q15
vcvt.f32.s32 q8, q8
vcvt.f32.s32 q9, q9
vrecpe.f32 q14, q14
vrecpe.f32 q15, q15

vbit.32 q8, q14, q10
vbit.32 q9, q15, q11

vmul.f32 q8, q8, q12
vmul.f32 q9, q9, q13

vst1.32 {q8, q9}, [r0]!


subs r3, r3, #1
bne Loop

vpop {q5, q6}
pop {r4, pc}


#endif
#endif
