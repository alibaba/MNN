//
//  MNNExpFP16.S
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
//void MNNExpFP16(FLOAT16* dest, const FLOAT16* source, const FLOAT16* parameters, size_t block)
asm_function MNNExpFP16

//r0: dest, r1:source, r2:parameters, r3:countC8
push {r4, lr}
vpush {q5, q6}

vld1.32 {q0, q1}, [r2]

vmov.i32 q2, #87
vcvt.f32.s32 q2, q2
vneg.f32 q3, q2

Loop:

vld1.32 {q8, q9}, [r1]!

vmin.f32 q8, q8, q2
vmin.f32 q9, q9, q2
vmax.f32 q8, q8, q3
vmax.f32 q9, q9, q3

vneg.f32 q10, q8
vneg.f32 q11, q9

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

vst1.32 {q12, q13}, [r0]!


subs r3, r3, #1
bne Loop

vpop {q5, q6}
pop {r4, pc}


#endif
#endif
