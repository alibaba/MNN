//
//  MNNCoefLine.S
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

asm_function MNNCoefLine
//void MNNCoefLine(float* dst, const float* prepare, float* t, size_t number)

push {lr}
//Auto: r0:dst, r1:prepare, r2:t, r3:number
vpush {q4-q7}

vld1.32 {d30[0]}, [r2]
vdup.32 q15, d30[0]//q15:t

cmp r3, #12
blt MNNCoefLineL1

sub r3, r3, #12

vld2.32 {q0, q1}, [r1]!
cmp r3, #12
vld2.32 {q2, q3}, [r1]!
vtrn.32 q0, q2
vld2.32 {q8, q9}, [r1]!
vtrn.32 q1, q3
vld2.32 {q10, q11}, [r1]!
vmla.f32 q1, q0, q15

vld2.32 {q4, q5}, [r1]!
vtrn.32 q8, q10
vld2.32 {q6, q7}, [r1]!

blt MNNCoefLineLoopL12End

MNNCoefLineLoopL12:

vmla.f32 q2, q1, q15
vtrn.32 q9, q11
vmla.f32 q3, q2, q15
vmla.f32 q9, q8, q15
vtrn.32 d6, d7
vtrn.32 q4, q6
vtrn.32 q5, q7
vmla.f32 q5, q4, q15
vst1.32 {q3}, [r0]!
vmla.f32 q10, q9, q15
vmla.f32 q6, q5, q15
vld2.32 {q0, q1}, [r1]!
vmla.f32 q11, q10, q15
vld2.32 {q2, q3}, [r1]!
vmla.f32 q7, q6, q15
vld2.32 {q8, q9}, [r1]!
vtrn.32 d22, d23
vtrn.32 d14, d15
vst1.32 {q11}, [r0]!
vtrn.32 q0, q2
vld2.32 {q10, q11}, [r1]!
vtrn.32 q1, q3
vst1.32 {q7}, [r0]!

vmla.f32 q1, q0, q15

vld2.32 {q4, q5}, [r1]!
vtrn.32 q8, q10
vld2.32 {q6, q7}, [r1]!


sub r3, r3, #12
cmp r3, #12
bge MNNCoefLineLoopL12


MNNCoefLineLoopL12End:

vmla.f32 q2, q1, q15
vtrn.32 q9, q11
vmla.f32 q3, q2, q15
vmla.f32 q9, q8, q15
vtrn.32 d6, d7
vtrn.32 q4, q6
vtrn.32 q5, q7
vmla.f32 q5, q4, q15
vst1.32 {q3}, [r0]!
vmla.f32 q10, q9, q15
vmla.f32 q6, q5, q15
vmla.f32 q11, q10, q15
vmla.f32 q7, q6, q15
vtrn.32 d22, d23
vtrn.32 d14, d15
vst1.32 {q11}, [r0]!
vst1.32 {q7}, [r0]!


MNNCoefLineL1:
cmp r3, #0
beq MNNCoefLineEnd

MNNCoefLineLoopL1:
vld1.32 {q8}, [r1]!
vdup.32 q0, d16[0]
vdup.32 q1, d16[1]
vdup.32 q2, d17[0]
vdup.32 q3, d17[1]

vmla.f32 q1, q0, q15
vmla.f32 q2, q1, q15
vmla.f32 q3, q2, q15

vst1.32 {d6[0]}, [r0]!
subs r3, r3, #1
bne MNNCoefLineLoopL1

MNNCoefLineEnd:

vpop {q4-q7}

pop {pc}

#endif
#endif
