//
//  MNNFloat2Int8.S
//  MNN
//
//  Created by MNN on 2019/01/22.
//  Copyright © 2018, Alibaba Group Holding Limited
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

asm_function MNNFloat2Int8
//void MNNFloat2Int8(const float* src, int8_t* dst, size_t sizeQuad, float* scale, ssize_t aMin, ssize_t aMax, float* zeroPoint, ssize_t quanParamVec);
// Auto load: r0:src, r1:dst, r2:sizeQuad, r3:scale
// Load from sp: aMin, aMax, lr: zeroPoint, r12: quanParamVec
push {lr}

vmov.f32 q10, #0.5
vmov.f32 q11, #-0.5
vmov.s32 q1, #1
// scale
vld1.32 {d30[0]}, [r3]
vdup.32 q15, d30[0]
// min
ldr r12, [sp, #4]
vdup.s8 d28, r12
// max
ldr r12, [sp, #8]
vdup.s8 d29, r12
// zeropoint
ldr lr, [sp, #12]
vld1.32 {d18[0]}, [lr]
vdup.32 q9, d18[0]

// quanParamVec
ldr r12, [sp, #16]
cmp r12, #3
bne LOAD_VEC_ZERO
vld1.f32 {q9}, [lr] // load vector zero
vld1.f32 {q15}, [r3] // load vector scale
b COMPUTE

LOAD_VEC_ZERO:
cmp r12, #2
bne LOAD_VEC_SCALE
vld1.f32 {q9}, [lr] // load vector zero
b COMPUTE

LOAD_VEC_SCALE:
cmp r12, #1
bne COMPUTE
vld1.f32 {q15}, [r3] // load vector scale


COMPUTE:
cmp r2, #3
ble FL1

FLLoop4:
vld1.32 {q0, q1}, [r0]!
vmul.f32 q0, q0, q15
vadd.f32 q0, q0, q9
vmul.f32 q1, q1, q15
vadd.f32 q1, q1, q9
vld1.32 {q2, q3}, [r0]!
// vcvtr.s32.f32 s0, s0
// vcvtr.s32.f32 s1, s1
// vcvtr.s32.f32 s2, s2
// vcvtr.s32.f32 s3, s3
// vcvtr.s32.f32 s4, s4
// vcvtr.s32.f32 s5, s5
// vcvtr.s32.f32 s6, s6
// vcvtr.s32.f32 s7, s7
_vroundq_f32 q10, q11, q0
_vroundq_f32 q10, q11, q1
vmul.f32 q2, q2, q15
vadd.f32 q2, q2, q9
vmul.f32 q3, q3, q15
vadd.f32 q3, q3, q9
// vcvtr.s32.f32 s8, s8
// vcvtr.s32.f32 s9, s9
// vcvtr.s32.f32 s10, s10
// vcvtr.s32.f32 s11, s11
// vcvtr.s32.f32 s12, s12
// vcvtr.s32.f32 s13, s13
// vcvtr.s32.f32 s14, s14
// vcvtr.s32.f32 s15, s15
_vroundq_f32 q10, q11, q2
_vroundq_f32 q10, q11, q3

vqmovn.s32 d0, q0
vqmovn.s32 d1, q1
vqmovn.s32 d2, q2
vqmovn.s32 d3, q3

vqmovn.s16 d0, q0
vqmovn.s16 d1, q1
vmax.s8 d0, d0, d28
vmax.s8 d1, d1, d28
vmin.s8 d0, d0, d29
vmin.s8 d1, d1, d29

vst1.32 {q0}, [r1]!

sub r2, r2, #4
cmp r2, #4
bge FLLoop4


FL1:
cmp r2, #0
beq FLEnd

FLLoop1:
vld1.32 {q0}, [r0]!
vmul.f32 q0, q0, q15
vadd.f32 q0, q0, q9
// vcvtr.s32.f32 s0, s0
// vcvtr.s32.f32 s1, s1
// vcvtr.s32.f32 s2, s2
// vcvtr.s32.f32 s3, s3
_vroundq_f32 q10, q11, q0

vqmovn.s32 d0, q0
vqmovn.s16 d0, q0
vmax.s8 d0, d0, d28
vmin.s8 d0, d0, d29

vst1.32 {d0[0]}, [r1]!

subs r2, r2, #1
bne FLLoop1

FLEnd:
pop {pc}
#endif
#endif
