//
//  MNNScaleBias2FloatC4.S
//  MNN
//
//  Created by MNN on 2018/07/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __arm__
#ifndef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNScaleBias2FloatC4
//void MNNScaleBias2FloatC4(float* dst, const int16_t* src, const float* alpha, const float* bias, size_t sizeQuad)

//Auto:
//r0:dst, r1:src, r2:alpha, r3:bias

push {r4, lr}

//Load:
//r4:sizeQuad
ldr r4, [sp, #8]
mov r12, #16

vld1.32 {q15}, [r3]//Bias
vld1.32 {q14}, [r2]//Alpha

L4:
cmp r4, #4
blt L1

L4Loop:
    vld1.16 {d0}, [r1], r12
    vmovl.s16 q2, d0
    vld1.16 {d1}, [r1], r12
    vmovl.s16 q3, d1
    vcvt.f32.s32 q2, q2
    vld1.16 {d2}, [r1], r12
    vcvt.f32.s32 q3, q3
    vld1.16 {d3}, [r1], r12
    vmovl.s16 q8, d2
    vmul.f32 q2, q2, q14
    vmul.f32 q3, q3, q14
    vmovl.s16 q9, d3
    vadd.f32 q2, q2, q15
    vadd.f32 q3, q3, q15
    vcvt.f32.s32 q8, q8
    vst1.32 {q2, q3}, [r0]!
    vcvt.f32.s32 q9, q9
    vmul.f32 q8, q8, q14
    vmul.f32 q9, q9, q14
    vadd.f32 q8, q8, q15
    vadd.f32 q9, q9, q15
    vst1.32 {q8, q9}, [r0]!
    sub r4, r4, #4
    cmp r4, #4
    bge L4Loop


L1:
cmp r4, #0
beq EndBasic

Loop1:
    vld1.16 {d0}, [r1], r12
    vmovl.s16 q0, d0
    vmov.s32 q1, q15
    vcvt.f32.s32 q0, q0
    vmla.f32 q1, q0, q14

    vst1.32 {q1}, [r0]!
    subs r4, r4, #1
    bne Loop1


EndBasic:

pop {r4, pc}

asm_function MNNScaleBias2FloatC4Relu
//void MNNScaleBias2FloatC4(float* dst, const int16_t* src, const float* alpha, const float* bias, size_t sizeQuad)

//Auto:
//r0:dst, r1:src, r2:alpha, r3:bias

push {r4, lr}

//Load:
//r4:sizeQuad
ldr r4, [sp, #8]
mov r12, #16

vld1.32 {q15}, [r3]//Bias
vmov.i32 q13, #0
vld1.32 {q14}, [r2]//Alpha

L4Relu:
cmp r4, #4
blt L1Relu

L4ReluLoop:
    vld1.16 {d0}, [r1], r12
    vmovl.s16 q2, d0
    vld1.16 {d1}, [r1], r12
    vmovl.s16 q3, d1
    vcvt.f32.s32 q2, q2
    vld1.16 {d2}, [r1], r12
    vcvt.f32.s32 q3, q3
    vld1.16 {d3}, [r1], r12
    vmovl.s16 q8, d2
    vmul.f32 q2, q2, q14
    vmul.f32 q3, q3, q14
    vmovl.s16 q9, d3

    vadd.f32 q2, q2, q15
    vadd.f32 q3, q3, q15
    vcvt.f32.s32 q8, q8
    vmax.f32 q2, q2, q13
    vmax.f32 q3, q3, q13
    vcvt.f32.s32 q9, q9
    vst1.32 {q2, q3}, [r0]!
    vmul.f32 q8, q8, q14
    vmul.f32 q9, q9, q14
    vadd.f32 q8, q8, q15
    vadd.f32 q9, q9, q15
    vmax.f32 q8, q8, q13
    vmax.f32 q9, q9, q13
    vst1.32 {q8, q9}, [r0]!
    sub r4, r4, #4
    cmp r4, #4
    bge L4ReluLoop


L1Relu:
cmp r4, #0
beq EndBasicRelu

Loop1Relu:
    vld1.16 {d0}, [r1], r12
    vmovl.s16 q0, d0
    vmov.s32 q1, q15
    vcvt.f32.s32 q0, q0
    vmla.f32 q1, q0, q14
    vmax.f32 q1, q1, q13

    vst1.32 {q1}, [r0]!
    subs r4, r4, #1
    bne Loop1Relu


EndBasicRelu:

pop {r4, pc}

#endif
#endif
