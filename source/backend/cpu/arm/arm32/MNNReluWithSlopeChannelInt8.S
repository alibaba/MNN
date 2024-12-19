//
//  MNNReluWithSlopeChannel.S
//  MNN
//
//  Created by MNN on 2019/02/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//

/*
    struct QuanPrePostParameters{
    float* inputScale;
    float* outputScale;
    ssize_t* inputZeroPoint;
    ssize_t* outputZeroPoint;
    ssize_t minValue;
    ssize_t maxValue;
};
 */

#ifdef __arm__
#ifndef __aarch64__
#include "MNNAsmGlobal.h"

.text
.align 5


asm_function MNNReluWithSlopeChannelInt8
// MNNReluWithSlopeChannelInt8(int8_t* dst, const int8_t* src, const float* slope, size_t planeNumber, size_t depthQuad, QuanPrePostParameters *params)
// Auto load:
// r0: dst, r1: src, r2: slope, r3: planeNumber
// Load from sp:
// r4: depthQuad, r5: params
// Load from r5:  r8: inputZeroPoint, r6: outputZeroPoint, r10: minValue, r11: maxValue
push {r4-r8, r10-r11, lr}
ldr r4, [sp, #32]
ldr r5, [sp, #36]

vpush {q4-q7}

ldr r8, [r5, #8]
ldr r6, [r5, #12]
ldr r10, [r5, #16]
ldr r11, [r5, #20]

cmp r4, #0
beq PReluEnd
cmp r3, #0
beq PReluEnd

.macro ROUND_TWO x0, x1
    vmov.f32 q12, #0.5
    vmov.f32 q13, #-0.5
    vcgt.f32 q10, \x0, #0
    vcgt.f32 q11, \x1, #0
    vbsl.f32 q10, q12, q13
    vbsl.f32 q11, q12, q13
    vadd.f32 \x0, q10, \x0
    vadd.f32 \x1, q11, \x1
    vcvt.s32.f32 \x0, \x0
    vcvt.s32.f32 \x1, \x1
.endm

.macro ROUND_ONE x0
    vmov.f32 q12, #0.5
    vmov.f32 q13, #-0.5
    vcgt.f32 q10, \x0, #0
    vbsl.f32 q10, q12, q13
    vadd.f32 \x0, q10, \x0
    vcvt.s32.f32 \x0, \x0
.endm

vld1.8 d30[0], [r8]
vld1.8 d31[0], [r6]
vdup.8 d30, d30[0]  // inputZeroPoint
vdup.8 d31, d31[0]  // outputZeroPoint

ldr r6, [r5, #0]    // inputScale
ldr r8, [r5, #4]    // outputScale

PReluZLoop:
vld1.32 {q14}, [r2]!

mov r5, r3
cmp r5, #3

ble PReluL1

PReluL4Loop:
vld1.8 {q0}, [r1]!
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
// *input_scale
vld1.f32 {d14[0]}, [r6]
vld1.f32 {d14[1]}, [r8] // outputscale
vmul.f32 q3, q3, d14[0]
vmul.f32 q4, q4, d14[0]
vmul.f32 q5, q5, d14[0]
vmul.f32 q6, q6, d14[0]

vclt.f32 q0, q3, #0
vclt.f32 q1, q4, #0
vclt.f32 q2, q5, #0
vclt.f32 q12, q6, #0

// *slope
vmul.f32 q8, q3, q14
vmul.f32 q9, q4, q14
vmul.f32 q10, q5, q14
vmul.f32 q11, q6, q14

vbit.32 q3, q8, q0
vbit.32 q4, q9, q1
vbit.32 q5, q10, q2
vbit.32 q6, q11, q12

vmul.f32 q3, q3, d14[1]
vmul.f32 q4, q4, d14[1]
vmul.f32 q5, q5, d14[1]
vmul.f32 q6, q6, d14[1]

ROUND_TWO q3, q4
ROUND_TWO q5, q6

vdup.8 q10, r10
vdup.8 q11, r11

vqmovn.s32 d14, q3
vqmovn.s32 d15, q4
vqmovn.s32 d16, q5
vqmovn.s32 d17, q6
vaddw.s8 q7, q7, d31
vaddw.s8 q8, q8, d31
vqmovn.s16 d18, q7
vqmovn.s16 d19, q8
vmax.s8 q9, q9, q10
vmin.s8 q9, q9, q11

vst1.8 {q9}, [r0]!

sub r5, r5, #4
cmp r5, #4
bge PReluL4Loop

PReluL1:
cmp r5, #0
beq PReluL1End

PReluL1Loop:
vld1.32 {d0[0]}, [r1]!
vmovl.s8 q1, d0
vsubw.s8 q1, q1, d30

vmovl.s16 q2, d2

vcvt.f32.s32 q2, q2
// *input_scale
vld1.f32 {d14[0]}, [r6]
vld1.f32 {d14[1]}, [r8] // outputscale
vmul.f32 q2, q2, d14[0]
vclt.f32 q4, q2, #0     // index
// *slope
vmul.f32 q3, q2, q14
vbit q2, q3, q4
// *output_scale
vmul.f32 q2, q2, d14[1]

ROUND_ONE q2

vqmovn.s32 d4, q2
vaddw.s8 q2, q2, d31
vqmovn.s16 d4, q2

vbit.8 d0, d4, d10
vst1.32 {d0[0]}, [r0]!

subs r5, r5, #1
bne PReluL1Loop

PReluL1End:

subs r4, r4, #1
bne PReluZLoop


PReluEnd:
vpop {q4-q7}
pop {r4-r8, r10-r11, pc}

#endif
#endif
