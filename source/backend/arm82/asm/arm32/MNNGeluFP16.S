//
//  MNNGelu.S
//  MNN
//
//  Created by MNN on 2023/2/27.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __arm__
#ifndef __aarch64__
#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNGeluFP16
//void MNNGeluFP16(FLOAT16* dst, const FLOAT16* src, size_t size, float* parameters);

//Auto Load:
//r0:dst, r1:src, r2:size, r3: *parameters


push {r4-r8, r10, r11, lr}
vpush {q4-q7}

cmp r2, #0
beq GeluEnd

ldr lr, [r3, #0]
ldr r4, [r3, #4]       // r4: 0.79788458f
ldr r5, [r3, #8]       // r5: 378.f
ldr r6, [r3, #12]      // r6: 17325.f
ldr r7, [r3, #16]      // r7: 135135.f
ldr r8, [r3, #20]      // r8: 28.f
ldr r10, [r3, #24]     // r10: 3150.f
ldr r11, [r3, #28]     // r11: 62370.f


vdup.16 q15, lr        //q15: [0.044715f]x8
vdup.16 q14, r4        //q16: [0.79788458f]x8
vdup.16 q13, r5        //q13: [378.f]x8
vdup.16 q12, r6        //q12: [17325.f]x8
vdup.16 q11, r7        //q11: [135135.f]x8
vdup.16 q10, r8        //q10: [28.f]x8
vdup.16 q9, r10        //q9: [3150.f]x8
vdup.16 q8, r11        //q8: [62370.f]x8

GeluZLoop:

vld1.16 {q0, q1}, [r1]!   // q0, q1: fp16x8

vmul.f16 q2, q0, q0
vmul.f16 q3, q1, q1
vmul.f16 q2, q2, q0
vmul.f16 q3, q3, q1

vmul.f16 q2, q2, q15
vadd.f16 q2, q2, q0
vmul.f16 q3, q3, q15
vadd.f16 q3, q3, q1

vmul.f16 q2, q2, q14
vmul.f16 q3, q3, q14

mov lr, #5
vdup.16 q4, lr
vcvt.f32.s32 q4, q4
mov lr, #-5
vdup.16 q5, lr
vcvt.f32.s32 q5, q5
vmax.f16 q2, q2, q5
vmin.f16 q2, q2, q4
vmax.f16 q3, q3, q5
vmin.f16 q3, q3, q4

// tanh(value)
vmul.f16 q4, q2, q2     // q4: value*value
vmul.f16 q5, q3, q3     // q5: value*value
// a
vadd.f16 q6, q4, q13
vadd.f16 q7, q5, q13
vmul.f16 q6, q6, q4
vmul.f16 q7, q7, q5
vadd.f16 q6, q6, q12
vadd.f16 q7, q7, q12
vmul.f16 q6, q6, q4
vmul.f16 q7, q7, q5
vadd.f16 q6, q6, q11
vadd.f16 q7, q7, q11
vmul.f16 q6, q6, q2
vmul.f16 q7, q7, q3
//b
vmul.f16 q2, q4, q10
vmul.f16 q3, q5, q10
vadd.f16 q2, q2, q9
vadd.f16 q3, q3, q9
vmul.f16 q2, q2, q4
vmul.f16 q3, q3, q5
vadd.f16 q2, q2, q8
vadd.f16 q3, q3, q8
vmul.f16 q2, q2, q4
vmul.f16 q3, q3, q5
vadd.f16 q2, q2, q11
vadd.f16 q3, q3, q11
//a/b
vdiv.f16 s24, s24, s8
vdiv.f16 s25, s25, s9
vdiv.f16 s26, s26, s10
vdiv.f16 s27, s27, s11
vdiv.f16 s28, s28, s12
vdiv.f16 s29, s29, s13
vdiv.f16 s30, s30, s14
vdiv.f16 s31, s31, s15

// border case
vmov.f32 q2, #1.0
vmov.f32 q3, #-1.0
vmov.f32 q4, #0.5
vcvt.f16.f32 d4, q2
vcvt.f16.f32 d6, q3
vcvt.f16.f32 d8, q4
//vmov.f16 d4, #1.0
//vmov.f16 d6, #-1.0
//vmov.f16 d8, #0.5
vdup.16 q2, d4[0]
vdup.16 q3, d6[0]
vdup.16 q4, d8[0]

vmin.f16 q6, q6, q2
vmin.f16 q7, q7, q2
vmax.f16 q6, q6, q3
vmax.f16 q7, q7, q3
// tanh(value)

vadd.f16 q6, q6, q2
vadd.f16 q7, q7, q2
vmul.f16 q6, q6, q0
vmul.f16 q7, q7, q1
vmul.f16 q6, q6, q4
vmul.f16 q7, q7, q4

vst1.16 {q6, q7}, [r0]!

subs r2, r2, #1
bne GeluZLoop


GeluEnd:
vpop {q4-q7}
pop {r4-r8, r10, r11, pc}


#endif
#endif
