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

asm_function NEON_MNNGelu_BF16
//void NEON_MNNGelu_BF16(int16_t* dst, const int16_t* src, size_t size, float* parameters);

//Auto Load:
//r0:dst, r1:src, r2:size


push {r4-r8, r10, r11, lr}
vpush {q4-q7}

cmp r2, #0
beq GeluEnd

ldr lr, [r3, #0]      // r3: 0.044715f
ldr r4, [r3, #4]       // r4: 0.79788458f
ldr r5, [r3, #8]       // r5: 378.f
ldr r6, [r3, #12]      // r6: 17325.f
ldr r7, [r3, #16]      // r7: 135135.f
ldr r8, [r3, #20]      // r8: 28.f
ldr r10, [r3, #24]     // r10: 3150.f
ldr r11, [r3, #28]     // r11: 62370.f


vdup.32 q15, lr        //q15: [0.044715f]x4
vdup.32 q14, r4        //q16: [0.79788458f]x4
vdup.32 q13, r5        //q13: [378.f]x4
vdup.32 q12, r6        //q12: [17325.f]x4
vdup.32 q11, r7        //q11: [135135.f]x4
vdup.32 q10, r8        //q10: [28.f]x4
vdup.32 q9, r10        //q9: [3150.f]x4
vdup.32 q8, r11        //q8: [62370.f]x4

mov r4, #5
mov r5, #-5

GeluZLoop:

vld1.16 {q0}, [r1]!   // q0: 8* sizeof(int16_t)

vshll.s16 q1, d1, #16     // shift left long of each int16_t as float32
vshll.s16 q0, d0, #16

vmul.f32 q2, q0, q0
vmul.f32 q3, q1, q1
vmul.f32 q2, q2, q0
vmul.f32 q3, q3, q1

vmul.f32 q2, q2, q15
vadd.f32 q2, q2, q0
vmul.f32 q3, q3, q15
vadd.f32 q3, q3, q1

vmul.f32 q2, q2, q14
vmul.f32 q3, q3, q14

vdup.32 q7, r4
vdup.32 q6, r5
vcvt.f32.s32 q7, q7
vcvt.f32.s32 q6, q6
vmax.f32 q2, q2, q6
vmax.f32 q3, q3, q6
vmin.f32 q2, q2, q7
vmin.f32 q3, q3, q7

// tanh(value)
vmul.f32 q4, q2, q2     // q4: value*value
vmul.f32 q5, q3, q3     // q5: value*value
// a
vadd.f32 q6, q4, q13
vadd.f32 q7, q5, q13
vmul.f32 q6, q6, q4
vmul.f32 q7, q7, q5
vadd.f32 q6, q6, q12
vadd.f32 q7, q7, q12
vmul.f32 q6, q6, q4
vmul.f32 q7, q7, q5
vadd.f32 q6, q6, q11
vadd.f32 q7, q7, q11
vmul.f32 q6, q6, q2
vmul.f32 q7, q7, q3
//b
vmul.f32 q2, q4, q10
vmul.f32 q3, q5, q10
vadd.f32 q2, q2, q9
vadd.f32 q3, q3, q9
vmul.f32 q2, q2, q4
vmul.f32 q3, q3, q5
vadd.f32 q2, q2, q8
vadd.f32 q3, q3, q8
vmul.f32 q2, q2, q4
vmul.f32 q3, q3, q5
vadd.f32 q2, q2, q11
vadd.f32 q3, q3, q11
//a/b
vdiv.f32 s24, s24, s8
vdiv.f32 s25, s25, s9
vdiv.f32 s26, s26, s10
vdiv.f32 s27, s27, s11
vdiv.f32 s28, s28, s12
vdiv.f32 s29, s29, s13
vdiv.f32 s30, s30, s14
vdiv.f32 s31, s31, s15

// border case
vmov.f32 q2, #1.0
vmov.f32 q3, #-1.0
vmov.f32 q4, #0.5
vmin.f32 q6, q6, q2
vmin.f32 q7, q7, q2
vmax.f32 q6, q6, q3
vmax.f32 q7, q7, q3
// tanh(value)

vadd.f32 q6, q6, q2
vadd.f32 q7, q7, q2
vmul.f32 q6, q6, q0
vmul.f32 q7, q7, q1
vmul.f32 q6, q6, q4
vmul.f32 q7, q7, q4

vshrn.i32 d12, q6, #16    // shift right 16bit of each float32 as int16_t.
vshrn.i32 d13, q7, #16

vst1.16 {q6}, [r0]!

subs r2, r2, #1
bne GeluZLoop


GeluEnd:
vpop {q4-q7}
pop {r4-r8, r10, r11, pc}

#endif
#endif
