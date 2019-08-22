//
//  MNNScaleAddInt8.S
//  MNN
//
//  Created by MNN on 2019/08/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __arm__
#ifndef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNScaleAddInt8
// MNNScaleAddInt8(int8_t* dst, const int8_t* src0, const int8_t* src1, 
//  const float* scale0, const float* scale1, const float* outputScale, const size_t size)
// Auto Load
// r0: dst, r1:src0, r2:src1, r3: scale0
// Load from sp
// r4:scale1, r5: outputScale, r6:size
push {r4, r5, r6, r7, r8, lr}

ldr r4, [sp, #24]
ldr r5, [sp, #28]
ldr r6, [sp, #32]

vpush {q4-q7}

vld1.32 {q13}, [r3]
vld1.32 {q14}, [r4]
vld1.32 {q15}, [r5]

L4:
cmp r6, #4
blt L1

L4Loop:
    vld1.32 {q11}, [r1]!
    sub r6, r6, #4
    vmovl.s8 q4, d22
    vmovl.s8 q5, d23
    
    vmovl.s16 q0, d8
    vmovl.s16 q1, d9
    vmovl.s16 q2, d10
    vmovl.s16 q3, d11

    vld1.32 {q12}, [r2]!

    vcvtq.f32.s32 q0, q0
    vcvtq.f32.s32 q1, q1
    vcvtq.f32.s32 q2, q2
    vcvtq.f32.s32 q3, q3

    vmovl.s8 q6, d24
    vmovl.s8 q7, d25

    vmulq.f32 q0, q0, q13
    vmulq.f32 q1, q1, q13
    vmulq.f32 q2, q2, q13
    vmulq.f32 q3, q3, q13

    vmovl.s16 q8, d12
    vmovl.s16 q9, d13
    vmovl.s16 q10, d14
    vmovl.s16 q11, d15
    vcvtq.f32.s32 q8, q8
    vcvtq.f32.s32 q9, q9
    vcvtq.f32.s32 q10, q10
    vcvtq.f32.s32 q11, q11
    vmulq.f32 q8, q8, q14
    vmulq.f32 q9, q9, q14
    vmulq.f32 q10, q10, q14
    vmulq.f32 q11, q11, q14

    vaddq.f32 q0, q0, q8
    vaddq.f32 q1, q1, q9
    vaddq.f32 q2, q2, q10
    vaddq.f32 q3, q3, q11

    vmulq.f32 q0, q0, q15
    vmulq.f32 q1, q1, q15
    vmulq.f32 q2, q2, q15
    vmulq.f32 q3, q3, q15

    vcvtq.s32.f32 q0, q0
    vcvtq.s32.f32 q1, q1
    vcvtq.s32.f32 q2, q2
    vcvtq.s32.f32 q3, q3

    vqmovn.s32 d8, q0
    vqmovn.s32 d9, q1
    vqmovn.s32 d10, q2
    vqmovn.s32 d11, q3
    
    vqmovn.s16 d0, q4
    vqmovn.s16 d1, q5
    cmp r6, #4
    vst1.32 {q0}, [r0]!
    bge L4Loop

L1:
cmp r6, #0
beq End

L1Loop:
    vld1.32 {d0[0]}, [r1]!
    subs r6, r6, #1
    vmovl.s8 q1, d0
    vmovl.s16 q2, d2
    vcvtq.f32.s32 q3, q2
    vmulq.f32 q3, q3, q13

    vld1.32 {d8[0]}, [r2]!
    vmovl.s8 q5, d8
    vmovl.s16 q6, d10
    vcvtq.f32.s32 q7, q6
    vmulq.f32 q7, q7, q14

    vaddq.f32 q3, q3, q7

    vmulq.f32 q3, q3, q15
    vcvtq.s32.f32 q0, q3
    vqmovn.s32 d2, q0
    vqmovn.s16 d6, q1
    vst1.32 {d6[0]}, [r0]!
    bne L1Loop
End:
vpop {q4-q7}
pop {r4, r5, r6, r7, r8, pc}

#endif
#endif