//
//  MNNInt8ScaleToFloat.S
//  MNN
//
//  Created by MNN on 2019/06/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __arm__
#ifndef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNInt8ScaleToFloat

// void MNNInt8ScaleToFloat(float* dst, const int8_t* src, const float* scale, size_t size, ssize_t zeroPoint)

push {lr}
ldr r12, [sp, #4]
vdup.s32 q13, r12
vcvt.f32.s32 q13, q13

vpush {q4-q7}

// Auto Load:
// r0: dst*, r1: src*, r2: scale*, r3: size, r4: zeroPoint

vld1.32 {q15}, [r2]

L4:
cmp r3, #4
blt L1

L4Loop:
    vld1.32 {q14}, [r1]!
    vmovl.s8 q4, d28
    vmovl.s8 q5, d29
    sub r3, #4
    vmovl.s16 q0, d8
    vmovl.s16 q2, d10
    vmovl.s16 q3, d11
    vmovl.s16 q1, d9
    vcvt.f32.s32 q0, q0 
    vsub.f32 q0, q13
    vmul.f32 q0, q15
    vcvt.f32.s32 q1, q1
    vsub.f32 q1, q13
    vmul.f32 q1, q15
    vst1.32 {q0, q1}, [r0]!
    vcvt.f32.s32 q2, q2
    vsub.f32 q2, q13
    vmul.f32 q2, q15
    vcvt.f32.s32 q3, q3
    vsub.f32 q3, q13
    vmul.f32 q3, q15
    vst1.32 {q2, q3}, [r0]!

    cmp r3, #4
    bge L4Loop

L1:
cmp r3, #0
beq End

L1Loop:
    vld1.32 {d28[0]}, [r1]!
    vmovl.s8 q4, d28
    vmovl.s16 q0, d8
    subs r3, r3, #1
    vcvt.f32.s32 q1, q0
    vsub.f32 q1, q13
    vmul.f32 q0, q1, q15
    vst1.32 {q0}, [r0]!
    bne L1Loop

End:
vpop {q4-q7}
pop {pc}


#endif
#endif