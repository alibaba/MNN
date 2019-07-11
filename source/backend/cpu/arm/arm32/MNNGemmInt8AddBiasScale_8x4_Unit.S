//
//  MNNGemmInt8AddBiasScale_8x4_Unit.S
//  MNN
//
//  Created by MNN on 2019/06/03.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __arm__
#ifndef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNGemmInt8AddBiasScale_8x4_Unit

// void MNNGemmInt8AddBiasScale_8x4_Unit(int8_t* dst, const int8_t* src, const int8_t* weight, const int32_t* bias, const float* scale, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad)

//Auto: r0: dst*, r1: src*, r2:weight*, r3: bias*
// Load from sp: r4: scale*, r5: src_depth_quad, r6: dst_step, r7: dst_depth_quad

push {r4, r5, r6, r7, r8, lr}

ldr r4, [sp, #24]
ldr r5, [sp, #28]
ldr r6, [sp, #32]
ldr r7, [sp, #36]

vpush {q4-q7}

L2LoopDz:
    mov r8, r1
    subs r12, r5, #1
    vld1.8 {q4}, [r1]!
    vld1.8 {q5, q6}, [r2]!
    // q8-q15 mul-add result
    vmull.s8 q0, d8, d10
    vpaddl.s16 q8, q0
    vmull.s8 q1, d8, d11
    vpaddl.s16 q9, q1
    vmull.s8 q0, d8, d12
    vpaddl.s16 q10, q0
    vmull.s8 q1, d8, d13
    vpaddl.s16 q11, q1
    vmull.s8 q0, d9, d10
    vpaddl.s16 q12, q0
    vmull.s8 q1, d9, d11
    vpaddl.s16 q13, q1
    vmull.s8 q0, d9, d12
    vpaddl.s16 q14, q0
    vmull.s8 q1, d9, d13
    vpaddl.s16 q15, q1
    beq L2LoopSzEnd

    subs r12, r12, #1
    vld1.8 {q4}, [r1]!
    vld1.8 {q5, q6}, [r2]!
    vmull.s8 q0, d8, d10
    beq L2LoopSzEndAdd

    L2LoopSz:
        vpadal.s16 q8, q0
        vmull.s8 q1, d8, d11
        vmull.s8 q2, d8, d12
        vpadal.s16 q9, q1
        vmull.s8 q3, d8, d13
        vpadal.s16 q10, q2
        vmull.s8 q0, d9, d10
        vpadal.s16 q11, q3
        vmull.s8 q1, d9, d11
        vpadal.s16 q12, q0
        vld1.8 {q5}, [r2]!
        vmull.s8 q0, d9, d12
        vpadal.s16 q13, q1
        vmull.s8 q2, d9, d13
        vpadal.s16 q14, q0
        vld1.8 {q6}, [r2]!
        vpadal.s16 q15, q2

        vld1.8 {q4}, [r1]!
        vmull.s8 q0, d8, d10

        subs r12, r12, #1
        bne L2LoopSz


    
    L2LoopSzEndAdd:
    vpadal.s16 q8, q0
    vmull.s8 q1, d8, d11
    vpadal.s16 q9, q1
    vmull.s8 q0, d8, d12
    vpadal.s16 q10, q0
    vmull.s8 q1, d8, d13
    vpadal.s16 q11, q1
    vmull.s8 q0, d9, d10
    vpadal.s16 q12, q0
    vmull.s8 q1, d9, d11
    vpadal.s16 q13, q1
    vmull.s8 q0, d9, d12
    vpadal.s16 q14, q0
    vmull.s8 q1, d9, d13
    vpadal.s16 q15, q1
    
    L2LoopSzEnd:

    vld1.s32 {q0}, [r3]!
    vld1.f32 {q1}, [r4]!

    vpadd.s32 d16, d16, d17
    vpadd.s32 d20, d20, d21
    vpadd.s32 d18, d18, d19
    vpadd.s32 d22, d22, d23

    vpadd.s32 d24, d24, d25
    vpadd.s32 d28, d28, d29
    vpadd.s32 d26, d26, d27
    vpadd.s32 d30, d30, d31

    // q8,q9
    vpadd.s32 d16, d16, d18
    vpadd.s32 d17, d20, d22
    vpadd.s32 d18, d24, d26
    vaddq.s32 q8, q8, q0
    vcvt.f32.s32 q8, q8
    vmulq.f32 q8, q8, q1
    vcvt.s32.f32 q8, q8
    vqmovn.s32 d4, q8

    vpadd.s32 d19, d28, d30
    vaddq.s32 q9, q9, q0
    vcvt.f32.s32 q9, q9
    vmulq.f32 q9, q9, q1
    vcvt.s32.f32 q9, q9

    vqmovn.s32 d5, q9

    vqmovn.s16 d6, q2
    
    vst1.s8 d6, [r0], r6

    subs r7, r7, #1
    mov r1, r8
    bne L2LoopDz



vpop {q4-q7}
pop {r4, r5, r6, r7, r8, pc}



#endif
#endif