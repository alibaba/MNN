//
//  MNNGemmInt8AddBiasScale_16x4_Unit.S
//  MNN
//
//  Created by MNN on 2020/03/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __arm__
#ifndef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNGemmInt8AddBiasScale_16x4_Unit_FAST

// void MNNGemmInt8AddBiasScale_16x4_Unit_FAST(int8_t* dst, const int8_t* src, 
//          const int8_t* weight, const int32_t* bias, const float* scale, 
//          size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad)

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
    vld1.8 {q0}, [r1]! // input
    vld1.8 {q2,q3}, [r2]! // weight
    vmull.s8 q8, d0, d4
    vld1.8 {q4,q5}, [r2]!
    vmull.s8 q9, d0, d6
    vld1.8 {q1}, [r1]!
    vmull.s8 q10, d0, d8
    subs r12, r5, #1
    vmull.s8 q11, d0, d10
    //----------------
    vmull.s8 q12, d2, d4
    vmull.s8 q13, d2, d6
    vmull.s8 q14, d2, d8
    vmull.s8 q15, d2, d10
    beq L2LoopSzEnd
    
    L2LoopSz:
        //*****acc*****
        vmlal.s8 q8,  d1, d5
        vmlal.s8 q9,  d1, d7
        vmlal.s8 q10, d1, d9
        vmlal.s8 q11, d1, d11
        //----------------
        vld1.8 {q0}, [r1]!
        vmlal.s8 q12, d3, d5
        vmlal.s8 q13, d3, d7
        vld1.8 {q2,q3}, [r2]!
        vmlal.s8 q14, d3, d9
        vmlal.s8 q15, d3, d11
        vld1.8 {q4,q5}, [r2]!
        
        vmlal.s8 q8, d0, d4
        vmlal.s8 q9, d0, d6
        vld1.8 {q1}, [r1]!
        vmlal.s8 q10, d0, d8
        vmlal.s8 q11, d0, d10
        vmlal.s8 q12, d2, d4
        vmlal.s8 q13, d2, d6
        vmlal.s8 q14, d2, d8
        vmlal.s8 q15, d2, d10

        subs r12, r12, #1
        bne L2LoopSz

    L2LoopSzEnd:

    // ------------------acc
    vmlal.s8 q8,  d1, d5
    vmlal.s8 q9,  d1, d7
    vmlal.s8 q10, d1, d9
    vmlal.s8 q11, d1, d11
    vmlal.s8 q12, d3, d5
    vmlal.s8 q13, d3, d7
    vmlal.s8 q14, d3, d9
    vmlal.s8 q15, d3, d11

    vpaddl.s16 q0, q8
    vpaddl.s16 q1, q9
    vpaddl.s16 q2, q10
    vpaddl.s16 q3, q11
    vpaddl.s16 q4, q12
    vpaddl.s16 q5, q13
    vpaddl.s16 q6, q14
    vpaddl.s16 q7, q15

    vld1.s32 {q14}, [r3]!
    vpadd.s32 d20, d0, d1
    vpadd.s32 d21, d2, d3
    vld1.f32 {q15}, [r4]!
    vpadd.s32 d22, d4, d5
    vpadd.s32 d23, d6, d7
    vpadd.s32 d24, d8, d9
    vpadd.s32 d25, d10, d11
    vpadd.s32 d26, d12, d13
    vpadd.s32 d27, d14, d15

    // q8,q9
    vpadd.s32 d16, d20, d21
    vpadd.s32 d17, d22, d23
    vpadd.s32 d18, d24, d25
    vpadd.s32 d19, d26, d27

    vaddq.s32 q0, q8, q14
    vaddq.s32 q1, q9, q14
    
    vcvt.f32.s32 q0, q0
    vcvt.f32.s32 q1, q1
    vmulq.f32 q0, q0, q15
    vmulq.f32 q1, q1, q15

    vcvtr.s32.f32 s0, s0
    vcvtr.s32.f32 s1, s1
    vcvtr.s32.f32 s2, s2
    vcvtr.s32.f32 s3, s3
    
    vcvtr.s32.f32 s4, s4
    vcvtr.s32.f32 s5, s5
    vcvtr.s32.f32 s6, s6
    vcvtr.s32.f32 s7, s7
    
    vqmovn.s32 d4, q0
    vqmovn.s32 d5, q1

    vqmovn.s16 d6, q2
    
    vst1.s8 d6, [r0], r6

    subs r7, r7, #1
    mov r1, r8
    bne L2LoopDz

vpop {q4-q7}
pop {r4, r5, r6, r7, r8, pc}

#endif
#endif