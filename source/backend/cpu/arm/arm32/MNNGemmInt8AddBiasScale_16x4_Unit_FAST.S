//
//  MNNGemmInt8AddBiasScale_16x4_Unit_FAST.S
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

//struct QuanPostTreatParameters {
//    const float* scale;
//    const int32_t* bias;
//    int32_t maxValue;
//    int32_t minValue;
//    int32_t useInt8;
//};

//void MNNGemmInt8AddBiasScale_16x4_Unit_FAST(int8_t* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad, size_t dst_step,
//                                              size_t dst_depth_quad, const QuanPostTreatParameters* post, size_t remain) {

//Auto: r0: dst*, r1: src*, r2:weight*, r3: src_depth_quad
// Load from sp: r4: dst_step, r5: dst_depth_quad, r6: post, r10: real
// Load from post: r8: scale, lr: bias, r7: maxValue, r6: minValue

push {r4-r8, r10, lr} // avoid to touch platform-register r-9

ldr r4, [sp, #28]
ldr r5, [sp, #32]
ldr r6, [sp, #36]
ldr r10, [sp, #40]

ldr r8, [r6, #0]
ldr lr, [r6, #4]

vpush {q4-q7}

// Only int8 output use this kernel.

ldr r12, [r6, #28] // srcKernelSum
str r12, [sp, #4]
ldr r12, [r6, #32] // weightBias
str r12, [sp, #8]
ldr r12, [r6, #36] // f32minmax
str r12, [sp, #12]
ldr r12, [r6, #8] // int8 max
str r12, [sp, #16]
ldr r12, [r6, #12] // int8 min
str r12, [sp, #20]

Start:
cmp r10, #2
blt L1LoopDz

L2LoopDz:
    mov r10, r1
    vld1.8 {q0}, [r1]! // input
    vld1.8 {q2,q3}, [r2]! // weight
    vmull.s8 q8, d0, d4
    vld1.8 {q4,q5}, [r2]!
    vmull.s8 q9, d0, d6
    vld1.8 {q1}, [r1]!
    vmull.s8 q10, d0, d8
    subs r12, r3, #1
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
    
    L2Quan:
    vld1.f32 {q14}, [lr]! // bias
    vld1.f32 {q15}, [r8]! // scale
    vpadd.s32 d20, d0, d1
    vpadd.s32 d21, d2, d3
    
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

    //vaddq.s32 q0, q8, q14 // add bias
    //vaddq.s32 q1, q9, q14

    vcvt.f32.s32 q0, q8
    vcvt.f32.s32 q1, q9
    vmulq.f32 q0, q0, q15 // mul scale
    vmulq.f32 q1, q1, q15

    ldr r6, [sp, #4] // srcKernelSum
    vld1.f32 {d12[0]}, [r6]! // tile 0
    vld1.f32 {d12[1]}, [r6] // tile 1
    ldr r6, [sp, #8] // weightBias
    vld1.f32 {q7}, [r6]!
    str r6, [sp, #8] // update next 4 weightBias

    vmla.f32 q0, q7, d12[0] // add srcKernelSum x weightBias
    vmla.f32 q1, q7, d12[1]

    vadd.f32 q0, q0, q14  // add bias
    vadd.f32 q1, q1, q14


    L2QuanUseInt8:
    vmov.f32 q10, #0.5
    vmov.f32 q11, #-0.5
    ldr r6, [sp, #16]
    vdup.32 q2, r6 // max
    ldr r6, [sp, #20]
    vdup.32 q3, r6 // min

    vcgt.f32 q12, q0, #0
    vcgt.f32 q13, q1, #0
    vbsl.f32 q12, q10, q11
    vbsl.f32 q13, q10, q11
    vadd.f32 q0, q12, q0
    vadd.f32 q1, q13, q1
    vcvt.s32.f32 q0, q0
    vcvt.s32.f32 q1, q1

    vmin.s32 q0, q2, q0
    vmin.s32 q1, q2, q1
    vmax.s32 q0, q3, q0
    vmax.s32 q1, q3, q1

    vqmovn.s32 d4, q0
    vqmovn.s32 d5, q1

    vqmovn.s16 d6, q2

    vst1.s8 {d6}, [r0], r4
L2LoopCheck:
    subs r5, r5, #1
    mov r1, r10
    bne L2LoopDz
b End

L1LoopDz:
    mov r10, r1
    vld1.8 {q0}, [r1]! // input
    vld1.8 {q2,q3}, [r2]! // weight
    vmull.s8 q8, d0, d4
    vld1.8 {q4,q5}, [r2]!
    vmull.s8 q9, d0, d6
    add r1, r1, #16
    vmull.s8 q10, d0, d8
    subs r12, r3, #1
    vmull.s8 q11, d0, d10

    beq L1LoopSzEnd

    L1LoopSz:
        //*****acc*****
        vmlal.s8 q8,  d1, d5
        vmlal.s8 q9,  d1, d7
        vmlal.s8 q10, d1, d9
        vmlal.s8 q11, d1, d11
        //----------------
        vld1.8 {q0}, [r1]!
        vld1.8 {q2,q3}, [r2]!
        vld1.8 {q4,q5}, [r2]!

        vmlal.s8 q8, d0, d4
        vmlal.s8 q9, d0, d6
        add r1, r1, #16
        vmlal.s8 q10, d0, d8
        vmlal.s8 q11, d0, d10

        subs r12, r12, #1
        bne L1LoopSz

    L1LoopSzEnd:

    // ------------------acc
    vmlal.s8 q8,  d1, d5
    vmlal.s8 q9,  d1, d7
    vmlal.s8 q10, d1, d9
    vmlal.s8 q11, d1, d11

    vpaddl.s16 q0, q8
    vpaddl.s16 q1, q9
    vpaddl.s16 q2, q10
    vpaddl.s16 q3, q11
    
    L1Quan:
    vld1.f32 {q14}, [lr]!
    vpadd.s32 d20, d0, d1
    vpadd.s32 d21, d2, d3
    vld1.f32 {q15}, [r8]!
    vpadd.s32 d22, d4, d5
    vpadd.s32 d23, d6, d7

    // q8,q9
    vpadd.s32 d16, d20, d21
    vpadd.s32 d17, d22, d23

    //vaddq.s32 q0, q8, q14

    vcvt.f32.s32 q0, q8
    vmulq.f32 q0, q0, q15

    ldr r6, [sp, #4] // srcKernelSum
    vld1.f32 {d12[0]}, [r6] // tile 0
    ldr r6, [sp, #8] // weightBias
    vld1.f32 {q7}, [r6]!
    str r6, [sp, #8] // update next 4 weightBias
    vmla.f32 q0, q7, d12[0]
    vadd.f32 q0, q0, q14 // add bias

    L1QuanUseInt8:
    vmov.f32 q10, #0.5
    vmov.f32 q11, #-0.5
    ldr r6, [sp, #16]
    vdup.32 q3, r6 // max
    ldr r6, [sp, #20]
    vdup.32 q2, r6 // min
    vcgt.f32 q12, q0, #0
    vbsl.f32 q12, q10, q11
    vbsl.f32 q13, q10, q11
    vadd.f32 q0, q12, q0
    vcvt.s32.f32 q0, q0

    vmax.s32 q0, q2, q0
    vmin.s32 q0, q3, q0

    vqmovn.s32 d4, q0

    vqmovn.s16 d6, q2

    vst1.s32 {d6[0]}, [r0], r4
L1LoopCheck:
    subs r5, r5, #1
    mov r1, r10
    bne L1LoopDz

End:
vpop {q4-q7}
pop {r4-r8, r10, pc}

#endif
#endif
