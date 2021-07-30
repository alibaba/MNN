//
//  MNNGemmInt8AddBiasScale_16x4_Unit.S
//  MNN
//
//  Created by MNN on 2019/06/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __arm__
#ifndef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNGemmInt8AddBiasScale_16x4_Unit

//struct QuanPostTreatParameters {
//    const float* scale;
//    const int32_t* bias;
//    int32_t maxValue;
//    int32_t minValue;
//};

//void MNNGemmInt8AddBiasScale_16x4_Unit(int8_t* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad, size_t dst_step,
//                                              size_t dst_depth_quad, const QuanPostTreatParameters* post, size_t real) {

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
ldr r7, [r6, #8]
ldr r6, [r6, #12]

vpush {q4-q7}

cmp r10, #2
blt L1LoopDz

L2LoopDz:
    mov r10, r1

    subs r12, r3, #1
    // first four output
    vld1.8 {q2}, [r1]!
    vld1.8 {q4,q5}, [r2]!
    vmull.s8 q0, d4, d8
    vmull.s8 q1, d4, d10
    vmlal.s8 q0, d5, d9
    vmlal.s8 q1, d5, d11
    vpaddl.s16 q8, q0
    vpaddl.s16 q9, q1
    vld1.8 {q6,q7}, [r2]!

    vmull.s8 q0, d4, d12
    vmull.s8 q1, d4, d14
    vmlal.s8 q0, d5, d13
    vmlal.s8 q1, d5, d15
    vpaddl.s16 q10, q0
    vld1.8 {q3}, [r1]!
    vpaddl.s16 q11, q1
    // second four output
    vmull.s8 q0, d6, d8
    vmull.s8 q1, d6, d10
    vmlal.s8 q0, d7, d9
    vmlal.s8 q1, d7, d11
    vpaddl.s16 q12, q0
    vpaddl.s16 q13, q1

    vmull.s8 q0, d6, d12
    vmull.s8 q1, d6, d14
    vmlal.s8 q0, d7, d13
    vmlal.s8 q1, d7, d15
    vpaddl.s16 q14, q0
    vpaddl.s16 q15, q1

    beq L2LoopSzEnd

    L2LoopSz:
        // first four output
        vld1.8 {q2}, [r1]!
        vld1.8 {q4,q5}, [r2]!
        vmull.s8 q0, d4, d8
        vmull.s8 q1, d4, d10
        vmlal.s8 q0, d5, d9
        vmlal.s8 q1, d5, d11
        vld1.8 {q6,q7}, [r2]!
        vpadal.s16 q8, q0
        vpadal.s16 q9, q1

        vmull.s8 q0, d4, d12
        vmull.s8 q1, d4, d14
        vmlal.s8 q0, d5, d13
        vmlal.s8 q1, d5, d15
        vld1.8 {q3}, [r1]!
        vpadal.s16 q10, q0
        vpadal.s16 q11, q1
        // second four output
        vmull.s8 q0, d6, d8
        vmull.s8 q1, d6, d10
        vmlal.s8 q0, d7, d9
        vmlal.s8 q1, d7, d11
        vpadal.s16 q12, q0
        vpadal.s16 q13, q1

        vmull.s8 q0, d6, d12
        vmull.s8 q1, d6, d14
        vmlal.s8 q0, d7, d13
        vmlal.s8 q1, d7, d15
        vpadal.s16 q14, q0
        vpadal.s16 q15, q1

        subs r12, r12, #1
        bne L2LoopSz

    L2LoopSzEnd:

    cmp r8, #0
    bne L2Quan
    vld1.s32 {q4}, [lr]!
    vpadd.s32 d16, d16, d17
    vpadd.s32 d20, d20, d21
    vpadd.s32 d18, d18, d19
    vpadd.s32 d22, d22, d23
    vpadd.s32 d24, d24, d25
    vpadd.s32 d28, d28, d29
    vpadd.s32 d26, d26, d27
    vpadd.s32 d30, d30, d31
    vpadd.s32 d16, d16, d18
    vpadd.s32 d17, d20, d22
    vpadd.s32 d18, d24, d26
    vpadd.s32 d19, d28, d30
    vaddq.s32 q0, q8, q4
    vaddq.s32 q1, q9, q4
    vcvt.f32.s32 q0, q0
    vcvt.f32.s32 q1, q1
    vst1.f32 {q0, q1}, [r0], r4
    b L2LoopCheck

    L2Quan:
    vld1.s32 {q4}, [lr]!
    vld1.f32 {q5}, [r8]!

    vpadd.s32 d16, d16, d17
    vpadd.s32 d20, d20, d21
    vpadd.s32 d18, d18, d19
    vpadd.s32 d22, d22, d23

    vpadd.s32 d24, d24, d25
    vpadd.s32 d28, d28, d29
    vpadd.s32 d26, d26, d27
    vpadd.s32 d30, d30, d31

    // q8,q9
    vdup.32 q2, r6
    vdup.32 q3, r7
    vpadd.s32 d16, d16, d18
    vpadd.s32 d17, d20, d22
    vpadd.s32 d18, d24, d26
    vpadd.s32 d19, d28, d30

    vaddq.s32 q0, q8, q4
    vaddq.s32 q1, q9, q4

    vcvt.f32.s32 q0, q0
    vcvt.f32.s32 q1, q1

    vmov.f32 q10, #0.5
    vmov.f32 q11, #-0.5

    vmulq.f32 q0, q0, q5
    vmulq.f32 q1, q1, q5

    vcgt.f32 q12, q0, #0
    vcgt.f32 q13, q1, #0
    vbsl.f32 q12, q10, q11
    vbsl.f32 q13, q10, q11
    vadd.f32 q0, q12, q0
    vadd.f32 q1, q13, q1
    vcvt.s32.f32 q0, q0
    vcvt.s32.f32 q1, q1

    vmax.s32 q0, q2, q0
    vmax.s32 q1, q2, q1
    vmin.s32 q0, q3, q0
    vmin.s32 q1, q3, q1

    vqmovn.s32 d4, q0
    vqmovn.s32 d5, q1

    vqmovn.s16 d6, q2

    vst1.s8 d6, [r0], r4
L2LoopCheck:
    subs r5, r5, #1
    mov r1, r10
    bne L2LoopDz

b End

L1LoopDz:
    mov r10, r1

    subs r12, r3, #1
    // first four output
    vld1.8 {q2}, [r1]!
    vld1.8 {q4,q5}, [r2]!
    vmull.s8 q0, d4, d8
    vmull.s8 q1, d4, d10
    vmlal.s8 q0, d5, d9
    vmlal.s8 q1, d5, d11
    vpaddl.s16 q8, q0
    vpaddl.s16 q9, q1
    vld1.8 {q6,q7}, [r2]!

    vmull.s8 q0, d4, d12
    vmull.s8 q1, d4, d14
    vmlal.s8 q0, d5, d13
    vmlal.s8 q1, d5, d15
    vpaddl.s16 q10, q0
    add r1, r1, #16
    vpaddl.s16 q11, q1

    beq L1LoopSzEnd

    L1LoopSz:
        // first four output
        vld1.8 {q2}, [r1]!
        vld1.8 {q4,q5}, [r2]!
        vmull.s8 q0, d4, d8
        vmull.s8 q1, d4, d10
        vmlal.s8 q0, d5, d9
        vmlal.s8 q1, d5, d11
        vld1.8 {q6,q7}, [r2]!
        vpadal.s16 q8, q0
        vpadal.s16 q9, q1

        vmull.s8 q0, d4, d12
        vmull.s8 q1, d4, d14
        vmlal.s8 q0, d5, d13
        vmlal.s8 q1, d5, d15
        add r1, r1, #16
        vpadal.s16 q10, q0
        vpadal.s16 q11, q1

        subs r12, r12, #1
        bne L1LoopSz

    L1LoopSzEnd:
    cmp r8, #0
    bne L1Quan
    vld1.s32 {q4}, [lr]!
    vpadd.s32 d16, d16, d17
    vpadd.s32 d20, d20, d21
    vpadd.s32 d18, d18, d19
    vpadd.s32 d22, d22, d23
    vpadd.s32 d16, d16, d18
    vpadd.s32 d17, d20, d22
    vaddq.s32 q0, q8, q4
    vcvt.f32.s32 q0, q0
    vst1.f32 {q0}, [r0], r4
    b L1LoopCheck

    L1Quan:
    vld1.s32 {q4}, [lr]!
    vld1.f32 {q5}, [r8]!

    vpadd.s32 d16, d16, d17
    vpadd.s32 d20, d20, d21
    vpadd.s32 d18, d18, d19
    vpadd.s32 d22, d22, d23

    // q8,q9
    vdup.32 q2, r6
    vdup.32 q3, r7
    vpadd.s32 d16, d16, d18
    vpadd.s32 d17, d20, d22

    vaddq.s32 q0, q8, q4

    vcvt.f32.s32 q0, q0

    vmov.f32 q10, #0.5
    vmov.f32 q11, #-0.5

    vmulq.f32 q0, q0, q5

    vcgt.f32 q12, q0, #0
    vbsl.f32 q12, q10, q11
    vbsl.f32 q13, q10, q11
    vadd.f32 q0, q12, q0
    vcvt.s32.f32 q0, q0

    vmax.s32 q0, q2, q0
    vmin.s32 q0, q3, q0

    vqmovn.s32 d4, q0

    vqmovn.s16 d6, q2

    vst1.s32 d6[0], [r0], r4
L1LoopCheck:
    subs r5, r5, #1
    mov r1, r10
    bne L1LoopDz

End:
vpop {q4-q7}
pop {r4-r8, r10, pc}

#endif
#endif
