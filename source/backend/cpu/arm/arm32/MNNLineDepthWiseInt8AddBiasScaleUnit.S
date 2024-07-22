//
//  MNNLineDepthWiseInt8AddBiasScaleUnit.S
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

asm_function MNNLineDepthWiseInt8AddBiasScaleUnit

//void MNNLineDepthWiseInt8AddBiasScaleUnit(int8_t* dst, const int8_t* src, const int8_t* weight, const QuanPostTreatParameters* parameters,
//                                          size_t width, size_t src_w_step, size_t fw, size_t fh, size_t dilateX_step,
//                                          size_t dilateY_step) {


//struct QuanPostTreatParameters {
//    const float* scale;
//    const int32_t* bias;
//    int32_t maxValue;
//    int32_t minValue;
//    float roundValuePos = 0.5f;
//    float roundValueNeg = -0.5f;
//};

// Auto Load:
// r0: dst*, r1: src*, r2: weight*, r3: parameters*
// Load from sp
// r4: width, r5: src_w_step, r6: fw, r7: fh, r8: dilateX_step, lr: dilateY_step, r10: scale_z, r11: mode
push {r4-r8, r10, r11, lr} // avoid to touch platform-register r-9

ldr r4, [sp, #32]
ldr r5, [sp, #36]
ldr r6, [sp, #40]
ldr r7, [sp, #44]
ldr r8, [sp, #48]
ldr lr, [sp, #52]

vpush {q4-q7}

ldr r10, [r3, #0]
vld1.32 {q12, q13}, [r10]! // scale, pack=16
vld1.32 {q14, q15}, [r10]!

ldr r11, [r3, #8]
// d23: max, d22: min
vdup.i8 d23, r11
ldr r11, [r3, #12]
vdup.i8 d22, r11
ldr r3, [r3, #44] // bias

mul r10, r6, r8
sub lr, lr, r10

L1:
cmp r4, #0
beq End

L1Loop:
    vld1.32 {q7, q8}, [r3]!
    vld1.32 {q9, q10}, [r3]
    sub r3, r3, #32

    vmov.i32 d12[0], r1
    vmov.i32 d12[1], r2
    mov r10, r7
    L1LoopH:
        mov r11, r6
        L1LoopW:
            vld1.32 {q1}, [r2]!
            vld1.32 {q0}, [r1], r8
            
            vmovl.s8 q2, d0
            vmovl.s8 q3, d1
            vmovl.s8 q4, d2
            vmovl.s8 q5, d3
            vmlal.s16 q7, d4, d8
            vmlal.s16 q8, d5, d9
            vmlal.s16 q9, d6, d10
            vmlal.s16 q10, d7, d11

            subs r11, r11, #1
            bne L1LoopW
        L1LoopWEnd:
        subs r10, r10, #1
        add r1, r1, lr
        bne L1LoopH

    vmov.f32 q0, #0.5
    vmov.f32 q1, #-0.5

    vcvt.f32.s32 q7,  q7
    vcvt.f32.s32 q8,  q8
    vcvt.f32.s32 q9,  q9
    vcvt.f32.s32 q10, q10

    vmul.f32 q7,  q7, q12
    vmul.f32 q8,  q8, q13
    vmul.f32 q9,  q9, q14
    vmul.f32 q10, q10, q15
    /*
    vrshrn.s32 d14, q7,  #14
    vrshrn.s32 d15, q8,  #14
    vrshrn.s32 d18, q9, #14
    vrshrn.s32 d19, q10, #14
    */
.macro ROUND_TWO x0, x1
    vcgt.f32 q2, \x0, #0
    vcgt.f32 q3, \x1, #0
    vbsl.f32 q2, q0, q1
    vbsl.f32 q3, q0, q1
    vadd.f32 \x0, q2, \x0
    vadd.f32 \x1, q3, \x1
    vcvt.s32.f32 \x0, \x0
    vcvt.s32.f32 \x1, \x1
.endm
    ROUND_TWO q7, q8
    ROUND_TWO q9, q10

    vqmovn.s32 d14, q7
    vqmovn.s32 d15, q8
    vqmovn.s32 d18, q9
    vqmovn.s32 d19, q10

    vqmovn.s16 d14, q7
    vqmovn.s16 d15, q9

    vmin.s8 d14, d14, d23
    vmax.s8 d14, d14, d22
    vmin.s8 d15, d15, d23
    vmax.s8 d15, d15, d22

    vst1.32 {q7}, [r0]!
    vmov.i32 r1, d12[0]
    vmov.i32 r2, d12[1]
    add r1, r1, r5
    subs r4, r4, #1
    bne L1Loop

End:
vpop {q4-q7}
pop {r4-r8, r10, r11, pc}


#endif
#endif
