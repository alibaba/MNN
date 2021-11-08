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
ldr r11, [r3, #8]
// d10: max, d11: min
vdup.i8 d10, r11
ldr r11, [r3, #12]
vdup.i8 d11, r11
ldr r3, [r3, #4]

vld1.32 {q6}, [r3] // q6
vld1.32 {q4}, [r10] // q4

mul r10, r6, r8
sub lr, lr, r10

L8:
cmp r4, #8
blt L4

mov r12, #8
mul r12, r5, r12

L8Loop_NORMAL:
    // load bias
    vmov.i32 q8, q6
    vmov.i32 q9, q6
    vmov.i32 q10, q6
    vmov.i32 q11, q6
    vmov.i32 q12, q6
    vmov.i32 q13, q6
    vmov.i32 q14, q6
    vmov.i32 q15, q6

    vmov.i32 d14[0], r1
    vmov.i32 d14[1], r2
    mov r10, r7
    L8LoopH_NORMAL:
        mov r11, r6
        L8LoopW_NORMAL:
            vld1.32 {d6[0]}, [r2]!
            vmovl.s8 q3, d6
            vld1.32 {d0[0]}, [r1], r5
            vld1.32 {d0[1]}, [r1], r5
            subs r11, r11, #1
            vmovl.s8 q1, d0
            vld1.32 {d1[0]}, [r1], r5
            vld1.32 {d1[1]}, [r1], r5
            vmovl.s8 q2, d1
            vmlal.s16 q8, d6, d2

            vld1.32 {d0[0]}, [r1], r5
            vmlal.s16 q9, d6, d3
            vld1.32 {d0[1]}, [r1], r5
            vld1.32 {d1[0]}, [r1], r5
            vld1.32 {d1[1]}, [r1], r5
            vmlal.s16 q10, d6, d4
            vmovl.s8 q1, d0
            vmlal.s16 q11, d6, d5
            vmovl.s8 q2, d1
            vmlal.s16 q12, d6, d2
            vmlal.s16 q13, d6, d3
            vmlal.s16 q14, d6, d4
            vmlal.s16 q15, d6, d5

            sub r1, r1, r12
            add r1, r1, r8
            bne L8LoopW_NORMAL

        subs r10, r10, #1
        add r1, r1, lr
        bne L8LoopH_NORMAL

    sub r4, r4, #8
    vmov.f32 q0, #0.5
    vmov.f32 q1, #-0.5

    vcvt.f32.s32 q8, q8
    vcvt.f32.s32 q9, q9
    vcvt.f32.s32 q10, q10
    vcvt.f32.s32 q11, q11
    vcvt.f32.s32 q12, q12
    vcvt.f32.s32 q13, q13
    vcvt.f32.s32 q14, q14
    vcvt.f32.s32 q15, q15

    vmul.f32 q8, q8, q4
    vmul.f32 q9, q9, q4
    vmul.f32 q10, q10, q4
    vmul.f32 q11, q11, q4
    vmul.f32 q12, q12, q4
    vmul.f32 q13, q13, q4
    vmul.f32 q14, q14, q4
    vmul.f32 q15, q15, q4

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
    ROUND_TWO q8, q9
    ROUND_TWO q10, q11
    ROUND_TWO q12, q13
    ROUND_TWO q14, q15

    vqmovn.s32 d4, q8
    vqmovn.s32 d5, q9
    vqmovn.s32 d0, q10
    vqmovn.s32 d1, q11
    vqmovn.s16 d6, q2
    vqmovn.s16 d7, q0

    vmin.s8 d6, d6, d10
    vmin.s8 d7, d7, d10
    vmax.s8 d6, d6, d11
    vmax.s8 d7, d7, d11

    vst1.32 {q3}, [r0]!

    vqmovn.s32 d4, q12
    vqmovn.s32 d5, q13
    vqmovn.s32 d0, q14
    vqmovn.s32 d1, q15
    vqmovn.s16 d6, q2
    vqmovn.s16 d7, q0

    vmin.s8 d6, d6, d10
    vmin.s8 d7, d7, d10
    vmax.s8 d6, d6, d11
    vmax.s8 d7, d7, d11

    vst1.32 {q3}, [r0]!

    vmov.i32 r1, d14[0]
    vmov.i32 r2, d14[1]
    add r1, r1, r12
    cmp r4, #8
    bge L8Loop_NORMAL

L4:
cmp r4, #4
blt L1

mov r12, #4
mul r12, r5, r12

L4Loop:
    vmov.i32 q8, q6
    vmov.i32 q9, q6
    vmov.i32 q10, q6
    vmov.i32 q11, q6

    vmov.i32 d14[0], r1
    vmov.i32 d14[1], r2
    mov r10, r7
    L4LoopH:
        mov r11, r6
        L4LoopW:
            vld1.32 {d6[0]}, [r2]!
            vmovl.s8 q3, d6
            vld1.32 {d0[0]}, [r1], r5
            vld1.32 {d0[1]}, [r1], r5
            subs r11, r11, #1
            vmovl.s8 q1, d0
            vld1.32 {d1[0]}, [r1], r5
            vld1.32 {d1[1]}, [r1], r5
            vmovl.s8 q2, d1
            vmlal.s16 q8, d6, d2
            vmlal.s16 q9, d6, d3
            vmlal.s16 q10, d6, d4
            vmlal.s16 q11, d6, d5

            sub r1, r1, r12
            add r1, r1, r8
            bne L4LoopW
        L4LoopWEnd:
        subs r10, r10, #1
        add r1, r1, lr
        bne L4LoopH
    sub r4, r4, #4
    vmov.f32 q0, #0.5
    vmov.f32 q1, #-0.5

    vcvt.f32.s32 q8, q8
    vcvt.f32.s32 q9, q9
    vcvt.f32.s32 q10, q10
    vcvt.f32.s32 q11, q11

    vmul.f32 q8, q8, q4
    vmul.f32 q9, q9, q4
    vmul.f32 q10, q10, q4
    vmul.f32 q11, q11, q4

    ROUND_TWO q8, q9
    ROUND_TWO q10, q11

    vqmovn.s32 d4, q8
    vqmovn.s32 d5, q9
    vqmovn.s32 d0, q10
    vqmovn.s32 d1, q11
    vqmovn.s16 d6, q2
    vqmovn.s16 d7, q0

    vmin.s8 d6, d6, d10
    vmin.s8 d7, d7, d10
    vmax.s8 d6, d6, d11
    vmax.s8 d7, d7, d11

    vst1.32 {q3}, [r0]!
    vmov.i32 r1, d14[0]
    vmov.i32 r2, d14[1]
    add r1, r1, r12
    cmp r4, #4
    bge L4Loop


L1:
cmp r4, #0
beq End
vmov.f32 q14, #0.5
vmov.f32 q15, #-0.5

L1Loop:
    vmov.i32 q8, q6
    mov r10, r7
    vmov.i32 d14[0], r1
    vmov.i32 d14[1], r2
    L1LoopH:
        mov r11, r6
        L1LoopW:
            vld1.32 {d2[0]}, [r1], r8
            vld1.32 {d4[0]}, [r2]!
            vmovl.s8 q1, d2
            vmovl.s8 q2, d4
            vmlal.s16 q8, d2, d4
            subs r11, r11, #1
            bne L1LoopW
        subs r10, r10, #1
        add r1, r1, lr
        bne L1LoopH

    subs r4, r4, #1

    vcvt.f32.s32 q8, q8
    vmul.f32 q8, q8, q4
    vcgt.f32 q2, q8, #0
    vbsl.f32 q2, q14, q15
    vadd.f32 q8, q2, q8
    vcvt.s32.f32 q8, q8

    vqmovn.s32 d4, q8
    vqmovn.s16 d6, q2

    vmin.s8 d6, d6, d10
    vmax.s8 d6, d6, d11

    vst1.32 {d6[0]}, [r0]!
    vmov.i32 r1, d14[0]
    vmov.i32 r2, d14[1]
    add r1, r1, r5
    bne L1Loop

End:
vpop {q4-q7}
pop {r4-r8, r10, r11, pc}


#endif
#endif
