//
//  MNNConvRunForLineDepthWiseUint8.S
//  MNN
//
//  Created by MNN on 2018/10/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __arm__
#ifndef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5
/*
struct MNN::ConstConvolutionParameter
{
    size_t kw;
    size_t kh;
    size_t weight_y_step;
    size_t dilate_x_step;
    size_t dilate_y_step;
    size_t stride_x_step;
    int32_t output_multiplier;
    int32_t output_shift_before;
    int32_t output_shift_after;
    int32_t output_offset;
    int32_t output_activation_min;
    int32_t output_activation_max;
};
*/

asm_function MNNConvRunForLineDepthWiseUint8
//void MNNConvRunForLineDepthWiseUint8(uint8_t* dst, const int16_t* src, const int16_t* weight, size_t width,
//                                       MNN::ConstConvolutionParameter* parameters,
//                                       const int32_t* bias_data)

push {r4-r8, r10, r11, lr} // avoid to touch platform-register r-9
//Auto: r0:dst, r1:src, r2:weight, r3:width

//Load from sp
//r4: parameters, r5: bias_data
ldr r4, [sp, #32]
ldr r5, [sp, #36]

//Load All parameters
//q15: output_multiplier, q14: output_shift_before, q13: output_shift_after
//q12: output_offset, d22: output_activation_min, d23: output_activation_max, q9: bias

vld1.32 {q9}, [r5]
ldr r6, [r4, #24]
vdup.32 q15, r6
ldr r6, [r4, #28]
vdup.32 q14, r6
ldr r6, [r4, #32]
vdup.32 q13, r6
ldr r6, [r4, #36]
vdup.32 q12, r6
ldr r6, [r4, #40]
vdup.32 q11, r6
ldr r6, [r4, #44]
vmovn.s32 d22, q11
vdup.32 q10, r6
vmovn.s32 d23, q10

//Now r5 is no used, set r5 as stride_x_step
ldr r5, [r4, #20]
//r6:fw, r7:fh
ldr r6, [r4, #0]
ldr r7, [r4, #4]

//r8:dilate_x_step, lr:dilate_y_step
ldr r8, [r4, #12]
ldr lr, [r4, #16]

//dilate_y_step->dilate_y_step-fw*dilate_x_step
mul r12, r6, r8
sub lr, lr, r12
vpush {q4-q7}

L8:
cmp r3, #8
blt L1

mov r12, #8
mul r10, r5, r12

LoopL8:
    push {r1, r2}
    vmov.s32 q0, q9
    vmov.s32 q1, q9
    vmov.s32 q2, q9
    vmov.s32 q3, q9
    vmov.s32 q4, q9
    vmov.s32 q5, q9
    vmov.s32 q6, q9
    vmov.s32 q7, q9

    mov r11, r7
    LoopL8FY:
        mov r12, r6
        LoopL8FX:
            vld1.32 {d16}, [r2]!
            vld1.32 {d20}, [r1], r5
            vmlal.s16 q0, d16, d20
            vld1.32 {d21}, [r1], r5
            vmlal.s16 q1, d16, d21
            vld1.32 {d20}, [r1], r5
            vmlal.s16 q2, d16, d20
            vld1.32 {d21}, [r1], r5
            vmlal.s16 q3, d16, d21
            vld1.32 {d20}, [r1], r5
            vmlal.s16 q4, d16, d20
            vld1.32 {d21}, [r1], r5
            vmlal.s16 q5, d16, d21
            vld1.32 {d20}, [r1], r5
            vmlal.s16 q6, d16, d20
            vld1.32 {d21}, [r1], r5
            vmlal.s16 q7, d16, d21
            sub r1, r1, r10
            add r1, r1, r8
            subs r12, r12, #1
            bne LoopL8FX
        subs r11, r11, #1
        add r1, r1, lr
        bne LoopL8FY
    pop {r1, r2}
    vshl.s32 q0, q0, q14
    vshl.s32 q1, q1, q14
    vshl.s32 q2, q2, q14
    vshl.s32 q3, q3, q14
    vshl.s32 q4, q4, q14
    vshl.s32 q5, q5, q14
    vshl.s32 q6, q6, q14
    vshl.s32 q7, q7, q14

    vqrdmulh.s32 q0, q15, q0
    vqrdmulh.s32 q1, q15, q1
    vqrdmulh.s32 q2, q15, q2
    vqrdmulh.s32 q3, q15, q3
    vqrdmulh.s32 q4, q15, q4
    vqrdmulh.s32 q5, q15, q5
    vqrdmulh.s32 q6, q15, q6
    vqrdmulh.s32 q7, q15, q7

    vrshl.s32 q0, q0, q13
    vrshl.s32 q1, q1, q13
    vrshl.s32 q2, q2, q13
    vrshl.s32 q3, q3, q13
    vrshl.s32 q4, q4, q13
    vrshl.s32 q5, q5, q13
    vrshl.s32 q6, q6, q13
    vrshl.s32 q7, q7, q13

    vadd.s32 q0, q0, q12
    vadd.s32 q1, q1, q12
    vadd.s32 q2, q2, q12
    vadd.s32 q3, q3, q12
    vadd.s32 q4, q4, q12
    vadd.s32 q5, q5, q12
    vadd.s32 q6, q6, q12
    vadd.s32 q7, q7, q12

    add r1, r1, r10
    vmovn.s32 d0, q0
    vmovn.s32 d1, q1
    vmax.s16 d0, d0, d22
    vmovn.s32 d2, q2
    vmax.s16 d1, d1, d22
    vmovn.s32 d3, q3
    vmin.s16 d0, d0, d23
    vmovn.s32 d4, q4
    vmin.s16 d1, d1, d23
    vmovn.s32 d5, q5
    vmovn.s32 d6, q6
    vmovn.s32 d7, q7

    vmax.s16 d2, d2, d22
    vmax.s16 d3, d3, d22
    vmax.s16 d4, d4, d22
    vmax.s16 d5, d5, d22
    vmax.s16 d6, d6, d22
    vmax.s16 d7, d7, d22

    vmin.s16 d2, d2, d23
    vmin.s16 d3, d3, d23
    vmin.s16 d4, d4, d23
    vmin.s16 d5, d5, d23
    vmin.s16 d6, d6, d23
    vmin.s16 d7, d7, d23

    vmovn.s16 d0, q0
    vmovn.s16 d1, q1
    vmovn.s16 d2, q2
    vmovn.s16 d3, q3
    vst1.32 {q0, q1}, [r0]!
    sub r3, r3, #8
    cmp r3, #8
    bge LoopL8


L1:
cmp r3, #0
beq End

LoopL1:
    push {r1, r2}
    vmov.s32 q0, q9
    mov r11, r7
    LoopL1FY:
        mov r12, r6
        LoopL1FX:
            vld1.32 {d2}, [r1], r8
            vld1.32 {d3}, [r2]!
            vmlal.s16 q0, d2, d3
            subs r12, r12, #1
            bne LoopL1FX
        subs r11, r11, #1
        add r1, r1, lr
        bne LoopL1FY
    pop {r1, r2}
    vshl.s32 q0, q0, q14
    vqrdmulh.s32 q0, q15, q0
    vrshl.s32 q0, q0, q13
    vadd.s32 q0, q0, q12

    add r1, r1, r5
    vmovn.s32 d0, q0
    vmax.s16 d0, d0, d22
    vmin.s16 d0, d0, d23
    vmovn.s16 d0, q0
    vst1.32 {d0[0]}, [r0]!
    subs r3, r3, #1
    bne LoopL1

End:

vpop {q4-q7}
pop {r4-r8, r10, r11, pc}



#endif
#endif
