//
//  MNNConvRunForLineDepthWiseInt8.S
//  MNN
//
//  Created by MNN on 2018/09/12.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __arm__
#ifndef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNConvRunForLineDepthWiseInt8
//void MNNConvRunForLineDepthWiseInt8(float* dst, const int8_t* src, const int8_t* weight, size_t width, size_t src_w_setup, size_t fw, size_t fh,
//size_t dilate_x_step, size_t dilate_y_step, const float* alpha_z)

//Auto Load:
//r0:dst, r1:src, r2:weight, r3:width

push {r4-r8, r10, r11, lr} // avoid to touch platform-register r-9

//Load From Sp
//r4:src_w_setup, r5:fw, r6:fh, r7:dilate_x_step, r8:dilate_y_step, lr:alpha_z
ldr r4, [sp, #32]
ldr r5, [sp, #36]
ldr r6, [sp, #40]
ldr r7, [sp, #44]
ldr r8, [sp, #48]
ldr lr, [sp, #52]

vpush {q4-q7}

vld1.32 {q4}, [lr]
//dilate_y_step -> dilate_y_step - fw*dilate_x_step
mul lr, r5, r7
sub r8, r8, lr

L8:
cmp r3, #8
blt L4

mov r12, #8
mul r12, r4, r12

L8Loop:
    vmov.i32 q8, #0
    vmov.i32 q9, #0
    vmov.i32 q10, #0
    vmov.i32 q11, #0
    vmov.i32 q12, #0
    vmov.i32 q13, #0
    vmov.i32 q14, #0
    vmov.i32 q15, #0

    vmov.i32 d14[0], r1
    vmov.i32 d14[1], r2
    mov lr, r6
    L8LoopH:
        mov r10, r5
        L8LoopW:
            vld1.32 {d6[0]}, [r2]!
            vmovl.s8 q3, d6
            vld1.32 {d0[0]}, [r1], r4
            vld1.32 {d0[1]}, [r1], r4
            subs r10, r10, #1
            vmovl.s8 q1, d0
            vld1.32 {d1[0]}, [r1], r4
            vld1.32 {d1[1]}, [r1], r4
            vmovl.s8 q2, d1
            vmlal.s16 q8, d6, d2
            vld1.32 {d0[0]}, [r1], r4
            vmlal.s16 q9, d6, d3
            vld1.32 {d0[1]}, [r1], r4
            vld1.32 {d1[0]}, [r1], r4
            vld1.32 {d1[1]}, [r1], r4
            vmlal.s16 q10, d6, d4
            vmovl.s8 q1, d0
            vmlal.s16 q11, d6, d5
            vmovl.s8 q2, d1
            vmlal.s16 q12, d6, d2
            vmlal.s16 q13, d6, d3
            vmlal.s16 q14, d6, d4
            vmlal.s16 q15, d6, d5

            sub r1, r1, r12
            add r1, r1, r7

            bne L8LoopW
        L8LoopWEnd:
        subs lr, lr, #1
        add r1, r1, r8
        bne L8LoopH

    sub r3, r3, #8
    vcvt.f32.s32 q8, q8
    vcvt.f32.s32 q9, q9
    vcvt.f32.s32 q10, q10
    vmul.f32 q8, q8, q4
    vmul.f32 q9, q9, q4
    vcvt.f32.s32 q11, q11
    vcvt.f32.s32 q12, q12
    vst1.32 {q8, q9}, [r0]!
    vmul.f32 q10, q10, q4
    vmul.f32 q11, q11, q4
    vmul.f32 q12, q12, q4
    vcvt.f32.s32 q13, q13
    vst1.32 {q10, q11}, [r0]!
    vcvt.f32.s32 q14, q14
    vcvt.f32.s32 q15, q15
    vmul.f32 q13, q13, q4
    vmul.f32 q14, q14, q4
    vmul.f32 q15, q15, q4
    vst1.32 {q12, q13}, [r0]!

    vmov.i32 r1, d14[0]
    vmov.i32 r2, d14[1]
    add r1, r1, r12
    cmp r3, #8
    vst1.32 {q14, q15}, [r0]!
    bge L8Loop


L4:
cmp r3, #4
blt L1

mov r12, #4
mul r12, r4, r12

L4Loop:
    vmov.i32 q8, #0
    vmov.i32 q9, #0
    vmov.i32 q10, #0
    vmov.i32 q11, #0

    vmov.i32 d14[0], r1
    vmov.i32 d14[1], r2
    mov lr, r6
    L4LoopH:
        mov r10, r5
        L4LoopW:
            vld1.32 {d6[0]}, [r2]!
            vmovl.s8 q3, d6
            vld1.32 {d0[0]}, [r1], r4
            vld1.32 {d0[1]}, [r1], r4
            subs r10, r10, #1
            vmovl.s8 q1, d0
            vld1.32 {d1[0]}, [r1], r4
            vld1.32 {d1[1]}, [r1], r4
            vmovl.s8 q2, d1
            vmlal.s16 q8, d6, d2
            vmlal.s16 q9, d6, d3
            vmlal.s16 q10, d6, d4
            vmlal.s16 q11, d6, d5

            sub r1, r1, r12
            add r1, r1, r7

            bne L4LoopW
        L4LoopWEnd:
        subs lr, lr, #1
        add r1, r1, r8
        bne L4LoopH

    sub r3, r3, #4
    vcvt.f32.s32 q8, q8
    vcvt.f32.s32 q9, q9
    vcvt.f32.s32 q10, q10
    vmul.f32 q8, q8, q4
    vmul.f32 q9, q9, q4
    vcvt.f32.s32 q11, q11
    vcvt.f32.s32 q12, q12
    vst1.32 {q8, q9}, [r0]!
    vmul.f32 q10, q10, q4
    vmul.f32 q11, q11, q4
    vst1.32 {q10, q11}, [r0]!

    vmov.i32 r1, d14[0]
    vmov.i32 r2, d14[1]
    add r1, r1, r12
    cmp r3, #4
    bge L4Loop


L1:
cmp r3, #0
beq End

L1Loop:
    vmov.i32 q0, #0
    mov lr, r6
    mov r11, r1
    mov r12, r2
    L1LoopH:
        mov r10, r5
        L1LoopW:
            vld1.32 {d2[0]}, [r1], r7
            vld1.32 {d4[0]}, [r2]!
            vmovl.s8 q1, d2
            vmovl.s8 q2, d4
            vmlal.s16 q0, d2, d4
            subs r10, r10, #1
            bne L1LoopW
        subs lr, lr, #1
        add r1, r1, r8
        bne L1LoopH

    subs r3, r3, #1

    vcvt.f32.s32 q0, q0
    vmul.f32 q0, q0, q4
    vst1.32 {q0}, [r0]!
    mov r2, r12
    add r1, r11, r4
    bne L1Loop


End:



vpop {q4-q7}
pop {r4-r8, r10, r11, pc}



#endif
#endif
