//
//  NEON_MNNConvRunForLineDepthwise_BF16.S
//  MNN
//
//  Created by MNN on 2021/03/09.
//  Copyright © 2018-2021 Alibaba Group Holding Limited
//

#ifdef __arm__
#ifndef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function NEON_MNNConvRunForLineDepthwise_BF16
//void NEON_MNNConvRunForLineDepthwise_BF16(float* dst, const float* src, const float* weight, size_t width, size_t src_w_setup,
//                                size_t fw, size_t fh, size_t dilateX_step, size_t dilateY_step, size_t height, size_t srcHStep, size_t dstHStep)


//Auto Load:
//r0:dst, r1:src, r2:weight, r3:width

push {r4-r8, r10, r11, lr} // avoid to touch platform-register r-9

//Load From Sp
//r4:src_w_setup, r5:fw, r6:fh, r7:dilate_x_step, r8:dilate_y_step, lr: height, r10:srcHStep, r11:dstHStep
ldr r4, [sp, #32]
ldr r5, [sp, #36]
ldr r6, [sp, #40]
ldr r7, [sp, #44]
ldr r8, [sp, #48]
ldr lr, [sp, #52]
ldr r10, [sp, #56]
ldr r11, [sp, #60]

vpush {q4-q7}

mov r12, #2
mul r4, r12, r4
mul r7, r12, r7 // r7(dilate_x_step in byte) = sizeof(int16_t) * dilate_x_step
mul r8, r12, r8 // r8(dilate_y_step in byte) = sizeof(int16_t) * dilate_y_step
mul r10, r12, r10
mul r11, r12, r11

//dilate_y_step -> dilate_y_step - fw*dilate_x_step
mul r12, r5, r7
sub r8, r8, r12

LoopDY:
push {r0, r1, r3, r10, r11, lr}

L8:
cmp r3, #7
ble L4

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
            vld1.16 {d6}, [r2]!
            vld1.16 {q0}, [r1], r4
            vshll.s16 q3, d6, #16
            vshll.s16 q0, d0, #16
            subs r10, r10, #1
            vmla.f32 q8, q3, q0
            vld1.16 {d2}, [r1], r4
            vshll.s16 q1, d2, #16

            vmla.f32 q9, q3, q1
            vld1.16 {d0}, [r1], r4
            vshll.s16 q0, d0, #16
            vmla.f32 q10, q0, q3
            vld1.16 {d2}, [r1], r4
            vshll.s16 q1, d2, #16
            vmla.f32 q11, q1, q3
            vld1.16 {d0}, [r1], r4
            vshll.s16 q0, d0, #16
            vmla.f32 q12, q0, q3
            vld1.16 {d2}, [r1], r4
            vshll.s16 q1, d2, #16
            vmla.f32 q13, q1, q3
            vld1.16 {q0}, [r1], r4
            vshll.s16 q0, d0, #16
            vmla.f32 q14, q0, q3
            vld1.16 {d2}, [r1], r4
            vshll.s16 q1, d2, #16
            vmla.f32 q15, q1, q3

            sub r1, r1, r12
            add r1, r1, r7

            bne L8LoopW
        L8LoopWEnd:
        subs lr, lr, #1
        add r1, r1, r8
        bne L8LoopH

    sub r3, r3, #8
    vshrn.i32 d16, q8, #16
    vshrn.i32 d17, q9, #16
    vst1.16 {d16, d17}, [r0]!
    vmov.i32 r1, d14[0]
    vmov.i32 r2, d14[1]
    vshrn.i32 d20, q10, #16
    vshrn.i32 d21, q11, #16
    vst1.16 {d20, d21}, [r0]!
    add r1, r1, r12
    vshrn.i32 d24, q12, #16
    vshrn.i32 d25, q13, #16
    vst1.16 {d24, d25}, [r0]!
    cmp r3, #8
    vshrn.i32 d28, q14, #16
    vshrn.i32 d29, q15, #16
    vst1.16 {d28, d29}, [r0]!
    bge L8Loop

L4:
cmp r3, #3
ble L1

mov r12, #4
mul r12, r4, r12

L4Loop:
    vmov.i32 q8, #0
    vmov.i32 q9, #0
    vmov.i32 q10, #0
    vmov.i32 q11, #0

    vmov.i32 d8[0], r1
    vmov.i32 d9[0], r2
    mov lr, r6
    L4LoopH:
        mov r10, r5
        L4LoopW:
            vld1.16 {d24}, [r2]!
            vld1.16 {d0}, [r1], r4
            vshll.s16 q12, d24, #16
            vshll.s16 q0, d0, #16
            subs r10, r10, #1
            vmla.f32 q8, q12, q0
            vld1.16 {d2}, [r1], r4
            vshll.s16 q1, d2, #16
            vmla.f32 q9, q12, q1
            vld1.16 {d4}, [r1], r4
            vshll.s16 q2, d4, #16
            vmla.f32 q10, q2, q12
            vld1.16 {d6}, [r1], r4
            vshll.s16 q3, d6, #16
            sub r1, r1, r12
            vmla.f32 q11, q3, q12

            add r1, r1, r7

            bne L4LoopW
        subs lr, lr, #1
        add r1, r1, r8
        bne L4LoopH

    sub r3, r3, #4
    vshrn.i32 d16, q8, #16
    vshrn.i32 d17, q9, #16
    vst1.16 {d16, d17}, [r0]!
    vmov.i32 r1, d8[0]
    vmov.i32 r2, d9[0]
    vshrn.i32 d20, q10, #16
    vshrn.i32 d21, q11, #16
    vst1.16 {d20, d21}, [r0]!
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
            vld1.16 {d2}, [r1], r7
            vld1.16 {d4}, [r2]!
            vshll.s16 q1, d2, #16
            vshll.s16 q2, d4, #16
            vmla.f32 q0, q1, q2
            subs r10, r10, #1
            bne L1LoopW
        subs lr, lr, #1
        add r1, r1, r8
        bne L1LoopH

    subs r3, r3, #1
    vshrn.i32 d0, q0, #16
    vst1.16 {d0}, [r0]!
    mov r2, r12
    add r1, r11, r4
    bne L1Loop


End:

pop {r0, r1, r3, r10, r11, lr}
add r0, r0, r11
subs lr, lr, #1
add r1, r1, r10
bne LoopDY


vpop {q4-q7}
pop {r4-r8, r10, r11, pc}


#endif
#endif
