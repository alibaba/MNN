//
//  MNNConvSlideWindowMiddle.S
//  MNN
//
//  Created by MNN on 2019/02/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __arm__
#ifndef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNConvSlideWindowMiddle
//void MNNConvSlideWindowMiddle(float* dst, const float* src, const float* weight, size_t width, size_t src_w_step, size_t src_depth_quad, size_t src_depth_step, size_t fw, size_t fh, size_t dilate_x_step, size_t dilate_y_step)

push {r4-r11, lr}

//Auto Load:
//r0:dst, r1:src, r2:weight, r3:width

//Load from sp
//r4:src_w_step, r5:src_depth_quad, r6: src_depth_step, r7:fw, r8:fh, r9:dilate_x_step, r10:dilate_y_step
ldr r4, [sp, #36]
ldr r5, [sp, #40]
ldr r6, [sp, #44]
ldr r7, [sp, #48]
ldr r8, [sp, #52]
ldr r9, [sp, #56]
ldr r10, [sp, #60]

vpush {q4-q7}

//step multi by sizeof(float)
mov r12, #4
mul r10, r12, r10
mul r9, r12, r9
mul r6, r12, r6
mul r4, r12, r4

//src_depth_step -> src_depth_step - fh*dilate_y_step
mul r12, r8, r10
sub r6, r6, r12

//dilate_y_step -> dilate_y_step-fw*dilate_x_step
mul r12, r7, r9
sub r10, r10, r12

L8:
cmp r3, #7
ble L4


L8Loop:
    vmov.i32 d6[0], r1
    vmov.i32 d6[1], r2
    vmov.i32 d7[0], r3
    vmov.i32 d7[1], r5
    mov r3, #8
    mul r3, r4, r3
    vmov.i32 q8,  #0
    vmov.i32 q9,  #0
    vmov.i32 q10, #0
    vmov.i32 q11, #0
    vmov.i32 q12, #0
    vmov.i32 q13, #0
    vmov.i32 q14, #0
    vmov.i32 q15, #0
    L8LoopZ:
        mov r11, r8
        L8LoopFY:
            mov r12, r7
            L8LoopFX:
                vld1.32 {q4, q5}, [r2]!
                vld1.32 {q6, q7}, [r2]!

                vld1.32 {q0}, [r1], r4

                vmla.f32 q8, q4, d0[0]

                vld1.32 {q1}, [r1], r4

                vmla.f32 q8, q5, d0[1]
                vmla.f32 q9, q4, d2[0]
                vmla.f32 q8, q6, d1[0]

                vld1.32 {q2}, [r1], r4

                vmla.f32 q8, q7, d1[1]
                vmla.f32 q9, q5, d2[1]
                vmla.f32 q10, q4, d4[0]

                vld1.32 {q0}, [r1], r4

                vmla.f32 q9, q6, d3[0]
                vmla.f32 q11, q7, d1[1]
                vmla.f32 q10, q5, d4[1]
                vmla.f32 q11, q5, d0[1]
                vmla.f32 q9, q7, d3[1]
                vmla.f32 q11, q6, d1[0]
                vmla.f32 q10, q6, d5[0]
                vmla.f32 q11, q4, d0[0]

                vld1.32 {q0}, [r1], r4

                vmla.f32 q10, q7, d5[1]
                vmla.f32 q12, q4, d0[0]

                vld1.32 {q1}, [r1], r4

                vmla.f32 q12, q5, d0[1]
                vmla.f32 q13, q4, d2[0]
                
                vld1.32 {q2}, [r1], r4

                vmla.f32 q12, q6, d1[0]
                vmla.f32 q13, q5, d2[1]
                vmla.f32 q12, q7, d1[1]
                vmla.f32 q14, q4, d4[0]

                vld1.32 {q0}, [r1], r4
                vmla.f32 q13, q6, d3[0]
                vmla.f32 q14, q5, d4[1]
                vmla.f32 q15, q5, d0[1]
                vmla.f32 q13, q7, d3[1]
                vmla.f32 q15, q6, d1[0]
                vmla.f32 q14, q6, d5[0]
                vmla.f32 q15, q4, d0[0]
                vmla.f32 q14, q7, d5[1]
                vmla.f32 q15, q7, d1[1]

                sub r1, r1, r3
                subs r7, r7, #1
                add r1, r1, r9
                bne L8LoopFX
            subs r8, r8, #1
            mov r7, r12
            add r1, r1, r10
            bne L8LoopFY
        subs r5, r5, #1
        mov r8, r11
        add r1, r1, r6
        bne L8LoopZ
    vmov.i32 r1, d6[0]
    add r1, r1, r3
    vmov.i32 r2, d6[1]
    vmov.i32 r3, d7[0]
    vmov.i32 r5, d7[1]
    vst1.32 {q8, q9}, [r0]!
    vst1.32 {q10, q11}, [r0]!
    sub r3, r3, #8
    vst1.32 {q12, q13}, [r0]!
    cmp r3, #8
    vst1.32 {q14, q15}, [r0]!
    bge L8Loop

L4:
cmp r3, #3
ble L1


L4Loop:
    vmov.i32 d30[0], r1
    vmov.i32 d30[1], r2
    vmov.i32 d31[0], r3
    vmov.i32 d31[1], r5
    mov r3, #4
    mul r3, r4, r3
    vmov.i32 q8, #0
    vmov.i32 q9, #0
    vmov.i32 q10, #0
    vmov.i32 q11, #0
    L4LoopZ:
        mov r11, r8
        L4LoopFY:
            mov r12, r7
            L4LoopFX:
                vld1.32 {q4, q5}, [r2]!
                vld1.32 {q6, q7}, [r2]!

                vld1.32 {q0}, [r1], r4

                vmla.f32 q8, q4, d0[0]

                vld1.32 {q1}, [r1], r4

                vmla.f32 q8, q5, d0[1]
                vmla.f32 q9, q4, d2[0]
                vmla.f32 q8, q6, d1[0]

                vld1.32 {q2}, [r1], r4

                vmla.f32 q8, q7, d1[1]
                vmla.f32 q9, q5, d2[1]
                vmla.f32 q10, q4, d4[0]

                vld1.32 {q3}, [r1], r4

                vmla.f32 q9, q6, d3[0]
                vmla.f32 q11, q7, d7[1]
                vmla.f32 q10, q5, d4[1]
                vmla.f32 q11, q5, d6[1]
                vmla.f32 q9, q7, d3[1]
                vmla.f32 q11, q6, d7[0]
                vmla.f32 q10, q6, d5[0]
                vmla.f32 q11, q4, d6[0]
                vmla.f32 q10, q7, d5[1]

                sub r1, r1, r3
                subs r7, r7, #1
                add r1, r1, r9
                bne L4LoopFX
            subs r8, r8, #1
            mov r7, r12
            add r1, r1, r10
            bne L4LoopFY
        subs r5, r5, #1
        mov r8, r11
        add r1, r1, r6
        bne L4LoopZ
    vmov.i32 r1, d30[0]
    add r1, r1, r3
    vmov.i32 r2, d30[1]
    vst1.32 {q8, q9}, [r0]!
    vmov.i32 r3, d31[0]
    vmov.i32 r5, d31[1]
    sub r3, r3, #4
    vst1.32 {q10, q11}, [r0]!


L1:
cmp r3, #0
ble End

L1Loop:
    vmov.i32 d16[0], r1
    vmov.i32 d16[1], r2
    vmov.i32 d17[0], r5
    vmov.i32 q0, #0
    vmov.i32 q1, #0
    L1LoopZ:
        mov r11, r8
        L1LoopFY:
            mov r12, r7
            L1LoopFX:
                vld1.32 {q3}, [r1], r9
                vld1.32 {q4, q5}, [r2]!
                vmla.f32 q0, q4, d6[0]
                vmla.f32 q1, q5, d6[1]
                vld1.32 {q6, q7}, [r2]!
                vmla.f32 q0, q6, d7[0]
                vmla.f32 q1, q7, d7[1]
                subs r7, r7, #1
                bne L1LoopFX
            subs r8, r8, #1
            mov r7, r12
            add r1, r1, r10
            bne L1LoopFY
        subs r5, r5, #1
        mov r8, r11
        add r1, r1, r6
        bne L1LoopZ

    vadd.f32 q0, q0, q1
    vmov.i32 r1, d16[0]
    vmov.i32 r2, d16[1]
    vmov.i32 r5, d17[0]
    add r1, r1, r4
    vst1.32 {q0}, [r0]!
    subs r3, r3, #1
    bne L1Loop

End:

vpop {q4-q7}
pop {r4-r11, pc}

#endif
#endif
