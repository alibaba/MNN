//
//  MNNConvSlideWindowBorder.S
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

asm_function MNNConvSlideWindowBorder
//void MNNConvSlideWindowBorder(float* dst, const float* src, const float* weight, size_t src_depth_quad, size_t src_depth_step, size_t fw, size_t fh, size_t weight_y_step, size_t weight_z_step, size_t dilate_x_step, size_t dilate_y_step)
push {r4-r11, lr}

//Default
//r0:dst, r1:src, r2:weight, r3:src_depth_quad

//Load from sp
//r4:src_depth_step, r5:fw, r6:fh, r7:weight_y_step
//r8:weight_z_step, r9:dilate_x_step, r10:dilate_y_step
ldr r4, [sp, #36]
ldr r5, [sp, #40]
ldr r6, [sp, #44]
ldr r7, [sp, #48]
ldr r8, [sp, #52]
ldr r9, [sp, #56]
ldr r10, [sp, #60]

vmov.i32 q0, #0
vmov.i32 q1, #0

cmp r6, #0
beq EndUnit

cmp r5, #0
beq EndUnit


//multi by sizeof(float)
mov r11, #4
mul r4, r11, r4
mul r7, r11, r7
mul r8, r11, r8
mul r9, r11, r9
mul r10, r11, r10

//weight_z_step -> weight_z_step - fh*weight_y_step
mul r11, r7, r6
sub r8, r8, r11

//weight_y_step -> weight_y_step - fw*16*sizeof(float)
mov r11, #64
mul r11, r5, r11
sub r7, r7, r11

//src_depth_step -> src_depth_step - fh*dilate_y_step
mul r11, r6, r10
sub r4, r4, r11

//dilate_y_step -> dilate_y_step - fw*dilate_x_step
mul r11, r9, r5
sub r10, r10, r11


LoopZ:
mov r11, r6
LoopFY:
mov r12, r5
LoopFX:
vld1.32 {q3}, [r1], r9
vld1.32 {q8,q9}, [r2]!
vmla.f32 q0, q8, d6[0]
vld1.32 {q10, q11}, [r2]!
vmla.f32 q1, q9, d6[1]
vmla.f32 q0, q10,d7[0]
vmla.f32 q1, q11,d7[1]
subs r5, r5, #1
bne LoopFX
subs r6, r6, #1
add r1, r10, r1
add r2, r2, r7
mov r5, r12
bne LoopFY
mov r6, r11
subs r3, r3, #1
add r1, r1, r4
add r2, r2, r8
bne LoopZ
vadd.f32 q0, q0, q1

EndUnit:
vst1.32 {q0}, [r0]

pop {r4-r11, pc}
#endif
#endif
