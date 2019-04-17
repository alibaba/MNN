//
//  MNNDeconvRunForUnitDepthWise.S
//  MNN
//
//  Created by MNN on 2018/07/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __arm__
#ifndef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNDeconvRunForUnitDepthWise
//void MNNDeconvRunForUnitDepthWise(const float* dst, float* src, const float* weight, size_t fw, size_t fh, size_t weight_y_step, size_t dilate_x_step, size_t dilate_y_step)
push {r4-r7, lr}

//Default
//r0:dst, r1:src, r2:weight, r3:fw

//Load from sp
//r4:fh, r5:weight_y_step, r6:dilate_x_step, r7:dilate_y_step
ldr r4, [sp, #20]
ldr r5, [sp, #24]
ldr r6, [sp, #28]
ldr r7, [sp, #32]

cmp r3, #0
beq EndUnit

cmp r4, #0
beq EndUnit

//multi by sizeof(float)
mov r12, #4
mul r5, r12, r5
mul r6, r12, r6
mul r7, r12, r7

//weight_y_step -> weight_y_step - fw*4*sizeof(float)
mov r12, #16
mul r12, r3, r12
sub r5, r5, r12

//dilate_y_step -> dilate_y_step - fw*dilate_x_step
mul r12, r6, r3
sub r7, r7, r12

vld1.f32 {q0}, [r0]

LoopFY:
    mov r12, r3
    LoopFX:
        vld1.32 {q8}, [r2]!
        vld1.32 {q1}, [r1]
        vmla.f32 q1, q8, q0
        vst1.32 {q1}, [r1], r6
        subs r12, r12, #1
        bne LoopFX

    add r1, r7, r1
    subs r4, r4, #1
    add r2, r2, r5
    bne LoopFY

EndUnit:

pop {r4-r7, pc}

#endif
#endif
