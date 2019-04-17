//
//  MNNConvRunForUnitDepthWiseInt8.S
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

asm_function MNNConvRunForUnitDepthWiseInt8

//void MNNConvRunForUnitDepthWiseInt8(float* dst, const int8_t* src, const int8_t* weight, size_t fw, size_t fh, size_t weight_y_step, size_t dilate_x_step,
// size_t dilate_y_step, const float* scale)

push {r4-r11, lr}

//Default
//r0:dst, r1:src, r2:weight, r3:fw

//Load from sp
//r4:fh, r5:weight_y_step, r6:dilate_x_step, r7:dilate_y_step, r8:scale
ldr r4, [sp, #36]
ldr r5, [sp, #40]
ldr r6, [sp, #44]
ldr r7, [sp, #48]
ldr r8, [sp, #52]

vld1.32 {q15}, [r8]

vmov.i32 q0, #0

cmp r3, #0
beq EndUnit

cmp r4, #0
beq EndUnit


//weight_y_step -> weight_y_step - fw*4*sizeof(int8_t)
mov r10, #4
mul r10, r3, r10
sub r5, r5, r10

//dilate_y_step -> dilate_y_step - fw*dilate_x_step
mul r10, r6, r3
sub r7, r7, r10
LoopFY:
    mov r12, r3
    LoopFX:
        vld1.32 {d4[0]}, [r1], r6
        vld1.32 {d16[0]}, [r2]!
        vmovl.s8 q2, d4
        vmovl.s8 q8, d16
        vmlal.s16 q0, d4, d16

        subs r12, r12, #1
        bne LoopFX
    subs r4, r4, #1
    add r1, r7, r1
    add r2, r2, r5
    bne LoopFY

vcvt.f32.s32 q1, q0
vmul.f32 q0, q1, q15


EndUnit:
vst1.32 {q0}, [r0]


pop {r4-r11, pc}




#endif
#endif
