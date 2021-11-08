//
//  MNNConvRunForUnitDepthWise.S
//  MNN
//
//  Created by MNN on 2019/02/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __arm__
#ifndef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNConvRunForUnitDepthWise
//void MNNConvRunForUnitDepthWise(float* dst, const float* src, const float* weight, size_t fw, size_t fh, size_t weight_y_step, size_t dilate_x_step, size_t dilate_y_step)

//Auto: r0:dst, r1:src, r2:weight, r3:fw

push {r4-r8, lr}

//Load from sp:
//r5:fh, r6:weight_y_step, r7:dilate_x_step, r8:dilate_y_step
mov r4, r3
ldr r5, [sp, #24]
ldr r6, [sp, #28]
ldr r7, [sp, #32]
ldr r8, [sp, #36]

cmp r4, #0
vmov.i32 q0, #0
beq UnitEnd
cmp r5, #0
beq UnitEnd

mov lr, #4
mul r6, lr, r6
mul r7, lr, r7
mul r8, lr, r8

//dilate_y_step -> dilate_y_step - dilate_x_step*fw
mul lr, r4, r7
sub r8, r8, lr

//weight_y_step -> weight_y_step - 4*sizeof(float)*fw
mov lr, #16
mul lr, r4, lr
sub r6, r6, lr


UnitLoopH:
mov lr, r4
UnitLoopW:
vld1.32 {q1}, [r1], r7
vld1.32 {q2}, [r2]!
vmla.f32 q0, q1, q2
subs lr, lr, #1
bne UnitLoopW
subs r5, r5, #1
add r1, r1, r8
add r2, r2, r6
bne UnitLoopH


UnitEnd:

vst1.32 {q0}, [r0]

pop {r4-r8, pc}

#endif
#endif
