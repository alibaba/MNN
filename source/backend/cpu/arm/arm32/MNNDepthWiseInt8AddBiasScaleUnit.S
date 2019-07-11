//
//  MNNDepthWiseInt8AddBiasScaleUnit.S
//  MNN
//
//  Created by MNN on 2019/06/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __arm__
#ifndef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNDepthWiseInt8AddBiasScaleUnit

// void MNNDepthWiseInt8AddBiasScaleUnit(int8_t* dst, const int8_t* src, const int8_t* weight, 
//      const int32_t* bias, size_t fw, size_t fh, size_t weight_y_step, size_t dilateX_step, 
//      size_t dilateY_step, const float* scale)

// Auto
// r0: dst, r1: src, r2: weight, r3: bias
// Load from sp
// r4: fw, r5: fh, r6: weight_y_step, r7: dilateX_step, r8: dilateY_step, r9: scale

push {r4-r11, lr}

ldr r4, [sp, #36]
ldr r5, [sp, #40]
ldr r6, [sp, #44]
ldr r7, [sp, #48]
ldr r8, [sp, #52]
ldr r9, [sp, #56]

vld1.32 {q0}, [r3]
vld1.32 {q15}, [r9]

cmp r4, #0
beq EndUnit

cmp r5, #0
beq EndUnit


mov r10, #4
mul r10, r4, r10
sub r6, r6, r10

mul r10, r7, r4
sub r8, r8, r10

LoopFy:
    mov r12, r4
    LoopFx:
        vld1.32 {d4[0]}, [r1], r7
        vld1.32 {d16[0]}, [r2]!
        vmovl.s8 q2, d4
        vmovl.s8 q8, d16
        vmlal.s16 q0, d4, d16
        subs r12, r12, #1
        bne LoopFx
    subs r5, r5, #1
    add r1, r8, r1
    add r2, r2, r6
    bne LoopFy

EndUnit:

vcvt.f32.s32 q1, q0
vmul.f32 q2, q1, q15
vcvt.s32.f32 q0, q2

vqmovn.s32 d2, q0
vqmovn.s16 d4, q1
vst1.s32 {d4[0]}, [r0]

pop {r4-r11, pc}


#endif
#endif