//
//  MNNScaleAndAddBias.S
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

asm_function MNNScaleAndAddBias
//void MNNScaleAndAddBias(float* dst, const float* src, const float* bias, const float* alpha, size_t planeNumber, size_t biasNumber)

//Auto: r0:dst, r1:src, r2:bias, r3:alpha

//Load from sp: r4:planeNumber, r5:biasNumber

push {r4, r5, r6, lr}
ldr r4, [sp, #16]
ldr r5, [sp, #20]
ldr r6, [sp, #24]

cmp r4, #0
beq BSEnd

cmp r5, #0
beq BSEnd

BSLoopZ:
    mov r6, r4
    vld1.32 {q15}, [r2]!
    vld1.32 {q14}, [r3]!
    cmp r6, #3
    ble BSLoopP1
    BSLoopP4:
        vld1.32 {q0, q1}, [r1]!
        vmul.f32 q0, q0, q14
        vmul.f32 q1, q1, q14
        vld1.32 {q2, q3}, [r1]!
        vadd.f32 q0, q0, q15
        vadd.f32 q1, q1, q15
        vmul.f32 q2, q2, q14
        vmul.f32 q3, q3, q14
        vst1.32 {q0, q1}, [r0]!
        vadd.f32 q2, q2, q15
        vadd.f32 q3, q3, q15
        sub r6, r6, #4
        vst1.32 {q2, q3}, [r0]!
        cmp r6, #4
        bge BSLoopP4

    cmp r6, #0
    beq BSLoopPEnd

    BSLoopP1:
        vld1.32 {q0}, [r1]!
        vmul.f32 q0, q0, q14
        vadd.f32 q0, q0, q15
        vst1.32 {q0}, [r0]!
        subs r6, r6, #1
        bne BSLoopP1
    BSLoopPEnd:

    subs r5, r5, #1
    bne BSLoopZ


BSEnd:


pop {r4, r5, r6, pc}

#endif
#endif
