//
//  MNNScaleAndAddBiasInt8.S
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

asm_function MNNScaleAndAddBiasInt8
// MNNScaleAndAddBiasInt8(int8_t* dst, const int8_t* src, const int32_t* bias, const int32_t* alpha, int32_t mShiftBits,
// ssize_t minValue, ssize_t maxValue, int8_t* inputZeroPoint, int8_t* outputZeroPoint, ssize_t planeNumber, ssize_t biasNumber, ssize_t pack)

//Auto: r0:dst, r1:src, r2:bias, r3:alpha
//Load from sp: r4:mShiftBits, r5:minValue, r6:maxValue, r7:inputZeroPoint, r12:outputZeroPoint, r8:planeNumber, r10:biasNumber

push {r4-r8, r10-r12, lr}
ldr r4, [sp, #36]
ldr r5, [sp, #40]
ldr r6, [sp, #44]
ldr r7, [sp, #48]
ldr r12, [sp, #52]
ldr r8, [sp, #56]
ldr r10, [sp, #60]

vpush {q4-q7}
vdup.s8 q7, r5
vdup.s8 q8, r6

vld1.8 {d24[0]}, [r7]
vld1.8 {d26[0]}, [r12]
vdup.8 d24, d24[0]
vdup.8 d26, d26[0]

cmp r8, #0
beq BSEnd

cmp r10, #0
beq BSEnd

BSLoopZ:
    mov r11, r8
    vld1.32 {q15}, [r2]!
    vld1.32 {q14}, [r3]!

    cmp r11, #2
    blt BSLoopP1
    cmp r11, #4
    blt BSLoopP2

    BSLoopP4:
        vld1.8 {q0}, [r1]!       // q0: 4x(4xint8_t)
        vmovl.s8 q1, d0
        vmovl.s8 q2, d1
        vsubw.s8 q1, q1, d24
        vsubw.s8 q2, q2, d24
        vmovl.s16 q3, d2
        vmovl.s16 q4, d3
        vmovl.s16 q5, d4
        vmovl.s16 q6, d5

        vmul.s32 q3, q3, q14
        vmul.s32 q4, q4, q14
        vmul.s32 q5, q5, q14
        vmul.s32 q6, q6, q14

        vadd.s32 q3, q3, q15
        vadd.s32 q4, q4, q15
        vadd.s32 q5, q5, q15
        vadd.s32 q6, q6, q15

        vrshrn.s32 d6, q3, #15
        vrshrn.s32 d7, q4, #15
        vrshrn.s32 d10, q5, #15
        vrshrn.s32 d11, q6, #15

        vaddw.s8 q3, q3, d26
        vaddw.s8 q5, q5, d26

        vqmovn.s16 d6, q3
        vqmovn.s16 d7, q5

        vmax.s8 q3, q3, q7
        vmin.s8 q3, q3, q8

        vst1.s8 {q3}, [r0]!

        sub r11, r11, #4
        cmp r11, #4
        bge BSLoopP4

    cmp r11, #0
    beq BSLoopPEnd
    cmp r11, #2
    blt BSLoopP1

    BSLoopP2:
        vld1.8 {d0}, [r1]!       // q0: 2x(4xint8_t)
        //vsub.s8 d0, d0, d24
        vmovl.s8 q1, d0
        vsubw.s8 q1, q1, d24
        vmovl.s16 q3, d2
        vmovl.s16 q4, d3

        vmul.s32 q3, q3, q14
        vmul.s32 q4, q4, q14

        vadd.s32 q3, q3, q15
        vadd.s32 q4, q4, q15

        vrshrn.s32 d6, q3, #15
        vrshrn.s32 d7, q4, #15
        vaddw.s8 q3, q3, d26

        vqmovn.s16 d6, q3

        vmax.s8 d6, d6, d14
        vmin.s8 d6, d6, d16

        vst1.s8 {d6}, [r0]!

        sub r11, r11, #2
        cmp r11, #2
        bge BSLoopP2

    cmp r11, #0
    beq BSLoopPEnd

    BSLoopP1:
        ldr lr, [r1], #4
        vdup.32 d0, lr

        vmovl.s8 q1, d0
        vsubw.s8 q1, q1, d24
        vmovl.s16 q3, d2

        vmul.s32 q3, q3, q14
        vadd.s32 q3, q3, q15

        vrshrn.s32 d6, q3, #15
        vmov.32 d7, d6
        vaddw.s8 q3, q3, d26
        vqmovn.s16 d6, q3
        
        vmax.s8 d6, d6, d14
        vmin.s8 d6, d6, d16

        vst1.32 {d6[0]}, [r0]!

        sub r11, r11, #1
        cmp r11, #1
        bge BSLoopP1

    BSLoopPEnd:

    subs r10, r10, #1
    bne BSLoopZ


BSEnd:

vpop {q4-q7}
pop {r4-r8, r10-r12, pc}

#endif
#endif
