//
//  MNNVectorTop1Float.S
//  MNN
//
//  Created by MNN on 2020/12/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __arm__
#ifndef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNVectorTop1Float
// void MNNVectorTop1Float(float* input, float* maxValue, int32_t* maxIndex, size_t inputCountUnit);

push {r4-r8, r10, r11, lr} // avoid to touch platform-register r-9

// Auto: r0: input, r1: maxValue, r2: maxIndex, r3: inputCountUnit

// q15 maxValue
vld1.f32 {q15}, [r0]

// q14 maxIndex
mov r11, #0
vmov.s32 d28[0], r11
mov r11, #1
vmov.s32 d28[1], r11
mov r11, #2
vmov.s32 d29[0], r11
mov r11, #3
vmov.s32 d29[1], r11

// q11 current index
vmov.s32 q11, q14

// all 4, increment
mov r11, #4
vmov.s32 d20[0], r11
vmov.s32 d20[1], r11
vmov.s32 d21[0], r11
vmov.s32 d21[1], r11


cmp r3, #0
beq End

Loop:
    vld1.f32 {q13}, [r0]!

    vcgt.f32 q12, q13, q15
    vbit.f32 q15, q13, q12
    vbit.s32 q14, q11, q12

    vadd.s32 q11, q11, q10
    subs r3, r3, #1

    bne Loop

// reduce result to single value and index
vcgt.f32 d24, d31, d30
vbit.f32 d30, d31, d24
vbit.s32 d28, d29, d24

vtrn.f32 d30, d31
vtrn.s32 d28, d29

vcgt.f32 d24, d31, d30
vbit.f32 d30, d31, d24
vbit.s32 d28, d29, d24

End:
    vst1.f32 d30[0], [r1]
    vst1.s32 d28[0], [r2]
    pop {r4-r8, r10, r11, pc}


#endif
#endif
