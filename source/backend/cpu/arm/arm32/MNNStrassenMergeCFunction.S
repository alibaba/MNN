//
//  MNNStrassenMergeCFunction.S
//  MNN
//
//  Created by MNN on 2019/02/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __arm__
#ifndef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNStrassenMergeCFunction
//void MNNStrassenMergeCFunction(float* c11, float* c12, float* c21, float* c22,
//      float* xAddr, size_t cStride, size_t eSub, size_t hSub) {

push {r4-r8, r10, r11, lr} // avoid to touch platform-register r-9
//Auto: r0: c11, r1:c12, r2:c21, r3:c22
//Load from sp: r4: xAddr, r5: cStride, r6: eSub, r7: hSub

ldr r4, [sp, #32]
ldr r5, [sp, #36]
ldr r6, [sp, #40]
ldr r7, [sp, #44]

//r5 -> cExtraOffset
mov r12, #4 //sizeof(float)
mul r5, r12, r5
mov r11, #16
mul r11, r6, r11
sub r5, r5, r11

LoopY:
    XL1:
    subs r12, r6, #1
    vld1.32 {q8}, [r4]!//x
    vld1.32 {q12}, [r1]//c12
    vld1.32 {q9}, [r2]//c21
    vadd.f32 q12, q12, q8
    vld1.32 {q10}, [r3]//c22
    vadd.f32 q9, q9, q12
    beq XEnd

    LoopXL1:
        vadd.f32 q12, q12, q10
        vst1.32 {q9}, [r2]!
        vld1.32 {q11}, [r0]! //c11
        vadd.f32 q10, q10, q9
        vld1.32 {q8}, [r4]!//x
        vadd.f32 q12, q12, q11
        vst1.32 {q10}, [r3]!
        vst1.32 {q12}, [r1]!

        vld1.32 {q12}, [r1]//c12
        vld1.32 {q9}, [r2]//c21
        vadd.f32 q12, q12, q8
        vld1.32 {q10}, [r3]//c22
        vadd.f32 q9, q9, q12

        subs r12, r12, #1
        bne LoopXL1

    XEnd:

    vadd.f32 q12, q12, q10
    vst1.32 {q9}, [r2]!
    vld1.32 {q11}, [r0]! //c11
    vadd.f32 q10, q10, q9
    vadd.f32 q12, q12, q11
    vst1.32 {q10}, [r3]!
    vst1.32 {q12}, [r1]!

    add r0, r0, r5
    add r1, r1, r5
    add r2, r2, r5
    add r3, r3, r5


    subs r7, r7, #1
    bne LoopY


pop {r4-r8, r10, r11, pc}

#endif
#endif
