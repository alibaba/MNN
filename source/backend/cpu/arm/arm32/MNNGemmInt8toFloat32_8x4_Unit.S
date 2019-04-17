//
//  MNNGemmInt8toFloat32_8x4_Unit.S
//  MNN
//
//  Created by MNN on 2018/12/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __arm__
#ifndef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNGemmInt8toFloat32_8x4_Unit
//void MNNGemmInt8toFloat32_8x4_Unit(float* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad);
push {r4, r5, r6, lr}

//Auto: r0: dst, r1: src, r2:weight, r3: src_depth_quad
//Load from sp: r4: dst_step, r5: dst_depth_quad

ldr r4, [sp, #16]
ldr r5, [sp, #20]


vpush {q4-q7}

L2LoopDz:
    mov r6, r1
    subs r12, r3, #1
    vld1.8 {q4}, [r1]!
    vld1.8 {q5, q6}, [r2]!

    vmull.s8 q0, d8, d10
    vpaddl.s16 q8, q0
    vmull.s8 q1, d8, d11
    vpaddl.s16 q9, q1
    vmull.s8 q0, d8, d12
    vpaddl.s16 q10, q0
    vmull.s8 q1, d8, d13
    vpaddl.s16 q11, q1
    vmull.s8 q0, d9, d10
    vpaddl.s16 q12, q0
    vmull.s8 q1, d9, d11
    vpaddl.s16 q13, q1
    vmull.s8 q0, d9, d12
    vpaddl.s16 q14, q0
    vmull.s8 q1, d9, d13
    vpaddl.s16 q15, q1
    beq L2LoopSzEnd

    subs r12, r12, #1
    vld1.8 {q4}, [r1]!
    vld1.8 {q5, q6}, [r2]!
    vmull.s8 q0, d8, d10

    beq L2LoopSzEndAdd
    L2LoopSz:
        vpadal.s16 q8, q0
        vmull.s8 q1, d8, d11
        vmull.s8 q2, d8, d12
        vpadal.s16 q9, q1
        vmull.s8 q3, d8, d13
        vpadal.s16 q10, q2
        vmull.s8 q0, d9, d10
        vpadal.s16 q11, q3
        vmull.s8 q1, d9, d11
        vpadal.s16 q12, q0
        vld1.8 {q5}, [r2]!
        vmull.s8 q0, d9, d12
        vpadal.s16 q13, q1
        vmull.s8 q2, d9, d13
        vpadal.s16 q14, q0
        vld1.8 {q6}, [r2]!
        vpadal.s16 q15, q2

        vld1.8 {q4}, [r1]!
        vmull.s8 q0, d8, d10

        subs r12, r12, #1
        bne L2LoopSz

    L2LoopSzEndAdd:
    vpadal.s16 q8, q0
    vmull.s8 q1, d8, d11
    vpadal.s16 q9, q1
    vmull.s8 q0, d8, d12
    vpadal.s16 q10, q0
    vmull.s8 q1, d8, d13
    vpadal.s16 q11, q1
    vmull.s8 q0, d9, d10
    vpadal.s16 q12, q0
    vmull.s8 q1, d9, d11
    vpadal.s16 q13, q1
    vmull.s8 q0, d9, d12
    vpadal.s16 q14, q0
    vmull.s8 q1, d9, d13
    vpadal.s16 q15, q1

    L2LoopSzEnd:

    vpadd.s32 d16, d16, d17
    vpadd.s32 d20, d20, d21
    vpadd.s32 d18, d18, d19
    vpadd.s32 d22, d22, d23

    vpadd.s32 d24, d24, d25
    vpadd.s32 d28, d28, d29
    vpadd.s32 d26, d26, d27
    vpadd.s32 d30, d30, d31

    vpadd.s32 d16, d16, d18
    vpadd.s32 d17, d20, d22
    vpadd.s32 d18, d24, d26
    vcvt.f32.s32 q8, q8
    vpadd.s32 d19, d28, d30

    vcvt.f32.s32 q9, q9

    vst1.32 {q8, q9}, [r0], r4

    subs r5, r5, #1
    mov r1, r6
    bne L2LoopDz

vpop {q4-q7}
pop {r4, r5, r6, pc}


#endif
#endif
