//
//  MNNGemmFloatUnit_4.S
//  MNN
//
//  Created by MNN on 2019/02/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNNAsmGlobal.h"
#ifdef __arm__
#ifndef __aarch64__

.text
.align 5

asm_function MNNGemmFloatUnit_4
//void MNNGemmFloatUnit_4(float* dstOrigin, const float* src, const float* weight, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad, size_t weight_depth_offset)

//Auto:
//r0:dstOrigin, r1:src, r2: weight, r3:src_depth_quad

//Load from sp

//r4: dst_step, r5:dst_depth_quad
//r8: weightExtraOffset

push {r4-r8, lr}
ldr r4, [sp, #24]
ldr r5, [sp, #28]
ldr r8, [sp, #32]
//step multi by sizeof(float)
mov r12, #4
mul r4, r12, r4
mul r8, r12, r8

vpush {q4-q7}

L8Dz:
    mov r6, r1
    mov r7, r3
    mov r12, r0
    subs r3, r3, #1
    vld1.32 {q4, q5}, [r2]!
    vld1.32 {q6, q7}, [r2]!
    vld1.32 {q0}, [r1]!

    vmul.f32 q8, q4, d0[0]

    vld1.32 {q1}, [r1]!

    vmla.f32 q8, q5, d0[1]
    vmul.f32 q9, q4, d2[0]
    vmla.f32 q8, q6, d1[0]

    vld1.32 {q2}, [r1]!

    vmla.f32 q8, q7, d1[1]
    vmla.f32 q9, q5, d2[1]
    vmul.f32 q10, q4, d4[0]

    vld1.32 {q0}, [r1]!

    vmla.f32 q9, q6, d3[0]
    vmul.f32 q11, q7, d1[1]
    vmla.f32 q10, q5, d4[1]
    vmla.f32 q11, q5, d0[1]
    vmla.f32 q9, q7, d3[1]
    vmla.f32 q11, q6, d1[0]
    vld1.32 {q1}, [r1]!
    vmla.f32 q10, q6, d5[0]
    vmla.f32 q11, q4, d0[0]

    vmla.f32 q10, q7, d5[1]
    vmul.f32 q12, q4, d2[0]

    vld1.32 {q2}, [r1]!

    vmla.f32 q12, q5, d2[1]
    vmul.f32 q13, q4, d4[0]
    
    vld1.32 {q0}, [r1]!

    vmla.f32 q12, q6, d3[0]
    vmla.f32 q13, q5, d4[1]
    vmla.f32 q12, q7, d3[1]
    vld1.32 {q1}, [r1]!
    vmul.f32 q14, q4, d0[0]
    vmla.f32 q13, q6, d5[0]
    vmul.f32 q15, q4, d2[0]
    beq L8LoopZEnd
    L8LoopZ:

        vmla.f32 q14, q5, d0[1]
        vmla.f32 q15, q5, d2[1]
        vmla.f32 q14, q6, d1[0]
        vmla.f32 q15, q6, d3[0]
        vmla.f32 q14, q7, d1[1]
        vmla.f32 q15, q7, d3[1]
        vld1.32 {q0, q1}, [r1]!
        vmla.f32 q13, q7, d5[1]
        vld1.32 {q4, q5}, [r2]!
        vmla.f32 q8, q4, d0[0]
        vmla.f32 q9, q4, d2[0]
        vld1.32 {q6, q7}, [r2]!
        vmla.f32 q8, q5, d0[1]

        vmla.f32 q9, q5, d2[1]
        vmla.f32 q8, q6, d1[0]
        vld1.32 {q2}, [r1]!

        vmla.f32 q8, q7, d1[1]
        vmla.f32 q10, q4, d4[0]

        vld1.32 {q0}, [r1]!

        vmla.f32 q9, q6, d3[0]
        vmla.f32 q11, q7, d1[1]
        vmla.f32 q10, q5, d4[1]
        vmla.f32 q11, q5, d0[1]
        vmla.f32 q9, q7, d3[1]
        vmla.f32 q11, q6, d1[0]
        vld1.32 {q1}, [r1]!
        vmla.f32 q10, q6, d5[0]
        vmla.f32 q11, q4, d0[0]

        vmla.f32 q10, q7, d5[1]
        vmla.f32 q12, q4, d2[0]

        vld1.32 {q2}, [r1]!

        vmla.f32 q12, q5, d2[1]
        vmla.f32 q13, q4, d4[0]
        
        vld1.32 {q0}, [r1]!

        vmla.f32 q12, q6, d3[0]
        vmla.f32 q13, q5, d4[1]
        vmla.f32 q12, q7, d3[1]
        vld1.32 {q1}, [r1]!
        vmla.f32 q14, q4, d0[0]
        vmla.f32 q13, q6, d5[0]
        vmla.f32 q15, q4, d2[0]

        subs r3, r3, #1
        bne L8LoopZ
    L8LoopZEnd:
    vst1.32 {q8, q9}, [r0]!
    vmla.f32 q14, q5, d0[1]
    vmla.f32 q15, q5, d2[1]
    vst1.32 {q10, q11}, [r0]!
    vmla.f32 q13, q7, d5[1]
    vmla.f32 q15, q6, d3[0]
    vst1.32 {q12, q13}, [r0]!
    vmla.f32 q14, q7, d1[1]
    vmla.f32 q15, q7, d3[1]
    vmla.f32 q14, q6, d1[0]

    vst1.32 {q14, q15}, [r0]!
    mov r1, r6
    mov r3, r7

    subs r5, r5, #1
    add r2, r2, r8
    add r0, r12, r4
    bne L8Dz


vpop {q4-q7}


pop {r4-r8, pc}

#endif
#endif
