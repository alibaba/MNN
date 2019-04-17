//
//  MNNGemmInt16to32_4x4_Unit.S
//  MNN
//
//  Created by MNN on 2018/08/07.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNNAsmGlobal.h"
#ifdef __arm__
#ifndef __aarch64__

.text
.align 5
asm_function MNNGemmInt16to32_4x4_Unit
//void MNNGemmInt16to32_4x4_Unit(int32_t* dst, const int16_t* src, const int16_t* weight, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad)
//Auto:
//r0:dstOrigin, r1:src, r2: weight, r3:src_depth_quad

//Load from sp
//r4: dst_step, r5:dst_depth_quad

push {r4-r7, lr}
ldr r4, [sp, #20]
ldr r5, [sp, #24]
//step multi by sizeof(int32_t)
mov r12, #4
mul r4, r12, r4

vpush {q4-q7}

L12Dz:
    mov r6, r1
    mov r7, r3
    mov r12, r0

    subs r3, r3, #1
    
    vld1.32 {q4, q5}, [r2]!

    //A
    vld1.32 {q0}, [r1]!
    vmull.s16 q6, d8,  d0[0]
    vmull.s16 q7, d8,  d1[0]
    vmlal.s16 q6, d9,  d0[1]
    vmlal.s16 q7, d9,  d1[1]
    vmlal.s16 q6, d10, d0[2]
    vmlal.s16 q7, d10, d1[2]
    vmlal.s16 q6, d11, d0[3]
    vmlal.s16 q7, d11, d1[3]
    vld1.32 {q1}, [r1]!
    vmull.s16 q8, d8,  d2[0]
    vmull.s16 q9, d8,  d3[0]
    vmlal.s16 q8, d9,  d2[1]
    vmlal.s16 q9, d9,  d3[1]
    vmlal.s16 q8, d10, d2[2]
    vmlal.s16 q9, d10, d3[2]
    vmlal.s16 q8, d11, d2[3]
    vmlal.s16 q9, d11, d3[3]

    //B
    vld1.32 {q0}, [r1]!
    vmull.s16 q10, d8,  d0[0]
    vmull.s16 q11, d8,  d1[0]
    vmlal.s16 q10, d9,  d0[1]
    vmlal.s16 q11, d9,  d1[1]
    vmlal.s16 q10, d10, d0[2]
    vmlal.s16 q11, d10, d1[2]
    vmlal.s16 q10, d11, d0[3]
    vmlal.s16 q11, d11, d1[3]
    vld1.32 {q1}, [r1]!
    vmull.s16 q12, d8,  d2[0]
    vmull.s16 q13, d8,  d3[0]
    vmlal.s16 q12, d9,  d2[1]
    vmlal.s16 q13, d9,  d3[1]
    vmlal.s16 q12, d10, d2[2]
    vmlal.s16 q13, d10, d3[2]
    vmlal.s16 q12, d11, d2[3]
    vmlal.s16 q13, d11, d3[3]

    //C
    vld1.32 {q0}, [r1]!
    vmull.s16 q14, d8,  d0[0]
    vmull.s16 q15, d8,  d1[0]
    vmlal.s16 q14, d9,  d0[1]
    vmlal.s16 q15, d9,  d1[1]
    vmlal.s16 q14, d10, d0[2]
    vmlal.s16 q15, d10, d1[2]
    vmlal.s16 q14, d11, d0[3]
    vmlal.s16 q15, d11, d1[3]
    vld1.32 {q1}, [r1]!
    vmull.s16 q2, d8,  d2[0]
    vmull.s16 q3, d8,  d3[0]

    beq LoopZEnd

    LoopZ:
        vmlal.s16 q2, d9,  d2[1]
        vld1.32 {q0}, [r1]!
        vmlal.s16 q3, d9,  d3[1]
        vmlal.s16 q2, d10, d2[2]
        vmlal.s16 q3, d10, d3[2]
        vmlal.s16 q2, d11, d2[3]
        vmlal.s16 q3, d11, d3[3]
        
        vld1.32 {q4, q5}, [r2]!

        //A
        vmlal.s16 q6, d8,  d0[0]
        vmlal.s16 q7, d8,  d1[0]
        vld1.32 {q1}, [r1]!
        vmlal.s16 q6, d9,  d0[1]
        vmlal.s16 q7, d9,  d1[1]
        vmlal.s16 q6, d10, d0[2]
        vmlal.s16 q7, d10, d1[2]
        vmlal.s16 q6, d11, d0[3]
        vmlal.s16 q7, d11, d1[3]
        vmlal.s16 q8, d8,  d2[0]
        vmlal.s16 q9, d8,  d3[0]
        vmlal.s16 q8, d9,  d2[1]
        vld1.32 {q0}, [r1]!
        vmlal.s16 q9, d9,  d3[1]
        vmlal.s16 q8, d10, d2[2]
        vmlal.s16 q9, d10, d3[2]
        vmlal.s16 q8, d11, d2[3]
        vmlal.s16 q9, d11, d3[3]

        //B
        vmlal.s16 q10, d8,  d0[0]
        vmlal.s16 q11, d8,  d1[0]
        vmlal.s16 q10, d9,  d0[1]
        vld1.32 {q1}, [r1]!
        vmlal.s16 q11, d9,  d1[1]
        vmlal.s16 q10, d10, d0[2]
        vmlal.s16 q11, d10, d1[2]
        vmlal.s16 q10, d11, d0[3]
        vmlal.s16 q11, d11, d1[3]
        vmlal.s16 q12, d8,  d2[0]
        vmlal.s16 q13, d8,  d3[0]
        vmlal.s16 q12, d9,  d2[1]
        vmlal.s16 q13, d9,  d3[1]
        vld1.32 {q0}, [r1]!
        vmlal.s16 q12, d10, d2[2]
        vmlal.s16 q13, d10, d3[2]
        vmlal.s16 q12, d11, d2[3]
        vmlal.s16 q13, d11, d3[3]

        //C
        vmlal.s16 q14, d8,  d0[0]
        vmlal.s16 q15, d8,  d1[0]
        vmlal.s16 q14, d9,  d0[1]
        vmlal.s16 q15, d9,  d1[1]
        vld1.32 {q1}, [r1]!
        vmlal.s16 q14, d10, d0[2]
        vmlal.s16 q15, d10, d1[2]
        vmlal.s16 q14, d11, d0[3]
        vmlal.s16 q15, d11, d1[3]
        vmlal.s16 q2, d8,  d2[0]
        vmlal.s16 q3, d8,  d3[0]


        subs r3, r3, #1
        bne LoopZ

    LoopZEnd:

    vst1.32 {q6, q7}, [r0]!
    vmlal.s16 q2, d9,  d2[1]
    vst1.32 {q8, q9}, [r0]!
    vmlal.s16 q3, d9,  d3[1]
    vst1.32 {q10, q11}, [r0]!
    vmlal.s16 q2, d10, d2[2]
    vst1.32 {q12, q13}, [r0]!
    vmlal.s16 q3, d10, d3[2]
    vmlal.s16 q2, d11, d2[3]
    vst1.32 {q14, q15}, [r0]!
    vmlal.s16 q3, d11, d3[3]


    mov r1, r6
    mov r3, r7
    vst1.32 {q2, q3}, [r0]!

    subs r5, r5, #1
    add r0, r12, r4
    bne L12Dz


vpop {q4-q7}


pop {r4-r7, pc}

#endif
#endif
