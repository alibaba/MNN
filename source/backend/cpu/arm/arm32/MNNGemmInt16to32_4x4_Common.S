//
//  MNNGemmInt16to32_4x4_Common.S
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
asm_function MNNGemmInt16to32_4x4_Common
//void MNNGemmInt16to32_4x4_Common(int32_t* dst, const int16_t* src, const int16_t* weight, size_t src_depth_quad, size_t width, size_t dst_step, size_t dst_depth_quad)
//Auto:
//r0:dstOrigin, r1:src, r2: weight, r3:src_depth_quad

//Load from sp
//r4: width, r5: dst_step, r6:dst_depth_quad

push {r4-r11, lr}
ldr r4, [sp, #36]
ldr r5, [sp, #40]
ldr r6, [sp, #44]
//step multi by sizeof(int32_t)
mov r12, #4
mul r5, r12, r5

//r11: weight_dz_step
mov r12, #32//16*sizeof(int16_t)
mul r11, r12, r3

//r12:src_step
mov r12, #8//4*sizeof(int16_t)
mul r12, r4, r12

LoopDz:
    push {r0, r1, r2, r4}

    cmp r4, #8
    blt L4
    
    L8Loop:
        mov r8, r1
        mov r9, r3
        mov r10, r2
    
        vmov.i32 q8, #0
        vmov.i32 q9, #0
        vmov.i32 q10, #0
        vmov.i32 q11, #0
        vmov.i32 q12, #0
        vmov.i32 q13, #0
        vmov.i32 q14, #0
        vmov.i32 q15, #0
    
        LoopZL8:
            vld1.32 {q2, q3}, [r2]!
    
            //A
            vld1.32 {q0}, [r1]!
            vmlal.s16 q8, d4, d0[0]
            vmlal.s16 q9, d4, d1[0]
            vmlal.s16 q8, d5, d0[1]
            vmlal.s16 q9, d5, d1[1]
            vmlal.s16 q8, d6, d0[2]
            vmlal.s16 q9, d6, d1[2]
            vmlal.s16 q8, d7, d0[3]
            vmlal.s16 q9, d7, d1[3]
    
            //B
            vld1.32 {q1}, [r1]!
            vmlal.s16 q10, d4, d2[0]
            vmlal.s16 q11, d4, d3[0]
            vmlal.s16 q10, d5, d2[1]
            vmlal.s16 q11, d5, d3[1]
            vmlal.s16 q10, d6, d2[2]
            vmlal.s16 q11, d6, d3[2]
            vmlal.s16 q10, d7, d2[3]
            vmlal.s16 q11, d7, d3[3]
    
            //C
            vld1.32 {q0}, [r1]!
            vmlal.s16 q12, d4, d0[0]
            vmlal.s16 q13, d4, d1[0]
            vmlal.s16 q12, d5, d0[1]
            vmlal.s16 q13, d5, d1[1]
            vmlal.s16 q12, d6, d0[2]
            vmlal.s16 q13, d6, d1[2]
            vmlal.s16 q12, d7, d0[3]
            vmlal.s16 q13, d7, d1[3]
    
            //D
            vld1.32 {q1}, [r1]!
            vmlal.s16 q14, d4, d2[0]
            vmlal.s16 q15, d4, d3[0]
            vmlal.s16 q14, d5, d2[1]
            vmlal.s16 q15, d5, d3[1]
            vmlal.s16 q14, d6, d2[2]
            vmlal.s16 q15, d6, d3[2]
            vmlal.s16 q14, d7, d2[3]
            vmlal.s16 q15, d7, d3[3]

            sub r1, r1, #64
            add r1, r1, r12
    
            subs r3, r3, #1
            bne LoopZL8
        vst1.32 {q8, q9}, [r0]!
        vst1.32 {q10, q11}, [r0]!
        vst1.32 {q12, q13}, [r0]!
        vst1.32 {q14, q15}, [r0]!
        add r1, r8, #64 // 8*4*sizeof(int16_t)
        mov r3, r9
        mov r2, r10

        sub r4, r4, #8
        cmp r4, #8
        bge L8Loop
    L4:
        cmp r4, #4
        blt L1
        vmov.i32 q12, #0
        vmov.i32 q13, #0
        vmov.i32 q14, #0
        vmov.i32 q15, #0
        mov r8, r1
        mov r9, r3
        mov r10, r2


        LoopZL4:
            vld1.32 {q2, q3}, [r2]!

            vld1.32 {q0}, [r1]!
            vmlal.s16 q12, d4, d0[0]
            vmlal.s16 q13, d4, d1[0]
            vmlal.s16 q12, d5, d0[1]
            vmlal.s16 q13, d5, d1[1]
            vld1.32 {q1}, [r1]!
            vmlal.s16 q12, d6, d0[2]
            vmlal.s16 q13, d6, d1[2]
            vmlal.s16 q12, d7, d0[3]
            vmlal.s16 q13, d7, d1[3]

            vmlal.s16 q14, d4, d2[0]
            vmlal.s16 q15, d4, d3[0]
            vmlal.s16 q14, d5, d2[1]
            vmlal.s16 q15, d5, d3[1]
            vmlal.s16 q14, d6, d2[2]
            vmlal.s16 q15, d6, d3[2]
            vmlal.s16 q14, d7, d2[3]
            vmlal.s16 q15, d7, d3[3]

            subs r3, r3, #1
            sub r1, r1, #32
            add r1, r1, r12
            bne LoopZL4

        sub r4, r4, #4
        add r1, r8, #32
        mov r3, r9
        mov r2, r10
        vst1.32 {q12, q13}, [r0]!
        vst1.32 {q14, q15}, [r0]!


    L1:
    cmp r4, #0
    beq EndL1

    LoopL1:
        mov r10, r2
        mov r8, r1
        mov r9, r3

        vmov.i32 q15, #0
        vmov.i32 q14, #0

        LoopZL1:
            vld1.32 {q2, q3}, [r2]!
            vld1.32 {d0}, [r1], r12

            vmlal.s16 q14, d4, d0[0]
            vmlal.s16 q15, d5, d0[1]
            vmlal.s16 q14, d6, d0[2]
            vmlal.s16 q15, d7, d0[3]

            subs r3, r3, #1
            bne LoopZL1

        subs r4, r4, #1
        vadd.s32 q15, q15, q14
        mov r2, r10
        mov r3, r9
        vst1.32 {q15}, [r0]!
        add r1, r8, #8
        bne LoopL1

    EndL1:

    pop {r0, r1, r2, r4}
    subs r6, r6, #1

    add r2, r2, r11
    add r0, r0, r5
    bne LoopDz


pop {r4-r11, pc}


#endif
#endif
