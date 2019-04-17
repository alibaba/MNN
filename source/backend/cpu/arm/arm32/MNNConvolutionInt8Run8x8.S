//
//  MNNConvolutionInt8Run8x8.S
//  MNN
//
//  Created by MNN on 2018/07/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __arm__
#ifndef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNConvolutionInt8Run8x8
//void MNNConvolutionInt8Run8x8(int16_t* dst_x, const int8_t* src_unit, const int8_t* weight_start,
//                                size_t icD8, size_t xCount, size_t yCount,
//                                size_t src_z_step, size_t dilate_y_step, size_t dilate_x_step,
//                                size_t weight_sz_step, size_t weight_sy_step
//                                );


push {r4-r11, lr}

//Auto
//r0: dst_x, r1:src_unit, r2: weight_start, r3: icD8


//Load
//r4: xCount, r5:yCount,r6:dilate_y_step, r7:dilate_x_step, r8:weight_sy_step

ldr r4, [sp, #36]
ldr r5, [sp, #40]
ldr r6, [sp, #44]
ldr r7, [sp, #48]
ldr r8, [sp, #52]

//Dst: q8-q15
vmov.i16 q15, #0
vmov.i16 q14, #0
vmov.i16 q13, #0
vmov.i16 q12, #0
vmov.i16 q11, #0
vmov.i16 q10, #0
vmov.i16 q9, #0
vmov.i16 q8, #0

LoopFY:
    mov r11, r4
    LoopFX:
        mov r12, r3
        LoopZ:
            vld1.32 {d0}, [r1]!

            vld1.32 {q1}, [r2]!

            vmlal.s8 q8, d0, d2

            vld1.32 {q2}, [r2]!
            vmlal.s8 q9, d0, d3
            vmlal.s8 q10, d0, d4
            vld1.32 {q3}, [r2]!
            vmlal.s8 q11, d0, d5
            vld1.32 {q1}, [r2]!
            vmlal.s8 q12, d0, d6
            vmlal.s8 q13, d0, d7
            vmlal.s8 q14, d0, d2
            vmlal.s8 q15, d0, d3

            subs r12, r12, #1
            
            bne LoopZ

        subs r11, r11, #1
        add r1, r1, r7
        bne LoopFX
    add r2, r2, r8
    add r1, r1, r6
    subs r5, r5, #1
    bne LoopFY

//Transpose
//Swap Step 1
vtrn.16 q8, q9
vtrn.16 q10, q11
vtrn.16 q12, q13
vtrn.16 q14, q15

//Swap Step 2
vtrn.32 q8, q10
vtrn.32 q9, q11
vtrn.32 q12, q14
vtrn.32 q13, q15

//Swap Step 3
vswp d17, d24
vswp d19, d26
vswp d21, d28
vswp d23, d30


vqadd.s16 q0, q8, q9
vqadd.s16 q1, q10, q11
vqadd.s16 q2, q12, q13
vqadd.s16 q3, q14, q15

vqadd.s16 q0, q0, q1
vqadd.s16 q2, q2, q3

vqadd.s16 q0, q0, q2


vst1.32 {q0}, [r0]!



pop {r4-r11, pc}


#endif
#endif
