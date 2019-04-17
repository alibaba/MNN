//
//  MNNGemmFloatOne_4.S
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

asm_function MNNGemmFloatOne_4
//void MNNGemmFloatOne_4(float* dst, const float* src, const float* weight, size_t src_depth_quad,
//                            size_t dst_step, size_t dst_depth_quad, size_t weight_depth_offset)

push {r4-r11, lr}

//Auto Load:
//r0:dst, r1:src, r2:weight, r3: src_depth_quad


//Load from sp
//r4:dst_step, r5:dst_depth_quad, r9:weight_depth_offset
ldr r4, [sp, #36]
ldr r5, [sp, #40]
ldr r9, [sp, #44]


//step multi by sizeof(float)
mov r12, #4
mul r4, r12, r4
mul r9, r12, r9

//r11: weight_dz_step
mov r12, #64 //16*sizeof(float)
mul r11, r12, r3
add r11, r9, r11

mov r6, r3
mov r10, r1

LoopDz:
mov r8, r0
mov r12, r2

L1:
cmp r3, #0
beq LZEnd

vld1.32 {q0}, [r1]!
vld1.32 {q8, q9}, [r2]!
vmul.f32 q2, q8, d0[0]
vld1.32 {q10, q11}, [r2]!
subs r3, r3, #1
vmul.f32 q3, q9, d0[1]
beq L1LoopZEnd
L1LoopZ:
    vld1.32 {q8, q9}, [r2]!
    vmla.f32 q2, q10, d1[0]
    vmla.f32 q3, q11, d1[1]
    vld1.32 {q0}, [r1]!
    vmla.f32 q2, q8, d0[0]
    vld1.32 {q10, q11}, [r2]!
    vmla.f32 q3, q9, d0[1]
    subs r3, r3, #1
    bne L1LoopZ
L1LoopZEnd:
vmla.f32 q2, q10, d1[0]
vmla.f32 q3, q11, d1[1]

vadd.f32 q0, q2, q3
vst1.32 {q0}, [r8]!

LZEnd:

subs r5, r5, #1
add r0, r0, r4
mov r1, r10
add r2, r12, r11
mov r3, r6
bne LoopDz

pop {r4-r11, pc}

#endif
#endif
