//
//  MNNCubicSampleC4.S
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

asm_function MNNCubicSampleC4
//void MNNCubicSampleC4(const float* src, float* dst, int* position, const float* factor, size_t number)

push {r4-r7, lr}

//Auto: r0:src, r1:dst, r2:position, r3:factor

//Load from sp:
//r4:number
ldr r4, [sp, #20]
vmov.i32 q15, #0x3f000000 //0.5
mov r12, #16//sizeof(float)*4

CubicSampleLoop:
vld1.32 {d2[0]}, [r3]!

vdup.f32 q14, d2[0]
vld1.32 {q13}, [r2]!
vdup.32 q12, r12
vmul.s32 q13, q12, q13

vmov.i32 r5, d26[0]
vmov.i32 r6, d26[1]

add r5, r0, r5
add r6, r0, r6

//A
vld1.32 {q0}, [r5]
vmov.i32 r7, d27[0]
vmov.i32 r5, d27[1]
//B
vld1.32 {q1}, [r6]
add r7, r0, r7
add r5, r0, r5
//C
vld1.32 {q2}, [r7]
vsub.f32 q8, q1, q0 //B-A
//D
vld1.32 {q3}, [r5]
vsub.f32 q9, q1, q2//B-C
vsub.f32 q10, q3, q2//D-C
vsub.f32 q11, q2, q0//C-A

//Compute a
//d = q1
vadd.f32 q12, q8, q10 //(D-C)+(B-A)
vadd.f32 q13, q9, q8 //(B-A)+(B-C), now B-A and B-C is no used
vmla.f32 q9, q12, q15//a=q9, now q12 is no used

vsub.f32 q2, q2, q13 //C-((B-A)+(B-C)), now q13 is no used
vadd.f32 q12, q1, q3//B+D
vmul.f32 q11, q11, q15//c=q11

vmls.f32 q2, q12, q15 //b=q2

vmla.f32 q2, q9, q14
vmla.f32 q11, q2, q14
vmla.f32 q1, q11, q14

vst1.32 {q1}, [r1]!

subs r4, r4, #1
bne CubicSampleLoop


pop {r4-r7, pc}

#endif
#endif
