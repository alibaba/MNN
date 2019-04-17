//
//  MNNCubicLineC4.S
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

asm_function MNNCubicLineC4
//void MNNCubicLineC4(float* dst, const float* A, const float* B, const float* C, const float* D, float* t, size_t number)
push {r4-r7, lr}
//Auto: r0:dst, r1:A, r2:B, r3:C

//Load from sp:
//r4: D, r5:t, r6:number
ldr r4, [sp, #20]
ldr r5, [sp, #24]
ldr r6, [sp, #28]

vld1.32 {d2[0]}, [r5]

vdup.32 q14, d2[0]
vmov.i32 q15, #0x3f000000 //0.5

subs r6, r6, #1
vld1.32 {q0}, [r1]!
vld1.32 {q1}, [r2]!
vsub.f32 q8, q1, q0 //B-A
vld1.32 {q2}, [r3]!
vsub.f32 q9, q1, q2//B-C
vld1.32 {q3}, [r4]!

vsub.f32 q10, q3, q2//D-C
vsub.f32 q11, q2, q0//C-A
beq MNNCubicLineC4LoopEnd

//Compute a
//d = q1
vadd.f32 q12, q8, q10 //(D-C)+(B-A)
vadd.f32 q13, q9, q8 //(B-A)+(B-C), now B-A and B-C is no used
vmla.f32 q9, q12, q15//a=q9, now q12 is no used

MNNCubicLineC4Loop:
    vsub.f32 q2, q2, q13 //C-((B-A)+(B-C)), now q13 is no used
    vadd.f32 q12, q1, q3//B+D
    vmul.f32 q11, q11, q15//c=q11

    vmls.f32 q2, q12, q15 //b=q2

    vmla.f32 q2, q9, q14
    vld1.32 {q0}, [r1]!
    vmla.f32 q11, q2, q14
    vld1.32 {q3}, [r4]!
    vmla.f32 q1, q11, q14
    vld1.32 {q2}, [r3]!

    vst1.32 {q1}, [r0]!

    vld1.32 {q1}, [r2]!
    
    vsub.f32 q8, q1, q0 //B-A
    vsub.f32 q9, q1, q2//B-C
    vsub.f32 q10, q3, q2//D-C
    vsub.f32 q11, q2, q0//C-A

//Compute a
//d = q1
vadd.f32 q12, q8, q10 //(D-C)+(B-A)
vadd.f32 q13, q9, q8 //(B-A)+(B-C), now B-A and B-C is no used
vmla.f32 q9, q12, q15//a=q9, now q12 is no used

subs r6, r6, #1
bne MNNCubicLineC4Loop

MNNCubicLineC4LoopEnd:
vsub.f32 q2, q2, q13 //C-((B-A)+(B-C)), now q13 is no used
vadd.f32 q12, q1, q3//B+D
vmul.f32 q11, q11, q15//c=q11

vmls.f32 q2, q12, q15 //b=q2

vmla.f32 q2, q9, q14
vmla.f32 q11, q2, q14
vmla.f32 q1, q11, q14

vst1.32 {q1}, [r0]

pop {r4-r7, pc}
//MNNCubicLineC4 End
#endif
#endif
