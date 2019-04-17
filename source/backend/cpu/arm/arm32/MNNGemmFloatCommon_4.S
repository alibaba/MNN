//
//  MNNGemmFloatCommon_4.S
//  MNN
//
//  Created by MNN on 2018/03/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __arm__
#ifndef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNGemmFloatCommon_4
//void MNNGemmFloatCommon_4(float* dst, const float* src, const float* weight, size_t src_depth_quad,
//                            size_t dst_step, size_t dst_depth_quad, size_t width, size_t weight_depth_offset)

push {r4-r11, lr}

//Auto Load:
//r0:dst, r1:src, r2:weight, r3: src_depth_quad


//Load from sp
//r4:dst_step, r5:dst_depth_quad, r6:width
ldr r4, [sp, #36]
ldr r5, [sp, #40]
ldr r6, [sp, #44]
ldr r9, [sp, #48]

vpush {q4-q7}

//step multi by sizeof(float)
mov r12, #4
mul r4, r12, r4
mul r9, r12, r9

//r7: src_z_step
mov r12, #16//4*sizeof(float)
mul r7, r12, r6

//r11: weight_dz_step
mov r12, #64 //16*sizeof(float)
mul r11, r12, r3
add r11, r9, r11


mov r9, r6
LoopDz:
mov r8, r0
mov r10, r1
mov r12, r2

.macro START_TWO z0 z1
vld1.32 {q0}, [r1]!
vmul.f32 \z0, q2, d0[0]
vld1.32 {q1}, [r1]!
vmla.f32 \z0, q3, d0[1]
vmul.f32 \z1, q2, d2[0]
vmla.f32 \z0, q4, d1[0]
vmla.f32 \z1, q3, d2[1]
vmla.f32 \z0, q5, d1[1]
vmla.f32 \z1, q4, d3[0]
vmla.f32 \z1, q5, d3[1]
.endm

.macro COMPUTE_TWO z0 z1
vld1.32 {q0}, [r1]!
vmla.f32 \z0, q2, d0[0]
vld1.32 {q1}, [r1]!
vmla.f32 \z0, q3, d0[1]
vmla.f32 \z1, q2, d2[0]
vmla.f32 \z0, q4, d1[0]
vmla.f32 \z1, q3, d2[1]
vmla.f32 \z0, q5, d1[1]
vmla.f32 \z1, q4, d3[0]
vmla.f32 \z1, q5, d3[1]
.endm

.macro START_FOUR z0 z1 z2 z3
vld1.32 {q0}, [r1]!
vmul.f32 \z0, q2, d0[0]
vld1.32 {q1}, [r1]!
vmla.f32 \z0, q3, d0[1]
vmul.f32 \z1, q2, d2[0]
vmla.f32 \z0, q4, d1[0]
vmla.f32 \z1, q3, d2[1]
vmla.f32 \z0, q5, d1[1]
vmla.f32 \z1, q4, d3[0]
vld1.32 {q0}, [r1]!
vmla.f32 \z1, q5, d3[1]
vmul.f32 \z2, q2, d0[0]
vld1.32 {q1}, [r1]!
vmla.f32 \z2, q3, d0[1]
vmul.f32 \z3, q2, d2[0]
vmla.f32 \z2, q4, d1[0]
vmla.f32 \z3, q3, d2[1]
vmla.f32 \z2, q5, d1[1]
vmla.f32 \z3, q4, d3[0]
vmla.f32 \z3, q5, d3[1]
.endm

.macro COMPUTE_FOUR z0 z1 z2 z3
vld1.32 {q0}, [r1]!
vmla.f32 \z0, q2, d0[0]
vld1.32 {q1}, [r1]!
vmla.f32 \z0, q3, d0[1]
vmla.f32 \z1, q2, d2[0]
vmla.f32 \z0, q4, d1[0]
vmla.f32 \z1, q3, d2[1]
vmla.f32 \z0, q5, d1[1]
vmla.f32 \z1, q4, d3[0]
vld1.32 {q0}, [r1]!
vmla.f32 \z1, q5, d3[1]
vmla.f32 \z2, q2, d0[0]
vld1.32 {q1}, [r1]!
vmla.f32 \z2, q3, d0[1]
vmla.f32 \z3, q2, d2[0]
vmla.f32 \z2, q4, d1[0]
vmla.f32 \z3, q3, d2[1]
vmla.f32 \z2, q5, d1[1]
vmla.f32 \z3, q4, d3[0]
vmla.f32 \z3, q5, d3[1]
.endm

L10:
cmp r6, #10
blt L4


L10Loop:
    push {r1, r2, r3}

    vld1.32 {q2, q3}, [r2]!
    vld1.32 {q4, q5}, [r2]!
    
    START_TWO q6, q7
    START_TWO q8, q9
    START_TWO q10, q11
    START_TWO q12, q13
    START_TWO q14, q15

    subs r3, r3, #1
    beq L10LoopZEnd
    L10LoopZ:
        sub r1, r1, #160
        vld1.32 {q2, q3}, [r2]!
        add r1, r1, r7
        vld1.32 {q4, q5}, [r2]!

        COMPUTE_TWO q6, q7
        COMPUTE_TWO q8, q9
        COMPUTE_TWO q10, q11
        COMPUTE_TWO q12, q13
        COMPUTE_TWO q14, q15

        subs r3, r3, #1
        bne L10LoopZ
    L10LoopZEnd:
    pop {r1, r2, r3}
    add r1, r1, #160
    vst1.32 {q6, q7}, [r8]!
    vst1.32 {q8, q9}, [r8]!
    vst1.32 {q10, q11}, [r8]!
    sub r6, r6, #10
    vst1.32 {q12, q13}, [r8]!
    cmp r6, #10
    vst1.32 {q14, q15}, [r8]!
    bge L10Loop

L4:
cmp r6, #3
ble L2


L4Loop:
    vmov.i32 d30[0], r1
    vmov.i32 d30[1], r2
    vmov.i32 d31[1], r3
    vld1.32 {q2, q3}, [r2]!
    vld1.32 {q4, q5}, [r2]!

    START_TWO q8, q9
    START_TWO q10, q11
    subs r3, r3, #1
    beq L4LoopZEnd
    L4LoopZ:
        sub r1, r1, #64
        vld1.32 {q2, q3}, [r2]!
        add r1, r1, r7
        vld1.32 {q4, q5}, [r2]!
        COMPUTE_TWO q8, q9
        COMPUTE_TWO q10, q11
        subs r3, r3, #1
        bne L4LoopZ
    L4LoopZEnd:
    vmov.i32 r1, d30[0]
    add r1, r1, #64
    vmov.i32 r2, d30[1]
    vst1.32 {q8, q9}, [r8]!
    sub r6, r6, #4
    vmov.i32 r3, d31[1]
    cmp r6, #4
    vst1.32 {q10, q11}, [r8]!
    bge L4Loop

L2:
cmp r6, #2
blt L1


L2Loop:
    vmov.i32 d30[0], r1
    vmov.i32 d30[1], r2
    vmov.i32 d31[1], r3
    vld1.32 {q2, q3}, [r2]!
    vld1.32 {q4, q5}, [r2]!

    START_TWO q8, q9
    subs r3, r3, #1
    beq L2LoopZEnd
    L2LoopZ:
        sub r1, r1, #32
        vld1.32 {q2, q3}, [r2]!
        add r1, r1, r7
        vld1.32 {q4, q5}, [r2]!
        COMPUTE_TWO q8, q9
        subs r3, r3, #1
        bne L2LoopZ
    L2LoopZEnd:
    vmov.i32 r1, d30[0]
    add r1, r1, #32
    vmov.i32 r2, d30[1]
    vst1.32 {q8, q9}, [r8]!
    sub r6, r6, #2
    vmov.i32 r3, d31[1]
    cmp r6, #2
    bge L2Loop


L1:
cmp r6, #0
beq End

L1Loop:
    vmov.i32 d16[0], r1
    vmov.i32 d16[1], r2
    vmov.i32 d17[0], r3
    vld1.32 {q3}, [r1], r7
    vld1.32 {q4, q5}, [r2]!
    vmul.f32 q0, q4, d6[0]
    vld1.32 {q6, q7}, [r2]!
    vmul.f32 q1, q5, d6[1]
    subs r3, r3, #1
    beq L1LoopZEnd
    L1LoopZ:
        vld1.32 {q4, q5}, [r2]!
        vmla.f32 q0, q6, d7[0]
        vmla.f32 q1, q7, d7[1]
        vld1.32 {q3}, [r1], r7
        vmla.f32 q0, q4, d6[0]
        vld1.32 {q6, q7}, [r2]!
        vmla.f32 q1, q5, d6[1]
        subs r3, r3, #1
        bne L1LoopZ
    L1LoopZEnd:
    vmla.f32 q0, q6, d7[0]
    vmla.f32 q1, q7, d7[1]

    vadd.f32 q0, q0, q1
    vmov.i32 r1, d16[0]
    vmov.i32 r2, d16[1]
    vmov.i32 r3, d17[0]
    add r1, r1, #16
    vst1.32 {q0}, [r8]!
    subs r6, r6, #1
    bne L1Loop

End:

subs r5, r5, #1
add r0, r0, r4
mov r6, r9
mov r1, r10
add r2, r12, r11
bne LoopDz

vpop {q4-q7}
pop {r4-r11, pc}

#endif
#endif
