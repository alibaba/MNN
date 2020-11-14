//
//  MNNGemmInt8toFloat32_8x4_Common.S
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

SET_ZERO:
vmov.i32 q12, #0
vmov.i32 q13, #0
vmov.i32 q14, #0
vmov.i32 q15, #0
b LoopZ1End

asm_function MNNGemmInt8toFloat32_8x4_Common
//void MNNGemmInt8toFloat32_8x4_Common(float* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad,
//                                     size_t width, size_t dst_step, size_t dst_depth_quad);
push {r4-r8, r10, r11, lr}

//Auto: r0: dst, r1: src, r2:weight, r3: src_depth_quad
//Load from sp: r4: width, r5: dst_step, r6: dst_depth_quad

ldr r4, [sp, #32] // 32 byte = 8 register (in push instruction's list) * 4 byte (aarch32, 32bit)
ldr r5, [sp, #36]
ldr r6, [sp, #40]
lsl r5, r5, #2 // sizeof(float)
lsl r7, r4, #3 // 8 * sizeof(int8)

vpush {q4-q7}

.macro MERGE z0, z1, z2, z3, z4
\z0 q12, \z1
\z0 q13, \z2
\z0 q14, \z3
\z0 q15, \z4
.endm

L1:
mov r8, r0
mov r11, r2

// r14 (lr) have been push in stack, and pop into pc at the end. So we can use it.
mov r14, r6

LoopDz:
    mov r10, r1
    cmp r3, #1
    beq SET_ZERO
    mov r12, r3
    vld1.8 {d0}, [r10], r7
    vld1.8 {d1}, [r10], r7
    vld1.8 {q1, q2}, [r11]!
    vld1.8 {q3, q4}, [r11]!

    vmull.s8 q5, d0, d2
    vmull.s8 q6, d0, d3
    vmull.s8 q7, d0, d4
    vmull.s8 q8, d0, d5
    vmull.s8 q9, d1, d6
    vmull.s8 q10, d1, d7
    vmull.s8 q11, d1, d8
    MERGE vpaddl.s16, q5, q6, q7, q8
    vmull.s8 q5, d1, d9
    MERGE vpadal.s16, q9, q10, q11, q5

    L2LoopZ:
        subs r12, r12, #2
        beq LoopZEnd
        cmp r12, #2
        blt LoopZ1End
        vld1.8 {d0}, [r10], r7
        vld1.8 {d1}, [r10], r7
        vld1.8 {q1, q2}, [r11]!
        vld1.8 {q3, q4}, [r11]!

        vmull.s8 q5, d0, d2
        vmull.s8 q6, d0, d3
        vmull.s8 q7, d0, d4
        vmull.s8 q8, d0, d5
        vmull.s8 q9, d1, d6
        vmull.s8 q10, d1, d7
        vmull.s8 q11, d1, d8
        MERGE vpadal.s16, q5, q6, q7, q8
        vmull.s8 q5, d1, d9
        MERGE vpadal.s16, q9, q10, q11, q5

        b L2LoopZ

LoopZ1End:
    vld1.8 {d0}, [r10], r7
    vld1.8 {q1, q2}, [r11]!
    vmull.s8 q5, d0, d2
    vmull.s8 q6, d0, d3
    vmull.s8 q7, d0, d4
    vmull.s8 q8, d0, d5
    MERGE vpadal.s16, q5, q6, q7, q8

LoopZEnd:
    vpadd.i32 d24, d24, d25
    vpadd.i32 d26, d26, d27
    vpadd.i32 d28, d28, d29
    vpadd.i32 d30, d30, d31
    vpadd.i32 d24, d24, d26
    vpadd.i32 d25, d28, d30
    vcvt.f32.s32 q12, q12

    vst1.32 {q12}, [r8], r5

    subs r14, r14, #1
    bne LoopDz

L1End:
add r0, r0, #16 // 1 * 4 * sizeof(float)
add r1, r1, #8  // 1 * 8 * sizeof(int8)
subs r4, r4, #1
bne L1

vpop {q4-q7}
pop {r4-r8, r10, r11, pc}


#endif
#endif
