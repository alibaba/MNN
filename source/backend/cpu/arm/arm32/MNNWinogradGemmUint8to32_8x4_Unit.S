//
//  MNNWinogradGemmUint8to32_8x4_Unit.S
//  MNN
//
//  Created by MNN on 2018/11/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __arm__
#ifndef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNWinogradGemmUint8to32_8x4_Unit
//void MNNWinogradGemmUint8to32_8x4_Unit(int32_t* dst, const uint8_t* src, const uint8_t* weight, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad, const int32_t* inputOffset);
push {r4, r5, r6, lr}

//Auto: r0: dst, r1: src, r2:weight, r3: src_depth_quad
//Load from sp: r4: dst_step, r5: dst_depth_quad

ldr r4, [sp, #16]
ldr r5, [sp, #20]
ldr r6, [sp, #24]

vld1.32 {d0}, [r6]

vpush {q4-q7}
vdup.32 q2, d0[0]
vdup.32 q3, d0[1]

L2LoopDz:
    mov r6, r1
    mov r12, r3

    vmov.i32 q8, #0
    vmov.i32 q9, #0
    vmov.i32 q10, #0
    vmov.i32 q11, #0

    vmov.i32 q12, #0
    vmov.i32 q13, #0
    vmov.i32 q14, #0
    vmov.i32 q15, #0

    L2LoopSz:
        vld1.8 {q4}, [r1]!
        vld1.8 {q5, q6}, [r2]!

        vaddl.u8 q0, d8, d10
        vaddl.u8 q1, d8, d11
        vmlal.u16 q8, d0, d1
        vmlal.u16 q9, d2, d3
        vaddl.u8 q0, d8, d12
        vaddl.u8 q1, d8, d13
        vmlal.u16 q10, d0, d1
        vmlal.u16 q11, d2, d3

        vaddl.u8 q0, d9, d10
        vaddl.u8 q1, d9, d11
        vmlal.u16 q12, d0, d1
        vmlal.u16 q13, d2, d3
        vaddl.u8 q0, d9, d12
        vaddl.u8 q1, d9, d13
        vmlal.u16 q14, d0, d1
        vmlal.u16 q15, d2, d3

        subs r12, r12, #1
        bne L2LoopSz
    L2LoopSzEnd:

    vpadd.u32 d16, d16, d17
    vpadd.u32 d20, d20, d21
    vpadd.u32 d18, d18, d19
    vpadd.u32 d22, d22, d23

    vpadd.u32 d24, d24, d25
    vpadd.u32 d28, d28, d29
    vpadd.u32 d26, d26, d27
    vpadd.u32 d30, d30, d31

    vpadd.u32 d16, d16, d18
    vpadd.u32 d17, d20, d22
    vpadd.u32 d18, d24, d26
    vpadd.u32 d19, d28, d30

    vsub.s32 q8, q8, q2
    vsub.s32 q9, q9, q3

    vst1.32 {q8, q9}, [r0]!

    subs r5, r5, #1
    mov r1, r6
    bne L2LoopDz

vpop {q4-q7}
pop {r4, r5, r6, pc}


#endif
#endif
