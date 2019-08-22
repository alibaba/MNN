//
//  MNNGemmint8to32_8x4_Unit.S
//  MNN
//
//  Created by MNN on 2019/08/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __arm__
#ifndef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNGemmint8to32_8x4_Unit
//void MNNGemmint8to32_8x4_Unit(int32_t* dst, const int8_t* src, const int8_t* weight, const int32_t* inputSummer, size_t src_depth_quad,
//                                  size_t dst_step, size_t dst_depth_quad);


//Auto: r0: dst*, r1: src*, r2:weight*, r3: bias*
// Load from sp: r5: src_depth_quad, r6: dst_step, r7: dst_depth_quad

push {r4, r5, r6, r7, r8, lr}

ldr r5, [sp, #24]
ldr r6, [sp, #28]
ldr r7, [sp, #32]

vpush {q4-q7}

L2LoopDz:
    mov r8, r1

    subs r12, r5, #1
    // first four output
    vld1.8 {q2}, [r1]!
    vld1.8 {q4,q5}, [r2]!
    vmull.s8 q0, d4, d8
    vmull.s8 q1, d4, d10
    vmlal.s8 q0, d5, d9
    vmlal.s8 q1, d5, d11
    vpaddl.s16 q9, q1
    vld1.8 {q6,q7}, [r2]!
    vpaddl.s16 q8, q0

    vmull.s8 q0, d4, d12
    vmull.s8 q1, d4, d14
    vmlal.s8 q1, d5, d15
    vmlal.s8 q0, d5, d13
    vpaddl.s16 q10, q0
    vld1.8 {q3}, [r1]!
    vpaddl.s16 q11, q1
    // second four output
    vmull.s8 q0, d6, d8
    vmull.s8 q1, d6, d10
    vmlal.s8 q1, d7, d11
    vmlal.s8 q0, d7, d9
    vpaddl.s16 q12, q0
    vpaddl.s16 q13, q1
    
    vmull.s8 q0, d6, d12
    vmlal.s8 q0, d7, d13
    vmull.s8 q1, d6, d14
    vmlal.s8 q1, d7, d15
    vpaddl.s16 q14, q0
    vpaddl.s16 q15, q1

    beq L2LoopSzEnd
    
    L2LoopSz:
        // first four output
        vld1.8 {q2}, [r1]!
        vld1.8 {q4,q5}, [r2]!
        vmull.s8 q0, d4, d8
        vmlal.s8 q0, d5, d9
        vmull.s8 q1, d4, d10
        vmlal.s8 q1, d5, d11
        vld1.8 {q6,q7}, [r2]!
        vpadal.s16 q8, q0
        vpadal.s16 q9, q1

        vmull.s8 q0, d4, d12
        vmull.s8 q1, d4, d14
        vmlal.s8 q0, d5, d13
        vld1.8 {q3}, [r1]!
        vmlal.s8 q1, d5, d15
        vpadal.s16 q10, q0
        vpadal.s16 q11, q1
        // second four output
        vmull.s8 q0, d6, d8
        vmull.s8 q1, d6, d10
        vmlal.s8 q0, d7, d9
        vmlal.s8 q1, d7, d11
        vpadal.s16 q12, q0
        vpadal.s16 q13, q1
        
        vmull.s8 q0, d6, d12
        vmull.s8 q1, d6, d14
        vmlal.s8 q0, d7, d13
        vmlal.s8 q1, d7, d15
        vpadal.s16 q14, q0
        vpadal.s16 q15, q1

        subs r12, r12, #1
        bne L2LoopSz

    L2LoopSzEnd:

    vld1.s32 {d2}, [r3]
    vdup.s32 q0, d2[0]
    vdup.s32 q1, d2[1]

    vpadd.s32 d16, d16, d17
    vpadd.s32 d20, d20, d21
    vpadd.s32 d18, d18, d19
    vpadd.s32 d22, d22, d23

    vpadd.s32 d24, d24, d25
    vpadd.s32 d28, d28, d29
    vpadd.s32 d26, d26, d27
    vpadd.s32 d30, d30, d31

    // q8,q9
    vpadd.s32 d16, d16, d18
    vpadd.s32 d17, d20, d22
    vpadd.s32 d18, d24, d26
    vsub.s32 q8, q8, q0

    vpadd.s32 d19, d28, d30
    vsub.s32 q9, q9, q1
    
    vst1.32 {q8, q9}, [r0], r6

    subs r7, r7, #1
    mov r1, r8
    bne L2LoopDz


vpop {q4-q7}
pop {r4, r5, r6, r7, r8, pc}

#endif
#endif
