//
//  MNNLineDepthWiseInt8AddBiasScaleUnit.S
//  MNN
//
//  Created by MNN on 2019/06/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __arm__
#ifndef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNLineDepthWiseInt8AddBiasScaleUnit

// void MNNLineDepthWiseInt8AddBiasScaleUnit(int8_t* dst, const int8_t* src, const int8_t* weight, 
//      const int32_t* bias_z, size_t width, size_t src_w_step, size_t fw, size_t fh, 
//      size_t dilateX_step, size_t dilateY_step, const float* scale_z)

// Auto Load:
// r0: dst*, r1: src*, r2: weight*, r3: bias_z*
// Load from sp
// r4: width, r5: src_w_step, r6: fw, r7: fh, r8: dilateX_step, r9: dilateY_step, r10: scale_z
push {r4-r11, lr}

ldr r4, [sp, #36]
ldr r5, [sp, #40]
ldr r6, [sp, #44]
ldr r7, [sp, #48]
ldr r8, [sp, #52]
ldr r9, [sp, #56]
ldr r10, [sp, #60]

vpush {q4-q7}

vld1.32 {q6}, [r3]
vld1.32 {q4}, [r10]

mul r10, r6, r8
sub r9, r9, r10

L8:
cmp r4, #8
blt L4

mov r12, #8
mul r12, r5, r12

L8Loop:
    // load bias
    vmov.i32 q8, q6
    vmov.i32 q9, q6
    vmov.i32 q10, q6
    vmov.i32 q11, q6
    vmov.i32 q12, q6
    vmov.i32 q13, q6
    vmov.i32 q14, q6
    vmov.i32 q15, q6

    vmov.i32 d14[0], r1
    vmov.i32 d14[1], r2
    mov r10, r7
    L8LoopH:
        mov r11, r6
        L8LoopW:
            vld1.32 {d6[0]}, [r2]!
            vmovl.s8 q3, d6
            vld1.32 {d0[0]}, [r1], r5
            vld1.32 {d0[1]}, [r1], r5
            subs r11, r11, #1
            vmovl.s8 q1, d0
            vld1.32 {d1[0]}, [r1], r5
            vld1.32 {d1[1]}, [r1], r5
            vmovl.s8 q2, d1
            vmlal.s16 q8, d6, d2

            vld1.32 {d0[0]}, [r1], r5
            vmlal.s16 q9, d6, d3
            vld1.32 {d0[1]}, [r1], r5
            vld1.32 {d1[0]}, [r1], r5
            vld1.32 {d1[1]}, [r1], r5
            vmlal.s16 q10, d6, d4
            vmovl.s8 q1, d0
            vmlal.s16 q11, d6, d5
            vmovl.s8 q2, d1
            vmlal.s16 q12, d6, d2
            vmlal.s16 q13, d6, d3
            vmlal.s16 q14, d6, d4
            vmlal.s16 q15, d6, d5

            sub r1, r1, r12
            add r1, r1, r8
            bne L8LoopW
        L8LoopWEnd:
        subs r10, r10, #1
        add r1, r1, r9
        bne L8LoopH
    
    sub r4, r4, #8
    vcvt.f32.s32 q8, q8
    vcvt.f32.s32 q9, q9
    vmul.f32 q8, q8, q4
    vmul.f32 q9, q9, q4
    vcvt.s32.f32 q8, q8
    vcvt.s32.f32 q9, q9
    vqmovn.s32 d0, q8
    vqmovn.s32 d1, q9
    vqmovn.s16 d4, q0

    vcvt.f32.s32 q10, q10
    vcvt.f32.s32 q11, q11
    vmul.f32 q10, q10, q4
    vmul.f32 q11, q11, q4
    vcvt.s32.f32 q10, q10
    vcvt.s32.f32 q11, q11
    vqmovn.s32 d2, q10
    vqmovn.s32 d3, q11
    vqmovn.s16 d5, q1

    vst1.32 {q2}, [r0]!

    vcvt.f32.s32 q12, q12
    vcvt.f32.s32 q13, q13
    vmul.f32 q12, q12, q4
    vmul.f32 q13, q13, q4
    vcvt.s32.f32 q12, q12
    vcvt.s32.f32 q13, q13
    vqmovn.s32 d0, q12
    vqmovn.s32 d1, q13
    vqmovn.s16 d6, q0

    vcvt.f32.s32 q14, q14
    vcvt.f32.s32 q15, q15
    vmul.f32 q14, q14, q4
    vmul.f32 q15, q15, q4
    vcvt.s32.f32 q14, q14
    vcvt.s32.f32 q15, q15
    vqmovn.s32 d2, q14
    vqmovn.s32 d3, q15
    vqmovn.s16 d7, q1
    vst1.32 {q3}, [r0]!

    vmov.i32 r1, d14[0]
    vmov.i32 r2, d14[1]
    add r1, r1, r12
    cmp r4, #8
    bge L8Loop


L4:
cmp r4, #4
blt L1

mov r12, #4
mul r12, r5, r12

L4Loop:
    vmov.i32 q8, q6
    vmov.i32 q9, q6
    vmov.i32 q10, q6
    vmov.i32 q11, q6
    
    vmov.i32 d14[0], r1
    vmov.i32 d14[1], r2
    mov r10, r7
    L4LoopH:
        mov r11, r6
        L4LoopW:
            vld1.32 {d6[0]}, [r2]!
            vmovl.s8 q3, d6
            vld1.32 {d0[0]}, [r1], r5
            vld1.32 {d0[1]}, [r1], r5
            subs r11, r11, #1
            vmovl.s8 q1, d0
            vld1.32 {d1[0]}, [r1], r5
            vld1.32 {d1[1]}, [r1], r5
            vmovl.s8 q2, d1
            vmlal.s16 q8, d6, d2
            vmlal.s16 q9, d6, d3
            vmlal.s16 q10, d6, d4
            vmlal.s16 q11, d6, d5

            sub r1, r1, r12
            add r1, r1, r8
            bne L4LoopW
        L4LoopWEnd:
        subs r10, r10, #1
        add r1, r1, r9
        bne L4LoopH
    sub r4, r4, #4
    vcvt.f32.s32 q8, q8
    vcvt.f32.s32 q9, q9
    vmul.f32 q8, q8, q4
    vmul.f32 q9, q9, q4
    vcvt.s32.f32 q8, q8
    vcvt.s32.f32 q9, q9
    vqmovn.s32 d0, q8
    vqmovn.s32 d1, q9
    vqmovn.s16 d4, q0

    vcvt.f32.s32 q10, q10
    vcvt.f32.s32 q11, q11
    vmul.f32 q10, q10, q4
    vmul.f32 q11, q11, q4
    vcvt.s32.f32 q10, q10
    vcvt.s32.f32 q11, q11
    vqmovn.s32 d2, q10
    vqmovn.s32 d3, q11
    vqmovn.s16 d5, q1

    vst1.32 {q2}, [r0]!
    vmov.i32 r1, d14[0]
    vmov.i32 r2, d14[1]
    add r1, r1, r12
    cmp r4, #4
    bge L4Loop


L1:
cmp r4, #0
beq End

L1Loop:
    vmov.i32 q0, q6
    mov r10, r7
    vmov.i32 d14[0], r1
    vmov.i32 d14[1], r2
    L1LoopH:
        mov r11, r6
        L1LoopW:
            vld1.32 {d2[0]}, [r1], r8
            vld1.32 {d4[0]}, [r2]!
            vmovl.s8 q1, d2
            vmovl.s8 q2, d4
            vmlal.s16 q0, d2, d4
            subs r11, r11, #1
            bne L1LoopW
        subs r10, r10, #1
        add r1, r1, r9
        bne L1LoopH
    
    subs r4, r4, #1
    vcvt.f32.s32 q0, q0
    vmul.f32 q0, q0, q4
    vcvt.s32.f32 q1, q0
    vqmovn.s32 d0, q1
    vqmovn.s16 d2, q0

    vst1.s32 {d2[0]}, [r0]!
    vmov.i32 r1, d14[0]
    vmov.i32 r2, d14[1]
    add r1, r1, r5
    bne L1Loop

End:
vpop {q4-q7}
pop {r4-r11, pc}

#endif
#endif