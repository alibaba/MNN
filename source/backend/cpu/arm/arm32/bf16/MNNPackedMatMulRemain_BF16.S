//
//  NEON_MNNPackedMatMulRemain_BF16.S
//  MNN
//
//  Created by MNN on 2021/02/24.
//  Copyright © 2018-2021 Alibaba Group Holding Limited.
//

#ifdef __arm__
#ifndef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5
// 12 * 8 MatMul
asm_function NEON_MNNPackedMatMulRemain_BF16
// treate float pointer as int16_t*
//void NEON_MNNPackedMatMulRemain_BF16(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter, const float* postParameters, const float* bias);
//Auto r0: C, r1:A, r2:B, r3:eSize,
//r4:parameter, r5: cache no usage, r6:postParameters, r7:bias

push {r4-r8, r10, r11, lr} // avoid to touch platform-register r-9
ldr r4, [sp, #32]
ldr r6, [sp, #36]
ldr r7, [sp, #40]
ldr r12, [r4, #0]
cmp r6, #0
beq Start
vld1.32 {q3}, [r6]
vdup.f32 q12, d7[0] // min
vdup.f32 q13, d7[1] // max
Start:
cmp r3, #4
blt L1

LoopE4:
    ldr r5, [r4, #8] // h
    add r5, r5, #3
    lsr r5, r5, #2 // r5 = UP_DIV(r5, 4)
    mov lr, r0
    mov r11, r2
    push {r7}
    LoopE4H:
        mov r10, r1
        ldr r8, [r4, #4] // l
        vmov.i32 q8, #0
        vmov.i32 q9, #0
        vmov.i32 q10, #0
        vmov.i32 q11, #0
        LoopE4L:
            vld1.16 {d0}, [r10], r12
            vld1.16 {d2}, [r11]! // load 4 * sizeof(int16_t)
            vshll.s16 q0, d0, #16 // shift left long of each int16_t as float32
            vshll.s16 q1, d2, #16
            vmla.f32 q8, q1, d0[0]
            vmla.f32 q9, q1, d0[1]
            vmla.f32 q10, q1, d1[0]
            vmla.f32 q11, q1, d1[1]
            subs r8, r8, #1
            bne LoopE4L
        cmp r6, #0
        beq StoreE4
        vld1.16 {d28}, [r7]! // load 4 * sizeof(int16_t)
        vshll.s16 q14, d28, #16 // shift left long of each int16_t as float32
        vmla.f32 q8, q14, d6[1]
        vmla.f32 q9, q14, d6[1]
        vmla.f32 q10, q14, d6[1]
        vmla.f32 q11, q14, d6[1]

        PostTreatE4:
        vmax.f32 q8, q8, q12
        vmax.f32 q9, q9, q12
        vmax.f32 q10, q10, q12
        vmax.f32 q11, q11, q12

        vmin.f32 q8, q8, q13
        vmin.f32 q9, q9, q13
        vmin.f32 q10, q10, q13
        vmin.f32 q11, q11, q13

        StoreE4:
        ldr r8, [r4, #20]
        add r11, r11, r8
        ldr r8, [r4, #12]

        vshrn.i32 d16, q8, #16 // shift right 16bit of each float32 as int16_t
        vshrn.i32 d17, q9, #16
        vshrn.i32 d18, q10, #16
        vshrn.i32 d19, q11, #16
        vst1.16 {d16, d17}, [lr]!
        vst1.16 {d18, d19}, [lr], r8
        sub lr, lr, #16
        subs r5, r5, #1 // move 4 colum along lP dim. lP = l / 4
        bne LoopE4H
    sub r3, r3, #4 // move 4 colum along e dim.
    add r0, r0, #32 // move address of 4 * 4 * sizeof(int16_t)
    add r1, r1, #8 // move address of 4 * sizeof(int16_t) in src tile block
    cmp r3, #4
    pop {r7}
    bge LoopE4

L1:
cmp r3, #0
beq End
LoopE1:
    ldr r5, [r4, #8] // h
    add r5, r5, #3
    lsr r5, r5, #2
    mov lr, r0
    mov r11, r2
    push {r7}
    LoopE1H:
        mov r10, r1
        ldr r8, [r4, #4] // l
        vmov.i32 q15, #0
        LoopE1L:
            vld1.16 {d0[0]}, [r10], r12
            vld1.16 {d2}, [r11]! // load 4 * sizeof(int16_t)
            vshll.s16 q0, d0, #16 // shift left long of each int16_t as float32
            vshll.s16 q1, d2, #16

            vmla.f32 q15, q1, d0[0]
            subs r8, r8, #1
            bne LoopE1L
        cmp r6, #0
        beq StoreE1
        vld1.16 {d28}, [r7]! // load 4 * sizeof(int16_t)
        vshll.s16 q14, d28, #16 // shift left long of each int16_t as float32
        vmla.f32 q15, q14, d6[1]

        PostTreatE1:
        vmax.f32 q15, q15, q12
        vmin.f32 q15, q15, q13

        StoreE1:
        ldr r8, [r4, #20]
        add r11, r11, r8
        ldr r8, [r4, #12]

        vshrn.i32 d30, q15, #16 // shift right 16bit of each float32 as int16_t
        vst1.16 {d30}, [lr], r8
        subs r5, r5, #1
        bne LoopE1H
    subs r3, r3, #1
    add r0, r0, #8 // move address of 4 * sizeof(int16_t)
    add r1, r1, #2 // move address of 1 * sizeof(int16_t)
    pop {r7}
    bne LoopE1
End:
pop {r4-r8, r10, r11, pc}

#endif
#endif
