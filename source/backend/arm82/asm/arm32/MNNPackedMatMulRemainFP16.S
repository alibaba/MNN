//
//  MNNPackedMatMulRemainFP16.S
//  MNN
//
//  Created by MNN on 2020/06/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __arm__
#ifndef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5
// 12 * 8 MatMul
asm_function MNNPackedMatMulRemainFP16
//void MNNPackedMatMulRemainFP16(FLOAT16* C, const FLOAT16* A, const FLOAT16* B, size_t eSize, const size_t* parameter, const FLOAT16* postParameters, const FLOAT16* bias);
//Auto r0: C, r1:A, r2:B, r3:eSize, 
//r4:parameter, r5: cache no usage, r6:postParameters, r7:bias

// r4: h, r8: l, r9: tmp r0, r10: tmp r1, r11: tmp r2, r12: aStride

push {r4-r11, lr}
ldr r4, [sp, #36]
ldr r6, [sp, #40]
ldr r7, [sp, #44]
ldr r12, [r4, #0]
vpush {q4}
cmp r6, #0
beq E8
// q0-q2
vld1.32 {q0}, [r6]
vcvt.f16.f32 d0, q0
vdup.16 q1, d0[2] // min
vdup.16 q2, d0[3] // max

.macro COMPUTE op, s0, s1, d0, d1, d2, d3
    \op \d0, \s0, \s1[0]
    \op \d1, \s0, \s1[1]
    \op \d2, \s0, \s1[2]
    \op \d3, \s0, \s1[3]
.endm

.macro CLIP op, s0, d0, d1, d2, d3
    \op \d0, \d0, \s0
    \op \d1, \d1, \s0
    \op \d2, \d2, \s0
    \op \d3, \d3, \s0
.endm

.macro ADD_BIAS s0, d0, d1, d2, d3
    vmla.f16 \d0, \s0, d0[1]
    vmla.f16 \d1, \s0, d0[1]
    vmla.f16 \d2, \s0, d0[1]
    vmla.f16 \d3, \s0, d0[1]
.endm

E8:
cmp r3, #8
blt E4
LoopE8:
    ldr r5, [r4, #8] // h
    add r5, r5, #7
    lsr r5, r5, #3
    mov r9, r0
    mov r11, r2
    push {r7}
    LoopE8H:
        mov r10, r1
        ldr r8, [r4, #4] // l
        subs r8, r8, #1
        vld1.16 {q3}, [r10], r12
        vld1.16 {q4}, [r11]!
        COMPUTE vmul.f16, q4, d6, q8, q9, q10, q11
        COMPUTE vmul.f16, q4, d7, q12, q13, q14, q15
        beq LoopE8LEnd
        LoopE8L:
            vld1.16 {q3}, [r10], r12
            vld1.16 {q4}, [r11]!
            COMPUTE vmla.f16, q4, d6, q8, q9, q10, q11
            COMPUTE vmla.f16, q4, d7, q12, q13, q14, q15
            subs r8, r8, #1
            bne LoopE8L

        LoopE8LEnd:
        cmp r6, #0
        beq StoreE8
        vld1.16 {q3}, [r7]!
        ADD_BIAS q3, q8, q9, q10, q11
        ADD_BIAS q3, q12, q13, q14, q15
        CLIP vmax.f16, q1, q8, q9, q10, q11
        CLIP vmax.f16, q1, q12, q13, q14, q15
        CLIP vmin.f16, q2, q8, q9, q10, q11
        CLIP vmin.f16, q2, q12, q13, q14, q15

        StoreE8:
        ldr r8, [r4, #20]
        add r11, r11, r8
        ldr r8, [r4, #12]
        vst1.16 {q8, q9}, [r9]!
        vst1.16 {q10, q11}, [r9]!
        vst1.16 {q12, q13}, [r9]!
        vst1.16 {q14, q15}, [r9], r8
        sub r9, r9, #96
        subs r5, r5, #1
        bne LoopE8H
        sub r3, r3, #8
        add r0, r0, #128
        add r1, r1, #16
        cmp r3, #8
        pop {r7}
        bge LoopE8
    

E4:
cmp r3, #4
blt E1
LoopE4:
    ldr r5, [r4, #8] // h
    add r5, r5, #7
    lsr r5, r5, #3
    mov r9, r0
    mov r11, r2
    push {r7}
    LoopE4H:
        mov r10, r1
        ldr r8, [r4, #4] // l
        subs r8, r8, #1
        vld1.16 {d6}, [r10], r12
        vld1.16 {q4}, [r11]!
        COMPUTE vmul.f16, q4, d6, q8, q9, q10, q11
        beq LoopE4LEnd
        LoopE4L:
            vld1.16 {d6}, [r10], r12
            vld1.16 {q4}, [r11]!
            COMPUTE vmla.f16, q4, d6, q8, q9, q10, q11
            subs r8, r8, #1
            bne LoopE4L

        LoopE4LEnd:
        cmp r6, #0
        beq StoreE4
        vld1.16 {q3}, [r7]!
        ADD_BIAS q3, q8, q9, q10, q11
        CLIP vmax.f16, q1, q8, q9, q10, q11
        CLIP vmin.f16, q2, q8, q9, q10, q11

        StoreE4:
        ldr r8, [r4, #20] // bExtraStride
        add r11, r11, r8
        ldr r8, [r4, #12] // cStride
        vst1.16 {q8, q9}, [r9]!
        vst1.16 {q10, q11}, [r9], r8
        sub r9, r9, #32
        subs r5, r5, #1
        bne LoopE4H
    sub r3, r3, #4
    add r0, r0, #64
    add r1, r1, #8
    cmp r3, #4
    pop {r7}
    bge LoopE4

E1:
cmp r3, #0
beq End
LoopE1:
    ldr r5, [r4, #8] // h
    add r5, r5, #7
    lsr r5, r5, #3
    mov r9, r0
    mov r11, r2
    push {r7}
    LoopE1H:
        mov r10, r1
        ldr r8, [r4, #4] // l
        vmov.i32 q15, #0
        LoopE1L:
            vld1.16 {d6[0]}, [r10], r12
            vld1.16 {q4}, [r11]!
            vmla.f16 q15, q4, d6[0]
            subs r8, r8, #1
            bne LoopE1L
        cmp r6, #0
        beq StoreE1
        vld1.16 {q14}, [r7]!
        vmla.f16 q15, q14, d0[1]

        PostTreatE1:
        vmax.f16 q15, q15, q1
        vmin.f16 q15, q15, q2

        StoreE1:
        ldr r8, [r4, #20]
        add r11, r11, r8
        ldr r8, [r4, #12]
        vst1.16 {q15}, [r9], r8
        subs r5, r5, #1
        bne LoopE1H
    subs r3, r3, #1
    add r0, r0, #16
    add r1, r1, #2
    pop {r7}
    bne LoopE1
End:
vpop {q4}
pop {r4-r11, pc}

#endif
#endif
