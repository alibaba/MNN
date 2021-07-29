//
//  MNNWinogradMatrixProductRight.S
//  MNN
//
//  Created by MNN on 2018/08/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __arm__
#ifndef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNWinogradMatrixProductRight
//void MNNWinogradMatrixProductRight(const float* S, const float* B, float* M, size_t w, size_t h, size_t k, size_t length);

//Auto: r0: S, r1:B, r2: M, r3:w
//Load From sp: r4:h, r5:k, r6:length

push {r4-r8, r10, r11, lr} // avoid to touch platform-register r-9
ldr r4, [sp, #32]
ldr r5, [sp, #36]
ldr r6, [sp, #40]

//unitStepInFloat
mov r8, #16 // 4*sizeof(float)
mul r8, r6, r8

//srcYUnitStep
mul lr, r5, r8

//B's step
mov r10, #4
mul r10, r4, r10

LoopY:
    push {r1, r3}
    LoopX:
        push {r0, r1}
        vmov.i32 q14, #0
        mov r11, r6
        LoopUnitSetZero:
            vst1.32 {q14}, [r2]!
            subs r11, r11, #1
            bne LoopUnitSetZero
        sub r2, r2, r8
        mov r12, r5

        LK7:
        cmp r12, #7
        blt LK4
        push {r3-r7}
        LoopK7:
            vld1.32 {d0[0]}, [r1], r10
            vld1.32 {d0[1]}, [r1], r10
            vld1.32 {d1[0]}, [r1], r10
            vld1.32 {d1[1]}, [r1], r10
            vld1.32 {d2[0]}, [r1], r10
            vld1.32 {d2[1]}, [r1], r10
            vld1.32 {d3[0]}, [r1], r10
            mov r11, r6
            vmov.32 d30[0], r1

            add r1, r0, r8
            add r3, r1, r8
            add r4, r3, r8
            add r5, r4, r8
            add r6, r5, r8
            add r7, r6, r8

            LoopUnitK7:
                vld1.32 {q8}, [r2]
                vld1.32 {q12}, [r0]!
                vmla.f32 q8, q12, d0[0]
                vld1.32 {q13}, [r1]!
                vmul.f32 q9, q13, d0[1]
                vld1.32 {q12}, [r3]!
                vmla.f32 q8, q12, d1[0]
                vld1.32 {q13}, [r4]!
                vmla.f32 q9, q13, d1[1]
                vld1.32 {q12}, [r5]!
                vmla.f32 q8, q12, d2[0]
                vld1.32 {q13}, [r6]!
                vmla.f32 q9, q13, d2[1]
                vld1.32 {q12}, [r7]!
                vmla.f32 q8, q12, d3[0]

                vadd.f32 q9, q8, q9
                vst1.32 {q9}, [r2]!
                subs r11, r11, #1
                bne LoopUnitK7
            sub r2, r2, r8
            sub r12, r12, #7
            mov r0, r7
            vmov.32 r1, d30[0]
            cmp r12, #7
            bge LoopK7
        pop {r3-r7}

        LK4:
        cmp r12, #4
        blt LK3
        vmov.32 d30[1], r3
        vmov.32 d31[0], r4
        LoopK4:
            vld1.32 {d0[0]}, [r1], r10
            vld1.32 {d0[1]}, [r1], r10
            vld1.32 {d1[0]}, [r1], r10
            vld1.32 {d1[1]}, [r1], r10
            mov r11, r6
            vmov.32 d30[0], r1

            add r1, r0, r8
            add r3, r1, r8
            add r4, r3, r8

            LoopUnitK4:
                vld1.32 {q8}, [r2]
                vld1.32 {q12}, [r0]!
                vmla.f32 q8, q12, d0[0]
                vld1.32 {q13}, [r1]!
                vmul.f32 q9, q13, d0[1]
                vld1.32 {q12}, [r3]!
                vmla.f32 q8, q12, d1[0]
                vld1.32 {q13}, [r4]!
                vmla.f32 q9, q13, d1[1]

                vadd.f32 q9, q8, q9
                vst1.32 {q9}, [r2]!
                subs r11, r11, #1
                bne LoopUnitK4
            sub r2, r2, r8

            sub r12, r12, #4

            mov r0, r4
            vmov.32 r1, d30[0]
            cmp r12, #4
            bge LoopK4
        vmov.32 r3, d30[1]
        vmov.32 r4, d31[0]

        LK3:
        cmp r12, #3
        blt LK1
        vmov.32 d30[1], r3
        LoopK3:
            vld1.32 {d0[0]}, [r1], r10
            vld1.32 {d0[1]}, [r1], r10
            vld1.32 {d1[0]}, [r1], r10
            mov r11, r6
            vmov.32 d30[0], r1

            add r1, r0, r8
            add r3, r1, r8

            LoopUnitK3:
                vld1.32 {q8}, [r2]
                vld1.32 {q12}, [r0]!
                vmla.f32 q8, q12, d0[0]
                vld1.32 {q13}, [r1]!
                vmul.f32 q9, q13, d0[1]
                vld1.32 {q12}, [r3]!
                vmla.f32 q8, q12, d1[0]

                vadd.f32 q9, q8, q9
                vst1.32 {q9}, [r2]!
                subs r11, r11, #1
                bne LoopUnitK3
            sub r2, r2, r8

            sub r12, r12, #3

            mov r0, r3
            vmov.32 r1, d30[0]
            cmp r12, #3
            bge LoopK3
        vmov.32 r3, d30[1]


        LK1:
        cmp r12, #0
        beq LKEnd

        LoopK:
            vld1.32 {d30[0]}, [r1], r10

            vdup.32 q15, d30[0]
            mov r11, r6
            LoopUnit:
                vld1.32 {q0}, [r2]
                vld1.32 {q1}, [r0]!
                vmla.f32 q0, q1, q15

                vst1.32 {q0}, [r2]!
                subs r11, r11, #1
                bne LoopUnit
            subs r12, r12, #1

            sub r2, r2, r8
            bne LoopK
        LKEnd:
        pop {r0, r1}
        subs r3, r3, #1
        add r2, r2, r8
        add r1, r1, #4 //sizeof(float)

        bne LoopX
    pop {r1, r3}
    add r0, r0, lr

    subs r4, r4, #1
    bne LoopY



pop {r4-r8, r10, r11, pc}

#endif
#endif
