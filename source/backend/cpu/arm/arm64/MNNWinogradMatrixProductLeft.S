//
//  MNNWinogradMatrixProductLeft.S
//  MNN
//
//  Created by MNN on 2018/08/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNWinogradMatrixProductLeft
//void MNNWinogradMatrixProductLeft(const float* S, const float* B, float* M, size_t w, size_t h, size_t k, size_t length);

//Auto: x0: S, x1:B, x2: M, x3:w, x4:h, x5:k, x6:length

//unitStepInFloat
mov x8, #16 // 4*sizeof(float)
mul x8, x6, x8

//srcYUnitStep
mul x9, x3, x8
sub x9, x9, x8
add x7, x9, x8

//B's step
mov x10, #4
mul x10, x4, x10

LoopY:
    mov v4.d[0], x0
    mov v4.d[1], x3
    LoopX:
        mov v5.d[0], x0
        mov v5.d[1], x1
        movi v30.4s, #0
        mov x11, x6
        LoopUnitSetZero:
            st1 {v30.4s}, [x2], #16
            subs x11, x11, #1
            bne LoopUnitSetZero
        sub x2, x2, x8
        mov x12, x5

        LK4:
        cmp x12, #4
        blt LK3
        mov v6.d[0], x3
        mov v6.d[1], x4
        LoopK4:
            ld1 {v0.s}[0], [x1], x10
            ld1 {v0.s}[1], [x1], x10
            ld1 {v0.s}[2], [x1], x10
            ld1 {v0.s}[3], [x1], x10
            mov x11, x6
            mov v7.d[0], x1

            add x1, x0, x7
            add x3, x1, x7
            add x4, x3, x7

            LoopUnitK4:
                ld1 {v16.4s}, [x2]
                ld1 {v20.4s}, [x0], #16
                fmla v16.4s, v20.4s, v0.s[0]
                ld1 {v21.4s}, [x1], #16
                fmul v17.4s, v21.4s, v0.s[1]
                ld1 {v20.4s}, [x3], #16
                fmla v16.4s, v20.4s, v0.s[2]
                ld1 {v21.4s}, [x4], #16
                fmla v17.4s, v21.4s, v0.s[3]

                fadd v17.4s, v16.4s, v17.4s
                st1 {v17.4s}, [x2], #16
                subs x11, x11, #1
                bne LoopUnitK4
            sub x2, x2, x8

            sub x12, x12, #4

            add x0, x4, x9
            mov x1, v7.d[0]
            cmp x12, #4
            bge LoopK4
        mov x3, v6.d[0]
        mov x4, v6.d[1]

        LK3:
        cmp x12, #3
        blt LK1
        mov v6.d[0], x3
        LoopK3:
            ld1 {v0.s}[0], [x1], x10
            ld1 {v0.s}[1], [x1], x10
            ld1 {v0.s}[2], [x1], x10
            mov x11, x6
            mov v7.d[0], x1

            add x1, x0, x7
            add x3, x1, x7

            LoopUnitK3:
                ld1 {v16.4s}, [x2]
                ld1 {v20.4s}, [x0], #16
                fmla v16.4s, v20.4s, v0.s[0]
                ld1 {v21.4s}, [x1], #16
                fmul v17.4s, v21.4s, v0.s[1]
                ld1 {v20.4s}, [x3], #16
                fmla v16.4s, v20.4s, v0.s[2]

                fadd v17.4s, v16.4s, v17.4s
                st1 {v17.4s}, [x2], #16
                subs x11, x11, #1
                bne LoopUnitK3
            sub x2, x2, x8

            sub x12, x12, #3

            add x0, x3, x9
            mov x1, v7.d[0]
            cmp x12, #3
            bge LoopK3
        mov x3, v6.d[0]


        LK1:
        cmp x12, #0
        beq LKEnd

        LoopK:
            ld1 {v31.s}[0], [x1], x10

            dup v31.4s, v31.s[0]
            mov x11, x6
            LoopUnit:
                ld1 {v0.4s}, [x2]
                ld1 {v1.4s}, [x0], #16
                fmla v0.4s, v1.4s, v31.4s

                st1 {v0.4s}, [x2], #16
                subs x11, x11, #1
                bne LoopUnit
            subs x12, x12, #1

            sub x2, x2, x8
            add x0, x0, x9
            bne LoopK
        LKEnd:
        mov x0, v5.d[0]
        mov x1, v5.d[1]
        subs x3, x3, #1
        add x0, x0, x8
        add x2, x2, x8

        bne LoopX
    mov x0, v4.d[0]
    mov x3, v4.d[1]
    add x1, x1, #4 //sizeof(float)

    subs x4, x4, #1
    bne LoopY



    ret

#endif
