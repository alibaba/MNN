//
//  MNNPackedMatMulRemain_BF16.S
//  MNN
//
//  Created by MNN on 2021/02/21.
//  Copyright © 2018-2021 Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5
// 12 * 8 MatMul
asm_function NEON_MNNPackedMatMulRemain_BF16
//void NEON_MNNPackedMatMulRemain_BF16(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter, const float* postParameters, const float* bias);
//Auto x0: C, x1:A, x2:B, x3:eSize, x4:parameter, x5:postParameters, x6:bias
sub sp, sp, #32
str x19, [sp, #0]
str x20, [sp, #8]
str x21, [sp, #16]
ldr x11, [x4, #0] // aStride
ldr x9, [x4, #8] // l
ldr x10, [x4, #16] // h

ldr x7, [x4, #24] // cStride
ldr x19, [x4, #40] // bExtraStride

add x10, x10, #3
lsr x10, x10, #2

cbz x5, Start
ld1 {v5.4s}, [x5]
dup v6.4s, v5.s[2] // Min Value
dup v7.4s, v5.s[3] // Max Value

Start:

E8:
cmp x3, #8
blt E4

LoopE8: // e, TILE_BLOCK size is 8
    mov x20, x6 // bias
    mov x8, x10 // updiv(h, 4)
    mov x21, x0 // dest, C
    mov x13, x2 // weight, B

    LH8:
    cmp x8, #2 // h/4 > 2
    blt LH4
    // sub x14, x7, #32 // in "StoreLH8", total 2 lines stride is x14, first line is  4 * 4 * size_t(int16_t) = 32byte
    LoopH8x8:
        mov x15, x1 // src, A
        subs x12, x9, #1 // l
        ld1 {v3.4h, v4.4h}, [x13], #16 // 2 * 4 * sizeof(int16_t)
        ld1 {v0.4h, v1.4h}, [x15], x11
        shll v3.4s, v3.4h, #16
        shll v4.4s, v4.4h, #16
        shll v0.4s, v0.4h, #16
        shll v1.4s, v1.4h, #16

        fmul v16.4s, v3.4s, v0.s[0]
        fmul v17.4s, v3.4s, v0.s[1]
        fmul v18.4s, v3.4s, v0.s[2]
        fmul v19.4s, v3.4s, v0.s[3]

        fmul v20.4s, v4.4s, v0.s[0]
        fmul v21.4s, v4.4s, v0.s[1]
        fmul v22.4s, v4.4s, v0.s[2]
        fmul v23.4s, v4.4s, v0.s[3]

        fmul v24.4s, v3.4s, v1.s[0]
        fmul v25.4s, v3.4s, v1.s[1]
        fmul v26.4s, v3.4s, v1.s[2]
        fmul v27.4s, v3.4s, v1.s[3]

        fmul v28.4s, v4.4s, v1.s[0]
        fmul v29.4s, v4.4s, v1.s[1]
        fmul v30.4s, v4.4s, v1.s[2]
        fmul v31.4s, v4.4s, v1.s[3]
        beq LoopLEnd

        LoopL:
            ld1 {v3.4h, v4.4h}, [x13], #16 // 2 * 4 * sizeof(int16_t)
            ld1 {v0.4h, v1.4h}, [x15], x11
            shll v3.4s, v3.4h, #16
            shll v4.4s, v4.4h, #16
            shll v0.4s, v0.4h, #16
            shll v1.4s, v1.4h, #16

            fmla v16.4s, v3.4s, v0.s[0]
            fmla v17.4s, v3.4s, v0.s[1]
            fmla v18.4s, v3.4s, v0.s[2]
            fmla v19.4s, v3.4s, v0.s[3]

            fmla v20.4s, v4.4s, v0.s[0]
            fmla v21.4s, v4.4s, v0.s[1]
            fmla v22.4s, v4.4s, v0.s[2]
            fmla v23.4s, v4.4s, v0.s[3]

            fmla v24.4s, v3.4s, v1.s[0]
            fmla v25.4s, v3.4s, v1.s[1]
            fmla v26.4s, v3.4s, v1.s[2]
            fmla v27.4s, v3.4s, v1.s[3]

            fmla v28.4s, v4.4s, v1.s[0]
            fmla v29.4s, v4.4s, v1.s[1]
            fmla v30.4s, v4.4s, v1.s[2]
            fmla v31.4s, v4.4s, v1.s[3]

            subs x12, x12, #1
            bne LoopL

        LoopLEnd:

        add x13, x13, x19
        sub x8, x8, #2
        cmp x8, #2

        cbz x5, StoreLH8
        AddBiasLH8:
        ld1 {v0.4h, v1.4h}, [x20], #16
        shll v0.4s, v0.4h, #16
        shll v1.4s, v1.4h, #16

        fmla v16.4s, v0.4s, v5.s[1]
        fmla v17.4s, v0.4s, v5.s[1]
        fmla v18.4s, v0.4s, v5.s[1]
        fmla v19.4s, v0.4s, v5.s[1]

        fmla v20.4s, v1.4s, v5.s[1]
        fmla v21.4s, v1.4s, v5.s[1]
        fmla v22.4s, v1.4s, v5.s[1]
        fmla v23.4s, v1.4s, v5.s[1]

        fmla v24.4s, v0.4s, v5.s[1]
        fmla v25.4s, v0.4s, v5.s[1]
        fmla v26.4s, v0.4s, v5.s[1]
        fmla v27.4s, v0.4s, v5.s[1]

        fmla v28.4s, v1.4s, v5.s[1]
        fmla v29.4s, v1.4s, v5.s[1]
        fmla v30.4s, v1.4s, v5.s[1]
        fmla v31.4s, v1.4s, v5.s[1]

        PostTreatLH8:
        fmax v16.4s, v16.4s, v6.4s
        fmax v17.4s, v17.4s, v6.4s
        fmax v18.4s, v18.4s, v6.4s
        fmax v19.4s, v19.4s, v6.4s
        fmax v20.4s, v20.4s, v6.4s
        fmax v21.4s, v21.4s, v6.4s
        fmax v22.4s, v22.4s, v6.4s
        fmax v23.4s, v23.4s, v6.4s
        fmax v24.4s, v24.4s, v6.4s
        fmax v25.4s, v25.4s, v6.4s
        fmax v26.4s, v26.4s, v6.4s
        fmax v27.4s, v27.4s, v6.4s
        fmax v28.4s, v28.4s, v6.4s
        fmax v29.4s, v29.4s, v6.4s
        fmax v30.4s, v30.4s, v6.4s
        fmax v31.4s, v31.4s, v6.4s

        fmin v16.4s, v16.4s, v7.4s
        fmin v17.4s, v17.4s, v7.4s
        fmin v18.4s, v18.4s, v7.4s
        fmin v19.4s, v19.4s, v7.4s
        fmin v20.4s, v20.4s, v7.4s
        fmin v21.4s, v21.4s, v7.4s
        fmin v22.4s, v22.4s, v7.4s
        fmin v23.4s, v23.4s, v7.4s
        fmin v24.4s, v24.4s, v7.4s
        fmin v25.4s, v25.4s, v7.4s
        fmin v26.4s, v26.4s, v7.4s
        fmin v27.4s, v27.4s, v7.4s
        fmin v28.4s, v28.4s, v7.4s
        fmin v29.4s, v29.4s, v7.4s
        fmin v30.4s, v30.4s, v7.4s
        fmin v31.4s, v31.4s, v7.4s

        StoreLH8:
        shrn v16.4h, v16.4s, #16
        shrn v17.4h, v17.4s, #16
        shrn v18.4h, v18.4s, #16
        shrn v19.4h, v19.4s, #16
        shrn v20.4h, v20.4s, #16
        shrn v21.4h, v21.4s, #16
        shrn v22.4h, v22.4s, #16
        shrn v23.4h, v23.4s, #16
        shrn v24.4h, v24.4s, #16
        shrn v25.4h, v25.4s, #16
        shrn v26.4h, v26.4s, #16
        shrn v27.4h, v27.4s, #16
        shrn v28.4h, v28.4s, #16
        shrn v29.4h, v29.4s, #16
        shrn v30.4h, v30.4s, #16
        shrn v31.4h, v31.4s, #16

        stp d16, d17, [x0]
        stp d18, d19, [x0, #(16 * 1)]
        stp d24, d25, [x0, #(16 * 2)]
        stp d26, d27, [x0, #(16 * 3)]
        add x0, x0, x7 // stp donot support post-index offset in register

        stp d20, d21, [x0]
        stp d22, d23, [x0, #(16 * 1)]
        stp d28, d29, [x0, #(16 * 2)]
        stp d30, d31, [x0, #(16 * 3)]
        add x0, x0, x7 // stp donot support post-index offset in register

        // st1 {v16.4h, v17.4h, v18.4h, v19.4h}, [x0], #32 // 4 * 4 * sizeof(int16_t)
        // st1 {v24.4h, v25.4h, v26.4h, v27.4h}, [x0], x14
        // st1 {v20.4h, v21.4h, v22.4h, v23.4h}, [x0], #32
        // st1 {v28.4h, v29.4h, v30.4h, v31.4h}, [x0], x14

        bge LoopH8x8

    LH4:
    cbz x8, E8End
    LoopHRemain:
        mov x15, x1
        subs x12, x9, #1
        ld1 {v3.4h}, [x13]
        ld1 {v0.4h}, [x15], #8
        shll v3.4s, v3.4h, #16
        shll v0.4s, v0.4h, #16

        fmul v16.4s, v3.4s, v0.s[0]
        fmul v17.4s, v3.4s, v0.s[1]
        add x13, x13, #16 // weight
        ld1 {v1.4h}, [x15]
        shll v1.4s, v1.4h, #16

        fmul v18.4s, v3.4s, v0.s[2]
        sub x15, x15, #8
        fmul v19.4s, v3.4s, v0.s[3]
        add x15, x15, x11
        fmul v20.4s, v3.4s, v1.s[0]
        fmul v21.4s, v3.4s, v1.s[1]
        fmul v22.4s, v3.4s, v1.s[2]
        fmul v23.4s, v3.4s, v1.s[3]
        beq LoopLREnd

        LoopLR:
            ld1 {v3.4h}, [x13]
            ld1 {v0.4h}, [x15], #8
            shll v3.4s, v3.4h, #16
            shll v0.4s, v0.4h, #16

            fmla v16.4s, v3.4s, v0.s[0]
            fmla v17.4s, v3.4s, v0.s[1]
            add x13, x13, #16 // weight
            ld1 {v1.4h}, [x15]
            shll v1.4s, v1.4h, #16

            fmla v18.4s, v3.4s, v0.s[2]
            sub x15, x15, #8
            fmla v19.4s, v3.4s, v0.s[3]
            add x15, x15, x11

            fmla v20.4s, v3.4s, v1.s[0]
            fmla v21.4s, v3.4s, v1.s[1]
            fmla v22.4s, v3.4s, v1.s[2]
            fmla v23.4s, v3.4s, v1.s[3]

            subs x12, x12, #1
            bne LoopLR
        LoopLREnd:

        cbz x5, StoreLH8x4
        AddBiasLH8x4:
        ld1 {v0.4h}, [x20]
        shll v0.4s, v0.4h, #16

        fmla v16.4s, v0.4s, v5.s[1]
        fmla v17.4s, v0.4s, v5.s[1]
        fmla v18.4s, v0.4s, v5.s[1]
        fmla v19.4s, v0.4s, v5.s[1]

        fmla v20.4s, v0.4s, v5.s[1]
        fmla v21.4s, v0.4s, v5.s[1]
        fmla v22.4s, v0.4s, v5.s[1]
        fmla v23.4s, v0.4s, v5.s[1]

        PostTreatLH8x4:
        fmax v16.4s, v16.4s, v6.4s
        fmax v17.4s, v17.4s, v6.4s
        fmax v18.4s, v18.4s, v6.4s
        fmax v19.4s, v19.4s, v6.4s
        fmax v20.4s, v20.4s, v6.4s
        fmax v21.4s, v21.4s, v6.4s
        fmax v22.4s, v22.4s, v6.4s
        fmax v23.4s, v23.4s, v6.4s

        fmin v16.4s, v16.4s, v7.4s
        fmin v17.4s, v17.4s, v7.4s
        fmin v18.4s, v18.4s, v7.4s
        fmin v19.4s, v19.4s, v7.4s
        fmin v20.4s, v20.4s, v7.4s
        fmin v21.4s, v21.4s, v7.4s
        fmin v22.4s, v22.4s, v7.4s
        fmin v23.4s, v23.4s, v7.4s

        StoreLH8x4:
        shrn v16.4h, v16.4s, #16
        shrn v17.4h, v17.4s, #16
        shrn v18.4h, v18.4s, #16
        shrn v19.4h, v19.4s, #16
        shrn v20.4h, v20.4s, #16
        shrn v21.4h, v21.4s, #16
        shrn v22.4h, v22.4s, #16
        shrn v23.4h, v23.4s, #16

        stp d16, d17, [x0]
        stp d18, d19, [x0, #(16 * 1)]
        stp d20, d21, [x0, #(16 * 2)]
        stp d22, d23, [x0, #(16 * 3)]
        add x0, x0, #(16 * 4)

        // st1 {v16.4h, v17.4h, v18.4h, v19.4h}, [x0], #32
        // st1 {v20.4h, v21.4h, v22.4h, v23.4h}, [x0], #32

    E8End:

    sub x3, x3, #8
    cmp x3, #8
    add x0, x21, #64 // move dest address of 8 * 4 * sizeof(int16_t)
    add x1, x1, #16 // move A matrix address of 8 * sizeof(int16_t)
    bge LoopE8

E4:
cmp x3, #4
mov x20, x6
blt E1
    mov x8, x10
    mov x21, x0
    mov x13, x2

    cmp x8, #2
    blt E4LH4

    E4LH8:
    E4LoopH8:
        mov x15, x1
        subs x12, x9, #1
        ld1 {v3.4h, v4.4h}, [x13], #16
        ld1 {v0.4h}, [x15], x11
        shll v3.4s, v3.4h, #16
        shll v4.4s, v4.4h, #16
        shll v0.4s, v0.4h, #16

        fmul v16.4s, v3.4s, v0.s[0]
        fmul v17.4s, v3.4s, v0.s[1]
        fmul v18.4s, v3.4s, v0.s[2]
        fmul v19.4s, v3.4s, v0.s[3]

        fmul v20.4s, v4.4s, v0.s[0]
        fmul v21.4s, v4.4s, v0.s[1]
        fmul v22.4s, v4.4s, v0.s[2]
        fmul v23.4s, v4.4s, v0.s[3]

        beq E4LoopLEnd

        subs x12, x12, #1
            ld1 {v3.4h, v4.4h}, [x13], #16
            ld1 {v0.4h}, [x15], x11
            shll v3.4s, v3.4h, #16
            shll v4.4s, v4.4h, #16
            shll v0.4s, v0.4h, #16

            fmla v16.4s, v3.4s, v0.s[0]
            fmla v17.4s, v3.4s, v0.s[1]

        beq E4LoopLComputeEnd

        E4LoopL:
            fmla v18.4s, v3.4s, v0.s[2]
            fmla v19.4s, v3.4s, v0.s[3]

            fmla v20.4s, v4.4s, v0.s[0]
            fmla v21.4s, v4.4s, v0.s[1]
            fmla v22.4s, v4.4s, v0.s[2]
            fmla v23.4s, v4.4s, v0.s[3]

            ld1 {v3.4h, v4.4h}, [x13], #16
            ld1 {v0.4h}, [x15], x11
            shll v3.4s, v3.4h, #16
            shll v4.4s, v4.4h, #16
            shll v0.4s, v0.4h, #16

            fmla v16.4s, v3.4s, v0.s[0]
            fmla v17.4s, v3.4s, v0.s[1]

            subs x12, x12, #1
            bne E4LoopL
        E4LoopLComputeEnd:
        fmla v18.4s, v3.4s, v0.s[2]
        fmla v19.4s, v3.4s, v0.s[3]

        fmla v20.4s, v4.4s, v0.s[0]
        fmla v21.4s, v4.4s, v0.s[1]
        fmla v22.4s, v4.4s, v0.s[2]
        fmla v23.4s, v4.4s, v0.s[3]

        E4LoopLEnd:
        add x13, x13, x19
        sub x8, x8, #2
        cmp x8, #2

        cbz x5, StoreLH4x8

        AddBiasLH4x8:
        ld1 {v0.4h, v1.4h}, [x20], #16
        shll v0.4s, v0.4h, #16
        shll v1.4s, v1.4h, #16

        fmla v16.4s, v0.4s, v5.s[1]
        fmla v17.4s, v0.4s, v5.s[1]
        fmla v18.4s, v0.4s, v5.s[1]
        fmla v19.4s, v0.4s, v5.s[1]

        fmla v20.4s, v1.4s, v5.s[1]
        fmla v21.4s, v1.4s, v5.s[1]
        fmla v22.4s, v1.4s, v5.s[1]
        fmla v23.4s, v1.4s, v5.s[1]

        PostTreatLH4x8:
        fmax v16.4s, v16.4s, v6.4s
        fmax v17.4s, v17.4s, v6.4s
        fmax v18.4s, v18.4s, v6.4s
        fmax v19.4s, v19.4s, v6.4s
        fmax v20.4s, v20.4s, v6.4s
        fmax v21.4s, v21.4s, v6.4s
        fmax v22.4s, v22.4s, v6.4s
        fmax v23.4s, v23.4s, v6.4s

        fmin v16.4s, v16.4s, v7.4s
        fmin v17.4s, v17.4s, v7.4s
        fmin v18.4s, v18.4s, v7.4s
        fmin v19.4s, v19.4s, v7.4s
        fmin v20.4s, v20.4s, v7.4s
        fmin v21.4s, v21.4s, v7.4s
        fmin v22.4s, v22.4s, v7.4s
        fmin v23.4s, v23.4s, v7.4s

        StoreLH4x8:
        shrn v16.4h, v16.4s, #16
        shrn v17.4h, v17.4s, #16
        shrn v18.4h, v18.4s, #16
        shrn v19.4h, v19.4s, #16
        shrn v20.4h, v20.4s, #16
        shrn v21.4h, v21.4s, #16
        shrn v22.4h, v22.4s, #16
        shrn v23.4h, v23.4s, #16


        stp d16, d17, [x0]
        stp d18, d19, [x0, #16]
        add x0, x0, x7
        stp d20, d21, [x0]
        stp d22, d23, [x0, #16]
        add x0, x0, x7

        // st1 {v16.4h, v17.4h, v18.4h, v19.4h}, [x0], x7
        // st1 {v20.4h, v21.4h, v22.4h, v23.4h}, [x0], x7

        bge E4LoopH8

    E4LH4:
    cbz x8, E4End
    mov x15, x1
    subs x12, x9, #1
    ld1 {v3.4h}, [x13]
    ld1 {v0.4h}, [x15], x11
    shll v3.4s, v3.4h, #16
    shll v0.4s, v0.4h, #16

    fmul v16.4s, v3.4s, v0.s[0]
    fmul v17.4s, v3.4s, v0.s[1]
    fmul v18.4s, v3.4s, v0.s[2]
    fmul v19.4s, v3.4s, v0.s[3]
    add x13, x13, #16 // weight

    beq E4LoopLREnd

    E4LoopLR:
        ld1 {v3.4h}, [x13]
        ld1 {v0.4h}, [x15], x11
        shll v3.4s, v3.4h, #16
        shll v0.4s, v0.4h, #16

        fmla v16.4s, v3.4s, v0.s[0]
        fmla v17.4s, v3.4s, v0.s[1]
        fmla v18.4s, v3.4s, v0.s[2]
        fmla v19.4s, v3.4s, v0.s[3]
        add x13, x13, #16 // weight

        subs x12, x12, #1
        bne E4LoopLR
    E4LoopLREnd:

    cbz x5, StoreLH4x4
    AddBiasLH4x4:
    ld1 {v0.4h}, [x20]
    shll v0.4s, v0.4h, #16

    fmla v16.4s, v0.4s, v5.s[1]
    fmla v17.4s, v0.4s, v5.s[1]
    fmla v18.4s, v0.4s, v5.s[1]
    fmla v19.4s, v0.4s, v5.s[1]


    PostTreatLH4x4:
    fmax v16.4s, v16.4s, v6.4s
    fmax v17.4s, v17.4s, v6.4s
    fmax v18.4s, v18.4s, v6.4s
    fmax v19.4s, v19.4s, v6.4s

    fmin v16.4s, v16.4s, v7.4s
    fmin v17.4s, v17.4s, v7.4s
    fmin v18.4s, v18.4s, v7.4s
    fmin v19.4s, v19.4s, v7.4s

    StoreLH4x4:

    shrn v16.4h, v16.4s, #16
    shrn v17.4h, v17.4s, #16
    shrn v18.4h, v18.4s, #16
    shrn v19.4h, v19.4s, #16

    stp d16, d17, [x0]
    stp d18, d19, [x0, #16]

    // st1 {v16.4h, v17.4h, v18.4h, v19.4h}, [x0]

    E4End:

    sub x3, x3, #4
    add x0, x21, #32 // move dest address of 4 * 4 * sizeof(int16_t)
    add x1, x1, #8 // move dest address of 4 * sizeof(int16_t)

E1:
cmp x3, #0
beq End

LoopE1:
    mov x20, x6
    mov x8, x10
    mov x21, x0
    mov x13, x2

    cmp x8, #2
    blt E1LH4

    E1LH8:
    E1LoopH8:
        mov x15, x1
        subs x12, x9, #1
        ld1 {v3.4h, v4.4h}, [x13], #16 //
        ld1 {v0.h}[0], [x15], x11
        shll v3.4s, v3.4h, #16
        shll v4.4s, v4.4h, #16
        shll v0.4s, v0.4h, #16

        fmul v16.4s, v3.4s, v0.s[0]
        fmul v20.4s, v4.4s, v0.s[0]

        beq E1LoopLEnd

        E1LoopL:
            ld1 {v3.4h, v4.4h}, [x13], #16 //
            ld1 {v0.h}[0], [x15], x11
            shll v3.4s, v3.4h, #16
            shll v4.4s, v4.4h, #16
            shll v0.4s, v0.4h, #16

            fmla v16.4s, v3.4s, v0.s[0]
            fmla v20.4s, v4.4s, v0.s[0]

            subs x12, x12, #1
            bne E1LoopL

        E1LoopLEnd:

        add x13, x13, x19
        sub x8, x8, #2
        cmp x8, #2

        cbz x5, StoreLH1x8
        AddBiasLH1x8:
        ld1 {v0.4h, v1.4h}, [x20], #16
        shll v1.4s, v1.4h, #16
        shll v0.4s, v0.4h, #16

        fmla v16.4s, v0.4s, v5.s[1]
        fmla v20.4s, v1.4s, v5.s[1]

        PostTreatLH1x8:
        fmax v16.4s, v16.4s, v6.4s
        fmax v20.4s, v20.4s, v6.4s
        fmin v16.4s, v16.4s, v7.4s
        fmin v20.4s, v20.4s, v7.4s

        StoreLH1x8:
        shrn v16.4h, v16.4s, #16
        shrn v20.4h, v20.4s, #16
        st1 {v16.4h}, [x0], x7
        st1 {v20.4h}, [x0], x7

        bge E1LoopH8

    E1LH4:
    cbz x8, E1End
    mov x15, x1
    subs x12, x9, #1
    ld1 {v3.4h}, [x13]
    ld1 {v0.h}[0], [x15], x11
    shll v3.4s, v3.4h, #16
    shll v0.4s, v0.4h, #16

    fmul v16.4s, v3.4s, v0.s[0]
    add x13, x13, #16 // weight

    beq E1LoopLREnd

    E1LoopLR:
        ld1 {v3.4h}, [x13]
        ld1 {v0.h}[0], [x15], x11
        shll v3.4s, v3.4h, #16
        shll v0.4s, v0.4h, #16

        fmla v16.4s, v3.4s, v0.s[0]
        add x13, x13, #16 // weight

        subs x12, x12, #1
        bne E1LoopLR
    E1LoopLREnd:

    cbz x5, StoreLH1x4
    AddBiasLH1x4:
    ld1 {v0.4h}, [x20]
    shll v0.4s, v0.4h, #16

    fmla v16.4s, v0.4s, v5.s[1]

    PostTreatLH1x4:
    fmax v16.4s, v16.4s, v6.4s
    fmin v16.4s, v16.4s, v7.4s

    StoreLH1x4:
    shrn v16.4h, v16.4s, #16
    st1 {v16.4h}, [x0]

    E1End:

    subs x3, x3, #1
    add x0, x21, #8
    add x1, x1, #2
    bne LoopE1


End:
ldr x19, [sp, #0]
ldr x20, [sp, #8]
ldr x21, [sp, #16]
add sp, sp, #32

ret

#endif
