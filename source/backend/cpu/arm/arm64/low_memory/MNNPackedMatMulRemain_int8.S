//
//  MNNPackedMatMulRemain_int8.S
//  MNN
//
//  Created by MNN on 2023/06/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

.macro MNN_ADD_FLAOT s0, s1, s2, s3, z0, z1, z2, z3
    fadd \s0\().4s, \s0\().4s, \z0\().4s
    fadd \s1\().4s, \s1\().4s, \z1\().4s
    fadd \s2\().4s, \s2\().4s, \z2\().4s
    fadd \s3\().4s, \s3\().4s, \z3\().4s

.endm
// 12 * 8 MatMul
asm_function MNNPackedMatMulRemain_int8
//void MNNPackedMatMulRemain_int4(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter, const float* postParameters, const float* bias, const float* k, const float* b);
//Auto x0: C, x1:A, x2:B, x3:eSize, x4:parameter, x5:postParameters, x6:bias, x7: k, x8: b
ldr x8, [sp]
stp d14, d15, [sp, #(-16 * 8)]!
stp d12, d13, [sp, #16]
stp d10, d11, [sp, #32]
stp d8,  d9,  [sp, #48]
stp x25, x26, [sp, #(16 * 4)]
stp x23, x24, [sp, #(16 * 5)]
stp x21, x22, [sp, #(16 * 6)]
stp x19, x20, [sp, #(16 * 7)]

mov x22, x7 // alpha
mov x23, x8 // bias
ldr x11, [x4, #0] // aStride
ldr x9, [x4, #8] // l
ldr x10, [x4, #16] // h

ldr x7, [x4, #24] // cStride
ldr x19, [x4, #40] // bExtraStride
ldr x26, [x4, #48] // blockId

add x10, x10, #3
lsr x10, x10, #2
//lsl x25, x9, #3 // l*hPack
//add x25, x25, x19
cbz x5, Start
ld1 {v5.4s}, [x5]
dup v6.4s, v5.s[2] // Min Value
dup v7.4s, v5.s[3] // Max Value

Start:
sub x25, x7, #64

E8:
cmp x3, #8
blt E4

LoopE8:
    mov x20, x6
    mov x8, x10
    mov x21, x0
    mov x13, x2
    mov x14, x22
    mov x24, x23

    LH8:
    cmp x8, #2
    blt LH4
    // sub x14, x7, #64
    LoopH8x8:
        mov x15, x1
        subs x12, x9, #1
        ld1 {v12.4s, v13.4s}, [x14], #32 // alpha
        ld1 {v14.4s, v15.4s}, [x24], #32 // bias

        ld1 {v3.8b}, [x13], #8
        sxtl v4.8h, v3.8b
        sxtl v8.4s, v4.4h
        sxtl2 v9.4s, v4.8h
        scvtf v0.4s, v8.4s
        scvtf v1.4s, v9.4s
        fmul v3.4s, v0.4s, v12.4s
        fmul v4.4s, v1.4s, v13.4s
        fadd v3.4s, v3.4s, v14.4s
        fadd v4.4s, v4.4s, v15.4s
        
        ld1 {v0.4s, v1.4s}, [x15], x11
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
            ld1 {v3.8b}, [x13], #8
            sxtl v4.8h, v3.8b
            sxtl v8.4s, v4.4h
            sxtl2 v9.4s, v4.8h
            scvtf v0.4s, v8.4s
            scvtf v1.4s, v9.4s
            fmul v3.4s, v0.4s, v12.4s
            fmul v4.4s, v1.4s, v13.4s
            fadd v3.4s, v3.4s, v14.4s
            fadd v4.4s, v4.4s, v15.4s

            ld1 {v0.4s, v1.4s}, [x15], x11
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


        cbz x26, AddBiasLH8
        ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x0], #64
        MNN_ADD_FLAOT v16, v17, v18, v19, v0, v1, v2, v3
        ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x0], x25
        MNN_ADD_FLAOT v24, v25, v26, v27, v0, v1, v2, v3
        ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x0], #64
        MNN_ADD_FLAOT v20, v21, v22, v23, v0, v1, v2, v3
        ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x0]
        MNN_ADD_FLAOT v28, v29, v30, v31, v0, v1, v2, v3
        sub x0, x0, #128
        sub x0, x0, x25

        AddBiasLH8:
        cbz x5, StoreLH8
        ld1 {v0.4s, v1.4s}, [x20], #32

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
        stp q16, q17, [x0]
        stp q18, q19, [x0, #(32 * 1)]
        stp q24, q25, [x0, #(32 * 2)]
        stp q26, q27, [x0, #(32 * 3)]
        add x0, x0, x7 // stp donot support post-index offset in register

        stp q20, q21, [x0]
        stp q22, q23, [x0, #(32 * 1)]
        stp q28, q29, [x0, #(32 * 2)]
        stp q30, q31, [x0, #(32 * 3)]
        add x0, x0, x7

        // st1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x0], #64
        // st1 {v24.4s, v25.4s, v26.4s, v27.4s}, [x0], x14
        // st1 {v20.4s, v21.4s, v22.4s, v23.4s}, [x0], #64
        // st1 {v28.4s, v29.4s, v30.4s, v31.4s}, [x0], x14

        bge LoopH8x8

    LH4:
    cbz x8, E8End
    LoopHRemain:
        mov x15, x1
        subs x12, x9, #1
        ld1 {v12.4s}, [x14] // alpha
        ld1 {v14.4s}, [x24] // bias
        ld1 {v3.s}[0], [x13]
        sxtl v4.8h, v3.8b
        sxtl v8.4s, v4.4h
        scvtf v0.4s, v8.4s
        fmul v3.4s, v0.4s, v12.4s
        fadd v3.4s, v3.4s, v14.4s
        add x13, x13, #8

        ld1 {v0.4s}, [x15], #16
        fmul v16.4s, v3.4s, v0.s[0]
        fmul v17.4s, v3.4s, v0.s[1]
        ld1 {v1.4s}, [x15]
        fmul v18.4s, v3.4s, v0.s[2]
        sub x15, x15, #16
        fmul v19.4s, v3.4s, v0.s[3]
        add x15, x15, x11
        fmul v20.4s, v3.4s, v1.s[0]
        fmul v21.4s, v3.4s, v1.s[1]
        fmul v22.4s, v3.4s, v1.s[2]
        fmul v23.4s, v3.4s, v1.s[3]
        beq LoopLREnd

        LoopLR:
            ld1 {v3.s}[0], [x13]
            sxtl v4.8h, v3.8b
            sxtl v8.4s, v4.4h
            scvtf v0.4s, v8.4s
            fmul v3.4s, v0.4s, v12.4s
            fadd v3.4s, v3.4s, v14.4s
            ld1 {v0.4s}, [x15], #16
            fmla v16.4s, v3.4s, v0.s[0]
            fmla v17.4s, v3.4s, v0.s[1]
            add x13, x13, #8
            ld1 {v1.4s}, [x15]
            fmla v18.4s, v3.4s, v0.s[2]
            sub x15, x15, #16
            fmla v19.4s, v3.4s, v0.s[3]
            add x15, x15, x11

            fmla v20.4s, v3.4s, v1.s[0]
            fmla v21.4s, v3.4s, v1.s[1]
            fmla v22.4s, v3.4s, v1.s[2]
            fmla v23.4s, v3.4s, v1.s[3]

            subs x12, x12, #1
            bne LoopLR
        LoopLREnd:

        cbz x26, AddBiasLH8x4
        ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x0], #64
        MNN_ADD_FLAOT v16, v17, v18, v19, v0, v1, v2, v3
        ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x0]
        MNN_ADD_FLAOT v20, v21, v22, v23, v0, v1, v2, v3
        sub x0, x0, #64

        AddBiasLH8x4:
        cbz x20, PostTreatLH8x4
        ld1 {v0.4s}, [x20]

        fmla v16.4s, v0.4s, v5.s[1]
        fmla v17.4s, v0.4s, v5.s[1]
        fmla v18.4s, v0.4s, v5.s[1]
        fmla v19.4s, v0.4s, v5.s[1]

        fmla v20.4s, v0.4s, v5.s[1]
        fmla v21.4s, v0.4s, v5.s[1]
        fmla v22.4s, v0.4s, v5.s[1]
        fmla v23.4s, v0.4s, v5.s[1]

        PostTreatLH8x4:
        cbz x5, StoreLH8x4
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

        stp q16, q17, [x0]
        stp q18, q19, [x0, #(32 * 1)]
        stp q20, q21, [x0, #(32 * 2)]
        stp q22, q23, [x0, #(32 * 3)]
        add x0, x0, #(32 * 4)

        // st1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x0], #64
        // st1 {v20.4s, v21.4s, v22.4s, v23.4s}, [x0], #64

    E8End:

    sub x3, x3, #8
    cmp x3, #8
    add x0, x21, #128
    add x1, x1, #32
    bge LoopE8

E4:
cmp x3, #4
mov x20, x6
blt E1
    mov x8, x10
    mov x21, x0
    mov x13, x2
    mov x14, x22
    mov x24, x23

    cmp x8, #2
    blt E4LH4

    E4LH8:
    E4LoopH8:
        mov x15, x1
        subs x12, x9, #1

        ld1 {v12.4s, v13.4s}, [x14], #32 // alpha
        ld1 {v14.4s, v15.4s}, [x24], #32 // bias

        ld1 {v3.8b}, [x13], #8
        sxtl v4.8h, v3.8b
        sxtl v8.4s, v4.4h
        sxtl2 v9.4s, v4.8h
        scvtf v0.4s, v8.4s
        scvtf v1.4s, v9.4s
        fmul v3.4s, v0.4s, v12.4s
        fmul v4.4s, v1.4s, v13.4s
        fadd v3.4s, v3.4s, v14.4s
        fadd v4.4s, v4.4s, v15.4s

        ld1 {v0.4s}, [x15], x11
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

        ld1 {v3.8b}, [x13], #8
        sxtl v4.8h, v3.8b
        sxtl v8.4s, v4.4h
        sxtl2 v9.4s, v4.8h
        scvtf v0.4s, v8.4s
        scvtf v1.4s, v9.4s
        fmul v3.4s, v0.4s, v12.4s
        fmul v4.4s, v1.4s, v13.4s
        fadd v3.4s, v3.4s, v14.4s
        fadd v4.4s, v4.4s, v15.4s

            ld1 {v0.4s}, [x15], x11
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

            ld1 {v3.8b}, [x13], #8
            sxtl v4.8h, v3.8b
            sxtl v8.4s, v4.4h
            sxtl2 v9.4s, v4.8h
            scvtf v0.4s, v8.4s
            scvtf v1.4s, v9.4s
            fmul v3.4s, v0.4s, v12.4s
            fmul v4.4s, v1.4s, v13.4s
            fadd v3.4s, v3.4s, v14.4s
            fadd v4.4s, v4.4s, v15.4s
            
            ld1 {v0.4s}, [x15], x11
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

        cbz x26, AddBiasLH4x8
        ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x0], x7
        MNN_ADD_FLAOT v16, v17, v18, v19, v0, v1, v2, v3
        ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x0]
        MNN_ADD_FLAOT v20, v21, v22, v23, v0, v1, v2, v3
        sub x0, x0, x7

        AddBiasLH4x8:
        cbz x20, PostTreatLH4x8
        ld1 {v0.4s, v1.4s}, [x20], #32

        fmla v16.4s, v0.4s, v5.s[1]
        fmla v17.4s, v0.4s, v5.s[1]
        fmla v18.4s, v0.4s, v5.s[1]
        fmla v19.4s, v0.4s, v5.s[1]

        fmla v20.4s, v1.4s, v5.s[1]
        fmla v21.4s, v1.4s, v5.s[1]
        fmla v22.4s, v1.4s, v5.s[1]
        fmla v23.4s, v1.4s, v5.s[1]

        PostTreatLH4x8:
        cbz x5, StoreLH4x8
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
        stp q16, q17, [x0]
        stp q18, q19, [x0, #32]
        add x0, x0, x7 // stp donot support post-index offset in register
        stp q20, q21, [x0]
        stp q22, q23, [x0, #32]
        add x0, x0, x7

        // st1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x0], x7
        // st1 {v20.4s, v21.4s, v22.4s, v23.4s}, [x0], x7

        bge E4LoopH8

    E4LH4:
    cbz x8, E4End
    mov x15, x1
    subs x12, x9, #1

    ld1 {v12.4s}, [x14] // alpha
    ld1 {v14.4s}, [x24] // bias
    ld1 {v3.s}[0], [x13]
    sxtl v4.8h, v3.8b
    sxtl v8.4s, v4.4h
    scvtf v0.4s, v8.4s
    fmul v3.4s, v0.4s, v12.4s
    fadd v3.4s, v3.4s, v14.4s
    add x13, x13, #8

    ld1 {v0.4s}, [x15], x11
    fmul v16.4s, v3.4s, v0.s[0]
    fmul v17.4s, v3.4s, v0.s[1]
    fmul v18.4s, v3.4s, v0.s[2]
    fmul v19.4s, v3.4s, v0.s[3]

    beq E4LoopLREnd

    E4LoopLR:

        ld1 {v3.s}[0], [x13]
        sxtl v4.8h, v3.8b
        sxtl v8.4s, v4.4h
        scvtf v0.4s, v8.4s
        fmul v3.4s, v0.4s, v12.4s
        fadd v3.4s, v3.4s, v14.4s
        add x13, x13, #8

        ld1 {v0.4s}, [x15], x11
        fmla v16.4s, v3.4s, v0.s[0]
        fmla v17.4s, v3.4s, v0.s[1]
        fmla v18.4s, v3.4s, v0.s[2]
        fmla v19.4s, v3.4s, v0.s[3]

        subs x12, x12, #1
        bne E4LoopLR
    E4LoopLREnd:

    cbz x26, AddBiasLH4x4
    ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x0]
    MNN_ADD_FLAOT v16, v17, v18, v19, v0, v1, v2, v3

    AddBiasLH4x4:
    cbz x5, StoreLH4x4
    ld1 {v0.4s}, [x20]

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
    stp q16, q17, [x0]
    stp q18, q19, [x0, #32]
    // st1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x0]

    E4End:

    sub x3, x3, #4
    add x0, x21, #64
    add x1, x1, #16

E1:
cmp x3, #0
beq End

LoopE1:
    mov x20, x6
    mov x8, x10
    mov x21, x0
    mov x13, x2
    mov x14, x22
    mov x24, x23

    cmp x8, #2
    blt E1LH4

    E1LH8:
    E1LoopH8:
        mov x15, x1
        subs x12, x9, #1

        ld1 {v3.8b}, [x13], #8
        sxtl v4.8h, v3.8b
        sxtl v8.4s, v4.4h
        sxtl2 v9.4s, v4.8h
        scvtf v3.4s, v8.4s
        scvtf v4.4s, v9.4s

        ld1 {v0.s}[0], [x15], x11
        fmul v16.4s, v3.4s, v0.s[0]
        fmul v20.4s, v4.4s, v0.s[0]

        mov v2.16b, v0.16b

        beq E1LoopLEnd

        E1LoopL:

            ld1 {v3.8b}, [x13], #8
            sxtl v4.8h, v3.8b
            sxtl v8.4s, v4.4h
            ld1 {v0.s}[0], [x15], x11
            sxtl2 v9.4s, v4.8h
            scvtf v3.4s, v8.4s
            scvtf v4.4s, v9.4s
            fmla v16.4s, v3.4s, v0.s[0]
            fmla v20.4s, v4.4s, v0.s[0]
            fadd v2.2s, v0.2s, v2.2s

            subs x12, x12, #1
            bne E1LoopL

        E1LoopLEnd:

        ld1 {v12.4s, v13.4s}, [x14], #32 // alpha
        ld1 {v14.4s, v15.4s}, [x24], #32 // bias

        fmul v16.4s, v16.4s, v12.4s
        fmul v20.4s, v20.4s, v13.4s
        fmla v16.4s, v14.4s, v2.s[0]
        fmla v20.4s, v15.4s, v2.s[0]

        add x13, x13, x19
        sub x8, x8, #2
        cmp x8, #2

        cbz x26, AddBiasLH1x8
        ld1 {v0.4s}, [x0], x7
        ld1 {v1.4s}, [x0]
        fadd v16.4s, v16.4s, v0.4s
        fadd v20.4s, v20.4s, v1.4s
        sub x0, x0, x7

        AddBiasLH1x8:
        cbz x5, StoreLH1x8
        ld1 {v0.4s, v1.4s}, [x20], #32

        fmla v16.4s, v0.4s, v5.s[1]
        fmla v20.4s, v1.4s, v5.s[1]

        PostTreatLH1x8:
        fmax v16.4s, v16.4s, v6.4s
        fmax v20.4s, v20.4s, v6.4s
        fmin v16.4s, v16.4s, v7.4s
        fmin v20.4s, v20.4s, v7.4s

        StoreLH1x8:

        st1 {v16.4s}, [x0], x7
        st1 {v20.4s}, [x0], x7

        bge E1LoopH8

    E1LH4:
    cbz x8, E1End
    mov x15, x1
    subs x12, x9, #1

    ld1 {v12.4s}, [x14] // alpha
    ld1 {v14.4s}, [x24] // bias
    ld1 {v3.s}[0], [x13]
    sxtl v4.8h, v3.8b
    sxtl v8.4s, v4.4h
    scvtf v0.4s, v8.4s
    fmul v3.4s, v0.4s, v12.4s
    fadd v3.4s, v3.4s, v14.4s
    add x13, x13, #8

    ld1 {v0.s}[0], [x15], x11
    fmul v16.4s, v3.4s, v0.s[0]

    beq E1LoopLREnd

    E1LoopLR:

        ld1 {v3.s}[0], [x13]
        sxtl v4.8h, v3.8b
        sxtl v8.4s, v4.4h
        scvtf v0.4s, v8.4s
        fmul v3.4s, v0.4s, v12.4s
        fadd v3.4s, v3.4s, v14.4s
        add x13, x13, #8
        ld1 {v0.s}[0], [x15], x11
        fmla v16.4s, v3.4s, v0.s[0]

        subs x12, x12, #1
        bne E1LoopLR
    E1LoopLREnd:

    cbz x26, AddBiasLH1x4
    ld1 {v0.4s}, [x0]
    fadd v16.4s, v16.4s, v0.4s

    AddBiasLH1x4:
    cbz x5, StoreLH1x4
    ld1 {v0.4s}, [x20]
    fmla v16.4s, v0.4s, v5.s[1]

    PostTreatLH1x4:
    fmax v16.4s, v16.4s, v6.4s
    fmin v16.4s, v16.4s, v7.4s

    StoreLH1x4:
    st1 {v16.4s}, [x0]

    E1End:

    subs x3, x3, #1
    add x0, x21, #16
    add x1, x1, #4
    bne LoopE1


End:
ldp x19, x20, [sp, #(16 * 7)]
ldp x21, x22, [sp, #(16 * 6)]
ldp x23, x24, [sp, #(16 * 5)]
ldp x25, x26, [sp, #(16 * 4)]
ldp d8,  d9,  [sp, #48]
ldp d10, d11, [sp, #32]
ldp d12, d13, [sp, #16]
ldp d14, d15, [sp], #(16 * 8)

ret

#endif
