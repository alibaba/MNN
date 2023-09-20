//
//  MNNPackedMatMulRemain_int4.S
//  MNN
//
//  Created by MNN on 2023/05/18.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5
// 12 * 8 MatMul
asm_function MNNPackedMatMulRemain_int4
//void MNNPackedMatMulRemain_int4(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter, const float* postParameters, const float* bias, const float* k, const float* b);
//Auto x0: C, x1:A, x2:B, x3:eSize, x4:parameter, x5:postParameters, x6:bias, x7: k, x8: b
ldr x8, [sp]
sub sp, sp, #64
str x19, [sp, #0]
str x20, [sp, #8]
str x21, [sp, #16]
str x22, [sp, #24]
str x23, [sp, #32]

mov x22, x7 // alpha
mov x23, x8 // bias
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

LoopE8:
    mov x20, x6
    mov x8, x10
    mov x21, x0
    mov x13, x2
    mov x14, x22
    mov x16, x23

    LH8:
    cmp x8, #2
    blt LH4

    // sub x14, x7, #64
    LoopH8x8:
        mov x15, x1
        ld1 {v12.4s, v13.4s}, [x14], #32 // alpha
        mov w17, #0x0f
        dup v3.8b, w17
        mov w17, #8
        dup v4.8b, w17
        ld1 {v14.4s, v15.4s}, [x16], #32 // bias
        subs x12, x9, #2
        ld1 {v0.4h}, [x13], #8
        ushr v1.8b, v0.8b, #4
        and v2.8b, v0.8b, v3.8b
        sub v1.8b, v1.8b, v4.8b
        sub v2.8b, v2.8b, v4.8b
        zip1 v9.8b, v1.8b, v2.8b
        zip2 v10.8b, v1.8b, v2.8b
        sxtl v0.8h, v9.8b
        sxtl v1.8h, v10.8b
        sxtl v8.4s, v0.4h
        sxtl2 v9.4s, v0.8h
        sxtl v10.4s, v1.4h
        sxtl2 v11.4s, v1.8h
        scvtf v0.4s, v8.4s
        scvtf v1.4s, v9.4s
        mov v8.16b, v14.16b
        mov v9.16b, v15.16b
        fmla v8.4s, v0.4s, v12.4s
        fmla v9.4s, v1.4s, v13.4s
        scvtf v0.4s, v10.4s
        scvtf v1.4s, v11.4s
        mov v10.16b, v14.16b
        mov v11.16b, v15.16b
        fmla v10.4s, v0.4s, v12.4s
        fmla v11.4s, v1.4s, v13.4s
        ld1 {v0.4s, v1.4s}, [x15], x11
        fmul v16.4s, v8.4s, v0.s[0]
        fmul v17.4s, v8.4s, v0.s[1]
        fmul v18.4s, v8.4s, v0.s[2]
        fmul v19.4s, v8.4s, v0.s[3]

        fmul v20.4s, v9.4s, v0.s[0]
        fmul v21.4s, v9.4s, v0.s[1]
        fmul v22.4s, v9.4s, v0.s[2]
        fmul v23.4s, v9.4s, v0.s[3]

        fmul v24.4s, v8.4s, v1.s[0]
        fmul v25.4s, v8.4s, v1.s[1]
        fmul v26.4s, v8.4s, v1.s[2]
        fmul v27.4s, v8.4s, v1.s[3]

        fmul v28.4s, v9.4s, v1.s[0]
        fmul v29.4s, v9.4s, v1.s[1]
        fmul v30.4s, v9.4s, v1.s[2]
        fmul v31.4s, v9.4s, v1.s[3]
        ld1 {v0.4s, v1.4s}, [x15], x11
        fmla v16.4s, v10.4s, v0.s[0]
        fmla v17.4s, v10.4s, v0.s[1]
        fmla v18.4s, v10.4s, v0.s[2]
        fmla v19.4s, v10.4s, v0.s[3]

        fmla v20.4s, v11.4s, v0.s[0]
        fmla v21.4s, v11.4s, v0.s[1]
        fmla v22.4s, v11.4s, v0.s[2]
        fmla v23.4s, v11.4s, v0.s[3]

        fmla v24.4s, v10.4s, v1.s[0]
        fmla v25.4s, v10.4s, v1.s[1]
        fmla v26.4s, v10.4s, v1.s[2]
        fmla v27.4s, v10.4s, v1.s[3]

        fmla v28.4s, v11.4s, v1.s[0]
        fmla v29.4s, v11.4s, v1.s[1]
        fmla v30.4s, v11.4s, v1.s[2]
        fmla v31.4s, v11.4s, v1.s[3]
        beq LoopLEnd

        LoopL:
            subs x12, x12, #2
            ld1 {v0.4h}, [x13], #8
            ushr v1.8b, v0.8b, #4
            and v2.8b, v0.8b, v3.8b
            sub v1.8b, v1.8b, v4.8b
            sub v2.8b, v2.8b, v4.8b
            zip1 v9.8b, v1.8b, v2.8b
            zip2 v10.8b, v1.8b, v2.8b
            sxtl v0.8h, v9.8b
            sxtl v1.8h, v10.8b
            sxtl v8.4s, v0.4h
            sxtl2 v9.4s, v0.8h
            sxtl v10.4s, v1.4h
            sxtl2 v11.4s, v1.8h
            scvtf v0.4s, v8.4s
            scvtf v1.4s, v9.4s
            mov v8.16b, v14.16b
            mov v9.16b, v15.16b
            fmla v8.4s, v0.4s, v12.4s
            fmla v9.4s, v1.4s, v13.4s
            scvtf v0.4s, v10.4s
            scvtf v1.4s, v11.4s
            mov v10.16b, v14.16b
            mov v11.16b, v15.16b
            fmla v10.4s, v0.4s, v12.4s
            fmla v11.4s, v1.4s, v13.4s
            ld1 {v0.4s, v1.4s}, [x15], x11
            fmla v16.4s, v8.4s, v0.s[0]
            fmla v17.4s, v8.4s, v0.s[1]
            fmla v18.4s, v8.4s, v0.s[2]
            fmla v19.4s, v8.4s, v0.s[3]

            fmla v20.4s, v9.4s, v0.s[0]
            fmla v21.4s, v9.4s, v0.s[1]
            fmla v22.4s, v9.4s, v0.s[2]
            fmla v23.4s, v9.4s, v0.s[3]

            fmla v24.4s, v8.4s, v1.s[0]
            fmla v25.4s, v8.4s, v1.s[1]
            fmla v26.4s, v8.4s, v1.s[2]
            fmla v27.4s, v8.4s, v1.s[3]

            fmla v28.4s, v9.4s, v1.s[0]
            fmla v29.4s, v9.4s, v1.s[1]
            fmla v30.4s, v9.4s, v1.s[2]
            fmla v31.4s, v9.4s, v1.s[3]

            ld1 {v0.4s, v1.4s}, [x15], x11
            fmla v16.4s, v10.4s, v0.s[0]
            fmla v17.4s, v10.4s, v0.s[1]
            fmla v18.4s, v10.4s, v0.s[2]
            fmla v19.4s, v10.4s, v0.s[3]

            fmla v20.4s, v11.4s, v0.s[0]
            fmla v21.4s, v11.4s, v0.s[1]
            fmla v22.4s, v11.4s, v0.s[2]
            fmla v23.4s, v11.4s, v0.s[3]

            fmla v24.4s, v10.4s, v1.s[0]
            fmla v25.4s, v10.4s, v1.s[1]
            fmla v26.4s, v10.4s, v1.s[2]
            fmla v27.4s, v10.4s, v1.s[3]

            fmla v28.4s, v11.4s, v1.s[0]
            fmla v29.4s, v11.4s, v1.s[1]
            fmla v30.4s, v11.4s, v1.s[2]
            fmla v31.4s, v11.4s, v1.s[3]

            bne LoopL

        LoopLEnd:

        add x13, x13, x19
        sub x8, x8, #2
        cmp x8, #2

        cbz x5, StoreLH8
        AddBiasLH8:
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
        ld1 {v4.4s}, [x14], #16 // alpha
        mov w17, #0x0f
        dup v30.8b, w17
        mov w17, #8
        dup v31.8b, w17
        ld1 {v14.4s}, [x16], #16 // bias
        subs x12, x9, #4
        // load 16xint4 to 4xfloat
        ld1 {v3.8h}, [x13]
        uzp1 v0.8h, v3.8h, v3.8h
        ushr v1.8b, v0.8b, #4
        and v2.8b, v0.8b, v30.8b
        sub v1.8b, v1.8b, v31.8b
        sub v2.8b, v2.8b, v31.8b

        zip1 v9.8b, v1.8b, v2.8b
        zip2 v10.8b, v1.8b, v2.8b
        sxtl v11.8h, v9.8b
        sxtl v12.8h, v10.8b
        sxtl v8.4s, v11.4h
        sxtl2 v9.4s, v11.8h
        sxtl v10.4s, v12.4h
        sxtl2 v11.4s, v12.8h
        scvtf v12.4s, v8.4s
        scvtf v13.4s, v9.4s
        mov v8.16b, v14.16b
        mov v9.16b, v14.16b
        fmla v8.4s, v12.4s, v4.4s
        fmla v9.4s, v13.4s, v4.4s
        scvtf v12.4s, v10.4s
        scvtf v13.4s, v11.4s
        mov v10.16b, v14.16b
        mov v11.16b, v14.16b
        fmla v10.4s, v12.4s, v4.4s
        fmla v11.4s, v13.4s, v4.4s

        ld1 {v0.4s}, [x15], #16
        fmul v16.4s, v8.4s, v0.s[0]
        fmul v17.4s, v8.4s, v0.s[1]
        ld1 {v1.4s}, [x15]
        fmul v18.4s, v8.4s, v0.s[2]
        sub x15, x15, #16
        fmul v19.4s, v8.4s, v0.s[3]
        add x15, x15, x11
        fmul v20.4s, v8.4s, v1.s[0]
        fmul v21.4s, v8.4s, v1.s[1]
        fmul v22.4s, v8.4s, v1.s[2]
        fmul v23.4s, v8.4s, v1.s[3]

        ld1 {v0.4s}, [x15], #16
        fmla v16.4s, v9.4s, v0.s[0]
        fmla v17.4s, v9.4s, v0.s[1]
        ld1 {v1.4s}, [x15]
        fmla v18.4s, v9.4s, v0.s[2]
        sub x15, x15, #16
        fmla v19.4s, v9.4s, v0.s[3]
        add x15, x15, x11
        fmla v20.4s, v9.4s, v1.s[0]
        fmla v21.4s, v9.4s, v1.s[1]
        fmla v22.4s, v9.4s, v1.s[2]
        fmla v23.4s, v9.4s, v1.s[3]

        ld1 {v0.4s}, [x15], #16
        fmla v16.4s, v10.4s, v0.s[0]
        fmla v17.4s, v10.4s, v0.s[1]
        ld1 {v1.4s}, [x15]
        fmla v18.4s, v10.4s, v0.s[2]
        sub x15, x15, #16
        fmla v19.4s, v10.4s, v0.s[3]
        add x15, x15, x11
        fmla v20.4s, v10.4s, v1.s[0]
        fmla v21.4s, v10.4s, v1.s[1]
        fmla v22.4s, v10.4s, v1.s[2]
        fmla v23.4s, v10.4s, v1.s[3]

        ld1 {v0.4s}, [x15], #16
        fmla v16.4s, v11.4s, v0.s[0]
        fmla v17.4s, v11.4s, v0.s[1]
        ld1 {v1.4s}, [x15]
        fmla v18.4s, v11.4s, v0.s[2]
        sub x15, x15, #16
        fmla v19.4s, v11.4s, v0.s[3]
        add x15, x15, x11
        fmla v20.4s, v11.4s, v1.s[0]
        fmla v21.4s, v11.4s, v1.s[1]
        fmla v22.4s, v11.4s, v1.s[2]
        fmla v23.4s, v11.4s, v1.s[3]

        add x13, x13, #16
        beq LoopLREnd

        LoopLR:
            ld1 {v3.8h}, [x13]
            uzp1 v0.8h, v3.8h, v3.8h
            ushr v1.8b, v0.8b, #4
            and v2.8b, v0.8b, v30.8b
            sub v1.8b, v1.8b, v31.8b
            sub v2.8b, v2.8b, v31.8b

            zip1 v9.8b, v1.8b, v2.8b
            zip2 v10.8b, v1.8b, v2.8b
            sxtl v11.8h, v9.8b
            sxtl v12.8h, v10.8b
            sxtl v8.4s, v11.4h
            sxtl2 v9.4s, v11.8h
            sxtl v10.4s, v12.4h
            sxtl2 v11.4s, v12.8h
            scvtf v12.4s, v8.4s
            scvtf v13.4s, v9.4s
            mov v8.16b, v14.16b
            mov v9.16b, v14.16b
            fmla v8.4s, v12.4s, v4.4s
            fmla v9.4s, v13.4s, v4.4s
            scvtf v12.4s, v10.4s
            scvtf v13.4s, v11.4s
            mov v10.16b, v14.16b
            mov v11.16b, v14.16b
            fmla v10.4s, v12.4s, v4.4s
            fmla v11.4s, v13.4s, v4.4s

            ld1 {v0.4s}, [x15], #16
            fmla v16.4s, v8.4s, v0.s[0]
            fmla v17.4s, v8.4s, v0.s[1]
            ld1 {v1.4s}, [x15]
            fmla v18.4s, v8.4s, v0.s[2]
            sub x15, x15, #16
            fmla v19.4s, v8.4s, v0.s[3]
            add x15, x15, x11
            fmla v20.4s, v8.4s, v1.s[0]
            fmla v21.4s, v8.4s, v1.s[1]
            fmla v22.4s, v8.4s, v1.s[2]
            fmla v23.4s, v8.4s, v1.s[3]

            ld1 {v0.4s}, [x15], #16
            fmla v16.4s, v9.4s, v0.s[0]
            fmla v17.4s, v9.4s, v0.s[1]
            ld1 {v1.4s}, [x15]
            fmla v18.4s, v9.4s, v0.s[2]
            sub x15, x15, #16
            fmla v19.4s, v9.4s, v0.s[3]
            add x15, x15, x11
            fmla v20.4s, v9.4s, v1.s[0]
            fmla v21.4s, v9.4s, v1.s[1]
            fmla v22.4s, v9.4s, v1.s[2]
            fmla v23.4s, v9.4s, v1.s[3]

            ld1 {v0.4s}, [x15], #16
            fmla v16.4s, v10.4s, v0.s[0]
            fmla v17.4s, v10.4s, v0.s[1]
            ld1 {v1.4s}, [x15]
            fmla v18.4s, v10.4s, v0.s[2]
            sub x15, x15, #16
            fmla v19.4s, v10.4s, v0.s[3]
            add x15, x15, x11
            fmla v20.4s, v10.4s, v1.s[0]
            fmla v21.4s, v10.4s, v1.s[1]
            fmla v22.4s, v10.4s, v1.s[2]
            fmla v23.4s, v10.4s, v1.s[3]

            ld1 {v0.4s}, [x15], #16
            fmla v16.4s, v11.4s, v0.s[0]
            fmla v17.4s, v11.4s, v0.s[1]
            ld1 {v1.4s}, [x15]
            fmla v18.4s, v11.4s, v0.s[2]
            sub x15, x15, #16
            fmla v19.4s, v11.4s, v0.s[3]
            add x15, x15, x11
            fmla v20.4s, v11.4s, v1.s[0]
            fmla v21.4s, v11.4s, v1.s[1]
            fmla v22.4s, v11.4s, v1.s[2]
            fmla v23.4s, v11.4s, v1.s[3]

            add x13, x13, #16
            subs x12, x12, #4
            bne LoopLR
        LoopLREnd:

        cbz x5, StoreLH8x4
        AddBiasLH8x4:
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
    mov x16, x23

    cmp x8, #2
    blt E4LH4

    E4LH8:
    E4LoopH8:
        mov x15, x1
        ld1 {v24.4s, v25.4s}, [x14], #32 // alpha
        mov w17, #0x0f
        dup v30.8b, w17
        mov w17, #8
        dup v31.8b, w17
        ld1 {v26.4s, v27.4s}, [x16], #32 // bias
        subs x12, x9, #2
        // ld1 {v3.4s, v4.4s}, [x13], #32
        ld1 {v0.4h}, [x13], #8
        ushr v1.8b, v0.8b, #4
        and v2.8b, v0.8b, v30.8b
        sub v1.8b, v1.8b, v31.8b
        sub v2.8b, v2.8b, v31.8b
        zip1 v9.8b, v1.8b, v2.8b
        zip2 v10.8b, v1.8b, v2.8b
        sxtl v11.8h, v9.8b
        sxtl v12.8h, v10.8b
        sxtl v8.4s, v11.4h
        sxtl2 v9.4s, v11.8h
        sxtl v10.4s, v12.4h
        sxtl2 v11.4s, v12.8h
        scvtf v12.4s, v8.4s
        scvtf v13.4s, v9.4s
        mov v8.16b, v26.16b
        mov v9.16b, v27.16b
        fmla v8.4s, v12.4s, v24.4s
        fmla v9.4s, v13.4s, v25.4s
        scvtf v12.4s, v10.4s
        scvtf v13.4s, v11.4s
        mov v10.16b, v26.16b
        mov v11.16b, v27.16b
        fmla v10.4s, v12.4s, v24.4s
        fmla v11.4s, v13.4s, v25.4s

        ld1 {v0.4s}, [x15], x11
        ld1 {v1.4s}, [x15], x11
        fmul v16.4s, v8.4s, v0.s[0]
        fmul v17.4s, v8.4s, v0.s[1]
        fmul v18.4s, v8.4s, v0.s[2]
        fmul v19.4s, v8.4s, v0.s[3]

        fmul v20.4s, v9.4s, v0.s[0]
        fmul v21.4s, v9.4s, v0.s[1]
        fmul v22.4s, v9.4s, v0.s[2]
        fmul v23.4s, v9.4s, v0.s[3]

        fmla v16.4s, v10.4s, v1.s[0]
        fmla v17.4s, v10.4s, v1.s[1]
        fmla v18.4s, v10.4s, v1.s[2]
        fmla v19.4s, v10.4s, v1.s[3]

        fmla v20.4s, v11.4s, v1.s[0]
        fmla v21.4s, v11.4s, v1.s[1]
        fmla v22.4s, v11.4s, v1.s[2]
        fmla v23.4s, v11.4s, v1.s[3]
        beq E4LoopLEnd

        E4LoopL:
            subs x12, x12, #2
            // ld1 {v3.4s, v4.4s}, [x13], #32
            ld1 {v0.4h}, [x13], #8
            ushr v1.8b, v0.8b, #4
            and v2.8b, v0.8b, v30.8b
            sub v1.8b, v1.8b, v31.8b
            sub v2.8b, v2.8b, v31.8b
            zip1 v9.8b, v1.8b, v2.8b
            zip2 v10.8b, v1.8b, v2.8b
            sxtl v11.8h, v9.8b
            sxtl v12.8h, v10.8b
            sxtl v8.4s, v11.4h
            sxtl2 v9.4s, v11.8h
            sxtl v10.4s, v12.4h
            sxtl2 v11.4s, v12.8h
            scvtf v12.4s, v8.4s
            scvtf v13.4s, v9.4s
            mov v8.16b, v26.16b
            mov v9.16b, v27.16b
            fmla v8.4s, v12.4s, v24.4s
            fmla v9.4s, v13.4s, v25.4s
            scvtf v12.4s, v10.4s
            scvtf v13.4s, v11.4s
            mov v10.16b, v26.16b
            mov v11.16b, v27.16b
            fmla v10.4s, v12.4s, v24.4s
            fmla v11.4s, v13.4s, v25.4s
            ld1 {v0.4s}, [x15], x11
            ld1 {v1.4s}, [x15], x11
            fmla v16.4s, v8.4s, v0.s[0]
            fmla v17.4s, v8.4s, v0.s[1]
            fmla v18.4s, v8.4s, v0.s[2]
            fmla v19.4s, v8.4s, v0.s[3]

            fmla v20.4s, v9.4s, v0.s[0]
            fmla v21.4s, v9.4s, v0.s[1]
            fmla v22.4s, v9.4s, v0.s[2]
            fmla v23.4s, v9.4s, v0.s[3]

            fmla v16.4s, v10.4s, v1.s[0]
            fmla v17.4s, v10.4s, v1.s[1]
            fmla v18.4s, v10.4s, v1.s[2]
            fmla v19.4s, v10.4s, v1.s[3]

            fmla v20.4s, v11.4s, v1.s[0]
            fmla v21.4s, v11.4s, v1.s[1]
            fmla v22.4s, v11.4s, v1.s[2]
            fmla v23.4s, v11.4s, v1.s[3]
            bne E4LoopL
        E4LoopLEnd:
        add x13, x13, x19
        sub x8, x8, #2
        cmp x8, #2

        cbz x5, StoreLH4x8

        AddBiasLH4x8:
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
#if 0
    // l_tile = 1 loop
    ld1 {v1.4s}, [x14], #16 // alpha
    ld1 {v2.4s}, [x16], #16 // bias
    subs x12, x9, #1
    ldrb w8, [x13]
    ldrb w9, [x13, #1]
    movi v8.4s, #15
    fmov s3, w8
    mov  v3.s[1], w9
    zip1 v3.4s, v3.4s, v3.4s
    ushr v4.4s, v3.4s, #4
    and  v3.16b, v3.16b, v8.16b
    rev64 v4.4s, v4.4s
    trn2  v3.4s, v4.4s, v3.4s
    mvni  v9.4s, #6
    add  v3.4s, v3.4s, v9.4s
    scvtf v3.4s, v3.4s
    mov v4.16b, v2.16b
    fmla v4.4s, v3.4s, v1.4s

    ld1 {v0.4s}, [x15], x11
    fmul v16.4s, v4.4s, v0.s[0]
    fmul v17.4s, v4.4s, v0.s[1]
    fmul v18.4s, v4.4s, v0.s[2]
    fmul v19.4s, v4.4s, v0.s[3]
    add x13, x13, #4
#else
    ld1 {v4.4s}, [x14], #16 // alpha
    mov w17, #0x0f
    dup v30.8b, w17
    mov w17, #8
    dup v31.8b, w17
    ld1 {v14.4s}, [x16], #16 // bias
    subs x12, x9, #4
    // load 16xint4 to 16xfloat
    ld1 {v3.8h}, [x13]
    uzp1 v0.8h, v3.8h, v3.8h
    ushr v1.8b, v0.8b, #4
    and v2.8b, v0.8b, v30.8b
    sub v1.8b, v1.8b, v31.8b
    sub v2.8b, v2.8b, v31.8b
    zip1 v9.8b, v1.8b, v2.8b
    zip2 v10.8b, v1.8b, v2.8b
    sxtl v11.8h, v9.8b
    sxtl v12.8h, v10.8b
    sxtl v8.4s, v11.4h
    sxtl2 v9.4s, v11.8h
    sxtl v10.4s, v12.4h
    sxtl2 v11.4s, v12.8h
    ld1 {v0.4s}, [x15], x11
    scvtf v12.4s, v8.4s
    scvtf v13.4s, v9.4s
    mov v8.16b, v14.16b
    mov v9.16b, v14.16b
    fmla v8.4s, v12.4s, v4.4s
    ld1 {v1.4s}, [x15], x11
    fmla v9.4s, v13.4s, v4.4s
    scvtf v12.4s, v10.4s
    ld1 {v2.4s}, [x15], x11
    scvtf v13.4s, v11.4s
    mov v10.16b, v14.16b
    mov v11.16b, v14.16b
    ld1 {v3.4s}, [x15], x11
    fmla v10.4s, v12.4s, v4.4s
    fmla v11.4s, v13.4s, v4.4s

    // ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x15], x11
    fmul v16.4s, v8.4s, v0.s[0]
    fmul v17.4s, v8.4s, v0.s[1]
    fmul v18.4s, v8.4s, v0.s[2]
    fmul v19.4s, v8.4s, v0.s[3]

    fmla v16.4s, v9.4s, v1.s[0]
    fmla v17.4s, v9.4s, v1.s[1]
    fmla v18.4s, v9.4s, v1.s[2]
    fmla v19.4s, v9.4s, v1.s[3]

    fmla v16.4s, v10.4s, v2.s[0]
    fmla v17.4s, v10.4s, v2.s[1]
    fmla v18.4s, v10.4s, v2.s[2]
    fmla v19.4s, v10.4s, v2.s[3]

    fmla v16.4s, v11.4s, v3.s[0]
    fmla v17.4s, v11.4s, v3.s[1]
    fmla v18.4s, v11.4s, v3.s[2]
    fmla v19.4s, v11.4s, v3.s[3]
    add x13, x13, #16
#endif
    beq E4LoopLREnd

    E4LoopLR:
#if 0
        // ld1 {v3.4s}, [x13]
        ldrb w8, [x13]
        ldrb w9, [x13, #1]
        movi v8.4s, #15
        fmov s3, w8
        mov  v3.s[1], w9
        zip1 v3.4s, v3.4s, v3.4s
        ushr v4.4s, v3.4s, #4
        and  v3.16b, v3.16b, v8.16b
        rev64 v4.4s, v4.4s
        trn2  v3.4s, v4.4s, v3.4s
        mvni  v9.4s, #6
        add  v3.4s, v3.4s, v9.4s
        scvtf v3.4s, v3.4s
        mov v4.16b, v2.16b
        fmla v4.4s, v3.4s, v1.4s

        ld1 {v0.4s}, [x15], x11
        fmla v16.4s, v4.4s, v0.s[0]
        fmla v17.4s, v4.4s, v0.s[1]
        fmla v18.4s, v4.4s, v0.s[2]
        fmla v19.4s, v4.4s, v0.s[3]
        add x13, x13, #4

        subs x12, x12, #1
#else
        ld1 {v3.8h}, [x13]
        // 0123xxxx4567xxxx => 01234567...
        uzp1 v0.8h, v3.8h, v3.8h
        ushr v1.8b, v0.8b, #4
        and v2.8b, v0.8b, v30.8b
        sub v1.8b, v1.8b, v31.8b
        sub v2.8b, v2.8b, v31.8b

        zip1 v9.8b, v1.8b, v2.8b
        zip2 v10.8b, v1.8b, v2.8b
        sxtl v11.8h, v9.8b
        sxtl v12.8h, v10.8b
        sxtl v8.4s, v11.4h
        sxtl2 v9.4s, v11.8h
        sxtl v10.4s, v12.4h
        sxtl2 v11.4s, v12.8h
        scvtf v12.4s, v8.4s
        scvtf v13.4s, v9.4s
        mov v8.16b, v14.16b
        mov v9.16b, v14.16b
        fmla v8.4s, v12.4s, v4.4s
        fmla v9.4s, v13.4s, v4.4s
        scvtf v12.4s, v10.4s
        scvtf v13.4s, v11.4s
        mov v10.16b, v14.16b
        mov v11.16b, v14.16b
        fmla v10.4s, v12.4s, v4.4s
        fmla v11.4s, v13.4s, v4.4s
        // ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x15], x11
        ld1 {v0.4s}, [x15], x11
        ld1 {v1.4s}, [x15], x11
        ld1 {v2.4s}, [x15], x11
        ld1 {v3.4s}, [x15], x11
        fmla v16.4s, v8.4s, v0.s[0]
        fmla v17.4s, v8.4s, v0.s[1]
        fmla v18.4s, v8.4s, v0.s[2]
        fmla v19.4s, v8.4s, v0.s[3]

        fmla v16.4s, v9.4s, v1.s[0]
        fmla v17.4s, v9.4s, v1.s[1]
        fmla v18.4s, v9.4s, v1.s[2]
        fmla v19.4s, v9.4s, v1.s[3]

        fmla v16.4s, v10.4s, v2.s[0]
        fmla v17.4s, v10.4s, v2.s[1]
        fmla v18.4s, v10.4s, v2.s[2]
        fmla v19.4s, v10.4s, v2.s[3]

        fmla v16.4s, v11.4s, v3.s[0]
        fmla v17.4s, v11.4s, v3.s[1]
        fmla v18.4s, v11.4s, v3.s[2]
        fmla v19.4s, v11.4s, v3.s[3]
        add x13, x13, #16
        subs x12, x12, #4
#endif
    bne E4LoopLR
    E4LoopLREnd:

    cbz x5, StoreLH4x4
    AddBiasLH4x4:
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
    mov x16, x23

    cmp x8, #2
    blt E1LH4

    E1LH8:
    E1LoopH8:
        mov x15, x1
        ld1 {v24.4s, v25.4s}, [x14], #32 // alpha
        mov w17, #0x0f
        dup v30.8b, w17
        mov w17, #8
        dup v31.8b, w17
        ld1 {v26.4s, v27.4s}, [x16], #32 // bias
        subs x12, x9, #2
        // ld1 {v3.4s, v4.4s}, [x13], #32
        ld1 {v0.4h}, [x13], #8
        ushr v1.8b, v0.8b, #4
        and v2.8b, v0.8b, v30.8b
        sub v1.8b, v1.8b, v31.8b
        sub v2.8b, v2.8b, v31.8b
        zip1 v9.8b, v1.8b, v2.8b
        zip2 v10.8b, v1.8b, v2.8b
        sxtl v11.8h, v9.8b
        sxtl v12.8h, v10.8b
        sxtl v8.4s, v11.4h
        sxtl2 v9.4s, v11.8h
        sxtl v10.4s, v12.4h
        sxtl2 v11.4s, v12.8h
        scvtf v12.4s, v8.4s
        scvtf v13.4s, v9.4s
        mov v8.16b, v26.16b
        mov v9.16b, v27.16b
        fmla v8.4s, v12.4s, v24.4s
        fmla v9.4s, v13.4s, v25.4s
        scvtf v12.4s, v10.4s
        scvtf v13.4s, v11.4s
        mov v10.16b, v26.16b
        mov v11.16b, v27.16b
        fmla v10.4s, v12.4s, v24.4s
        fmla v11.4s, v13.4s, v25.4s
        ld1 {v0.s}[0], [x15], x11
        ld1 {v0.s}[1], [x15], x11
        fmul v16.4s, v8.4s, v0.s[0]
        fmul v20.4s, v9.4s, v0.s[0]
        fmla v16.4s, v10.4s, v0.s[1]
        fmla v20.4s, v11.4s, v0.s[1]
        beq E1LoopLEnd

        E1LoopL:
            subs x12, x12, #2
            // ld1 {v3.4s, v4.4s}, [x13], #32
            ld1 {v0.4h}, [x13], #8
            ushr v1.8b, v0.8b, #4
            and v2.8b, v0.8b, v30.8b
            sub v1.8b, v1.8b, v31.8b
            sub v2.8b, v2.8b, v31.8b
            zip1 v9.8b, v1.8b, v2.8b
            zip2 v10.8b, v1.8b, v2.8b
            sxtl v11.8h, v9.8b
            sxtl v12.8h, v10.8b
            sxtl v8.4s, v11.4h
            sxtl2 v9.4s, v11.8h
            sxtl v10.4s, v12.4h
            sxtl2 v11.4s, v12.8h
            scvtf v12.4s, v8.4s
            scvtf v13.4s, v9.4s
            mov v8.16b, v26.16b
            mov v9.16b, v27.16b
            fmla v8.4s, v12.4s, v24.4s
            fmla v9.4s, v13.4s, v25.4s
            scvtf v12.4s, v10.4s
            scvtf v13.4s, v11.4s
            mov v10.16b, v26.16b
            mov v11.16b, v27.16b
            fmla v10.4s, v12.4s, v24.4s
            fmla v11.4s, v13.4s, v25.4s
            ld1 {v0.s}[0], [x15], x11
            ld1 {v0.s}[1], [x15], x11
            fmla v16.4s, v8.4s, v0.s[0]
            fmla v20.4s, v9.4s, v0.s[0]
            fmla v16.4s, v10.4s, v0.s[1]
            fmla v20.4s, v11.4s, v0.s[1]
            bne E1LoopL

        E1LoopLEnd:

        add x13, x13, x19
        sub x8, x8, #2
        cmp x8, #2

        cbz x5, StoreLH1x8
        AddBiasLH1x8:
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
    ld1 {v4.4s}, [x14], #16 // alpha
    mov w17, #0x0f
    dup v30.8b, w17
    mov w17, #8
    dup v31.8b, w17
    ld1 {v14.4s}, [x16], #16 // bias
    subs x12, x9, #4
    ld1 {v3.8h}, [x13]
    uzp1 v0.8h, v3.8h, v3.8h
    ushr v1.8b, v0.8b, #4
    and v2.8b, v0.8b, v30.8b
    sub v1.8b, v1.8b, v31.8b
    sub v2.8b, v2.8b, v31.8b
    zip1 v9.8b, v1.8b, v2.8b
    zip2 v10.8b, v1.8b, v2.8b
    sxtl v11.8h, v9.8b
    sxtl v12.8h, v10.8b
    sxtl v8.4s, v11.4h
    sxtl2 v9.4s, v11.8h
    sxtl v10.4s, v12.4h
    sxtl2 v11.4s, v12.8h
    scvtf v12.4s, v8.4s
    scvtf v13.4s, v9.4s
    mov v8.16b, v14.16b
    mov v9.16b, v14.16b
    fmla v8.4s, v12.4s, v4.4s
    fmla v9.4s, v13.4s, v4.4s
    scvtf v12.4s, v10.4s
    scvtf v13.4s, v11.4s
    mov v10.16b, v14.16b
    mov v11.16b, v14.16b
    fmla v10.4s, v12.4s, v4.4s
    fmla v11.4s, v13.4s, v4.4s

    ld1 {v0.s}[0], [x15], x11
    ld1 {v0.s}[1], [x15], x11
    ld1 {v0.s}[2], [x15], x11
    ld1 {v0.s}[3], [x15], x11
    fmul v16.4s, v8.4s, v0.s[0]
    fmla v16.4s, v9.4s, v0.s[1]
    fmla v16.4s, v10.4s, v0.s[2]
    fmla v16.4s, v11.4s, v0.s[3]
    add x13, x13, #16

    beq E1LoopLREnd

    E1LoopLR:
        // weight: load 16 x int4 to 16 x float
        ld1 {v3.8h}, [x13]
        uzp1 v0.8h, v3.8h, v3.8h
        ushr v1.8b, v0.8b, #4
        and v2.8b, v0.8b, v30.8b
        sub v1.8b, v1.8b, v31.8b
        sub v2.8b, v2.8b, v31.8b

        zip1 v9.8b, v1.8b, v2.8b
        zip2 v10.8b, v1.8b, v2.8b
        sxtl v11.8h, v9.8b
        sxtl v12.8h, v10.8b

        sxtl v8.4s, v11.4h
        sxtl2 v9.4s, v11.8h
        sxtl v10.4s, v12.4h
        sxtl2 v11.4s, v12.8h
        scvtf v12.4s, v8.4s
        scvtf v13.4s, v9.4s
        mov v8.16b, v15.16b
        mov v9.16b, v15.16b
        fmla v8.4s, v12.4s, v4.4s
        fmla v9.4s, v13.4s, v4.4s
        scvtf v12.4s, v10.4s
        scvtf v13.4s, v11.4s
        mov v10.16b, v15.16b
        mov v11.16b, v15.16b
        fmla v10.4s, v12.4s, v4.4s
        fmla v11.4s, v13.4s, v4.4s

        // input: load 4 x float
        ld1 {v0.s}[0], [x15], x11
        ld1 {v0.s}[1], [x15], x11
        ld1 {v0.s}[2], [x15], x11
        ld1 {v0.s}[3], [x15], x11
        // compute
        fmla v16.4s, v8.4s, v0.s[0]
        fmla v16.4s, v9.4s, v0.s[1]
        fmla v16.4s, v10.4s, v0.s[2]
        fmla v16.4s, v11.4s, v0.s[3]
        add x13, x13, #16
        subs x12, x12, #4
        bne E1LoopLR
    E1LoopLREnd:

    cbz x5, StoreLH1x4
    AddBiasLH1x4:
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
ldr x19, [sp, #0]
ldr x20, [sp, #8]
ldr x21, [sp, #16]
ldr x22, [sp, #24]
ldr x23, [sp, #32]
add sp, sp, #64

ret

#endif
