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

add x10, x10, #3
lsr x10, x10, #2
lsl x25, x9, #3 // l*hPack
add x25, x25, x19
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
        ld1 {v14.4s, v15.4s}, [x16], #32 // bias
        subs x12, x9, #2
        ld1 {v3.16b}, [x13], #16
        sxtl v0.8h, v3.8b
        sxtl2 v1.8h, v3.16b
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
            ld1 {v3.16b}, [x13], #16
            sxtl v0.8h, v3.8b
            sxtl2 v1.8h, v3.16b
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
        ld1 {v14.4s}, [x16], #16 // bias
        subs x12, x9, #4
        // load 16xint4 to 4xfloat
        ld1 {v3.4s}, [x13]
        uzp1 v0.4s, v3.4s, v3.4s
        sxtl v11.8h, v0.8b
        sxtl2 v12.8h, v0.16b
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

        add x13, x13, #32
        beq LoopLREnd

        LoopLR:
            ld1 {v3.4s}, [x13]
            uzp1 v0.4s, v3.4s, v3.4s
            sxtl v11.8h, v0.8b
            sxtl2 v12.8h, v0.16b
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

            add x13, x13, #32
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
        ld1 {v26.4s, v27.4s}, [x16], #32 // bias
        subs x12, x9, #2
        // ld1 {v3.4s, v4.4s}, [x13], #32
        ld1 {v0.16b}, [x13], #16
        sxtl v11.8h, v0.8b
        sxtl2 v12.8h, v0.16b
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
        // st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x0]
        // b End

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
            ld1 {v0.16b}, [x13], #16
            sxtl v11.8h, v0.8b
            sxtl2 v12.8h, v0.16b
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
    ld1 {v4.4s}, [x14], #16 // alpha
    ld1 {v14.4s}, [x16], #16 // bias
    subs x12, x9, #4
    // load 16xint4 to 16xfloat
    ld1 {v3.4s}, [x13]
    uzp1 v0.4s, v3.4s, v3.4s
    sxtl v11.8h, v0.8b
    sxtl2 v12.8h, v0.16b
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
    add x13, x13, #32
    beq E4LoopLREnd

    E4LoopLR:
        ld1 {v3.4s}, [x13]
        uzp1 v0.4s, v3.4s, v3.4s
        sxtl v11.8h, v0.8b
        sxtl2 v12.8h, v0.16b
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
        add x13, x13, #32
        subs x12, x12, #4
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

    E1H16:
    cmp x8, #4
    blt E1LH8
    E1LoopH16:
        mov x15, x1
        add x24, x13, x25
        ld1 {v24.4s, v25.4s, v26.4s, v27.4s}, [x14], #64 // Weight dequantize scale
        mov x12, x9
        ld1 {v28.4s, v29.4s, v30.4s, v31.4s}, [x20], #64 // Bias

        E1LoopH16L:
            ld1 {v0.s}[0], [x15], x11 // Input Float
            ld1 {v0.s}[1], [x15], x11
            ld1 {v1.16b}, [x13], #16  // Weight Int8
            ld1 {v2.16b}, [x24], #16
            ld1 {v20.4s, v21.4s, v22.4s, v23.4s}, [x16] // bias for dequantizing weight.
            ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x16]
            subs x12, x12, #2 // Compute 2 point along L.

            // Dequantize Int8 weight.
            sxtl v3.8h, v1.8b
            sxtl2 v4.8h, v1.16b
            sxtl v5.8h, v2.8b
            sxtl2 v15.8h, v2.16b

            sxtl2 v8.4s, v3.8h
            sxtl v9.4s, v4.4h
            sxtl2 v10.4s, v4.8h
            sxtl v11.4s, v5.4h
            sxtl2 v12.4s, v5.8h
            sxtl v13.4s, v15.4h
            sxtl2 v14.4s, v15.8h
            sxtl v1.4s, v3.4h

            
            scvtf v8.4s, v8.4s
            scvtf v9.4s, v9.4s
            scvtf v10.4s, v10.4s
            scvtf v11.4s, v11.4s
            scvtf v12.4s, v12.4s
            scvtf v13.4s, v13.4s
            scvtf v14.4s, v14.4s
            scvtf v1.4s, v1.4s

            fmla v17.4s, v8.4s, v25.4s
            fmla v20.4s, v9.4s, v24.4s
            fmla v21.4s, v10.4s, v25.4s
            fmla v18.4s, v11.4s, v26.4s
            fmla v19.4s, v12.4s, v27.4s
            fmla v22.4s, v13.4s, v26.4s
            fmla v23.4s, v14.4s, v27.4s
            fmla v16.4s, v1.4s, v24.4s

            fmla v28.4s, v16.4s, v0.s[0]
            fmla v29.4s, v17.4s, v0.s[0]
            fmla v28.4s, v20.4s, v0.s[1]
            fmla v29.4s, v21.4s, v0.s[1]
            fmla v30.4s, v18.4s, v0.s[0]
            fmla v31.4s, v19.4s, v0.s[0]
            fmla v30.4s, v22.4s, v0.s[1]
            fmla v31.4s, v23.4s, v0.s[1]

            bne E1LoopH16L
        E1LoopH16LEnd:
            sub x8, x8, #4
            add x13, x24, x19
            add x16, x16, #64

            cbz x5, StoreE1H16

            PostTreatE1H16:
            fmax v28.4s, v28.4s, v6.4s
            fmax v29.4s, v29.4s, v6.4s
            fmin v30.4s, v30.4s, v7.4s
            fmin v31.4s, v31.4s, v7.4s
            
            StoreE1H16:
            cmp x8, #4
            st1 {v28.4s}, [x0], x7
            st1 {v29.4s}, [x0], x7
            st1 {v30.4s}, [x0], x7
            st1 {v31.4s}, [x0], x7
            bge E1LoopH16

    E1LH8:
    cmp x8, #2
    blt E1LH4
    E1LoopH8:
        mov x15, x1
        ld1 {v24.4s, v25.4s}, [x14], #32 // alpha
        mov x12, x9
        ld1 {v27.4s, v28.4s}, [x20], #32 // Bias

        E1LoopH8L:
            ld1 {v0.16b}, [x13], #16
            ld1 {v29.s}[0], [x15], x11 // Input Float
            ld1 {v29.s}[1], [x15], x11
            ld1 {v8.4s, v9.4s},[x16] // bias for dequantizing weight.
            ld1 {v10.4s, v11.4s}, [x16]

            subs x12, x12, #2
            // Weight Int8->Int32.
            sxtl v30.8h, v0.8b
            sxtl2 v31.8h, v0.16b
            sxtl v1.4s, v30.4h
            sxtl2 v2.4s, v30.8h
            sxtl v3.4s, v31.4h
            sxtl2 v4.4s, v31.8h
            scvtf v1.4s, v1.4s
            scvtf v2.4s, v2.4s
            scvtf v3.4s, v3.4s
            scvtf v4.4s, v4.4s

            // Dequantize Weight.
            fmla v8.4s, v1.4s, v24.4s
            fmla v9.4s, v2.4s, v25.4s
            fmla v10.4s, v3.4s, v24.4s
            fmla v11.4s, v4.4s, v25.4s

            fmla v27.4s, v8.4s, v29.s[0]
            fmla v28.4s, v9.4s, v29.s[0]
            fmla v27.4s, v10.4s, v29.s[1]
            fmla v28.4s, v11.4s, v29.s[1]

            bne E1LoopH8L

        E1LoopH8LEnd:

        add x13, x13, x19
        add x16, x16, #32
        sub x8, x8, #2
        cmp x8, #2

        cbz x5, StoreLH1x8

        PostTreatLH1x8:
        fmax v27.4s, v27.4s, v6.4s
        fmax v28.4s, v28.4s, v6.4s
        fmin v27.4s, v27.4s, v7.4s
        fmin v28.4s, v28.4s, v7.4s

        StoreLH1x8:

        st1 {v27.4s}, [x0], x7
        st1 {v28.4s}, [x0], x7

        bge E1LoopH8

    E1LH4:
    cbz x8, E1End
    mov x15, x1
    ld1 {v4.4s}, [x14], #16 // alpha
    ld1 {v14.4s}, [x16], #16 // bias
    subs x12, x9, #4
    ld1 {v3.4s}, [x13]
    uzp1 v0.4s, v3.4s, v3.4s
    sxtl v11.8h, v0.8b
    sxtl2 v12.8h, v0.16b
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
    add x13, x13, #32

    beq E1LoopLREnd

    E1LoopLR:
        // weight: load 16 x int4 to 16 x float
        ld1 {v3.4s}, [x13]
        uzp1 v0.4s, v3.4s, v3.4s
        sxtl v11.8h, v0.8b
        sxtl2 v12.8h, v0.16b
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
        add x13, x13, #32
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
