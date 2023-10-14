//
//  MNNPackedMatMulRemainFP16_int4.S
//  MNN
//
//  Created by MNN on 2023/05/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5
// 8 * 24 MatMul, C(UP_DIV(h,8), e, h8) = B(UP_DIV(h,hP), l, hP) * A(l, e), hP = 24
// Remain meaning is eSize is any value
asm_function MNNPackedMatMulRemainFP16_int4
//void MNNPackedMatMulRemainFP16_int4(FLOAT16* C, const FLOAT16* A, const FLOAT16* B, size_t eSize, const size_t* parameter, const FLOAT16* postParameters, const FLOAT16* bias);
//Auto x0: C, x1:A, x2:B, x3:eSize, x4:parameter, x5:postParameters, x6:bias, x7: k, x8: b
// parameter: {aStride, l, h, cStride, bExtraStride}
ldr x8, [sp] // bias
stp d14, d15, [sp, #-112]!
stp d12, d13, [sp, #16]
stp d10, d11, [sp, #32]
stp d8,  d9,  [sp, #48]
stp x19, x20, [sp, #64]
stp x21, x22, [sp, #80]
stp x23, x24, [sp, #96]

mov x22, x7 // alpha
mov x23, x8 // bias
ldr x11, [x4, #0] // aStride
ldr x9, [x4, #8] // l
ldr x10, [x4, #16] // h

ldr x7, [x4, #24] // cStride
ldr x19, [x4, #40] // bExtraStride

add x10, x10, #7
lsr x10, x10, #3

cbz x5, Start
ld1 {v5.4s}, [x5]
fcvtn v5.4h, v5.4s
dup v6.8h, v5.h[2] // Min Value
dup v7.8h, v5.h[3] // Max Value

Start:

E8:
cmp x3, #8
blt E4

// 8x16
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
    sub x24, x7, #64
    LoopH8x8:
        mov x15, x1
        ld1 {v12.8h, v13.8h}, [x14], #32 // alpha
        mov w17, #0x0f
        dup v3.16b, w17
        mov w17, #8
        dup v4.16b, w17
        ld1 {v14.8h, v15.8h}, [x16], #32 // bias
        subs x12, x9, #2
        ld1 {v0.8h}, [x13], #16
        ushr v1.16b, v0.16b, #4
        and v2.16b, v0.16b, v3.16b
        sub v1.16b, v1.16b, v4.16b
        sub v2.16b, v2.16b, v4.16b
        zip1 v10.16b, v1.16b, v2.16b
        zip2 v11.16b, v1.16b, v2.16b
        sxtl v1.8h, v10.8b
        sxtl2 v2.8h, v10.16b
        scvtf v1.8h, v1.8h
        scvtf v2.8h, v2.8h
        mov v8.8h, v14.8h
        mov v9.8h, v15.8h
        fmla v8.8h, v1.8h, v12.8h
        fmla v9.8h, v2.8h, v13.8h

        ld1 {v0.8h}, [x15], x11
        fmul v16.8h, v8.8h, v0.h[0]
        fmul v17.8h, v8.8h, v0.h[1]
        fmul v18.8h, v8.8h, v0.h[2]
        fmul v19.8h, v8.8h, v0.h[3]

        fmul v20.8h, v9.8h, v0.h[0]
        fmul v21.8h, v9.8h, v0.h[1]
        fmul v22.8h, v9.8h, v0.h[2]
        fmul v23.8h, v9.8h, v0.h[3]

        fmul v24.8h, v8.8h, v0.h[4]
        fmul v25.8h, v8.8h, v0.h[5]
        fmul v26.8h, v8.8h, v0.h[6]
        fmul v27.8h, v8.8h, v0.h[7]

        fmul v28.8h, v9.8h, v0.h[4]
        fmul v29.8h, v9.8h, v0.h[5]
        fmul v30.8h, v9.8h, v0.h[6]
        fmul v31.8h, v9.8h, v0.h[7]

        sxtl v1.8h, v11.8b
        sxtl2 v2.8h, v11.16b
        scvtf v1.8h, v1.8h
        scvtf v2.8h, v2.8h
        mov v8.8h, v14.8h
        mov v9.8h, v15.8h
        fmla v8.8h, v1.8h, v12.8h
        fmla v9.8h, v2.8h, v13.8h

        ld1 {v0.8h}, [x15], x11
        fmla v16.8h, v8.8h, v0.h[0]
        fmla v17.8h, v8.8h, v0.h[1]
        fmla v18.8h, v8.8h, v0.h[2]
        fmla v19.8h, v8.8h, v0.h[3]

        fmla v20.8h, v9.8h, v0.h[0]
        fmla v21.8h, v9.8h, v0.h[1]
        fmla v22.8h, v9.8h, v0.h[2]
        fmla v23.8h, v9.8h, v0.h[3]

        fmla v24.8h, v8.8h, v0.h[4]
        fmla v25.8h, v8.8h, v0.h[5]
        fmla v26.8h, v8.8h, v0.h[6]
        fmla v27.8h, v8.8h, v0.h[7]

        fmla v28.8h, v9.8h, v0.h[4]
        fmla v29.8h, v9.8h, v0.h[5]
        fmla v30.8h, v9.8h, v0.h[6]
        fmla v31.8h, v9.8h, v0.h[7]
        beq LoopLEnd

        LoopL:
            subs x12, x12, #2
            ld1 {v0.8h}, [x13], #16
            ushr v1.16b, v0.16b, #4
            and v2.16b, v0.16b, v3.16b
            sub v1.16b, v1.16b, v4.16b
            sub v2.16b, v2.16b, v4.16b
            zip1 v10.16b, v1.16b, v2.16b
            zip2 v11.16b, v1.16b, v2.16b
            sxtl v1.8h, v10.8b
            sxtl2 v2.8h, v10.16b
            scvtf v1.8h, v1.8h
            scvtf v2.8h, v2.8h
            mov v8.8h, v14.8h
            mov v9.8h, v15.8h
            fmla v8.8h, v1.8h, v12.8h
            fmla v9.8h, v2.8h, v13.8h

            ld1 {v0.8h}, [x15], x11
            fmla v16.8h, v8.8h, v0.h[0]
            fmla v17.8h, v8.8h, v0.h[1]
            fmla v18.8h, v8.8h, v0.h[2]
            fmla v19.8h, v8.8h, v0.h[3]

            fmla v20.8h, v9.8h, v0.h[0]
            fmla v21.8h, v9.8h, v0.h[1]
            fmla v22.8h, v9.8h, v0.h[2]
            fmla v23.8h, v9.8h, v0.h[3]

            fmla v24.8h, v8.8h, v0.h[4]
            fmla v25.8h, v8.8h, v0.h[5]
            fmla v26.8h, v8.8h, v0.h[6]
            fmla v27.8h, v8.8h, v0.h[7]

            fmla v28.8h, v9.8h, v0.h[4]
            fmla v29.8h, v9.8h, v0.h[5]
            fmla v30.8h, v9.8h, v0.h[6]
            fmla v31.8h, v9.8h, v0.h[7]

            sxtl v1.8h, v11.8b
            sxtl2 v2.8h, v11.16b
            scvtf v1.8h, v1.8h
            scvtf v2.8h, v2.8h
            mov v8.8h, v14.8h
            mov v9.8h, v15.8h
            fmla v8.8h, v1.8h, v12.8h
            fmla v9.8h, v2.8h, v13.8h
            ld1 {v0.8h}, [x15], x11
            fmla v16.8h, v8.8h, v0.h[0]
            fmla v17.8h, v8.8h, v0.h[1]
            fmla v18.8h, v8.8h, v0.h[2]
            fmla v19.8h, v8.8h, v0.h[3]

            fmla v20.8h, v9.8h, v0.h[0]
            fmla v21.8h, v9.8h, v0.h[1]
            fmla v22.8h, v9.8h, v0.h[2]
            fmla v23.8h, v9.8h, v0.h[3]

            fmla v24.8h, v8.8h, v0.h[4]
            fmla v25.8h, v8.8h, v0.h[5]
            fmla v26.8h, v8.8h, v0.h[6]
            fmla v27.8h, v8.8h, v0.h[7]

            fmla v28.8h, v9.8h, v0.h[4]
            fmla v29.8h, v9.8h, v0.h[5]
            fmla v30.8h, v9.8h, v0.h[6]
            fmla v31.8h, v9.8h, v0.h[7]
            bne LoopL

        LoopLEnd:

        add x13, x13, x19
        sub x8, x8, #2

        cbz x5, StoreLH8
        AddBiasLH8:
        ld1 {v0.8h, v1.8h}, [x20], #32

        fmla v16.8h, v0.8h, v5.h[1]
        fmla v17.8h, v0.8h, v5.h[1]
        fmla v18.8h, v0.8h, v5.h[1]
        fmla v19.8h, v0.8h, v5.h[1]

        fmla v20.8h, v1.8h, v5.h[1]
        fmla v21.8h, v1.8h, v5.h[1]
        fmla v22.8h, v1.8h, v5.h[1]
        fmla v23.8h, v1.8h, v5.h[1]

        fmla v24.8h, v0.8h, v5.h[1]
        fmla v25.8h, v0.8h, v5.h[1]
        fmla v26.8h, v0.8h, v5.h[1]
        fmla v27.8h, v0.8h, v5.h[1]

        fmla v28.8h, v1.8h, v5.h[1]
        fmla v29.8h, v1.8h, v5.h[1]
        fmla v30.8h, v1.8h, v5.h[1]
        fmla v31.8h, v1.8h, v5.h[1]

        PostTreatLH8:
        fmax v16.8h, v16.8h, v6.8h
        fmax v17.8h, v17.8h, v6.8h
        fmax v18.8h, v18.8h, v6.8h
        fmax v19.8h, v19.8h, v6.8h
        fmax v20.8h, v20.8h, v6.8h
        fmax v21.8h, v21.8h, v6.8h
        fmax v22.8h, v22.8h, v6.8h
        fmax v23.8h, v23.8h, v6.8h
        fmax v24.8h, v24.8h, v6.8h
        fmax v25.8h, v25.8h, v6.8h
        fmax v26.8h, v26.8h, v6.8h
        fmax v27.8h, v27.8h, v6.8h
        fmax v28.8h, v28.8h, v6.8h
        fmax v29.8h, v29.8h, v6.8h
        fmax v30.8h, v30.8h, v6.8h
        fmax v31.8h, v31.8h, v6.8h

        fmin v16.8h, v16.8h, v7.8h
        fmin v17.8h, v17.8h, v7.8h
        fmin v18.8h, v18.8h, v7.8h
        fmin v19.8h, v19.8h, v7.8h
        fmin v20.8h, v20.8h, v7.8h
        fmin v21.8h, v21.8h, v7.8h
        fmin v22.8h, v22.8h, v7.8h
        fmin v23.8h, v23.8h, v7.8h
        fmin v24.8h, v24.8h, v7.8h
        fmin v25.8h, v25.8h, v7.8h
        fmin v26.8h, v26.8h, v7.8h
        fmin v27.8h, v27.8h, v7.8h
        fmin v28.8h, v28.8h, v7.8h
        fmin v29.8h, v29.8h, v7.8h
        fmin v30.8h, v30.8h, v7.8h
        fmin v31.8h, v31.8h, v7.8h

        StoreLH8:
        st1 {v16.8h, v17.8h, v18.8h, v19.8h}, [x0], #64
        st1 {v24.8h, v25.8h, v26.8h, v27.8h}, [x0], x24

        st1 {v20.8h, v21.8h, v22.8h, v23.8h}, [x0], #64
        st1 {v28.8h, v29.8h, v30.8h, v31.8h}, [x0], x24
        cmp x8, #2
        bge LoopH8x8

    LH4:
    cbz x8, E8End
    LoopHRemain:
        mov x15, x1
        subs x12, x9, #1
        ld1 {v3.8h}, [x13]
        ld1 {v0.8h}, [x15], x11
        fmul v16.8h, v3.8h, v0.h[0]
        fmul v17.8h, v3.8h, v0.h[1]
        add x13, x13, #32
        fmul v18.8h, v3.8h, v0.h[2]
        fmul v19.8h, v3.8h, v0.h[3]
        fmul v20.8h, v3.8h, v0.h[4]
        fmul v21.8h, v3.8h, v0.h[5]
        fmul v22.8h, v3.8h, v0.h[6]
        fmul v23.8h, v3.8h, v0.h[7]
        beq LoopLREnd

        LoopLR:
            ld1 {v3.8h}, [x13]
            ld1 {v0.8h}, [x15], x11
            fmla v16.8h, v3.8h, v0.h[0]
            fmla v17.8h, v3.8h, v0.h[1]
            fmla v18.8h, v3.8h, v0.h[2]
            fmla v19.8h, v3.8h, v0.h[3]
            add x13, x13, #32

            fmla v20.8h, v3.8h, v0.h[4]
            fmla v21.8h, v3.8h, v0.h[5]
            fmla v22.8h, v3.8h, v0.h[6]
            fmla v23.8h, v3.8h, v0.h[7]

            subs x12, x12, #1
            bne LoopLR
        LoopLREnd:

        cbz x5, StoreLH8x4
        AddBiasLH8x4:
        ld1 {v0.8h}, [x20]

        fmla v16.8h, v0.8h, v5.h[1]
        fmla v17.8h, v0.8h, v5.h[1]
        fmla v18.8h, v0.8h, v5.h[1]
        fmla v19.8h, v0.8h, v5.h[1]

        fmla v20.8h, v0.8h, v5.h[1]
        fmla v21.8h, v0.8h, v5.h[1]
        fmla v22.8h, v0.8h, v5.h[1]
        fmla v23.8h, v0.8h, v5.h[1]
        
        PostTreatLH8x4:
        fmax v16.8h, v16.8h, v6.8h
        fmax v17.8h, v17.8h, v6.8h
        fmax v18.8h, v18.8h, v6.8h
        fmax v19.8h, v19.8h, v6.8h
        fmax v20.8h, v20.8h, v6.8h
        fmax v21.8h, v21.8h, v6.8h
        fmax v22.8h, v22.8h, v6.8h
        fmax v23.8h, v23.8h, v6.8h

        fmin v16.8h, v16.8h, v7.8h
        fmin v17.8h, v17.8h, v7.8h
        fmin v18.8h, v18.8h, v7.8h
        fmin v19.8h, v19.8h, v7.8h
        fmin v20.8h, v20.8h, v7.8h
        fmin v21.8h, v21.8h, v7.8h
        fmin v22.8h, v22.8h, v7.8h
        fmin v23.8h, v23.8h, v7.8h

        StoreLH8x4:

        st1 {v16.8h, v17.8h, v18.8h, v19.8h}, [x0], #64
        st1 {v20.8h, v21.8h, v22.8h, v23.8h}, [x0], #64

    E8End:

    sub x3, x3, #8
    add x0, x21, #128
    add x1, x1, #16

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
        ld1 {v24.8h, v25.8h}, [x14], #32 // alpha
        mov w17, #0x0f
        dup v30.16b, w17
        mov w17, #8
        dup v31.16b, w17
        ld1 {v26.8h, v27.8h}, [x16], #32 // bias
        subs x12, x9, #2
        ld1 {v0.8h}, [x13], #16
        ushr v1.16b, v0.16b, #4
        and v2.16b, v0.16b, v30.16b
        sub v1.16b, v1.16b, v31.16b
        sub v2.16b, v2.16b, v31.16b
        zip1 v3.16b, v1.16b, v2.16b
        zip2 v4.16b, v1.16b, v2.16b
        sxtl v0.8h, v3.8b
        sxtl2 v1.8h, v3.16b
        sxtl v2.8h, v4.8b
        sxtl2 v3.8h, v4.16b
        scvtf v12.8h, v0.8h
        scvtf v13.8h, v1.8h
        scvtf v14.8h, v2.8h
        scvtf v15.8h, v3.8h
        mov v8.8h, v26.8h
        mov v9.8h, v27.8h
        fmla v8.8h, v12.8h, v24.8h
        fmla v9.8h, v13.8h, v25.8h
        mov v10.8h, v26.8h
        mov v11.8h, v27.8h
        fmla v10.8h, v14.8h, v24.8h
        fmla v11.8h, v15.8h, v25.8h

        ld1 {v0.4h}, [x15], x11
        ld1 {v1.4h}, [x15], x11
        fmul v16.8h, v8.8h, v0.h[0]
        fmul v17.8h, v8.8h, v0.h[1]
        fmul v18.8h, v8.8h, v0.h[2]
        fmul v19.8h, v8.8h, v0.h[3]

        fmul v20.8h, v9.8h, v0.h[0]
        fmul v21.8h, v9.8h, v0.h[1]
        fmul v22.8h, v9.8h, v0.h[2]
        fmul v23.8h, v9.8h, v0.h[3]

        fmla v16.8h, v10.8h, v1.h[0]
        fmla v17.8h, v10.8h, v1.h[1]
        fmla v18.8h, v10.8h, v1.h[2]
        fmla v19.8h, v10.8h, v1.h[3]

        fmla v20.8h, v11.8h, v1.h[0]
        fmla v21.8h, v11.8h, v1.h[1]
        fmla v22.8h, v11.8h, v1.h[2]
        fmla v23.8h, v11.8h, v1.h[3]

        beq E4LoopLEnd

        E4LoopL:
#if 0
            subs x12, x12, #1
            ld1 {v0.4h}, [x13], #8
            ushr v1.8b, v0.8b, #4
            and v2.8b, v0.8b, v30.8b
            sub v1.8b, v1.8b, v31.8b
            sub v2.8b, v2.8b, v31.8b
            zip1 v8.8b, v1.8b, v2.8b
            zip2 v9.8b, v1.8b, v2.8b
            sxtl v10.8h, v8.8b
            sxtl v11.8h, v9.8b
            scvtf v12.8h, v10.8h
            scvtf v13.8h, v11.8h
            mov v8.8h, v26.8h
            mov v9.8h, v27.8h
            fmla v8.8h, v12.8h, v24.8h
            fmla v9.8h, v13.8h, v25.8h
            ld1 {v0.4h}, [x15], x11
            fmla v16.8h, v8.8h, v0.h[0]
            fmla v17.8h, v8.8h, v0.h[1]
            fmla v18.8h, v8.8h, v0.h[2]
            fmla v19.8h, v8.8h, v0.h[3]

            fmla v20.8h, v9.8h, v0.h[0]
            fmla v21.8h, v9.8h, v0.h[1]
            fmla v22.8h, v9.8h, v0.h[2]
            fmla v23.8h, v9.8h, v0.h[3]
#else
            subs x12, x12, #2
            ld1 {v0.8h}, [x13], #16
            ushr v1.16b, v0.16b, #4
            and v2.16b, v0.16b, v30.16b
            sub v1.16b, v1.16b, v31.16b
            sub v2.16b, v2.16b, v31.16b
            zip1 v3.16b, v1.16b, v2.16b
            zip2 v4.16b, v1.16b, v2.16b
            sxtl v0.8h, v3.8b
            sxtl2 v1.8h, v3.16b
            sxtl v2.8h, v4.8b
            sxtl2 v3.8h, v4.16b
            scvtf v12.8h, v0.8h
            scvtf v13.8h, v1.8h
            scvtf v14.8h, v2.8h
            scvtf v15.8h, v3.8h
            mov v8.8h, v26.8h
            mov v9.8h, v27.8h
            fmla v8.8h, v12.8h, v24.8h
            fmla v9.8h, v13.8h, v25.8h
            mov v10.8h, v26.8h
            mov v11.8h, v27.8h
            fmla v10.8h, v14.8h, v24.8h
            fmla v11.8h, v15.8h, v25.8h

            ld1 {v0.4h}, [x15], x11
            ld1 {v1.4h}, [x15], x11
            fmla v16.8h, v8.8h, v0.h[0]
            fmla v17.8h, v8.8h, v0.h[1]
            fmla v18.8h, v8.8h, v0.h[2]
            fmla v19.8h, v8.8h, v0.h[3]

            fmla v20.8h, v9.8h, v0.h[0]
            fmla v21.8h, v9.8h, v0.h[1]
            fmla v22.8h, v9.8h, v0.h[2]
            fmla v23.8h, v9.8h, v0.h[3]

            fmla v16.8h, v10.8h, v1.h[0]
            fmla v17.8h, v10.8h, v1.h[1]
            fmla v18.8h, v10.8h, v1.h[2]
            fmla v19.8h, v10.8h, v1.h[3]

            fmla v20.8h, v11.8h, v1.h[0]
            fmla v21.8h, v11.8h, v1.h[1]
            fmla v22.8h, v11.8h, v1.h[2]
            fmla v23.8h, v11.8h, v1.h[3]
#endif
            bne E4LoopL

        E4LoopLEnd:
        add x13, x13, x19
        sub x8, x8, #2
        cmp x8, #2

        cbz x5, StoreLH4x8

        AddBiasLH4x8:
        ld1 {v0.8h, v1.8h}, [x20], #32

        fmla v16.8h, v0.8h, v5.h[1]
        fmla v17.8h, v0.8h, v5.h[1]
        fmla v18.8h, v0.8h, v5.h[1]
        fmla v19.8h, v0.8h, v5.h[1]

        fmla v20.8h, v1.8h, v5.h[1]
        fmla v21.8h, v1.8h, v5.h[1]
        fmla v22.8h, v1.8h, v5.h[1]
        fmla v23.8h, v1.8h, v5.h[1]
        
        PostTreatLH4x8:
        fmax v16.8h, v16.8h, v6.8h
        fmax v17.8h, v17.8h, v6.8h
        fmax v18.8h, v18.8h, v6.8h
        fmax v19.8h, v19.8h, v6.8h
        fmax v20.8h, v20.8h, v6.8h
        fmax v21.8h, v21.8h, v6.8h
        fmax v22.8h, v22.8h, v6.8h
        fmax v23.8h, v23.8h, v6.8h

        fmin v16.8h, v16.8h, v7.8h
        fmin v17.8h, v17.8h, v7.8h
        fmin v18.8h, v18.8h, v7.8h
        fmin v19.8h, v19.8h, v7.8h
        fmin v20.8h, v20.8h, v7.8h
        fmin v21.8h, v21.8h, v7.8h
        fmin v22.8h, v22.8h, v7.8h
        fmin v23.8h, v23.8h, v7.8h

        StoreLH4x8:

        st1 {v16.8h, v17.8h, v18.8h, v19.8h}, [x0], x7
        st1 {v20.8h, v21.8h, v22.8h, v23.8h}, [x0], x7

        bge E4LoopH8

    E4LH4:
    cbz x8, E4End
    mov x15, x1
    ld1 {v4.8h}, [x14], #16 // alpha
    // mov v4.d[1], v4.d[0]
    mov w17, #0x0f
    dup v30.8b, w17
    mov w17, #8
    dup v31.8b, w17
    ld1 {v14.8h}, [x16], #16 // bias
    // mov v14.d[1], v14.d[0]
    subs x12, x9, #2
    // load 16xint4 to 16xfloat
    ld1 {v3.8h}, [x13]
    // 01234567xxxxxxx89... => 0123456789...
    uzp1 v0.4s, v3.4s, v3.4s
    ushr v1.8b, v0.8b, #4
    and v2.8b, v0.8b, v30.8b
    sub v1.8b, v1.8b, v31.8b
    sub v2.8b, v2.8b, v31.8b
    zip1 v8.8b, v1.8b, v2.8b
    zip2 v9.8b, v1.8b, v2.8b
    sxtl v10.8h, v8.8b
    sxtl v11.8h, v9.8b
    scvtf v12.8h, v10.8h
    scvtf v13.8h, v11.8h
    mov v8.8h, v14.8h
    mov v9.8h, v14.8h
    fmla v8.8h, v12.8h, v4.8h
    fmla v9.8h, v13.8h, v4.8h
    // st1 {v8.8h, v9.8h}, [x0]
    // b End

    ld1 {v0.4h}, [x15], x11
    fmul v16.8h, v8.8h, v0.h[0]
    fmul v17.8h, v8.8h, v0.h[1]
    fmul v18.8h, v8.8h, v0.h[2]
    fmul v19.8h, v8.8h, v0.h[3]
    ld1 {v1.4h}, [x15], x11
    fmla v16.8h, v9.8h, v1.h[0]
    fmla v17.8h, v9.8h, v1.h[1]
    fmla v18.8h, v9.8h, v1.h[2]
    fmla v19.8h, v9.8h, v1.h[3]
    add x13, x13, #16

    beq E4LoopLREnd

    E4LoopLR:
        ld1 {v3.8h}, [x13]
        uzp1 v0.4s, v3.4s, v3.4s
        ushr v1.8b, v0.8b, #4
        and v2.8b, v0.8b, v30.8b
        sub v1.8b, v1.8b, v31.8b
        sub v2.8b, v2.8b, v31.8b
        zip1 v8.8b, v1.8b, v2.8b
        zip2 v9.8b, v1.8b, v2.8b
        sxtl v10.8h, v8.8b
        sxtl v11.8h, v9.8b
        scvtf v12.8h, v10.8h
        scvtf v13.8h, v11.8h
        mov v8.8h, v14.8h
        mov v9.8h, v14.8h
        fmla v8.8h, v12.8h, v4.8h
        fmla v9.8h, v13.8h, v4.8h

        ld1 {v0.4h}, [x15], x11
        fmla v16.8h, v8.8h, v0.h[0]
        fmla v17.8h, v8.8h, v0.h[1]
        fmla v18.8h, v8.8h, v0.h[2]
        fmla v19.8h, v8.8h, v0.h[3]
        ld1 {v1.4h}, [x15], x11
        fmla v16.8h, v9.8h, v1.h[0]
        fmla v17.8h, v9.8h, v1.h[1]
        fmla v18.8h, v9.8h, v1.h[2]
        fmla v19.8h, v9.8h, v1.h[3]
        add x13, x13, #16

        subs x12, x12, #2
        bne E4LoopLR
    E4LoopLREnd:

    cbz x5, StoreLH4x4
    AddBiasLH4x4:
    ld1 {v0.8h}, [x20]

    fmla v16.8h, v0.8h, v5.h[1]
    fmla v17.8h, v0.8h, v5.h[1]
    fmla v18.8h, v0.8h, v5.h[1]
    fmla v19.8h, v0.8h, v5.h[1]

    
    PostTreatLH4x4:
    fmax v16.8h, v16.8h, v6.8h
    fmax v17.8h, v17.8h, v6.8h
    fmax v18.8h, v18.8h, v6.8h
    fmax v19.8h, v19.8h, v6.8h

    fmin v16.8h, v16.8h, v7.8h
    fmin v17.8h, v17.8h, v7.8h
    fmin v18.8h, v18.8h, v7.8h
    fmin v19.8h, v19.8h, v7.8h

    StoreLH4x4:
    st1 {v16.8h, v17.8h, v18.8h, v19.8h}, [x0]

    E4End:

    sub x3, x3, #4
    add x0, x21, #64
    add x1, x1, #8

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
        ld1 {v24.8h, v25.8h}, [x14], #32 // alpha
        mov w17, #0x0f
        dup v30.16b, w17
        mov w17, #8
        dup v31.16b, w17
        ld1 {v26.8h, v27.8h}, [x16], #32 // bias
        subs x12, x9, #2
        ld1 {v0.8h}, [x13], #16
        ushr v1.16b, v0.16b, #4
        and v2.16b, v0.16b, v30.16b
        sub v1.16b, v1.16b, v31.16b
        sub v2.16b, v2.16b, v31.16b
        zip1 v3.16b, v1.16b, v2.16b
        zip2 v4.16b, v1.16b, v2.16b
        sxtl v0.8h, v3.8b
        sxtl2 v1.8h, v3.16b
        sxtl v2.8h, v4.8b
        sxtl2 v3.8h, v4.16b
        scvtf v12.8h, v0.8h
        scvtf v13.8h, v1.8h
        scvtf v14.8h, v2.8h
        scvtf v15.8h, v3.8h
        mov v8.8h, v26.8h
        mov v9.8h, v27.8h
        fmla v8.8h, v12.8h, v24.8h
        fmla v9.8h, v13.8h, v25.8h
        mov v10.8h, v26.8h
        mov v11.8h, v27.8h
        fmla v10.8h, v14.8h, v24.8h
        fmla v11.8h, v15.8h, v25.8h

        ld1 {v0.h}[0], [x15], x11
        ld1 {v0.h}[1], [x15], x11
        fmul v16.8h, v8.8h, v0.h[0]
        fmul v20.8h, v9.8h, v0.h[0]

        fmla v16.8h, v10.8h, v0.h[1]
        fmla v20.8h, v11.8h, v0.h[1]

        beq E1LoopLEnd

        E1LoopL:
            subs x12, x12, #2
            ld1 {v0.16b}, [x13], #16
            ushr v1.16b, v0.16b, #4
            and v2.16b, v0.16b, v30.16b
            sub v1.16b, v1.16b, v31.16b
            sub v2.16b, v2.16b, v31.16b
            zip1 v3.16b, v1.16b, v2.16b
            zip2 v4.16b, v1.16b, v2.16b
            sxtl v0.8h, v3.8b
            sxtl2 v1.8h, v3.16b
            sxtl v2.8h, v4.8b
            sxtl2 v3.8h, v4.16b
            mov v8.8h, v26.8h
            mov v9.8h, v27.8h
            scvtf v12.8h, v0.8h
            scvtf v13.8h, v1.8h
            mov v10.8h, v26.8h
            mov v11.8h, v27.8h
            scvtf v14.8h, v2.8h
            ld1 {v0.h}[0], [x15], x11
            scvtf v15.8h, v3.8h
            fmla v8.8h, v12.8h, v24.8h
            fmla v9.8h, v13.8h, v25.8h
            ld1 {v0.h}[1], [x15], x11
            fmla v10.8h, v14.8h, v24.8h
            fmla v11.8h, v15.8h, v25.8h

            fmla v16.8h, v8.8h, v0.h[0]
            fmla v20.8h, v9.8h, v0.h[0]
            fmla v16.8h, v10.8h, v0.h[1]
            fmla v20.8h, v11.8h, v0.h[1]
            bne E1LoopL

        E1LoopLEnd:

        add x13, x13, x19
        sub x8, x8, #2
        cmp x8, #2

        cbz x5, StoreLH1x8
        AddBiasLH1x8:
        ld1 {v0.8h, v1.8h}, [x20], #32

        fmla v16.8h, v0.8h, v5.h[1]
        fmla v20.8h, v1.8h, v5.h[1]
        
        PostTreatLH1x8:
        fmax v16.8h, v16.8h, v6.8h
        fmax v20.8h, v20.8h, v6.8h
        fmin v16.8h, v16.8h, v7.8h
        fmin v20.8h, v20.8h, v7.8h

        StoreLH1x8:

        st1 {v16.8h}, [x0], x7
        st1 {v20.8h}, [x0], x7

        bge E1LoopH8

    E1LH4:
    cbz x8, E1End
    mov x15, x1
    subs x12, x9, #1
    ld1 {v3.8h}, [x13]
    ld1 {v0.h}[0], [x15], x11
    fmul v16.8h, v3.8h, v0.h[0]
    add x13, x13, #32

    beq E1LoopLREnd

    E1LoopLR:
        ld1 {v3.8h}, [x13]
        ld1 {v0.h}[0], [x15], x11
        fmla v16.8h, v3.8h, v0.h[0]
        add x13, x13, #32

        subs x12, x12, #1
        bne E1LoopLR
    E1LoopLREnd:

    cbz x5, StoreLH1x4
    AddBiasLH1x4:
    ld1 {v0.8h}, [x20]
    fmla v16.8h, v0.8h, v5.h[1]
    
    PostTreatLH1x4:
    fmax v16.8h, v16.8h, v6.8h
    fmin v16.8h, v16.8h, v7.8h

    StoreLH1x4:
    st1 {v16.8h}, [x0]

    E1End:

    subs x3, x3, #1
    add x0, x21, #16
    add x1, x1, #2
    bne LoopE1


End:
ldp x23, x24, [sp, #96]
ldp x21, x22, [sp, #80]
ldp x19, x20, [sp, #64]
ldp d8, d9, [sp, #48]
ldp d10, d11, [sp, #32]
ldp d12, d13, [sp, #16]
ldp d14, d15, [sp], #112
ret


#endif
