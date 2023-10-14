//
//  MNNPackedMatMulFP16_int4.S
//  MNN
//
//  Created by MNN on 2023/05/29.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5
// 8 * 24 MatMul
asm_function MNNPackedMatMulFP16_int4
//void MNNPackedMatMulFP16(FLOAT16* C, const FLOAT16* A, const FLOAT16* B, const size_t* parameter, const FLOAT16* postParameters, const FLOAT16* bias, const FLOAT16* k, const FLOAT16* b);
// x0: C, x1:A, x2:B, x3:parameter, x4: postParameters, x5:bias, x6: quant_alpha, x7: quant_bias
stp d14, d15, [sp, #-80]!
stp d12, d13, [sp, #16]
stp d10, d11, [sp, #32]
stp d8,  d9,  [sp, #48]
stp x19, x20, [sp, #64]

ldr x9, [x3, #8] // l
ldr x10, [x3, #16] // h

ldr x13, [x3, #24] // cStride
ldr x11, [x3, #40] // bExtraStride

// v0, v1, v2: A
// v3, v4: B
// v8 - v31: C
add x10, x10, #7
lsr x10, x10, #3

Start:

cmp x10, #2
blt LH4

LH8:
sub x14, x13, #128
mov x19, x6
mov x20, x7
LoopH:
    mov x15, x1
    ld1 {v4.8h, v5.8h}, [x19], #32 // alpha
    ld1 {v6.8h, v7.8h}, [x20], #32 // bias
    subs x12, x9, #2
    // ld1 {v3.8h, v4.8h}, [x2], #32
    ld1 {v0.8h}, [x2], #16
    ushr v1.16b, v0.16b, #4
    mov w17, #0x0f
    dup v3.16b, w17
    and v2.16b, v0.16b, v3.16b
    mov w17, #8
    dup v0.16b, w17
    sub v1.16b, v1.16b, v0.16b
    sub v2.16b, v2.16b, v0.16b
    zip1 v0.16b, v1.16b, v2.16b
    zip2 v3.16b, v1.16b, v2.16b
    sxtl v1.8h, v0.8b
    sxtl2 v2.8h, v0.16b
    scvtf v0.8h, v1.8h
    scvtf v1.8h, v2.8h
    mov v2.8h, v7.8h
    fmla v2.8h, v1.8h, v5.8h
    mov v1.8h, v6.8h
    fmla v1.8h, v0.8h, v4.8h

    ld1 {v0.8h}, [x15], #16
    fmul v8.8h, v1.8h, v0.h[0]
    fmul v9.8h, v1.8h, v0.h[1]
    fmul v10.8h, v1.8h, v0.h[2]
    fmul v11.8h, v1.8h, v0.h[3]
    fmul v12.8h, v1.8h, v0.h[4]
    fmul v13.8h, v1.8h, v0.h[5]
    fmul v14.8h, v1.8h, v0.h[6]
    fmul v15.8h, v1.8h, v0.h[7]

    fmul v20.8h, v2.8h, v0.h[0]
    fmul v21.8h, v2.8h, v0.h[1]
    fmul v22.8h, v2.8h, v0.h[2]
    fmul v23.8h, v2.8h, v0.h[3]
    fmul v24.8h, v2.8h, v0.h[4]
    fmul v25.8h, v2.8h, v0.h[5]
    fmul v26.8h, v2.8h, v0.h[6]
    fmul v27.8h, v2.8h, v0.h[7]

    ld1 {v1.4h}, [x15], #8
    fmul v16.8h, v1.8h, v1.h[0]
    fmul v17.8h, v1.8h, v1.h[1]
    fmul v18.8h, v1.8h, v1.h[2]
    fmul v19.8h, v1.8h, v1.h[3]
    fmul v28.8h, v2.8h, v1.h[0]
    fmul v29.8h, v2.8h, v1.h[1]
    fmul v30.8h, v2.8h, v1.h[2]
    fmul v31.8h, v2.8h, v1.h[3]

    sxtl v0.8h, v3.8b
    sxtl2 v1.8h, v3.16b
    scvtf v0.8h, v0.8h
    scvtf v1.8h, v1.8h
    mov v2.8h, v7.8h
    fmla v2.8h, v1.8h, v5.8h
    mov v1.8h, v6.8h
    fmla v1.8h, v0.8h, v4.8h

    ld1 {v0.8h}, [x15], #16
    fmla v8.8h, v1.8h, v0.h[0]
    fmla v9.8h, v1.8h, v0.h[1]
    fmla v10.8h, v1.8h, v0.h[2]
    fmla v11.8h, v1.8h, v0.h[3]
    fmla v12.8h, v1.8h, v0.h[4]
    fmla v13.8h, v1.8h, v0.h[5]
    fmla v14.8h, v1.8h, v0.h[6]
    fmla v15.8h, v1.8h, v0.h[7]

    fmla v20.8h, v2.8h, v0.h[0]
    fmla v21.8h, v2.8h, v0.h[1]
    fmla v22.8h, v2.8h, v0.h[2]
    fmla v23.8h, v2.8h, v0.h[3]
    fmla v24.8h, v2.8h, v0.h[4]
    fmla v25.8h, v2.8h, v0.h[5]
    fmla v26.8h, v2.8h, v0.h[6]
    fmla v27.8h, v2.8h, v0.h[7]

    ld1 {v0.4h}, [x15], #8
    fmla v16.8h, v1.8h, v0.h[0]
    fmla v17.8h, v1.8h, v0.h[1]
    fmla v18.8h, v1.8h, v0.h[2]
    fmla v19.8h, v1.8h, v0.h[3]
    fmla v28.8h, v2.8h, v0.h[0]
    fmla v29.8h, v2.8h, v0.h[1]
    fmla v30.8h, v2.8h, v0.h[2]
    fmla v31.8h, v2.8h, v0.h[3]

    beq LoopLEnd

    LoopL2:
        // ld1 {v3.8h, v4.8h}, [x2], #32
        subs x12, x12, #2
        ld1 {v0.8h}, [x2], #16
        ushr v1.16b, v0.16b, #4
        mov w17, #0x0f
        dup v3.16b, w17
        and v2.16b, v0.16b, v3.16b
        mov w17, #8
        dup v0.16b, w17
        sub v1.16b, v1.16b, v0.16b
        sub v2.16b, v2.16b, v0.16b
        zip1 v0.16b, v1.16b, v2.16b
        zip2 v3.16b, v1.16b, v2.16b
        sxtl v1.8h, v0.8b
        sxtl2 v2.8h, v0.16b
        scvtf v0.8h, v1.8h
        scvtf v1.8h, v2.8h
        mov v2.8h, v7.8h
        fmla v2.8h, v1.8h, v5.8h
        mov v1.8h, v6.8h
        fmla v1.8h, v0.8h, v4.8h

        ld1 {v0.8h}, [x15], #16
        fmla v8.8h, v1.8h, v0.h[0]
        fmla v9.8h, v1.8h, v0.h[1]
        fmla v10.8h, v1.8h, v0.h[2]
        fmla v11.8h, v1.8h, v0.h[3]
        fmla v12.8h, v1.8h, v0.h[4]
        fmla v13.8h, v1.8h, v0.h[5]
        fmla v14.8h, v1.8h, v0.h[6]
        fmla v15.8h, v1.8h, v0.h[7]

        fmla v20.8h, v2.8h, v0.h[0]
        fmla v21.8h, v2.8h, v0.h[1]
        fmla v22.8h, v2.8h, v0.h[2]
        fmla v23.8h, v2.8h, v0.h[3]
        fmla v24.8h, v2.8h, v0.h[4]
        fmla v25.8h, v2.8h, v0.h[5]
        fmla v26.8h, v2.8h, v0.h[6]
        fmla v27.8h, v2.8h, v0.h[7]

        ld1 {v0.4h}, [x15], #8
        fmla v16.8h, v1.8h, v0.h[0]
        fmla v17.8h, v1.8h, v0.h[1]
        fmla v18.8h, v1.8h, v0.h[2]
        fmla v19.8h, v1.8h, v0.h[3]
        fmla v28.8h, v2.8h, v0.h[0]
        fmla v29.8h, v2.8h, v0.h[1]
        fmla v30.8h, v2.8h, v0.h[2]
        fmla v31.8h, v2.8h, v0.h[3]

        sxtl v0.8h, v3.8b
        sxtl2 v1.8h, v3.16b
        scvtf v0.8h, v0.8h
        scvtf v1.8h, v1.8h
        mov v2.8h, v7.8h
        fmla v2.8h, v1.8h, v5.8h
        mov v1.8h, v6.8h
        fmla v1.8h, v0.8h, v4.8h

        ld1 {v0.8h}, [x15], #16
        fmla v8.8h, v1.8h, v0.h[0]
        fmla v9.8h, v1.8h, v0.h[1]
        fmla v10.8h, v1.8h, v0.h[2]
        fmla v11.8h, v1.8h, v0.h[3]
        fmla v12.8h, v1.8h, v0.h[4]
        fmla v13.8h, v1.8h, v0.h[5]
        fmla v14.8h, v1.8h, v0.h[6]
        fmla v15.8h, v1.8h, v0.h[7]

        fmla v20.8h, v2.8h, v0.h[0]
        fmla v21.8h, v2.8h, v0.h[1]
        fmla v22.8h, v2.8h, v0.h[2]
        fmla v23.8h, v2.8h, v0.h[3]
        fmla v24.8h, v2.8h, v0.h[4]
        fmla v25.8h, v2.8h, v0.h[5]
        fmla v26.8h, v2.8h, v0.h[6]
        fmla v27.8h, v2.8h, v0.h[7]

        ld1 {v0.4h}, [x15], #8
        fmla v16.8h, v1.8h, v0.h[0]
        fmla v17.8h, v1.8h, v0.h[1]
        fmla v18.8h, v1.8h, v0.h[2]
        fmla v19.8h, v1.8h, v0.h[3]
        fmla v28.8h, v2.8h, v0.h[0]
        fmla v29.8h, v2.8h, v0.h[1]
        fmla v30.8h, v2.8h, v0.h[2]
        fmla v31.8h, v2.8h, v0.h[3]
        bne LoopL2
    
    LoopLEnd:

    add x2, x2, x11
    sub x10, x10, #2
    cmp x10, #2

    cbz x4, StoreLH8

    AddBiasLH8:
    ld1 {v5.8h}, [x4]
    fcvtn v5.4h, v5.4s
    dup v6.8h, v5.h[2] // Min Value
    dup v7.8h, v5.h[3] // Max Value
    ld1 {v0.8h, v1.8h}, [x5], #32

    fmla v8.8h, v0.8h, v5.h[1]
    fmla v9.8h, v0.8h, v5.h[1]
    fmla v10.8h, v0.8h, v5.h[1]
    fmla v11.8h, v0.8h, v5.h[1]

    fmla v12.8h, v0.8h, v5.h[1]
    fmla v13.8h, v0.8h, v5.h[1]
    fmla v14.8h, v0.8h, v5.h[1]
    fmla v15.8h, v0.8h, v5.h[1]

    fmla v16.8h, v0.8h, v5.h[1]
    fmla v17.8h, v0.8h, v5.h[1]
    fmla v18.8h, v0.8h, v5.h[1]
    fmla v19.8h, v0.8h, v5.h[1]

    fmla v20.8h, v1.8h, v5.h[1]
    fmla v21.8h, v1.8h, v5.h[1]
    fmla v22.8h, v1.8h, v5.h[1]
    fmla v23.8h, v1.8h, v5.h[1]

    fmla v24.8h, v1.8h, v5.h[1]
    fmla v25.8h, v1.8h, v5.h[1]
    fmla v26.8h, v1.8h, v5.h[1]
    fmla v27.8h, v1.8h, v5.h[1]

    fmla v28.8h, v1.8h, v5.h[1]
    fmla v29.8h, v1.8h, v5.h[1]
    fmla v30.8h, v1.8h, v5.h[1]
    fmla v31.8h, v1.8h, v5.h[1]

    PostTreatLH8:
    fmax v8.8h, v8.8h, v6.8h
    fmax v9.8h, v9.8h, v6.8h
    fmax v10.8h, v10.8h, v6.8h
    fmax v11.8h, v11.8h, v6.8h
    fmax v12.8h, v12.8h, v6.8h
    fmax v13.8h, v13.8h, v6.8h
    fmax v14.8h, v14.8h, v6.8h
    fmax v15.8h, v15.8h, v6.8h
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

    fmin v8.8h,  v8.8h,  v7.8h
    fmin v9.8h,  v9.8h,  v7.8h
    fmin v10.8h, v10.8h, v7.8h
    fmin v11.8h, v11.8h, v7.8h
    fmin v12.8h, v12.8h, v7.8h
    fmin v13.8h, v13.8h, v7.8h
    fmin v14.8h, v14.8h, v7.8h
    fmin v15.8h, v15.8h, v7.8h
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

    st1 {v8.8h, v9.8h, v10.8h, v11.8h}, [x0], #64
    st1 {v12.8h, v13.8h, v14.8h, v15.8h}, [x0], #64
    st1 {v16.8h, v17.8h, v18.8h, v19.8h}, [x0], x14

    st1 {v20.8h, v21.8h, v22.8h, v23.8h}, [x0], #64
    st1 {v24.8h, v25.8h, v26.8h, v27.8h}, [x0], #64
    st1 {v28.8h, v29.8h, v30.8h, v31.8h}, [x0], x14

    bge LoopH

LH4:
cbz x10, End
mov x19, x6
mov x20, x7
LoopHRemain:
    mov x15, x1
    subs x12, x9, #2
    ld1 {v20.8h}, [x19], #16 // alpha
    ld1 {v21.8h}, [x20], #16 // bias
    mov w17, #0x0f
    dup v22.16b, w17
    mov w17, #8
    dup v23.16b, w17
    // ld1 {v3.8h}, [x2]
    ld1 {v3.8h}, [x2], #16
    // 01234567xxxxxxx89... => 0123456789...
    uzp1 v0.4s, v3.4s, v3.4s
    ushr v1.8b, v0.8b, #4
    and v2.8b, v0.8b, v22.8b
    sub v1.8b, v1.8b, v23.8b
    sub v2.8b, v2.8b, v23.8b
    zip1 v6.8b, v1.8b, v2.8b
    zip2 v7.8b, v1.8b, v2.8b
    sxtl v0.8h, v6.8b
    sxtl v1.8h, v7.8b
    scvtf v6.8h, v0.8h
    scvtf v7.8h, v1.8h
    mov v3.8h, v21.8h
    mov v4.8h, v21.8h
    fmla v3.8h, v6.8h, v20.8h
    fmla v4.8h, v7.8h, v20.8h

    ld1 {v0.4h, v1.4h, v2.4h}, [x15], #24
    fmul v8.8h, v3.8h, v0.h[0]
    fmul v9.8h, v3.8h, v0.h[1]
    fmul v10.8h, v3.8h, v0.h[2]
    fmul v11.8h, v3.8h, v0.h[3]
    fmul v12.8h, v3.8h, v1.h[0]
    fmul v13.8h, v3.8h, v1.h[1]
    fmul v14.8h, v3.8h, v1.h[2]
    fmul v15.8h, v3.8h, v1.h[3]
    fmul v16.8h, v3.8h, v2.h[0]
    fmul v17.8h, v3.8h, v2.h[1]
    fmul v18.8h, v3.8h, v2.h[2]
    fmul v19.8h, v3.8h, v2.h[3]

    ld1 {v0.4h, v1.4h, v2.4h}, [x15], #24
    fmla v8.8h, v4.8h, v0.h[0]
    fmla v9.8h, v4.8h, v0.h[1]
    fmla v10.8h, v4.8h, v0.h[2]
    fmla v11.8h, v4.8h, v0.h[3]
    fmla v12.8h, v4.8h, v1.h[0]
    fmla v13.8h, v4.8h, v1.h[1]
    fmla v14.8h, v4.8h, v1.h[2]
    fmla v15.8h, v4.8h, v1.h[3]
    fmla v16.8h, v4.8h, v2.h[0]
    fmla v17.8h, v4.8h, v2.h[1]
    fmla v18.8h, v4.8h, v2.h[2]
    fmla v19.8h, v4.8h, v2.h[3]

    beq LoopLREnd

    LoopLR:
        subs x12, x12, #2
        // ld1 {v3.8h}, [x2]
        ld1 {v3.8h}, [x2], #16
        // 01234567xxxxxxx89... => 0123456789...
        uzp1 v0.4s, v3.4s, v3.4s
        ushr v1.8b, v0.8b, #4
        and v2.8b, v0.8b, v22.8b
        sub v1.8b, v1.8b, v23.8b
        sub v2.8b, v2.8b, v23.8b
        zip1 v6.8b, v1.8b, v2.8b
        zip2 v7.8b, v1.8b, v2.8b
        sxtl v0.8h, v6.8b
        sxtl v1.8h, v7.8b
        scvtf v6.8h, v0.8h
        scvtf v7.8h, v1.8h
        mov v3.8h, v21.8h
        mov v4.8h, v21.8h
        fmla v3.8h, v6.8h, v20.8h
        fmla v4.8h, v7.8h, v20.8h
        ld1 {v0.4h, v1.4h, v2.4h}, [x15], #24
        fmla v8.8h, v3.8h, v0.h[0]
        fmla v9.8h, v3.8h, v0.h[1]
        fmla v10.8h, v3.8h, v0.h[2]
        fmla v11.8h, v3.8h, v0.h[3]
        fmla v12.8h, v3.8h, v1.h[0]
        fmla v13.8h, v3.8h, v1.h[1]
        fmla v14.8h, v3.8h, v1.h[2]
        fmla v15.8h, v3.8h, v1.h[3]
        fmla v16.8h, v3.8h, v2.h[0]
        fmla v17.8h, v3.8h, v2.h[1]
        fmla v18.8h, v3.8h, v2.h[2]
        fmla v19.8h, v3.8h, v2.h[3]

        ld1 {v0.4h, v1.4h, v2.4h}, [x15], #24
        fmla v8.8h, v4.8h, v0.h[0]
        fmla v9.8h, v4.8h, v0.h[1]
        fmla v10.8h, v4.8h, v0.h[2]
        fmla v11.8h, v4.8h, v0.h[3]
        fmla v12.8h, v4.8h, v1.h[0]
        fmla v13.8h, v4.8h, v1.h[1]
        fmla v14.8h, v4.8h, v1.h[2]
        fmla v15.8h, v4.8h, v1.h[3]
        fmla v16.8h, v4.8h, v2.h[0]
        fmla v17.8h, v4.8h, v2.h[1]
        fmla v18.8h, v4.8h, v2.h[2]
        fmla v19.8h, v4.8h, v2.h[3]

        bne LoopLR
    LoopLREnd:

    cbz x4, StoreLH4
    AddBiasLH4:
    ld1 {v5.8h}, [x4]
    fcvtn v5.4h, v5.4s
    dup v6.8h, v5.h[2] // Min Value
    dup v7.8h, v5.h[3] // Max Value
    ld1 {v0.8h}, [x5], #16

    fmla v8.8h, v0.8h, v5.h[1]
    fmla v9.8h, v0.8h, v5.h[1]
    fmla v10.8h, v0.8h, v5.h[1]
    fmla v11.8h, v0.8h, v5.h[1]

    fmla v12.8h, v0.8h, v5.h[1]
    fmla v13.8h, v0.8h, v5.h[1]
    fmla v14.8h, v0.8h, v5.h[1]
    fmla v15.8h, v0.8h, v5.h[1]

    fmla v16.8h, v0.8h, v5.h[1]
    fmla v17.8h, v0.8h, v5.h[1]
    fmla v18.8h, v0.8h, v5.h[1]
    fmla v19.8h, v0.8h, v5.h[1]

    PostTreatLH4:
    fmax v8.8h, v8.8h, v6.8h
    fmax v9.8h, v9.8h, v6.8h
    fmax v10.8h, v10.8h, v6.8h
    fmax v11.8h, v11.8h, v6.8h
    fmax v12.8h, v12.8h, v6.8h
    fmax v13.8h, v13.8h, v6.8h
    fmax v14.8h, v14.8h, v6.8h
    fmax v15.8h, v15.8h, v6.8h
    fmax v16.8h, v16.8h, v6.8h
    fmax v17.8h, v17.8h, v6.8h
    fmax v18.8h, v18.8h, v6.8h
    fmax v19.8h, v19.8h, v6.8h

    fmin v8.8h,  v8.8h,  v7.8h
    fmin v9.8h,  v9.8h,  v7.8h
    fmin v10.8h, v10.8h, v7.8h
    fmin v11.8h, v11.8h, v7.8h
    fmin v12.8h, v12.8h, v7.8h
    fmin v13.8h, v13.8h, v7.8h
    fmin v14.8h, v14.8h, v7.8h
    fmin v15.8h, v15.8h, v7.8h
    fmin v16.8h, v16.8h, v7.8h
    fmin v17.8h, v17.8h, v7.8h
    fmin v18.8h, v18.8h, v7.8h
    fmin v19.8h, v19.8h, v7.8h

    StoreLH4:

    st1 {v8.8h, v9.8h, v10.8h, v11.8h}, [x0], #64
    st1 {v12.8h, v13.8h, v14.8h, v15.8h}, [x0], #64
    st1 {v16.8h, v17.8h, v18.8h, v19.8h}, [x0]
    sub x10, x10, #1


End:
ldp x19, x20, [sp, #64]
ldp d8,  d9,  [sp, #48]
ldp d10, d11, [sp, #32]
ldp d12, d13, [sp, #16]
ldp d14, d15, [sp], #80

ret

#endif
