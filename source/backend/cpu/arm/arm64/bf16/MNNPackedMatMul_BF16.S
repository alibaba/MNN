//
//  MNNPackedMatMul_BF16.S
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
asm_function NEON_MNNPackedMatMul_BF16
//void NEON_MNNPackedMatMul_BF16(float* C, const float* A, const float* B, const size_t* parameter, const float* postParameters, const float* bias);
// x0: C, x1:A, x2:B, x3:parameter, x4: postParameters, x5:bias
stp d14, d15, [sp, #-64]!
stp d12, d13, [sp, #16]
stp d10, d11, [sp, #32]
stp d8,  d9,  [sp, #48]

//ldr x8, [x3, #0] // deprecated
ldr x9, [x3, #8] // l
ldr x10, [x3, #16] // h

ldr x13, [x3, #24] // cStride
ldr x7, [x3, #40] // bExtraStride

// v0, v1, v2: A
// v3, v4: B
// v8 - v31: C
add x10, x10, #3
lsr x10, x10, #2

cbz x4, Start
ld1 {v5.4s}, [x4]
dup v6.4s, v5.s[2] // Min Value
dup v7.4s, v5.s[3] // Max Value

Start:

cmp x10, #2
blt LH4

LH8:
// sub x14, x13, #80 // in "StoreLH8", total 3 lines Cstride is x13, first 5 line stp is  5 * 8 * sizeof(int16_t) = 64byte
                  // stp should add at last
LoopH:
    mov x15, x1
    subs x12, x9, #1
    ld1 {v3.4h, v4.4h}, [x2], #16 // 8 * sizeof(int16_t)
    ld1 {v0.4h, v1.4h, v2.4h}, [x15], #24 // 12 * sizeof(int16_t)

    shll v3.4s, v3.4h, #16
    shll v4.4s, v4.4h, #16
    shll v0.4s, v0.4h, #16
    shll v1.4s, v1.4h, #16
    shll v2.4s, v2.4h, #16

    fmul v8.4s, v3.4s, v0.s[0]
    fmul v9.4s, v3.4s, v0.s[1]
    fmul v10.4s, v3.4s, v0.s[2]
    fmul v11.4s, v3.4s, v0.s[3]
    fmul v12.4s, v3.4s, v1.s[0]
    fmul v13.4s, v3.4s, v1.s[1]
    fmul v14.4s, v3.4s, v1.s[2]
    fmul v15.4s, v3.4s, v1.s[3]
    fmul v16.4s, v3.4s, v2.s[0]
    fmul v17.4s, v3.4s, v2.s[1]
    fmul v18.4s, v3.4s, v2.s[2]
    fmul v19.4s, v3.4s, v2.s[3]

    fmul v20.4s, v4.4s, v0.s[0]
    fmul v21.4s, v4.4s, v0.s[1]
    fmul v22.4s, v4.4s, v0.s[2]
    fmul v23.4s, v4.4s, v0.s[3]

    fmul v24.4s, v4.4s, v1.s[0]
    fmul v25.4s, v4.4s, v1.s[1]
    fmul v26.4s, v4.4s, v1.s[2]
    fmul v27.4s, v4.4s, v1.s[3]

    fmul v28.4s, v4.4s, v2.s[0]
    fmul v29.4s, v4.4s, v2.s[1]
    fmul v30.4s, v4.4s, v2.s[2]
    fmul v31.4s, v4.4s, v2.s[3]

    beq LoopLEnd

    cmp x12, #2
    blt L1
    LoopL2:
        ld1 {v3.4h, v4.4h}, [x2], #16 // 8 * sizeof(int16_t)
        ld1 {v0.4h, v1.4h, v2.4h}, [x15], #24 // 12 * sizeof(int16_t) //  * sizeof(int16_t)

        shll v3.4s, v3.4h, #16
        shll v4.4s, v4.4h, #16
        shll v0.4s, v0.4h, #16
        shll v1.4s, v1.4h, #16
        shll v2.4s, v2.4h, #16

        fmla v8.4s, v3.4s, v0.s[0]
        fmla v9.4s, v3.4s, v0.s[1]
        fmla v10.4s, v3.4s, v0.s[2]
        fmla v11.4s, v3.4s, v0.s[3]
        fmla v12.4s, v3.4s, v1.s[0]
        fmla v13.4s, v3.4s, v1.s[1]
        fmla v14.4s, v3.4s, v1.s[2]
        fmla v15.4s, v3.4s, v1.s[3]
        fmla v16.4s, v3.4s, v2.s[0]
        fmla v17.4s, v3.4s, v2.s[1]
        fmla v18.4s, v3.4s, v2.s[2]
        fmla v19.4s, v3.4s, v2.s[3]

        fmla v20.4s, v4.4s, v0.s[0]
        fmla v21.4s, v4.4s, v0.s[1]
        fmla v22.4s, v4.4s, v0.s[2]
        fmla v23.4s, v4.4s, v0.s[3]

        fmla v24.4s, v4.4s, v1.s[0]
        fmla v25.4s, v4.4s, v1.s[1]
        fmla v26.4s, v4.4s, v1.s[2]
        fmla v27.4s, v4.4s, v1.s[3]

        fmla v28.4s, v4.4s, v2.s[0]
        fmla v29.4s, v4.4s, v2.s[1]
        fmla v30.4s, v4.4s, v2.s[2]
        fmla v31.4s, v4.4s, v2.s[3]

        ld1 {v3.4h, v4.4h}, [x2], #16 // 8 * sizeof(int16_t)
        ld1 {v0.4h, v1.4h, v2.4h}, [x15], #24 // 12 * sizeof(int16_t) //  * sizeof(int16_t)

        shll v3.4s, v3.4h, #16
        shll v4.4s, v4.4h, #16
        shll v0.4s, v0.4h, #16
        shll v1.4s, v1.4h, #16
        shll v2.4s, v2.4h, #16

        fmla v8.4s, v3.4s, v0.s[0]
        fmla v9.4s, v3.4s, v0.s[1]
        fmla v10.4s, v3.4s, v0.s[2]
        fmla v11.4s, v3.4s, v0.s[3]
        fmla v12.4s, v3.4s, v1.s[0]
        fmla v13.4s, v3.4s, v1.s[1]
        fmla v14.4s, v3.4s, v1.s[2]
        fmla v15.4s, v3.4s, v1.s[3]
        fmla v16.4s, v3.4s, v2.s[0]
        fmla v17.4s, v3.4s, v2.s[1]
        fmla v18.4s, v3.4s, v2.s[2]
        fmla v19.4s, v3.4s, v2.s[3]

        fmla v20.4s, v4.4s, v0.s[0]
        fmla v21.4s, v4.4s, v0.s[1]
        fmla v22.4s, v4.4s, v0.s[2]
        fmla v23.4s, v4.4s, v0.s[3]

        fmla v24.4s, v4.4s, v1.s[0]
        fmla v25.4s, v4.4s, v1.s[1]
        fmla v26.4s, v4.4s, v1.s[2]
        fmla v27.4s, v4.4s, v1.s[3]

        fmla v28.4s, v4.4s, v2.s[0]
        fmla v29.4s, v4.4s, v2.s[1]
        fmla v30.4s, v4.4s, v2.s[2]
        fmla v31.4s, v4.4s, v2.s[3]
        sub x12, x12, #2
        cmp x12, #2
        bge LoopL2

    cbz x12, LoopLEnd

    L1:
        ld1 {v3.4h, v4.4h}, [x2], #16 // 8 * sizeof(int16_t)
        ld1 {v0.4h, v1.4h, v2.4h}, [x15], #24 // 12 * sizeof(int16_t) //  * sizeof(int16_t)

        shll v3.4s, v3.4h, #16
        shll v4.4s, v4.4h, #16
        shll v0.4s, v0.4h, #16
        shll v1.4s, v1.4h, #16
        shll v2.4s, v2.4h, #16

        fmla v8.4s, v3.4s, v0.s[0]
        fmla v9.4s, v3.4s, v0.s[1]
        fmla v10.4s, v3.4s, v0.s[2]
        fmla v11.4s, v3.4s, v0.s[3]
        fmla v12.4s, v3.4s, v1.s[0]
        fmla v13.4s, v3.4s, v1.s[1]
        fmla v14.4s, v3.4s, v1.s[2]
        fmla v15.4s, v3.4s, v1.s[3]
        fmla v16.4s, v3.4s, v2.s[0]
        fmla v17.4s, v3.4s, v2.s[1]
        fmla v18.4s, v3.4s, v2.s[2]
        fmla v19.4s, v3.4s, v2.s[3]

        fmla v20.4s, v4.4s, v0.s[0]
        fmla v21.4s, v4.4s, v0.s[1]
        fmla v22.4s, v4.4s, v0.s[2]
        fmla v23.4s, v4.4s, v0.s[3]

        fmla v24.4s, v4.4s, v1.s[0]
        fmla v25.4s, v4.4s, v1.s[1]
        fmla v26.4s, v4.4s, v1.s[2]
        fmla v27.4s, v4.4s, v1.s[3]

        fmla v28.4s, v4.4s, v2.s[0]
        fmla v29.4s, v4.4s, v2.s[1]
        fmla v30.4s, v4.4s, v2.s[2]
        fmla v31.4s, v4.4s, v2.s[3]

    LoopLEnd:

    add x2, x2, x7 // weight stride
    sub x10, x10, #2
    cmp x10, #2

    cbz x4, StoreLH8

    AddBiasLH8:
    ld1 {v0.4h, v1.4h}, [x5], #16 // 8 * sizeof(int16_t)
    shll v0.4s, v0.4h, #16
    shll v1.4s, v1.4h, #16

    fmla v8.4s, v0.4s, v5.s[1]
    fmla v9.4s, v0.4s, v5.s[1]
    fmla v10.4s, v0.4s, v5.s[1]
    fmla v11.4s, v0.4s, v5.s[1]

    fmla v12.4s, v0.4s, v5.s[1]
    fmla v13.4s, v0.4s, v5.s[1]
    fmla v14.4s, v0.4s, v5.s[1]
    fmla v15.4s, v0.4s, v5.s[1]

    fmla v16.4s, v0.4s, v5.s[1]
    fmla v17.4s, v0.4s, v5.s[1]
    fmla v18.4s, v0.4s, v5.s[1]
    fmla v19.4s, v0.4s, v5.s[1]

    fmla v20.4s, v1.4s, v5.s[1]
    fmla v21.4s, v1.4s, v5.s[1]
    fmla v22.4s, v1.4s, v5.s[1]
    fmla v23.4s, v1.4s, v5.s[1]

    fmla v24.4s, v1.4s, v5.s[1]
    fmla v25.4s, v1.4s, v5.s[1]
    fmla v26.4s, v1.4s, v5.s[1]
    fmla v27.4s, v1.4s, v5.s[1]

    fmla v28.4s, v1.4s, v5.s[1]
    fmla v29.4s, v1.4s, v5.s[1]
    fmla v30.4s, v1.4s, v5.s[1]
    fmla v31.4s, v1.4s, v5.s[1]

    PostTreatLH8:
    fmax v8.4s, v8.4s, v6.4s
    fmax v9.4s, v9.4s, v6.4s
    fmax v10.4s, v10.4s, v6.4s
    fmax v11.4s, v11.4s, v6.4s
    fmax v12.4s, v12.4s, v6.4s
    fmax v13.4s, v13.4s, v6.4s
    fmax v14.4s, v14.4s, v6.4s
    fmax v15.4s, v15.4s, v6.4s
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

    fmin v8.4s,  v8.4s,  v7.4s
    fmin v9.4s,  v9.4s,  v7.4s
    fmin v10.4s, v10.4s, v7.4s
    fmin v11.4s, v11.4s, v7.4s
    fmin v12.4s, v12.4s, v7.4s
    fmin v13.4s, v13.4s, v7.4s
    fmin v14.4s, v14.4s, v7.4s
    fmin v15.4s, v15.4s, v7.4s
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

    shrn v8.4h, v8.4s, #16
    shrn v9.4h, v9.4s, #16
    shrn v10.4h, v10.4s, #16
    shrn v11.4h, v11.4s, #16
    shrn v12.4h, v12.4s, #16
    shrn v13.4h, v13.4s, #16
    shrn v14.4h, v14.4s, #16
    shrn v15.4h, v15.4s, #16
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

    stp d8, d9, [x0]
    stp d10, d11, [x0, #(16 * 1)] // 2 * 4 * sizeof(int16_t)
    stp d12, d13, [x0, #(16 * 2)]
    stp d14, d15, [x0, #(16 * 3)]
    stp d16, d17, [x0, #(16 * 4)]
    stp d18, d19, [x0, #(16 * 5)]
    add x0, x0, x13 // stp donot support post-index offset in register
    stp d20, d21, [x0]
    stp d22, d23, [x0, #(16 * 1)]
    stp d24, d25, [x0, #(16 * 2)]
    stp d26, d27, [x0, #(16 * 3)]
    stp d28, d29, [x0, #(16 * 4)]
    stp d30, d31, [x0, #(16 * 5)]
    add x0, x0, x13

    // st1 {v8.4h, v9.4h, v10.4h, v11.4h}, [x0], #32 // 16 * sizeof(int16_t)
    // st1 {v12.4h, v13.4h, v14.4h, v15.4h}, [x0], #32 // 16 * sizeof(int16_t)
    // st1 {v16.4h, v17.4h, v18.4h, v19.4h}, [x0], x14
    // st1 {v20.4h, v21.4h, v22.4h, v23.4h}, [x0], #32 // 16 * sizeof(int16_t)
    // st1 {v24.4h, v25.4h, v26.4h, v27.4h}, [x0], #32 // 16 * sizeof(int16_t)
    // st1 {v28.4h, v29.4h, v30.4h, v31.4h}, [x0], x14

    bge LoopH

LH4:
cbz x10, End
LoopHRemain:
    mov x15, x1
    subs x12, x9, #1
    ld1 {v3.4h}, [x2]
    ld1 {v0.4h}, [x15], #8
    shll v3.4s, v3.4h, #16
    shll v0.4s, v0.4h, #16

    fmul v8.4s, v3.4s, v0.s[0]
    fmul v9.4s, v3.4s, v0.s[1]
    add x2, x2, #16 //
    ld1 {v1.4h}, [x15], #8
    shll v1.4s, v1.4h, #16

    fmul v10.4s, v3.4s, v0.s[2]
    fmul v11.4s, v3.4s, v0.s[3]
    fmul v12.4s, v3.4s, v1.s[0]

    ld1 {v2.4h}, [x15], #8
    shll v2.4s, v2.4h, #16

    fmul v13.4s, v3.4s, v1.s[1]
    fmul v14.4s, v3.4s, v1.s[2]
    fmul v15.4s, v3.4s, v1.s[3]
    fmul v16.4s, v3.4s, v2.s[0]
    fmul v17.4s, v3.4s, v2.s[1]
    fmul v18.4s, v3.4s, v2.s[2]
    fmul v19.4s, v3.4s, v2.s[3]

    beq LoopLREnd

    LoopLR:
        ld1 {v3.4h}, [x2]
        ld1 {v0.4h, v1.4h, v2.4h}, [x15], #24 // 12 * sizeof(int16_t)
        shll v3.4s, v3.4h, #16
        shll v0.4s, v0.4h, #16
        shll v1.4s, v1.4h, #16
        shll v2.4s, v2.4h, #16

        fmla v8.4s, v3.4s, v0.s[0]
        fmla v9.4s, v3.4s, v0.s[1]
        fmla v10.4s, v3.4s, v0.s[2]
        fmla v11.4s, v3.4s, v0.s[3]
        add x2, x2, #16 //
        fmla v12.4s, v3.4s, v1.s[0]
        fmla v13.4s, v3.4s, v1.s[1]
        fmla v14.4s, v3.4s, v1.s[2]
        fmla v15.4s, v3.4s, v1.s[3]
        fmla v16.4s, v3.4s, v2.s[0]
        fmla v17.4s, v3.4s, v2.s[1]
        fmla v18.4s, v3.4s, v2.s[2]
        fmla v19.4s, v3.4s, v2.s[3]

        subs x12, x12, #1
        bne LoopLR
    LoopLREnd:

    cbz x4, StoreLH4
    AddBiasLH4:
    ld1 {v0.4h}, [x5], #8
    shll v0.4s, v0.4h, #16

    fmla v8.4s, v0.4s, v5.s[1]
    fmla v9.4s, v0.4s, v5.s[1]
    fmla v10.4s, v0.4s, v5.s[1]
    fmla v11.4s, v0.4s, v5.s[1]

    fmla v12.4s, v0.4s, v5.s[1]
    fmla v13.4s, v0.4s, v5.s[1]
    fmla v14.4s, v0.4s, v5.s[1]
    fmla v15.4s, v0.4s, v5.s[1]

    fmla v16.4s, v0.4s, v5.s[1]
    fmla v17.4s, v0.4s, v5.s[1]
    fmla v18.4s, v0.4s, v5.s[1]
    fmla v19.4s, v0.4s, v5.s[1]

    PostTreatLH4:
    fmax v8.4s, v8.4s, v6.4s
    fmax v9.4s, v9.4s, v6.4s
    fmax v10.4s, v10.4s, v6.4s
    fmax v11.4s, v11.4s, v6.4s
    fmax v12.4s, v12.4s, v6.4s
    fmax v13.4s, v13.4s, v6.4s
    fmax v14.4s, v14.4s, v6.4s
    fmax v15.4s, v15.4s, v6.4s
    fmax v16.4s, v16.4s, v6.4s
    fmax v17.4s, v17.4s, v6.4s
    fmax v18.4s, v18.4s, v6.4s
    fmax v19.4s, v19.4s, v6.4s

    fmin v8.4s,  v8.4s,  v7.4s
    fmin v9.4s,  v9.4s,  v7.4s
    fmin v10.4s, v10.4s, v7.4s
    fmin v11.4s, v11.4s, v7.4s
    fmin v12.4s, v12.4s, v7.4s
    fmin v13.4s, v13.4s, v7.4s
    fmin v14.4s, v14.4s, v7.4s
    fmin v15.4s, v15.4s, v7.4s
    fmin v16.4s, v16.4s, v7.4s
    fmin v17.4s, v17.4s, v7.4s
    fmin v18.4s, v18.4s, v7.4s
    fmin v19.4s, v19.4s, v7.4s

    StoreLH4:

    shrn v8.4h, v8.4s, #16
    shrn v9.4h, v9.4s, #16
    shrn v10.4h, v10.4s, #16
    shrn v11.4h, v11.4s, #16
    shrn v12.4h, v12.4s, #16
    shrn v13.4h, v13.4s, #16
    shrn v14.4h, v14.4s, #16
    shrn v15.4h, v15.4s, #16
    shrn v16.4h, v16.4s, #16
    shrn v17.4h, v17.4s, #16
    shrn v18.4h, v18.4s, #16
    shrn v19.4h, v19.4s, #16

    stp d8,  d9,  [x0]
    stp d10, d11, [x0, #(16 * 1)]
    stp d12, d13, [x0, #(16 * 2)]
    stp d14, d15, [x0, #(16 * 3)]
    stp d16, d17, [x0, #(16 * 4)]
    stp d18, d19, [x0, #(16 * 5)]

    // st1 {v8.4h, v9.4h, v10.4h, v11.4h}, [x0], #32
    // st1 {v12.4h, v13.4h, v14.4h, v15.4h}, [x0], #32
    // st1 {v16.4h, v17.4h, v18.4h, v19.4h}, [x0]

    sub x10, x10, #1


End:
ldp d8,  d9,  [sp, #48]
ldp d10, d11, [sp, #32]
ldp d12, d13, [sp, #16]
ldp d14, d15, [sp], #64


ret

#endif
