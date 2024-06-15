//
//  MNNPackedMatMulFP16_int4.S
//  MNN
//
//  Created by MNN on 2023/06/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

.macro FMLA_8 d0, d1, d2, d3, d4, d5, d6, d7, s0, s1
    fmla \d0\().8h, \s0\().8h, \s1\().h[0]
    fmla \d1\().8h, \s0\().8h, \s1\().h[1]
    fmla \d2\().8h, \s0\().8h, \s1\().h[2]
    fmla \d3\().8h, \s0\().8h, \s1\().h[3]
    fmla \d4\().8h, \s0\().8h, \s1\().h[4]
    fmla \d5\().8h, \s0\().8h, \s1\().h[5]
    fmla \d6\().8h, \s0\().8h, \s1\().h[6]
    fmla \d7\().8h, \s0\().8h, \s1\().h[7]
.endm

.macro FMLA_4 d0, d1, d2, d3, s0, s1
    fmla \d0\().8h, \s0\().8h, \s1\().h[0]
    fmla \d1\().8h, \s0\().8h, \s1\().h[1]
    fmla \d2\().8h, \s0\().8h, \s1\().h[2]
    fmla \d3\().8h, \s0\().8h, \s1\().h[3]
.endm

.macro FMUL_8 d0, d1, d2, d3, d4, d5, d6, d7, s0, s1
    fmul \d0\().8h, \s0\().8h, \s1\().h[0]
    fmul \d1\().8h, \s0\().8h, \s1\().h[1]
    fmul \d2\().8h, \s0\().8h, \s1\().h[2]
    fmul \d3\().8h, \s0\().8h, \s1\().h[3]
    fmul \d4\().8h, \s0\().8h, \s1\().h[4]
    fmul \d5\().8h, \s0\().8h, \s1\().h[5]
    fmul \d6\().8h, \s0\().8h, \s1\().h[6]
    fmul \d7\().8h, \s0\().8h, \s1\().h[7]
.endm

.macro FMUL_4 d0, d1, d2, d3, s0, s1
    fmul \d0\().8h, \s0\().8h, \s1\().h[0]
    fmul \d1\().8h, \s0\().8h, \s1\().h[1]
    fmul \d2\().8h, \s0\().8h, \s1\().h[2]
    fmul \d3\().8h, \s0\().8h, \s1\().h[3]
.endm

.macro FADD_12 d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, z0
    fadd \d0\().8h, \d0\().8h, \z0\().8h
    fadd \d1\().8h, \d1\().8h, \z0\().8h
    fadd \d2\().8h, \d2\().8h, \z0\().8h
    fadd \d3\().8h, \d3\().8h, \z0\().8h
    fadd \d4\().8h, \d4\().8h, \z0\().8h
    fadd \d5\().8h, \d5\().8h, \z0\().8h
    fadd \d6\().8h, \d6\().8h, \z0\().8h
    fadd \d7\().8h, \d7\().8h, \z0\().8h
    fadd \d8\().8h, \d8\().8h, \z0\().8h
    fadd \d9\().8h, \d9\().8h, \z0\().8h
    fadd \d10\().8h, \d10\().8h, \z0\().8h
    fadd \d11\().8h, \d11\().8h, \z0\().8h
.endm

.macro FMAX_12 d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, z0
    fmax \d0\().8h, \d0\().8h, \z0\().8h
    fmax \d1\().8h, \d1\().8h, \z0\().8h
    fmax \d2\().8h, \d2\().8h, \z0\().8h
    fmax \d3\().8h, \d3\().8h, \z0\().8h
    fmax \d4\().8h, \d4\().8h, \z0\().8h
    fmax \d5\().8h, \d5\().8h, \z0\().8h
    fmax \d6\().8h, \d6\().8h, \z0\().8h
    fmax \d7\().8h, \d7\().8h, \z0\().8h
    fmax \d8\().8h, \d8\().8h, \z0\().8h
    fmax \d9\().8h, \d9\().8h, \z0\().8h
    fmax \d10\().8h, \d10\().8h, \z0\().8h
    fmax \d11\().8h, \d11\().8h, \z0\().8h
.endm

.macro FMIN_12 d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, z0
    fmin \d0\().8h, \d0\().8h, \z0\().8h
    fmin \d1\().8h, \d1\().8h, \z0\().8h
    fmin \d2\().8h, \d2\().8h, \z0\().8h
    fmin \d3\().8h, \d3\().8h, \z0\().8h
    fmin \d4\().8h, \d4\().8h, \z0\().8h
    fmin \d5\().8h, \d5\().8h, \z0\().8h
    fmin \d6\().8h, \d6\().8h, \z0\().8h
    fmin \d7\().8h, \d7\().8h, \z0\().8h
    fmin \d8\().8h, \d8\().8h, \z0\().8h
    fmin \d9\().8h, \d9\().8h, \z0\().8h
    fmin \d10\().8h, \d10\().8h, \z0\().8h
    fmin \d11\().8h, \d11\().8h, \z0\().8h
.endm

// 8 * 24 MatMul
asm_function MNNPackedMatMulFP16_int4
//void MNNPackedMatMulFP16_int4(FLOAT16* C, const FLOAT16* A, const FLOAT16* B, const size_t* parameter, const FLOAT16* postParameters, const FLOAT16* bias, const FLOAT16* k, const FLOAT16* b);
// x0: C, x1:A, x2:B, x3:parameter, x4: postParameters, x5:bias, x6: quant_alpha, x7: quant_bias
stp d14, d15, [sp, #-96]!
stp d12, d13, [sp, #16]
stp d10, d11, [sp, #32]
stp d8,  d9,  [sp, #48]
stp x19, x20, [sp, #64]
stp x21, x22, [sp, #80]

ldr x9, [x3, #8] // l
ldr x10, [x3, #16] // h

ldr x13, [x3, #24] // cStride
ldr x11, [x3, #40] // bExtraStride
ldr x19, [x3, #48] // blockId
mov x20, x0
add x10, x10, #7
lsr x10, x10, #3

Start:

cmp x10, #2
blt LH4

LH8:
sub x14, x13, #128
LoopH:
    mov x15, x1
    mov x22, x2
    ld1 {v4.8h, v5.8h}, [x6], #32 // alpha
    ld1 {v6.8h, v7.8h}, [x7], #32 // bias
    subs x12, x9, #1

    ld1 {v0.8b}, [x2], #8
    ushr v1.8b, v0.8b, #4
    movi v3.8b, #0x0f
    and v2.8b, v0.8b, v3.8b
    zip1 v3.8b, v1.8b, v2.8b
    zip2 v2.8b, v1.8b, v2.8b

    sxtl v0.8h, v3.8b
    sxtl v1.8h, v2.8b
    scvtf v0.8h, v0.8h
    scvtf v1.8h, v1.8h
    mov v2.8h, v7.8h
    fmla v2.8h, v1.8h, v5.8h
    mov v1.8h, v6.8h
    fmla v1.8h, v0.8h, v4.8h

    cbnz x19, LH8_BLOCK_GT_0

    LH8_BLOCK0:
    ld1 {v0.8h}, [x15], #16
    FMUL_8 v8, v9, v10, v11, v12, v13, v14, v15, v1, v0
    FMUL_8 v20, v21, v22, v23, v24, v25, v26, v27, v2, v0
    ld1 {v0.4h}, [x15], #8
    FMUL_4 v16, v17, v18, v19, v1, v0
    FMUL_4 v28, v29, v30, v31, v2, v0
    b LH8_INIT_END

    LH8_BLOCK_GT_0:
    ld1 {v8.8h, v9.8h, v10.8h, v11.8h}, [x20], #64
    ld1 {v12.8h, v13.8h, v14.8h, v15.8h}, [x20], #64
    ld1 {v16.8h, v17.8h, v18.8h, v19.8h}, [x20], x14

    ld1 {v20.8h, v21.8h, v22.8h, v23.8h}, [x20], #64
    ld1 {v24.8h, v25.8h, v26.8h, v27.8h}, [x20], #64
    ld1 {v28.8h, v29.8h, v30.8h, v31.8h}, [x20], x14

    ld1 {v0.8h}, [x15], #16
    FMLA_8 v8, v9, v10, v11, v12, v13, v14, v15, v1, v0
    FMLA_8 v20, v21, v22, v23, v24, v25, v26, v27, v2, v0
    ld1 {v0.4h}, [x15], #8
    FMLA_4 v16, v17, v18, v19, v1, v0
    FMLA_4 v28, v29, v30, v31, v2, v0

    LH8_INIT_END:
    beq LoopLEnd

    LoopL1:
        subs x12, x12, #1
        ld1 {v0.8b}, [x2], #8
        ushr v1.8b, v0.8b, #4
        movi v3.8b, #0x0f
        and v2.8b, v0.8b, v3.8b
        zip1 v3.8b, v1.8b, v2.8b
        zip2 v2.8b, v1.8b, v2.8b

        sxtl v0.8h, v3.8b
        sxtl v1.8h, v2.8b
        scvtf v0.8h, v0.8h
        scvtf v1.8h, v1.8h
        mov v2.8h, v7.8h
        fmla v2.8h, v1.8h, v5.8h
        mov v1.8h, v6.8h
        fmla v1.8h, v0.8h, v4.8h

        ld1 {v0.8h}, [x15], #16
        FMLA_8 v8, v9, v10, v11, v12, v13, v14, v15, v1, v0
        FMLA_8 v20, v21, v22, v23, v24, v25, v26, v27, v2, v0
        ld1 {v0.4h}, [x15], #8
        FMLA_4 v16, v17, v18, v19, v1, v0
        FMLA_4 v28, v29, v30, v31, v2, v0
        bne LoopL1
    
    LoopLEnd:

    add x2, x2, x11
    sub x10, x10, #2
    cmp x10, #2

    cbz x4, StoreLH8 // If postParameter* is nullptr, not the last blockId, just store the intemediate results.

    ld1 {v5.8h}, [x4]
    fcvtn v5.4h, v5.4s
    dup v6.8h, v5.h[2] // Min Value
    dup v7.8h, v5.h[3] // Max Value

    AddBiasLH8:
    cbz x5, PostTreatLH8
    ld1 {v0.8h, v1.8h}, [x5], #32 // gemm bias
    FADD_12 v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v0
    FADD_12 v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v1

    PostTreatLH8:
    FMAX_12 v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v6
    FMAX_12 v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v6
    FMIN_12 v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v7
    FMIN_12 v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v7

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
LoopHRemain:
    //mov x22, x2
    mov x15, x1
    subs x12, x9, #1
    ld1 {v20.8h}, [x6], #16 // alpha
    ld1 {v21.8h}, [x7], #16 // bias

    ld1 {v0.8b}, [x2], #8
    ushr v1.8b, v0.8b, #4
    movi v3.8b, #0x0f
    and v2.8b, v0.8b, v3.8b
    zip1 v3.8b, v1.8b, v2.8b

    sxtl v3.8h, v3.8b
    scvtf v6.8h, v3.8h
    mov v3.8h, v21.8h
    fmla v3.8h, v6.8h, v20.8h

    ld1 {v0.4h, v1.4h, v2.4h}, [x15], #24
    cbnz x19, LH4_BLOCK_GT_0

    LH4_BLOCK0:
    
    FMUL_4 v8, v9, v10, v11, v3, v0
    FMUL_4 v12, v13, v14, v15, v3, v1
    FMUL_4 v16, v17, v18, v19, v3, v2
    b LH4_INIT_END
    /* 
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
    */

    LH4_BLOCK_GT_0:
    ld1 {v8.8h, v9.8h, v10.8h, v11.8h}, [x20], #64
    ld1 {v12.8h, v13.8h, v14.8h, v15.8h}, [x20], #64
    ld1 {v16.8h, v17.8h, v18.8h, v19.8h}, [x20]
    FMLA_4 v8, v9, v10, v11, v3, v0
    FMLA_4 v12, v13, v14, v15, v3, v1
    FMLA_4 v16, v17, v18, v19, v3, v2

    LH4_INIT_END:
    beq LoopLREnd

    LoopLR:
        subs x12, x12, #1
        ld1 {v0.8b}, [x2], #8
        ushr v1.8b, v0.8b, #4
        movi v3.8b, #0x0f
        and v2.8b, v0.8b, v3.8b
        zip1 v3.8b, v1.8b, v2.8b

        sxtl v3.8h, v3.8b

        scvtf v6.8h, v3.8h
        mov v3.8h, v21.8h
        fmla v3.8h, v6.8h, v20.8h

        ld1 {v0.4h, v1.4h, v2.4h}, [x15], #24
        FMLA_4 v8, v9, v10, v11, v3, v0
        FMLA_4 v12, v13, v14, v15, v3, v1
        FMLA_4 v16, v17, v18, v19, v3, v2
        bne LoopLR
    LoopLREnd:

    cbz x4, StoreLH4
    
    ld1 {v5.8h}, [x4]
    fcvtn v5.4h, v5.4s
    dup v6.8h, v5.h[2] // Min Value
    dup v7.8h, v5.h[3] // Max Value
    AddBiasLH4:
    cbz x5, PostTreatLH4
    ld1 {v0.8h}, [x5], #16
    FADD_12 v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v0 

    PostTreatLH4:
    FMAX_12 v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v6
    FMIN_12 v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v7

    StoreLH4:

    st1 {v8.8h, v9.8h, v10.8h, v11.8h}, [x0], #64
    st1 {v12.8h, v13.8h, v14.8h, v15.8h}, [x0], #64
    st1 {v16.8h, v17.8h, v18.8h, v19.8h}, [x0]


End:
ldp x21, x22, [sp, #80]
ldp x19, x20, [sp, #64]
ldp d8,  d9,  [sp, #48]
ldp d10, d11, [sp, #32]
ldp d12, d13, [sp, #16]
ldp d14, d15, [sp], #96

ret

#endif
