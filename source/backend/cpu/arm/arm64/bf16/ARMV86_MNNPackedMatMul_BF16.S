//
//  ARMV86_MNNPackedMatMul_BF16.S
//  MNN
//
//  Created by MNN on 2022/10/09.
//  Copyright © 2018-2021 Alibaba Group Holding Limited
//
#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

.macro SET_ZERO d0, d1, d2, d3
    movi \d0\().4s, #0
    movi \d1\().4s, #0
    movi \d2\().4s, #0
    movi \d3\().4s, #0
.endm

.macro Float32ToBf16 d0, d1, d2, d3
    shrn \d0\().4h, \d0\().4s, #16
    shrn \d1\().4h, \d1\().4s, #16
    shrn \d2\().4h, \d2\().4s, #16
    shrn \d3\().4h, \d3\().4s, #16
.endm

.macro FOURFMAX s, d0, d1, d2, d3
    fmax \d0\().4s, \d0\().4s, \s\().4s
    fmax \d1\().4s, \d1\().4s, \s\().4s
    fmax \d2\().4s, \d2\().4s, \s\().4s
    fmax \d3\().4s, \d3\().4s, \s\().4s
.endm

.macro FOURFMIN s, d0, d1, d2, d3
    fmin \d0\().4s, \d0\().4s, \s\().4s
    fmin \d1\().4s, \d1\().4s, \s\().4s
    fmin \d2\().4s, \d2\().4s, \s\().4s
    fmin \d3\().4s, \d3\().4s, \s\().4s
.endm

.macro SET_BIAS s, d0, d1, d2, d3
    mov \d0\().16b, \s\().16b
    mov \d1\().16b, \s\().16b
    mov \d2\().16b, \s\().16b
    mov \d3\().16b, \s\().16b
.endm

// 12 * 8 * 4 MatMul
asm_function ARMV86_MNNPackedMatMul_BF16
//void ARMV86_MNNPackedMatMul_BF16(float* C, const float* A, const float* B, const size_t* parameter, const float* postParameters, const float* bias);
// x0: C, x1:A, x2:B, x3:parameter, x4: postParameters, x5:bias
stp d14, d15, [sp, #-80]!
stp d12, d13, [sp, #16]
stp d10, d11, [sp, #32]
stp d8,  d9,  [sp, #48]
stp x19, x21, [sp, #64]

//ldr x8, [x3, #0] // deprecated
ldr x9, [x3, #8] // l
ldr x10, [x3, #16] // h
mov x11, #64  // B_stride = LP * HP = 4 * 8 * sizeof(int16_t)

ldr x13, [x3, #24] // cStride
ldr x7, [x3, #40] // bExtraStride

add x10, x10, #3
lsr x10, x10, #2
add x9, x9, #3
lsr x9, x9, #2

cbz x4, Start
ld1 {v5.4s}, [x4]
mov w19, v5.s[2] // min value
mov w20, v5.s[3] // max value

Start:
    cmp x10, #2
    blt LH4
LH8:
    sub x14, x13, #96 // cStride - 96
LoopH:
    mov x15, x1
    mov x12, x9
    cbz x5, NoBiasH8
    ld1 {v0.4h, v1.4h}, [x5], #16 // 8 * sizeof(int16_t)
    shll v0.4s, v0.4h, #16
    shll v1.4s, v1.4h, #16
    mov v2.16b, v0.16b
    mov v3.16b, v1.16b
    uzp1 v18.2d, v0.2d, v2.2d   // bias_0, bias_1, bias_0, bias_1
    uzp2 v19.2d, v0.2d, v2.2d   // bias_2, bias_3, bias_2, bias_3
    uzp1 v30.2d, v1.2d, v3.2d   // bias_0, bias_1, bias_0, bias_1
    uzp2 v31.2d, v1.2d, v3.2d   // bias_2, bias_3, bias_2, bias_3
    SET_BIAS v18, v8, v10, v12, v14
    mov v16.16b, v18.16b
    SET_BIAS v19, v9, v11, v13, v15
    mov v17.16b, v19.16b
    SET_BIAS v30, v20, v22, v24, v26
    mov v28.16b, v30.16b
    SET_BIAS v31, v21, v23, v25, v27
    mov v29.16b, v31.16b
    b LoopL
    NoBiasH8:
        SET_ZERO v8, v9, v10, v11
        SET_ZERO v12, v13, v14, v15
        SET_ZERO v16, v17, v18, v19
        SET_ZERO v20, v21, v22, v23
        SET_ZERO v24, v25, v26, v27
        SET_ZERO v28, v29, v30, v31
    LoopL:
        // A [12, 4, bf16] : rn = 6  : v2 - v7
        // B [ 8, 4, bf16] : rn = 2  : v0 - v1
        // C [12, 8, fp32] : rn = 24 : v8 - v31
        ld1 {v2.8h, v3.8h, v4.8h, v5.8h}, [x15], #64 // A: 8 * 4 * sizeof(int16_t)
        ld1 {v6.8h, v7.8h}, [x15], #32               // A: 4 * 4 * sizeof(int16_t)
        ld1 {v0.8h, v1.8h}, [x2],  #32               // B: 4 * 4 * sizeof(int16_t) 
        .inst 0x6e40ec48 // bfmmla v8.4s, v2.8h, v0.8h
        .inst 0x6e41ec49 // bfmmla v9.4s, v2.8h, v1.8h
        .inst 0x6e40ec6a // bfmmla v10.4s, v3.8h, v0.8h
        .inst 0x6e41ec6b // bfmmla v11.4s, v3.8h, v1.8h
        .inst 0x6e40ec8c // bfmmla v12.4s, v4.8h, v0.8h
        .inst 0x6e41ec8d // bfmmla v13.4s, v4.8h, v1.8h
        .inst 0x6e40ecae // bfmmla v14.4s, v5.8h, v0.8h
        .inst 0x6e41ecaf // bfmmla v15.4s, v5.8h, v1.8h
        .inst 0x6e40ecd0 // bfmmla v16.4s, v6.8h, v0.8h
        .inst 0x6e41ecd1 // bfmmla v17.4s, v6.8h, v1.8h
        .inst 0x6e40ecf2 // bfmmla v18.4s, v7.8h, v0.8h
        .inst 0x6e41ecf3 // bfmmla v19.4s, v7.8h, v1.8h
        ld1 {v0.8h, v1.8h}, [x2],  #32               // B: 4 * 4 * sizeof(int16_t) 
        .inst 0x6e40ec54 // bfmmla v20.4s, v2.8h, v0.8h
        .inst 0x6e41ec55 // bfmmla v21.4s, v2.8h, v1.8h
        .inst 0x6e40ec76 // bfmmla v22.4s, v3.8h, v0.8h
        .inst 0x6e41ec77 // bfmmla v23.4s, v3.8h, v1.8h
        .inst 0x6e40ec98 // bfmmla v24.4s, v4.8h, v0.8h
        .inst 0x6e41ec99 // bfmmla v25.4s, v4.8h, v1.8h
        .inst 0x6e40ecba // bfmmla v26.4s, v5.8h, v0.8h
        .inst 0x6e41ecbb // bfmmla v27.4s, v5.8h, v1.8h
        .inst 0x6e40ecdc // bfmmla v28.4s, v6.8h, v0.8h
        .inst 0x6e41ecdd // bfmmla v29.4s, v6.8h, v1.8h
        .inst 0x6e40ecfe // bfmmla v30.4s, v7.8h, v0.8h
        .inst 0x6e41ecff // bfmmla v31.4s, v7.8h, v1.8h
        subs x12, x12, #1
        bgt LoopL
    LoopLEnd:
        uzp1 v7.2d, v8.2d, v9.2d
        uzp2 v8.2d, v8.2d, v9.2d
        uzp1 v9.2d, v10.2d, v11.2d
        uzp2 v10.2d, v10.2d, v11.2d
        uzp1 v11.2d, v12.2d, v13.2d
        uzp2 v12.2d, v12.2d, v13.2d
        uzp1 v13.2d, v14.2d, v15.2d
        uzp2 v14.2d, v14.2d, v15.2d
        uzp1 v15.2d, v16.2d, v17.2d
        uzp2 v16.2d, v16.2d, v17.2d
        uzp1 v17.2d, v18.2d, v19.2d
        uzp2 v18.2d, v18.2d, v19.2d
        uzp1 v19.2d, v20.2d, v21.2d
        uzp2 v20.2d, v20.2d, v21.2d
        uzp1 v21.2d, v22.2d, v23.2d
        uzp2 v22.2d, v22.2d, v23.2d
        uzp1 v23.2d, v24.2d, v25.2d
        uzp2 v24.2d, v24.2d, v25.2d
        uzp1 v25.2d, v26.2d, v27.2d
        uzp2 v26.2d, v26.2d, v27.2d
        uzp1 v27.2d, v28.2d, v29.2d
        uzp2 v28.2d, v28.2d, v29.2d
        uzp1 v29.2d, v30.2d, v31.2d
        uzp2 v30.2d, v30.2d, v31.2d
        cbz x4, StoreLH8
    PostTreatLH8:
        dup v5.4s, w19
        dup v6.4s, w20
        FOURFMAX v5, v7, v8, v9, v10
        FOURFMAX v5, v11, v12, v13, v14
        FOURFMAX v5, v15, v16, v17, v18
        FOURFMAX v5, v19, v20, v21, v22
        FOURFMAX v5, v23, v24, v25, v26
        FOURFMAX v5, v27, v28, v29, v30
        FOURFMIN v6, v7, v8, v9, v10
        FOURFMIN v6, v11, v12, v13, v14
        FOURFMIN v6, v15, v16, v17, v18
        FOURFMIN v6, v19, v20, v21, v22
        FOURFMIN v6, v23, v24, v25, v26
        FOURFMIN v6, v27, v28, v29, v30
    StoreLH8:
        Float32ToBf16 v7, v8, v9, v10
        Float32ToBf16 v11, v12, v13, v14
        Float32ToBf16 v15, v16, v17, v18
        Float32ToBf16 v19, v20, v21, v22
        Float32ToBf16 v23, v24, v25, v26
        Float32ToBf16 v27, v28, v29, v30
        st1 {v7.4h, v8.4h, v9.4h, v10.4h},    [x0], #32 // 16 * sizeof(int16_t)
        st1 {v11.4h, v12.4h, v13.4h, v14.4h}, [x0], #32 // 16 * sizeof(int16_t)
        st1 {v15.4h, v16.4h, v17.4h, v18.4h}, [x0], #32 // 16 * sizeof(int16_t)
        add x0, x0, x14
        st1 {v19.4h, v20.4h, v21.4h, v22.4h}, [x0], #32 // 16 * sizeof(int16_t)
        st1 {v23.4h, v24.4h, v25.4h, v26.4h}, [x0], #32 // 16 * sizeof(int16_t)
        st1 {v27.4h, v28.4h, v29.4h, v30.4h}, [x0], #32 // 16 * sizeof(int16_t)
        add x0, x0, x14
        add x2, x2, x7 // weight stride
        sub x10, x10, #2
        cmp x10, #2
        bge LoopH
LH4:
cbz x10, End
LoopHR:
    mov x15, x1
    mov x12, x9
    cbz x5, NoBiasH4
    ld1 {v0.4h}, [x5], #8 // 8 * sizeof(int16_t)
    shll v0.4s, v0.4h, #16
    mov v2.16b, v0.16b
    uzp1 v18.2d, v0.2d, v2.2d   // bias_0, bias_1, bias_0, bias_1
    uzp2 v19.2d, v0.2d, v2.2d   // bias_2, bias_3, bias_2, bias_3
    SET_BIAS v18, v8, v10, v12, v14
    mov v16.16b, v18.16b
    SET_BIAS v19, v9, v11, v13, v15
    mov v17.16b, v19.16b
    b LoopLR
    NoBiasH4:
        SET_ZERO v8, v9, v10, v11
        SET_ZERO v12, v13, v14, v15
        SET_ZERO v16, v17, v18, v19
    LoopLR:
        // A [12, 4, bf16] : rn = 6  : v2 - v7
        // B [ 4, 4, bf16] : rn = 2  : v0 - v1
        // C [12, 4, fp32] : rn = 12 : v8 - v19
        ld1 {v2.8h, v3.8h, v4.8h, v5.8h}, [x15], #64 // A: 8 * 4 * sizeof(int16_t)
        ld1 {v6.8h, v7.8h}, [x15], #32               // A: 4 * 4 * sizeof(int16_t)
        ld1 {v0.8h, v1.8h}, [x2],  x11               // B: 4 * 4 * sizeof(int16_t)
        .inst 0x6e40ec48 // bfmmla v8.4s, v2.8h, v0.8h
        .inst 0x6e41ec49 // bfmmla v9.4s, v2.8h, v1.8h
        .inst 0x6e40ec6a // bfmmla v10.4s, v3.8h, v0.8h
        .inst 0x6e41ec6b // bfmmla v11.4s, v3.8h, v1.8h
        .inst 0x6e40ec8c // bfmmla v12.4s, v4.8h, v0.8h
        .inst 0x6e41ec8d // bfmmla v13.4s, v4.8h, v1.8h
        .inst 0x6e40ecae // bfmmla v14.4s, v5.8h, v0.8h
        .inst 0x6e41ecaf // bfmmla v15.4s, v5.8h, v1.8h
        .inst 0x6e40ecd0 // bfmmla v16.4s, v6.8h, v0.8h
        .inst 0x6e41ecd1 // bfmmla v17.4s, v6.8h, v1.8h
        .inst 0x6e40ecf2 // bfmmla v18.4s, v7.8h, v0.8h
        .inst 0x6e41ecf3 // bfmmla v19.4s, v7.8h, v1.8h
        subs x12, x12, #1
        bgt LoopLR
    LoopLREnd:
        add x2, x2, x7 // weight stride
        uzp1 v7.2d, v8.2d, v9.2d
        uzp2 v8.2d, v8.2d, v9.2d
        uzp1 v9.2d, v10.2d, v11.2d
        uzp2 v10.2d, v10.2d, v11.2d
        uzp1 v11.2d, v12.2d, v13.2d
        uzp2 v12.2d, v12.2d, v13.2d
        uzp1 v13.2d, v14.2d, v15.2d
        uzp2 v14.2d, v14.2d, v15.2d
        uzp1 v15.2d, v16.2d, v17.2d
        uzp2 v16.2d, v16.2d, v17.2d
        uzp1 v17.2d, v18.2d, v19.2d
        uzp2 v18.2d, v18.2d, v19.2d
        cbz x4, StoreLH4
    PostTreatLH4:
        dup v5.4s, w19
        dup v6.4s, w20
        FOURFMAX v5, v7, v8, v9, v10
        FOURFMAX v5, v11, v12, v13, v14
        FOURFMAX v5, v15, v16, v17, v18
        FOURFMIN v6, v7, v8, v9, v10
        FOURFMIN v6, v11, v12, v13, v14
        FOURFMIN v6, v15, v16, v17, v18
    StoreLH4:
        Float32ToBf16 v7, v8, v9, v10
        Float32ToBf16 v11, v12, v13, v14
        Float32ToBf16 v15, v16, v17, v18
        st1 {v7.4h, v8.4h, v9.4h, v10.4h},    [x0], #32 // 16 * sizeof(int16_t)
        st1 {v11.4h, v12.4h, v13.4h, v14.4h}, [x0], #32 // 16 * sizeof(int16_t)
        st1 {v15.4h, v16.4h, v17.4h, v18.4h}, [x0], #32 // 16 * sizeof(int16_t)
End:
ldp x19, x21, [sp, #64]
ldp d8,  d9,  [sp, #48]
ldp d10, d11, [sp, #32]
ldp d12, d13, [sp, #16]
ldp d14, d15, [sp], #80
ret

#endif
