//
//  ARMV86_MNNPackedMatMulRemain_BF16.S
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

.macro SET_BIAS s, d0, d1, d2
    mov \d0\().16b, \s\().16b
    mov \d1\().16b, \s\().16b
    mov \d2\().16b, \s\().16b
.endm

// 12 * 8 * 4 MatMul
asm_function ARMV86_MNNPackedMatMulRemain_BF16
//void ARMV86_MNNPackedMatMulRemain_BF16(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter, const float* postParameters, const float* bias);
//Auto x0: C, x1:A, x2:B, x3:eSize, x4:parameter, x5:postParameters, x6:bias
sub sp, sp, #64
str x19, [sp, #0]
str x20, [sp, #8]
str x21, [sp, #16]
str x22, [sp, #24]
ldr x11, [x4, #0] // aStride
ldr x9, [x4, #8] // l
ldr x10, [x4, #16] // h
lsl x11, x11, #2 // aStride * 4
mov x22, #64  // B_stride = LP * HP = 4 * 8 * sizeof(int16_t)

ldr x7, [x4, #24] // cStride
ldr x19, [x4, #40] // bExtraStride

add x10, x10, #3
lsr x10, x10, #2
add x9, x9, #3
lsr x9, x9, #2

cbz x5, Start
ld1 {v5.4s}, [x5]
dup v9.4s, v5.s[2] // Min Value
dup v10.4s, v5.s[3] // Max Value

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
    sub x14, x7, #64 // cStride - 64
    LoopH8x8:
        mov x15, x1 // src, A
        mov x12, x9 // l
        cbz x5, NoBiasLH8
        ld1 {v0.4h, v1.4h}, [x20], #16 // 8 * sizeof(int16_t)
        shll v0.4s, v0.4h, #16
        shll v1.4s, v1.4h, #16
        mov v2.16b, v0.16b
        mov v3.16b, v1.16b
        uzp1 v16.2d, v0.2d, v2.2d   // bias_0, bias_1, bias_0, bias_1
        uzp2 v17.2d, v0.2d, v2.2d   // bias_2, bias_3, bias_2, bias_3
        uzp1 v24.2d, v1.2d, v3.2d   // bias_0, bias_1, bias_0, bias_1
        uzp2 v25.2d, v1.2d, v3.2d   // bias_2, bias_3, bias_2, bias_3
        SET_BIAS v16, v18, v20, v22
        SET_BIAS v17, v19, v21, v23
        SET_BIAS v24, v26, v28, v30
        SET_BIAS v25, v27, v29, v31
        b LoopL
        NoBiasLH8:
            SET_ZERO v16, v17, v18, v19
            SET_ZERO v20, v21, v22, v23
            SET_ZERO v24, v25, v26, v27
            SET_ZERO v28, v29, v30, v31
        LoopL:
            // A [8, 4, bf16] : rn = 4  : v4 - v7
            // B [8, 4, bf16] : rn = 4  : v0 - v3
            // C [8, 8, fp32] : rn = 16 : v16 - v31
            ld1 {v4.8h, v5.8h, v6.8h, v7.8h}, [x15], x11 // A: 8 * 4 * sizeof(int16_t)
            ld1 {v0.8h, v1.8h, v2.8h, v3.8h}, [x13], x22 // B: 8 * 4 * sizeof(int16_t)
            .inst 0x6e40ec90 // bfmmla v16.4s, v4.8h, v0.8h
            .inst 0x6e41ec91 // bfmmla v17.4s, v4.8h, v1.8h
            .inst 0x6e40ecb2 // bfmmla v18.4s, v5.8h, v0.8h
            .inst 0x6e41ecb3 // bfmmla v19.4s, v5.8h, v1.8h
            .inst 0x6e40ecd4 // bfmmla v20.4s, v6.8h, v0.8h
            .inst 0x6e41ecd5 // bfmmla v21.4s, v6.8h, v1.8h
            .inst 0x6e40ecf6 // bfmmla v22.4s, v7.8h, v0.8h
            .inst 0x6e41ecf7 // bfmmla v23.4s, v7.8h, v1.8h
            .inst 0x6e42ec98 // bfmmla v24.4s, v4.8h, v2.8h
            .inst 0x6e43ec99 // bfmmla v25.4s, v4.8h, v3.8h
            .inst 0x6e42ecba // bfmmla v26.4s, v5.8h, v2.8h
            .inst 0x6e43ecbb // bfmmla v27.4s, v5.8h, v3.8h
            .inst 0x6e42ecdc // bfmmla v28.4s, v6.8h, v2.8h
            .inst 0x6e43ecdd // bfmmla v29.4s, v6.8h, v3.8h
            .inst 0x6e42ecfe // bfmmla v30.4s, v7.8h, v2.8h
            .inst 0x6e43ecff // bfmmla v31.4s, v7.8h, v3.8h
            subs x12, x12, #1
            bgt LoopL
        LoopLEnd:
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
            cbz x5, StoreLH8
        PostTreatLH8:
            FOURFMAX v9, v15, v16, v17, v18
            FOURFMAX v9, v19, v20, v21, v22
            FOURFMAX v9, v23, v24, v25, v26
            FOURFMAX v9, v27, v28, v29, v30
            FOURFMIN v10, v15, v16, v17, v18
            FOURFMIN v10, v19, v20, v21, v22
            FOURFMIN v10, v23, v24, v25, v26
            FOURFMIN v10, v27, v28, v29, v30
        StoreLH8:
            Float32ToBf16 v15, v16, v17, v18
            Float32ToBf16 v19, v20, v21, v22
            Float32ToBf16 v23, v24, v25, v26
            Float32ToBf16 v27, v28, v29, v30
            st1 {v15.4h, v16.4h, v17.4h, v18.4h}, [x0], #32 // 16 * sizeof(int16_t)
            st1 {v19.4h, v20.4h, v21.4h, v22.4h}, [x0], #32 // 16 * sizeof(int16_t)
            add x0, x0, x14
            st1 {v23.4h, v24.4h, v25.4h, v26.4h}, [x0], #32 // 16 * sizeof(int16_t)
            st1 {v27.4h, v28.4h, v29.4h, v30.4h}, [x0], #32 // 16 * sizeof(int16_t)
            add x0, x0, x14
            add x13, x13, x19 // weight stride
            sub x8, x8, #2
            cmp x8, #2
            bge LoopH8x8
    LH4:
    cbz x8, E8End
    LoopHRemain:
        mov x15, x1
        mov x12, x9
        cbz x5, NoBiasHRemain
        ld1 {v0.4h}, [x20]
        shll v0.4s, v0.4h, #16
        mov v2.16b, v0.16b
        uzp1 v16.2d, v0.2d, v2.2d   // bias_0, bias_1, bias_0, bias_1
        uzp2 v17.2d, v0.2d, v2.2d   // bias_2, bias_3, bias_2, bias_3
        SET_BIAS v16, v18, v20, v22
        SET_BIAS v17, v19, v21, v23
        b LoopLR
        NoBiasHRemain:
            SET_ZERO v16, v17, v18, v19
            SET_ZERO v20, v21, v22, v23
        LoopLR:
            // A [8, 4, bf16] : rn = 4  : v4 - v7
            // B [4, 4, bf16] : rn = 2  : v0 - v1
            // C [8, 4, fp32] : rn = 8  : v16 - v23
            ld1 {v4.8h, v5.8h, v6.8h, v7.8h}, [x15], x11 // A: 8 * 4 * sizeof(int16_t)
            ld1 {v0.8h, v1.8h}, [x13],  x22              // B: 4 * 4 * sizeof(int16_t)
            .inst 0x6e40ec90 // bfmmla v16.4s, v4.8h, v0.8h
            .inst 0x6e41ec91 // bfmmla v17.4s, v4.8h, v1.8h
            .inst 0x6e40ecb2 // bfmmla v18.4s, v5.8h, v0.8h
            .inst 0x6e41ecb3 // bfmmla v19.4s, v5.8h, v1.8h
            .inst 0x6e40ecd4 // bfmmla v20.4s, v6.8h, v0.8h
            .inst 0x6e41ecd5 // bfmmla v21.4s, v6.8h, v1.8h
            .inst 0x6e40ecf6 // bfmmla v22.4s, v7.8h, v0.8h
            .inst 0x6e41ecf7 // bfmmla v23.4s, v7.8h, v1.8h
            subs x12, x12, #1
            bne LoopLR
        LoopLREnd:
            uzp1 v15.2d, v16.2d, v17.2d
            uzp2 v16.2d, v16.2d, v17.2d
            uzp1 v17.2d, v18.2d, v19.2d
            uzp2 v18.2d, v18.2d, v19.2d
            uzp1 v19.2d, v20.2d, v21.2d
            uzp2 v20.2d, v20.2d, v21.2d
            uzp1 v21.2d, v22.2d, v23.2d
            uzp2 v22.2d, v22.2d, v23.2d
            cbz x5, StoreLH8x4
        PostTreatLH8x4:
            FOURFMAX v9, v15, v16, v17, v18
            FOURFMAX v9, v19, v20, v21, v22
            FOURFMIN v10, v15, v16, v17, v18
            FOURFMIN v10, v19, v20, v21, v22
        StoreLH8x4:
            Float32ToBf16 v15, v16, v17, v18
            Float32ToBf16 v19, v20, v21, v22
            st1 {v15.4h, v16.4h, v17.4h, v18.4h}, [x0], #32 // 16 * sizeof(int16_t)
            st1 {v19.4h, v20.4h, v21.4h, v22.4h}, [x0], #32 // 16 * sizeof(int16_t)
    E8End:
        sub x3, x3, #8
        cmp x3, #8
        add x0, x21, #64 // move dest address of 8 * 4 * sizeof(int16_t)
        add x1, x1, #64  // move A matrix address of 8 * 4 * sizeof(int16_t)
        bge LoopE8

E4:
cmp x3, #4
mov x20, x6
blt E2

mov x8, x10
mov x21, x0
mov x13, x2

cmp x8, #2
blt E4LH4
E4LH8:
    E4LoopH8:
        mov x15, x1
        mov x12, x9
        cbz x5, NoBiasE4
        ld1 {v0.4h, v1.4h}, [x20], #16 // 8 * sizeof(int16_t)
        shll v0.4s, v0.4h, #16
        shll v1.4s, v1.4h, #16
        mov v2.16b, v0.16b
        mov v3.16b, v1.16b
        uzp1 v16.2d, v0.2d, v2.2d   // bias_0, bias_1, bias_0, bias_1
        uzp2 v17.2d, v0.2d, v2.2d   // bias_2, bias_3, bias_2, bias_3
        uzp1 v20.2d, v1.2d, v3.2d   // bias_0, bias_1, bias_0, bias_1
        uzp2 v21.2d, v1.2d, v3.2d   // bias_2, bias_3, bias_2, bias_3
        mov v18.16b, v16.16b
        mov v19.16b, v17.16b
        mov v22.16b, v20.16b
        mov v23.16b, v21.16b
        b E4LoopL
        NoBiasE4:
            SET_ZERO v16, v17, v18, v19
            SET_ZERO v20, v21, v22, v23
        E4LoopL:
            // A [4, 4, bf16] : rn = 4  : v4 - v5
            // B [8, 4, bf16] : rn = 4  : v0 - v3
            // C [4, 8, fp32] : rn = 8  : v16 - v23
            ld1 {v4.8h, v5.8h}, [x15], x11               // A: 4 * 4 * sizeof(int16_t)
            ld1 {v0.8h, v1.8h, v2.8h, v3.8h}, [x13], x22 // B: 8 * 4 * sizeof(int16_t)
            .inst 0x6e40ec90 // bfmmla v16.4s, v4.8h, v0.8h
            .inst 0x6e41ec91 // bfmmla v17.4s, v4.8h, v1.8h
            .inst 0x6e40ecb2 // bfmmla v18.4s, v5.8h, v0.8h
            .inst 0x6e41ecb3 // bfmmla v19.4s, v5.8h, v1.8h
            .inst 0x6e42ec94 // bfmmla v20.4s, v4.8h, v2.8h
            .inst 0x6e43ec95 // bfmmla v21.4s, v4.8h, v3.8h
            .inst 0x6e42ecb6 // bfmmla v22.4s, v5.8h, v2.8h
            .inst 0x6e43ecb7 // bfmmla v23.4s, v5.8h, v3.8h
            subs x12, x12, #1
            bgt E4LoopL
        E4LoopLEnd:
            uzp1 v15.2d, v16.2d, v17.2d
            uzp2 v16.2d, v16.2d, v17.2d
            uzp1 v17.2d, v18.2d, v19.2d
            uzp2 v18.2d, v18.2d, v19.2d
            uzp1 v19.2d, v20.2d, v21.2d
            uzp2 v20.2d, v20.2d, v21.2d
            uzp1 v21.2d, v22.2d, v23.2d
            uzp2 v22.2d, v22.2d, v23.2d
            cbz x5, StoreLH4x8
        PostTreatLH4x8:
            FOURFMAX v9, v15, v16, v17, v18
            FOURFMAX v9, v19, v20, v21, v22
            FOURFMIN v10, v15, v16, v17, v18
            FOURFMIN v10, v19, v20, v21, v22
        StoreLH4x8:
            Float32ToBf16 v15, v16, v17, v18
            Float32ToBf16 v19, v20, v21, v22          
            st1 {v15.4h, v16.4h, v17.4h, v18.4h}, [x0], x7 // 16 * sizeof(int16_t)
            st1 {v19.4h, v20.4h, v21.4h, v22.4h}, [x0], x7 // 16 * sizeof(int16_t)
            add x13, x13, x19 // weight stride
            sub x8, x8, #2
            cmp x8, #2
            bge E4LoopH8
    E4LH4:
        cbz x8, E4End
        mov x15, x1
        mov x12, x9
        cbz x5, NoBiasE4R
        ld1 {v0.4h}, [x20]
        shll v0.4s, v0.4h, #16
        mov v2.16b, v0.16b
        uzp1 v16.2d, v0.2d, v2.2d   // bias_0, bias_1, bias_0, bias_1
        uzp2 v17.2d, v0.2d, v2.2d   // bias_2, bias_3, bias_2, bias_3
        mov v18.16b, v16.16b
        mov v19.16b, v17.16b
        b E4LoopLR
        NoBiasE4R:
            SET_ZERO v16, v17, v18, v19
        E4LoopLR:
            // A [4, 4, bf16] : rn = 4  : v4 - v5
            // B [4, 4, bf16] : rn = 4  : v0 - v1
            // C [4, 4, fp32] : rn = 4  : v16 - v19
            ld1 {v4.8h, v5.8h}, [x15], x11 // A: 4 * 4 * sizeof(int16_t)
            ld1 {v0.8h, v1.8h}, [x13], x22 // B: 4 * 4 * sizeof(int16_t)
            .inst 0x6e40ec90 // bfmmla v16.4s, v4.8h, v0.8h
            .inst 0x6e41ec91 // bfmmla v17.4s, v4.8h, v1.8h
            .inst 0x6e40ecb2 // bfmmla v18.4s, v5.8h, v0.8h
            .inst 0x6e41ecb3 // bfmmla v19.4s, v5.8h, v1.8h
            subs x12, x12, #1
            bgt E4LoopLR
        E4LoopLREnd:
            uzp1 v15.2d, v16.2d, v17.2d
            uzp2 v16.2d, v16.2d, v17.2d
            uzp1 v17.2d, v18.2d, v19.2d
            uzp2 v18.2d, v18.2d, v19.2d
            cbz x5, StoreLH4x4
        PostTreatLH4x4:
            FOURFMAX v9, v15, v16, v17, v18
            FOURFMIN v10, v19, v20, v21, v22
        StoreLH4x4:
            Float32ToBf16 v15, v16, v17, v18
            st1 {v15.4h, v16.4h, v17.4h, v18.4h}, [x0] // 16 * sizeof(int16_t)
    E4End:
        sub x3, x3, #4
        add x0, x21, #32 // move dest address of 4 * 4 * sizeof(int16_t)
        add x1, x1, #32 // move dest address of 4 * 4 * sizeof(int16_t)

E2:
cmp x3, #2
mov x20, x6
blt E1

mov x8, x10
mov x21, x0
mov x13, x2

cmp x8, #2
blt E2LH4
E2LH8:
    E2LoopH8:
        mov x15, x1
        mov x12, x9
        cbz x5, NoBiasE2
        ld1 {v0.4h, v1.4h}, [x20], #16
        shll v0.4s, v0.4h, #16
        shll v1.4s, v1.4h, #16
        mov v2.16b, v0.16b
        mov v3.16b, v1.16b
        uzp1 v16.2d, v0.2d, v2.2d   // bias_0, bias_1, bias_0, bias_1
        uzp2 v17.2d, v0.2d, v2.2d   // bias_2, bias_3, bias_2, bias_3
        uzp1 v18.2d, v1.2d, v3.2d   // bias_0, bias_1, bias_0, bias_1
        uzp2 v19.2d, v1.2d, v3.2d   // bias_2, bias_3, bias_2, bias_3
        b E2LoopL
        NoBiasE2:
            SET_ZERO v16, v17, v18, v19
        E2LoopL:
            // A [2, 4, bf16] : rn = 1  : v4
            // B [8, 4, bf16] : rn = 2  : v0 - v3
            // C [2, 8, fp32] : rn = 4  : v16 - v19
            ld1 {v4.8h}, [x15], x11                      // A: 2 * 4 * sizeof(int16_t)
            ld1 {v0.8h, v1.8h, v2.8h, v3.8h}, [x13], x22 // B: 8 * 4 * sizeof(int16_t)
            .inst 0x6e40ec90 // bfmmla v16.4s, v4.8h, v0.8h
            .inst 0x6e41ec91 // bfmmla v17.4s, v4.8h, v1.8h
            .inst 0x6e42ec92 // bfmmla v18.4s, v4.8h, v2.8h
            .inst 0x6e43ec93 // bfmmla v19.4s, v4.8h, v3.8h
            subs x12, x12, #1
            bgt E2LoopL
        E2LoopLEnd:
            uzp1 v15.2d, v16.2d, v17.2d
            uzp2 v16.2d, v16.2d, v17.2d
            uzp1 v17.2d, v18.2d, v19.2d
            uzp2 v18.2d, v18.2d, v19.2d
            cbz x5, StoreLH2x8
        PostTreatLH2x8:
            FOURFMAX v9, v15, v16, v17, v18
            FOURFMIN v10, v15, v16, v17, v18
        StoreLH2x8:
            Float32ToBf16 v15, v16, v17, v18
            st1 {v15.4h, v16.4h}, [x0], x7 // 8 * sizeof(int16_t)
            st1 {v17.4h, v18.4h}, [x0], x7 // 8 * sizeof(int16_t)
            add x13, x13, x19 // weight stride
            sub x8, x8, #2
            cmp x8, #2
            bge E2LoopH8
    E2LH4:
        cbz x8, E2End
        mov x15, x1
        mov x12, x9
        cbz x5, NoBiasE2R
        ld1 {v0.4h}, [x20]
        shll v0.4s, v0.4h, #16
        mov v2.16b, v0.16b
        uzp1 v16.2d, v0.2d, v2.2d   // bias_0, bias_1, bias_0, bias_1
        uzp2 v17.2d, v0.2d, v2.2d   // bias_2, bias_3, bias_2, bias_3
        b E2LoopLR
        NoBiasE2R:
            movi v16.4s, #0
            movi v17.4s, #0
        E2LoopLR:
            // A [2, 4, bf16] : rn = 1  : v4
            // B [4, 4, bf16] : rn = 2  : v0 - v1
            // C [2, 4, fp32] : rn = 2  : v16 - v17
            ld1 {v4.8h}, [x15], x11        // A: 2 * 4 * sizeof(int16_t)
            ld1 {v0.8h, v1.8h}, [x13], x22 // B: 4 * 4 * sizeof(int16_t) 
            .inst 0x6e40ec90 // bfmmla v16.4s, v4.8h, v0.8h
            .inst 0x6e41ec91 // bfmmla v17.4s, v4.8h, v1.8h
            subs x12, x12, #1
            bgt E2LoopLR
        E2LoopLREnd:
            uzp1 v15.2d, v16.2d, v17.2d
            uzp2 v16.2d, v16.2d, v17.2d
            cbz x5, StoreLH2x4
        PostTreatLH2x4:
            fmax v15.4s, v15.4s, v9.4s
            fmax v16.4s, v16.4s, v9.4s
            fmin v15.4s, v15.4s, v10.4s
            fmin v16.4s, v16.4s, v10.4s
        StoreLH2x4:
            shrn v15.4h, v15.4s, #16
            shrn v16.4h, v16.4s, #16
            st1 {v15.4h, v16.4h}, [x0] // 8 * sizeof(int16_t)
    E2End:
        sub x3, x3, #2
        add x0, x21, #16 // move dest address of 2 * 4 * sizeof(int16_t)
        add x1, x1, #16 // move dest address of 2 * 4 * sizeof(int16_t)

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
        mov x12, x9
        cbz x5, NoBiasE1
        ld1 {v0.4h, v1.4h}, [x20], #16
        shll v0.4s, v0.4h, #16
        shll v1.4s, v1.4h, #16
        mov v2.16b, v0.16b
        mov v3.16b, v1.16b
        uzp1 v16.2d, v0.2d, v2.2d   // bias_0, bias_1, bias_0, bias_1
        uzp2 v17.2d, v0.2d, v2.2d   // bias_2, bias_3, bias_2, bias_3
        uzp1 v18.2d, v1.2d, v3.2d   // bias_0, bias_1, bias_0, bias_1
        uzp2 v19.2d, v1.2d, v3.2d   // bias_2, bias_3, bias_2, bias_3
        b E1LoopL
        NoBiasE1:
            SET_ZERO v16, v17, v18, v19
        E1LoopL:
            // A [1, 4, bf16] : rn = 1  : v4
            // B [8, 4, bf16] : rn = 4  : v0 - v3
            // C [1, 8, fp32] : rn = 4  : v16 - v19
            ld1 {v4.4h}, [x15], x11                      // A: 1 * 4 * sizeof(int16_t)
            ld1 {v0.8h, v1.8h, v2.8h, v3.8h}, [x13], x22 // B: 8 * 4 * sizeof(int16_t)
            .inst 0x6e40ec90 // bfmmla v16.4s, v4.8h, v0.8h
            .inst 0x6e41ec91 // bfmmla v17.4s, v4.8h, v1.8h
            .inst 0x6e42ec92 // bfmmla v18.4s, v4.8h, v2.8h
            .inst 0x6e43ec93 // bfmmla v19.4s, v4.8h, v3.8h
            subs x12, x12, #1
            bgt E1LoopL
        E1LoopLEnd:
            // v16-v19: [r0, r1, 0, 0]
            uzp1 v15.2d, v16.2d, v17.2d
            uzp1 v16.2d, v18.2d, v19.2d
            cbz x5, StoreLH1x8
        PostTreatLH1x8:
            fmax v15.4s, v15.4s, v9.4s
            fmax v16.4s, v16.4s, v9.4s
            fmin v15.4s, v15.4s, v10.4s
            fmin v16.4s, v16.4s, v10.4s
        StoreLH1x8:
            shrn v15.4h, v15.4s, #16
            shrn v16.4h, v16.4s, #16
            st1 {v15.4h}, [x0], x7
            st1 {v16.4h}, [x0], x7
            add x13, x13, x19
            sub x8, x8, #2
            cmp x8, #2
            bge E1LoopH8

    E1LH4:
    cbz x8, E1End
    mov x15, x1
    mov x12, x9
    cbz x5, NoBiasE1R
    ld1 {v0.4h}, [x20]
    shll v0.4s, v0.4h, #16
    mov v2.16b, v0.16b
    uzp1 v16.2d, v0.2d, v2.2d   // bias_0, bias_1, bias_0, bias_1
    uzp2 v17.2d, v0.2d, v2.2d   // bias_2, bias_3, bias_2, bias_3
    b E1LoopLR
    NoBiasE1R:
        movi v16.4s, #0
        movi v17.4s, #0
    E1LoopLR:
        // A [1, 4, bf16] : rn = 1  : v4
        // B [4, 4, bf16] : rn = 2  : v0 - v1
        // C [1, 8, fp32] : rn = 4  : v16 - v17
        ld1 {v4.4h}, [x15], x11        // A: 1 * 4 * sizeof(int16_t)
        ld1 {v0.8h, v1.8h}, [x13], x22 // B: 4 * 4 * sizeof(int16_t)
        .inst 0x6e40ec90 // bfmmla v16.4s, v4.8h, v0.8h
        .inst 0x6e41ec91 // bfmmla v17.4s, v4.8h, v1.8h
        subs x12, x12, #1
        bgt E1LoopLR
    E1LoopLREnd:
        uzp1 v15.2d, v16.2d, v17.2d
        cbz x5, StoreLH1x4
    PostTreatLH1x4:
        fmax v15.4s, v15.4s, v9.4s
        fmin v15.4s, v15.4s, v10.4s
    StoreLH1x4:
        shrn v15.4h, v15.4s, #16
        st1 {v15.4h}, [x0]
    E1End:
        subs x3, x3, #1
        add x0, x21, #8
        add x1, x1, #8
        bne LoopE1
End:
ldr x19, [sp, #0]
ldr x20, [sp, #8]
ldr x21, [sp, #16]
ldr x22, [sp, #24]
add sp, sp, #64

ret
#endif
