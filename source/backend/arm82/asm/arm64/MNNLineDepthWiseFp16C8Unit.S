//
//  MNNLineDepthWiseFp16C8Unit.S
//  MNN
//
//  Created by MNN on 2019/01/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNLineDepthWiseFp16C8Unit
// void MNNLineDepthWiseFp16C8Unit(FLOAT16* dst, const FLOAT16* src, 
//    const FLOAT16* weight, const FLOAT16* bias_z, size_t width, size_t src_w_step, 
//    size_t fw, size_t fh, size_t dilateX_step, size_t dilateY_step, size_t relu, size_t relu6)

// Auto:
//  x0: dst, x1:src, x2:weight, x3:bias_z, x4:width
//  x5:src_w_step, x6:fw, x7:fh
// Load:
//  x8:dilateX_step, x9:dilateY_step, x10:relu, x11:relu6

ldr x8, [sp, #0]
ldr x9, [sp, #8]
ldr x10, [sp, #16]
ldr x11, [sp, #24]

mov x12, #2 // sizeof(fp16) == 2
mul x5, x12, x5
mul x8, x12, x8
mul x9, x12, x9

// fw * dilateX_step
mul x12, x6, x8
sub x9, x9, x12
ldr q0, [x3] // bias

L8:
cmp x4, #7
ble L4

mov x12, #8
mul x12, x5, x12

LOOP_TILE_8:
    mov v16.16b, v0.16b
    mov v17.16b, v0.16b
    mov v18.16b, v0.16b
    mov v19.16b, v0.16b
    mov v20.16b, v0.16b
    mov v21.16b, v0.16b
    mov v22.16b, v0.16b
    mov v23.16b, v0.16b
    // x7 -> kh
    mov x13, x7
    // keep x1
    mov x3, x1
    // keep x2
    mov x15, x2
    LOOP_TILE_8_KH:
        // x6 -> kw
        mov x14, x6
        LOOP_TILE_8_KW:
            ldr q1, [x2], #16 // weight
            ld1 {v24.16b}, [x1], x5 // input
            ld1 {v25.16b}, [x1], x5 // input
            ld1 {v26.16b}, [x1], x5
            ld1 {v27.16b}, [x1], x5
            fmla v16.8h, v1.8h, v24.8h
            fmla v17.8h, v1.8h, v25.8h
            subs x14, x14, #1
            fmla v18.8h, v1.8h, v26.8h
            fmla v19.8h, v1.8h, v27.8h
            ld1 {v28.16b}, [x1], x5
            ld1 {v29.16b}, [x1], x5
            ld1 {v30.16b}, [x1], x5
            ld1 {v31.16b}, [x1], x5
            fmla v20.8h, v1.8h, v28.8h
            fmla v21.8h, v1.8h, v29.8h
            sub x1, x1, x12
            fmla v22.8h, v1.8h, v30.8h
            fmla v23.8h, v1.8h, v31.8h
            add x1, x1, x8
            bne LOOP_TILE_8_KW
        subs x13, x13, #1
        add x1, x1, x9
        bne LOOP_TILE_8_KH

    sub x4, x4, #8
    cbz x10, LOOP_TILE_8_RELU6
    eor v6.16b, v6.16b, v6.16b
    fmax v16.8h, v16.8h, v6.8h
    fmax v17.8h, v17.8h, v6.8h
    fmax v18.8h, v18.8h, v6.8h
    fmax v19.8h, v19.8h, v6.8h
    fmax v20.8h, v20.8h, v6.8h
    fmax v21.8h, v21.8h, v6.8h
    fmax v22.8h, v22.8h, v6.8h
    fmax v23.8h, v23.8h, v6.8h

    LOOP_TILE_8_RELU6:
    cbz x11, STORE_TILE_8
    eor v6.16b, v6.16b, v6.16b
    movi v7.8h, #0x46, lsl #8 // 0x4600 -> fp16(6.0)
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

    STORE_TILE_8:
    mov x2, x15
    str q16, [x0], #16
    str q17, [x0], #16
    str q18, [x0], #16
    str q19, [x0], #16
    add x1, x12, x3
    cmp x4, #8
    str q20, [x0], #16
    str q21, [x0], #16
    str q22, [x0], #16
    str q23, [x0], #16
    bge LOOP_TILE_8

L4:
cmp x4, #3
ble L1

mov x12, #4
mul x12, x5, x12

LOOP_TILE_4:
    mov v16.16b, v0.16b
    mov v17.16b, v0.16b
    mov v18.16b, v0.16b
    mov v19.16b, v0.16b
    // x7 -> kh
    mov x13, x7
    mov x3, x1
    mov x15, x2
    LOOP_TILE_4_KH:
        // x6 -> kw
        mov x14, x6
        LOOP_TILE_4_KW:
            ldr q1, [x2], #16 // weight
            ld1 {v24.16b}, [x1], x5 // input
            ld1 {v25.16b}, [x1], x5 // input
            ld1 {v26.16b}, [x1], x5
            ld1 {v27.16b}, [x1], x5
            fmla v16.8h, v1.8h, v24.8h
            fmla v17.8h, v1.8h, v25.8h
            subs x14, x14, #1
            fmla v18.8h, v1.8h, v26.8h
            fmla v19.8h, v1.8h, v27.8h
            sub x1, x1, x12
            add x1, x1, x8
            bne LOOP_TILE_4_KW
        subs x13, x13, #1
        add x1, x1, x9
        bne LOOP_TILE_4_KH

    sub x4, x4, #4
    cbz x10, LOOP_TILE_4_RELU6
    eor v6.16b, v6.16b, v6.16b
    fmax v16.8h, v16.8h, v6.8h
    fmax v17.8h, v17.8h, v6.8h
    fmax v18.8h, v18.8h, v6.8h
    fmax v19.8h, v19.8h, v6.8h

    LOOP_TILE_4_RELU6:
    cbz x11, STORE_TILE_4
    eor v6.16b, v6.16b, v6.16b
    movi v7.8h, #0x46, lsl #8 // 0x4600 -> fp16(6.0)
    fmax v16.8h, v16.8h, v6.8h
    fmax v17.8h, v17.8h, v6.8h
    fmax v18.8h, v18.8h, v6.8h
    fmax v19.8h, v19.8h, v6.8h

    fmin v16.8h, v16.8h, v7.8h
    fmin v17.8h, v17.8h, v7.8h
    fmin v18.8h, v18.8h, v7.8h
    fmin v19.8h, v19.8h, v7.8h

    STORE_TILE_4:
    mov x2, x15 
    str q16, [x0], #16
    str q17, [x0], #16
    str q18, [x0], #16
    str q19, [x0], #16
    add x1, x12, x3
    cmp x4, #4
    bge LOOP_TILE_4

L1:
cmp x4, #0
beq REAL_END

LOOP_TILE_1:
    mov v16.16b, v0.16b
    // x7 -> kh
    mov x13, x7
    mov x3, x1
    mov x15, x2
    LOOP_TILE_1_KH:
        // x6 -> kw
        mov x14, x6
        LOOP_TILE_1_KW:
            ld1 {v1.16b}, [x2], #16 // weight
            ld1 {v24.16b}, [x1], x8 // input
            fmla v16.8h, v1.8h, v24.8h
            subs x14, x14, #1
            bne LOOP_TILE_1_KW
        subs x13, x13, #1
        add x1, x1, x9
        bne LOOP_TILE_1_KH
    
    cbz x10, LOOP_TILE_1_RELU6
    eor v6.16b, v6.16b, v6.16b
    fmax v16.8h, v16.8h, v6.8h

    LOOP_TILE_1_RELU6:
    cbz x11, STORE_TILE_1
    eor v6.16b, v6.16b, v6.16b
    movi v7.8h, #0x46, lsl #8 // 0x4600 -> fp16(6.0)
    fmax v16.8h, v16.8h, v6.8h
    fmin v16.8h, v16.8h, v7.8h

    STORE_TILE_1:
    subs x4, x4, #1
    mov x2, x15
    str q16, [x0], #16
    add x1, x5, x3
    bne LOOP_TILE_1

REAL_END:
ret
#endif
