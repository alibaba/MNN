//
//  MNNGemmFP16C8_UNIT.S
//  MNN
//
//  Created by MNN on 2020/01/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__
#include "MNNAsmGlobal.h"

.text
.align 5
asm_function MNNGemmFP16C8_UNIT
// void MNNGemmFP16C8_UNIT(FLOAT16* dst, const FLOAT16* src, const FLOAT16* weight, 
//  const FLOAT16* bias, size_t src_loop, size_t dst_step, size_t dst_loop, size_t relu, 
//  size_t relu6, size_t realDstCount)

// Auto:
//      x0:dst, x1:src, x2:weight, x3:bias, x4:src_loop
//      x5:dst_step, x6:dst_loop, x7:relu
// load from sp:
//      x8:relu6, x9:realDstCount

ldr x8, [sp, #0]
ldr x9, [sp, #8]

mov x10, #8
mul x4, x4, x10 // x4 * 8 = (inputChannelUnit * kernelCount) * 8

cmp x9, #4
ble TILE_4

TILE_8:
cmp x6, #2
blt LoopDz_TILE_8_ONE_OC

LoopDz_TILE_8_DOUBLE_OC:
    ldr q6, [x3], #16 // bias
    mov x11, x1
    mov x12, x4
    ldr q7, [x3], #16 // bias + 8

    mov v16.16b, v6.16b
    mov v17.16b, v6.16b
    mov v18.16b, v6.16b
    mov v19.16b, v6.16b
    mov v20.16b, v6.16b
    mov v21.16b, v6.16b
    mov v22.16b, v6.16b
    mov v23.16b, v6.16b
    mov v24.16b, v7.16b
    mov v25.16b, v7.16b
    mov v26.16b, v7.16b
    mov v27.16b, v7.16b
    mov v28.16b, v7.16b
    mov v29.16b, v7.16b
    mov v30.16b, v7.16b
    mov v31.16b, v7.16b

    LoopSz_TILE_8_DOUBLE_OC:
        ldr q0, [x2] // weight
        ldr q4, [x11] // input
        fmla v16.8h, v0.8h, v4.h[0]
        fmla v17.8h, v0.8h, v4.h[1]
        fmla v18.8h, v0.8h, v4.h[2]
        fmla v19.8h, v0.8h, v4.h[3]
        ldr q1, [x2, #16] // weight
        fmla v20.8h, v0.8h, v4.h[4]
        fmla v21.8h, v0.8h, v4.h[5]
        fmla v22.8h, v0.8h, v4.h[6]
        fmla v23.8h, v0.8h, v4.h[7]

        ldr q2, [x2, #32] // weight
        fmla v24.8h, v1.8h, v4.h[0]
        fmla v25.8h, v1.8h, v4.h[1]
        fmla v26.8h, v1.8h, v4.h[2]
        fmla v27.8h, v1.8h, v4.h[3]
        ldr q5, [x11, #16] // input
        fmla v28.8h, v1.8h, v4.h[4]
        fmla v29.8h, v1.8h, v4.h[5]
        fmla v30.8h, v1.8h, v4.h[6]
        fmla v31.8h, v1.8h, v4.h[7]

        fmla v16.8h, v2.8h, v5.h[0]
        fmla v17.8h, v2.8h, v5.h[1]
        ldr q3, [x2, #48] // weight
        fmla v18.8h, v2.8h, v5.h[2]
        fmla v19.8h, v2.8h, v5.h[3]
        add x11, x11, #32
        fmla v20.8h, v2.8h, v5.h[4]
        fmla v21.8h, v2.8h, v5.h[5]
        fmla v22.8h, v2.8h, v5.h[6]
        fmla v23.8h, v2.8h, v5.h[7]

        fmla v24.8h, v3.8h, v5.h[0]
        fmla v25.8h, v3.8h, v5.h[1]
        subs x12, x12, #2
        fmla v26.8h, v3.8h, v5.h[2]
        fmla v27.8h, v3.8h, v5.h[3]
        add x2, x2, #64
        fmla v28.8h, v3.8h, v5.h[4]
        fmla v29.8h, v3.8h, v5.h[5]
        fmla v30.8h, v3.8h, v5.h[6]
        fmla v31.8h, v3.8h, v5.h[7]
        bne LoopSz_TILE_8_DOUBLE_OC
    
    cbz x7, RELU6_DOUBLE_OC
    eor v0.16b, v0.16b, v0.16b
    fmax v16.8h, v16.8h, v0.8h
    fmax v17.8h, v17.8h, v0.8h
    fmax v18.8h, v18.8h, v0.8h
    fmax v19.8h, v19.8h, v0.8h
    fmax v20.8h, v20.8h, v0.8h
    fmax v21.8h, v21.8h, v0.8h
    fmax v22.8h, v22.8h, v0.8h
    fmax v23.8h, v23.8h, v0.8h
    fmax v24.8h, v24.8h, v0.8h
    fmax v25.8h, v25.8h, v0.8h
    fmax v26.8h, v26.8h, v0.8h
    fmax v27.8h, v27.8h, v0.8h
    fmax v28.8h, v28.8h, v0.8h
    fmax v29.8h, v29.8h, v0.8h
    fmax v30.8h, v30.8h, v0.8h
    fmax v31.8h, v31.8h, v0.8h
    
    RELU6_DOUBLE_OC:
    cbz x8, STORE_TILE_8_DOUBLE_OC
    eor v0.16b, v0.16b, v0.16b
    movi v1.8h, #0x46, lsl #8 // 0x4600 -> fp16(6.0)
    fmax v16.8h, v16.8h, v0.8h
    fmax v17.8h, v17.8h, v0.8h
    fmax v18.8h, v18.8h, v0.8h
    fmax v19.8h, v19.8h, v0.8h
    fmax v20.8h, v20.8h, v0.8h
    fmax v21.8h, v21.8h, v0.8h
    fmax v22.8h, v22.8h, v0.8h
    fmax v23.8h, v23.8h, v0.8h
    fmax v24.8h, v24.8h, v0.8h
    fmax v25.8h, v25.8h, v0.8h
    fmax v26.8h, v26.8h, v0.8h
    fmax v27.8h, v27.8h, v0.8h
    fmax v28.8h, v28.8h, v0.8h
    fmax v29.8h, v29.8h, v0.8h
    fmax v30.8h, v30.8h, v0.8h
    fmax v31.8h, v31.8h, v0.8h
    fmin v16.8h, v16.8h, v1.8h
    fmin v17.8h, v17.8h, v1.8h
    fmin v18.8h, v18.8h, v1.8h
    fmin v19.8h, v19.8h, v1.8h
    fmin v20.8h, v20.8h, v1.8h
    fmin v21.8h, v21.8h, v1.8h
    fmin v22.8h, v22.8h, v1.8h
    fmin v23.8h, v23.8h, v1.8h
    fmin v24.8h, v24.8h, v1.8h
    fmin v25.8h, v25.8h, v1.8h
    fmin v26.8h, v26.8h, v1.8h
    fmin v27.8h, v27.8h, v1.8h
    fmin v28.8h, v28.8h, v1.8h
    fmin v29.8h, v29.8h, v1.8h
    fmin v30.8h, v30.8h, v1.8h
    fmin v31.8h, v31.8h, v1.8h

    STORE_TILE_8_DOUBLE_OC:
    str q16, [x0]
    str q17, [x0, #16]
    str q18, [x0, #32]
    str q19, [x0, #48]
    str q20, [x0, #64]
    str q21, [x0, #80]
    str q22, [x0, #96]
    str q23, [x0, #112]
    add x0, x0, x5
    str q24, [x0]
    str q25, [x0, #16]
    str q26, [x0, #32]
    str q27, [x0, #48]
    str q28, [x0, #64]
    str q29, [x0, #80]
    str q30, [x0, #96]
    str q31, [x0, #112]
    sub x6, x6, #2
    cmp x6, #2
    add x0, x0, x5
    BGE LoopDz_TILE_8_DOUBLE_OC


LoopDz_TILE_8_ONE_OC:    
cmp x6, #0
beq REAL_END

ldr q6, [x3] // bias
mov x11, x1
mov x12, x4

mov v24.16b, v6.16b
mov v25.16b, v6.16b
mov v26.16b, v6.16b
mov v27.16b, v6.16b
mov v28.16b, v6.16b
mov v29.16b, v6.16b
mov v30.16b, v6.16b
mov v31.16b, v6.16b

LoopSz_TILE_8_ONE_OC:
    ldr q0, [x2] // weight
    ldr q4, [x11] // input
    fmla v24.8h, v0.8h, v4.h[0]
    fmla v25.8h, v0.8h, v4.h[1]
    ldr q2, [x2, #16] // weight
    fmla v26.8h, v0.8h, v4.h[2]
    fmla v27.8h, v0.8h, v4.h[3]
    ldr q5, [x11, #16] // input
    fmla v28.8h, v0.8h, v4.h[4]
    fmla v29.8h, v0.8h, v4.h[5]
    fmla v30.8h, v0.8h, v4.h[6]
    fmla v31.8h, v0.8h, v4.h[7]

    fmla v24.8h, v2.8h, v5.h[0]
    fmla v25.8h, v2.8h, v5.h[1]
    subs x12, x12, #2
    fmla v26.8h, v2.8h, v5.h[2]
    fmla v27.8h, v2.8h, v5.h[3]
    add x2, x2, #32
    fmla v28.8h, v2.8h, v5.h[4]
    fmla v29.8h, v2.8h, v5.h[5]
    add x11, x11, #32
    fmla v30.8h, v2.8h, v5.h[6]
    fmla v31.8h, v2.8h, v5.h[7]
    bne LoopSz_TILE_8_ONE_OC

cbz x7, RELU6_ONE_OC
eor v0.16b, v0.16b, v0.16b
fmax v24.8h, v24.8h, v0.8h
fmax v25.8h, v25.8h, v0.8h
fmax v26.8h, v26.8h, v0.8h
fmax v27.8h, v27.8h, v0.8h
fmax v28.8h, v28.8h, v0.8h
fmax v29.8h, v29.8h, v0.8h
fmax v30.8h, v30.8h, v0.8h
fmax v31.8h, v31.8h, v0.8h

RELU6_ONE_OC:
cbz x8, STORE_TILE_8_ONE_OC
eor v0.16b, v0.16b, v0.16b
movi v1.8h, #0x46, lsl #8 // 0x4600 -> fp16(6.0)
fmax v24.8h, v24.8h, v0.8h
fmax v25.8h, v25.8h, v0.8h
fmax v26.8h, v26.8h, v0.8h
fmax v27.8h, v27.8h, v0.8h
fmax v28.8h, v28.8h, v0.8h
fmax v29.8h, v29.8h, v0.8h
fmax v30.8h, v30.8h, v0.8h
fmax v31.8h, v31.8h, v0.8h

fmin v24.8h, v24.8h, v1.8h
fmin v25.8h, v25.8h, v1.8h
fmin v26.8h, v26.8h, v1.8h
fmin v27.8h, v27.8h, v1.8h
fmin v28.8h, v28.8h, v1.8h
fmin v29.8h, v29.8h, v1.8h
fmin v30.8h, v30.8h, v1.8h
fmin v31.8h, v31.8h, v1.8h

STORE_TILE_8_ONE_OC:
str q24, [x0]
str q25, [x0, #16]
str q26, [x0, #32]
str q27, [x0, #48]
str q28, [x0, #64]
str q29, [x0, #80]
str q30, [x0, #96]
str q31, [x0, #112]

b REAL_END

# remain tile is (0, 4]
TILE_4:
cmp x6, #2
blt LoopDz_TILE_4_ONE_OC

LoopDz_TILE_4_DOUBLE_OC:
    ldr q6, [x3], #16 // bias
    mov x11, x1
    mov x12, x4
    ldr q7, [x3], #16 // bias + 8

    mov v24.16b, v6.16b
    mov v25.16b, v6.16b
    mov v26.16b, v6.16b
    mov v27.16b, v6.16b

    mov v28.16b, v7.16b
    mov v29.16b, v7.16b
    mov v30.16b, v7.16b
    mov v31.16b, v7.16b

    LoopSz_TILE_4_DOUBLE_OC:
        ldr q0, [x2] // weight
        ldr d4, [x11] // input
        fmla v24.8h, v0.8h, v4.h[0]
        fmla v25.8h, v0.8h, v4.h[1]
        ldr q1, [x2, #16] // weight
        fmla v26.8h, v0.8h, v4.h[2]
        fmla v27.8h, v0.8h, v4.h[3]
        ldr d5, [x11, #8] // input
        fmla v28.8h, v1.8h, v4.h[0]
        fmla v29.8h, v1.8h, v4.h[1]
        ldr q2, [x2, #32] // weight
        fmla v30.8h, v1.8h, v4.h[2]
        fmla v31.8h, v1.8h, v4.h[3]
        
        ldr q3, [x2, #48] // weight
        fmla v24.8h, v2.8h, v5.h[0]
        fmla v25.8h, v2.8h, v5.h[1]
        subs x12, x12, #2
        fmla v26.8h, v2.8h, v5.h[2]
        fmla v27.8h, v2.8h, v5.h[3]
        add x2, x2, #64
        fmla v28.8h, v3.8h, v5.h[0]
        fmla v29.8h, v3.8h, v5.h[1]
        add x11, x11, #16
        fmla v30.8h, v3.8h, v5.h[2]
        fmla v31.8h, v3.8h, v5.h[3]
        bne LoopSz_TILE_4_DOUBLE_OC
    
    cbz x7, RELU6_TILE_4_DOUBLE_OC
    eor v0.16b, v0.16b, v0.16b
    fmax v24.8h, v24.8h, v0.8h
    fmax v25.8h, v25.8h, v0.8h
    fmax v26.8h, v26.8h, v0.8h
    fmax v27.8h, v27.8h, v0.8h
    fmax v28.8h, v28.8h, v0.8h
    fmax v29.8h, v29.8h, v0.8h
    fmax v30.8h, v30.8h, v0.8h
    fmax v31.8h, v31.8h, v0.8h
    
    RELU6_TILE_4_DOUBLE_OC:
    cbz x8, STORE_TILE_4_DOUBLE_OC
    eor v0.16b, v0.16b, v0.16b
    movi v1.8h, #0x46, lsl #8 // 0x4600 -> fp16(6.0)
    fmax v24.8h, v24.8h, v0.8h
    fmax v25.8h, v25.8h, v0.8h
    fmax v26.8h, v26.8h, v0.8h
    fmax v27.8h, v27.8h, v0.8h
    fmax v28.8h, v28.8h, v0.8h
    fmax v29.8h, v29.8h, v0.8h
    fmax v30.8h, v30.8h, v0.8h
    fmax v31.8h, v31.8h, v0.8h
    fmin v24.8h, v24.8h, v1.8h
    fmin v25.8h, v25.8h, v1.8h
    fmin v26.8h, v26.8h, v1.8h
    fmin v27.8h, v27.8h, v1.8h
    fmin v28.8h, v28.8h, v1.8h
    fmin v29.8h, v29.8h, v1.8h
    fmin v30.8h, v30.8h, v1.8h
    fmin v31.8h, v31.8h, v1.8h

    STORE_TILE_4_DOUBLE_OC:
    str q24, [x0]
    str q25, [x0, #16]
    str q26, [x0, #32]
    str q27, [x0, #48]
    add x0, x0, x5
    sub x6, x6, #2
    str q28, [x0]
    str q29, [x0, #16]
    str q30, [x0, #32]
    str q31, [x0, #48]
    cmp x6, #2
    add x0, x0, x5
    BGE LoopDz_TILE_4_DOUBLE_OC


LoopDz_TILE_4_ONE_OC:    
cmp x6, #0
beq REAL_END

ldr q6, [x3] // bias
mov x11, x1
mov x12, x4

mov v28.16b, v6.16b
mov v29.16b, v6.16b
mov v30.16b, v6.16b
mov v31.16b, v6.16b

LoopSz_TILE_4_ONE_OC:
    ldr q0, [x2] // weight
    ldr d4, [x11] // input
    ldr q2, [x2, #16] // weight
    ldr d5, [x11, #8] // input
    fmla v28.8h, v0.8h, v4.h[0]
    fmla v29.8h, v0.8h, v4.h[1]
    subs x12, x12, #2
    fmla v30.8h, v0.8h, v4.h[2]
    fmla v31.8h, v0.8h, v4.h[3]
    add x2, x2, #32
    fmla v28.8h, v2.8h, v5.h[0]
    fmla v29.8h, v2.8h, v5.h[1]
    add x11, x11, #16
    fmla v30.8h, v2.8h, v5.h[2]
    fmla v31.8h, v2.8h, v5.h[3]
    bne LoopSz_TILE_4_ONE_OC

cbz x7, RELU6_TILE_4_ONE_OC
eor v0.16b, v0.16b, v0.16b
fmax v28.8h, v28.8h, v0.8h
fmax v29.8h, v29.8h, v0.8h
fmax v30.8h, v30.8h, v0.8h
fmax v31.8h, v31.8h, v0.8h

RELU6_TILE_4_ONE_OC:
cbz x8, STORE_TILE_4_ONE_OC
eor v0.16b, v0.16b, v0.16b
movi v1.8h, #0x46, lsl #8 // 0x4600 -> fp16(6.0)
fmax v28.8h, v28.8h, v0.8h
fmax v29.8h, v29.8h, v0.8h
fmax v30.8h, v30.8h, v0.8h
fmax v31.8h, v31.8h, v0.8h

fmin v28.8h, v28.8h, v1.8h
fmin v29.8h, v29.8h, v1.8h
fmin v30.8h, v30.8h, v1.8h
fmin v31.8h, v31.8h, v1.8h

STORE_TILE_4_ONE_OC:
str q28, [x0]
str q29, [x0, #16]
str q30, [x0, #32]
str q31, [x0, #48]

REAL_END:

ret

#endif