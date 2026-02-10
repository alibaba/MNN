//
//  MNNLocalMinMaxFP16_Pack4.S
//
//  Created by MNN on 2023/10/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"
.text
.align 5

.macro Max4 d0, d1, d2, d3, z0, z1, z2, z3
    fmax \d0\().4h, \d0\().4h, \z0\().4h
    fmax \d1\().4h, \d1\().4h, \z1\().4h
    fmax \d2\().4h, \d2\().4h, \z2\().4h
    fmax \d3\().4h, \d3\().4h, \z3\().4h
.endm

.macro Min4 d0, d1, d2, d3, z0, z1, z2, z3
    fmin \d0\().4h, \d0\().4h, \z0\().4h
    fmin \d1\().4h, \d1\().4h, \z1\().4h
    fmin \d2\().4h, \d2\().4h, \z2\().4h
    fmin \d3\().4h, \d3\().4h, \z3\().4h
.endm

.macro ReduceMax s0, s1, s2, s3, z0
    fmaxp \s0\().4h, \s0\().4h, \s1\().4h // 0 0 1 1
    fmaxp \s2\().4h, \s2\().4h, \s3\().4h // 2 2 3 3
    fmaxp \z0\().4h, \s0\().4h, \s2\().4h // 0 1 2 3
.endm

.macro ReduceMin s0, s1, s2, s3, z0
    fminp \s0\().4h, \s0\().4h, \s1\().4h // 0 0 1 1
    fminp \s2\().4h, \s2\().4h, \s3\().4h // 2 2 3 3
    fminp \z0\().4h, \s0\().4h, \s2\().4h // 0 1 2 3
.endm

.macro L4Copy s0, s1, s2, s3, z0, z1, z2, z3
    mov \s0\().8b, \z0\().8b
    mov \s1\().8b, \z1\().8b
    mov \s2\().8b, \z2\().8b
    mov \s3\().8b, \z3\().8b
.endm

//void MNNLocalMinMaxFP16_Pack4(float* dstMin, float* dstMax, const float* source, size_t blockNum, size_t blockLU, size_t EP, size_t LP, size_t loadDstBuffer)
asm_function MNNLocalMinMaxFP16_Pack4

// x0: dstMin, x1:dstMax, x2:source, x3:blockNum, x4: blockLU, x5: EP, x6: LP=4, x7: loadDstBuffer
// input shape: [blocknum, blockLU, EP, LP]
stp d14, d15, [sp, #(-16 * 4)]!
stp d12, d13, [sp, #(16 * 1)]
stp d10, d11, [sp, #(16 * 2)]
stp d8,  d9,  [sp, #(16 * 3)]

lsl x6, x5, #3 // src_step = batch * 4 * sizeof(float16_t) = batch << 3
mul x13, x5, x4       // blockLU * EP * LP * sizeof(float16_t)
lsl x13, x13, #3
mov x9, x5
mov x10, x4


Loop_BlockNum:
sub x3, x3, #1 // blocknum--
mov x5, x9     // EP
mov x12, x2    // block's source

TILE_8:
cmp x5, #8
blt TILE_1
mov x4, x10  // blockLU
mov x11, x2  // src
sub x8, x6, #32 // src_step

ld1 {v0.4h, v1.4h, v2.4h, v3.4h}, [x11], #32
ld1 {v4.4h, v5.4h, v6.4h, v7.4h}, [x11], x8
L4Copy v8, v9, v10, v11, v0, v1, v2, v3
L4Copy v12, v13, v14, v15, v4, v5, v6, v7
subs x4, x4, #1
beq Tile8End

LoopSz_8:
ld1 {v16.4h, v17.4h, v18.4h, v19.4h}, [x11], #32
ld1 {v20.4h, v21.4h, v22.4h, v23.4h}, [x11], x8

Max4 v0, v1, v2, v3, v16, v17, v18, v19
Max4 v4, v5, v6, v7, v20, v21, v22, v23
Min4 v8, v9, v10, v11, v16, v17, v18, v19
Min4 v12, v13, v14, v15, v20, v21, v22, v23

subs x4, x4, #1
bne LoopSz_8

Tile8End:
ReduceMax v0, v1, v2, v3, v16
ReduceMax v4, v5, v6, v7, v17

ReduceMin v8, v9, v10, v11, v18
ReduceMin v12, v13, v14, v15, v19
cbz x7, TILE_8_Store
ld1 {v4.4h, v5.4h}, [x0] // dstMin
ld1 {v6.4h, v7.4h}, [x1] // dstMax
fmax v16.4h, v16.4h, v6.4h
fmax v17.4h, v17.4h, v7.4h
fmin v18.4h, v18.4h, v4.4h
fmin v19.4h, v19.4h, v5.4h

TILE_8_Store:
st1 {v16.4h, v17.4h}, [x1], #16
st1 {v18.4h, v19.4h}, [x0], #16
sub x5, x5, #8
add x2, x2, #64 // src += 8 * 4 * sizeof(float16_t)
b TILE_8


TILE_1:
cbz x5, Loop_Block_End

mov x4, x10  // src_depth_quad
mov x11, x2  // src

ld1 {v8.4h}, [x11], x6
mov v9.8b, v8.8b

subs x4, x4, #1
beq Tile1End

LoopSz_1:
ld1 {v16.4h}, [x11], x6

fmax v8.4h, v8.4h, v16.4h
fmin v9.4h, v9.4h, v16.4h

subs x4, x4, #1
bne LoopSz_1

Tile1End:
// reduce max/min
fmaxp v8.4h, v8.4h, v8.4h
fminp v9.4h, v9.4h, v9.4h
fmaxp v8.4h, v8.4h, v8.4h
fminp v9.4h, v9.4h, v9.4h
cbz x7, TILE_1_Store
ld1 {v10.h}[0], [x1]
ld1 {v11.h}[0], [x0]
fmax v8.4h, v8.4h, v10.4h
fmin v9.4h, v9.4h, v11.4h

TILE_1_Store:
st1 {v8.h}[0], [x1], #2
st1 {v9.h}[0], [x0], #2
subs x5, x5, #1
add x2, x2, #8 // src += 1 * 4(pack) * 2(sizeof(float16_t))
bne TILE_1

Loop_Block_End:
add x2, x12, x13
cbnz x3, Loop_BlockNum


End:
ldp d8,  d9,  [sp, #(16 * 3)]
ldp d10, d11, [sp, #(16 * 2)]
ldp d12, d13, [sp, #(16 * 1)]
ldp d14, d15, [sp], #(16 * 4)
ret

#endif
