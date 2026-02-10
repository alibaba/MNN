//
//  MNNLocalMinMaxFP16_Pack8.S
//
//  Created by MNN on 2023/10/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"
.text
.align 5

.macro Max4 d0, d1, d2, d3, z0, z1, z2, z3
    fmax \d0\().8h, \d0\().8h, \z0\().8h
    fmax \d1\().8h, \d1\().8h, \z1\().8h
    fmax \d2\().8h, \d2\().8h, \z2\().8h
    fmax \d3\().8h, \d3\().8h, \z3\().8h
.endm

.macro Min4 d0, d1, d2, d3, z0, z1, z2, z3
    fmin \d0\().8h, \d0\().8h, \z0\().8h
    fmin \d1\().8h, \d1\().8h, \z1\().8h
    fmin \d2\().8h, \d2\().8h, \z2\().8h
    fmin \d3\().8h, \d3\().8h, \z3\().8h
.endm

.macro ReduceMax s0, s1, s2, s3, s4, s5, s6, s7, z0
    fmaxp \s0\().8h, \s0\().8h, \s1\().8h // 0 0 0 0 1 1 1 1
    fmaxp \s2\().8h, \s2\().8h, \s3\().8h // 2 2 2 2 3 3 3 3
    fmaxp \s0\().8h, \s0\().8h, \s2\().8h // 0 0 1 1 2 2 3 3
    fmaxp \s4\().8h, \s4\().8h, \s5\().8h // 4 4 4 4 5 5 5 5
    fmaxp \s6\().8h, \s6\().8h, \s7\().8h // 6 6 6 6 7 7 7 7
    fmaxp \s4\().8h, \s4\().8h, \s6\().8h // 4 4 5 5 6 6 7 7
    fmaxp \z0\().8h, \s0\().8h, \s4\().8h // 0 1 2 3 4 5 6 7
.endm

.macro ReduceMin s0, s1, s2, s3, s4, s5, s6, s7, z0
    fminp \s0\().8h, \s0\().8h, \s1\().8h // 0 0 0 0 1 1 1 1
    fminp \s2\().8h, \s2\().8h, \s3\().8h // 2 2 2 2 3 3 3 3
    fminp \s0\().8h, \s0\().8h, \s2\().8h // 0 0 1 1 2 2 3 3
    fminp \s4\().8h, \s4\().8h, \s5\().8h // 4 4 4 4 5 5 5 5
    fminp \s6\().8h, \s6\().8h, \s7\().8h // 6 6 6 6 7 7 7 7
    fminp \s4\().8h, \s4\().8h, \s6\().8h // 4 4 5 5 6 6 7 7
    fminp \z0\().8h, \s0\().8h, \s4\().8h // 0 1 2 3 4 5 6 7
.endm

.macro L4Copy s0, s1, s2, s3, z0, z1, z2, z3
    mov \s0\().16b, \z0\().16b
    mov \s1\().16b, \z1\().16b
    mov \s2\().16b, \z2\().16b
    mov \s3\().16b, \z3\().16b
.endm

//void MNNLocalMinMaxFP16_Pack8(float* dstMin, float* dstMax, const float* source, size_t blockNum, size_t blockLU, size_t EP, size_t LP, size_t loadDstBuffer)
asm_function MNNLocalMinMaxFP16_Pack8

// x0: dstMin, x1:dstMax, x2:source, x3:blockNum, x4: blockLU, x5: EP, x6: LP=4, x7: loadDstBuffer
// input shape: [blocknum, blockLU, EP, LP]
stp d14, d15, [sp, #(-16 * 4)]!
stp d12, d13, [sp, #(16 * 1)]
stp d10, d11, [sp, #(16 * 2)]
stp d8,  d9,  [sp, #(16 * 3)]

// LP=8
lsl x6, x5, #4 // src_step = batch * 8 * sizeof(float16_t) = batch << 4
mul x13, x5, x4       // blockLU * EP * LP * sizeof(float16_t)
lsl x13, x13, #4
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
sub x8, x6, #64 // src_step

ld1 {v0.8h, v1.8h, v2.8h, v3.8h}, [x11], #64
ld1 {v4.8h, v5.8h, v6.8h, v7.8h}, [x11], x8
L4Copy v8, v9, v10, v11, v0, v1, v2, v3
L4Copy v12, v13, v14, v15, v4, v5, v6, v7
subs x4, x4, #1
beq Tile8End

LoopSz_8:
ld1 {v16.8h, v17.8h, v18.8h, v19.8h}, [x11], #64
ld1 {v20.8h, v21.8h, v22.8h, v23.8h}, [x11], x8

Max4 v0, v1, v2, v3, v16, v17, v18, v19
Max4 v4, v5, v6, v7, v20, v21, v22, v23
Min4 v8, v9, v10, v11, v16, v17, v18, v19
Min4 v12, v13, v14, v15, v20, v21, v22, v23

subs x4, x4, #1
bne LoopSz_8

Tile8End:
ReduceMax v0, v1, v2, v3, v4, v5, v6, v7, v16
ReduceMin v8, v9, v10, v11, v12, v13, v14, v15, v18

cbz x7, TILE_8_Store
ld1 {v4.8h}, [x0] // dstMin
ld1 {v6.8h}, [x1] // dstMax
fmax v16.8h, v16.8h, v6.8h
fmin v18.8h, v18.8h, v4.8h

TILE_8_Store:
st1 {v16.8h}, [x1], #16
st1 {v18.8h}, [x0], #16
sub x5, x5, #8
add x2, x2, #128 // src += 8 * 8 * sizeof(float16_t)
b TILE_8


TILE_1:
cbz x5, Loop_Block_End

mov x4, x10  // src_depth_quad
mov x11, x2  // src

ld1 {v8.8h}, [x11], x6
mov v9.16b, v8.16b

subs x4, x4, #1
beq Tile1End

LoopSz_1:
ld1 {v16.8h}, [x11], x6

fmax v8.8h, v8.8h, v16.8h
fmin v9.8h, v9.8h, v16.8h

subs x4, x4, #1
bne LoopSz_1

Tile1End:
// reduce max/min
fmaxp v8.8h, v8.8h, v8.8h
fminp v9.8h, v9.8h, v9.8h
fmaxp v8.8h, v8.8h, v8.8h
fminp v9.8h, v9.8h, v9.8h
fmaxp v8.8h, v8.8h, v8.8h
fminp v9.8h, v9.8h, v9.8h
cbz x7, TILE_1_Store
ld1 {v10.h}[0], [x1]
ld1 {v11.h}[0], [x0]
fmax v8.4h, v8.4h, v10.4h
fmin v9.4h, v9.4h, v11.4h

TILE_1_Store:
st1 {v8.h}[0], [x1], #2
st1 {v9.h}[0], [x0], #2
subs x5, x5, #1
add x2, x2, #16 // src += 1 * 8(pack) * 2(sizeof(float16_t))
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
