//
//  MNNAbsMaxFP32.S
//
//  Created by MNN on 2023/10/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"
.text
.align 5

.macro Abs z0, z1, z2, z3
    fabs \z0\().4s, \z0\().4s
    fabs \z1\().4s, \z1\().4s
    fabs \z2\().4s, \z2\().4s
    fabs \z3\().4s, \z3\().4s
.endm

.macro Max d0, d1, d2, d3, z0, z1, z2, z3
    fmax \d0\().4s, \d0\().4s, \z0\().4s
    fmax \d1\().4s, \d1\().4s, \z1\().4s
    fmax \d2\().4s, \d2\().4s, \z2\().4s
    fmax \d3\().4s, \d3\().4s, \z3\().4s
.endm

.macro ReduceMax s0, s1, s2, s3, z0
    fmaxp \s0\().4s, \s0\().4s, \s1\().4s // 0 0 1 1
    fmaxp \s2\().4s, \s2\().4s, \s3\().4s // 2 2 3 3
    fmaxp \z0\().4s, \s0\().4s, \s2\().4s // 0 1 2 3
.endm
//void MNNAbsMaxFP32(const float* source, float* absmax, size_t src_depth_quad, size_t realSize, int pack)
asm_function MNNAbsMaxFP32

// x0: source, x1:absmax, x2:src_depth_quad, x3:realSize
stp d14, d15, [sp, #(-16 * 4)]!
stp d12, d13, [sp, #(16 * 1)]
stp d10, d11, [sp, #(16 * 2)]
stp d8,  d9,  [sp, #(16 * 3)]

Start:
lsl x6, x3, #4 // src_step = batch * 4 * sizeof(float32_t) = batch << 4

TILE_8:
cmp x3, #8
blt TILE_1
mov x5, x2  // src_depth_quad
mov x7, x0  // src
sub x8, x6, #64 // src_step

// absmax: v8-15
ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x7], #64
ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x7], x8
fabs v8.4s, v0.4s
fabs v9.4s, v1.4s
fabs v10.4s, v2.4s
fabs v11.4s, v3.4s
fabs v12.4s, v4.4s
fabs v13.4s, v5.4s
fabs v14.4s, v6.4s
fabs v15.4s, v7.4s
subs x5, x5, #1
beq Tile8End

LoopSz_8:
ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x7], #64
ld1 {v20.4s, v21.4s, v22.4s, v23.4s}, [x7], x8

// absmax = fmax(absmax, abs(x))
Abs v16, v17, v18, v19
Abs v20, v21, v22, v23
Max v8, v9, v10, v11, v16, v17, v18, v19
Max v12, v13, v14, v15, v20, v21, v22, v23

subs x5, x5, #1
bne LoopSz_8

Tile8End:

ReduceMax v8, v9, v10, v11, v0
ReduceMax v12, v13, v14, v15, v1
st1 {v0.4s, v1.4s}, [x1], #32
sub x3, x3, #8
add x0, x0, #128 // src += 8 * 4 * 4
b TILE_8


TILE_1:
cmp x3, #1
blt End
mov x5, x2  // src_depth_quad
mov x7, x0  // src

//    sum: v0
// absmax: v8
ld1 {v0.4s}, [x7], x6
fabs v8.4s, v0.4s
subs x5, x5, #1
beq Tile1End

LoopSz_1:
ld1 {v16.4s}, [x7], x6

// absmax = fmax(absmax, abs(x))
fabs v16.4s, v16.4s
fmax v8.4s, v8.4s, v16.4s

subs x5, x5, #1
bne LoopSz_1

Tile1End:
// reduce max
mov v1.d[0], v8.d[1]
fmax v8.4s, v8.4s, v1.4s
mov v5.s[0], v8.s[1]
fmax v8.4s, v5.4s, v8.4s
st1 {v8.s}[0], [x1], #4
subs x3, x3, #1
add x0, x0, #16 // src += 1 * 4(pack) * 4(sizeof(float32_t))
bne TILE_1

End:
ldp d8,  d9,  [sp, #(16 * 3)]
ldp d10, d11, [sp, #(16 * 2)]
ldp d12, d13, [sp, #(16 * 1)]
ldp d14, d15, [sp], #(16 * 4)
ret

#endif