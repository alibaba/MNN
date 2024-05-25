//
//  MNNAbsMaxFP16.S
//  MNN
//
//  Created by MNN on 2023/10/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"
.text
.align 5

.macro Abs z0, z1, z2, z3
    fabs \z0\().8h, \z0\().8h
    fabs \z1\().8h, \z1\().8h
    fabs \z2\().8h, \z2\().8h
    fabs \z3\().8h, \z3\().8h
.endm

.macro Max d0, d1, d2, d3, z0, z1, z2, z3
    fmax \d0\().8h, \d0\().8h, \z0\().8h
    fmax \d1\().8h, \d1\().8h, \z1\().8h
    fmax \d2\().8h, \d2\().8h, \z2\().8h
    fmax \d3\().8h, \d3\().8h, \z3\().8h
.endm

.macro ReduceSum s0, s1, s2, s3, zero
    faddp \s0\().8h, \s0\().8h, \s1\().8h // 0 0 0 0 1 1 1 1
    faddp \s2\().8h, \s2\().8h, \s3\().8h // 2 2 2 2 3 3 3 3
    faddp \s0\().8h, \s0\().8h, \s2\().8h // 0 0 1 1 2 2 3 3
    faddp \s0\().8h, \s0\().8h, \zero\().8h // 0 1 2 3
.endm

.macro ReduceMax_8 s0, s1, s2, s3, s4, s5, s6, s7
    fmaxp \s0\().8h, \s0\().8h, \s1\().8h // 0 0 0 0 1 1 1 1
    fmaxp \s2\().8h, \s2\().8h, \s3\().8h // 2 2 2 2 3 3 3 3
    fmaxp \s4\().8h, \s4\().8h, \s5\().8h // 0 0 0 0 1 1 1 1
    fmaxp \s6\().8h, \s6\().8h, \s7\().8h // 2 2 2 2 3 3 3 3
    fmaxp \s0\().8h, \s0\().8h, \s2\().8h // 0 0 1 1 2 2 3 3
    fmaxp \s4\().8h, \s4\().8h, \s6\().8h // 0 0 1 1 2 2 3 3
    fmaxp \s0\().8h, \s0\().8h, \s4\().8h // 0 1 2 3 0 1 2 3
.endm

.macro ReduceMax_4 s0, s1, s2, s3
    fmaxp \s0\().8h, \s0\().8h, \s1\().8h // 0 0 0 0 1 1 1 1
    fmaxp \s2\().8h, \s2\().8h, \s3\().8h // 2 2 2 2 3 3 3 3
    fmaxp \s0\().8h, \s0\().8h, \s2\().8h // 0 0 1 1 2 2 3 3
    fmaxp \s0\().8h, \s0\().8h, \s0\().8h // 0 0 1 1 2 2 3 3
.endm

//void MNNAbsMaxFP16(const float* source, float* absmax, size_t src_depth_quad, size_t realSize, int pack)
asm_function MNNAbsMaxFP16

// x0: source, x1:absmax, x2:src_depth_quad, x3:realSize, x4: pack(no used)
stp d14, d15, [sp, #(-16 * 4)]!
stp d12, d13, [sp, #(16 * 1)]
stp d10, d11, [sp, #(16 * 2)]
stp d8,  d9,  [sp, #(16 * 3)]

Start:
lsl x6, x3, #4 // src_step = batch * 8 * sizeof(float16_t) = batch << 4

TILE_12:
cmp x3, #12
blt TILE_10
mov x5, x2  // src_depth_quad
mov x7, x0  // src
sub x8, x6, #128 // src_step

// absmax: v0-11
ld1 {v0.8h, v1.8h, v2.8h, v3.8h}, [x7], #64
ld1 {v4.8h, v5.8h, v6.8h, v7.8h}, [x7], #64
ld1 {v8.8h, v9.8h, v10.8h, v11.8h}, [x7], x8
Abs v0, v1, v2, v3
Abs v4, v5, v6, v7
Abs v8, v9, v10, v11
subs x5, x5, #1
beq Tile12End

LoopSz_12:
ld1 {v12.8h, v13.8h, v14.8h, v15.8h}, [x7], #64
ld1 {v16.8h, v17.8h, v18.8h, v19.8h}, [x7], #64
ld1 {v20.8h, v21.8h, v22.8h, v23.8h}, [x7], x8
// absmax = fmax(absmax, abs(x))
Abs v12, v13, v14, v15
Abs v16, v17, v18, v19
Abs v20, v21, v22, v23
Max v0, v1, v2, v3, v12, v13, v14, v15
Max v4, v5, v6, v7, v16, v17, v18, v19
Max v8, v9, v10, v11, v20, v21, v22, v23

subs x5, x5, #1
bne LoopSz_12

Tile12End:

ReduceMax_8 v0, v1, v2, v3, v4, v5, v6, v7
ReduceMax_4 v8, v9, v10, v11
st1 {v0.8h}, [x1], #16
st1 {v8.d}[0], [x1], #8

sub x3, x3, #12
add x0, x0, #192 // src += 12 * 8 * 2
b TILE_12

TILE_10:
cmp x3, #10
blt TILE_8
mov x5, x2  // src_depth_quad
mov x7, x0  // src
sub x8, x6, #128 // src_step

// absmax: v0-9
ld1 {v0.8h, v1.8h, v2.8h, v3.8h}, [x7], #64
ld1 {v4.8h, v5.8h, v6.8h, v7.8h}, [x7], #64
ld1 {v8.8h, v9.8h}, [x7], x8
Abs v0, v1, v2, v3
Abs v4, v5, v6, v7
fabs v8.8h, v8.8h
fabs v9.8h, v9.8h

subs x5, x5, #1
beq Tile10End

LoopSz_10:
ld1 {v20.8h, v21.8h, v22.8h, v23.8h}, [x7], #64
ld1 {v24.8h, v25.8h, v26.8h, v27.8h}, [x7], #64
ld1 {v28.8h, v29.8h}, [x7], x8

// absmax = fmax(absmax, abs(x))
Abs v20, v21, v22, v23
Abs v24, v25, v26, v27
fabs v28.8h, v28.8h
fabs v29.8h, v29.8h

Max v0, v1, v2, v3, v20, v21, v22, v23
Max v4, v5, v6, v7, v24, v25, v26, v27
fmax v8.8h, v8.8h, v28.8h
fmax v9.8h, v9.8h, v29.8h

subs x5, x5, #1
bne LoopSz_10

Tile10End:
fmaxp v0.8h, v0.8h, v1.8h
fmaxp v0.8h, v0.8h, v0.8h
fmaxp v0.8h, v0.8h, v0.8h
st1 {v0.s}[0], [x1], #4
ReduceMax_8 v2, v3, v4, v5, v6, v7, v8, v9
st1 {v2.8h}, [x1], #16

sub x3, x3, #10
add x0, x0, #160 // src += 10 * 8 * 2
b TILE_10

TILE_8:
cmp x3, #8
blt TILE_1
mov x5, x2  // src_depth_quad
mov x7, x0  // src
sub x8, x6, #64 // src_step

// absmax: v0-7
ld1 {v0.8h, v1.8h, v2.8h, v3.8h}, [x7], #64
ld1 {v4.8h, v5.8h, v6.8h, v7.8h}, [x7], x8

Abs v0, v1, v2, v3
Abs v4, v5, v6, v7

subs x5, x5, #1
beq Tile8End

LoopSz_8:
ld1 {v16.8h, v17.8h, v18.8h, v19.8h}, [x7], #64
ld1 {v20.8h, v21.8h, v22.8h, v23.8h}, [x7], x8

// absmax = fmax(absmax, abs(x))
Abs v16, v17, v18, v19
Abs v20, v21, v22, v23
Max v0, v1, v2, v3, v16, v17, v18, v19
Max v4, v5, v6, v7, v20, v21, v22, v23

subs x5, x5, #1
bne LoopSz_8

Tile8End:
ReduceMax_8 v0, v1, v2, v3, v4, v5, v6, v7
st1 {v0.8h}, [x1], #16
sub x3, x3, #8
add x0, x0, #128 // src += 8 * 8 * 2
b TILE_8

TILE_1:
cmp x3, #1
blt End
mov x5, x2  // src_depth_quad
mov x7, x0  // src

// absmax: v0
ld1 {v0.8h}, [x7], x6
fabs v0.8h, v0.8h
subs x5, x5, #1
beq Tile1End

LoopSz_1:
ld1 {v16.8h}, [x7], x6

// absmax = fmax(absmax, abs(x))
fabs v16.8h, v16.8h
fmax v0.8h, v0.8h, v16.8h

subs x5, x5, #1
bne LoopSz_1

Tile1End:

fmaxp v2.8h, v0.8h, v0.8h
fmaxp v3.8h, v2.8h, v2.8h
fmaxp v4.8h, v3.8h, v3.8h
st1 {v4.h}[0], [x1], #2

sub x3, x3, #1
add x0, x0, #16 // src += 1 * 8 * 2
b TILE_1

End:
ldp d8,  d9,  [sp, #(16 * 3)]
ldp d10, d11, [sp, #(16 * 2)]
ldp d12, d13, [sp, #(16 * 1)]
ldp d14, d15, [sp], #(16 * 4)
ret

#endif