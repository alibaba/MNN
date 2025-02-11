//
//  MNNAbsMaxFP16_Pack4.S
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
    fabs \z0\().4h, \z0\().4h
    fabs \z1\().4h, \z1\().4h
    fabs \z2\().4h, \z2\().4h
    fabs \z3\().4h, \z3\().4h
.endm

.macro Max d0, d1, d2, d3, z0, z1, z2, z3
    fmax \d0\().4h, \d0\().4h, \z0\().4h
    fmax \d1\().4h, \d1\().4h, \z1\().4h
    fmax \d2\().4h, \d2\().4h, \z2\().4h
    fmax \d3\().4h, \d3\().4h, \z3\().4h
.endm

.macro ReduceMax_8 s0, s1, s2, s3, s4, s5, s6, s7
    fmaxp \s0\().4h, \s0\().4h, \s1\().4h // 0 0 1 1
    fmaxp \s2\().4h, \s2\().4h, \s3\().4h // 2 2 3 3
    fmaxp \s4\().4h, \s4\().4h, \s5\().4h // 4 4 5 6
    fmaxp \s6\().4h, \s6\().4h, \s7\().4h // 6 6 7 7
    fmaxp \s0\().4h, \s0\().4h, \s2\().4h // 0 1 2 3
    fmaxp \s4\().4h, \s4\().4h, \s6\().4h // 4 5 6 7
    mov \s0\().d[1], \s4\().d[0] // 0 1 2 3 4 5 6 7
.endm

.macro ReduceMax_4 s0, s1, s2, s3
    fmaxp \s0\().4h, \s0\().4h, \s1\().4h // 0 0 1 1
    fmaxp \s2\().4h, \s2\().4h, \s3\().4h // 2 2 3 3
    fmaxp \s0\().4h, \s0\().4h, \s2\().4h // 0 1 2 3
.endm

//void MNNAbsMaxFP16_Pack4(const float* source, float* absmax, size_t LU, size_t EP, int LP)
// LP = 4
asm_function MNNAbsMaxFP16_Pack4

// x0: source, x1:absmax, x2:src_depth_quad, x3:realSize, x4: pack(no used)
stp d14, d15, [sp, #(-16 * 4)]!
stp d12, d13, [sp, #(16 * 1)]
stp d10, d11, [sp, #(16 * 2)]
stp d8,  d9,  [sp, #(16 * 3)]

Start:
lsl x6, x3, #3 // src_step = batch * 4 * sizeof(float16_t) = batch << 3

TILE_12:
cmp x3, #12
blt TILE_10
mov x5, x2  // src_depth_quad
mov x7, x0  // src
sub x8, x6, #64 // src_step

// absmax: v0-11
ld1 {v0.4h, v1.4h, v2.4h, v3.4h}, [x7], #32
ld1 {v4.4h, v5.4h, v6.4h, v7.4h}, [x7], #32
ld1 {v8.4h, v9.4h, v10.4h, v11.4h}, [x7], x8
Abs v0, v1, v2, v3
Abs v4, v5, v6, v7
Abs v8, v9, v10, v11
subs x5, x5, #1
beq Tile12End

LoopSz_12:
ld1 {v12.4h, v13.4h, v14.4h, v15.4h}, [x7], #32
ld1 {v16.4h, v17.4h, v18.4h, v19.4h}, [x7], #32
ld1 {v20.4h, v21.4h, v22.4h, v23.4h}, [x7], x8
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
st1 {v8.4h}, [x1], #8

sub x3, x3, #12
add x0, x0, #96 // src += 12 * 4 * 2
b TILE_12

TILE_10:
cmp x3, #10
blt TILE_8
mov x5, x2  // src_depth_quad
mov x7, x0  // src
sub x8, x6, #64 // src_step

// absmax: v0-9
ld1 {v0.4h, v1.4h, v2.4h, v3.4h}, [x7], #32
ld1 {v4.4h, v5.4h, v6.4h, v7.4h}, [x7], #32
ld1 {v8.4h, v9.4h}, [x7], x8
Abs v0, v1, v2, v3
Abs v4, v5, v6, v7
fabs v8.4h, v8.4h
fabs v9.4h, v9.4h

subs x5, x5, #1
beq Tile10End

LoopSz_10:
ld1 {v20.4h, v21.4h, v22.4h, v23.4h}, [x7], #32
ld1 {v24.4h, v25.4h, v26.4h, v27.4h}, [x7], #32
ld1 {v28.4h, v29.4h}, [x7], x8

// absmax = fmax(absmax, abs(x))
Abs v20, v21, v22, v23
Abs v24, v25, v26, v27
fabs v28.4h, v28.4h
fabs v29.4h, v29.4h

Max v0, v1, v2, v3, v20, v21, v22, v23
Max v4, v5, v6, v7, v24, v25, v26, v27
fmax v8.4h, v8.4h, v28.4h
fmax v9.4h, v9.4h, v29.4h

subs x5, x5, #1
bne LoopSz_10

Tile10End:
fmaxp v0.4h, v0.4h, v1.4h // 0 0 1 1
fmaxp v0.4h, v0.4h, v0.4h // 0 1
ReduceMax_8 v2, v3, v4, v5, v6, v7, v8, v9
st1 {v0.s}[0], [x1], #4
st1 {v2.8h}, [x1], #16

sub x3, x3, #10
add x0, x0, #80 // src += 10 * 4 * 2
b TILE_10

TILE_8:
cmp x3, #8
blt TILE_1
mov x5, x2  // src_depth_quad
mov x7, x0  // src
sub x8, x6, #32 // src_step

// absmax: v0-7
ld1 {v0.4h, v1.4h, v2.4h, v3.4h}, [x7], #32
ld1 {v4.4h, v5.4h, v6.4h, v7.4h}, [x7], x8

Abs v0, v1, v2, v3
Abs v4, v5, v6, v7

subs x5, x5, #1
beq Tile8End

LoopSz_8:
ld1 {v16.4h, v17.4h, v18.4h, v19.4h}, [x7], #32
ld1 {v20.4h, v21.4h, v22.4h, v23.4h}, [x7], x8

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
add x0, x0, #64 // src += 8 * 4 * 2
b TILE_8

TILE_1:
cmp x3, #1
blt End
mov x5, x2  // src_depth_quad
mov x7, x0  // src

// absmax: v0
ld1 {v0.4h}, [x7], x6
fabs v0.8h, v0.8h
subs x5, x5, #1
beq Tile1End

LoopSz_1:
ld1 {v16.4h}, [x7], x6

// absmax = fmax(absmax, abs(x))
fabs v16.4h, v16.4h
fmax v0.4h, v0.4h, v16.4h

subs x5, x5, #1
bne LoopSz_1

Tile1End:

fmaxp v2.4h, v0.4h, v0.4h
fmaxp v3.4h, v2.4h, v2.4h
st1 {v3.h}[0], [x1], #2

sub x3, x3, #1
add x0, x0, #8 // src += 1 * 4 * 2
b TILE_1

End:
ldp d8,  d9,  [sp, #(16 * 3)]
ldp d10, d11, [sp, #(16 * 2)]
ldp d12, d13, [sp, #(16 * 1)]
ldp d14, d15, [sp], #(16 * 4)
ret

#endif