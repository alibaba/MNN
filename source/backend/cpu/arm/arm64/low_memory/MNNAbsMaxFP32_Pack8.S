//
//  MNNAbsMaxFP32_Pack8.S
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
.macro ReduceMax8 s0, s1, s2, s3, s4, s5, s6, s7, z0
    fmaxp \s0\().4s, \s0\().4s, \s1\().4s // 0 0 0 0
    fmaxp \s2\().4s, \s2\().4s, \s3\().4s // 1 1 1 1
    fmaxp \s4\().4s, \s4\().4s, \s5\().4s // 2 2 2 2
    fmaxp \s6\().4s, \s6\().4s, \s7\().4s // 3 3 3 3
    fmaxp \s0\().4s, \s0\().4s, \s2\().4s // 0 0 1 1
    fmaxp \s4\().4s, \s4\().4s, \s6\().4s // 2 2 3 3
    fmaxp \z0\().4s, \s0\().4s, \s4\().4s // 0 1 2 3
.endm

.macro ReduceMax4 s0, s1, s2, s3, z0, z1, z2
    fmaxp \z1\().4s, \s0\().4s, \s1\().4s // 0 0 0 0
    fmaxp \z2\().4s, \s2\().4s, \s3\().4s // 1 1 1 1
    fmaxp \s0\().4s, \z1\().4s, \z2\().4s // 0 0 1 1
    fmaxp \z0\().4s, \s0\().4s, \s0\().4s // 0 1
.endm

//void MNNAbsMaxFP32_Pack8(const float* source, float* absmax, size_t src_depth_quad, size_t realSize, int pack)
asm_function MNNAbsMaxFP32_Pack8

// x0: source, x1:absmax, x2:src_depth_quad, x3:realSize
stp d14, d15, [sp, #(-16 * 4)]!
stp d12, d13, [sp, #(16 * 1)]
stp d10, d11, [sp, #(16 * 2)]
stp d8,  d9,  [sp, #(16 * 3)]

Start:
lsl x6, x3, #5 // src_step = batch * 8 * sizeof(float32_t) = batch << 5
TILE_10:
cmp x3, #10
blt TILE_8
mov x5, x2  // src_depth_quad
mov x7, x0  // src
sub x8, x6, #256 // src_step

ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x7], #64 // E0, E0, E1, E1
ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x7], #64 // E2, E2, E3, E3
ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x7], #64 // E4, E4, E5, E5
ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [x7], #64 // E6, E6, E7, E7
ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x7], x8  // E8, E8, E9, E9
Abs v0, v1, v2, v3
Abs v4, v5, v6, v7
Abs v8, v9, v10, v11
Abs v12, v13, v14, v15
Abs v16, v17, v18, v19
subs x5, x5, #1
beq Tile10End

LoopSz_10:
ld1 {v20.4s, v21.4s, v22.4s, v23.4s}, [x7], #64 // E0, E0, E1, E1
ld1 {v24.4s, v25.4s, v26.4s, v27.4s}, [x7], #64 // E2, E2, E3, E3
ld1 {v28.4s, v29.4s, v30.4s, v31.4s}, [x7], #64 // E4, E4, E5, E5

Abs v20, v21, v22, v23
Abs v24, v25, v26, v27
Abs v28, v29, v30, v31
Max v0, v1, v2, v3, v20, v21, v22, v23
Max v4, v5, v6, v7, v24, v25, v26, v27
Max v8, v9, v10, v11, v28, v29, v30, v31

ld1 {v20.4s, v21.4s, v22.4s, v23.4s}, [x7], #64 // E6, E6, E7, E7
ld1 {v24.4s, v25.4s, v26.4s, v27.4s}, [x7], x8  // E8, E8, E9, E9
Abs v20, v21, v22, v23
Abs v24, v25, v26, v27
Max v12, v13, v14, v15, v20, v21, v22, v23 
Max v16, v17, v18, v19, v24, v25, v26, v27

subs x5, x5, #1
bne LoopSz_10

Tile10End:

ReduceMax8 v0, v1, v2, v3, v4, v5, v6, v7, v20
ReduceMax8 v8, v9, v10, v11, v12, v13, v14, v15, v21
ReduceMax4 v16, v17, v18, v19, v22, v23, v24
st1 {v20.4s, v21.4s}, [x1], #32
st1 {v22.2s}, [x1], #8
sub x3, x3, #10
add x0, x0, #320 // src += 10 * pack * sizeof(float)
cbz x3, End
b TILE_10

TILE_8:
cmp x3, #8
blt TILE_4
mov x5, x2  // src_depth_quad
mov x7, x0  // src
sub x8, x6, #192 // src_step

ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x7], #64 // E0, E0, E1, E1
ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x7], #64 // E2, E2, E3, E3
ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x7], #64 // E4, E4, E5, E5
ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [x7], x8 // E6, E6, E7, E7
Abs v0, v1, v2, v3
Abs v4, v5, v6, v7
Abs v8, v9, v10, v11
Abs v12, v13, v14, v15
subs x5, x5, #1
beq Tile8End

LoopSz_8:
ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x7], #64 // E0, E0, E1, E1
ld1 {v20.4s, v21.4s, v22.4s, v23.4s}, [x7], #64 // E2, E2, E3, E3
ld1 {v24.4s, v25.4s, v26.4s, v27.4s}, [x7], #64 // E4, E4, E5, E5
ld1 {v28.4s, v29.4s, v30.4s, v31.4s}, [x7], x8 // E6, E6, E7, E7

Abs v16, v17, v18, v19
Abs v20, v21, v22, v23
Abs v24, v25, v26, v27
Abs v28, v29, v30, v31

Max v0, v1, v2, v3, v16, v17, v18, v19
Max v4, v5, v6, v7, v20, v21, v22, v23
Max v8, v9, v10, v11, v24, v25, v26, v27
Max v12, v13, v14, v15, v28, v29, v30, v31

subs x5, x5, #1
bne LoopSz_8

Tile8End:

ReduceMax8 v0, v1, v2, v3, v4, v5, v6, v7, v16
ReduceMax8 v8, v9, v10, v11, v12, v13, v14, v15, v17
st1 {v16.4s, v17.4s}, [x1], #32
sub x3, x3, #8
add x0, x0, #256 // src += 8 * pack * sizeof(float)
b TILE_8

TILE_4:
cmp x3, #4
blt TILE_1
mov x5, x2  // src_depth_quad
mov x7, x0  // src
sub x8, x6, #64 // src_step

ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x7], #64 // E0, E0, E1, E1
ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x7], x8 // E2, E2, E3, E3
Abs v0, v1, v2, v3
Abs v4, v5, v6, v7
subs x5, x5, #1
beq Tile4End

LoopSz_4:
ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x7], #64 // E0, E0, E1, E1
ld1 {v20.4s, v21.4s, v22.4s, v23.4s}, [x7], x8 // E2, E2, E3, E3

Abs v16, v17, v18, v19
Abs v20, v21, v22, v23

Max v0, v1, v2, v3, v16, v17, v18, v19
Max v4, v5, v6, v7, v20, v21, v22, v23

subs x5, x5, #1
bne LoopSz_4

Tile4End:

ReduceMax8 v0, v1, v2, v3, v4, v5, v6, v7, v16
st1 {v16.4s}, [x1], #16
sub x3, x3, #4
add x0, x0, #128 // src += 4 * pack * sizeof(float)
b TILE_4

TILE_1:
cmp x3, #1
blt End
mov x5, x2  // src_depth_quad
mov x7, x0  // src

//    sum: v0
// absmax: v8
ld1 {v0.4s, v1.4s}, [x7], x6
fabs v0.4s, v0.4s
fabs v1.4s, v1.4s
subs x5, x5, #1
beq Tile1End

LoopSz_1:
ld1 {v16.4s, v17.4s}, [x7], x6

// absmax = fmax(absmax, abs(x))
fabs v16.4s, v16.4s
fabs v17.4s, v17.4s
fmax v0.4s, v0.4s, v16.4s
fmax v1.4s, v1.4s, v17.4s

subs x5, x5, #1
bne LoopSz_1

Tile1End:
// reduce max
fmaxp v2.4s, v0.4s, v1.4s // 0 0 0 0
fmaxp v3.4s, v2.4s, v2.4s // 0 0
fmaxp v4.4s, v3.4s, v3.4s
st1 {v4.s}[0], [x1], #4
subs x3, x3, #1
add x0, x0, #32 // src += 1 * 8(pack) * 4(sizeof(float32_t))
bne TILE_1

End:
ldp d8,  d9,  [sp, #(16 * 3)]
ldp d10, d11, [sp, #(16 * 2)]
ldp d12, d13, [sp, #(16 * 1)]
ldp d14, d15, [sp], #(16 * 4)
ret

#endif
