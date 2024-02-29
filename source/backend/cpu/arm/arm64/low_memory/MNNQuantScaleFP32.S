//
//  MNNQuantScaleFP32.S
//  MNN
//
//  Created by MNN on 2023/11/01.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"
.text
.align 5

// void MNNQuantScaleFP32(float* absmax, float* quant_scale, float* dequant_scale, size_t thread, size_t batch)
asm_function MNNQuantScaleFP32

// x0:absmax, x1:quant_scale, x2:dequant_scale, x3:thread, x4:batch
stp d14, d15, [sp, #(-16 * 4)]!
stp d12, d13, [sp, #(16 * 1)]
stp d10, d11, [sp, #(16 * 2)]
stp d8,  d9,  [sp, #(16 * 3)]

Start:
movi v31.4s, #127
scvtf v31.4s, v31.4s
lsl x9, x4, #2 // src_step = batch * sizeof(float32_t)

TILE_8:
cmp x4, #8
blt TILE_4
mov x7, x0
mov x8, x3

ld1 {v1.4s, v2.4s}, [x7], x9
subs x8, x8, #1
beq Tile8End

LoopSz_8:
ld1 {v3.4s, v4.4s}, [x7], x9
fmax v1.4s, v1.4s, v3.4s
fmax v2.4s, v2.4s, v4.4s
subs x8, x8, #1
bne LoopSz_8

Tile8End:
sub x4, x4, #8
add x0, x0, #32
fdiv v5.4s, v31.4s, v1.4s
fdiv v6.4s, v31.4s, v2.4s
fdiv v7.4s, v1.4s, v31.4s
fdiv v8.4s, v2.4s, v31.4s

st1 {v5.4s, v6.4s}, [x1], #32
st1 {v7.4s, v8.4s}, [x2], #32
b TILE_8

TILE_4:
cmp x4, #4
blt TILE_1
mov x7, x0  // max_ptr
mov x8, x3  // thread

// absmax: v1
ld1 {v1.4s}, [x7], x9
subs x8, x8, #1
beq Tile4End

LoopSz_4:
ld1 {v3.4s}, [x7], x9

// absmax = fmax(absmax, absmax[i])
fmax v1.4s, v1.4s, v3.4s

subs x8, x8, #1
bne LoopSz_4

Tile4End:
sub x4, x4, #4
add x0, x0, #16
// quant_scale = 127 / absmax
// dequant_scale = absmax / 127
fdiv v2.4s, v31.4s, v1.4s
fdiv v3.4s, v1.4s, v31.4s
st1 {v2.4s}, [x1], #16
st1 {v3.4s}, [x2], #16
b TILE_4


TILE_1:
cmp x4, #1
blt End
mov x7, x0  // max_ptr
mov x8, x3  // thread

//    sum: v0
// absmax: v1
ld1 {v1.s}[0], [x7], x9
subs x8, x8, #1
beq Tile1End

LoopSz_1:
ld1 {v3.s}[0], [x7], x9

// absmax = fmax(absmax, absmax[i])
fmax s1, s1, s3

subs x8, x8, #1
bne LoopSz_1

Tile1End:
sub x4, x4, #1
add x0, x0, #4
// quant_scale = 127 / absmax
// dequant_scale = absmax / 127
fdiv s2, s31, s1
fdiv s3, s1, s31
st1 {v2.s}[0], [x1], #4
st1 {v3.s}[0], [x2], #4
b TILE_1


End:
ldp d8,  d9,  [sp, #(16 * 3)]
ldp d10, d11, [sp, #(16 * 2)]
ldp d12, d13, [sp, #(16 * 1)]
ldp d14, d15, [sp], #(16 * 4)
ret

#endif

