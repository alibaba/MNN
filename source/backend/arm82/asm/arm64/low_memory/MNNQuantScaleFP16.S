//
//  MNNQuantScaleFP16.S
//  MNN
//
//  Created by MNN on 2023/11/01.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"
.text
.align 5

.macro Round z0, z1, z2, z3
    fcvtas \z0\().8h, \z0\().8h
    fcvtas \z1\().8h, \z1\().8h
    fcvtas \z2\().8h, \z2\().8h
    fcvtas \z3\().8h, \z3\().8h
.endm

//void MNNQuantScaleFP16(float* absmax, float* quant_scale, float* dequant_scale, size_t thread, size_t batch)
asm_function MNNQuantScaleFP16

// x0:absmax, x1:quant_scale, x2:dequant_scale, x3:thread, x4:batch
stp d14, d15, [sp, #(-16 * 4)]!
stp d12, d13, [sp, #(16 * 1)]
stp d10, d11, [sp, #(16 * 2)]
stp d8,  d9,  [sp, #(16 * 3)]

Start:
movi v31.4s, #127
scvtf v31.4s, v31.4s
//fcvtn v31.4h, v0.4s
//fcvtn2 v31.8h, v0.4s
lsl x9, x4, #1 // src_step = batch * sizeof(float16_t)

TILE_12:
cmp x4, #12
blt TILE_10
sub x10, x9, #16
mov x6, x0  // max_ptr
mov x7, x3  // thread

// absmax: v0, v1
ld1 {v0.8h}, [x6], #16
ld1 {v1.d}[0], [x6], x10
subs x7, x7, #1
beq Tile12End

LoopSz_12:
ld1 {v4.8h}, [x6], #16
ld1 {v5.d}[0], [x6], x10

// absmax = fmax(absmax, absmax[i])
fmax v0.8h, v0.8h, v4.8h
fmax v1.8h, v1.8h, v5.8h

subs x7, x7, #1
bne LoopSz_12

Tile12End:
sub x4, x4, #12
add x0, x0, #24
// quant_scale = 127 / absmax
// dequant_scale = absmax / 127

// float16->float32
fcvtl v4.4s, v0.4h
fcvtl2 v5.4s, v0.8h
fcvtl v6.4s, v1.4h

fdiv v8.4s, v31.4s, v4.4s
fdiv v9.4s, v31.4s, v5.4s
fdiv v10.4s, v31.4s, v6.4s

fdiv v12.4s, v4.4s, v31.4s
fdiv v13.4s, v5.4s, v31.4s
fdiv v14.4s, v6.4s, v31.4s

st1 {v8.4s, v9.4s, v10.4s}, [x1], #48
st1 {v12.4s, v13.4s, v14.4s}, [x2], #48

//fdiv v4.8h, v31.8h, v0.8h
//fdiv v5.8h, v31.8h, v1.8h
//fdiv v6.8h, v0.8h, v31.8h
//fdiv v7.8h, v1.8h, v31.8h

//st1 {v4.8h}, [x1], #16
//st1 {v5.d}[0], [x1], #8
//st1 {v6.8h}, [x2], #16
//st1 {v7.d}[0], [x2], #8
b TILE_12

TILE_10:
cmp x4, #10
blt TILE_8
sub x10, x9, #16
mov x6, x0  // max_ptr
mov x7, x3  // thread

// absmax: v0, v1
ld1 {v0.8h}, [x6], #16
ld1 {v1.s}[0], [x6], x10
subs x7, x7, #1
beq Tile10End

LoopSz_10:
ld1 {v4.8h}, [x6], #16
ld1 {v5.s}[0], [x6], x10

// absmax = fmax(absmax, absmax[i])
fmax v0.8h, v0.8h, v4.8h
fmax v1.8h, v1.8h, v5.8h

subs x7, x7, #1
bne LoopSz_10

Tile10End:
sub x4, x4, #10
add x0, x0, #20
// quant_scale = 127 / absmax
// dequant_scale = absmax / 127

// float16->float32
fcvtl v4.4s, v0.4h
fcvtl2 v5.4s, v0.8h
fcvtl v6.4s, v1.4h

fdiv v8.4s, v31.4s, v4.4s
fdiv v9.4s, v31.4s, v5.4s
fdiv v10.4s, v31.4s, v6.4s

fdiv v12.4s, v4.4s, v31.4s
fdiv v13.4s, v5.4s, v31.4s
fdiv v14.4s, v6.4s, v31.4s

st1 {v8.4s, v9.4s}, [x1], #32
st1 {v10.d}[0], [x1], #8
st1 {v12.4s, v13.4s}, [x2], #32
st1 {v14.d}[0], [x2], #8

// fdiv v4.8h, v31.8h, v0.8h
// fdiv v5.8h, v31.8h, v1.8h
// fdiv v6.8h, v0.8h, v31.8h
// fdiv v7.8h, v1.8h, v31.8h
// st1 {v4.8h}, [x1], #16
// st1 {v5.s}[0], [x1], #4
// st1 {v6.8h}, [x2], #16
// st1 {v7.s}[0], [x2], #4
b TILE_10


TILE_8:
cmp x4, #8
blt TILE_1
mov x6, x0  // max_ptr
mov x7, x3  // thread

// absmax: v0
ld1 {v0.8h}, [x6], x9
subs x7, x7, #1
beq Tile8End

LoopSz_8:
ld1 {v2.8h}, [x6], x9

// absmax = fmax(absmax, absmax[i])
fmax v0.8h, v0.8h, v2.8h

subs x7, x7, #1
bne LoopSz_8

Tile8End:
sub x4, x4, #8
add x0, x0, #16
// quant_scale = 127 / absmax
// dequant_scale = absmax / 127
// float16->float32
fcvtl v4.4s, v0.4h
fcvtl2 v5.4s, v0.8h

fdiv v8.4s, v31.4s, v4.4s
fdiv v9.4s, v31.4s, v5.4s

fdiv v12.4s, v4.4s, v31.4s
fdiv v13.4s, v5.4s, v31.4s

st1 {v8.4s, v9.4s}, [x1], #32
st1 {v12.4s, v13.4s}, [x2], #32

// fdiv v2.8h, v31.8h, v0.8h
// fdiv v3.8h, v0.8h, v31.8h
// st1 {v2.8h}, [x1], #16
// st1 {v3.8h}, [x2], #16
b TILE_8


TILE_1:
cmp x4, #1
blt End
mov x6, x0  // absmax
mov x7, x3  // thread

// absmax: v0
ld1 {v0.h}[0], [x6], x9
subs x7, x7, #1
beq Tile1End

LoopSz_1:
ld1 {v2.h}[0], [x6], x9

// absmax = fmax(absmax, absmax[i])
fmax h0, h0, h2

subs x7, x7, #1
bne LoopSz_1

Tile1End:
sub x4, x4, #1
add x0, x0, #2
// quant_scale = 127 / absmax
// dequant_scale = absmax / 127
fcvtl v4.4s, v0.4h

fdiv v8.4s, v31.4s, v4.4s
fdiv v12.4s, v4.4s, v31.4s

st1 {v8.s}[0], [x1], #4
st1 {v12.s}[0], [x2], #4

// fdiv h2, h31, h0
// fdiv h3, h0, h31
// st1 {v2.h}[0], [x1], #2
// st1 {v3.h}[0], [x2], #2
b TILE_1


End:
ldp d8,  d9,  [sp, #(16 * 3)]
ldp d10, d11, [sp, #(16 * 2)]
ldp d12, d13, [sp, #(16 * 1)]
ldp d14, d15, [sp], #(16 * 4)
ret

#endif

