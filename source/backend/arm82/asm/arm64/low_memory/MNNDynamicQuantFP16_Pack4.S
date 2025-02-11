//
//  MNNDynamicQuantFP16_Pack4.S
//  MNN
//
//  Created by MNN on 2023/10/31.
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

//void MNNDynamicQuantFP16(const float* src, int8_t* dst, const float* scale, size_t src_depth_quad, size_t realSize, int pack)
asm_function MNNDynamicQuantFP16_Pack4

// x0: src, x1:dst, x2:scale, x3:src_depth_quad, x4:realSize
stp d14, d15, [sp, #(-16 * 4)]!
stp d12, d13, [sp, #(16 * 1)]
stp d10, d11, [sp, #(16 * 2)]
stp d8,  d9,  [sp, #(16 * 3)]

// pack=4=LP
Start:
lsl x6, x4, #2  // dst_step = batch * pack * sizeof(int8_t) = batch * 4 = batch << 2
lsl x7, x4, #3  // src_step = batch * pack * sizeof(float16) = batch * 4 * 2 = batch << 3

TILE_24:
cmp x4, #24
blt TILE_16
mov x9, x0   // src
mov x10, x1  // dst
sub x15, x6, #64
mov x12, x3  // src_depth_quad
sub x13, x7, #160 // src_step - 160

ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [x2], #64
ld1 {v16.4s, v17.4s}, [x2], #32
fcvtn v12.4h, v12.4s
fcvtn2 v12.8h, v13.4s
fcvtn v13.4h, v14.4s
fcvtn2 v13.8h, v15.4s
fcvtn v14.4h, v16.4s
fcvtn2 v14.8h, v17.4s

LoopSz_24:
ld1 {v0.4h, v1.4h, v2.4h, v3.4h}, [x9], #32
ld1 {v4.4h, v5.4h, v6.4h, v7.4h}, [x9], #32
ld1 {v8.4h, v9.4h, v10.4h, v11.4h}, [x9], #32
ld1 {v15.4h, v16.4h, v17.4h, v18.4h}, [x9], #32
ld1 {v19.4h, v20.4h, v21.4h, v22.4h}, [x9], #32
ld1 {v23.4h, v24.4h, v25.4h, v26.4h}, [x9], x13

// float16_t x = x * quant_scale
fmul v0.4h, v0.4h, v12.h[0]
fmul v1.4h, v1.4h, v12.h[1]
fmul v2.4h, v2.4h, v12.h[2]
fmul v3.4h, v3.4h, v12.h[3]
fmul v4.4h, v4.4h, v12.h[4]
fmul v5.4h, v5.4h, v12.h[5]
fmul v6.4h, v6.4h, v12.h[6]
fmul v7.4h, v7.4h, v12.h[7]
fmul v8.4h, v8.4h, v13.h[0]
fmul v9.4h, v9.4h, v13.h[1]
fmul v10.4h, v10.4h, v13.h[2]
fmul v11.4h, v11.4h, v13.h[3]
fmul v15.4h, v15.4h, v13.h[4]
fmul v16.4h, v16.4h, v13.h[5]
fmul v17.4h, v17.4h, v13.h[6]
fmul v18.4h, v18.4h, v13.h[7]

fmul v19.4h, v19.4h, v14.h[0]
fmul v20.4h, v20.4h, v14.h[1]
fmul v21.4h, v21.4h, v14.h[2]
fmul v22.4h, v22.4h, v14.h[3]
fmul v23.4h, v23.4h, v14.h[4]
fmul v24.4h, v24.4h, v14.h[5]
fmul v25.4h, v25.4h, v14.h[6]
fmul v26.4h, v26.4h, v14.h[7]

mov v0.d[1], v1.d[0]
mov v2.d[1], v3.d[0]
mov v4.d[1], v5.d[0]
mov v6.d[1], v7.d[0]
mov v8.d[1], v9.d[0]
mov v10.d[1], v11.d[0]
mov v15.d[1], v16.d[0]
mov v17.d[1], v18.d[0]
mov v19.d[1], v20.d[0]
mov v21.d[1], v22.d[0]
mov v23.d[1], v24.d[0]
mov v25.d[1], v26.d[0]

// int16_t x = round(x)
Round v0, v2, v4, v6
Round v8, v10, v15, v17
Round v19, v21, v23, v25

// y = (int8_t)x
sqxtn v27.8b, v0.8h
sqxtn2 v27.16b, v2.8h
sqxtn v28.8b, v4.8h
sqxtn2 v28.16b, v6.8h
sqxtn v29.8b, v8.8h
sqxtn2 v29.16b, v10.8h
sqxtn v30.8b, v15.8h
sqxtn2 v30.16b, v17.8h
sqxtn v0.8b, v19.8h
sqxtn2 v0.16b, v21.8h
sqxtn v1.8b, v23.8h
sqxtn2 v1.16b, v25.8h

st1 {v27.16b, v28.16b, v29.16b, v30.16b}, [x10], #64
st1 {v0.16b, v1.16b}, [x10], x15

subs x12, x12, #1
bne LoopSz_24

Tile24End:
sub x4, x4, #24   // batch -= 24
add x0, x0, #192  // src += 24 * 4 * sizeof(float16_t)
add x1, x1, #96   // dst += 24 * 4 * sizeof(int8_t)
b TILE_24

TILE_16:
cmp x4, #16
blt TILE_12
mov x9, x0   // src
mov x10, x1  // dst
// sub x15, x6, #32
mov x12, x3  // src_depth_quad
sub x13, x7, #96 // src_step - 192

ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [x2], #64
fcvtn v12.4h, v12.4s
fcvtn2 v12.8h, v13.4s
fcvtn v13.4h, v14.4s
fcvtn2 v13.8h, v15.4s

LoopSz_16:
ld1 {v0.4h, v1.4h, v2.4h, v3.4h}, [x9], #32
ld1 {v4.4h, v5.4h, v6.4h, v7.4h}, [x9], #32
ld1 {v8.4h, v9.4h, v10.4h, v11.4h}, [x9], #32
ld1 {v15.4h, v16.4h, v17.4h, v18.4h}, [x9], x13

// float16_t x = x * quant_scale
fmul v0.4h, v0.4h, v12.h[0]
fmul v1.4h, v1.4h, v12.h[1]
fmul v2.4h, v2.4h, v12.h[2]
fmul v3.4h, v3.4h, v12.h[3]
fmul v4.4h, v4.4h, v12.h[4]
fmul v5.4h, v5.4h, v12.h[5]
fmul v6.4h, v6.4h, v12.h[6]
fmul v7.4h, v7.4h, v12.h[7]
fmul v8.4h, v8.4h, v13.h[0]
fmul v9.4h, v9.4h, v13.h[1]
fmul v10.4h, v10.4h, v13.h[2]
fmul v11.4h, v11.4h, v13.h[3]
fmul v15.4h, v15.4h, v13.h[4]
fmul v16.4h, v16.4h, v13.h[5]
fmul v17.4h, v17.4h, v13.h[6]
fmul v18.4h, v18.4h, v13.h[7]

mov v0.d[1], v1.d[0]
mov v2.d[1], v3.d[0]
mov v4.d[1], v5.d[0]
mov v6.d[1], v7.d[0]
mov v8.d[1], v9.d[0]
mov v10.d[1], v11.d[0]
mov v15.d[1], v16.d[0]
mov v17.d[1], v18.d[0]

// int16_t x = round(x)
Round v0, v2, v4, v6
Round v8, v10, v15, v17

// y = (int8_t)x
sqxtn v19.8b, v0.8h
sqxtn2 v19.16b, v2.8h
sqxtn v20.8b, v4.8h
sqxtn2 v20.16b, v6.8h
sqxtn v21.8b, v8.8h
sqxtn2 v21.16b, v10.8h
sqxtn v22.8b, v15.8h
sqxtn2 v22.16b, v17.8h

st1 {v19.16b, v20.16b, v21.16b, v22.16b}, [x10], x6

subs x12, x12, #1
bne LoopSz_16

Tile16End:
sub x4, x4, #16   // batch -= 16
add x0, x0, #128  // src += 16 * 4 * sizeof(float16_t)
add x1, x1, #64   // dst += 16 * 4 * sizeof(int8_t)
b TILE_16

TILE_12:
cmp x4, #12
blt TILE_10
mov x9, x0   // src
mov x10, x1  // dst
mov x12, x3  // src_depth_quad
sub x13, x7, #64 // src_step - 128

ld1 {v12.4s, v13.4s, v14.4s}, [x2], #48
fcvtn v12.4h, v12.4s
fcvtn2 v12.8h, v13.4s
fcvtn v13.4h, v14.4s

LoopSz_12:
ld1 {v0.4h, v1.4h, v2.4h, v3.4h}, [x9], #32
ld1 {v4.4h, v5.4h, v6.4h, v7.4h}, [x9], #32
ld1 {v8.4h, v9.4h, v10.4h, v11.4h}, [x9], x13

// float16_t x = x * quant_scale
fmul v0.4h, v0.4h, v12.h[0]
fmul v1.4h, v1.4h, v12.h[1]
fmul v2.4h, v2.4h, v12.h[2]
fmul v3.4h, v3.4h, v12.h[3]
fmul v4.4h, v4.4h, v12.h[4]
fmul v5.4h, v5.4h, v12.h[5]
fmul v6.4h, v6.4h, v12.h[6]
fmul v7.4h, v7.4h, v12.h[7]
fmul v8.4h, v8.4h, v13.h[0]
fmul v9.4h, v9.4h, v13.h[1]
fmul v10.4h, v10.4h, v13.h[2]
fmul v11.4h, v11.4h, v13.h[3]

mov v0.d[1], v1.d[0]
mov v2.d[1], v3.d[0]
mov v4.d[1], v5.d[0]
mov v6.d[1], v7.d[0]
mov v8.d[1], v9.d[0]
mov v10.d[1], v11.d[0]

// int16_t x = round(x)
Round v0, v2, v4, v6
fcvtas v8.8h, v8.8h
fcvtas v10.8h, v10.8h

// y = (int8_t)x
sqxtn  v14.8b, v0.8h
sqxtn2 v14.16b, v2.8h
sqxtn  v15.8b, v4.8h
sqxtn2 v15.16b, v6.8h
sqxtn  v16.8b, v8.8h
sqxtn2 v16.16b, v10.8h

st1 {v14.16b, v15.16b, v16.16b}, [x10], x6

subs x12, x12, #1
bne LoopSz_12

Tile12End:
sub x4, x4, #12   // batch -= 12
add x0, x0, #96  // src += 12 * 4 * sizeof(float16_t)
add x1, x1, #48   // dst += 12 * 4 * sizeof(int8_t)
b TILE_12

TILE_10:
cmp x4, #10
blt TILE_8
mov x9, x0   // src
mov x10, x1  // dst
mov x12, x3  // src_depth_quad
sub x13, x7, #64 // src_step - 64
sub x15, x6, #32

ld1 {v12.4s, v13.4s}, [x2], #32
ld1 {v14.d}[0], [x2], #8
fcvtn v10.4h, v12.4s
fcvtn2 v10.8h, v13.4s
fcvtn v11.4h, v14.4s

LoopSz_10:
ld1 {v0.4h, v1.4h, v2.4h, v3.4h}, [x9], #32
ld1 {v4.4h, v5.4h, v6.4h, v7.4h}, [x9], #32
ld1 {v8.4h, v9.4h}, [x9], x13

// float16_t x = x * quant_scale
fmul v0.4h, v0.4h, v10.h[0]
fmul v1.4h, v1.4h, v10.h[1]
fmul v2.4h, v2.4h, v10.h[2]
fmul v3.4h, v3.4h, v10.h[3]
fmul v4.4h, v4.4h, v10.h[4]
fmul v5.4h, v5.4h, v10.h[5]
fmul v6.4h, v6.4h, v10.h[6]
fmul v7.4h, v7.4h, v10.h[7]
fmul v8.4h, v8.4h, v11.h[0]
fmul v9.4h, v9.4h, v11.h[1]

mov v0.d[1], v1.d[0]
mov v2.d[1], v3.d[0]
mov v4.d[1], v5.d[0]
mov v6.d[1], v7.d[0]
mov v8.d[1], v9.d[0]

// int16_t x = round(x)
Round v0, v2, v4, v6
fcvtas v8.8h, v8.8h

// y = (int8_t)x
sqxtn v0.8b, v0.8h
sqxtn2 v0.16b, v2.8h
sqxtn v1.8b, v4.8h
sqxtn2 v1.16b, v6.8h
sqxtn v2.8b, v8.8h

st1 {v0.16b, v1.16b}, [x10], #32
st1 {v2.8b}, [x10], x15

subs x12, x12, #1
bne LoopSz_10

Tile10End:
sub x4, x4, #10   // batch -= 10
add x0, x0, #80  // src += 10 * 4 * sizeof(float16_t)
add x1, x1, #40   // dst += 10 * 4 * sizeof(int8_t)
b TILE_10


TILE_8:
cmp x4, #8
blt TILE_4
sub x8, x7, #32 // src_step - 64
mov x9, x0   // src
mov x10, x1  // dst
mov x12, x3  // src_depth_quad

ld1 {v12.4s, v13.4s}, [x2], #32
fcvtn v8.4h, v12.4s
fcvtn2 v8.8h, v13.4s

LoopSz_8:
ld1 {v0.4h, v1.4h, v2.4h, v3.4h}, [x9], #32
ld1 {v4.4h, v5.4h, v6.4h, v7.4h}, [x9], x8

// float16_t x = x * quant_scale
fmul v0.4h, v0.4h, v8.h[0]
fmul v1.4h, v1.4h, v8.h[1]
fmul v2.4h, v2.4h, v8.h[2]
fmul v3.4h, v3.4h, v8.h[3]
fmul v4.4h, v4.4h, v8.h[4]
fmul v5.4h, v5.4h, v8.h[5]
fmul v6.4h, v6.4h, v8.h[6]
fmul v7.4h, v7.4h, v8.h[7]

mov v0.d[1], v1.d[0]
mov v2.d[1], v3.d[0]
mov v4.d[1], v5.d[0]
mov v6.d[1], v7.d[0]

// int16_t x = round(x)
Round v0, v2, v4, v6

// y = (int8_t)x
sqxtn v9.8b, v0.8h
sqxtn2 v9.16b, v2.8h
sqxtn v10.8b, v4.8h
sqxtn2 v10.16b, v6.8h

st1 {v9.16b, v10.16b}, [x10], x6

subs x12, x12, #1
bne LoopSz_8

Tile8End:
sub x4, x4, #8    // batch -= 8
add x0, x0, #64  // src += 8 * 4 * sizeof(float16_t)
add x1, x1, #32   // dst += 8 * 4 * sizeof(int8_t)
b TILE_8

TILE_4:
cmp x4, #4
blt TILE_2
mov x9, x0   // src
mov x10, x1  // dst
mov x12, x3  // src_depth_quad

ld1 {v12.4s}, [x2], #16
fcvtn v8.4h, v12.4s

LoopSz_4:
ld1 {v0.4h, v1.4h, v2.4h, v3.4h}, [x9], x7

// float16_t x = x * quant_scale
fmul v0.4h, v0.4h, v8.h[0]
fmul v1.4h, v1.4h, v8.h[1]
fmul v2.4h, v2.4h, v8.h[2]
fmul v3.4h, v3.4h, v8.h[3]

mov v0.d[1], v1.d[0]
mov v2.d[1], v3.d[0]
// int16_t x = round(x)
fcvtas v0.8h, v0.8h
fcvtas v2.8h, v2.8h

// y = (int8_t)x
sqxtn v4.8b, v0.8h
sqxtn2 v4.16b, v2.8h

st1 {v4.16b}, [x10], x6

subs x12, x12, #1
bne LoopSz_4

Tile4End:
sub x4, x4, #4    // batch -= 4
add x0, x0, #32   // src += 4 * 4 * sizeof(float16_t)
add x1, x1, #16   // dst += 4 * 4 * sizeof(int8_t)
b TILE_4


TILE_2:
cmp x4, #2
blt TILE_1
mov x9, x0   // src
mov x10, x1  // dst
mov x12, x3  // src_depth_quad

ld1 {v12.d}[0], [x2], #8
fcvtn v8.4h, v12.4s

LoopSz_2:
ld1 {v0.4h, v1.4h}, [x9], x7

// float16_t x = x * quant_scale
fmul v0.4h, v0.4h, v8.h[0]
fmul v1.4h, v1.4h, v8.h[1]
mov v0.d[1], v1.d[0]

// int16_t x = round(x)
fcvtas v0.8h, v0.8h

// y = (int8_t)x
sqxtn v2.8b, v0.8h

st1 {v2.8b}, [x10], x6

subs x12, x12, #1
bne LoopSz_2

Tile2End:
sub x4, x4, #2    // batch -= 2
add x0, x0, #16   // src += 2 * 4 * sizeof(float16_t)
add x1, x1, #8   // dst += 2 * 4 * sizeof(int8_t)
b TILE_2


TILE_1:
cmp x4, #1
blt End
mov x9, x0   // src
mov x10, x1  // dst
mov x12, x3  // src_depth_quad

ld1 {v12.s}[0], [x2], #4
fcvtn v8.4h, v12.4s

LoopSz_1:
ld1 {v0.4h}, [x9], x7

// float16_t x = x * quant_scale
fmul v0.4h, v0.4h, v8.h[0]
// int16_t x = round(x)
fcvtas v0.8h, v0.8h
// y = (int8_t)x
sqxtn v0.8b, v0.8h

st1 {v0.s}[0], [x10], x6

subs x12, x12, #1
bne LoopSz_1


End:
ldp d8,  d9,  [sp, #(16 * 3)]
ldp d10, d11, [sp, #(16 * 2)]
ldp d12, d13, [sp, #(16 * 1)]
ldp d14, d15, [sp], #(16 * 4)
ret

#endif
