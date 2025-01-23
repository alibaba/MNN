//
//  MNNDynamicQuantFP16_Pack8.S
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
asm_function MNNDynamicQuantFP16_Pack8

// x0: src, x1:dst, x2:scale, x3:src_depth_quad, x4:realSize
stp d14, d15, [sp, #(-16 * 4)]!
stp d12, d13, [sp, #(16 * 1)]
stp d10, d11, [sp, #(16 * 2)]
stp d8,  d9,  [sp, #(16 * 3)]

Start:
lsl x6, x4, #3  // dst_step = batch * (2*unit) * sizeof(int8_t) = batch * 8 = batch << 3
lsl x7, x4, #4  // src_step = batch * pack * sizeof(float16) = batch * 8 * 2 = batch << 4

TILE_24:
cmp x4, #24
blt TILE_16
mov x9, x0   // src
mov x10, x1  // dst
sub x15, x6, #128
mov x12, x3  // src_depth_quad
sub x13, x7, #320 // src_step - 320

ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [x2], #64
ld1 {v16.4s, v17.4s}, [x2], #32
fcvtn v12.4h, v12.4s
fcvtn2 v12.8h, v13.4s
fcvtn v13.4h, v14.4s
fcvtn2 v13.8h, v15.4s
fcvtn v14.4h, v16.4s
fcvtn2 v14.8h, v17.4s

LoopSz_24:
ld1 {v0.8h, v1.8h, v2.8h, v3.8h}, [x9], #64
ld1 {v4.8h, v5.8h, v6.8h, v7.8h}, [x9], #64
ld1 {v8.8h, v9.8h, v10.8h, v11.8h}, [x9], #64
ld1 {v15.8h, v16.8h, v17.8h, v18.8h}, [x9], #64
ld1 {v19.8h, v20.8h, v21.8h, v22.8h}, [x9], #64
ld1 {v23.8h, v24.8h, v25.8h, v26.8h}, [x9], x13

// float16_t x = x * quant_scale
fmul v0.8h, v0.8h, v12.h[0]
fmul v1.8h, v1.8h, v12.h[1]
fmul v2.8h, v2.8h, v12.h[2]
fmul v3.8h, v3.8h, v12.h[3]
fmul v4.8h, v4.8h, v12.h[4]
fmul v5.8h, v5.8h, v12.h[5]
fmul v6.8h, v6.8h, v12.h[6]
fmul v7.8h, v7.8h, v12.h[7]
fmul v8.8h, v8.8h, v13.h[0]
fmul v9.8h, v9.8h, v13.h[1]
fmul v10.8h, v10.8h, v13.h[2]
fmul v11.8h, v11.8h, v13.h[3]
fmul v15.8h, v15.8h, v13.h[4]
fmul v16.8h, v16.8h, v13.h[5]
fmul v17.8h, v17.8h, v13.h[6]
fmul v18.8h, v18.8h, v13.h[7]

fmul v19.8h, v19.8h, v14.h[0]
fmul v20.8h, v20.8h, v14.h[1]
fmul v21.8h, v21.8h, v14.h[2]
fmul v22.8h, v22.8h, v14.h[3]
fmul v23.8h, v23.8h, v14.h[4]
fmul v24.8h, v24.8h, v14.h[5]
fmul v25.8h, v25.8h, v14.h[6]
fmul v26.8h, v26.8h, v14.h[7]

// int16_t x = round(x)
Round v0, v1, v2, v3
Round v4, v5, v6, v7
Round v8, v9, v10, v11
Round v15, v16, v17, v18
Round v19, v20, v21, v22
Round v23, v24, v25, v26

// y = (int8_t)x
sqxtn v27.8b, v0.8h
sqxtn2 v27.16b, v1.8h
sqxtn v28.8b, v2.8h
sqxtn2 v28.16b, v3.8h
sqxtn v29.8b, v4.8h
sqxtn2 v29.16b, v5.8h
sqxtn v30.8b, v6.8h
sqxtn2 v30.16b, v7.8h
sqxtn v0.8b, v8.8h
sqxtn2 v0.16b, v9.8h
sqxtn v1.8b, v10.8h
sqxtn2 v1.16b, v11.8h
sqxtn v2.8b, v15.8h
sqxtn2 v2.16b, v16.8h
sqxtn v3.8b, v17.8h
sqxtn2 v3.16b, v18.8h
sqxtn v4.8b, v19.8h
sqxtn2 v4.16b, v20.8h
sqxtn v5.8b, v21.8h
sqxtn2 v5.16b, v22.8h
sqxtn v6.8b, v23.8h
sqxtn2 v6.16b, v24.8h
sqxtn v7.8b, v25.8h
sqxtn2 v7.16b, v26.8h

st1 {v27.16b, v28.16b, v29.16b, v30.16b}, [x10], #64
st1 {v0.16b, v1.16b, v2.16b, v3.16b}, [x10], #64
st1 {v4.16b, v5.16b, v6.16b, v7.16b}, [x10], x15

subs x12, x12, #1
bne LoopSz_24

Tile24End:
sub x4, x4, #24   // batch -= 24
add x0, x0, #384  // src += 24 * 8 * sizeof(float16_t)
add x1, x1, #192   // dst += 24 * 8 * sizeof(int8_t)
b TILE_24

TILE_16:
cmp x4, #16
blt TILE_12
mov x9, x0   // src
mov x10, x1  // dst
sub x15, x6, #64
mov x12, x3  // src_depth_quad
sub x13, x7, #192 // src_step - 192

ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [x2], #64
fcvtn v12.4h, v12.4s
fcvtn2 v12.8h, v13.4s
fcvtn v13.4h, v14.4s
fcvtn2 v13.8h, v15.4s

LoopSz_16:
ld1 {v0.8h, v1.8h, v2.8h, v3.8h}, [x9], #64
ld1 {v4.8h, v5.8h, v6.8h, v7.8h}, [x9], #64
ld1 {v8.8h, v9.8h, v10.8h, v11.8h}, [x9], #64
ld1 {v15.8h, v16.8h, v17.8h, v18.8h}, [x9], x13

// float16_t x = x * quant_scale
fmul v0.8h, v0.8h, v12.h[0]
fmul v1.8h, v1.8h, v12.h[1]
fmul v2.8h, v2.8h, v12.h[2]
fmul v3.8h, v3.8h, v12.h[3]
fmul v4.8h, v4.8h, v12.h[4]
fmul v5.8h, v5.8h, v12.h[5]
fmul v6.8h, v6.8h, v12.h[6]
fmul v7.8h, v7.8h, v12.h[7]
fmul v8.8h, v8.8h, v13.h[0]
fmul v9.8h, v9.8h, v13.h[1]
fmul v10.8h, v10.8h, v13.h[2]
fmul v11.8h, v11.8h, v13.h[3]
fmul v15.8h, v15.8h, v13.h[4]
fmul v16.8h, v16.8h, v13.h[5]
fmul v17.8h, v17.8h, v13.h[6]
fmul v18.8h, v18.8h, v13.h[7]

// int16_t x = round(x)
Round v0, v1, v2, v3
Round v4, v5, v6, v7
Round v8, v9, v10, v11
Round v15, v16, v17, v18

// y = (int8_t)x
sqxtn v19.8b, v0.8h
sqxtn2 v19.16b, v1.8h
sqxtn v20.8b, v2.8h
sqxtn2 v20.16b, v3.8h
sqxtn v21.8b, v4.8h
sqxtn2 v21.16b, v5.8h
sqxtn v22.8b, v6.8h
sqxtn2 v22.16b, v7.8h
sqxtn v23.8b, v8.8h
sqxtn2 v23.16b, v9.8h
sqxtn v24.8b, v10.8h
sqxtn2 v24.16b, v11.8h
sqxtn v25.8b, v15.8h
sqxtn2 v25.16b, v16.8h
sqxtn v26.8b, v17.8h
sqxtn2 v26.16b, v18.8h

st1 {v19.16b, v20.16b, v21.16b, v22.16b}, [x10], #64
st1 {v23.16b, v24.16b, v25.16b, v26.16b}, [x10], x15

subs x12, x12, #1
bne LoopSz_16

Tile16End:
sub x4, x4, #16   // batch -= 16
add x0, x0, #256  // src += 16 * 8 * sizeof(float16_t)
add x1, x1, #128   // dst += 16 * 8 * sizeof(int8_t)
b TILE_16

TILE_12:
cmp x4, #12
blt TILE_10
mov x9, x0   // src
mov x10, x1  // dst
sub x15, x6, #64
mov x12, x3  // src_depth_quad
sub x13, x7, #128 // src_step - 128

ld1 {v12.4s, v13.4s, v14.4s}, [x2], #48
fcvtn v12.4h, v12.4s
fcvtn2 v12.8h, v13.4s
fcvtn v13.4h, v14.4s

LoopSz_12:
ld1 {v0.8h, v1.8h, v2.8h, v3.8h}, [x9], #64
ld1 {v4.8h, v5.8h, v6.8h, v7.8h}, [x9], #64
ld1 {v8.8h, v9.8h, v10.8h, v11.8h}, [x9], x13

// float16_t x = x * quant_scale
fmul v0.8h, v0.8h, v12.h[0]
fmul v1.8h, v1.8h, v12.h[1]
fmul v2.8h, v2.8h, v12.h[2]
fmul v3.8h, v3.8h, v12.h[3]
fmul v4.8h, v4.8h, v12.h[4]
fmul v5.8h, v5.8h, v12.h[5]
fmul v6.8h, v6.8h, v12.h[6]
fmul v7.8h, v7.8h, v12.h[7]
fmul v8.8h, v8.8h, v13.h[0]
fmul v9.8h, v9.8h, v13.h[1]
fmul v10.8h, v10.8h, v13.h[2]
fmul v11.8h, v11.8h, v13.h[3]

// int16_t x = round(x)
Round v0, v1, v2, v3
Round v4, v5, v6, v7
Round v8, v9, v10, v11

// y = (int8_t)x
sqxtn  v14.8b, v0.8h
sqxtn2 v14.16b, v1.8h
sqxtn  v15.8b, v2.8h
sqxtn2 v15.16b, v3.8h
sqxtn  v16.8b, v4.8h
sqxtn2 v16.16b, v5.8h
sqxtn  v17.8b, v6.8h
sqxtn2 v17.16b, v7.8h
sqxtn  v18.8b, v8.8h
sqxtn2 v18.16b, v9.8h
sqxtn  v19.8b, v10.8h
sqxtn2 v19.16b, v11.8h

st1 {v14.16b, v15.16b, v16.16b, v17.16b}, [x10], #64
st1 {v18.16b, v19.16b}, [x10], x15

subs x12, x12, #1
bne LoopSz_12

Tile12End:
sub x4, x4, #12   // batch -= 12
add x0, x0, #192  // src += 12 * 8 * sizeof(float16_t)
add x1, x1, #96   // dst += 12 * 8 * sizeof(int8_t)
b TILE_12

TILE_10:
cmp x4, #10
blt TILE_8
mov x9, x0   // src
mov x10, x1  // dst
mov x12, x3  // src_depth_quad
sub x13, x7, #128 // src_step - 128
sub x15, x6, #64

// quant_scale: v10, v11
//ld1 {v10.8h}, [x2], #16
//ld1 {v11.s}[0], [x2], #4
ld1 {v12.4s, v13.4s}, [x2], #32
ld1 {v14.d}[0], [x2], #8
fcvtn v10.4h, v12.4s
fcvtn2 v10.8h, v13.4s
fcvtn v11.4h, v14.4s

LoopSz_10:
ld1 {v0.8h, v1.8h, v2.8h, v3.8h}, [x9], #64
ld1 {v4.8h, v5.8h, v6.8h, v7.8h}, [x9], #64
ld1 {v8.8h, v9.8h}, [x9], x13

// float16_t x = x * quant_scale
fmul v0.8h, v0.8h, v10.h[0]
fmul v1.8h, v1.8h, v10.h[1]
fmul v2.8h, v2.8h, v10.h[2]
fmul v3.8h, v3.8h, v10.h[3]
fmul v4.8h, v4.8h, v10.h[4]
fmul v5.8h, v5.8h, v10.h[5]
fmul v6.8h, v6.8h, v10.h[6]
fmul v7.8h, v7.8h, v10.h[7]
fmul v8.8h, v8.8h, v11.h[0]
fmul v9.8h, v9.8h, v11.h[1]

// int16_t x = round(x)
Round v0, v1, v2, v3
Round v4, v5, v6, v7
fcvtas v8.8h, v8.8h
fcvtas v9.8h, v9.8h

// y = (int8_t)x
sqxtn v0.8b, v0.8h
sqxtn2 v0.16b, v1.8h
sqxtn v1.8b, v2.8h
sqxtn2 v1.16b, v3.8h
sqxtn v2.8b, v4.8h
sqxtn2 v2.16b, v5.8h
sqxtn v3.8b, v6.8h
sqxtn2 v3.16b, v7.8h
sqxtn v4.8b, v8.8h
sqxtn2 v4.16b, v9.8h

st1 {v0.16b, v1.16b, v2.16b, v3.16b}, [x10], #64
st1 {v4.16b}, [x10], x15


subs x12, x12, #1
bne LoopSz_10

Tile10End:
sub x4, x4, #10   // batch -= 10
add x0, x0, #160  // src += 10 * 8 * sizeof(float16_t)
add x1, x1, #80   // dst += 10 * 4 * sizeof(int8_t)
b TILE_10


TILE_8:
cmp x4, #8
blt TILE_1
sub x8, x7, #64 // src_step - 64
mov x9, x0   // src
mov x10, x1  // dst
mov x12, x3  // src_depth_quad

// quant_scale: v8
//ld1 {v8.8h}, [x2], #16
ld1 {v12.4s, v13.4s}, [x2], #32
fcvtn v8.4h, v12.4s
fcvtn2 v8.8h, v13.4s

LoopSz_8:
ld1 {v0.8h, v1.8h, v2.8h, v3.8h}, [x9], #64
ld1 {v4.8h, v5.8h, v6.8h, v7.8h}, [x9], x8

// float16_t x = x * quant_scale
fmul v0.8h, v0.8h, v8.h[0]
fmul v1.8h, v1.8h, v8.h[1]
fmul v2.8h, v2.8h, v8.h[2]
fmul v3.8h, v3.8h, v8.h[3]
fmul v4.8h, v4.8h, v8.h[4]
fmul v5.8h, v5.8h, v8.h[5]
fmul v6.8h, v6.8h, v8.h[6]
fmul v7.8h, v7.8h, v8.h[7]

// int16_t x = round(x)
Round v0, v1, v2, v3
Round v4, v5, v6, v7

// y = (int8_t)x
sqxtn v9.8b, v0.8h
sqxtn2 v9.16b, v1.8h
sqxtn v10.8b, v2.8h
sqxtn2 v10.16b, v3.8h
sqxtn v11.8b, v4.8h
sqxtn2 v11.16b, v5.8h
sqxtn v12.8b, v6.8h
sqxtn2 v12.16b, v7.8h

st1 {v9.16b, v10.16b, v11.16b, v12.16b}, [x10], x6

subs x12, x12, #1
bne LoopSz_8

Tile8End:
sub x4, x4, #8    // batch -= 8
add x0, x0, #128  // src += 8 * 8 * sizeof(float16_t)
add x1, x1, #64   // dst += 8 * 8 * sizeof(int8_t)
b TILE_8

TILE_4:
cmp x4, #4
blt TILE_2
mov x9, x0   // src
mov x10, x1  // dst
mov x12, x3  // src_depth_quad

// quant_scale: v8
//ld1 {v8.d}[0], [x2], #8
ld1 {v12.4s}, [x2], #16
fcvtn v8.4h, v12.4s

LoopSz_4:
ld1 {v0.8h, v1.8h, v2.8h, v3.8h}, [x9], x7

// float16_t x = x * quant_scale
fmul v0.8h, v0.8h, v8.h[0]
fmul v1.8h, v1.8h, v8.h[1]
fmul v2.8h, v2.8h, v8.h[2]
fmul v3.8h, v3.8h, v8.h[3]

// int16_t x = round(x)
Round v0, v1, v2, v3

// y = (int8_t)x
sqxtn v4.8b, v0.8h
sqxtn2 v4.16b, v1.8h
sqxtn v5.8b, v2.8h
sqxtn2 v5.16b, v3.8h

st1 {v4.16b, v5.16b}, [x10], x6

subs x12, x12, #1
bne LoopSz_4

Tile4End:
sub x4, x4, #4    // batch -= 4
add x0, x0, #64   // src += 4 * 8 * sizeof(float16_t)
add x1, x1, #32   // dst += 4 * 8 * sizeof(int8_t)
b TILE_4


TILE_2:
cmp x4, #2
blt TILE_1
mov x9, x0   // src
mov x10, x1  // dst
mov x12, x3  // src_depth_quad

// quant_scale: v8
//ld1 {v8.s}[0], [x2], #4
ld1 {v12.d}[0], [x2], #8
fcvtn v8.4h, v12.4s

LoopSz_2:
ld1 {v0.8h, v1.8h}, [x9], x7

// float16_t x = x * quant_scale
fmul v0.8h, v0.8h, v8.h[0]
fmul v1.8h, v1.8h, v8.h[1]

// int16_t x = round(x)
fcvtas v0.8h, v0.8h
fcvtas v1.8h, v1.8h

// y = (int8_t)x
sqxtn v2.8b, v0.8h
sqxtn2 v2.16b, v1.8h

st1 {v2.16b}, [x10], x6

subs x12, x12, #1
bne LoopSz_2

Tile2End:
sub x4, x4, #2    // batch -= 2
add x0, x0, #32   // src += 2 * 8 * sizeof(float16_t)
add x1, x1, #16   // dst += 2 * 8 * sizeof(int8_t)
b TILE_2


TILE_1:
cmp x4, #1
blt End
mov x9, x0   // src
mov x10, x1  // dst
mov x12, x3  // src_depth_quad

// quant_scale: v8
//ld1 {v8.h}[0], [x2], #2
ld1 {v12.s}[0], [x2], #4
fcvtn v8.4h, v12.4s

LoopSz_1:
ld1 {v0.8h}, [x9], x7

// float16_t x = x * quant_scale
fmul v0.8h, v0.8h, v8.h[0]
// int16_t x = round(x)
fcvtas v0.8h, v0.8h
// y = (int8_t)x
sqxtn v0.8b, v0.8h

st1 {v0.8b}, [x10], x6

subs x12, x12, #1
bne LoopSz_1

Tile1End:
sub x4, x4, #1   // batch -= 1
add x0, x0, #16  // src += 1 * 8 * sizeof(float16_t)
add x1, x1, #8   // dst += 1 * 8 * sizeof(int8_t)
b TILE_1


End:
ldp d8,  d9,  [sp, #(16 * 3)]
ldp d10, d11, [sp, #(16 * 2)]
ldp d12, d13, [sp, #(16 * 1)]
ldp d14, d15, [sp], #(16 * 4)
ret

#endif
