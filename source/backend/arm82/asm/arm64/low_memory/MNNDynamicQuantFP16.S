//
//  MNNDynamicQuantFP16.S
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

//void MNNDynamicQuantFP16(const float* src, int8_t* dst, const float* scale, float* sum, size_t src_depth_quad, size_t realSize)
asm_function MNNDynamicQuantFP16

// x0: src, x1:dst, x2:scale, x3:sum, x4:src_depth_quad, x5:realSize
stp d14, d15, [sp, #(-16 * 4)]!
stp d12, d13, [sp, #(16 * 1)]
stp d10, d11, [sp, #(16 * 2)]
stp d8,  d9,  [sp, #(16 * 3)]

Start:
lsl x6, x5, #3  // dst_step = batch * unit * sizeof(int8_t) = batch * 8 = batch << 3
lsl x7, x6, #1  // src_step = dst_step * 2 (float16_t) = dst_step << 1

movi v29.16b, #1

TILE_12:
cmp x5, #12
blt TILE_10
mov x9, x0   // src
mov x10, x1  // dst
mov x12, x4  // src_depth_quad
sub x13, x7, #128 // src_step - 64
sub x14, x6, #64 // dst_step - 64

// quant_scale: v12, v13
ld1 {v12.8h}, [x2], #16
ld1 {v13.d}[0], [x2], #8
movi v23.4s, #0
movi v24.4s, #0
movi v25.4s, #0
movi v26.4s, #0
movi v27.4s, #0
movi v28.4s, #0

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
sqxtn v5.8b, v10.8h
sqxtn2 v5.16b, v11.8h

.inst 0x4e9d9417 // sdot v23.4s, v0.16b, v29.16b
.inst 0x4e9d9438 // sdot v24.4s, v1.16b, v29.16b
.inst 0x4e9d9459 // sdot v25.4s, v2.16b, v29.16b
.inst 0x4e9d947a // sdot v26.4s, v3.16b, v29.16b
.inst 0x4e9d949b // sdot v27.4s, v4.16b, v29.16b
.inst 0x4e9d94bc // sdot v28.4s, v5.16b, v29.16b

st1 {v0.16b, v1.16b, v2.16b, v3.16b}, [x10], #64
st1 {v4.16b, v5.16b}, [x10], x14

subs x12, x12, #1
bne LoopSz_12

addp v12.4s, v23.4s, v24.4s
addp v13.4s, v25.4s, v26.4s
addp v14.4s, v27.4s, v28.4s
st1 {v12.4s, v13.4s, v14.4s}, [x3], #48

Tile12End:
sub x5, x5, #12   // batch -= 12
add x0, x0, #192  // src += 12 * 8 * sizeof(float16_t)
add x1, x1, #96   // dst += 12 * 8 * sizeof(int8_t)
b TILE_12

TILE_10:
cmp x5, #10
blt TILE_8
mov x9, x0   // src
mov x10, x1  // dst
mov x12, x4  // src_depth_quad
sub x13, x7, #128 // src_step - 64
sub x14, x6, #64 // dst_step - 64

// quant_scale: v10, v11
ld1 {v10.8h}, [x2], #16
ld1 {v11.s}[0], [x2], #4
movi v24.4s, #0
movi v25.4s, #0
movi v26.4s, #0
movi v27.4s, #0
movi v28.4s, #0

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

.inst 0x4e9d9418 // sdot v24.4s, v0.16b, v29.16b
.inst 0x4e9d9439 // sdot v25.4s, v1.16b, v29.16b
.inst 0x4e9d945a // sdot v26.4s, v2.16b, v29.16b
.inst 0x4e9d947b // sdot v27.4s, v3.16b, v29.16b
.inst 0x4e9d949c // sdot v28.4s, v4.16b, v29.16b

st1 {v0.16b, v1.16b, v2.16b, v3.16b}, [x10], #64
st1 {v4.16b}, [x10], x14

subs x12, x12, #1
bne LoopSz_10

addp v13.4s, v24.4s, v25.4s
addp v14.4s, v26.4s, v27.4s
addp v15.4s, v28.4s, v28.4s
st1 {v13.4s, v14.4s}, [x3], #32
st1 {v15.d}[0], [x3], #8

Tile10End:
sub x5, x5, #10   // batch -= 10
add x0, x0, #160  // src += 10 * 8 * sizeof(float16_t)
add x1, x1, #80   // dst += 10 * 8 * sizeof(int8_t)
b TILE_10


TILE_8:
cmp x5, #8
blt TILE_1
sub x8, x7, #64 // src_step - 64
mov x9, x0   // src
mov x10, x1  // dst
mov x12, x4  // src_depth_quad

// quant_scale: v8
ld1 {v8.8h}, [x2], #16
movi v25.4s, #0
movi v26.4s, #0
movi v27.4s, #0
movi v28.4s, #0

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

.inst 0x4e9d9539 // sdot v25.4s, v9.16b, v29.16b
.inst 0x4e9d955a // sdot v26.4s, v10.16b, v29.16b
.inst 0x4e9d957b // sdot v27.4s, v11.16b, v29.16b
.inst 0x4e9d959c // sdot v28.4s, v12.16b, v29.16b

st1 {v9.16b, v10.16b, v11.16b, v12.16b}, [x10], x6

subs x12, x12, #1
bne LoopSz_8

addp v14.4s, v25.4s, v26.4s
addp v15.4s, v27.4s, v28.4s
st1 {v14.4s, v15.4s}, [x3], #32

Tile8End:
sub x5, x5, #8    // batch -= 8
add x0, x0, #128  // src += 8 * 8 * sizeof(float16_t)
add x1, x1, #64   // dst += 8 * 8 * sizeof(int8_t)
b TILE_8

TILE_4:
cmp x5, #4
blt TILE_2
mov x9, x0   // src
mov x10, x1  // dst
mov x12, x4  // src_depth_quad

// quant_scale: v8
ld1 {v8.d}[0], [x2], #8
movi v27.4s, #0
movi v28.4s, #0

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

.inst 0x4e9d949b // sdot v27.4s, v4.16b, v29.16b
.inst 0x4e9d94bc // sdot v28.4s, v5.16b, v29.16b

st1 {v4.16b, v5.16b}, [x10], x6

subs x12, x12, #1
bne LoopSz_4

addp v26.4s, v27.4s, v28.4s
st1 {v26.4s}, [x3], #16

Tile4End:
sub x5, x5, #4    // batch -= 4
add x0, x0, #64   // src += 4 * 8 * sizeof(float16_t)
add x1, x1, #32   // dst += 4 * 8 * sizeof(int8_t)
b TILE_4


TILE_2:
cmp x5, #2
blt TILE_1
mov x9, x0   // src
mov x10, x1  // dst
mov x12, x4  // src_depth_quad

// quant_scale: v8
ld1 {v8.s}[0], [x2], #4
movi v28.4s, #0

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
.inst 0x4e9d945c // sdot v28.4s, v2.16b, v29.16b

st1 {v2.16b}, [x10], x6

subs x12, x12, #1
bne LoopSz_2

addp v27.4s, v28.4s, v28.4s
st1 {v27.d}[0], [x3], #8

Tile2End:
sub x5, x5, #2    // batch -= 2
add x0, x0, #32   // src += 2 * 8 * sizeof(float16_t)
add x1, x1, #16   // dst += 2 * 8 * sizeof(int8_t)
b TILE_2


TILE_1:
cmp x5, #1
blt End
mov x9, x0   // src
mov x10, x1  // dst
mov x12, x4  // src_depth_quad

// quant_scale: v8
ld1 {v8.h}[0], [x2], #2
movi v28.4s, #0

LoopSz_1:
ld1 {v0.8h}, [x9], x7

// float16_t x = x * quant_scale
fmul v0.8h, v0.8h, v8.h[0]
// int16_t x = round(x)
fcvtas v0.8h, v0.8h
// y = (int8_t)x
sqxtn v0.8b, v0.8h
.inst 0x4e9d941c // sdot v28.4s, v0.16b, v29.16b

st1 {v0.8b}, [x10], x6

subs x12, x12, #1
bne LoopSz_1

addp v27.4s, v28.4s, v28.4s
st1 {v27.s}[0], [x3], #4

Tile1End:
sub x5, x5, #1   // batch -= 1
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