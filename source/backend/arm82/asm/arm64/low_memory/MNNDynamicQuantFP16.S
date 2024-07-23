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

//void MNNDynamicQuantFP16(const float* src, int8_t* dst, const float* scale, size_t src_depth_quad, size_t realSize, int pack)
asm_function MNNDynamicQuantFP16

// Feature: quant and reorder C8->C4

// x0: src, x1:dst, x2:scale, x3:src_depth_quad, x4:realSize
stp d14, d15, [sp, #(-16 * 4)]!
stp d12, d13, [sp, #(16 * 1)]
stp d10, d11, [sp, #(16 * 2)]
stp d8,  d9,  [sp, #(16 * 3)]

Start:
lsl x6, x4, #3  // dst_step = batch * (2*unit) * sizeof(int8_t) = batch * 8 = batch << 3
lsl x7, x4, #4  // src_step = batch * pack * sizeof(float16) = batch * 8 * 2 = batch << 4
lsl x8, x4, #2  // 4 * plane
add x11, x1, x8 // second N*4

TILE_12:
cmp x4, #12
blt TILE_10
mov x9, x0   // src
mov x10, x1  // dst
mov x15, x11 // second dst 
mov x12, x3  // src_depth_quad
sub x13, x7, #128 // src_step - 64

// quant_scale: v12, v13, v14
// ld1 {v12.8h}, [x2], #16
// ld1 {v13.d}[0], [x2], #8
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

uzp1 v6.4s, v0.4s, v1.4s
uzp1 v7.4s, v2.4s, v3.4s
uzp1 v8.4s, v4.4s, v5.4s
uzp2 v9.4s, v0.4s, v1.4s
uzp2 v10.4s, v2.4s, v3.4s
uzp2 v11.4s, v4.4s, v5.4s

st1 {v6.16b, v7.16b, v8.16b}, [x10], x6
st1 {v9.16b, v10.16b, v11.16b}, [x15], x6

//st1 {v0.16b, v1.16b, v2.16b, v3.16b}, [x10], #64
//st1 {v4.16b, v5.16b}, [x10], x14

subs x12, x12, #1
bne LoopSz_12

Tile12End:
sub x4, x4, #12   // batch -= 12
add x0, x0, #192  // src += 12 * 8 * sizeof(float16_t)
add x1, x1, #48   // dst += 12 * 4 * sizeof(int8_t)
add x11, x11, #48
b TILE_12

TILE_10:
cmp x4, #10
blt TILE_8
mov x9, x0   // src
mov x10, x1  // dst
mov x15, x11 // second dst 
mov x12, x3  // src_depth_quad
sub x13, x7, #128 // src_step - 128
sub x14, x6, #32 // dst_step - 32

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

uzp1 v6.4s, v0.4s, v1.4s // 0 1 2 3
uzp1 v7.4s, v2.4s, v3.4s // 4 5 6 7
uzp1 v8.4s, v4.4s, v4.4s // 8 9 8 9
uzp2 v12.4s, v0.4s, v1.4s
uzp2 v13.4s, v2.4s, v3.4s
uzp2 v14.4s, v4.4s, v4.4s
st1 {v6.16b, v7.16b}, [x10], #32
st1 {v8.d}[0], [x10], x14
st1 {v12.16b, v13.16b}, [x15], #32
st1 {v14.d}[0], [x15], x14

// st1 {v0.16b, v1.16b, v2.16b, v3.16b}, [x10], #64
// st1 {v4.16b}, [x10], x14

subs x12, x12, #1
bne LoopSz_10

Tile10End:
sub x4, x4, #10   // batch -= 10
add x0, x0, #160  // src += 10 * 8 * sizeof(float16_t)
add x1, x1, #40   // dst += 10 * 4 * sizeof(int8_t)
add x11, x11, #40
b TILE_10


TILE_8:
cmp x4, #8
blt TILE_1
sub x8, x7, #64 // src_step - 64
mov x9, x0   // src
mov x10, x1  // dst
mov x15, x11 // second dst 
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

uzp1 v6.4s, v9.4s, v10.4s // 0 1 2 3 first
uzp1 v7.4s, v11.4s, v12.4s // 4 5 6 7
uzp2 v14.4s, v9.4s, v10.4s // 0 1 2 3 second
uzp2 v15.4s, v11.4s, v12.4s // 4 5 6 7
st1 {v6.16b, v7.16b}, [x10], x6
st1 {v14.16b, v15.16b}, [x15], x6
//st1 {v9.16b, v10.16b, v11.16b, v12.16b}, [x10], x6

subs x12, x12, #1
bne LoopSz_8

Tile8End:
sub x4, x4, #8    // batch -= 8
add x0, x0, #128  // src += 8 * 8 * sizeof(float16_t)
add x1, x1, #32   // dst += 8 * 4 * sizeof(int8_t)
add x11, x11, #32
b TILE_8

TILE_4:
cmp x4, #4
blt TILE_2
mov x9, x0   // src
mov x10, x1  // dst
mov x15, x11 // second dst 
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

uzp1 v6.4s, v4.4s, v5.4s // 0 1 2 3 first
uzp2 v14.4s, v4.4s, v5.4s // 0 1 2 3 second
st1 {v6.16b}, [x10], x6
st1 {v14.16b}, [x15], x6
//st1 {v4.16b, v5.16b}, [x10], x6

subs x12, x12, #1
bne LoopSz_4

Tile4End:
sub x4, x4, #4    // batch -= 4
add x0, x0, #64   // src += 4 * 8 * sizeof(float16_t)
add x1, x1, #16   // dst += 4 * 4 * sizeof(int8_t)
add x11, x11, #16
b TILE_4


TILE_2:
cmp x4, #2
blt TILE_1
mov x9, x0   // src
mov x10, x1  // dst
mov x15, x11 // second dst 
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

st1 {v2.d}[0], [x10], x6
st1 {v2.d}[1], [x15], x6
//st1 {v2.16b}, [x10], x6

subs x12, x12, #1
bne LoopSz_2

Tile2End:
sub x4, x4, #2    // batch -= 2
add x0, x0, #32   // src += 2 * 8 * sizeof(float16_t)
add x1, x1, #8   // dst += 2 * 4 * sizeof(int8_t)
add x11, x11, #8
b TILE_2


TILE_1:
cmp x4, #1
blt End
mov x9, x0   // src
mov x10, x1  // dst
mov x15, x11 // second dst 
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

st1 {v0.s}[0], [x10], x6
st1 {v0.s}[1], [x15], x6

subs x12, x12, #1
bne LoopSz_1

Tile1End:
sub x4, x4, #1   // batch -= 1
add x0, x0, #16  // src += 1 * 8 * sizeof(float16_t)
add x1, x1, #4   // dst += 1 * 4 * sizeof(int8_t)
add x11, x11, #4
b TILE_1


End:
ldp d8,  d9,  [sp, #(16 * 3)]
ldp d10, d11, [sp, #(16 * 2)]
ldp d12, d13, [sp, #(16 * 1)]
ldp d14, d15, [sp], #(16 * 4)
ret

#endif
