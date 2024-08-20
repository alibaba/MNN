//
//  MNNDynamicQuantFP32.S
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
    fcvtas \z0\().4s, \z0\().4s
    fcvtas \z1\().4s, \z1\().4s
    fcvtas \z2\().4s, \z2\().4s
    fcvtas \z3\().4s, \z3\().4s
.endm

.macro Transpose z0, z1, z2, z3, t0, t1, t2, t3
    trn1 \t0\().4s, \z0\().4s, \z1\().4s
    trn1 \t1\().4s, \z2\().4s, \z3\().4s
    trn2 \t2\().4s, \z0\().4s, \z1\().4s
    trn2 \t3\().4s, \z2\().4s, \z3\().4s

    trn1 \z0\().2d, \t0\().2d, \t1\().2d
    trn1 \z1\().2d, \t2\().2d, \t3\().2d
    trn2 \z2\().2d, \t0\().2d, \t1\().2d
    trn2 \z3\().2d, \t2\().2d, \t3\().2d
.endm

.macro Add_4x4 d0, d1, d2, d3
    add \d0\().4s, \d1\().4s, \d0\().4s
    add \d2\().4s, \d3\().4s, \d2\().4s
    add \d0\().4s, \d0\().4s, \d2\().4s
.endm

//void MNNDynamicQuantFP32(const float* src, int8_t* dst, const float* scale, size_t src_depth_quad, size_t realSize, int pack)
asm_function MNNDynamicQuantFP32

// x0: src, x1:dst, x2:scale, x3:src_depth_quad, x4:realSize
stp d14, d15, [sp, #(-16 * 4)]!
stp d12, d13, [sp, #(16 * 1)]
stp d10, d11, [sp, #(16 * 2)]
stp d8,  d9,  [sp, #(16 * 3)]

Start:
lsl x6, x4, #2  // dst_step = batch * unit * sizeof(int8_t) = batch * 4 = batch << 2
lsl x7, x6, #2  // src_step = dst_step * 4 (sizeof(float32_t)) = dst_step << 2

TILE_8:
cmp x4, #8
blt TILE_4
sub x8, x7, #64 // src_step - 64
mov x9, x0   // src
mov x10, x1  // dst
//mov x11, x2  // scale
mov x12, x3  // src_depth_quad

// quant_scale: v8, 8(batch)*sizeof(float32_t)
ld1 {v8.4s, v9.4s}, [x2], #32

// int8 sum
LoopSz_8:
ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x9], #64
ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x9], x8

// float32_t x = x * quant_scale
fmul v0.4s, v0.4s, v8.s[0]
fmul v1.4s, v1.4s, v8.s[1]
fmul v2.4s, v2.4s, v8.s[2]
fmul v3.4s, v3.4s, v8.s[3]
fmul v4.4s, v4.4s, v9.s[0]
fmul v5.4s, v5.4s, v9.s[1]
fmul v6.4s, v6.4s, v9.s[2]
fmul v7.4s, v7.4s, v9.s[3]

// int32_t x = round(x)
Round v0, v1, v2, v3
Round v4, v5, v6, v7

// y = (int8_t)x
sqxtn v12.4h, v0.4s
sqxtn2 v12.8h, v1.4s
sqxtn v13.4h, v2.4s
sqxtn2 v13.8h, v3.4s
sqxtn v14.4h, v4.4s
sqxtn2 v14.8h, v5.4s
sqxtn v15.4h, v6.4s
sqxtn2 v15.8h, v7.4s

sqxtn v16.8b, v12.8h
sqxtn2 v16.16b, v13.8h
sqxtn v17.8b, v14.8h
sqxtn2 v17.16b, v15.8h

st1 {v16.16b, v17.16b}, [x10], x6

subs x12, x12, #1
bne LoopSz_8

Tile8End:
sub x4, x4, #8    // batch -= 8
add x0, x0, #128  // src += 8 * 4 * sizeof(float32_t)
add x1, x1, #32   // dst += 8 * 4 * sizeof(int8_t)
b TILE_8

TILE_4:
cmp x4, #4
blt TILE_1
mov x9, x0   // src
mov x10, x1  // dst
//mov x11, x2  // scale
mov x12, x3  // src_depth_quad

// quant_scale: v8, 4(batch)*sizeof(float32_t)
ld1 {v8.4s}, [x2], #16

// int8 sum
movi v10.4s, #0

LoopSz_4:
ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x9], x7

// float16_t x = x * quant_scale
fmul v0.4s, v0.4s, v8.s[0]
fmul v1.4s, v1.4s, v8.s[1]
fmul v2.4s, v2.4s, v8.s[2]
fmul v3.4s, v3.4s, v8.s[3]

// int32_t x = round(x)
Round v0, v1, v2, v3

// y = (int8_t)x
sqxtn v4.4h, v0.4s
sqxtn2 v4.8h, v1.4s
sqxtn v5.4h, v2.4s
sqxtn2 v5.8h, v3.4s

sqxtn v6.8b, v4.8h
sqxtn2 v6.16b, v5.8h

st1 {v6.16b}, [x10], x6

subs x12, x12, #1
bne LoopSz_4

Tile4End:
sub x4, x4, #4    // batch -= 4
add x0, x0, #64  // src += 4 * 4 * sizeof(float32_t)
add x1, x1, #16   // dst += 4 * 4 * sizeof(int8_t)
//add x2, x2, #16   // scale += 4 * sizeof(float32_t)
b TILE_4

TILE_1:
cmp x4, #1
blt End
mov x9, x0   // src
mov x10, x1  // dst
mov x12, x3  // src_depth_quad

// quant_scale: v8
ld1 {v8.s}[0], [x2], #4
movi v4.4s, #0
LoopSz_1:
ld1 {v0.4s}, [x9], x7

// float16_t x = x * quant_scale
fmul v0.4s, v0.4s, v8.s[0]
// int16_t x = round(x)
fcvtas v0.4s, v0.4s

// y = (int8_t)x
sqxtn v7.4h, v0.4s
sqxtn v7.8b, v7.8h

st1 {v7.s}[0], [x10], x6

subs x12, x12, #1
bne LoopSz_1

Tile1End:
subs x4, x4, #1    // batch -= 1
add x0, x0, #16    // src += 1 * 4 * sizeof(float32_t)
add x1, x1, #4    // dst += 1 * 4 * sizeof(int8_t)
//add x2, x2, #4    // scale += 1 * sizeof(float32_t)
bne TILE_1

End:
ldp d8,  d9,  [sp, #(16 * 3)]
ldp d10, d11, [sp, #(16 * 2)]
ldp d12, d13, [sp, #(16 * 1)]
ldp d14, d15, [sp], #(16 * 4)
ret

#endif
