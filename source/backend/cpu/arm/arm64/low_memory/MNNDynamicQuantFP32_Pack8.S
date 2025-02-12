//
//  MNNDynamicQuantFP32_Pack8.S
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

//void MNNDynamicQuantFP32_Pack8(const float* src, int8_t* dst, const float* scale, size_t src_depth_quad, size_t realSize, int pack)
asm_function MNNDynamicQuantFP32_Pack8

// x0: src, x1:dst, x2:scale, x3:src_depth_quad, x4:realSize
stp d14, d15, [sp, #(-16 * 4)]!
stp d12, d13, [sp, #(16 * 1)]
stp d10, d11, [sp, #(16 * 2)]
stp d8,  d9,  [sp, #(16 * 3)]

Start:
lsl x6, x4, #3  // dst_step = batch * unit * sizeof(int8_t) = batch * 8 = batch << 3
lsl x7, x6, #2  // src_step = dst_step * 4 (sizeof(float32_t)) = dst_step << 2

TILE_10:
cmp x4, #8
blt TILE_8
sub x8, x7, #256 // src_step - 256
sub x11, x6, #64 // dst_step-64
mov x9, x0   // src
mov x10, x1  // dst
mov x12, x3  // src_depth_quad

// quant_scale: v8, 8(batch)*sizeof(float32_t)
ld1 {v16.4s, v17.4s}, [x2], #32
ld1 {v30.8b}, [x2], #8

LoopSz_10:
ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x9], #64 // E0, E0, E1, E1
ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x9], #64 // E2, E2, E3, E3
ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x9], #64 // E4, E4, E5, E5
ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [x9], #64 // E6, E6, E7, E7
ld1 {v18.4s, v19.4s, v20.4s, v21.4s}, [x9], x8 // E8, E8, E9, E9

// float32_t x = x * quant_scale
fmul v0.4s, v0.4s, v16.s[0]
fmul v1.4s, v1.4s, v16.s[0]
fmul v2.4s, v2.4s, v16.s[1]
fmul v3.4s, v3.4s, v16.s[1]
fmul v4.4s, v4.4s, v16.s[2]
fmul v5.4s, v5.4s, v16.s[2]
fmul v6.4s, v6.4s, v16.s[3]
fmul v7.4s, v7.4s, v16.s[3]

fmul v8.4s, v8.4s, v17.s[0]
fmul v9.4s, v9.4s, v17.s[0]
fmul v10.4s, v10.4s, v17.s[1]
fmul v11.4s, v11.4s, v17.s[1]
fmul v12.4s, v12.4s, v17.s[2]
fmul v13.4s, v13.4s, v17.s[2]
fmul v14.4s, v14.4s, v17.s[3]
fmul v15.4s, v15.4s, v17.s[3]

fmul v18.4s, v18.4s, v30.s[0]
fmul v19.4s, v19.4s, v30.s[0]
fmul v20.4s, v20.4s, v30.s[1]
fmul v21.4s, v21.4s, v30.s[1]

// int32_t x = round(x)
Round v0, v1, v2, v3
Round v4, v5, v6, v7
Round v8, v9, v10, v11
Round v12, v13, v14, v15
Round v18, v19, v20, v21

// y = (int8_t)x
sqxtn v22.4h, v0.4s
sqxtn2 v22.8h, v1.4s
sqxtn v23.4h, v2.4s
sqxtn2 v23.8h, v3.4s
sqxtn v24.4h, v4.4s
sqxtn2 v24.8h, v5.4s
sqxtn v25.4h, v6.4s
sqxtn2 v25.8h, v7.4s

sqxtn v26.4h, v8.4s
sqxtn2 v26.8h, v9.4s
sqxtn v27.4h, v10.4s
sqxtn2 v27.8h, v11.4s
sqxtn v28.4h, v12.4s
sqxtn2 v28.8h, v13.4s
sqxtn v29.4h, v14.4s
sqxtn2 v29.8h, v15.4s

sqxtn v0.4h, v18.4s
sqxtn2 v0.8h, v19.4s
sqxtn v1.4h, v20.4s
sqxtn2 v1.8h, v21.4s

sqxtn v2.8b, v22.8h
sqxtn2 v2.16b, v23.8h
sqxtn v3.8b, v24.8h
sqxtn2 v3.16b, v25.8h
sqxtn v4.8b, v26.8h
sqxtn2 v4.16b, v27.8h
sqxtn v5.8b, v28.8h
sqxtn2 v5.16b, v29.8h
sqxtn v6.8b, v0.8h
sqxtn2 v6.16b, v1.8h


st1 {v2.16b, v3.16b, v4.16b, v5.16b}, [x10], #64
st1 {v6.16b}, [x10], x11

subs x12, x12, #1
bne LoopSz_10

Tile10End:
sub x4, x4, #10    // batch -= 10
add x0, x0, #320  // src += 10 * 8 * sizeof(float32_t)
add x1, x1, #80   // dst += 10 * 8 * sizeof(int8_t)
b TILE_10

TILE_8:
cmp x4, #8
blt TILE_4
sub x8, x7, #192 // src_step - 192
mov x9, x0   // src
mov x10, x1  // dst
mov x12, x3  // src_depth_quad

// quant_scale: v8, 8(batch)*sizeof(float32_t)
ld1 {v16.4s, v17.4s}, [x2], #32

LoopSz_8:
ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x9], #64 // E0, E0, E1, E1
ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x9], #64 // E2, E2, E3, E3
ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x9], #64 // E4, E4, E5, E5
ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [x9], x8 // E6, E6, E7, E7

// float32_t x = x * quant_scale
fmul v0.4s, v0.4s, v16.s[0]
fmul v1.4s, v1.4s, v16.s[0]
fmul v2.4s, v2.4s, v16.s[1]
fmul v3.4s, v3.4s, v16.s[1]
fmul v4.4s, v4.4s, v16.s[2]
fmul v5.4s, v5.4s, v16.s[2]
fmul v6.4s, v6.4s, v16.s[3]
fmul v7.4s, v7.4s, v16.s[3]

fmul v8.4s, v8.4s, v17.s[0]
fmul v9.4s, v9.4s, v17.s[0]
fmul v10.4s, v10.4s, v17.s[1]
fmul v11.4s, v11.4s, v17.s[1]
fmul v12.4s, v12.4s, v17.s[2]
fmul v13.4s, v13.4s, v17.s[2]
fmul v14.4s, v14.4s, v17.s[3]
fmul v15.4s, v15.4s, v17.s[3]

// int32_t x = round(x)
Round v0, v1, v2, v3
Round v4, v5, v6, v7
Round v8, v9, v10, v11
Round v12, v13, v14, v15

// y = (int8_t)x
sqxtn v18.4h, v0.4s
sqxtn2 v18.8h, v1.4s
sqxtn v19.4h, v2.4s
sqxtn2 v19.8h, v3.4s
sqxtn v20.4h, v4.4s
sqxtn2 v20.8h, v5.4s
sqxtn v21.4h, v6.4s
sqxtn2 v21.8h, v7.4s

sqxtn v22.4h, v8.4s
sqxtn2 v22.8h, v9.4s
sqxtn v23.4h, v10.4s
sqxtn2 v23.8h, v11.4s
sqxtn v24.4h, v12.4s
sqxtn2 v24.8h, v13.4s
sqxtn v25.4h, v14.4s
sqxtn2 v25.8h, v15.4s

sqxtn v26.8b, v18.8h
sqxtn2 v26.16b, v19.8h
sqxtn v27.8b, v20.8h
sqxtn2 v27.16b, v21.8h
sqxtn v28.8b, v22.8h
sqxtn2 v28.16b, v23.8h
sqxtn v29.8b, v24.8h
sqxtn2 v29.16b, v25.8h

st1 {v26.16b, v27.16b, v28.16b, v29.16b}, [x10], x6

subs x12, x12, #1
bne LoopSz_8

Tile8End:
sub x4, x4, #8    // batch -= 8
add x0, x0, #256  // src += 8 * 8 * sizeof(float32_t)
add x1, x1, #64   // dst += 8 * 8 * sizeof(int8_t)
b TILE_8

TILE_4:
cmp x4, #4
blt TILE_1
sub x8, x7, #64
mov x9, x0   // src
mov x10, x1  // dst
mov x12, x3  // src_depth_quad

ld1 {v16.4s}, [x2], #16

LoopSz_4:
ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x9], #64 // E0, E0, E1, E1
ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x9], x8 // E2, E2, E3, E3

// float32_t x = x * quant_scale
fmul v0.4s, v0.4s, v16.s[0]
fmul v1.4s, v1.4s, v16.s[0]
fmul v2.4s, v2.4s, v16.s[1]
fmul v3.4s, v3.4s, v16.s[1]
fmul v4.4s, v4.4s, v16.s[2]
fmul v5.4s, v5.4s, v16.s[2]
fmul v6.4s, v6.4s, v16.s[3]
fmul v7.4s, v7.4s, v16.s[3]

// int32_t x = round(x)
Round v0, v1, v2, v3
Round v4, v5, v6, v7

// y = (int8_t)x
sqxtn v18.4h, v0.4s
sqxtn2 v18.8h, v1.4s
sqxtn v19.4h, v2.4s
sqxtn2 v19.8h, v3.4s
sqxtn v20.4h, v4.4s
sqxtn2 v20.8h, v5.4s
sqxtn v21.4h, v6.4s
sqxtn2 v21.8h, v7.4s

sqxtn v26.8b, v18.8h
sqxtn2 v26.16b, v19.8h
sqxtn v27.8b, v20.8h
sqxtn2 v27.16b, v21.8h

st1 {v26.16b, v27.16b}, [x10], x6

subs x12, x12, #1
bne LoopSz_4

Tile4End:
sub x4, x4, #4    // batch -= 4
add x0, x0, #128  // src += 4 * 8 * sizeof(float32_t)
add x1, x1, #32   // dst += 4 * 8 * sizeof(int8_t)
b TILE_4

TILE_1:
cmp x4, #1
blt End
mov x9, x0   // src
mov x10, x1  // dst
mov x12, x3  // src_depth_quad

// quant_scale: v8
ld1 {v8.s}[0], [x2], #4
LoopSz_1:
ld1 {v0.4s, v1.4s}, [x9], x7

fmul v0.4s, v0.4s, v8.s[0]
fmul v1.4s, v1.4s, v8.s[0]
// int16_t x = round(x)
fcvtas v0.4s, v0.4s
fcvtas v1.4s, v1.4s

// y = (int8_t)x
sqxtn v7.4h, v0.4s
sqxtn2 v7.8h, v1.4s
sqxtn v7.8b, v7.8h

st1 {v7.8b}, [x10], x6

subs x12, x12, #1
bne LoopSz_1

Tile1End:
subs x4, x4, #1    // batch -= 1
add x0, x0, #32    // src += 1 * 8 * sizeof(float32_t)
add x1, x1, #8    // dst += 1 * 8 * sizeof(int8_t)
bne TILE_1

End:
ldp d8,  d9,  [sp, #(16 * 3)]
ldp d10, d11, [sp, #(16 * 2)]
ldp d12, d13, [sp, #(16 * 1)]
ldp d14, d15, [sp], #(16 * 4)
ret

#endif
