//
//  MNNQuantSumFP16.S
//  MNN
//
//  Created by MNN on 2023/11/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"
.text
.align 5

//void MNNQuantSumFP16(float* sum, const float* dequant_scale, size_t thread, size_t batch)
asm_function MNNQuantSumFP16

// x0: sum, x1:dequant_scale, x2:thread, x3:batch
stp d14, d15, [sp, #(-16 * 4)]!
stp d12, d13, [sp, #(16 * 1)]
stp d10, d11, [sp, #(16 * 2)]
stp d8,  d9,  [sp, #(16 * 3)]

Start:
lsl x9, x3, #2 // src_step = batch * sizeof(int32_t)
mov x10, #0
b TILE_8
TILE_12:
cmp x3, #12
blt TILE_8
add x6, x0, x10  // sum_ptr
mov x7, x2  // thread

// sum: v0
ld1 {v0.4s, v1.4s, v2.4s}, [x6], x9
subs x7, x7, #1
beq Tile12End

LoopSz_12:
ld1 {v3.4s, v4.4s, v5.4s}, [x6], x9

// sum += sum[i]
add v0.4s, v0.4s, v3.4s
add v1.4s, v1.4s, v4.4s
add v2.4s, v2.4s, v5.4s

subs x7, x7, #1
bne LoopSz_12

Tile12End:
sub x3, x3, #12
// load dequant_scale
ld1 {v4.8h}, [x1], #16
ld1 {v10.4h}, [x1], #8
fcvtl v5.4s, v4.4h
fcvtl2 v6.4s, v4.8h
fcvtl v10.4s, v10.4h
// sum_half = (half)((float)sum_int * dequant_scale)
scvtf v0.4s, v0.4s
scvtf v1.4s, v1.4s
scvtf v2.4s, v2.4s
fmul v7.4s, v0.4s, v5.4s
fmul v8.4s, v1.4s, v6.4s
fmul v10.4s, v2.4s, v10.4s
fcvtn v9.4h, v7.4s
fcvtn2 v9.8h, v8.4s
fcvtn v10.4h, v10.4s
st1 {v9.8h}, [x0], #16
st1 {v10.4h}, [x0], #8
add x10, x10, #24
b TILE_12

TILE_8:
cmp x3, #8
blt TILE_4
add x6, x0, x10  // sum_ptr
mov x7, x2  // thread

// sum: v0
ld1 {v0.4s, v1.4s}, [x6], x9
subs x7, x7, #1
beq Tile8End

LoopSz_8:
ld1 {v2.4s, v3.4s}, [x6], x9

// sum += sum[i]
add v0.4s, v0.4s, v2.4s
add v1.4s, v1.4s, v3.4s

subs x7, x7, #1
bne LoopSz_8

Tile8End:
sub x3, x3, #8
// load dequant_scale
ld1 {v4.8h}, [x1], #16
fcvtl v5.4s, v4.4h
fcvtl2 v6.4s, v4.8h
// sum_half = (half)((float)sum_int * dequant_scale)
scvtf v0.4s, v0.4s
scvtf v1.4s, v1.4s
fmul v7.4s, v0.4s, v5.4s
fmul v8.4s, v1.4s, v6.4s
fcvtn v9.4h, v7.4s
fcvtn2 v9.8h, v8.4s
st1 {v9.8h}, [x0], #16
add x10, x10, #16
b TILE_8

TILE_4:
cmp x3, #4
blt TILE_1
add x6, x0, x10  // sum_ptr
mov x7, x2  // thread

// sum: v0
ld1 {v0.4s}, [x6], x9
subs x7, x7, #1
beq Tile4End

LoopSz_4:
ld1 {v1.4s}, [x6], x9

// sum += sum[i]
add v0.4s, v0.4s, v1.4s

subs x7, x7, #1
bne LoopSz_4

Tile4End:
sub x3, x3, #4
// load dequant_scale
ld1 {v1.4h}, [x1], #8
fcvtl v2.4s, v1.4h
// sum_half = (half)((float)sum_int * dequant_scale)
scvtf v3.4s, v0.4s
fmul v4.4s, v3.4s, v2.4s
fcvtn v5.4h, v4.4s
st1 {v5.d}[0], [x0], #8
add x10, x10, #8
b TILE_4

// x0: sum, x1:dequant_scale, x2:thread, x3:batch
TILE_1:
cmp x3, #1
blt End
add x6, x0, x10  // sum_ptr
mov x7, x2  // thread

// sum: v0
ld1 {v0.s}[0], [x6], x9
subs x7, x7, #1
beq Tile1End

LoopSz_1:
ld1 {v1.s}[0], [x6], x9

// sum += sum[i]
// add s0, s0, s1
add v0.4s, v0.4s, v1.4s

subs x7, x7, #1
bne LoopSz_1

Tile1End:
sub x3, x3, #1
// load dequant_scale
ld1 {v1.h}[0], [x1], #2
fcvtl v2.4s, v1.4h
// sum_half = (half)((float)sum_int * dequant_scale)
scvtf s3, s0
fmul s4, s3, s2
fcvtn v5.4h, v4.4s
st1 {v5.h}[0], [x0], #2
add x10, x10, #2
b TILE_1


End:
ldp d8,  d9,  [sp, #(16 * 3)]
ldp d10, d11, [sp, #(16 * 2)]
ldp d12, d13, [sp, #(16 * 1)]
ldp d14, d15, [sp], #(16 * 4)
ret

#endif

