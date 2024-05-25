//
//  MNNQuantSumFP32.S
//  MNN
//
//  Created by MNN on 2023/11/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"
.text
.align 5

//void MNNQuantSumFP32(float* sum, const float* dequant_scale, size_t thread, size_t batch)
asm_function MNNQuantSumFP32

// x0: sum, x1:dequant_scale, x2:thread, x3:batch
stp d14, d15, [sp, #(-16 * 4)]!
stp d12, d13, [sp, #(16 * 1)]
stp d10, d11, [sp, #(16 * 2)]
stp d8,  d9,  [sp, #(16 * 3)]

Start:
lsl x9, x3, #2 // src_step = batch * sizeof(int32_t)

TILE_8:
cmp x3, #8
blt TILE_4
// add x6, x0, x10  // sum_ptr
mov x6, x0
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
ld1 {v4.4s, v5.4s}, [x1], #32
// sum_half = (float)sum_int * dequant_scale
scvtf v0.4s, v0.4s
scvtf v1.4s, v1.4s
fmul v0.4s, v0.4s, v4.4s
fmul v1.4s, v1.4s, v5.4s
st1 {v0.4s, v1.4s}, [x0], #32
b TILE_8

TILE_4:
cmp x3, #4
blt TILE_1
// add x6, x0, x10  // sum_ptr
mov x6, x0
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
ld1 {v2.4s}, [x1], #16
// sum_half = (float)sum_int * dequant_scale
scvtf v3.4s, v0.4s
fmul v4.4s, v3.4s, v2.4s
st1 {v4.4s}, [x0], #16
b TILE_4

// x0: sum, x1:dequant_scale, x2:thread, x3:batch
TILE_1:
cmp x3, #1
blt End
mov x6, x0
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
ld1 {v2.s}[0], [x1], #4

// sum_half = (float)sum_int * dequant_scale
scvtf v3.4s, v0.4s
fmul s4, s3, s2
st1 {v4.s}[0], [x0], #4 
b TILE_1


End:
ldp d8,  d9,  [sp, #(16 * 3)]
ldp d10, d11, [sp, #(16 * 2)]
ldp d12, d13, [sp, #(16 * 1)]
ldp d14, d15, [sp], #(16 * 4)
ret

#endif

