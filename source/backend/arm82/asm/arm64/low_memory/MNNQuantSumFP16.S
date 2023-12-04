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

