//
//  MNNFloat16ToFloat32C4.S
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5
asm_function MNNFloat16ToFloat32C4
//void MNNFloat16ToFloat32C4(float* dst, const int16_t* src, size_t sizeQuad, size_t depth, size_t dstStride);
//Auto: x0: dst, x1:src, x2:sizeQuad, x3: depth
//Auto: x4:dstStride

LoopZ:
mov x8, x2
mov x9, x0


L4:
cmp x2, #4
blt L1
Loop4:

ld1 {v0.4h, v1.4h}, [x1], #16
fcvtl v0.4s, v0.4h
ld1 {v2.4h, v3.4h}, [x1], #16
fcvtl v1.4s, v1.4h
fcvtl v2.4s, v2.4h
st1 {v0.4s, v1.4s}, [x0], #32
sub x2, x2, #4
fcvtl v3.4s, v3.4h
cmp x2, #4
st1 {v2.4s, v3.4s}, [x0], #32
bge Loop4

L1:
cmp x2, #0
beq End
Loop1:

ld1 {v0.4h}, [x1], #8
fcvtl v0.4s, v0.4h
st1 {v0.4s}, [x0], #16

subs x2, x2, #1
bne Loop1


End:

subs x3, x3, #1
add x0, x9, x4
mov x2, x8

bne LoopZ

ret
#endif
