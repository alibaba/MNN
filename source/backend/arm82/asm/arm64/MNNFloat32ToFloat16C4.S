//
//  MNNFloat32ToFloat16C4.S
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5
asm_function MNNFloat32ToFloat16C4
//    void MNNFloat32ToFloat16C4(int16_t* dst, const float* src, size_t sizeQuad);
//Auto: x0: dst, x1:src, x2:sizeQuad
L4:
cmp x2, #4
blt L1
Loop4:

ld1 {v0.4s, v1.4s}, [x1], #32
fcvtn v0.4h, v0.4s
ld1 {v2.4s, v3.4s}, [x1], #32
fcvtn v1.4h, v1.4s
fcvtn v2.4h, v2.4s
st1 {v0.4h, v1.4h}, [x0], #16
sub x2, x2, #4
fcvtn v3.4h, v3.4s
cmp x2, #4
st1 {v2.4h, v3.4h}, [x0], #16
bge Loop4

L1:
cmp x2, #0
beq End
Loop1:

ld1 {v0.4s}, [x1], #16
fcvtn v0.4h, v0.4s
st1 {v0.4h}, [x0], #8

subs x2, x2, #1
bne Loop1


End:

ret
#endif
