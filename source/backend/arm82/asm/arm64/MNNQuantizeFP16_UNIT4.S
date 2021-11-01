//
//  MNNQuantizeFP16_UNIT4.S
//  MNN
//
//  Created by MNN on 2020/02/13.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__
#include "MNNAsmGlobal.h"

.text
.align 5
asm_function MNNQuantizeFP16_UNIT4
// void MNNQuantizeFP16_UNIT4(FLOAT16* dst, const float* src, int size);

// Auto:
//      x0:dst, x1:src, x2:size, x3:min_max

// Init min / max to v30, v31
ldr w4, [x3]
ldr w5, [x3, #4]
dup v30.4s, w4
dup v31.4s, w5
L4:
cmp x2, #4
blt L1
Loop4:

ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x1], #64
fmax v0.4s, v0.4s, v30.4s
fmax v1.4s, v1.4s, v30.4s
fmax v2.4s, v2.4s, v30.4s
fmax v3.4s, v3.4s, v30.4s
fmin v0.4s, v0.4s, v31.4s
fmin v1.4s, v1.4s, v31.4s
fmin v2.4s, v2.4s, v31.4s
fmin v3.4s, v3.4s, v31.4s
fcvtn v0.4h, v0.4s
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
fmax v0.4s, v0.4s, v30.4s
fmin v0.4s, v0.4s, v31.4s
fcvtn v0.4h, v0.4s
st1 {v0.4h}, [x0], #8

subs x2, x2, #1
bne Loop1


End:

ret
#endif
