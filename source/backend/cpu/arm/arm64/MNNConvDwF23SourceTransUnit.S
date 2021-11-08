//
//  MNNConvDwF23SourceTransUnit.S
//  MNN
//
//  Created by MNN on 2019/4/4.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNConvDwF23SourceTransUnit
//    void MNNConvDwF23SourceTransUnit(const float *source, float *dest, size_t unit);

//Auto:
//x0: source, x1:dest, x2:unit

L1:
cmp x2, #0
beq End

ld1 {v16.4s, v17.4s}, [x0], #32
ld1 {v18.4s, v19.4s}, [x0], #32
subs x2, x2, #1
fsub v0.4s, v16.4s, v18.4s
fadd v1.4s, v17.4s, v18.4s
beq L1LoopEnd

L1Loop:
    fsub v2.4s, v18.4s, v17.4s
    // st1 {v0.4s, v1.4s}, [x1], #32
    stp q0, q1, [x1], #32
    fsub v3.4s, v19.4s, v17.4s
    mov v16.16b, v18.16b
    // st1 {v2.4s, v3.4s}, [x1], #32
    stp q2, q3, [x1], #32
    mov v17.16b, v19.16b
    ld1 {v18.4s, v19.4s}, [x0], #32
    fsub v0.4s, v16.4s, v18.4s
    fadd v1.4s, v17.4s, v18.4s

    subs x2, x2, #1
    bne L1Loop
L1LoopEnd:
fsub v2.4s, v18.4s, v17.4s
fsub v3.4s, v19.4s, v17.4s

// st1 {v0.4s, v1.4s}, [x1], #32
// st1 {v2.4s, v3.4s}, [x1], #32
stp q0, q1, [x1], #32
stp q2, q3, [x1], #32


End:
ret

#endif
