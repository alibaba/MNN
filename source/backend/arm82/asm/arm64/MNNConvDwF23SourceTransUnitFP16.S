//
//  MNNConvDwF23SourceTransUnitFP16.S
//  MNN
//
//  Created by MNN on 2019/4/4.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNConvDwF23SourceTransUnitFP16
//    void MNNConvDwF23SourceTransUnitFP16(const FLOAT16 *source, FLOAT16 *dest, size_t unit);

//Auto:
//x0: source, x1:dest, x2:unit

L1:
cmp x2, #0
beq End

ld1 {v16.8h, v17.8h}, [x0], #32
ld1 {v18.8h, v19.8h}, [x0], #32
subs x2, x2, #1
fsub v0.8h, v16.8h, v18.8h
fadd v1.8h, v17.8h, v18.8h
beq L1LoopEnd

L1Loop:
    fsub v2.8h, v18.8h, v17.8h
    st1 {v0.8h, v1.8h}, [x1], #32
    fsub v3.8h, v19.8h, v17.8h
    mov v16.16b, v18.16b
    st1 {v2.8h, v3.8h}, [x1], #32
    mov v17.16b, v19.16b
    ld1 {v18.8h, v19.8h}, [x0], #32
    fsub v0.8h, v16.8h, v18.8h
    fadd v1.8h, v17.8h, v18.8h

    subs x2, x2, #1
    bne L1Loop
L1LoopEnd:
fsub v2.8h, v18.8h, v17.8h
fsub v3.8h, v19.8h, v17.8h

st1 {v0.8h, v1.8h}, [x1], #32
st1 {v2.8h, v3.8h}, [x1], #32


End:
ret

#endif
