//
//  MNNMaxFloat.S
//  MNN
//
//  Created by MNN on 2018/09/18.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5
asm_function MNNMaxFloat
//void MNNMaxFloat(float* input, float * maxBuffer, int32_t inputCountUnit);

//Auto: X0: input, X1:maxBuffer, X2:inputCountUnit

cmp x2, #0
beq End


# maxBuffer : v30
ld1 {v30.4s}, [x1]
mov v31.16b, v30.16b

cmp x2, #2
blt Loop
mov v29.16b, v30.16b
mov v28.16b, v30.16b
LoopC4:
    ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x0], #64
    fmax v30.4s, v1.4s, v30.4s
    fmax v31.4s, v0.4s, v31.4s
    fmax v29.4s, v2.4s, v29.4s
    fmax v28.4s, v3.4s, v28.4s
    sub x2, x2, #2
    cmp x2, #2
    bge LoopC4

fmax v30.4s, v30.4s, v29.4s
fmax v31.4s, v31.4s, v28.4s

cmp x2, #0
beq ComputeEnd


Loop:
    ld1 {v0.4s, v1.4s}, [x0], #32
    fmax v30.4s, v1.4s, v30.4s
    fmax v31.4s, v0.4s, v31.4s
    subs x2, x2, #1
    bne Loop

ComputeEnd:
fmax v30.4s, v30.4s, v31.4s

st1 {v30.4s}, [x1]

End:

ret

#endif
