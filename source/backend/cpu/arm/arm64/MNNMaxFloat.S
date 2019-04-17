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

Loop:
    ld1 {v0.4s, v1.4s}, [x0], #32
    fmax v30.4s, v1.4s, v30.4s
    fmax v30.4s, v0.4s, v30.4s
    subs x2, x2, #1
    bne Loop

st1 {v30.4s}, [x1]


End:
ret

#endif
