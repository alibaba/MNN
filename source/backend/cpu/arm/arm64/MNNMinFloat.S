//
//  MNNMinFloat.S
//  MNN
//
//  Created by MNN on 2018/09/18.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5
asm_function MNNMinFloat
//void MNNMinFloat(float* input, float * minBuffer, int32_t inputCountUnit);

//Auto: X0: input, X1:minBuffer, X2:inputCountUnit

cmp x2, #0
beq End


# minBuffer : v31
ld1 {v31.4s}, [x1]

Loop:
    ld1 {v0.4s, v1.4s}, [x0], #32
    fmin v31.4s, v0.4s, v31.4s
    fmin v31.4s, v1.4s, v31.4s
    subs x2, x2, #1
    bne Loop

st1 {v31.4s}, [x1]

End:
ret

#endif
