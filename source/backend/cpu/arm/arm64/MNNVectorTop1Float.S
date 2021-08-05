//
//  MNNVectorTop1Float.S
//  MNN
//
//  Created by MNN on 2020/12/09.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNVectorTop1Float
// void MNNVectorTop1Float(float* input, float* maxValue, int32_t* maxIndex, size_t inputCountUnit);

// Auto: x0: input, x1: maxValue, x2: maxIndex, x3: inputCountUnit

// v30 maxValue
ld1 {v30.4s}, [x0]

// v29 maxIndex
mov w11, #0
mov v29.s[0], w11
mov w11, #1
mov v29.s[1], w11
mov w11, #2
mov v29.s[2], w11
mov w11, #3
mov v29.s[3], w11

// v28 current index
mov v28.16b, v29.16b

// v27, all 4, increment
mov w11, #4
mov v27.s[0], w11
mov v27.s[1], w11
mov v27.s[2], w11
mov v27.s[3], w11


cmp x3, #0
beq End

Loop:
    ld1 {v26.4s}, [x0], #16

    fcmgt v25.4s, v26.4s, v30.4s
    bit v30.16b, v26.16b, v25.16b
    bit v29.16b, v28.16b, v25.16b

    add v28.4s, v28.4s, v27.4s
    subs x3, x3, #1

    bne Loop

// reduce result to single value and index
mov v20.d[0], v30.d[1]
mov v21.d[0], v29.d[1]

fcmgt v25.2s, v20.2s, v30.2s
bit v30.8b, v20.8b, v25.8b
bit v29.8b, v21.8b, v25.8b

mov v20.s[0], v30.s[1]
mov v21.s[0], v29.s[1]

fcmgt v25.2s, v20.2s, v30.2s
bit v30.8b, v20.8b, v25.8b
bit v29.8b, v21.8b, v25.8b

End:
    st1 {v30.s}[0], [x1]
    st1 {v29.s}[0], [x2]
    ret


#endif
