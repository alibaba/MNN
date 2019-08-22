//
//  MNNLoadU8AndSum.S
//  MNN
//
//  Created by MNN on 2018/11/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNLoadU8AndSum
//void MNNLoadU8AndSum(int32_t* inputSum, uint8_t* colAddr, const uint8_t* inputOrigin, size_t srcZStep, size_t icDiv8, size_t realDstCount, size_t filter_offset)
//Auto: x0:inputSum, x1:colAddr, x2:inputOrigin, x3:srcZStep
//x4: icDiv8, x5: realDstCount, x6:filter_offset
mov v22.s[0], w6

mov x11, #64//SRC_UNIT*DST_XUNIT
movi v31.8b, #128
LoopCount:
    mov x12, x4
    mov x7, x2
    mov x8, x1
    movi v23.4s, #0
    movi v21.4s, #0
    LoopSz:
        subs x12, x12, #1
        ld1 {v0.s}[0], [x7], x3
        ld1 {v0.s}[1], [x7], x3
        ld1 {v1.s}[0], [x7], x3
        ld1 {v1.s}[1], [x7], x3

        usubl v2.8h, v0.8b, v31.8b
        usubl v3.8h, v1.8b, v31.8b

        sqxtn v4.8b, v2.8h
        sadalp v23.4s, v2.8h
        sqxtn2 v4.16b, v3.8h
        sadalp v21.4s, v3.8h
        st1 {v4.16b}, [x8], x11
        bne LoopSz
    add v21.4s, v21.4s, v23.4s
    addv s21, v21.4s
    mul v21.2s, v21.2s, v22.2s

    subs x5, x5, #1
    st1 {v21.s}[0], [x0], #4
    add x2, x2, #4 //UNIT 
    add x1, x1, #16 //SRC_UNIT
    bne LoopCount

ret

#endif
