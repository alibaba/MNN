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

mov x11, #32//SRC_UNIT*DST_XUNIT
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
        uxtl v1.8h, v0.8b
        st1 {v0.8b}, [x8], x11
        mov v2.d[0], v1.d[1]
        uadalp v23.4s, v1.8h
        umlal v21.4s, v2.4h, v1.4h
        bne LoopSz
    addv s23, v23.4s
    addv s21, v21.4s
    mla v21.2s, v23.2s, v22.2s

    subs x5, x5, #1
    st1 {v21.s}[0], [x0], #4
    add x2, x2, #4 //UNIT 
    add x1, x1, #8 //SRC_UNIT
    bne LoopCount

ret

#endif
