//
//  MNNUInt8ToInt16WithOffsetC4Fast.S
//  MNN
//
//  Created by MNN on 2018/10/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNUInt8ToInt16WithOffsetC4Fast

//void MNNUInt8ToInt16WithOffsetC4Fast(int16_t* dst, const uint8_t* src, size_t zeroPoint, size_t sizeQuad, size_t depthQuad, size_t dstZStep, size_t srcZStep);

//Auto: x0:dst, x1:src, x2: zeroPoint, x3: sizeQuad
//x4: depthQuad, x5: dstStride, x6:srcStride
dup v23.16b, w2

LoopZ:
    mov x7, x0
    mov x8, x1
    mov x9, x3
    
    L8:
    cmp x3, #8
    blt L1
    
    LoopL8:
    ld1 {v0.8h}, [x1], #16

    usubl v16.8h, v0.8b, v23.8b

    ld1 {v1.8h}, [x1], #16
    usubl2 v17.8h, v0.16b, v23.16b
    st1 {v16.8h}, [x0], #16
    usubl v18.8h, v1.8b, v23.8b
    st1 {v17.8h}, [x0], #16
    usubl2 v19.8h, v1.16b, v23.16b
    st1 {v18.8h}, [x0], #16
    st1 {v19.8h}, [x0], #16
    
    sub x3, x3, #8
    cmp x3, #8
    bge LoopL8
    
    
    L1:
    cmp x3, #0
    beq End
    
    LoopL1:
    ld1 {v0.s}[0], [x1], #4
    usubl v0.8h, v0.8b, v23.8b
    st1 {v0.4h}, [x0], #8
    
    subs x3, x3, #1
    bne LoopL1
    
    
    
    End:
    
    subs x4, x4, #1
    add x0, x7, x5
    add x1, x8, x6
    mov x3, x9
    bne LoopZ

ret

#endif
