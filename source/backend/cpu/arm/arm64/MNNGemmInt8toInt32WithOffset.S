//
//  MNNGemmInt8toInt32WithOffset.S
//  MNN
//
//  Created by MNN on 2018/08/24.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNGemmInt8toInt32WithOffset
//void MNNGemmInt8toInt32WithOffset(int32_t* gemmOutputAddr, const uint8_t* colAddr, const int16_t* weightOrigin, const uint8_t* inputOffset,
//const uint8_t* kernelOffset, size_t outputStride, size_t colStride, size_t ocUnit, size_t kernelCountUnit)

//Auto: x0: gemmOutputAddr, x1:colAddr, x2: weightOrigin, x3:inputOffset
//x4: kernelOffset, x5:outputStride, x6:colStride, x7:ocUnit

//Load From Sp
//x8:kernelCountUnit
ldr x8, [sp, #0]

mov x9, #8

ld1 {v23.8b}, [x3]
ld1 {v24.8b}, [x4]
//Now x3,x4 can be reused
uxtl v22.8h, v23.8b
uxtl v23.8h, v24.8b

LoopDx:
    mov x11, x7
    mov x3, x2
    LoopDz:
        mov x4, x1
        mov x12, x8
        movi v1.4s, #0
        movi v2.4s, #0
        LoopSz:
            ld1 {v0.8b}, [x1], #8
            uxtl v0.8h, v0.8b
            ld1 {v16.8h, v17.8h, v18.8h, v19.8h}, [x2], #64
            sqsub v0.8h, v0.8h, v22.8h
            smlal v1.4s, v16.4h, v0.h[0]
            smlal2 v2.4s, v16.8h, v0.h[0]
            smlal v1.4s, v17.4h, v0.h[1]
            smlal2 v2.4s, v17.8h, v0.h[1]
            smlal v1.4s, v18.4h, v0.h[2]
            smlal2 v2.4s, v18.8h, v0.h[2]
            smlal v1.4s, v19.4h, v0.h[3]
            smlal2 v2.4s, v19.8h, v0.h[3]
            ld1 {v16.8h, v17.8h, v18.8h, v19.8h}, [x2], #64
            smlal v1.4s, v16.4h, v0.h[4]
            smlal2 v2.4s, v16.8h, v0.h[4]
            smlal v1.4s, v17.4h, v0.h[5]
            smlal2 v2.4s, v17.8h, v0.h[5]
            smlal v1.4s, v18.4h, v0.h[6]
            smlal2 v2.4s, v18.8h, v0.h[6]
            smlal v1.4s, v19.4h, v0.h[7]
            smlal2 v2.4s, v19.8h, v0.h[7]

            subs x12, x12, #1
            bne LoopSz

        st1 {v1.4s, v2.4s}, [x0], #32
        
        subs x11, x11, #1
        mov x1, x4
        bne LoopDz

    subs x9, x9, #1
    mov x2, x3
    add x1, x1, x6
    bne LoopDx

ret

#endif
