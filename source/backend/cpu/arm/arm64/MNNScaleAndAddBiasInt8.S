//
//  MNNScaleAndAddBiasInt8.S
//  MNN
//
//  Created by MNN on 2019/02/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__
#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNScaleAndAddBiasInt8
// MNNScaleAndAddBiasInt8(int8_t* dst, const int8_t* src, const int32_t* bias, const int32_t* alpha, int32_t mShiftBits,
//                        ssize_t minValue, ssize_t maxValue, int8_t* inputZeroPoint, int8_t* outputZeroPoint, ssize_t planeNumber, ssize_t biasNumber, ssize_t pack)

//Auto: x0:dst, x1:src, x2:bias, x3:alpha, x4:mShiftBits, x5:minValue, x6:maxValue, x7:inputZeroPoint
//Load from sp: x11:outputZeroPoint, x8:planeNumber, x9:biasNumber
//avoid to touch platform-register x-18

ldr x11, [sp, #0]
ldr x8, [sp, #8]
ldr x9, [sp, #16]

stp d14, d15, [sp, #-64]!
stp d12, d13, [sp, #16]
stp d10, d11, [sp, #32]
stp d8,  d9,  [sp, #48]

cmp x8, #0
beq BSEnd

cmp x9, #0
beq BSEnd

dup v27.16b, w5                // min
dup v28.16b, w6                // max

ld1r {v29.8b}, [x7]            // inputZeroPoint

BSLoopZ:
    mov x10, x8
    ld1 {v31.4s}, [x2], #16    // bias
    ld1 {v30.4s}, [x3], #16    // scale

    cmp x10, #4
    blt BSLoopP1
    cmp x10, #8
    blt BSLoopP4
    cmp x10, #16
    blt BSLoopP8

BSLoopP16:
        ld1 {v0.16b, v1.16b, v2.16b, v3.16b}, [x1], #64

        sxtl  v4.8h,  v0.8b
        sxtl2 v5.8h,  v0.16b
        sxtl  v6.8h,  v1.8b
        sxtl2 v7.8h,  v1.16b
        sxtl  v8.8h,  v2.8b
        sxtl2 v9.8h,  v2.16b
        sxtl  v10.8h, v3.8b
        sxtl2 v11.8h, v3.16b

        ssubw v4.8h, v4.8h, v29.8b
        ssubw v5.8h, v5.8h, v29.8b
        ssubw v6.8h, v6.8h, v29.8b
        ssubw v7.8h, v7.8h, v29.8b
        ssubw v8.8h, v8.8h, v29.8b
        ssubw v9.8h, v9.8h, v29.8b
        ssubw v10.8h, v10.8h, v29.8b
        ssubw v11.8h, v11.8h, v29.8b

        sxtl  v12.4s, v4.4h
        sxtl2 v13.4s, v4.8h
        sxtl  v14.4s, v5.4h
        sxtl2 v15.4s, v5.8h
        sxtl  v16.4s, v6.4h
        sxtl2 v17.4s, v6.8h
        sxtl  v18.4s, v7.4h
        sxtl2 v19.4s, v7.8h
        sxtl  v20.4s, v8.4h
        sxtl2 v21.4s, v8.8h
        sxtl  v22.4s, v9.4h
        sxtl2 v23.4s, v9.8h
        sxtl  v24.4s, v10.4h
        sxtl2 v25.4s, v10.8h
        sxtl  v26.4s, v11.4h
        sxtl2 v11.4s, v11.8h

        ld1r {v0.8b}, [x11]

        mul v12.4s, v12.4s, v30.4s
        mul v13.4s, v13.4s, v30.4s
        mul v14.4s, v14.4s, v30.4s
        mul v15.4s, v15.4s, v30.4s
        mul v16.4s, v16.4s, v30.4s
        mul v17.4s, v17.4s, v30.4s
        mul v18.4s, v18.4s, v30.4s
        mul v19.4s, v19.4s, v30.4s
        mul v20.4s, v20.4s, v30.4s
        mul v21.4s, v21.4s, v30.4s
        mul v22.4s, v22.4s, v30.4s
        mul v23.4s, v23.4s, v30.4s
        mul v24.4s, v24.4s, v30.4s
        mul v25.4s, v25.4s, v30.4s
        mul v26.4s, v26.4s, v30.4s
        mul v11.4s, v11.4s, v30.4s

        add v12.4s, v12.4s, v31.4s
        add v13.4s, v13.4s, v31.4s
        add v14.4s, v14.4s, v31.4s
        add v15.4s, v15.4s, v31.4s
        add v16.4s, v16.4s, v31.4s
        add v17.4s, v17.4s, v31.4s
        add v18.4s, v18.4s, v31.4s
        add v19.4s, v19.4s, v31.4s
        add v20.4s, v20.4s, v31.4s
        add v21.4s, v21.4s, v31.4s
        add v22.4s, v22.4s, v31.4s
        add v23.4s, v23.4s, v31.4s
        add v24.4s, v24.4s, v31.4s
        add v25.4s, v25.4s, v31.4s
        add v26.4s, v26.4s, v31.4s
        add v11.4s, v11.4s, v31.4s

        sqrshrn  v12.4h, v12.4s, #15
        sqrshrn2 v12.8h, v13.4s, #15
        sqrshrn  v14.4h, v14.4s, #15
        sqrshrn2 v14.8h, v15.4s, #15
        sqrshrn  v16.4h, v16.4s, #15
        sqrshrn2 v16.8h, v17.4s, #15
        sqrshrn  v18.4h, v18.4s, #15
        sqrshrn2 v18.8h, v19.4s, #15
        sqrshrn  v20.4h, v20.4s, #15
        sqrshrn2 v20.8h, v21.4s, #15
        sqrshrn  v22.4h, v22.4s, #15
        sqrshrn2 v22.8h, v23.4s, #15
        sqrshrn  v24.4h, v24.4s, #15
        sqrshrn2 v24.8h, v25.4s, #15
        sqrshrn  v26.4h, v26.4s, #15
        sqrshrn2 v26.8h, v11.4s, #15

        saddw  v12.8h, v12.8h, v0.8b
        saddw  v14.8h, v14.8h, v0.8b
        saddw  v16.8h, v16.8h, v0.8b
        saddw  v18.8h, v18.8h, v0.8b
        saddw  v20.8h, v20.8h, v0.8b
        saddw  v22.8h, v22.8h, v0.8b
        saddw  v24.8h, v24.8h, v0.8b
        saddw  v26.8h, v26.8h, v0.8b

        sqxtn  v12.8b,  v12.8h
        sqxtn2 v12.16b, v14.8h
        sqxtn  v13.8b,  v16.8h
        sqxtn2 v13.16b, v18.8h
        sqxtn  v14.8b,  v20.8h
        sqxtn2 v14.16b, v22.8h
        sqxtn  v15.8b,  v24.8h
        sqxtn2 v15.16b, v26.8h

        smax v12.16b, v12.16b, v27.16b
        smin v12.16b, v12.16b, v28.16b
        smax v13.16b, v13.16b, v27.16b
        smin v13.16b, v13.16b, v28.16b
        smax v14.16b, v14.16b, v27.16b
        smin v14.16b, v14.16b, v28.16b
        smax v15.16b, v15.16b, v27.16b
        smin v15.16b, v15.16b, v28.16b

        st1 {v12.16b, v13.16b, v14.16b, v15.16b}, [x0], #64
        sub x10, x10, #16

        cmp x10, #16
        bge BSLoopP16
        cmp x10, #0
        beq BSLoopPEnd
        cmp x10, #4
        blt BSLoopP1
        cmp x10, #8
        blt BSLoopP4

    BSLoopP8:
        ld1 {v0.16b, v1.16b}, [x1], #32

        sxtl  v2.8h, v0.8b
        sxtl2 v3.8h, v0.16b
        sxtl  v4.8h, v1.8b
        sxtl2 v5.8h, v1.16b

        ssubw v2.8h, v2.8h, v29.8b
        ssubw v3.8h, v3.8h, v29.8b
        ssubw v4.8h, v4.8h, v29.8b
        ssubw v5.8h, v5.8h, v29.8b

        sxtl  v16.4s, v2.4h
        sxtl2 v17.4s, v2.8h
        sxtl  v18.4s, v3.4h
        sxtl2 v19.4s, v3.8h
        sxtl  v20.4s, v4.4h
        sxtl2 v21.4s, v4.8h
        sxtl  v22.4s, v5.4h
        sxtl2 v23.4s, v5.8h
        ld1r {v24.8b}, [x11]

        mul v16.4s, v16.4s, v30.4s
        mul v17.4s, v17.4s, v30.4s
        mul v18.4s, v18.4s, v30.4s
        mul v19.4s, v19.4s, v30.4s
        mul v20.4s, v20.4s, v30.4s
        mul v21.4s, v21.4s, v30.4s
        mul v22.4s, v22.4s, v30.4s
        mul v23.4s, v23.4s, v30.4s

        add v16.4s, v16.4s, v31.4s
        add v17.4s, v17.4s, v31.4s
        add v18.4s, v18.4s, v31.4s
        add v19.4s, v19.4s, v31.4s
        add v20.4s, v20.4s, v31.4s
        add v21.4s, v21.4s, v31.4s
        add v22.4s, v22.4s, v31.4s
        add v23.4s, v23.4s, v31.4s

        sqrshrn  v16.4h, v16.4s, #15
        sqrshrn2 v16.8h, v17.4s, #15
        sqrshrn  v18.4h, v18.4s, #15
        sqrshrn2 v18.8h, v19.4s, #15
        sqrshrn  v20.4h, v20.4s, #15
        sqrshrn2 v20.8h, v21.4s, #15
        sqrshrn  v22.4h, v22.4s, #15
        sqrshrn2 v22.8h, v23.4s, #15

        saddw v16.8h, v16.8h, v24.8b
        saddw v18.8h, v18.8h, v24.8b
        saddw v20.8h, v20.8h, v24.8b
        saddw v22.8h, v22.8h, v24.8b

        sqxtn  v0.8b,  v16.8h
        sqxtn2 v0.16b, v18.8h
        sqxtn  v1.8b,  v20.8h
        sqxtn2 v1.16b, v22.8h

        smax v0.16b, v0.16b, v27.16b
        smin v0.16b, v0.16b, v28.16b
        smax v1.16b, v1.16b, v27.16b
        smin v1.16b, v1.16b, v28.16b

        st1 {v0.16b, v1.16b}, [x0], #32
        sub x10, x10, #8

        cmp x10, #8
        bge BSLoopP8
        cmp x10, #0
        beq BSLoopPEnd
        cmp x10, #4
        blt BSLoopP1

    BSLoopP4:
        ld1 {v0.16b}, [x1], #16

        sxtl  v2.8h, v0.8b
        sxtl2 v3.8h, v0.16b

        ssubw v2.8h, v2.8h, v29.8b
        ssubw v3.8h, v2.8h, v29.8b
        sxtl  v16.4s, v2.4h
        sxtl2 v17.4s, v2.8h
        sxtl  v18.4s, v3.4h
        sxtl2 v19.4s, v3.8h

        mul v16.4s, v16.4s, v30.4s
        mul v17.4s, v17.4s, v30.4s
        mul v18.4s, v18.4s, v30.4s
        mul v19.4s, v19.4s, v30.4s
        ld1r {v20.8b}, [x11]

        add v16.4s, v16.4s, v31.4s
        add v17.4s, v17.4s, v31.4s
        add v18.4s, v18.4s, v31.4s
        add v19.4s, v19.4s, v31.4s

        sqrshrn  v16.4h, v16.4s, #15
        sqrshrn2 v16.8h, v17.4s, #15
        sqrshrn  v18.4h, v18.4s, #15
        sqrshrn2 v18.8h, v19.4s, #15

        saddw v16.8h, v16.8h, v20.8b
        saddw v18.8h, v18.8h, v20.8b
        sqxtn  v0.8b,  v16.8h
        sqxtn2 v0.16b, v18.8h

        smax v0.16b, v0.16b, v27.16b
        smin v0.16b, v0.16b, v28.16b

        st1 {v0.16b}, [x0], #16
        sub x10, x10, #4

        cmp x10, #4
        bge BSLoopP4

    cmp x10, #0
    beq BSLoopPEnd

    BSLoopP1:
        ld1 {v0.s}[0], [x1], #4
        dup v0.4s, v0.s[0]
        ld1r {v20.8b}, [x11]

        sxtl  v2.8h, v0.8b
        ssubw v2.8h, v2.8h, v29.8b
        sxtl  v1.4s, v2.4h

        mul v1.4s, v1.4s, v30.4s
        add v1.4s, v1.4s, v31.4s

        sqrshrn v1.4h, v1.4s, #15
        dup    v1.2d, v1.d[0]
        saddw v1.8h, v1.8h, v20.8b
        sqxtn v1.8b, v1.8h

        smax v1.8b, v1.8b, v27.8b
        smin v1.8b, v1.8b, v28.8b
    
        st1 {v1.s}[0], [x0], #4
        subs x10, x10, #1
        bne BSLoopP1
    BSLoopPEnd:
    subs x9, x9, #1
    bne BSLoopZ


BSEnd:
ldp d8,  d9,  [sp, #48]
ldp d10, d11, [sp, #32]
ldp d12, d13, [sp, #16]
ldp d14, d15, [sp], #64
ret


#endif
