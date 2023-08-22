//
//  MNNScaleAndAddBias.S
//  MNN
//
//  Created by MNN on 2019/02/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__
#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNScaleAndAddBias
//void MNNScaleAndAddBias(float* dst, const float* src, const float* bias, const float* alpha, size_t planeNumber, size_t biasNumber)

//Auto: x0:dst, x1:src, x2:bias, x3:alpha, x4:planeNumber, x5:biasNumber
stp d14, d15, [sp, #-64]!
stp d12, d13, [sp, #16]
stp d10, d11, [sp, #32]
stp d8,  d9,  [sp, #48]

cmp x4, #0
beq BSEnd

cmp x5, #0
beq BSEnd

BSLoopZ:
    mov x6, x4
    ld1 {v31.4s}, [x2], #16
    ld1 {v30.4s}, [x3], #16

    BSL32:
    cmp x6, #31
    ble BSL16
    BSLoopP32:
        ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x1], #64
        ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x1], #64
        ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x1], #64
        ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [x1], #64
        ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x1], #64
        ld1 {v20.4s, v21.4s, v22.4s, v23.4s}, [x1], #64
        ld1 {v24.4s, v25.4s, v26.4s, v27.4s}, [x1], #64
        fmul v0.4s, v0.4s, v30.4s
        fmul v1.4s, v1.4s, v30.4s
        fmul v2.4s, v2.4s, v30.4s
        fmul v3.4s, v3.4s, v30.4s
        fmul v4.4s, v4.4s, v30.4s
        fmul v5.4s, v5.4s, v30.4s
        fmul v6.4s, v6.4s, v30.4s
        fmul v7.4s, v7.4s, v30.4s
        sub x6, x6, #32
        fmul v8.4s, v8.4s, v30.4s
        fmul v9.4s, v9.4s, v30.4s
        fmul v10.4s, v10.4s, v30.4s
        fmul v11.4s, v11.4s, v30.4s
        fmul v12.4s, v12.4s, v30.4s
        fmul v13.4s, v13.4s, v30.4s
        fmul v14.4s, v14.4s, v30.4s
        fmul v15.4s, v15.4s, v30.4s

        fmul v16.4s, v16.4s, v30.4s
        fmul v17.4s, v17.4s, v30.4s
        fmul v18.4s, v18.4s, v30.4s
        fmul v19.4s, v19.4s, v30.4s
        fmul v20.4s, v20.4s, v30.4s
        fmul v21.4s, v21.4s, v30.4s
        fmul v22.4s, v22.4s, v30.4s
        fmul v23.4s, v23.4s, v30.4s

        fmul v24.4s, v24.4s, v30.4s
        fmul v25.4s, v25.4s, v30.4s
        fmul v26.4s, v26.4s, v30.4s
        fmul v27.4s, v27.4s, v30.4s
    
        fadd v0.4s, v0.4s, v31.4s
        fadd v1.4s, v1.4s, v31.4s
        fadd v2.4s, v2.4s, v31.4s
        fadd v3.4s, v3.4s, v31.4s
        fadd v4.4s, v4.4s, v31.4s
        fadd v5.4s, v5.4s, v31.4s
        fadd v6.4s, v6.4s, v31.4s
        fadd v7.4s, v7.4s, v31.4s
        cmp x6, #32
        fadd v8.4s, v8.4s, v31.4s
        fadd v9.4s, v9.4s, v31.4s
        fadd v10.4s, v10.4s, v31.4s
        fadd v11.4s, v11.4s, v31.4s
        fadd v12.4s, v12.4s, v31.4s
        fadd v13.4s, v13.4s, v31.4s
        fadd v14.4s, v14.4s, v31.4s
        fadd v15.4s, v15.4s, v31.4s
        st1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x0], #64

        fadd v16.4s, v16.4s, v31.4s
        fadd v17.4s, v17.4s, v31.4s
        fadd v18.4s, v18.4s, v31.4s
        fadd v19.4s, v19.4s, v31.4s
        fadd v20.4s, v20.4s, v31.4s
        fadd v21.4s, v21.4s, v31.4s
        fadd v22.4s, v22.4s, v31.4s
        fadd v23.4s, v23.4s, v31.4s
        ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x1], #64

        fadd v24.4s, v24.4s, v31.4s
        fadd v25.4s, v25.4s, v31.4s
        fadd v26.4s, v26.4s, v31.4s
        fadd v27.4s, v27.4s, v31.4s

        fmul v0.4s, v0.4s, v30.4s
        fmul v1.4s, v1.4s, v30.4s
        fmul v2.4s, v2.4s, v30.4s
        fmul v3.4s, v3.4s, v30.4s

        st1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x0], #64
        st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x0], #64
        st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [x0], #64
        fadd v0.4s, v0.4s, v31.4s
        fadd v1.4s, v1.4s, v31.4s
        fadd v2.4s, v2.4s, v31.4s
        fadd v3.4s, v3.4s, v31.4s
        st1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x0], #64
        st1 {v20.4s, v21.4s, v22.4s, v23.4s}, [x0], #64
        st1 {v24.4s, v25.4s, v26.4s, v27.4s}, [x0], #64
        
        st1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x0], #64

        bge BSLoopP32

    BSL16:
    cmp x6, #15
    ble BSL8_
    BSLoopP16:
        ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x1], #64
        ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x1], #64
        ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x1], #64
        ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [x1], #64
        fmul v0.4s, v0.4s, v30.4s
        fmul v1.4s, v1.4s, v30.4s
        fmul v2.4s, v2.4s, v30.4s
        fmul v3.4s, v3.4s, v30.4s
        fmul v4.4s, v4.4s, v30.4s
        fmul v5.4s, v5.4s, v30.4s
        fmul v6.4s, v6.4s, v30.4s
        fmul v7.4s, v7.4s, v30.4s
        sub x6, x6, #16
        fmul v8.4s, v8.4s, v30.4s
        fmul v9.4s, v9.4s, v30.4s
        fmul v10.4s, v10.4s, v30.4s
        fmul v11.4s, v11.4s, v30.4s
        fmul v12.4s, v12.4s, v30.4s
        fmul v13.4s, v13.4s, v30.4s
        fmul v14.4s, v14.4s, v30.4s
        fmul v15.4s, v15.4s, v30.4s
    
        fadd v0.4s, v0.4s, v31.4s
        fadd v1.4s, v1.4s, v31.4s
        fadd v2.4s, v2.4s, v31.4s
        fadd v3.4s, v3.4s, v31.4s
        fadd v4.4s, v4.4s, v31.4s
        fadd v5.4s, v5.4s, v31.4s
        fadd v6.4s, v6.4s, v31.4s
        fadd v7.4s, v7.4s, v31.4s
        cmp x6, #16
        fadd v8.4s, v8.4s, v31.4s
        fadd v9.4s, v9.4s, v31.4s
        fadd v10.4s, v10.4s, v31.4s
        fadd v11.4s, v11.4s, v31.4s
        fadd v12.4s, v12.4s, v31.4s
        fadd v13.4s, v13.4s, v31.4s
        fadd v14.4s, v14.4s, v31.4s
        fadd v15.4s, v15.4s, v31.4s

        st1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x0], #64
        st1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x0], #64
        st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x0], #64
        st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [x0], #64

        bge BSLoopP16

    BSL8_:
    cmp x6, #7
    ble BSL1
    BSLoopP8:
        ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x1], #64
        fmul v0.4s, v0.4s, v30.4s
        fmul v1.4s, v1.4s, v30.4s
        fmul v2.4s, v2.4s, v30.4s
        fmul v3.4s, v3.4s, v30.4s
        ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x1], #64
        fadd v0.4s, v0.4s, v31.4s
        fadd v1.4s, v1.4s, v31.4s
        fadd v2.4s, v2.4s, v31.4s
        fadd v3.4s, v3.4s, v31.4s
        fmul v4.4s, v4.4s, v30.4s
        fmul v5.4s, v5.4s, v30.4s
        fmul v6.4s, v6.4s, v30.4s
        fmul v7.4s, v7.4s, v30.4s
        st1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x0], #64
        sub x6, x6, #8
        fadd v4.4s, v4.4s, v31.4s
        fadd v5.4s, v5.4s, v31.4s
        fadd v6.4s, v6.4s, v31.4s
        fadd v7.4s, v7.4s, v31.4s
        st1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x0], #64
        cmp x6, #8
        bge BSLoopP8
    BSL1:
    cmp x6, #0
    beq BSLoopPEnd

    BSLoopP1:
        ld1 {v0.4s}, [x1], #16
        fmul v0.4s, v0.4s, v30.4s
        fadd v0.4s, v0.4s, v31.4s
        st1 {v0.4s}, [x0], #16
        subs x6, x6, #1
        bne BSLoopP1
    BSLoopPEnd:

    subs x5, x5, #1
    bne BSLoopZ


BSEnd:
ldp d8,  d9,  [sp, #48]
ldp d10, d11, [sp, #32]
ldp d12, d13, [sp, #16]
ldp d14, d15, [sp], #64

ret


#endif
