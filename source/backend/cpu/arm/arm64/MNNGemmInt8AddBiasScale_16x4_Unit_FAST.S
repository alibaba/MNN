//
//  MNNGemmInt8AddBiasScale_16x4_Unit_FAST.S
//  MNN
//
//  Created by MNN on 2020/03/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNGemmInt8AddBiasScale_16x4_Unit_FAST

// void MNNGemmInt8AddBiasScale_16x4_Unit_FAST(int8_t* dst, const int8_t* src, 
//          const int8_t* weight, const int32_t* bias, const float* scale, 
//          size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad)

//Auto: x0: dst*, x1: src*, x2:weight*, x3: bias*
// x4: scale*, x5: src_depth_quad, x6: dst_step, 
// x7: dst_depth_quad

sub sp, sp, #128
st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64

sub x6, x6, #8

L4LoopDz:
    mov x8, x1
    // load four weights
    ld1 {v0.16b}, [x2], #16
    ld1 {v1.16b}, [x2], #16
    ld1 {v2.16b}, [x2], #16
    ld1 {v3.16b}, [x2], #16
    // load one tile input
    ld1 {v4.16b}, [x1], #16
    smull v16.8h, v0.8b, v4.8b
    smull v17.8h, v1.8b, v4.8b
    ld1 {v5.16b}, [x1], #16
    smull v18.8h, v2.8b, v4.8b
    mov x9, x5
    smull v19.8h, v3.8b, v4.8b
    smull v20.8h, v0.8b, v5.8b
    smull v21.8h, v1.8b, v5.8b
    ld1 {v6.16b}, [x1], #16
    smull v22.8h, v2.8b, v5.8b
    smull v23.8h, v3.8b, v5.8b
    smull v24.8h, v0.8b, v6.8b
    smull v25.8h, v1.8b, v6.8b
    ld1 {v7.16b}, [x1], #16
    smull v26.8h, v2.8b, v6.8b
    smull v27.8h, v3.8b, v6.8b
    smull v28.8h, v0.8b, v7.8b
    smull v29.8h, v1.8b, v7.8b
    subs x9, x9, #1
    smull v30.8h, v2.8b, v7.8b
    smull v31.8h, v3.8b, v7.8b
    
    beq L4LoopSzEnd

    L4LoopSz:
        smlal2 v16.8h, v0.16b, v4.16b
        smlal2 v17.8h, v1.16b, v4.16b
        smlal2 v18.8h, v2.16b, v4.16b
        smlal2 v19.8h, v3.16b, v4.16b
        smlal2 v20.8h, v0.16b, v5.16b
        ld1 {v4.16b}, [x1], #16
        smlal2 v21.8h, v1.16b, v5.16b
        smlal2 v22.8h, v2.16b, v5.16b
        smlal2 v23.8h, v3.16b, v5.16b
        smlal2 v24.8h, v0.16b, v6.16b
        ld1 {v5.16b}, [x1], #16
        smlal2 v25.8h, v1.16b, v6.16b
        smlal2 v26.8h, v2.16b, v6.16b
        smlal2 v27.8h, v3.16b, v6.16b
        smlal2 v28.8h, v0.16b, v7.16b
        ld1 {v6.16b}, [x1], #16
        smlal2 v29.8h, v1.16b, v7.16b
        ld1 {v0.16b}, [x2], #16
        smlal2 v30.8h, v2.16b, v7.16b
        ld1 {v1.16b}, [x2], #16
        smlal2 v31.8h, v3.16b, v7.16b
        ld1 {v2.16b}, [x2], #16

        smlal v16.8h, v0.8b, v4.8b
        ld1 {v7.16b}, [x1], #16
        smlal v17.8h, v1.8b, v4.8b
        ld1 {v3.16b}, [x2], #16
        smlal v18.8h, v2.8b, v4.8b
        smlal v19.8h, v3.8b, v4.8b
        smlal v20.8h, v0.8b, v5.8b
        smlal v21.8h, v1.8b, v5.8b
        smlal v22.8h, v2.8b, v5.8b
        smlal v23.8h, v3.8b, v5.8b
        smlal v24.8h, v0.8b, v6.8b
        smlal v25.8h, v1.8b, v6.8b
        smlal v26.8h, v2.8b, v6.8b
        smlal v27.8h, v3.8b, v6.8b
        smlal v28.8h, v0.8b, v7.8b
        smlal v29.8h, v1.8b, v7.8b
        smlal v30.8h, v2.8b, v7.8b
        subs x9, x9, #1
        smlal v31.8h, v3.8b, v7.8b
        bne L4LoopSz
    L4LoopSzEnd:

    smlal2 v16.8h, v0.16b, v4.16b
    smlal2 v17.8h, v1.16b, v4.16b
    smlal2 v18.8h, v2.16b, v4.16b
    smlal2 v19.8h, v3.16b, v4.16b
    smlal2 v20.8h, v0.16b, v5.16b
    smlal2 v21.8h, v1.16b, v5.16b
    smlal2 v22.8h, v2.16b, v5.16b
    smlal2 v23.8h, v3.16b, v5.16b
    smlal2 v24.8h, v0.16b, v6.16b
    smlal2 v25.8h, v1.16b, v6.16b
    smlal2 v26.8h, v2.16b, v6.16b
    smlal2 v27.8h, v3.16b, v6.16b
    smlal2 v28.8h, v0.16b, v7.16b
    smlal2 v29.8h, v1.16b, v7.16b
    smlal2 v30.8h, v2.16b, v7.16b
    smlal2 v31.8h, v3.16b, v7.16b

    saddlp v15.4s, v16.8h
    saddlp v14.4s, v17.8h
    saddlp v13.4s, v18.8h
    saddlp v12.4s, v19.8h
    saddlp v11.4s, v20.8h
    saddlp v10.4s, v21.8h
    saddlp v9.4s,  v22.8h
    saddlp v8.4s,  v23.8h
    saddlp v7.4s,  v24.8h
    saddlp v6.4s,  v25.8h
    saddlp v5.4s,  v26.8h
    saddlp v4.4s,  v27.8h
    saddlp v3.4s,  v28.8h
    saddlp v2.4s,  v29.8h
    saddlp v1.4s,  v30.8h
    saddlp v0.4s,  v31.8h
    
    addp v16.4s, v15.4s, v14.4s
    addp v17.4s, v13.4s, v12.4s
    addp v18.4s, v11.4s, v10.4s
    addp v19.4s, v9.4s, v8.4s
    addp v20.4s, v7.4s, v6.4s
    addp v21.4s, v5.4s, v4.4s
    addp v22.4s, v3.4s, v2.4s
    addp v23.4s, v1.4s, v0.4s

    addp v12.4s, v16.4s, v17.4s
    addp v13.4s, v18.4s, v19.4s
    ld1 {v0.4s}, [x3], #16
    addp v14.4s, v20.4s, v21.4s
    addp v15.4s, v22.4s, v23.4s

    add v16.4s, v12.4s, v0.4s
    add v17.4s, v13.4s, v0.4s
    ld1 {v1.4s}, [x4], #16
    add v18.4s, v14.4s, v0.4s
    add v19.4s, v15.4s, v0.4s

    scvtf v4.4s, v16.4s
    scvtf v5.4s, v17.4s
    scvtf v6.4s, v18.4s
    scvtf v7.4s, v19.4s

    fmul v12.4s, v4.4s, v1.4s
    fmul v13.4s, v5.4s, v1.4s
    fmul v14.4s, v6.4s, v1.4s
    fmul v15.4s, v7.4s, v1.4s

    fcvtas v8.4s, v12.4s
    fcvtas v9.4s, v13.4s
    fcvtas v10.4s, v14.4s
    fcvtas v11.4s, v15.4s

    sqxtn v0.4h, v8.4s
    sqxtn2 v0.8h, v9.4s
    sqxtn v1.4h, v10.4s
    sqxtn2 v1.8h, v11.4s

    sqxtn v2.8b, v0.8h
    sqxtn v3.8b, v1.8h
    st1 {v2.8b}, [x0], #8
    st1 {v3.8b}, [x0], x6
    subs x7, x7, #1
    mov x1, x8

    bne L4LoopDz


sub sp, sp, #128
ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64
ret

#endif
