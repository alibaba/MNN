//
//  MNNGemmInt8AddBiasScale_16x4_Unit.S
//  MNN
//
//  Created by MNN on 2019/06/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNGemmInt8AddBiasScale_16x4_Unit

// void MNNGemmInt8AddBiasScale_16x4_Unit(int8_t* dst, const int8_t* src, 
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
    ld1 {v12.16b, v13.16b}, [x2], #32
    ld1 {v8.16b, v9.16b}, [x1], #32
    subs x9, x5, #1
    // first two
    smull v0.8h, v12.8b, v8.8b
    smull v1.8h, v13.8b, v8.8b
    ld1 {v14.16b, v15.16b}, [x2], #32
    smlal2 v0.8h, v12.16b, v8.16b
    smlal2 v1.8h, v13.16b, v8.16b
    saddlp v16.4s, v0.8h
    saddlp v17.4s, v1.8h

    smull v2.8h, v14.8b, v8.8b
    smull v3.8h, v15.8b, v8.8b
    smull v4.8h, v12.8b, v9.8b
    smull v5.8h, v13.8b, v9.8b
    smull v6.8h, v14.8b, v9.8b
    smull v7.8h, v15.8b, v9.8b
    smlal2 v2.8h, v14.16b, v8.16b
    smlal2 v3.8h, v15.16b, v8.16b
    smlal2 v4.8h, v12.16b, v9.16b
    smlal2 v5.8h, v13.16b, v9.16b
    smlal2 v6.8h, v14.16b, v9.16b
    smlal2 v7.8h, v15.16b, v9.16b
    ld1 {v10.16b, v11.16b}, [x1], #32
    saddlp v18.4s, v2.8h
    saddlp v19.4s, v3.8h
    saddlp v20.4s, v4.8h
    saddlp v21.4s, v5.8h
    saddlp v22.4s, v6.8h
    saddlp v23.4s, v7.8h

    // second two
    smull v0.8h, v12.8b, v10.8b
    smull v1.8h, v13.8b, v10.8b
    smull v2.8h, v14.8b, v10.8b
    smull v3.8h, v15.8b, v10.8b
    smlal2 v0.8h, v12.16b, v10.16b
    smlal2 v1.8h, v13.16b, v10.16b
    smlal2 v2.8h, v14.16b, v10.16b
    smlal2 v3.8h, v15.16b, v10.16b
    saddlp v24.4s, v0.8h
    saddlp v25.4s, v1.8h
    saddlp v26.4s, v2.8h
    saddlp v27.4s, v3.8h

    smull v4.8h, v12.8b, v11.8b
    smull v5.8h, v13.8b, v11.8b
    smull v6.8h, v14.8b, v11.8b
    smull v7.8h, v15.8b, v11.8b
    smlal2 v4.8h, v12.16b, v11.16b
    smlal2 v5.8h, v13.16b, v11.16b
    smlal2 v6.8h, v14.16b, v11.16b
    smlal2 v7.8h, v15.16b, v11.16b
    saddlp v28.4s, v4.8h
    saddlp v29.4s, v5.8h
    saddlp v30.4s, v6.8h
    saddlp v31.4s, v7.8h
    
    beq L4LoopSzEnd

    L4LoopSz:
        ld1 {v12.16b, v13.16b}, [x2], #32
        ld1 {v8.16b, v9.16b}, [x1], #32
        // first two
        smull v0.8h, v12.8b, v8.8b
        smull v1.8h, v13.8b, v8.8b
        ld1 {v14.16b, v15.16b}, [x2], #32
        smlal2 v0.8h, v12.16b, v8.16b
        smlal2 v1.8h, v13.16b, v8.16b
        sadalp v16.4s, v0.8h
        sadalp v17.4s, v1.8h

        smull v2.8h, v14.8b, v8.8b
        smull v3.8h, v15.8b, v8.8b
        smull v4.8h, v12.8b, v9.8b
        smull v5.8h, v13.8b, v9.8b
        smull v6.8h, v14.8b, v9.8b
        smull v7.8h, v15.8b, v9.8b
        ld1 {v10.16b, v11.16b}, [x1], #32
        smlal2 v2.8h, v14.16b, v8.16b
        smlal2 v3.8h, v15.16b, v8.16b
        smlal2 v4.8h, v12.16b, v9.16b
        smlal2 v5.8h, v13.16b, v9.16b
        smlal2 v6.8h, v14.16b, v9.16b
        smlal2 v7.8h, v15.16b, v9.16b
        sadalp v18.4s, v2.8h
        sadalp v19.4s, v3.8h
        sadalp v20.4s, v4.8h
        sadalp v21.4s, v5.8h
        sadalp v22.4s, v6.8h
        sadalp v23.4s, v7.8h

        // second two
        smull v0.8h, v12.8b, v10.8b
        smull v1.8h, v13.8b, v10.8b
        smull v2.8h, v14.8b, v10.8b
        smull v3.8h, v15.8b, v10.8b
        smlal2 v0.8h, v12.16b, v10.16b
        smlal2 v1.8h, v13.16b, v10.16b
        smlal2 v2.8h, v14.16b, v10.16b
        smlal2 v3.8h, v15.16b, v10.16b
        sadalp v24.4s, v0.8h
        sadalp v25.4s, v1.8h
        sadalp v26.4s, v2.8h
        sadalp v27.4s, v3.8h

        smull v4.8h, v12.8b, v11.8b
        smull v5.8h, v13.8b, v11.8b
        smull v6.8h, v14.8b, v11.8b
        smull v7.8h, v15.8b, v11.8b
        smlal2 v4.8h, v12.16b, v11.16b
        smlal2 v5.8h, v13.16b, v11.16b
        smlal2 v6.8h, v14.16b, v11.16b
        smlal2 v7.8h, v15.16b, v11.16b
        sadalp v28.4s, v4.8h
        sadalp v29.4s, v5.8h
        sadalp v30.4s, v6.8h
        sadalp v31.4s, v7.8h
        subs x9, x9, #1
        bne L4LoopSz

    L4LoopSzEnd:

    ld1 {v0.4s}, [x3], #16
    addp v4.4s, v16.4s, v17.4s
    addp v5.4s, v18.4s, v19.4s
    addp v6.4s, v20.4s, v21.4s
    addp v7.4s, v22.4s, v23.4s
    addp v8.4s, v24.4s, v25.4s
    addp v9.4s, v26.4s, v27.4s
    addp v10.4s, v28.4s, v29.4s
    addp v11.4s, v30.4s, v31.4s

    addp v12.4s, v4.4s, v5.4s
    addp v13.4s, v6.4s, v7.4s
    addp v14.4s, v8.4s, v9.4s
    addp v15.4s, v10.4s, v11.4s
    ld1 {v1.4s}, [x4], #16
    add v16.4s, v12.4s, v0.4s
    add v17.4s, v13.4s, v0.4s
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

    fcvtzs v8.4s, v12.4s
    fcvtzs v9.4s, v13.4s
    fcvtzs v10.4s, v14.4s
    fcvtzs v11.4s, v15.4s

    sqxtn v0.4h, v8.4s
    sqxtn2 v0.8h, v9.4s
    sqxtn v1.4h, v10.4s
    sqxtn2 v1.8h, v11.4s

    sqxtn v2.8b, v0.8h
    sqxtn v3.8b, v1.8h
    st1 {v2.8b}, [x0], #8
    subs x7, x7, #1
    mov x1, x8
    st1 {v3.8b}, [x0], x6
    bne L4LoopDz


sub sp, sp, #128
ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64
ret

#endif
