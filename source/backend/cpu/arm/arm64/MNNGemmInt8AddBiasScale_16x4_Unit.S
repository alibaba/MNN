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
    ld1 {v0.16b}, [x2], #16
    ld1 {v1.16b}, [x2], #16
    ld1 {v2.16b}, [x2], #16
    ld1 {v3.16b}, [x2], #16
    dup v16.4s, wzr
    dup v17.4s, wzr
    ld1 {v4.16b}, [x1], #16
    ld1 {v5.16b}, [x1], #16
    ld1 {v6.16b}, [x1], #16
    ld1 {v7.16b}, [x1], #16
    
    smull v8.8h, v0.8b, v4.8b
    dup v18.4s, wzr
    smull v9.8h, v1.8b, v4.8b
    dup v19.4s, wzr
    smull v10.8h, v2.8b, v4.8b
    dup v20.4s, wzr
    smull v11.8h, v3.8b, v4.8b
    dup v21.4s, wzr
    smull v12.8h, v0.8b, v5.8b
    dup v22.4s, wzr
    smull v13.8h, v1.8b, v5.8b
    dup v23.4s, wzr
    smull v14.8h, v2.8b, v5.8b
    dup v24.4s, wzr
    smull v15.8h, v3.8b, v5.8b
    subs x9, x5, #1
    smlal2 v8.8h, v0.16b, v4.16b
    dup v25.4s, wzr
    smlal2 v9.8h, v1.16b, v4.16b
    dup v26.4s, wzr
    smlal2 v10.8h, v2.16b, v4.16b
    dup v27.4s, wzr
    smlal2 v11.8h, v3.16b, v4.16b
    dup v28.4s, wzr
    smlal2 v12.8h, v0.16b, v5.16b
    dup v29.4s, wzr
    smlal2 v13.8h, v1.16b, v5.16b
    dup v30.4s, wzr
    smlal2 v14.8h, v2.16b, v5.16b
    dup v31.4s, wzr
    smlal2 v15.8h, v3.16b, v5.16b
    beq L4LoopSzEnd

    L4LoopSz:
        
        sadalp v16.4s, v8.8h
        ld1 {v4.16b}, [x1], #16
        smull v8.8h, v0.8b, v6.8b
        sadalp v17.4s, v9.8h
        ld1 {v5.16b}, [x1], #16
        smull v9.8h, v1.8b, v6.8b
        sadalp v18.4s, v10.8h
        smull v10.8h, v2.8b, v6.8b
        sadalp v19.4s, v11.8h
        smull v11.8h, v3.8b, v6.8b
        sadalp v20.4s, v12.8h
        smull v12.8h, v0.8b, v7.8b
        sadalp v21.4s, v13.8h
        smull v13.8h, v1.8b, v7.8b
        sadalp v22.4s, v14.8h
        smull v14.8h, v2.8b, v7.8b
        sadalp v23.4s, v15.8h
        smull v15.8h, v3.8b, v7.8b

        smlal2 v8.8h,  v0.16b, v6.16b
        smlal2 v9.8h,  v1.16b, v6.16b
        smlal2 v10.8h, v2.16b, v6.16b
        smlal2 v11.8h, v3.16b, v6.16b

        ld1 {v6.16b}, [x1], #16

        smlal2 v12.8h, v0.16b, v7.16b
        ld1 {v0.16b}, [x2], #16
        smlal2 v13.8h, v1.16b, v7.16b
        ld1 {v1.16b}, [x2], #16
        smlal2 v14.8h, v2.16b, v7.16b
        ld1 {v2.16b}, [x2], #16
        smlal2 v15.8h, v3.16b, v7.16b
        ld1 {v3.16b}, [x2], #16

        sadalp v24.4s, v8.8h
        smull v8.8h, v0.8b, v4.8b
        sadalp v25.4s, v9.8h
        ld1 {v7.16b}, [x1], #16
        smull v9.8h, v1.8b, v4.8b
        sadalp v26.4s, v10.8h
        smull v10.8h, v2.8b, v4.8b
        sadalp v27.4s, v11.8h
        smull v11.8h, v3.8b, v4.8b
        sadalp v28.4s, v12.8h
        smull v12.8h, v0.8b, v5.8b
        sadalp v29.4s, v13.8h
        smull v13.8h, v1.8b, v5.8b
        sadalp v30.4s, v14.8h
        smull v14.8h, v2.8b, v5.8b
        sadalp v31.4s, v15.8h
        smull v15.8h, v3.8b, v5.8b

        smlal2 v8.8h, v0.16b, v4.16b
        smlal2 v9.8h, v1.16b, v4.16b
        smlal2 v10.8h, v2.16b, v4.16b
        smlal2 v11.8h, v3.16b, v4.16b

        smlal2 v12.8h, v0.16b, v5.16b
        smlal2 v13.8h, v1.16b, v5.16b
        smlal2 v14.8h, v2.16b, v5.16b
        smlal2 v15.8h, v3.16b, v5.16b

        subs x9, x9, #1
        bne L4LoopSz

    L4LoopSzEnd:
    sadalp v16.4s, v8.8h
    smull v8.8h, v0.8b, v6.8b
    sadalp v17.4s, v9.8h
    smull v9.8h, v1.8b, v6.8b
    sadalp v18.4s, v10.8h
    smull v10.8h, v2.8b, v6.8b
    sadalp  v19.4s, v11.8h
    smull v11.8h, v3.8b, v6.8b
    sadalp  v20.4s, v12.8h
    smull v12.8h, v0.8b, v7.8b
    sadalp  v21.4s, v13.8h
    smull v13.8h, v1.8b, v7.8b
    sadalp  v22.4s, v14.8h
    smull v14.8h, v2.8b, v7.8b
    sadalp  v23.4s, v15.8h
    smull v15.8h, v3.8b, v7.8b

    smlal2 v8.8h, v0.16b, v6.16b
    smlal2 v9.8h, v1.16b, v6.16b
    smlal2 v10.8h, v2.16b, v6.16b
    smlal2 v11.8h, v3.16b, v6.16b

    smlal2 v12.8h, v0.16b, v7.16b
    smlal2 v13.8h, v1.16b, v7.16b
    smlal2 v14.8h, v2.16b, v7.16b
    smlal2 v15.8h, v3.16b, v7.16b

    sadalp v24.4s, v8.8h
    sadalp v25.4s, v9.8h
    sadalp v26.4s, v10.8h
    sadalp v27.4s, v11.8h
    sadalp v28.4s, v12.8h
    sadalp v29.4s, v13.8h
    sadalp v30.4s, v14.8h
    sadalp v31.4s, v15.8h

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
    subs x7, x7, #1
    mov x1, x8
    st1 {v3.8b}, [x0], x6
    bne L4LoopDz


sub sp, sp, #128
ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64
ret

#endif
