//
//  MNNGemmInt8toFloat32_8x4_Unit.S
//  MNN
//
//  Created by MNN on 2018/12/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNGemmInt8toFloat32_8x4_Unit
//void MNNGemmInt8toFloat32_8x4_Unit(float* dst, const uint8_t* src, const uint8_t* weight, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad);
//Auto: x0: dst, x1: src, x2:weight, x3: src_depth_quad
//x4: dst_step, x5: dst_depth_quad

sub sp, sp, #128
st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64

mov x12, #64
sub x4, x4, x12

L6LoopDz:
    mov x6, x1
    subs x7, x3, #1
    ld1 {v4.8b, v5.8b, v6.8b, v7.8b}, [x2], #32
.macro START_TWO z0 z1 z2 z3 z4 z5 z6 z7
        ld1 {v2.8b, v3.8b}, [x1], #16
        smull v0.8h, v4.8b, v2.8b
        saddlp \z0, v0.8h
        smull v1.8h, v4.8b, v3.8b
        saddlp \z4, v1.8h
        smull v0.8h, v5.8b, v2.8b
        saddlp \z1, v0.8h
        smull v1.8h, v5.8b, v3.8b
        saddlp \z5, v1.8h
        smull v0.8h, v6.8b, v2.8b
        saddlp \z2, v0.8h
        smull v1.8h, v6.8b, v3.8b
        saddlp \z6, v1.8h
        smull v0.8h, v7.8b, v2.8b
        saddlp \z3, v0.8h
        smull v1.8h, v7.8b, v3.8b
        saddlp \z7, v1.8h
.endm
    START_TWO v8.4s, v9.4s, v10.4s, v11.4s, v12.4s, v13.4s, v14.4s, v15.4s
    START_TWO v16.4s, v17.4s, v18.4s, v19.4s, v20.4s, v21.4s, v22.4s, v23.4s
    START_TWO v24.4s, v25.4s, v26.4s, v27.4s, v28.4s, v29.4s, v30.4s, v31.4s
    beq L6LoopSzEnd
    L6LoopSz:
        ld1 {v4.8b, v5.8b, v6.8b, v7.8b}, [x2], #32
.macro COMPUTE_THREE z0 z1 z2 z3 z4 z5 z6 z7 z8 z9 z10 z11
        ld1 {v3.8b}, [x1], #8
        smull v0.8h, v4.8b, v3.8b
        smull v1.8h, v5.8b, v3.8b
        sadalp \z0, v0.8h
        smull v2.8h, v6.8b, v3.8b
        sadalp \z1, v1.8h
        smull v0.8h, v7.8b, v3.8b
        sadalp \z2, v2.8h
        ld1 {v3.8b}, [x1], #8
        smull v1.8h, v4.8b, v3.8b
        sadalp \z3, v0.8h
        smull v2.8h, v5.8b, v3.8b
        sadalp \z4, v1.8h
        smull v0.8h, v6.8b, v3.8b
        sadalp \z5, v2.8h
        smull v1.8h, v7.8b, v3.8b
        ld1 {v3.8b}, [x1], #8
        sadalp \z6, v0.8h
        smull v2.8h, v4.8b, v3.8b
        sadalp \z7, v1.8h
        smull v0.8h, v5.8b, v3.8b
        sadalp \z8, v2.8h
        smull v1.8h, v6.8b, v3.8b
        sadalp \z9, v0.8h
        smull v2.8h, v7.8b, v3.8b
        sadalp \z10, v1.8h
        sadalp \z11, v2.8h
.endm
        COMPUTE_THREE v8.4s, v9.4s, v10.4s, v11.4s, v12.4s, v13.4s, v14.4s, v15.4s, v16.4s, v17.4s, v18.4s, v19.4s
        COMPUTE_THREE v20.4s, v21.4s, v22.4s, v23.4s, v24.4s, v25.4s, v26.4s, v27.4s, v28.4s, v29.4s, v30.4s, v31.4s
        subs x7, x7, #1
        bne L6LoopSz
    L6LoopSzEnd:

    addp v8.4s, v8.4s, v9.4s
    addp v10.4s, v10.4s, v11.4s
    addp v12.4s, v12.4s, v13.4s
    addp v14.4s, v14.4s, v15.4s
    addp v16.4s, v16.4s, v17.4s
    addp v18.4s, v18.4s, v19.4s
    addp v20.4s, v20.4s, v21.4s
    addp v22.4s, v22.4s, v23.4s
    addp v24.4s, v24.4s, v25.4s
    addp v26.4s, v26.4s, v27.4s
    addp v28.4s, v28.4s, v29.4s
    addp v30.4s, v30.4s, v31.4s

    addp v8.4s, v8.4s, v10.4s
    addp v9.4s, v12.4s, v14.4s
    scvtf v8.4s, v8.4s
    addp v10.4s, v16.4s, v18.4s
    scvtf v9.4s, v9.4s
    addp v11.4s, v20.4s, v22.4s
    scvtf v10.4s, v10.4s
    addp v12.4s, v24.4s, v26.4s
    scvtf v11.4s, v11.4s
    addp v13.4s, v28.4s, v30.4s

    scvtf v12.4s, v12.4s
    st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x0], #64
    scvtf v13.4s, v13.4s

    st1 {v12.4s, v13.4s}, [x0], x4

    subs x5, x5, #1
    mov x1, x6
    bne L6LoopDz

sub sp, sp, #128
ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64
ret

#endif
