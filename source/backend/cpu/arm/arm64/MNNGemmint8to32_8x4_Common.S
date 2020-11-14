//
//  MNNGemmint8to32_8x4_Common.S
//  MNN
//
//  Created by MNN on 2019/08/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNGemmint8to32_8x4_Common
//void MNNGemmint8to32_8x4_Common(int32_t* dst, const int8_t* src, const int8_t* weight, const int32_t* inputSummer, size_t src_depth_quad,
//                                  size_t dst_step, size_t dst_depth_quad, size_t width);


//Auto: x0: dst*, x1: src*, x2:weight*, x3: inputSummer*
//x4: src_depth_quad, x5: dst_step, x6: dst_depth_quad, x7: width

sub sp, sp, #128
st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64

.macro COMPUTE_L2_L4 z0, z1, z2, z3
\z0 v0.8h, v12.\z1, \z2
\z0 v1.8h, v13.\z1, \z2
\z0 v2.8h, v14.\z1, \z2
\z0 v3.8h, v15.\z1, \z2
\z0 v4.8h, v12.\z1, \z3
\z0 v5.8h, v13.\z1, \z3
\z0 v6.8h, v14.\z1, \z3
\z0 v7.8h, v15.\z1, \z3
.endm

.macro MERGE_L2_L4_0 z0
\z0 v16.4s, v0.8h
\z0 v17.4s, v1.8h
\z0 v18.4s, v2.8h
\z0 v19.4s, v3.8h
\z0 v20.4s, v4.8h
\z0 v21.4s, v5.8h
\z0 v22.4s, v6.8h
\z0 v23.4s, v7.8h
.endm

.macro MERGE_L2_L4_1 z0
\z0 v24.4s, v0.8h
\z0 v25.4s, v1.8h
\z0 v26.4s, v2.8h
\z0 v27.4s, v3.8h
\z0 v28.4s, v4.8h
\z0 v29.4s, v5.8h
\z0 v30.4s, v6.8h
\z0 v31.4s, v7.8h
.endm

mov x11, #16
mul x11, x11, x7

L4:
cmp x7, #2
ble L2
mov x10, x2
mov x13, x0
mov x12, x6

L4LoopDz:
    mov x8, x1
    ld1 {v8.16b, v9.16b, v10.16b, v11.16b}, [x1], x11
    ld1 {v12.16b, v13.16b, v14.16b, v15.16b}, [x2], #64
    subs x9, x4, #1
    // first two

    COMPUTE_L2_L4 smull, 8b, v8.8b, v9.8b
    COMPUTE_L2_L4 smlal2, 16b, v8.16b, v9.16b
    MERGE_L2_L4_0 saddlp

    COMPUTE_L2_L4 smull, 8b, v10.8b, v11.8b
    COMPUTE_L2_L4 smlal2, 16b, v10.16b, v11.16b
    MERGE_L2_L4_1 saddlp
    
    beq L4LoopSzEnd

    L4LoopSz:
        ld1 {v8.16b, v9.16b, v10.16b, v11.16b}, [x1], x11
        ld1 {v12.16b, v13.16b, v14.16b, v15.16b}, [x2], #64

        COMPUTE_L2_L4 smull, 8b, v8.8b, v9.8b
        COMPUTE_L2_L4 smlal2, 16b, v8.16b, v9.16b
        MERGE_L2_L4_0 sadalp

        COMPUTE_L2_L4 smull, 8b, v10.8b, v11.8b
        COMPUTE_L2_L4 smlal2, 16b, v10.16b, v11.16b
        MERGE_L2_L4_1 sadalp

        subs x9, x9, #1
        bne L4LoopSz

    L4LoopSzEnd:
    
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

    st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [x0], x5
    subs x12, x12, #1
    mov x1, x8
    bne L4LoopDz

L4End:
add x0, x13, #64
add x1, x1, #64
mov x2, x10
sub x7, x7, #4
cmp x7, #4
bge L4

L2:
cmp x7, #1
ble L1
mov x10, x2
mov x13, x0
mov x12, x6

L2LoopDz:
    mov x8, x1
    ld1 {v12.16b, v13.16b, v14.16b, v15.16b}, [x2], #64
    ld1 {v8.16b, v9.16b}, [x1], x11
    subs x9, x4, #1
    
    COMPUTE_L2_L4 smull, 8b, v8.8b, v9.8b
    COMPUTE_L2_L4 smlal2, 16b, v8.16b, v9.16b
    MERGE_L2_L4_0 saddlp

    beq L2LoopSzEnd

    L2LoopSz:
        ld1 {v12.16b, v13.16b, v14.16b, v15.16b}, [x2], #64
        ld1 {v8.16b, v9.16b}, [x1], x11
        
        COMPUTE_L2_L4 smull, 8b, v8.8b, v9.8b
        COMPUTE_L2_L4 smlal2, 16b, v8.16b, v9.16b
        MERGE_L2_L4_0 sadalp

        subs x9, x9, #1
        bne L2LoopSz

    L2LoopSzEnd:

    addp v4.4s, v16.4s, v17.4s
    addp v5.4s, v18.4s, v19.4s
    addp v6.4s, v20.4s, v21.4s
    addp v7.4s, v22.4s, v23.4s

    addp v12.4s, v4.4s, v5.4s
    addp v13.4s, v6.4s, v7.4s

    st1 {v12.4s, v13.4s}, [x0], x5
    subs x12, x12, #1
    mov x1, x8
    bne L2LoopDz

L2End:
add x0, x13, #32
add x1, x1, #32
mov x2, x10
sub x7, x7, #2

.macro COMPUTE_L1 z0, z1, z2
\z0 v0.8h, v12.\z1, \z2
\z0 v1.8h, v13.\z1, \z2
\z0 v2.8h, v14.\z1, \z2
\z0 v3.8h, v15.\z1, \z2
.endm

.macro MERGE_L1 z0
\z0 v16.4s, v0.8h
\z0 v17.4s, v1.8h
\z0 v18.4s, v2.8h
\z0 v19.4s, v3.8h
.endm

L1:
cbz x7, End

L1LoopDz:
    mov x8, x1
    ld1 {v12.16b, v13.16b, v14.16b, v15.16b}, [x2], #64
    ld1 {v8.16b}, [x1], x11

    COMPUTE_L1 smull, 8b, v8.8b
    COMPUTE_L1 smlal2, 16b, v8.16b
    MERGE_L1 saddlp

    subs x9, x4, #1
    beq L1LoopSzEnd
    L1LoopSz:
        ld1 {v12.16b, v13.16b, v14.16b, v15.16b}, [x2], #64
        ld1 {v8.16b}, [x1], x11

        COMPUTE_L1 smull, 8b, v8.8b
        COMPUTE_L1 smlal2, 16b, v8.16b
        MERGE_L1 sadalp

        subs x9, x9, #1
        bne L1LoopSz
    L1LoopSzEnd:

    addp v4.4s, v16.4s, v17.4s
    addp v5.4s, v18.4s, v19.4s
    addp v12.4s, v4.4s, v5.4s

    st1 {v12.4s}, [x0], x5
    subs x6, x6, #1
    mov x1, x8
    bne L1LoopDz

End:
sub sp, sp, #128
ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64
ret

#endif
