//
//  MNNGemmInt8toFloat32_8x4_Common.S
//  MNN
//
//  Created by MNN on 2018/12/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNGemmInt8toFloat32_8x4_Common
//void MNNGemmInt8toFloat32_8x4_Common(float* dst, const uint8_t* src, const uint8_t* weight, size_t src_depth_quad,
//                                     size_t width, size_t dst_step, size_t dst_depth_quad);
//Auto: x0: dst, x1: src, x2:weight, x3: src_depth_quad
//      x4: width, x5: dst_step, x6: dst_depth_quad

sub sp, sp, #128
st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64

// step multi by sizeof(float)
mov x12, #4
mul x5, x12, x5

// src_z_step
mov x12, #8
mul x7, x12, x4

cmp x4, #3
blt L2

.macro COMPUTE z0, z1, z2, z3, z4
smull \z1, v0.8b, \z0
smull \z2, v1.8b, \z0
smull \z3, v2.8b, \z0
smull \z4, v3.8b, \z0
.endm

.macro MERGE z0, z1, z2, z3, z4, z5, z6, z7, z8
\z0 \z5, \z1
\z0 \z6, \z2
\z0 \z7, \z3
\z0 \z8, \z4
.endm

L3:
mov x8, x0
mov x9, x2
mov x10, x6

L3LoopDz:
mov x11, x1
mov x12, x3
ld1 {v0.8b, v1.8b, v2.8b, v3.8b}, [x9], #32
ld1 {v4.8b, v5.8b, v6.8b}, [x11], x7

COMPUTE v4.8b, v7.8h, v8.8h, v9.8h, v10.8h
COMPUTE v5.8b, v11.8h, v12.8h, v13.8h, v14.8h
COMPUTE v6.8b, v15.8h, v16.8h, v17.8h, v18.8h
MERGE saddlp, v7.8h, v8.8h, v9.8h, v10.8h, v19.4s, v20.4s, v21.4s, v22.4s
MERGE saddlp, v11.8h, v12.8h, v13.8h, v14.8h, v23.4s, v24.4s, v25.4s, v26.4s
MERGE saddlp, v15.8h, v16.8h, v17.8h, v18.8h, v27.4s, v28.4s, v29.4s, v30.4s

subs x12, x12, #1
beq L3LoopZEnd

L3LoopZ:
    ld1 {v0.8b, v1.8b, v2.8b, v3.8b}, [x9], #32
    ld1 {v4.8b, v5.8b, v6.8b}, [x11], x7

    COMPUTE v4.8b, v7.8h, v8.8h, v9.8h, v10.8h
    COMPUTE v5.8b, v11.8h, v12.8h, v13.8h, v14.8h
    COMPUTE v6.8b, v15.8h, v16.8h, v17.8h, v18.8h
    MERGE sadalp, v7.8h, v8.8h, v9.8h, v10.8h, v19.4s, v20.4s, v21.4s, v22.4s
    MERGE sadalp, v11.8h, v12.8h, v13.8h, v14.8h, v23.4s, v24.4s, v25.4s, v26.4s
    MERGE sadalp, v15.8h, v16.8h, v17.8h, v18.8h, v27.4s, v28.4s, v29.4s, v30.4s

    subs x12, x12, #1
    bne L3LoopZ

L3LoopZEnd:
addp v19.4s, v19.4s, v20.4s
addp v21.4s, v21.4s, v22.4s
addp v23.4s, v23.4s, v24.4s
addp v25.4s, v25.4s, v26.4s
addp v27.4s, v27.4s, v28.4s
addp v29.4s, v29.4s, v30.4s

addp v19.4s, v19.4s, v21.4s
addp v23.4s, v23.4s, v25.4s
addp v27.4s, v27.4s, v29.4s
scvtf v19.4s, v19.4s
scvtf v20.4s, v23.4s
scvtf v21.4s, v27.4s
st1 {v19.4s, v20.4s, v21.4s}, [x8], x5
subs x10, x10, #1
bne L3LoopDz

L3End:
add x0, x0, #48  // 3 * 4 * sizeof(float)
add x1, x1, #24  // 3 * 8 * sizeof(int8)
sub x4, x4, #3
cmp x4, #3
bge L3

L2:
cmp x4, #2
blt L1
mov x8, x0
mov x9, x2
mov x10, x6

L2LoopDz:
mov x11, x1
mov x12, x3
ld1 {v0.8b, v1.8b, v2.8b, v3.8b}, [x9], #32
ld1 {v4.8b, v5.8b}, [x11], x7

COMPUTE v4.8b, v6.8h, v7.8h, v8.8h, v9.8h
COMPUTE v5.8b, v10.8h, v11.8h, v12.8h, v13.8h
MERGE saddlp, v6.8h, v7.8h, v8.8h, v9.8h, v14.4s, v15.4s, v16.4s, v17.4s
MERGE saddlp, v10.8h, v11.8h, v12.8h, v13.8h, v18.4s, v19.4s, v20.4s, v21.4s

subs x12, x12, #1
beq L2LoopZEnd

L2LoopZ:
    ld1 {v0.8b, v1.8b, v2.8b, v3.8b}, [x9], #32
    ld1 {v4.8b, v5.8b}, [x11], x7

    COMPUTE v4.8b, v6.8h, v7.8h, v8.8h, v9.8h
    COMPUTE v5.8b, v10.8h, v11.8h, v12.8h, v13.8h
    MERGE sadalp, v6.8h, v7.8h, v8.8h, v9.8h, v14.4s, v15.4s, v16.4s, v17.4s
    MERGE sadalp, v10.8h, v11.8h, v12.8h, v13.8h, v18.4s, v19.4s, v20.4s, v21.4s

    subs x12, x12, #1
    bne L2LoopZ

L2LoopZEnd:
addp v14.4s, v14.4s, v15.4s
addp v16.4s, v16.4s, v17.4s
addp v18.4s, v18.4s, v19.4s
addp v20.4s, v20.4s, v21.4s

addp v14.4s, v14.4s, v16.4s
addp v18.4s, v18.4s, v20.4s
scvtf v14.4s, v14.4s
scvtf v15.4s, v18.4s
st1 {v14.4s, v15.4s}, [x8], x5
subs x10, x10, #1
bne L2LoopDz

L2End:
add x0, x0, #32 // 2 * 4 * sizeof(float)
add x1, x1, #16 // 2 * 8 * sizeof(int8)
sub x4, x4, #2

L1:
cmp x4, #1
blt End
mov x8, x0
mov x9, x2
mov x10, x6

L1LoopDz:
mov x11, x1
mov x12, x3
ld1 {v0.8b, v1.8b, v2.8b, v3.8b}, [x9], #32
ld1 {v4.8b}, [x11], x7

COMPUTE v4.8b, v5.8h, v6.8h, v7.8h, v8.8h
MERGE saddlp, v5.8h, v6.8h, v7.8h, v8.8h, v9.4s, v10.4s, v11.4s, v12.4s

subs x12, x12, #1
beq L1LoopZEnd

L1LoopZ:
    ld1 {v0.8b, v1.8b, v2.8b, v3.8b}, [x9], #32
    ld1 {v4.8b}, [x11], x7

    COMPUTE v4.8b, v5.8h, v6.8h, v7.8h, v8.8h
    MERGE sadalp, v5.8h, v6.8h, v7.8h, v8.8h, v9.4s, v10.4s, v11.4s, v12.4s

    subs x12, x12, #1
    bne L1LoopZ

L1LoopZEnd:
addp v9.4s, v9.4s, v10.4s
addp v11.4s, v11.4s, v12.4s

addp v9.4s, v9.4s, v11.4s
scvtf v9.4s, v9.4s
st1 {v9.4s}, [x8], x5
subs x10, x10, #1
bne L1LoopDz

End:

sub sp, sp, #128
ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64
ret
#endif
