//
//  MNNMatrixSub.S
//  MNN
//
//  Created by MNN on 2019/02/12.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNMatrixSub
//void MNNMatrixSub(float* C, const float* A, const float* B, size_t widthC4, size_t cStride, size_t aStride, size_t bStride, size_t height)

//Auto: x0: C, x1:A, x2:B, x3:widthC4
//x4:cStride, x5:aStride, x6:bStride, x7:height
stp d14, d15, [sp, #-64]!
stp d12, d13, [sp, #16]
stp d10, d11, [sp, #32]
stp d8,  d9,  [sp, #48]

mov x12, #4 //sizeof(float)
mul x4, x12, x4
mul x5, x12, x5
mul x6, x12, x6

LoopY:
mov x8, x0
mov x9, x1
mov x10, x2

mov x11, x3

L16:
cmp x11, #16
blt L8

L16Loop:
ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x1], #64
ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x1], #64
ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x2], #64
ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [x2], #64
ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x1], #64
ld1 {v20.4s, v21.4s, v22.4s, v23.4s}, [x1], #64
ld1 {v24.4s, v25.4s, v26.4s, v27.4s}, [x2], #64
ld1 {v28.4s, v29.4s, v30.4s, v31.4s}, [x2], #64

fsub v0.4s, v0.4s, v8.4s
fsub v1.4s, v1.4s, v9.4s
fsub v2.4s, v2.4s, v10.4s
fsub v3.4s, v3.4s, v11.4s
fsub v4.4s, v4.4s, v12.4s
fsub v5.4s, v5.4s, v13.4s
fsub v6.4s, v6.4s, v14.4s
fsub v7.4s, v7.4s, v15.4s

sub x11, x11, #16

fsub v16.4s, v16.4s, v24.4s
fsub v17.4s, v17.4s, v25.4s
fsub v18.4s, v18.4s, v26.4s
fsub v19.4s, v19.4s, v27.4s
fsub v20.4s, v20.4s, v28.4s
fsub v21.4s, v21.4s, v29.4s
fsub v22.4s, v22.4s, v30.4s
fsub v23.4s, v23.4s, v31.4s

st1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x0], #64
st1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x0], #64
st1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x0], #64
st1 {v20.4s, v21.4s, v22.4s, v23.4s}, [x0], #64
cmp x11, #16
bge L16Loop

L8:
cmp x11, #8
blt L4

L8Loop:
ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x1], #64
ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x1], #64
ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x2], #64
ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [x2], #64

fsub v0.4s, v0.4s, v8.4s
fsub v1.4s, v1.4s, v9.4s
fsub v2.4s, v2.4s, v10.4s
fsub v3.4s, v3.4s, v11.4s
fsub v4.4s, v4.4s, v12.4s
fsub v5.4s, v5.4s, v13.4s
fsub v6.4s, v6.4s, v14.4s
fsub v7.4s, v7.4s, v15.4s
sub x11, x11, #8

st1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x0], #64
st1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x0], #64
cmp x11, #8
bge L8Loop

L4:
cmp x11, #4
blt L1
sub x11, x11, #4
ld1 {v0.4s, v1.4s}, [x1], #32
ld1 {v2.4s, v3.4s}, [x2], #32

fsub v0.4s, v0.4s, v2.4s
ld1 {v16.4s, v17.4s}, [x1], #32
fsub v1.4s, v1.4s, v3.4s

cmp x11, #4
blt L4LoopEnd

L4Loop:
ld1 {v18.4s, v19.4s}, [x2], #32
st1 {v0.4s, v1.4s}, [x0], #32
fsub v16.4s, v16.4s, v18.4s
fsub v17.4s, v17.4s, v19.4s

ld1 {v0.4s, v1.4s}, [x1], #32
st1 {v16.4s, v17.4s}, [x0], #32
ld1 {v2.4s, v3.4s}, [x2], #32
fsub v0.4s, v0.4s, v2.4s
ld1 {v16.4s, v17.4s}, [x1], #32
fsub v1.4s, v1.4s, v3.4s

sub x11, x11, #4
cmp x11, #4
bge L4Loop

L4LoopEnd:
ld1 {v18.4s, v19.4s}, [x2], #32
st1 {v0.4s, v1.4s}, [x0], #32
fsub v16.4s, v16.4s, v18.4s
fsub v17.4s, v17.4s, v19.4s
st1 {v16.4s, v17.4s}, [x0], #32

L1:
cmp x11, #0
beq EndLine

L1Loop:
ld1 {v0.4s}, [x1], #16
ld1 {v1.4s}, [x2], #16
fsub v0.4s, v0.4s, v1.4s
st1 {v0.4s}, [x0], #16
subs x11, x11, #1
bne L1Loop

EndLine:
add x0, x8, x4
add x1, x9, x5
add x2, x10, x6

subs x7, x7, #1
bne LoopY

End:
ldp d8,  d9,  [sp, #48]
ldp d10, d11, [sp, #32]
ldp d12, d13, [sp, #16]
ldp d14, d15, [sp], #64
ret

#endif
