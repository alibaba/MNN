//
//  MNNPackC8_BF16.S
//  MNN
//
//  Created by MNN on 2021/02/20.
//  Copyright © 2018-2021 Alibaba Group Holding Limited.
//
#ifdef __aarch64__

#include "MNNAsmGlobal.h"


.text
.align 5
asm_function MNNPackC8_BF16
// treate float pointer as int16_t*
//void MNNPackC8_BF16(float* dest, const float* source, size_t l, size_t h);
// h, l ->  hC8, l, 8
// Auto: x0:dest, x1:source, x2: l, x3: h
// x4: lC8, x5:hC8, x6: sourceStride, x7: destStride

lsr x4, x2, #3
lsr x5, x3, #3
mov x12, #2 // sizeof(int16_t)
mov x13, #16 // 8 * sizeof(int16_t)
mov x15, #4
mul x6, x15, x2
mul x7, x13, x2
mov x12, #32 // 8 * sizeof(float)
mul x15, x12, x2

.macro transpose_4x4 x0, x1, x2, x3, x5, x6
    trn1 \x5\().4s,  \x0\().4s, \x1\().4s
    trn2 \x1\().4s,  \x0\().4s, \x1\().4s
    trn1 \x6\().4s,  \x2\().4s, \x3\().4s
    trn2 \x3\().4s,  \x2\().4s, \x3\().4s
    trn1 \x0\().2d,  \x5\().2d, \x6\().2d
    trn2 \x2\().2d,  \x5\().2d, \x6\().2d
    trn1 \x6\().2d,  \x1\().2d, \x3\().2d
    trn2 \x3\().2d,  \x1\().2d, \x3\().2d
    mov \x1\().16b, \x6\().16b
.endm

LoopH:
mov x8, x0
mov x9, x1
mov x12, x4

LoopL:
mov x10, x9
ld1 {v16.4s, v17.4s}, [x9], x6
ld1 {v18.4s, v19.4s}, [x9], x6
ld1 {v20.4s, v21.4s}, [x9], x6
ld1 {v22.4s, v23.4s}, [x9], x6

ld1 {v24.4s, v25.4s}, [x9], x6
ld1 {v26.4s, v27.4s}, [x9], x6
ld1 {v28.4s, v29.4s}, [x9], x6
ld1 {v30.4s, v31.4s}, [x9], x6


transpose_4x4 v16, v18, v20, v22, v0, v1
transpose_4x4 v17, v19, v21, v23, v2, v3
transpose_4x4 v24, v26, v28, v30, v4, v5
transpose_4x4 v25, v27, v29, v31, v6, v7


shrn v16.4h, v16.4s, #16
shrn v17.4h, v17.4s, #16
shrn v18.4h, v18.4s, #16
shrn v19.4h, v19.4s, #16
shrn v20.4h, v20.4s, #16
shrn v21.4h, v21.4s, #16
shrn v22.4h, v22.4s, #16
shrn v23.4h, v23.4s, #16
shrn v24.4h, v24.4s, #16
shrn v25.4h, v25.4s, #16
shrn v26.4h, v26.4s, #16
shrn v27.4h, v27.4s, #16
shrn v28.4h, v28.4s, #16
shrn v29.4h, v29.4s, #16
shrn v30.4h, v30.4s, #16
shrn v31.4h, v31.4s, #16


stp d16, d24, [x8], #16
stp d18, d26, [x8], #16
stp d20, d28, [x8], #16
stp d22, d30, [x8], #16

stp d17, d25, [x8], #16
stp d19, d27, [x8], #16
stp d21, d29, [x8], #16
stp d23, d31, [x8], #16

add x9, x10, #32  // 8 * sizeof(float)

subs x12, x12, #1
bne LoopL


subs x5, x5, #1
add x0, x0, x7
add x1, x1, x15
bne LoopH


ret

#endif
