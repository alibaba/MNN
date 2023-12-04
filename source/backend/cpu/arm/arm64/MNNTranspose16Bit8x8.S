//
//  MNNTranspose16Bit8x8.S
//  MNN
//
//  Created by MNN on 2023/11/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

.macro TRANSPOSE_8x8 s0, s1, s2, s3, s4, s5, s6, s7, d0, d1, d2, d3, d4, d5, d6, d7, t0, t1, t2, t3, t4, t5, t6, t7
    zip1 \t0\().8h, \s0\().8h, \s1\().8h
    zip2 \t1\().8h, \s0\().8h, \s1\().8h
    zip1 \t2\().8h, \s2\().8h, \s3\().8h
    zip2 \t3\().8h, \s2\().8h, \s3\().8h
    zip1 \t4\().8h, \s4\().8h, \s5\().8h
    zip2 \t5\().8h, \s4\().8h, \s5\().8h
    zip1 \t6\().8h, \s6\().8h, \s7\().8h
    zip2 \t7\().8h, \s6\().8h, \s7\().8h
    zip1 \s0\().4s, \t0\().4s, \t2\().4s
    zip2 \s1\().4s, \t0\().4s, \t2\().4s
    zip1 \s2\().4s, \t1\().4s, \t3\().4s
    zip2 \s3\().4s, \t1\().4s, \t3\().4s
    zip1 \s4\().4s, \t4\().4s, \t6\().4s
    zip2 \s5\().4s, \t4\().4s, \t6\().4s
    zip1 \s6\().4s, \t5\().4s, \t7\().4s
    zip2 \s7\().4s, \t5\().4s, \t7\().4s
    zip1 \d0\().2d, \s0\().2d, \s4\().2d
    zip2 \d1\().2d, \s0\().2d, \s4\().2d
    zip1 \d2\().2d, \s1\().2d, \s5\().2d
    zip2 \d3\().2d, \s1\().2d, \s5\().2d
    zip1 \d4\().2d, \s2\().2d, \s6\().2d
    zip2 \d5\().2d, \s2\().2d, \s6\().2d
    zip1 \d6\().2d, \s3\().2d, \s7\().2d
    zip2 \d7\().2d, \s3\().2d, \s7\().2d
.endm

asm_function MNNTranspose16Bit8x8
//void MNNTranspose16Bit8x8(int16_t* dstO, const int16_t* srcO, int* dim)
//Auto: x0: dstO, x1:srcO, x2: dim
stp d14, d15, [sp, #(-16 * 4)]!
stp d12, d13, [sp, #(16 * 1)]
stp d10, d11, [sp, #(16 * 2)]
stp d8,  d9,  [sp, #(16 * 3)]

mov x4, #0
mov x5, #0
mov x6, #0
mov x7, #0
ldr w4, [x2, #0]
ldr w5, [x2, #4]
ldr w6, [x2, #8]
ldr w7, [x2, #12]

// x4, x5 -> wC8, hC8
lsr x4, x4, #3
lsr x5, x5, #3

// x6, x7 -> srcStride * sizeof(half), dstStride * sizeof(half)
lsl x6, x6, #1
lsl x7, x7, #1

LoopY:
    mov x2, x4
    mov x8, x0
    mov x9, x1
    LoopX:
        ld1 {v0.8h}, [x1], x6
        ld1 {v1.8h}, [x1], x6
        ld1 {v2.8h}, [x1], x6
        ld1 {v3.8h}, [x1], x6
        ld1 {v4.8h}, [x1], x6
        ld1 {v5.8h}, [x1], x6
        ld1 {v6.8h}, [x1], x6
        ld1 {v7.8h}, [x1], x6

        TRANSPOSE_8x8  v0,  v1,  v2,  v3,  v4,  v5,  v6,  v7, \
                       v8,  v9, v10, v11, v12, v13, v14, v15, \
                      v16, v17, v18, v19, v20, v21, v22, v23

        mov x12, x0

        st1 {v8.8h}, [x12], x7
        st1 {v9.8h}, [x12], x7
        st1 {v10.8h}, [x12], x7
        st1 {v11.8h}, [x12], x7
        st1 {v12.8h}, [x12], x7
        st1 {v13.8h}, [x12], x7
        st1 {v14.8h}, [x12], x7
        st1 {v15.8h}, [x12], x7

        add x0, x0, #16 // 8 * sizeof(half)

        subs x2, x2, #1
        bne LoopX


    lsl x12, x7, #3 // 8 * dstStride
    subs x5, x5, #1
    add x1, x9, #16 // 8 * sizeof(half)
    add x0, x8, x12
    bne LoopY

End:
ldp d8,  d9,  [sp, #(16 * 3)]
ldp d10, d11, [sp, #(16 * 2)]
ldp d12, d13, [sp, #(16 * 1)]
ldp d14, d15, [sp], #(16 * 4)
ret

#endif