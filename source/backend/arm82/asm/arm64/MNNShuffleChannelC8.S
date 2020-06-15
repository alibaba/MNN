//
//  MNNShuffleChannelC8.S
//  MNN
//
//  Created by MNN on 2020/01/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__
#include "MNNAsmGlobal.h"

.text
.align 5
asm_function MNNShuffleChannelC8
// void MNNShuffleChannelC8(FLOAT16* dst, const FLOAT16* src, size_t size, size_t halfFlag)
// Auto:
//    x0:dst, x1:src, x2:size, x3:halfFlag

cbz x3, LOOP_SIZE

mov x4, #128
LOOP_SIZE_4:
ldr q0, [x1]
ldr q1, [x1, #16]
ldr q2, [x1, #32]
ldr q3, [x1, #48]
subs x2, x2, #1
add x1, x1, x4
st4 {v0.8h, v1.8h, v2.8h, v3.8h}, [x0]
add x0, x0, #64
bne LOOP_SIZE_4

b REAL_END

LOOP_SIZE:
    ldr q0, [x1], #16
    ldr q1, [x1], #16
    ldr q2, [x1], #16
    ldr q3, [x1], #16
    ldr q4, [x1], #16
    ldr q5, [x1], #16
    ldr q6, [x1], #16
    ldr q7, [x1], #16
    zip1 v16.8h, v0.8h, v4.8h
    zip1 v17.8h, v2.8h, v6.8h
    zip1 v18.8h, v1.8h, v5.8h
    zip1 v19.8h, v3.8h, v7.8h

    zip1 v24.8h, v16.8h, v17.8h
    zip1 v25.8h, v18.8h, v19.8h
    zip2 v26.8h, v16.8h, v17.8h
    zip2 v27.8h, v18.8h, v19.8h

    zip1 v28.8h, v24.8h, v25.8h
    zip2 v29.8h, v24.8h, v25.8h
    zip1 v30.8h, v26.8h, v27.8h
    zip2 v31.8h, v26.8h, v27.8h
    
    st1 {v28.16b, v29.16b, v30.16b, v31.16b}, [x0], #64
    // -----
    zip2 v20.8h, v0.8h, v4.8h
    zip2 v21.8h, v2.8h, v6.8h
    zip2 v22.8h, v1.8h, v5.8h
    zip2 v23.8h, v3.8h, v7.8h

    zip1 v24.8h, v20.8h, v21.8h
    zip1 v25.8h, v22.8h, v23.8h
    zip2 v26.8h, v20.8h, v21.8h
    zip2 v27.8h, v22.8h, v23.8h

    subs x2, x2, #1
    zip1 v28.8h, v24.8h, v25.8h
    zip2 v29.8h, v24.8h, v25.8h
    zip1 v30.8h, v26.8h, v27.8h
    zip2 v31.8h, v26.8h, v27.8h
    st1 {v28.16b, v29.16b, v30.16b, v31.16b}, [x0], #64
    bne LOOP_SIZE


REAL_END:
ret
#endif