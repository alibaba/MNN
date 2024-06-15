//
//  MNNPackedMatMulFP16.S
//  MNN
//
//  Created by MNN on 2020/06/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5
// 8 * 24 MatMul
asm_function MNNPackedMatMulFP16
//void MNNPackedMatMulFP16(FLOAT16* C, const FLOAT16* A, const FLOAT16* B, const size_t* parameter, const FLOAT16* postParameters, const FLOAT16* bias);
// x0: C, x1:A, x2:B, x3:parameter, x4: postParameters, x5:bias
stp d14, d15, [sp, #-64]!
stp d12, d13, [sp, #16]
stp d10, d11, [sp, #32]
stp d8,  d9,  [sp, #48]

//ldr x8, [x3, #0] // deprecated
ldr x9, [x3, #8] // l
ldr x10, [x3, #16] // h

ldr x13, [x3, #24] // cStride
ldr x7, [x3, #40] // bExtraStride

// v0, v1, v2: A
// v3, v4: B
// v8 - v31: C
add x10, x10, #7
lsr x10, x10, #3

cbz x4, Start
ld1 {v5.8h}, [x4]
fcvtn v5.4h, v5.4s
dup v6.8h, v5.h[2] // Min Value
dup v7.8h, v5.h[3] // Max Value

Start:

cmp x10, #2
blt LH4

LH8:
sub x14, x13, #128
LoopH:
    mov x15, x1
    subs x12, x9, #1
    ld1 {v3.8h, v4.8h}, [x2], #32
    ld1 {v0.4h, v1.4h, v2.4h}, [x15], #24

    fmul v8.8h, v3.8h, v0.h[0]
    fmul v9.8h, v3.8h, v0.h[1]
    fmul v10.8h, v3.8h, v0.h[2]
    fmul v11.8h, v3.8h, v0.h[3]
    fmul v12.8h, v3.8h, v1.h[0]
    fmul v13.8h, v3.8h, v1.h[1]
    fmul v14.8h, v3.8h, v1.h[2]
    fmul v15.8h, v3.8h, v1.h[3]
    fmul v16.8h, v3.8h, v2.h[0]
    fmul v17.8h, v3.8h, v2.h[1]
    fmul v18.8h, v3.8h, v2.h[2]
    fmul v19.8h, v3.8h, v2.h[3]

    fmul v20.8h, v4.8h, v0.h[0]
    fmul v21.8h, v4.8h, v0.h[1]
    fmul v22.8h, v4.8h, v0.h[2]
    fmul v23.8h, v4.8h, v0.h[3]

    fmul v24.8h, v4.8h, v1.h[0]
    fmul v25.8h, v4.8h, v1.h[1]
    fmul v26.8h, v4.8h, v1.h[2]
    fmul v27.8h, v4.8h, v1.h[3]

    fmul v28.8h, v4.8h, v2.h[0]
    fmul v29.8h, v4.8h, v2.h[1]
    fmul v30.8h, v4.8h, v2.h[2]
    fmul v31.8h, v4.8h, v2.h[3]

    beq LoopLEnd

    cmp x12, #2
    blt L1
    LoopL2:
        ld1 {v3.8h, v4.8h}, [x2], #32
        ld1 {v0.4h, v1.4h, v2.4h}, [x15], #24

        fmla v8.8h, v3.8h, v0.h[0]
        fmla v9.8h, v3.8h, v0.h[1]
        fmla v10.8h, v3.8h, v0.h[2]
        fmla v11.8h, v3.8h, v0.h[3]
        fmla v12.8h, v3.8h, v1.h[0]
        fmla v13.8h, v3.8h, v1.h[1]
        fmla v14.8h, v3.8h, v1.h[2]
        fmla v15.8h, v3.8h, v1.h[3]
        fmla v16.8h, v3.8h, v2.h[0]
        fmla v17.8h, v3.8h, v2.h[1]
        fmla v18.8h, v3.8h, v2.h[2]
        fmla v19.8h, v3.8h, v2.h[3]

        fmla v20.8h, v4.8h, v0.h[0]
        fmla v21.8h, v4.8h, v0.h[1]
        fmla v22.8h, v4.8h, v0.h[2]
        fmla v23.8h, v4.8h, v0.h[3]

        fmla v24.8h, v4.8h, v1.h[0]
        fmla v25.8h, v4.8h, v1.h[1]
        fmla v26.8h, v4.8h, v1.h[2]
        fmla v27.8h, v4.8h, v1.h[3]

        fmla v28.8h, v4.8h, v2.h[0]
        fmla v29.8h, v4.8h, v2.h[1]
        fmla v30.8h, v4.8h, v2.h[2]
        fmla v31.8h, v4.8h, v2.h[3]

        ld1 {v3.8h, v4.8h}, [x2], #32
        ld1 {v0.4h, v1.4h, v2.4h}, [x15], #24

        fmla v8.8h, v3.8h, v0.h[0]
        fmla v9.8h, v3.8h, v0.h[1]
        fmla v10.8h, v3.8h, v0.h[2]
        fmla v11.8h, v3.8h, v0.h[3]
        fmla v12.8h, v3.8h, v1.h[0]
        fmla v13.8h, v3.8h, v1.h[1]
        fmla v14.8h, v3.8h, v1.h[2]
        fmla v15.8h, v3.8h, v1.h[3]
        fmla v16.8h, v3.8h, v2.h[0]
        fmla v17.8h, v3.8h, v2.h[1]
        fmla v18.8h, v3.8h, v2.h[2]
        fmla v19.8h, v3.8h, v2.h[3]

        fmla v20.8h, v4.8h, v0.h[0]
        fmla v21.8h, v4.8h, v0.h[1]
        fmla v22.8h, v4.8h, v0.h[2]
        fmla v23.8h, v4.8h, v0.h[3]

        fmla v24.8h, v4.8h, v1.h[0]
        fmla v25.8h, v4.8h, v1.h[1]
        fmla v26.8h, v4.8h, v1.h[2]
        fmla v27.8h, v4.8h, v1.h[3]

        fmla v28.8h, v4.8h, v2.h[0]
        fmla v29.8h, v4.8h, v2.h[1]
        fmla v30.8h, v4.8h, v2.h[2]
        fmla v31.8h, v4.8h, v2.h[3]
        sub x12, x12, #2
        cmp x12, #2
        bge LoopL2

    cbz x12, LoopLEnd

    L1:
        ld1 {v3.8h, v4.8h}, [x2], #32
        ld1 {v0.4h, v1.4h, v2.4h}, [x15], #24

        fmla v8.8h, v3.8h, v0.h[0]
        fmla v9.8h, v3.8h, v0.h[1]
        fmla v10.8h, v3.8h, v0.h[2]
        fmla v11.8h, v3.8h, v0.h[3]
        fmla v12.8h, v3.8h, v1.h[0]
        fmla v13.8h, v3.8h, v1.h[1]
        fmla v14.8h, v3.8h, v1.h[2]
        fmla v15.8h, v3.8h, v1.h[3]
        fmla v16.8h, v3.8h, v2.h[0]
        fmla v17.8h, v3.8h, v2.h[1]
        fmla v18.8h, v3.8h, v2.h[2]
        fmla v19.8h, v3.8h, v2.h[3]

        fmla v20.8h, v4.8h, v0.h[0]
        fmla v21.8h, v4.8h, v0.h[1]
        fmla v22.8h, v4.8h, v0.h[2]
        fmla v23.8h, v4.8h, v0.h[3]

        fmla v24.8h, v4.8h, v1.h[0]
        fmla v25.8h, v4.8h, v1.h[1]
        fmla v26.8h, v4.8h, v1.h[2]
        fmla v27.8h, v4.8h, v1.h[3]

        fmla v28.8h, v4.8h, v2.h[0]
        fmla v29.8h, v4.8h, v2.h[1]
        fmla v30.8h, v4.8h, v2.h[2]
        fmla v31.8h, v4.8h, v2.h[3]

    LoopLEnd:

    add x2, x2, x7
    sub x10, x10, #2
    cmp x10, #2

    cbz x4, StoreLH8

    AddBiasLH8:
    ld1 {v0.8h, v1.8h}, [x5], #32

    fmla v8.8h, v0.8h, v5.h[1]
    fmla v9.8h, v0.8h, v5.h[1]
    fmla v10.8h, v0.8h, v5.h[1]
    fmla v11.8h, v0.8h, v5.h[1]

    fmla v12.8h, v0.8h, v5.h[1]
    fmla v13.8h, v0.8h, v5.h[1]
    fmla v14.8h, v0.8h, v5.h[1]
    fmla v15.8h, v0.8h, v5.h[1]

    fmla v16.8h, v0.8h, v5.h[1]
    fmla v17.8h, v0.8h, v5.h[1]
    fmla v18.8h, v0.8h, v5.h[1]
    fmla v19.8h, v0.8h, v5.h[1]

    fmla v20.8h, v1.8h, v5.h[1]
    fmla v21.8h, v1.8h, v5.h[1]
    fmla v22.8h, v1.8h, v5.h[1]
    fmla v23.8h, v1.8h, v5.h[1]

    fmla v24.8h, v1.8h, v5.h[1]
    fmla v25.8h, v1.8h, v5.h[1]
    fmla v26.8h, v1.8h, v5.h[1]
    fmla v27.8h, v1.8h, v5.h[1]

    fmla v28.8h, v1.8h, v5.h[1]
    fmla v29.8h, v1.8h, v5.h[1]
    fmla v30.8h, v1.8h, v5.h[1]
    fmla v31.8h, v1.8h, v5.h[1]

    PostTreatLH8:
    fmax v8.8h, v8.8h, v6.8h
    fmax v9.8h, v9.8h, v6.8h
    fmax v10.8h, v10.8h, v6.8h
    fmax v11.8h, v11.8h, v6.8h
    fmax v12.8h, v12.8h, v6.8h
    fmax v13.8h, v13.8h, v6.8h
    fmax v14.8h, v14.8h, v6.8h
    fmax v15.8h, v15.8h, v6.8h
    fmax v16.8h, v16.8h, v6.8h
    fmax v17.8h, v17.8h, v6.8h
    fmax v18.8h, v18.8h, v6.8h
    fmax v19.8h, v19.8h, v6.8h
    fmax v20.8h, v20.8h, v6.8h
    fmax v21.8h, v21.8h, v6.8h
    fmax v22.8h, v22.8h, v6.8h
    fmax v23.8h, v23.8h, v6.8h
    fmax v24.8h, v24.8h, v6.8h
    fmax v25.8h, v25.8h, v6.8h
    fmax v26.8h, v26.8h, v6.8h
    fmax v27.8h, v27.8h, v6.8h
    fmax v28.8h, v28.8h, v6.8h
    fmax v29.8h, v29.8h, v6.8h
    fmax v30.8h, v30.8h, v6.8h
    fmax v31.8h, v31.8h, v6.8h

    fmin v8.8h,  v8.8h,  v7.8h
    fmin v9.8h,  v9.8h,  v7.8h
    fmin v10.8h, v10.8h, v7.8h
    fmin v11.8h, v11.8h, v7.8h
    fmin v12.8h, v12.8h, v7.8h
    fmin v13.8h, v13.8h, v7.8h
    fmin v14.8h, v14.8h, v7.8h
    fmin v15.8h, v15.8h, v7.8h
    fmin v16.8h, v16.8h, v7.8h
    fmin v17.8h, v17.8h, v7.8h
    fmin v18.8h, v18.8h, v7.8h
    fmin v19.8h, v19.8h, v7.8h
    fmin v20.8h, v20.8h, v7.8h
    fmin v21.8h, v21.8h, v7.8h
    fmin v22.8h, v22.8h, v7.8h
    fmin v23.8h, v23.8h, v7.8h
    fmin v24.8h, v24.8h, v7.8h
    fmin v25.8h, v25.8h, v7.8h
    fmin v26.8h, v26.8h, v7.8h
    fmin v27.8h, v27.8h, v7.8h
    fmin v28.8h, v28.8h, v7.8h
    fmin v29.8h, v29.8h, v7.8h
    fmin v30.8h, v30.8h, v7.8h
    fmin v31.8h, v31.8h, v7.8h

    StoreLH8:

    st1 {v8.8h, v9.8h, v10.8h, v11.8h}, [x0], #64
    st1 {v12.8h, v13.8h, v14.8h, v15.8h}, [x0], #64
    st1 {v16.8h, v17.8h, v18.8h, v19.8h}, [x0], x14

    st1 {v20.8h, v21.8h, v22.8h, v23.8h}, [x0], #64
    st1 {v24.8h, v25.8h, v26.8h, v27.8h}, [x0], #64
    st1 {v28.8h, v29.8h, v30.8h, v31.8h}, [x0], x14

    bge LoopH

LH4:
cbz x10, End
LoopHRemain:
    mov x15, x1
    subs x12, x9, #1
    ld1 {v3.8h}, [x2]
    ld1 {v0.4h, v1.4h, v2.4h}, [x15], #24

    fmul v8.8h, v3.8h, v0.h[0]
    fmul v9.8h, v3.8h, v0.h[1]
    add x2, x2, #32
    fmul v10.8h, v3.8h, v0.h[2]
    fmul v11.8h, v3.8h, v0.h[3]
    fmul v12.8h, v3.8h, v1.h[0]
    fmul v13.8h, v3.8h, v1.h[1]
    fmul v14.8h, v3.8h, v1.h[2]
    fmul v15.8h, v3.8h, v1.h[3]
    fmul v16.8h, v3.8h, v2.h[0]
    fmul v17.8h, v3.8h, v2.h[1]
    fmul v18.8h, v3.8h, v2.h[2]
    fmul v19.8h, v3.8h, v2.h[3]

    beq LoopLREnd

    LoopLR:
        ld1 {v3.8h}, [x2]
        ld1 {v0.4h, v1.4h, v2.4h}, [x15], #24

        fmla v8.8h, v3.8h, v0.h[0]
        fmla v9.8h, v3.8h, v0.h[1]
        fmla v10.8h, v3.8h, v0.h[2]
        fmla v11.8h, v3.8h, v0.h[3]
        add x2, x2, #32
        fmla v12.8h, v3.8h, v1.h[0]
        fmla v13.8h, v3.8h, v1.h[1]
        fmla v14.8h, v3.8h, v1.h[2]
        fmla v15.8h, v3.8h, v1.h[3]
        fmla v16.8h, v3.8h, v2.h[0]
        fmla v17.8h, v3.8h, v2.h[1]
        fmla v18.8h, v3.8h, v2.h[2]
        fmla v19.8h, v3.8h, v2.h[3]

        subs x12, x12, #1
        bne LoopLR
    LoopLREnd:

    cbz x4, StoreLH4
    AddBiasLH4:
    ld1 {v0.8h}, [x5], #16

    fmla v8.8h, v0.8h, v5.h[1]
    fmla v9.8h, v0.8h, v5.h[1]
    fmla v10.8h, v0.8h, v5.h[1]
    fmla v11.8h, v0.8h, v5.h[1]

    fmla v12.8h, v0.8h, v5.h[1]
    fmla v13.8h, v0.8h, v5.h[1]
    fmla v14.8h, v0.8h, v5.h[1]
    fmla v15.8h, v0.8h, v5.h[1]

    fmla v16.8h, v0.8h, v5.h[1]
    fmla v17.8h, v0.8h, v5.h[1]
    fmla v18.8h, v0.8h, v5.h[1]
    fmla v19.8h, v0.8h, v5.h[1]

    PostTreatLH4:
    fmax v8.8h, v8.8h, v6.8h
    fmax v9.8h, v9.8h, v6.8h
    fmax v10.8h, v10.8h, v6.8h
    fmax v11.8h, v11.8h, v6.8h
    fmax v12.8h, v12.8h, v6.8h
    fmax v13.8h, v13.8h, v6.8h
    fmax v14.8h, v14.8h, v6.8h
    fmax v15.8h, v15.8h, v6.8h
    fmax v16.8h, v16.8h, v6.8h
    fmax v17.8h, v17.8h, v6.8h
    fmax v18.8h, v18.8h, v6.8h
    fmax v19.8h, v19.8h, v6.8h

    fmin v8.8h,  v8.8h,  v7.8h
    fmin v9.8h,  v9.8h,  v7.8h
    fmin v10.8h, v10.8h, v7.8h
    fmin v11.8h, v11.8h, v7.8h
    fmin v12.8h, v12.8h, v7.8h
    fmin v13.8h, v13.8h, v7.8h
    fmin v14.8h, v14.8h, v7.8h
    fmin v15.8h, v15.8h, v7.8h
    fmin v16.8h, v16.8h, v7.8h
    fmin v17.8h, v17.8h, v7.8h
    fmin v18.8h, v18.8h, v7.8h
    fmin v19.8h, v19.8h, v7.8h

    StoreLH4:

    st1 {v8.8h, v9.8h, v10.8h, v11.8h}, [x0], #64
    st1 {v12.8h, v13.8h, v14.8h, v15.8h}, [x0], #64
    st1 {v16.8h, v17.8h, v18.8h, v19.8h}, [x0]
    sub x10, x10, #1


End:
ldp d8,  d9,  [sp, #48]
ldp d10, d11, [sp, #32]
ldp d12, d13, [sp, #16]
ldp d14, d15, [sp], #64

ret

#endif
