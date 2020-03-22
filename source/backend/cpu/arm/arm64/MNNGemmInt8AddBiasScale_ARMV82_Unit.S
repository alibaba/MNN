//
//  MNNGemmInt8AddBiasScale_ARMV82_Unit.S
//  MNN
//
//  Created by MNN on 2019/12/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#if defined(__aarch64__) && defined(ENABLE_ARMV82)

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNGemmInt8AddBiasScale_ARMV82_Unit

// void MNNGemmInt8AddBiasScale_ARMV82_Unit(int8_t* dst, const int8_t* src, 
//          const int8_t* weight, const int32_t* bias, const float* scale, 
//          size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad, size_t relu)

//Auto: x0: dst*, x1: src*, x2:weight*, x3: bias*
// x4: scale*, x5: src_depth_quad, x6: dst_step, 
// x7: dst_depth_quad, x8: relu, x9: realDstCount

ldr x8, [sp, #0]
ldr x9, [sp, #8]

sub sp, sp, #128
st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64

cmp x9, #8
bgt TILE_16
cmp x9, #4
bgt TILE_8
b TILE_4

TILE_16:
//sub x6, x6, #48
L4LoopDz_TILE_16:
    ld1 {v7.4s}, [x3], #16 // bias
    mov x10, x1
    mov x11, x5

    mov v8.16b, v7.16b
    mov v9.16b, v7.16b
    mov v10.16b, v7.16b
    mov v11.16b, v7.16b
    mov v12.16b, v7.16b
    mov v13.16b, v7.16b
    mov v14.16b, v7.16b
    mov v15.16b, v7.16b
    mov v16.16b, v7.16b
    mov v17.16b, v7.16b
    mov v18.16b, v7.16b
    mov v19.16b, v7.16b
    mov v20.16b, v7.16b
    mov v21.16b, v7.16b
    mov v22.16b, v7.16b
    mov v23.16b, v7.16b

    L4LoopSz_TILE_16:
        ld1 {v6.16b}, [x2], #16 // weight
        ld1 {v0.16b}, [x1], #16 // src

        sdot v8.4s, v6.16b, v0.4b[0]
        sdot v9.4s, v6.16b, v0.4b[1]
        ld1 {v1.16b}, [x1], #16
        sdot v10.4s, v6.16b, v0.4b[2]
        sdot v11.4s, v6.16b, v0.4b[3]

        sdot v12.4s, v6.16b, v1.4b[0]
        sdot v13.4s, v6.16b, v1.4b[1]
        ld1 {v2.16b}, [x1], #16
        sdot v14.4s, v6.16b, v1.4b[2]
        sdot v15.4s, v6.16b, v1.4b[3]

        sdot v16.4s, v6.16b, v2.4b[0]
        sdot v17.4s, v6.16b, v2.4b[1]
        ld1 {v3.16b}, [x1], #16
        sdot v18.4s, v6.16b, v2.4b[2]
        sdot v19.4s, v6.16b, v2.4b[3]
        subs x11, x11, #1
        sdot v20.4s, v6.16b, v3.4b[0]
        sdot v21.4s, v6.16b, v3.4b[1]
        sdot v22.4s, v6.16b, v3.4b[2]
        sdot v23.4s, v6.16b, v3.4b[3]
        bne L4LoopSz_TILE_16

    L4LoopSzEnd_TILE_16:

    ld1 {v1.4s}, [x4], #16 // scale

    scvtf v8.4s, v8.4s
    scvtf v9.4s, v9.4s
    scvtf v10.4s, v10.4s
    scvtf v11.4s, v11.4s
    scvtf v12.4s, v12.4s
    scvtf v13.4s, v13.4s
    scvtf v14.4s, v14.4s
    scvtf v15.4s, v15.4s
    scvtf v16.4s, v16.4s
    scvtf v17.4s, v17.4s
    scvtf v18.4s, v18.4s
    scvtf v19.4s, v19.4s
    scvtf v20.4s, v20.4s
    scvtf v21.4s, v21.4s
    scvtf v22.4s, v22.4s
    scvtf v23.4s, v23.4s
    
    fmul v8.4s, v8.4s, v1.4s
    fmul v9.4s, v9.4s, v1.4s
    fmul v10.4s, v10.4s, v1.4s
    fmul v11.4s, v11.4s, v1.4s
    fmul v12.4s, v12.4s, v1.4s
    fmul v13.4s, v13.4s, v1.4s
    fmul v14.4s, v14.4s, v1.4s
    fmul v15.4s, v15.4s, v1.4s
    fmul v16.4s, v16.4s, v1.4s
    fmul v17.4s, v17.4s, v1.4s
    fmul v18.4s, v18.4s, v1.4s
    fmul v19.4s, v19.4s, v1.4s
    fmul v20.4s, v20.4s, v1.4s
    fmul v21.4s, v21.4s, v1.4s
    fmul v22.4s, v22.4s, v1.4s
    fmul v23.4s, v23.4s, v1.4s


    fcvtas v8.4s, v8.4s
    fcvtas v9.4s, v9.4s
    fcvtas v10.4s, v10.4s
    fcvtas v11.4s, v11.4s
    fcvtas v12.4s, v12.4s
    fcvtas v13.4s, v13.4s
    fcvtas v14.4s, v14.4s
    fcvtas v15.4s, v15.4s
    fcvtas v16.4s, v16.4s
    fcvtas v17.4s, v17.4s
    fcvtas v18.4s, v18.4s
    fcvtas v19.4s, v19.4s
    fcvtas v20.4s, v20.4s
    fcvtas v21.4s, v21.4s
    fcvtas v22.4s, v22.4s
    fcvtas v23.4s, v23.4s
    

    sqxtn v0.4h,  v8.4s
    sqxtn2 v0.8h, v9.4s
    sqxtn v1.4h,  v10.4s
    sqxtn2 v1.8h, v11.4s
    sqxtn v2.4h,  v12.4s
    sqxtn2 v2.8h, v13.4s
    sqxtn v3.4h,  v14.4s
    sqxtn2 v3.8h, v15.4s
    sqxtn v4.4h,  v16.4s
    sqxtn2 v4.8h, v17.4s
    sqxtn v5.4h,  v18.4s
    sqxtn2 v5.8h, v19.4s
    sqxtn v6.4h,  v20.4s
    sqxtn2 v6.8h, v21.4s
    sqxtn v7.4h,  v22.4s
    sqxtn2 v7.8h, v23.4s
    

    sqxtn v16.8b, v0.8h
    sqxtn2 v16.16b, v1.8h
    sqxtn v17.8b, v2.8h
    sqxtn2 v17.16b, v3.8h
    sqxtn v18.8b, v4.8h
    sqxtn2 v18.16b, v5.8h
    sqxtn v19.8b, v6.8h
    sqxtn2 v19.16b, v7.8h

    cbz x8, STORE_TILE_16
    eor v0.16b, v0.16b, v0.16b
    smax v16.16b, v0.16b, v16.16b
    smax v17.16b, v0.16b, v17.16b
    smax v18.16b, v0.16b, v18.16b
    smax v19.16b, v0.16b, v19.16b
    

    STORE_TILE_16:
    subs x7, x7, #1
    st1 {v16.16b, v17.16b, v18.16b, v19.16b}, [x0], x6
    // st1 {v17.16b}, [x0], #16
    // st1 {v18.16b}, [x0], #16
    // st1 {v19.16b}, [x0], x6

    mov x1, x10
    bne L4LoopDz_TILE_16

b REAL_END

TILE_8:
// sub x6, x6, #16
L4LoopDz_TILE_8:
    ld1 {v7.4s}, [x3], #16 // bias
    mov x10, x1
    mov x11, x5

    mov v8.16b, v7.16b
    mov v9.16b, v7.16b
    mov v10.16b, v7.16b
    mov v11.16b, v7.16b
    mov v12.16b, v7.16b
    mov v13.16b, v7.16b
    mov v14.16b, v7.16b
    mov v15.16b, v7.16b

    L4LoopSz_TILE_8:
        ld1 {v6.16b}, [x2], #16 // weight
        ld1 {v0.16b}, [x1], #16 // src

        sdot v8.4s, v6.16b, v0.4b[0]
        sdot v9.4s, v6.16b, v0.4b[1]
        ld1 {v1.16b}, [x1], #16
        sdot v10.4s, v6.16b, v0.4b[2]
        sdot v11.4s, v6.16b, v0.4b[3]

        subs x11, x11, #1
        sdot v12.4s, v6.16b, v1.4b[0]
        sdot v13.4s, v6.16b, v1.4b[1]
        add x1, x1, #32
        sdot v14.4s, v6.16b, v1.4b[2]
        sdot v15.4s, v6.16b, v1.4b[3]

        bne L4LoopSz_TILE_8

    L4LoopSzEnd_TILE_8:

    ld1 {v1.4s}, [x4], #16 // scale

    scvtf v8.4s, v8.4s
    scvtf v9.4s, v9.4s
    scvtf v10.4s, v10.4s
    scvtf v11.4s, v11.4s
    scvtf v12.4s, v12.4s
    scvtf v13.4s, v13.4s
    scvtf v14.4s, v14.4s
    scvtf v15.4s, v15.4s
    
    fmul v8.4s, v8.4s, v1.4s
    fmul v9.4s, v9.4s, v1.4s
    fmul v10.4s, v10.4s, v1.4s
    fmul v11.4s, v11.4s, v1.4s
    fmul v12.4s, v12.4s, v1.4s
    fmul v13.4s, v13.4s, v1.4s
    fmul v14.4s, v14.4s, v1.4s
    fmul v15.4s, v15.4s, v1.4s


    fcvtas v8.4s, v8.4s
    fcvtas v9.4s, v9.4s
    fcvtas v10.4s, v10.4s
    fcvtas v11.4s, v11.4s
    fcvtas v12.4s, v12.4s
    fcvtas v13.4s, v13.4s
    fcvtas v14.4s, v14.4s
    fcvtas v15.4s, v15.4s
    

    sqxtn v0.4h,  v8.4s
    sqxtn2 v0.8h, v9.4s
    sqxtn v1.4h,  v10.4s
    sqxtn2 v1.8h, v11.4s
    sqxtn v2.4h,  v12.4s
    sqxtn2 v2.8h, v13.4s
    sqxtn v3.4h,  v14.4s
    sqxtn2 v3.8h, v15.4s
    

    sqxtn v16.8b, v0.8h
    sqxtn2 v16.16b, v1.8h
    sqxtn v17.8b, v2.8h
    sqxtn2 v17.16b, v3.8h

    cbz x8, STORE_TILE_8
    eor v0.16b, v0.16b, v0.16b
    smax v16.16b, v0.16b, v16.16b
    smax v17.16b, v0.16b, v17.16b
    

    STORE_TILE_8:
    subs x7, x7, #1
    st1 {v16.16b, v17.16b}, [x0], x6
    // st1 {v17.16b}, [x0], x6
    mov x1, x10
    bne L4LoopDz_TILE_8

b REAL_END

TILE_4:
L4LoopDz_TILE_4:
    ld1 {v7.4s}, [x3], #16 // bias
    mov x10, x1
    mov x11, x5

    mov v8.16b, v7.16b
    mov v9.16b, v7.16b
    mov v10.16b, v7.16b
    mov v11.16b, v7.16b

    L4LoopSz_TILE_4:
        ld1 {v6.16b}, [x2], #16 // weight
        ld1 {v0.16b}, [x1], #16// src

        sdot v8.4s, v6.16b, v0.4b[0]
        sdot v9.4s, v6.16b, v0.4b[1]
        subs x11, x11, #1
        add x1, x1, #48
        sdot v10.4s, v6.16b, v0.4b[2]
        sdot v11.4s, v6.16b, v0.4b[3]

        bne L4LoopSz_TILE_4

    L4LoopSzEnd_TILE_4:

    ld1 {v1.4s}, [x4], #16 // scale

    scvtf v8.4s, v8.4s
    scvtf v9.4s, v9.4s
    scvtf v10.4s, v10.4s
    scvtf v11.4s, v11.4s
    
    fmul v8.4s, v8.4s, v1.4s
    fmul v9.4s, v9.4s, v1.4s
    fmul v10.4s, v10.4s, v1.4s
    fmul v11.4s, v11.4s, v1.4s


    fcvtas v8.4s, v8.4s
    fcvtas v9.4s, v9.4s
    fcvtas v10.4s, v10.4s
    fcvtas v11.4s, v11.4s
    

    sqxtn v0.4h,  v8.4s
    sqxtn2 v0.8h, v9.4s
    sqxtn v1.4h,  v10.4s
    sqxtn2 v1.8h, v11.4s
    

    sqxtn v16.8b, v0.8h
    sqxtn2 v16.16b, v1.8h

    cbz x8, STORE_TILE_4
    eor v0.16b, v0.16b, v0.16b
    smax v16.16b, v0.16b, v16.16b
    
    STORE_TILE_4:
    subs x7, x7, #1
    st1 {v16.16b}, [x0], x6
    mov x1, x10
    bne L4LoopDz_TILE_4

REAL_END:
sub sp, sp, #128
ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64
ret

#endif
