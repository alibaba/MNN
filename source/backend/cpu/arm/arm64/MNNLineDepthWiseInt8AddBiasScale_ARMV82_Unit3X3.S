//
//  MNNLineDepthWiseInt8AddBiasScale_ARMV82_Unit3X3.S
//  MNN
//
//  Created by MNN on 2019/06/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#if defined(__aarch64__)
#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNLineDepthWiseInt8AddBiasScale_ARMV82_Unit3X3

//void MNNLineDepthWiseInt8AddBiasScale_ARMV82_Unit3X3(int8_t* dst, const int8_t* src, const int8_t* weight, const QuanPostTreatParameters* parameters,
//                                          size_t width, size_t src_w_step, size_t fw, size_t fh, size_t dilateX_step,
//                                          size_t dilateY_step, int8_t* idx) {

// kernelx=3, kernely=3,dilatex=1,dilatey=1
/*
struct QuanPostTreatParameters {
    const float* scale;
    const float* biasFloat;
    int32_t maxValue;
    int32_t minValue;
    int32_t useInt8 = 1; // Save result as int8_t dataType; otherwise float32.
    float roundValuePos = 0.5f;
    float roundValueNeg = -0.5f;
    float* srcKernelSum;
    float* weightQuanBias;
    float* fp32minmax;
    ssize_t blockNum = 1;
    const int32_t* bias;

};*/

// Auto Load:
// x0: dst*, x1: src*, x2: weight*, x3: parameters*
// x4: width, x5: src_w_step, x6: fw, x7: fh
// Load from sp
// x8: dilateX_step, x9: dilateY_step

ldr x8, [sp, #0]
ldr x9, [sp, #8]
ldr x14, [sp, #16]

stp d14, d15, [sp, #(-16 * 6)]!
stp d12, d13, [sp, #(16 * 1)]
stp d10, d11, [sp, #(16 * 2)]
stp d8,  d9,  [sp, #(16 * 3)]
stp x21, x22, [sp, #(16 * 4)]
stp x19, x20, [sp, #(16 * 5)]


ldr x19, [x3, #0]  // scale
ldr w20, [x3, #16] // max
ldr w15, [x3, #20] // min
ldr x3, [x3, #72]   // bias
ld1 {v24.16b, v25.16b, v26.16b, v27.16b}, [x14]
ld1 {v0.16b, v1.16b}, [x2], #32 // v0,v1:weight
ld1 {v12.s}[0], [x2]           // weight:    k:8
ld1 {v29.4s}, [x19] // scale
sxtl v12.8h, v12.8b
mov v12.d[1], v12.d[0]
L16:
cmp x4, #16
blt L16End

mov x10, #16
mov x14, #8

mul x10, x5, x10
mul x14, x5, x14

add x11, x1, #8 // move 2 points(8=2x4)
add x12, x1, #16
add x21, x1, #24

L16Loop:
    mov x13, x1
    mov x6, x11

    ld1 {v2.16b}, [x1], x9
    ld1 {v13.16b}, [x11], x9
    mov x7, x12
    mov x22, x21
    ld1 {v3.16b}, [x1], x9
    ld1 {v14.16b}, [x11], x9
    ld1 {v5.4s, v6.4s, v7.4s, v8.4s}, [x3]
    ld1 {v4.16b}, [x1]
    ld1 {v15.16b}, [x11]
    
    sxtl2 v11.8h, v4.16b
    sxtl2 v22.8h, v15.16b

    tbl v16.16b, {v2.16b, v3.16b}, v24.16b  // src0
    tbl v17.16b, {v3.16b, v4.16b}, v25.16b  // src0
    tbl v9.16b, {v2.16b, v3.16b}, v26.16b  // src1
    tbl v10.16b, {v3.16b, v4.16b}, v27.16b // src1
    tbl v18.16b, {v13.16b, v14.16b}, v24.16b // src2
    tbl v19.16b, {v14.16b, v15.16b}, v25.16b // src2
    tbl v20.16b, {v13.16b, v14.16b}, v26.16b // src3
    tbl v21.16b, {v14.16b, v15.16b}, v27.16b // src3

    .inst 0x4e909405 // sdot v5.4s, v0.16b, v16.16b
    .inst 0x4e899406 // sdot v6.4s, v0.16b, v9.16b
    .inst 0x4e929407 // sdot v7.4s, v0.16b, v18.16b
    .inst 0x4e949408 // sdot v8.4s, v0.16b, v20.16b
    .inst 0x4e919425 // sdot v5.4s, v1.16b, v17.16b
    .inst 0x4e8a9426 // sdot v6.4s, v1.16b, v10.16b
    .inst 0x4e939427 // sdot v7.4s, v1.16b, v19.16b
    .inst 0x4e959428 // sdot v8.4s, v1.16b, v21.16b

    ld1 {v2.16b}, [x12], x9
    ld1 {v13.16b}, [x21], x9
    ld1 {v3.16b}, [x12], x9
    ld1 {v14.16b}, [x21], x9
    ld1 {v4.16b}, [x12]
    ld1 {v15.16b}, [x21]

    smlal v5.4s, v11.4h, v12.4h
    smlal2 v6.4s, v11.8h, v12.8h
    smlal v7.4s, v22.4h, v12.4h
    smlal2 v8.4s, v22.8h, v12.8h
    ld1 {v18.4s, v19.4s, v20.4s, v21.4s}, [x3]

    tbl v16.16b, {v2.16b, v3.16b}, v24.16b  // src4
    tbl v17.16b, {v3.16b, v4.16b}, v25.16b  // src4
    tbl v9.16b, {v2.16b, v3.16b}, v26.16b  // src5
    tbl v10.16b, {v3.16b, v4.16b}, v27.16b // src5
    tbl v23.16b, {v13.16b, v14.16b}, v24.16b // src6
    tbl v30.16b, {v14.16b, v15.16b}, v25.16b // src6
    tbl v31.16b, {v13.16b, v14.16b}, v26.16b // src7
    tbl v11.16b, {v14.16b, v15.16b}, v27.16b // src7
    
    add x1, x13, x14
    add x11, x6, x14
    add x12, x7, x14
    add x21, x22, x14

    .inst 0x4e909412 // sdot v18.4s, v0.16b, v16.16b
    .inst 0x4e899413 // sdot v19.4s, v0.16b, v9.16b
    .inst 0x4e979414 // sdot v20.4s, v0.16b, v23.16b
    .inst 0x4e9f9415 // sdot v21.4s, v0.16b, v31.16b
    .inst 0x4e919432 // sdot v18.4s, v1.16b, v17.16b
    .inst 0x4e8a9433 // sdot v19.4s, v1.16b, v10.16b
    .inst 0x4e9e9434 // sdot v20.4s, v1.16b, v30.16b
    .inst 0x4e8b9435 // sdot v21.4s, v1.16b, v11.16b

    sxtl2 v9.8h, v4.16b
    sxtl2 v22.8h, v15.16b

    /*src 8-11 */
    ld1 {v2.16b}, [x1], x9
    ld1 {v13.16b}, [x11], x9
    ld1 {v3.16b}, [x1], x9
    ld1 {v14.16b}, [x11], x9
    ld1 {v4.16b}, [x1]
    ld1 {v15.16b}, [x11]

    scvtf v5.4s, v5.4s
    scvtf v6.4s, v6.4s
    scvtf v7.4s, v7.4s
    scvtf v8.4s, v8.4s

    smlal  v18.4s, v9.4h, v12.4h
    smlal2 v19.4s, v9.8h, v12.8h
    smlal  v20.4s, v22.4h, v12.4h
    smlal2 v21.4s, v22.8h, v12.8h

    tbl v16.16b, {v2.16b, v3.16b}, v24.16b  // src8
    tbl v17.16b, {v3.16b, v4.16b}, v25.16b  // src8
    tbl v9.16b, {v2.16b, v3.16b}, v26.16b  // src9
    tbl v10.16b, {v3.16b, v4.16b}, v27.16b // src9
    tbl v23.16b, {v13.16b, v14.16b}, v24.16b // src10
    tbl v30.16b, {v14.16b, v15.16b}, v25.16b // src10
    tbl v31.16b, {v13.16b, v14.16b}, v26.16b // src11
    tbl v28.16b, {v14.16b, v15.16b}, v27.16b // src11

    fmul v5.4s, v5.4s, v29.4s
    fmul v6.4s, v6.4s, v29.4s
    fmul v7.4s, v7.4s, v29.4s
    fmul v8.4s, v8.4s, v29.4s

    scvtf v18.4s, v18.4s
    scvtf v19.4s, v19.4s
    scvtf v20.4s, v20.4s
    scvtf v21.4s, v21.4s

    fcvtas v5.4s, v5.4s
    fcvtas v6.4s, v6.4s
    fcvtas v7.4s, v7.4s
    fcvtas v8.4s, v8.4s

    fmul v18.4s, v18.4s, v29.4s
    fmul v19.4s, v19.4s, v29.4s
    fmul v20.4s, v20.4s, v29.4s
    fmul v21.4s, v21.4s, v29.4s

    sqxtn v11.4h, v5.4s
    sqxtn2 v11.8h, v6.4s
    sqxtn v22.4h, v7.4s
    sqxtn2 v22.8h, v8.4s

    fcvtas v18.4s, v18.4s
    fcvtas v19.4s, v19.4s
    fcvtas v20.4s, v20.4s
    fcvtas v21.4s, v21.4s

    ld1 {v5.4s, v6.4s, v7.4s, v8.4s}, [x3]

    sqxtn v14.8b, v11.8h
    sqxtn2 v14.16b, v22.8h
    sub x4, x4, #16
    sqxtn v13.4h, v18.4s
    sqxtn2 v13.8h, v19.4s
    sqxtn v11.4h, v20.4s
    sqxtn2 v11.8h, v21.4s

    sxtl2 v4.8h, v4.16b
    sxtl2 v18.8h, v15.16b
    .inst 0x4e909405 // sdot v5.4s, v0.16b, v16.16b
    .inst 0x4e899406 // sdot v6.4s, v0.16b, v9.16b
    .inst 0x4e979407 // sdot v7.4s, v0.16b, v23.16b
    .inst 0x4e9f9408 // sdot v8.4s, v0.16b, v31.16b
    .inst 0x4e919425 // sdot v5.4s, v1.16b, v17.16b
    .inst 0x4e8a9426 // sdot v6.4s, v1.16b, v10.16b
    .inst 0x4e9e9427 // sdot v7.4s, v1.16b, v30.16b
    .inst 0x4e9c9428 // sdot v8.4s, v1.16b, v28.16b


    smlal v5.4s, v4.4h, v12.4h
    smlal2 v6.4s, v4.8h, v12.8h
    smlal v7.4s, v18.4h, v12.4h
    smlal2 v8.4s, v18.8h, v12.8h

    dup v23.16b, w15
    dup v28.16b, w20
    /* src: 12-15 */
    ld1 {v2.16b}, [x12], x9
    ld1 {v9.16b}, [x21], x9
    ld1 {v3.16b}, [x12], x9
    ld1 {v10.16b}, [x21], x9

    sqxtn v15.8b, v13.8h
    sqxtn2 v15.16b, v11.8h

    scvtf v5.4s, v5.4s
    scvtf v6.4s, v6.4s
    scvtf v7.4s, v7.4s
    scvtf v8.4s, v8.4s
    ld1 {v17.4s, v18.4s, v19.4s, v20.4s}, [x3]
    ld1 {v4.16b}, [x12]
    ld1 {v11.16b}, [x21]

    smin v14.16b, v14.16b, v28.16b
    smax v14.16b, v14.16b, v23.16b
    smin v15.16b, v15.16b, v28.16b
    smax v15.16b, v15.16b, v23.16b

    fmul v5.4s, v5.4s, v29.4s
    fmul v6.4s, v6.4s, v29.4s
    fmul v7.4s, v7.4s, v29.4s
    fmul v8.4s, v8.4s, v29.4s

    tbl v13.16b, {v2.16b, v3.16b}, v24.16b  // src8
    tbl v21.16b, {v3.16b, v4.16b}, v25.16b  // src8
    tbl v22.16b, {v2.16b, v3.16b}, v26.16b  // src9
    tbl v16.16b, {v3.16b, v4.16b}, v27.16b // src9

    fcvtas v5.4s, v5.4s
    fcvtas v6.4s, v6.4s
    fcvtas v7.4s, v7.4s
    fcvtas v8.4s, v8.4s
    tbl v3.16b, {v9.16b, v10.16b}, v24.16b // src10
    tbl v30.16b, {v10.16b, v11.16b}, v25.16b // src10
    tbl v31.16b, {v9.16b, v10.16b}, v26.16b // src11
    tbl v2.16b, {v10.16b, v11.16b}, v27.16b // src11

    .inst 0x4e8d9411 // sdot v17.4s, v0.16b, v13.16b
    .inst 0x4e969412 // sdot v18.4s, v0.16b, v22.16b
    .inst 0x4e839413 // sdot v19.4s, v0.16b, v3.16b
    .inst 0x4e9f9414 // sdot v20.4s, v0.16b, v31.16b
    .inst 0x4e959431 // sdot v17.4s, v1.16b, v21.16b
    .inst 0x4e909432 // sdot v18.4s, v1.16b, v16.16b
    .inst 0x4e9e9433 // sdot v19.4s, v1.16b, v30.16b
    .inst 0x4e829434 // sdot v20.4s, v1.16b, v2.16b

    sxtl2 v4.8h, v4.16b
    sxtl2 v11.8h, v11.16b

    add x1, x13, x10
    add x11, x6, x10
    add x12, x7, x10
    add x21, x22, x10

    smlal v17.4s, v4.4h, v12.4h
    smlal2 v18.4s, v4.8h, v12.8h
    smlal v19.4s, v11.4h, v12.4h
    smlal2 v20.4s, v11.8h, v12.8h

    scvtf v17.4s, v17.4s
    scvtf v18.4s, v18.4s
    scvtf v19.4s, v19.4s
    scvtf v20.4s, v20.4s

    sqxtn v13.4h, v5.4s
    sqxtn2 v13.8h, v6.4s
    sqxtn v22.4h, v7.4s
    sqxtn2 v22.8h, v8.4s

    fmul v17.4s, v17.4s, v29.4s
    fmul v18.4s, v18.4s, v29.4s
    fmul v19.4s, v19.4s, v29.4s
    fmul v20.4s, v20.4s, v29.4s

    fcvtas v17.4s, v17.4s
    fcvtas v18.4s, v18.4s
    fcvtas v19.4s, v19.4s
    fcvtas v20.4s, v20.4s

    sqxtn v30.4h, v17.4s
    sqxtn2 v30.8h, v18.4s
    sqxtn v31.4h, v19.4s
    sqxtn2 v31.8h, v20.4s

    sqxtn v16.8b, v13.8h
    sqxtn2 v16.16b, v22.8h
    sqxtn v17.8b, v30.8h
    sqxtn2 v17.16b, v31.8h

    smin v16.16b, v16.16b, v28.16b
    smin v17.16b, v17.16b, v28.16b
    smax v16.16b, v16.16b, v23.16b
    smax v17.16b, v17.16b, v23.16b


   st1 {v14.16b, v15.16b, v16.16b, v17.16b}, [x0], #64
    
    cmp x4, #16
    bge L16Loop

L16End:

L8:
cmp x4, #8
blt L8End

mov x10, #8

mul x10, x5, x10


add x11, x1, #8 // move 2 points(8=2x4)
add x12, x1, #16
add x21, x1, #24

L8Loop:
    mov x13, x1
    mov x6, x11

    ld1 {v2.16b}, [x1], x9
    ld1 {v13.16b}, [x11], x9
    mov x7, x12
    mov x22, x21
    ld1 {v3.16b}, [x1], x9
    ld1 {v14.16b}, [x11], x9
    ld1 {v5.4s, v6.4s, v7.4s, v8.4s}, [x3]
    ld1 {v4.16b}, [x1]
    ld1 {v15.16b}, [x11]
    
    sxtl2 v11.8h, v4.16b
    sxtl2 v22.8h, v15.16b

    tbl v16.16b, {v2.16b, v3.16b}, v24.16b  // src0
    tbl v17.16b, {v3.16b, v4.16b}, v25.16b  // src0
    tbl v9.16b, {v2.16b, v3.16b}, v26.16b  // src1
    tbl v10.16b, {v3.16b, v4.16b}, v27.16b // src1
    tbl v18.16b, {v13.16b, v14.16b}, v24.16b // src2
    tbl v19.16b, {v14.16b, v15.16b}, v25.16b // src2
    tbl v20.16b, {v13.16b, v14.16b}, v26.16b // src3
    tbl v21.16b, {v14.16b, v15.16b}, v27.16b // src3

    .inst 0x4e909405 // sdot v5.4s, v0.16b, v16.16b
    .inst 0x4e899406 // sdot v6.4s, v0.16b, v9.16b
    .inst 0x4e929407 // sdot v7.4s, v0.16b, v18.16b
    .inst 0x4e949408 // sdot v8.4s, v0.16b, v20.16b
    .inst 0x4e919425 // sdot v5.4s, v1.16b, v17.16b
    .inst 0x4e8a9426 // sdot v6.4s, v1.16b, v10.16b
    .inst 0x4e939427 // sdot v7.4s, v1.16b, v19.16b
    .inst 0x4e959428 // sdot v8.4s, v1.16b, v21.16b

    ld1 {v2.16b}, [x12], x9
    ld1 {v13.16b}, [x21], x9
    ld1 {v3.16b}, [x12], x9
    ld1 {v14.16b}, [x21], x9
    ld1 {v4.16b}, [x12]
    ld1 {v15.16b}, [x21]

    smlal v5.4s, v11.4h, v12.4h
    smlal2 v6.4s, v11.8h, v12.8h
    smlal v7.4s, v22.4h, v12.4h
    smlal2 v8.4s, v22.8h, v12.8h
    

    tbl v16.16b, {v2.16b, v3.16b}, v24.16b  // src0
    tbl v17.16b, {v3.16b, v4.16b}, v25.16b  // src0
    tbl v9.16b, {v2.16b, v3.16b}, v26.16b  // src1
    tbl v10.16b, {v3.16b, v4.16b}, v27.16b // src1
    tbl v23.16b, {v13.16b, v14.16b}, v24.16b // src2
    tbl v30.16b, {v14.16b, v15.16b}, v25.16b // src2
    tbl v31.16b, {v13.16b, v14.16b}, v26.16b // src3
    tbl v11.16b, {v14.16b, v15.16b}, v27.16b // src3
    
    ld1 {v18.4s, v19.4s, v20.4s, v21.4s}, [x3]
    sxtl2 v2.8h, v4.16b
    sxtl2 v22.8h, v15.16b

    .inst 0x4e909412 // sdot v18.4s, v0.16b, v16.16b
    .inst 0x4e899413 // sdot v19.4s, v0.16b, v9.16b
    .inst 0x4e979414 // sdot v20.4s, v0.16b, v23.16b
    .inst 0x4e9f9415 // sdot v21.4s, v0.16b, v31.16b
    .inst 0x4e919432 // sdot v18.4s, v1.16b, v17.16b
    .inst 0x4e8a9433 // sdot v19.4s, v1.16b, v10.16b
    .inst 0x4e9e9434 // sdot v20.4s, v1.16b, v30.16b
    .inst 0x4e8b9435 // sdot v21.4s, v1.16b, v11.16b

    
    scvtf v5.4s, v5.4s
    scvtf v6.4s, v6.4s
    scvtf v7.4s, v7.4s
    scvtf v8.4s, v8.4s

    smlal  v18.4s, v2.4h, v12.4h
    smlal2 v19.4s, v2.8h, v12.8h
    smlal  v20.4s, v22.4h, v12.4h
    smlal2 v21.4s, v22.8h, v12.8h

    fmul v5.4s, v5.4s, v29.4s
    fmul v6.4s, v6.4s, v29.4s
    fmul v7.4s, v7.4s, v29.4s
    fmul v8.4s, v8.4s, v29.4s

    scvtf v18.4s, v18.4s
    scvtf v19.4s, v19.4s
    scvtf v20.4s, v20.4s
    scvtf v21.4s, v21.4s

    fcvtas v5.4s, v5.4s
    fcvtas v6.4s, v6.4s
    fcvtas v7.4s, v7.4s
    fcvtas v8.4s, v8.4s

    dup v23.16b, w15
    dup v9.16b, w20

    fmul v18.4s, v18.4s, v29.4s
    fmul v19.4s, v19.4s, v29.4s
    fmul v20.4s, v20.4s, v29.4s
    fmul v21.4s, v21.4s, v29.4s

    fcvtas v18.4s, v18.4s
    fcvtas v19.4s, v19.4s
    fcvtas v20.4s, v20.4s
    fcvtas v21.4s, v21.4s
    sqxtn v16.4h, v5.4s
    sqxtn2 v16.8h, v6.4s
    sqxtn v17.4h, v7.4s
    sqxtn2 v17.8h, v8.4s
    sub x4, x4, #8
    sqxtn v11.4h, v18.4s
    sqxtn2 v11.8h, v19.4s
    sqxtn v10.4h, v20.4s
    sqxtn2 v10.8h, v21.4s

    sqxtn v14.8b, v16.8h
    sqxtn2 v14.16b, v17.8h
    sqxtn v15.8b, v11.8h
    sqxtn2 v15.16b, v10.8h

    smin v14.16b, v14.16b, v9.16b
    smax v14.16b, v14.16b, v23.16b
    smin v15.16b, v15.16b, v9.16b
    smax v15.16b, v15.16b, v23.16b
    st1 {v14.16b, v15.16b}, [x0], #32

    add x1, x13, x10
    add x11, x6, x10
    add x12, x7, x10
    add x21, x22, x10
    
    cmp x4, #8
    bge L8Loop

L8End:

L4:
cmp x4, #4
blt L4End

mov x10, #4
dup v30.16b, w20 // max
dup v31.16b, w15 // min

mul x10, x5, x10
add x11, x1, #8 // move 2 points(8=2x4)  

L4Loop:
    mov x13, x1
    mov x12, x11

    ld1 {v2.16b}, [x1], x9
    ld1 {v13.16b}, [x11], x9
    ld1 {v3.16b}, [x1], x9
    ld1 {v14.16b}, [x11], x9
    ld1 {v4.16b}, [x1]
    ld1 {v15.16b}, [x11]

    ld1 {v5.4s, v6.4s, v7.4s, v8.4s}, [x3]
    sxtl2 v11.8h, v4.16b
    sxtl2 v22.8h, v15.16b

    tbl v16.16b, {v2.16b, v3.16b}, v24.16b  // src0
    tbl v17.16b, {v3.16b, v4.16b}, v25.16b  // src0
    tbl v9.16b, {v2.16b, v3.16b}, v26.16b  // src1
    tbl v10.16b, {v3.16b, v4.16b}, v27.16b // src1
    tbl v18.16b, {v13.16b, v14.16b}, v24.16b // src2
    tbl v19.16b, {v14.16b, v15.16b}, v25.16b // src2
    tbl v20.16b, {v13.16b, v14.16b}, v26.16b // src3
    tbl v21.16b, {v14.16b, v15.16b}, v27.16b // src3

    .inst 0x4e909405 // sdot v5.4s, v0.16b, v16.16b
    .inst 0x4e899406 // sdot v6.4s, v0.16b, v9.16b
    .inst 0x4e929407 // sdot v7.4s, v0.16b, v18.16b
    .inst 0x4e949408 // sdot v8.4s, v0.16b, v20.16b
    .inst 0x4e919425 // sdot v5.4s, v1.16b, v17.16b
    .inst 0x4e8a9426 // sdot v6.4s, v1.16b, v10.16b
    .inst 0x4e939427 // sdot v7.4s, v1.16b, v19.16b
    .inst 0x4e959428 // sdot v8.4s, v1.16b, v21.16b
    sub x4, x4, #4
    smlal v5.4s, v11.4h, v12.4h
    smlal2 v6.4s, v11.8h, v12.8h
    smlal v7.4s, v22.4h, v12.4h
    smlal2 v8.4s, v22.8h, v12.8h

    scvtf v5.4s, v5.4s
    scvtf v6.4s, v6.4s
    scvtf v7.4s, v7.4s
    scvtf v8.4s, v8.4s
    fmul v5.4s, v5.4s, v29.4s
    fmul v6.4s, v6.4s, v29.4s
    fmul v7.4s, v7.4s, v29.4s
    fmul v8.4s, v8.4s, v29.4s
    fcvtas v5.4s, v5.4s
    fcvtas v6.4s, v6.4s
    fcvtas v7.4s, v7.4s
    fcvtas v8.4s, v8.4s
    sqxtn v20.4h, v5.4s
    sqxtn2 v20.8h, v6.4s
    sqxtn v21.4h, v7.4s
    sqxtn2 v21.8h, v8.4s
    sqxtn v9.8b, v20.8h
    sqxtn2 v9.16b, v21.8h

    smin v9.16b, v9.16b, v30.16b
    smax v9.16b, v9.16b, v31.16b
    st1 {v9.16b}, [x0], #16

    add x1, x13, x10
    add x11, x12, x10
    
    cmp x4, #4
    bge L4Loop

L4End:

L2:
cmp x4, #2
blt L2End

mov x10, #2
dup v30.16b, w20 // max
dup v31.16b, w15 // min

mul x10, x5, x10

L2Loop:
    mov x13, x1

    ld1 {v2.16b}, [x1], x9
    ld1 {v3.16b}, [x1], x9
    ld1 {v4.16b}, [x1]

    ld1 {v5.4s, v6.4s}, [x3]

    tbl v7.16b, {v2.16b, v3.16b}, v24.16b  // src0
    tbl v8.16b, {v3.16b, v4.16b}, v25.16b  // src0
    tbl v9.16b, {v2.16b, v3.16b}, v26.16b  // src1
    tbl v10.16b, {v3.16b, v4.16b}, v27.16b // src1

    sxtl2 v11.8h, v4.16b 

    .inst 0x4e879405 // sdot v5.4s, v0.16b, v7.16b
    .inst 0x4e899406 // sdot v6.4s, v0.16b, v9.16b
    .inst 0x4e889425 // sdot v5.4s, v1.16b, v8.16b
    .inst 0x4e8a9426 // sdot v6.4s, v1.16b, v10.16b

    smlal v5.4s, v11.4h, v12.4h
    smlal2 v6.4s, v11.8h, v12.8h

    scvtf v5.4s, v5.4s
    scvtf v6.4s, v6.4s
    fmul v5.4s, v5.4s, v29.4s
    fmul v6.4s, v6.4s, v29.4s
    fcvtas v5.4s, v5.4s
    fcvtas v6.4s, v6.4s
    sqxtn v7.4h, v5.4s
    sqxtn2 v7.8h, v6.4s
    sqxtn v7.8b, v7.8h

    smin v7.8b, v7.8b, v30.8b
    smax v7.8b, v7.8b, v31.8b
    st1 {v7.8b}, [x0], #8

    add x1, x13, x10
    sub x4, x4, #2
    cmp x4, #2
    bge L2Loop

L2End:

L1:
cmp x4, #1
blt End
ld1 {v20.4s}, [x19] // load scale
ld1 {v16.4s}, [x3] // load bias

dup v30.16b, w20 // max
dup v31.16b, w15 // min
L1Loop:
    mov x13, x1

    // v2, v3, v4: src
    ld1 {v2.8b}, [x1], #8   // src: k:0,1
    ld1 {v2.s}[2], [x1], #4 // src: k:2
    sub x1, x1, #12
    add x1, x1, x9
    ld1 {v3.8b}, [x1], #8  // src: k:3
    ld1 {v3.s}[2], [x1], #4    // src: k:4,5
    sub x1, x1, #12
    add x1, x1, x9
    ld1 {v4.8b}, [x1], #8    // src: k:6,7
    ld1 {v4.s}[2], [x1]

    mov v9.16b, v16.16b
    sxtl2 v6.8h, v4.16b

    tbl v7.16b, {v2.16b, v3.16b}, v24.16b  // src0
    tbl v8.16b, {v3.16b, v4.16b}, v25.16b  // src0

    .inst 0x4e8094e9 // sdot v9.4s, v7.16b, v0.16b
    .inst 0x4e819509 // sdot v9.4s, v8.16b, v1.16b

    smlal v9.4s, v6.4h, v12.4h

    scvtf v9.4s, v9.4s
    fmul v9.4s, v9.4s, v20.4s

    fcvtas v9.4s, v9.4s
    sqxtn  v9.4h, v9.4s
    sqxtn  v9.8b, v9.8h

    smin v9.8b, v9.8b, v30.8b
    smax v9.8b, v9.8b, v31.8b

    st1 {v9.s}[0], [x0], #4

    add x1, x13, x5
    subs x4, x4, #1
    bne L1Loop


End:
ldp x19, x20, [sp, #(16 * 5)]
ldp x21, x22, [sp, #(16 * 4)]
ldp d8,  d9,  [sp, #(16 * 3)]
ldp d10, d11, [sp, #(16 * 2)]
ldp d12, d13, [sp, #(16 * 1)]
ldp d14, d15, [sp], #(16 * 6)
ret

#endif
