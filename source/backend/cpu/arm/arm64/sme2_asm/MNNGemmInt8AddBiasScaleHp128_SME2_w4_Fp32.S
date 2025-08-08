//  MNNGemmInt8AddBiasScaleHp128_SME2_w4_Fp32.S
//  Created by MNN on 2022/09/26.
//  Copyright © 2018, Alibaba Group Holding Limited
 
#if defined(__aarch64__)
#include "MNNAsmGlobal.h"
 
.text
 
.macro REVERT_INPUT_DEQUANT_BIAS rg0, rg1, rg2, rg3
mul \rg1, \rg2, \rg3
sub \rg0, \rg0, \rg1
.endm

 
asm_function MNNGemmInt8AddBiasScaleHp128_SME2_w4_Fp32
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
    const float* extraScale = nullptr;
};
*/
//void MNNGemmInt8AddBiasScaleHp128_SME2_w4_Fp32(int8_t* dst, const int8_t* src,
//    const int8_t* weight, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad,
// const QuanPostTreatParameters* parameters, size_t realDstCount);
 
//Auto: x0:dst, x1:src, x2:weight, x3:src_depth_quad, x4:dst_step x5:dst_depth_quad, x6: parameters, x7: realDstCount
// sme2 Ep=16, LP=4, HP=16

stp x29, x30, [sp, #-320]!
mov x29, sp
stp x19, x20, [sp, #224]
stp x21, x22, [sp, #208]
stp x23, x24, [sp, #192]
stp x25, x26, [sp, #176]
stp x27, x28, [sp, #160]
stp d8, d9,   [sp, #80]
stp d10, d11, [sp, #64]
stp d12, d13, [sp, #48]
stp d14, d15, [sp, #32]
.inst 0xd503477f  // smstart


ldr x9, [x6, #8]  // biasFloat
ldr x13, [x6, #40] // srcKernelSum
ldr x28, [x6, #48] // weightKernelSum
ldr x26, [x6, #64]  // blockNum
ldr x23, [x6, #80]  // input scale
ldr x27, [x6, #88]  // input bias
ldr x8, [x6, #104]  // indices
ldr x14, [x6, #56]  // float32 maxmin ptr

.inst 0xe11f8100  // ldr zt0, [x8]

/* initialize predicates */
mov x22, #48
.inst 0x2518e080  // ptrue p0.b, #4            // first 4 bytes
.inst 0x2518e125  // ptrue p5.b, vl16          // first 16 bytes
.inst 0x2518e3e3  // ptrue p3.b                // 64 bytes valid
.inst 0x25207810  // ptrue pn8.b


lsl x22, x7, #2 // eDest * GEMM_INT8_SRC_UNIT
mov x25, 0       // inputBlockNum=1
cbz x27, TILE_1
mov x25, x22         // input block quant: realDstCount * sizeof(float)


TILE_1:
    cmp x7, #1
    blt End

    mov x24, x5 // dst_depth_quad
    mov x6, x0 // dst
    mov x20, x9 // bias
    mov x19, x13      // input kernel sum
    mov x21, x23     // input dequant scale
 
LoopDz_TILE1:
    mov x11, x1             // src
    mov x15, x26
    .inst 0xa040c298  // ld1w {z24.s-z27.s}, pn8/z, [x20]    // bias
    .inst 0xa041c29c  // ld1w {z28.s-z31.s}, pn8/z, [x20, #4, MUL VL]    // bias
    .inst 0x04345114  // addvl x20, x20, #8
TILE1_BLOCKNUM:
    mov x10, x3             // src_depth_quad

    .inst 0xc00800ff  // zero {za}
    mov w8, #0
    LoopSz_TILE_1_lu1:
        .inst 0xa400216c  // ld1rqb {z12.b}, p0/z, [x11]        // src
        .inst 0xa0408040  // ld1b {z0.b-z3.b}, pn8/z, [x2]        // weight
        // int4->int8
        .inst 0xc08a4004  // luti4 {z4.b-z5.b}, zt0, z0[0]
        .inst 0xc08a4026  // luti4 {z6.b-z7.b}, zt0, z1[0]
        .inst 0xc08a4048  // luti4 {z8.b-z9.b}, zt0, z2[0]
        .inst 0xc08a406a  // luti4 {z10.b-z11.b}, zt0, z3[0]

        .inst 0xc15c90a0  // sdot za.s[w8, 0, VGx4], {z4.b-z7.b}, z12.b[0]
        .inst 0xc15c9124  // sdot za.s[w8, 4, VGx4], {z8.b-z11.b}, z12.b[0]
        subs x10, x10, #1
        add x11, x11, x22
        .inst 0x04225082  // addvl x2, x2, #4
 
    bne LoopSz_TILE_1_lu1
 
 
LoopSzEnd_TILE_1:
    .inst 0xa040c040  // ld1w {z0.s-z3.s}, pn8/z, [x2]     // weight scale: 64*sizeof(float)
    .inst 0xa041c050  // ld1w {z16.s-z19.s}, pn8/z, [x2, #4, MUL VL]    // weight scale
    .inst 0xa042c048  // ld1w {z8.s-z11.s}, pn8/z, [x2, #8, MUL VL]    // weight bias
    .inst 0xa043c04c  // ld1w {z12.s-z15.s}, pn8/z, [x2, #12, MUL VL]    // weight bias

    .inst 0x8540cda4  // ld1rw {z4.s}, p3/z, [x13]     // x kernel sum
    .inst 0x8540cef7  // ld1rw {z23.s}, p3/z, [x23]    // x scale

    // input kernel sum x weight quant bias
    .inst 0x64a40118  // fmla z24.s, z8.s, z4.s[0]
    .inst 0x64a40139  // fmla z25.s, z9.s, z4.s[0]
    .inst 0x64a4015a  // fmla z26.s, z10.s, z4.s[0]
    .inst 0x64a4017b  // fmla z27.s, z11.s, z4.s[0]

    .inst 0x64a4019c  // fmla z28.s, z12.s, z4.s[0]
    .inst 0x64a401bd  // fmla z29.s, z13.s, z4.s[0]
    .inst 0x64a401de  // fmla z30.s, z14.s, z4.s[0]
    .inst 0x64a401ff  // fmla z31.s, z15.s, z4.s[0]

    // input scale x weight scale
    .inst 0x65970800  // fmul z0.s, z0.s, z23.s
    .inst 0x65970821  // fmul z1.s, z1.s, z23.s
    .inst 0x65970842  // fmul z2.s, z2.s, z23.s
    .inst 0x65970863  // fmul z3.s, z3.s, z23.s

    .inst 0x65970a10  // fmul z16.s, z16.s, z23.s
    .inst 0x65970a31  // fmul z17.s, z17.s, z23.s
    .inst 0x65970a52  // fmul z18.s, z18.s, z23.s
    .inst 0x65970a73  // fmul z19.s, z19.s, z23.s

    // blocksum
    .inst 0xc0060c08  // mova {z8.s-z11.s}, za.s[w8, 0, VGx4]
    .inst 0xc0060c8c  // mova {z12.s-z15.s}, za.s[w8, 4, VGx4]

    .inst 0x6594ad08  // scvtf z8.s, p3/m, z8.s
    .inst 0x6594ad29  // scvtf z9.s, p3/m, z9.s
    .inst 0x6594ad4a  // scvtf z10.s, p3/m, z10.s
    .inst 0x6594ad6b  // scvtf z11.s, p3/m, z11.s
    .inst 0x6594ad8c  // scvtf z12.s, p3/m, z12.s
    .inst 0x6594adad  // scvtf z13.s, p3/m, z13.s
    .inst 0x6594adce  // scvtf z14.s, p3/m, z14.s
    .inst 0x6594adef  // scvtf z15.s, p3/m, z15.s

    // accumulate
    .inst 0x65a00d18  // fmla z24.s, p3/m, z8.s, z0.s
    .inst 0x65a10d39  // fmla z25.s, p3/m, z9.s, z1.s
    .inst 0x65a20d5a  // fmla z26.s, p3/m, z10.s, z2.s
    .inst 0x65a30d7b  // fmla z27.s, p3/m, z11.s, z3.s
    .inst 0x65b00d9c  // fmla z28.s, p3/m, z12.s, z16.s
    .inst 0x65b10dbd  // fmla z29.s, p3/m, z13.s, z17.s
    .inst 0x65b20dde  // fmla z30.s, p3/m, z14.s, z18.s
    .inst 0x65b30dff  // fmla z31.s, p3/m, z15.s, z19.s

    .inst 0x04225202  // addvl x2, x2, #16  // weight ptr
    add x13, x13, x22
 
    cbz x27, TILE1_ADD_DSTV
    .inst 0x8540cf64  // ld1rw {z4.s}, p3/z, [x27]    // input dequant bias
    .inst 0xa0408388  // ld1b {z8.b-z11.b}, pn8/z, [x28]     // weight kernel sum
    .inst 0xa041838c  // ld1b {z12.b-z15.b}, pn8/z, [x28, #4, MUL VL]    // weight kernel sum
    .inst 0x64a40118  // fmla z24.s, z8.s, z4.s[0]
    .inst 0x64a40139  // fmla z25.s, z9.s, z4.s[0]
    .inst 0x64a4015a  // fmla z26.s, z10.s, z4.s[0]
    .inst 0x64a4017b  // fmla z27.s, z11.s, z4.s[0]
    .inst 0x64a4019c  // fmla z28.s, z12.s, z4.s[0]
    .inst 0x64a401bd  // fmla z29.s, z13.s, z4.s[0]
    .inst 0x64a401de  // fmla z30.s, z14.s, z4.s[0]
    .inst 0x64a401ff  // fmla z31.s, z15.s, z4.s[0]
    add x27, x27, x25
    add x23, x23, x25
    .inst 0x043c511c  // addvl x28, x28, #8

    TILE1_ADD_DSTV:
    subs x15, x15, #1
    bne TILE1_BLOCKNUM

    TILE1_STORE:
    .inst 0x8540cdc8  // ld1rw {z8.s}, p3/z, [x14]
    .inst 0x8541cdc9  // ld1rw {z9.s}, p3/z, [x14, #4]
    .inst 0xc1a9c918  // fclamp {z24.s-z27.s}, z8.s, z9.s
    .inst 0xc1a9c91c  // fclamp {z28.s-z31.s}, z8.s, z9.s

    cmp x24, #32
    bge TILE1_STORE128
    cmp x24, #31
    beq TILE1_STORE124
    cmp x24, #30
    beq TILE1_STORE120
    cmp x24, #29
    beq TILE1_STORE116
    cmp x24, #28
    beq TILE1_STORE112
    cmp x24, #27
    beq TILE1_STORE108
    cmp x24, #26
    beq TILE1_STORE104
    cmp x24, #25
    beq TILE1_STORE100
    cmp x24, #24
    beq TILE1_STORE96
    cmp x24, #23
    beq TILE1_STORE92
    cmp x24, #22
    beq TILE1_STORE88
    cmp x24, #21
    beq TILE1_STORE84
    cmp x24, #20
    beq TILE1_STORE80
    cmp x24, #19
    beq TILE1_STORE76
    cmp x24, #18
    beq TILE1_STORE72
    cmp x24, #17
    beq TILE1_STORE68
    cmp x24, #16
    beq TILE1_STORE64
    cmp x24, #15
    beq TILE1_STORE60
    cmp x24, #14
    beq TILE1_STORE56
    cmp x24, #13
    beq TILE1_STORE52
    cmp x24, #12
    beq TILE1_STORE48
    cmp x24, #11
    beq TILE1_STORE44
    cmp x24, #10
    beq TILE1_STORE40
    cmp x24, #9
    beq TILE1_STORE36
    cmp x24, #8
    beq TILE1_STORE32
    cmp x24, #7
    beq TILE1_STORE28
    cmp x24, #6
    beq TILE1_STORE24
    cmp x24, #5
    beq TILE1_STORE20
    cmp x24, #4
    beq TILE1_STORE16
    cmp x24, #3
    beq TILE1_STORE12
    cmp x24, #2
    beq TILE1_STORE8

    TILE1_STORE4:
    .inst 0xe400f4d8  // st1b {z24.b}, p5, [x6]
    b TILE1_Dz_End

    TILE1_STORE8:
    .inst 0x05702300  // dup z0.q, z24.q[1]

    .inst 0xe400f4d8  // st1b {z24.b}, p5, [x6]
    .inst 0xe40454c0  // st1b {z0.b}, p5, [x6, x4]
    b TILE1_Dz_End

    TILE1_STORE12:
    .inst 0x05702300  // dup z0.q, z24.q[1]
    .inst 0x05b02301  // dup z1.q, z24.q[2]

    add x9, x6, x4, lsl #1

    .inst 0xe400f4d8  // st1b {z24.b}, p5, [x6]
    .inst 0xe40454c0  // st1b {z0.b}, p5, [x6, x4]
    .inst 0xe400f521  // st1b {z1.b}, p5, [x9]
    b TILE1_Dz_End

    TILE1_STORE16:
    .inst 0x05702300  // dup z0.q, z24.q[1]
    .inst 0x05b02301  // dup z1.q, z24.q[2]
    .inst 0x05f02302  // dup z2.q, z24.q[3]

    add x9, x6, x4, lsl #1

    .inst 0xe400f4d8  // st1b {z24.b}, p5, [x6]
    .inst 0xe40454c0  // st1b {z0.b}, p5, [x6, x4]
    .inst 0xe400f521  // st1b {z1.b}, p5, [x9]
    .inst 0xe4045522  // st1b {z2.b}, p5, [x9, x4]
    b TILE1_Dz_End

    TILE1_STORE20:
    .inst 0x05702300  // dup z0.q, z24.q[1]
    .inst 0x05b02301  // dup z1.q, z24.q[2]
    .inst 0x05f02302  // dup z2.q, z24.q[3]

    add x9, x6, x4, lsl #1
    add x15, x6, x4, lsl #2

    .inst 0xe400f4d8  // st1b {z24.b}, p5, [x6]
    .inst 0xe40454c0  // st1b {z0.b}, p5, [x6, x4]
    .inst 0xe400f521  // st1b {z1.b}, p5, [x9]
    .inst 0xe4045522  // st1b {z2.b}, p5, [x9, x4]
    .inst 0xe400f5f9  // st1b {z25.b}, p5, [x15]
    b TILE1_Dz_End

    TILE1_STORE24:
    .inst 0x05702300  // dup z0.q, z24.q[1]
    .inst 0x05b02301  // dup z1.q, z24.q[2]
    .inst 0x05f02302  // dup z2.q, z24.q[3]
    .inst 0x05702323  // dup z3.q, z25.q[1]

    add x9, x6, x4, lsl #1
    add x15, x6, x4, lsl #2

    .inst 0xe400f4d8  // st1b {z24.b}, p5, [x6]
    .inst 0xe40454c0  // st1b {z0.b}, p5, [x6, x4]
    .inst 0xe400f521  // st1b {z1.b}, p5, [x9]
    .inst 0xe4045522  // st1b {z2.b}, p5, [x9, x4]
    .inst 0xe400f5f9  // st1b {z25.b}, p5, [x15]
    .inst 0xe40455e3  // st1b {z3.b}, p5, [x15, x4]
    b TILE1_Dz_End

    TILE1_STORE28:
    .inst 0x05702300  // dup z0.q, z24.q[1]
    .inst 0x05b02301  // dup z1.q, z24.q[2]
    .inst 0x05f02302  // dup z2.q, z24.q[3]
    .inst 0x05702323  // dup z3.q, z25.q[1]
    .inst 0x05b02324  // dup z4.q, z25.q[2]

    add x9, x6, x4, lsl #1
    add x15, x6, x4, lsl #2
    add x8, x9, x4, lsl #2

    .inst 0xe400f4d8  // st1b {z24.b}, p5, [x6]
    .inst 0xe40454c0  // st1b {z0.b}, p5, [x6, x4]
    .inst 0xe400f521  // st1b {z1.b}, p5, [x9]
    .inst 0xe4045522  // st1b {z2.b}, p5, [x9, x4]
    .inst 0xe400f5f9  // st1b {z25.b}, p5, [x15]
    .inst 0xe40455e3  // st1b {z3.b}, p5, [x15, x4]
    .inst 0xe400f504  // st1b {z4.b}, p5, [x8]
    b TILE1_Dz_End

    TILE1_STORE32:
    .inst 0x05702300  // dup z0.q, z24.q[1]
    .inst 0x05b02301  // dup z1.q, z24.q[2]
    .inst 0x05f02302  // dup z2.q, z24.q[3]
    .inst 0x05702323  // dup z3.q, z25.q[1]
    .inst 0x05b02324  // dup z4.q, z25.q[2]
    .inst 0x05f02325  // dup z5.q, z25.q[3]

    add x9, x6, x4, lsl #1
    add x15, x6, x4, lsl #2
    add x8, x9, x4, lsl #2

    .inst 0xe400f4d8  // st1b {z24.b}, p5, [x6]
    .inst 0xe40454c0  // st1b {z0.b}, p5, [x6, x4]
    .inst 0xe400f521  // st1b {z1.b}, p5, [x9]
    .inst 0xe4045522  // st1b {z2.b}, p5, [x9, x4]
    .inst 0xe400f5f9  // st1b {z25.b}, p5, [x15]
    .inst 0xe40455e3  // st1b {z3.b}, p5, [x15, x4]
    .inst 0xe400f504  // st1b {z4.b}, p5, [x8]
    .inst 0xe4045505  // st1b {z5.b}, p5, [x8, x4]
    b TILE1_Dz_End

    TILE1_STORE36:
    .inst 0x05702300  // dup z0.q, z24.q[1]
    .inst 0x05b02301  // dup z1.q, z24.q[2]
    .inst 0x05f02302  // dup z2.q, z24.q[3]
    .inst 0x05702323  // dup z3.q, z25.q[1]
    .inst 0x05b02324  // dup z4.q, z25.q[2]
    .inst 0x05f02325  // dup z5.q, z25.q[3]

    add x9, x6, x4, lsl #1
    add x15, x6, x4, lsl #2
    add x8, x9, x4, lsl #2
    add x11, x6, x4, lsl #3

    .inst 0xe400f4d8  // st1b {z24.b}, p5, [x6]
    .inst 0xe40454c0  // st1b {z0.b}, p5, [x6, x4]
    .inst 0xe400f521  // st1b {z1.b}, p5, [x9]
    .inst 0xe4045522  // st1b {z2.b}, p5, [x9, x4]
    .inst 0xe400f5f9  // st1b {z25.b}, p5, [x15]
    .inst 0xe40455e3  // st1b {z3.b}, p5, [x15, x4]
    .inst 0xe400f504  // st1b {z4.b}, p5, [x8]
    .inst 0xe4045505  // st1b {z5.b}, p5, [x8, x4]
    .inst 0xe400f57a  // st1b {z26.b}, p5, [x11]
    b TILE1_Dz_End

    TILE1_STORE40:
    .inst 0x05702300  // dup z0.q, z24.q[1]
    .inst 0x05b02301  // dup z1.q, z24.q[2]
    .inst 0x05f02302  // dup z2.q, z24.q[3]
    .inst 0x05702323  // dup z3.q, z25.q[1]
    .inst 0x05b02324  // dup z4.q, z25.q[2]
    .inst 0x05f02325  // dup z5.q, z25.q[3]
    .inst 0x05702346  // dup z6.q, z26.q[1]

    add x9, x6, x4, lsl #1
    add x15, x6, x4, lsl #2
    add x8, x9, x4, lsl #2
    add x11, x6, x4, lsl #3

    .inst 0xe400f4d8  // st1b {z24.b}, p5, [x6]
    .inst 0xe40454c0  // st1b {z0.b}, p5, [x6, x4]
    .inst 0xe400f521  // st1b {z1.b}, p5, [x9]
    .inst 0xe4045522  // st1b {z2.b}, p5, [x9, x4]
    .inst 0xe400f5f9  // st1b {z25.b}, p5, [x15]
    .inst 0xe40455e3  // st1b {z3.b}, p5, [x15, x4]
    .inst 0xe400f504  // st1b {z4.b}, p5, [x8]
    .inst 0xe4045505  // st1b {z5.b}, p5, [x8, x4]
    .inst 0xe400f57a  // st1b {z26.b}, p5, [x11]
    .inst 0xe4045566  // st1b {z6.b}, p5, [x11, x4]
    b TILE1_Dz_End

    TILE1_STORE44:
    .inst 0x05702300  // dup z0.q, z24.q[1]
    .inst 0x05b02301  // dup z1.q, z24.q[2]
    .inst 0x05f02302  // dup z2.q, z24.q[3]
    .inst 0x05702323  // dup z3.q, z25.q[1]
    .inst 0x05b02324  // dup z4.q, z25.q[2]
    .inst 0x05f02325  // dup z5.q, z25.q[3]
    .inst 0x05702346  // dup z6.q, z26.q[1]
    .inst 0x05b02347  // dup z7.q, z26.q[2]

    add x9, x6, x4, lsl #1
    add x15, x6, x4, lsl #2
    add x8, x9, x4, lsl #2
    add x11, x6, x4, lsl #3
    add x13, x9, x4, lsl #3

    .inst 0xe400f4d8  // st1b {z24.b}, p5, [x6]
    .inst 0xe40454c0  // st1b {z0.b}, p5, [x6, x4]
    .inst 0xe400f521  // st1b {z1.b}, p5, [x9]
    .inst 0xe4045522  // st1b {z2.b}, p5, [x9, x4]
    .inst 0xe400f5f9  // st1b {z25.b}, p5, [x15]
    .inst 0xe40455e3  // st1b {z3.b}, p5, [x15, x4]
    .inst 0xe400f504  // st1b {z4.b}, p5, [x8]
    .inst 0xe4045505  // st1b {z5.b}, p5, [x8, x4]
    .inst 0xe400f57a  // st1b {z26.b}, p5, [x11]
    .inst 0xe4045566  // st1b {z6.b}, p5, [x11, x4]
    .inst 0xe400f5a7  // st1b {z7.b}, p5, [x13]
    b TILE1_Dz_End

TILE1_STORE48:
    .inst 0x05702300  // dup z0.q, z24.q[1]
    .inst 0x05b02301  // dup z1.q, z24.q[2]
    .inst 0x05f02302  // dup z2.q, z24.q[3]
    .inst 0x05702323  // dup z3.q, z25.q[1]
    .inst 0x05b02324  // dup z4.q, z25.q[2]
    .inst 0x05f02325  // dup z5.q, z25.q[3]
    .inst 0x05702346  // dup z6.q, z26.q[1]
    .inst 0x05b02347  // dup z7.q, z26.q[2]
    .inst 0x05f02348  // dup z8.q, z26.q[3]

    add x9, x6, x4, lsl #1
    add x15, x6, x4, lsl #2
    add x8, x9, x4, lsl #2
    add x11, x6, x4, lsl #3
    add x13, x9, x4, lsl #3

    .inst 0xe400f4d8  // st1b {z24.b}, p5, [x6]
    .inst 0xe40454c0  // st1b {z0.b}, p5, [x6, x4]
    .inst 0xe400f521  // st1b {z1.b}, p5, [x9]
    .inst 0xe4045522  // st1b {z2.b}, p5, [x9, x4]
    .inst 0xe400f5f9  // st1b {z25.b}, p5, [x15]
    .inst 0xe40455e3  // st1b {z3.b}, p5, [x15, x4]
    .inst 0xe400f504  // st1b {z4.b}, p5, [x8]
    .inst 0xe4045505  // st1b {z5.b}, p5, [x8, x4]
    .inst 0xe400f57a  // st1b {z26.b}, p5, [x11]
    .inst 0xe4045566  // st1b {z6.b}, p5, [x11, x4]
    .inst 0xe400f5a7  // st1b {z7.b}, p5, [x13]
    .inst 0xe40455a8  // st1b {z8.b}, p5, [x13, x4]
    b TILE1_Dz_End

TILE1_STORE52:
    .inst 0x05702300  // dup z0.q, z24.q[1]
    .inst 0x05b02301  // dup z1.q, z24.q[2]
    .inst 0x05f02302  // dup z2.q, z24.q[3]
    .inst 0x05702323  // dup z3.q, z25.q[1]
    .inst 0x05b02324  // dup z4.q, z25.q[2]
    .inst 0x05f02325  // dup z5.q, z25.q[3]
    .inst 0x05702346  // dup z6.q, z26.q[1]
    .inst 0x05b02347  // dup z7.q, z26.q[2]
    .inst 0x05f02348  // dup z8.q, z26.q[3]

    add x9, x6, x4, lsl #1
    add x15, x6, x4, lsl #2
    add x8, x9, x4, lsl #2
    add x11, x6, x4, lsl #3
    add x13, x9, x4, lsl #3
    add x23, x15, x4, lsl #3

    .inst 0xe400f4d8  // st1b {z24.b}, p5, [x6]
    .inst 0xe40454c0  // st1b {z0.b}, p5, [x6, x4]
    .inst 0xe400f521  // st1b {z1.b}, p5, [x9]
    .inst 0xe4045522  // st1b {z2.b}, p5, [x9, x4]
    .inst 0xe400f5f9  // st1b {z25.b}, p5, [x15]
    .inst 0xe40455e3  // st1b {z3.b}, p5, [x15, x4]
    .inst 0xe400f504  // st1b {z4.b}, p5, [x8]
    .inst 0xe4045505  // st1b {z5.b}, p5, [x8, x4]
    .inst 0xe400f57a  // st1b {z26.b}, p5, [x11]
    .inst 0xe4045566  // st1b {z6.b}, p5, [x11, x4]
    .inst 0xe400f5a7  // st1b {z7.b}, p5, [x13]
    .inst 0xe40455a8  // st1b {z8.b}, p5, [x13, x4]
    .inst 0xe400f6fb  // st1b {z27.b}, p5, [x23]
    b TILE1_Dz_End

TILE1_STORE56:
    .inst 0x05702300  // dup z0.q, z24.q[1]
    .inst 0x05b02301  // dup z1.q, z24.q[2]
    .inst 0x05f02302  // dup z2.q, z24.q[3]
    .inst 0x05702323  // dup z3.q, z25.q[1]
    .inst 0x05b02324  // dup z4.q, z25.q[2]
    .inst 0x05f02325  // dup z5.q, z25.q[3]
    .inst 0x05702346  // dup z6.q, z26.q[1]
    .inst 0x05b02347  // dup z7.q, z26.q[2]
    .inst 0x05f02348  // dup z8.q, z26.q[3]
    .inst 0x05702369  // dup z9.q, z27.q[1]

    add x9, x6, x4, lsl #1
    add x15, x6, x4, lsl #2
    add x8, x9, x4, lsl #2
    add x11, x6, x4, lsl #3
    add x13, x9, x4, lsl #3
    add x23, x15, x4, lsl #3

    .inst 0xe400f4d8  // st1b {z24.b}, p5, [x6]
    .inst 0xe40454c0  // st1b {z0.b}, p5, [x6, x4]
    .inst 0xe400f521  // st1b {z1.b}, p5, [x9]
    .inst 0xe4045522  // st1b {z2.b}, p5, [x9, x4]
    .inst 0xe400f5f9  // st1b {z25.b}, p5, [x15]
    .inst 0xe40455e3  // st1b {z3.b}, p5, [x15, x4]
    .inst 0xe400f504  // st1b {z4.b}, p5, [x8]
    .inst 0xe4045505  // st1b {z5.b}, p5, [x8, x4]
    .inst 0xe400f57a  // st1b {z26.b}, p5, [x11]
    .inst 0xe4045566  // st1b {z6.b}, p5, [x11, x4]
    .inst 0xe400f5a7  // st1b {z7.b}, p5, [x13]
    .inst 0xe40455a8  // st1b {z8.b}, p5, [x13, x4]
    .inst 0xe400f6fb  // st1b {z27.b}, p5, [x23]
    .inst 0xe40456e9  // st1b {z9.b}, p5, [x23, x4]
    b TILE1_Dz_End

TILE1_STORE60:
    .inst 0x05702300  // dup z0.q, z24.q[1]
    .inst 0x05b02301  // dup z1.q, z24.q[2]
    .inst 0x05f02302  // dup z2.q, z24.q[3]
    .inst 0x05702323  // dup z3.q, z25.q[1]
    .inst 0x05b02324  // dup z4.q, z25.q[2]
    .inst 0x05f02325  // dup z5.q, z25.q[3]
    .inst 0x05702346  // dup z6.q, z26.q[1]
    .inst 0x05b02347  // dup z7.q, z26.q[2]
    .inst 0x05f02348  // dup z8.q, z26.q[3]
    .inst 0x05702369  // dup z9.q, z27.q[1]
    .inst 0x05b0236a  // dup z10.q, z27.q[2]

    add x9, x6, x4, lsl #1
    add x15, x6, x4, lsl #2
    add x8, x9, x4, lsl #2
    add x11, x6, x4, lsl #3
    add x13, x9, x4, lsl #3
    add x23, x15, x4, lsl #3
    add x5, x8, x4, lsl #3

    .inst 0xe400f4d8  // st1b {z24.b}, p5, [x6]
    .inst 0xe40454c0  // st1b {z0.b}, p5, [x6, x4]
    .inst 0xe400f521  // st1b {z1.b}, p5, [x9]
    .inst 0xe4045522  // st1b {z2.b}, p5, [x9, x4]
    .inst 0xe400f5f9  // st1b {z25.b}, p5, [x15]
    .inst 0xe40455e3  // st1b {z3.b}, p5, [x15, x4]
    .inst 0xe400f504  // st1b {z4.b}, p5, [x8]
    .inst 0xe4045505  // st1b {z5.b}, p5, [x8, x4]
    .inst 0xe400f57a  // st1b {z26.b}, p5, [x11]
    .inst 0xe4045566  // st1b {z6.b}, p5, [x11, x4]
    .inst 0xe400f5a7  // st1b {z7.b}, p5, [x13]
    .inst 0xe40455a8  // st1b {z8.b}, p5, [x13, x4]
    .inst 0xe400f6fb  // st1b {z27.b}, p5, [x23]
    .inst 0xe40456e9  // st1b {z9.b}, p5, [x23, x4]
    .inst 0xe400f4aa  // st1b {z10.b}, p5, [x5]
    b TILE1_Dz_End

TILE1_STORE64:
    .inst 0x05702300  // dup z0.q, z24.q[1]
    .inst 0x05b02301  // dup z1.q, z24.q[2]
    .inst 0x05f02302  // dup z2.q, z24.q[3]
    .inst 0x05702323  // dup z3.q, z25.q[1]
    .inst 0x05b02324  // dup z4.q, z25.q[2]
    .inst 0x05f02325  // dup z5.q, z25.q[3]
    .inst 0x05702346  // dup z6.q, z26.q[1]
    .inst 0x05b02347  // dup z7.q, z26.q[2]
    .inst 0x05f02348  // dup z8.q, z26.q[3]
    .inst 0x05702369  // dup z9.q, z27.q[1]
    .inst 0x05b0236a  // dup z10.q, z27.q[2]
    .inst 0x05f0236b  // dup z11.q, z27.q[3]

    add x9, x6, x4, lsl #1
    add x15, x6, x4, lsl #2
    add x8, x9, x4, lsl #2
    add x11, x6, x4, lsl #3
    add x13, x9, x4, lsl #3
    add x23, x15, x4, lsl #3
    add x5, x8, x4, lsl #3

    .inst 0xe400f4d8  // st1b {z24.b}, p5, [x6]
    .inst 0xe40454c0  // st1b {z0.b}, p5, [x6, x4]
    .inst 0xe400f521  // st1b {z1.b}, p5, [x9]
    .inst 0xe4045522  // st1b {z2.b}, p5, [x9, x4]
    .inst 0xe400f5f9  // st1b {z25.b}, p5, [x15]
    .inst 0xe40455e3  // st1b {z3.b}, p5, [x15, x4]
    .inst 0xe400f504  // st1b {z4.b}, p5, [x8]
    .inst 0xe4045505  // st1b {z5.b}, p5, [x8, x4]
    .inst 0xe400f57a  // st1b {z26.b}, p5, [x11]
    .inst 0xe4045566  // st1b {z6.b}, p5, [x11, x4]
    .inst 0xe400f5a7  // st1b {z7.b}, p5, [x13]
    .inst 0xe40455a8  // st1b {z8.b}, p5, [x13, x4]
    .inst 0xe400f6fb  // st1b {z27.b}, p5, [x23]
    .inst 0xe40456e9  // st1b {z9.b}, p5, [x23, x4]
    .inst 0xe400f4aa  // st1b {z10.b}, p5, [x5]
    .inst 0xe40454ab  // st1b {z11.b}, p5, [x5, x4]
    b TILE1_Dz_End

TILE1_STORE68:
    .inst 0x05702300  // dup z0.q, z24.q[1]
    .inst 0x05b02301  // dup z1.q, z24.q[2]
    .inst 0x05f02302  // dup z2.q, z24.q[3]
    .inst 0x05702323  // dup z3.q, z25.q[1]
    .inst 0x05b02324  // dup z4.q, z25.q[2]
    .inst 0x05f02325  // dup z5.q, z25.q[3]
    .inst 0x05702346  // dup z6.q, z26.q[1]
    .inst 0x05b02347  // dup z7.q, z26.q[2]
    .inst 0x05f02348  // dup z8.q, z26.q[3]
    .inst 0x05702369  // dup z9.q, z27.q[1]
    .inst 0x05b0236a  // dup z10.q, z27.q[2]
    .inst 0x05f0236b  // dup z11.q, z27.q[3]
    .inst 0x0570238c  // dup z12.q, z28.q[1]

    add x9, x6, x4, lsl #1
    add x15, x6, x4, lsl #2
    add x8, x9, x4, lsl #2
    add x11, x6, x4, lsl #3
    add x13, x9, x4, lsl #3
    add x23, x15, x4, lsl #3
    add x5, x8, x4, lsl #3

    .inst 0xe400f4d8  // st1b {z24.b}, p5, [x6]
    .inst 0xe40454c0  // st1b {z0.b}, p5, [x6, x4]
    .inst 0xe400f521  // st1b {z1.b}, p5, [x9]
    .inst 0xe4045522  // st1b {z2.b}, p5, [x9, x4]
    .inst 0xe400f5f9  // st1b {z25.b}, p5, [x15]
    .inst 0xe40455e3  // st1b {z3.b}, p5, [x15, x4]
    .inst 0xe400f504  // st1b {z4.b}, p5, [x8]
    .inst 0xe4045505  // st1b {z5.b}, p5, [x8, x4]
    .inst 0xe400f57a  // st1b {z26.b}, p5, [x11]
    .inst 0xe4045566  // st1b {z6.b}, p5, [x11, x4]
    .inst 0xe400f5a7  // st1b {z7.b}, p5, [x13]
    .inst 0xe40455a8  // st1b {z8.b}, p5, [x13, x4]
    .inst 0xe400f6fb  // st1b {z27.b}, p5, [x23]
    .inst 0xe40456e9  // st1b {z9.b}, p5, [x23, x4]
    .inst 0xe400f4aa  // st1b {z10.b}, p5, [x5]
    .inst 0xe40454ab  // st1b {z11.b}, p5, [x5, x4]

    add x9, x6, x4, lsl #4

    .inst 0xe400f53c  // st1b {z28.b}, p5, [x9]
    b TILE1_Dz_End

TILE1_STORE72:
    .inst 0x05702300  // dup z0.q, z24.q[1]
    .inst 0x05b02301  // dup z1.q, z24.q[2]
    .inst 0x05f02302  // dup z2.q, z24.q[3]
    .inst 0x05702323  // dup z3.q, z25.q[1]
    .inst 0x05b02324  // dup z4.q, z25.q[2]
    .inst 0x05f02325  // dup z5.q, z25.q[3]
    .inst 0x05702346  // dup z6.q, z26.q[1]
    .inst 0x05b02347  // dup z7.q, z26.q[2]
    .inst 0x05f02348  // dup z8.q, z26.q[3]
    .inst 0x05702369  // dup z9.q, z27.q[1]
    .inst 0x05b0236a  // dup z10.q, z27.q[2]
    .inst 0x05f0236b  // dup z11.q, z27.q[3]

    add x9, x6, x4, lsl #1
    add x15, x6, x4, lsl #2
    add x8, x9, x4, lsl #2
    add x11, x6, x4, lsl #3
    add x13, x9, x4, lsl #3
    add x23, x15, x4, lsl #3
    add x5, x8, x4, lsl #3

    .inst 0xe400f4d8  // st1b {z24.b}, p5, [x6]
    .inst 0xe40454c0  // st1b {z0.b}, p5, [x6, x4]
    .inst 0xe400f521  // st1b {z1.b}, p5, [x9]
    .inst 0xe4045522  // st1b {z2.b}, p5, [x9, x4]
    .inst 0xe400f5f9  // st1b {z25.b}, p5, [x15]
    .inst 0xe40455e3  // st1b {z3.b}, p5, [x15, x4]
    .inst 0xe400f504  // st1b {z4.b}, p5, [x8]
    .inst 0xe4045505  // st1b {z5.b}, p5, [x8, x4]
    .inst 0xe400f57a  // st1b {z26.b}, p5, [x11]
    .inst 0xe4045566  // st1b {z6.b}, p5, [x11, x4]
    .inst 0xe400f5a7  // st1b {z7.b}, p5, [x13]
    .inst 0xe40455a8  // st1b {z8.b}, p5, [x13, x4]
    .inst 0xe400f6fb  // st1b {z27.b}, p5, [x23]
    .inst 0xe40456e9  // st1b {z9.b}, p5, [x23, x4]
    .inst 0xe400f4aa  // st1b {z10.b}, p5, [x5]
    .inst 0xe40454ab  // st1b {z11.b}, p5, [x5, x4]

    add x9, x6, x4, lsl #4

    .inst 0xe400f53c  // st1b {z28.b}, p5, [x9]
    .inst 0xe404552c  // st1b {z12.b}, p5, [x9, x4]
    b TILE1_Dz_End

TILE1_STORE76:
    .inst 0x05702300  // dup z0.q, z24.q[1]
    .inst 0x05b02301  // dup z1.q, z24.q[2]
    .inst 0x05f02302  // dup z2.q, z24.q[3]
    .inst 0x05702323  // dup z3.q, z25.q[1]
    .inst 0x05b02324  // dup z4.q, z25.q[2]
    .inst 0x05f02325  // dup z5.q, z25.q[3]
    .inst 0x05702346  // dup z6.q, z26.q[1]
    .inst 0x05b02347  // dup z7.q, z26.q[2]
    .inst 0x05f02348  // dup z8.q, z26.q[3]
    .inst 0x05702369  // dup z9.q, z27.q[1]
    .inst 0x05b0236a  // dup z10.q, z27.q[2]
    .inst 0x05f0236b  // dup z11.q, z27.q[3]
    .inst 0x0570238c  // dup z12.q, z28.q[1]

    add x9, x6, x4, lsl #1
    add x15, x6, x4, lsl #2
    add x8, x9, x4, lsl #2
    add x11, x6, x4, lsl #3
    add x13, x9, x4, lsl #3
    add x23, x15, x4, lsl #3
    add x5, x8, x4, lsl #3

    .inst 0xe400f4d8  // st1b {z24.b}, p5, [x6]
    .inst 0xe40454c0  // st1b {z0.b}, p5, [x6, x4]
    .inst 0xe400f521  // st1b {z1.b}, p5, [x9]
    .inst 0xe4045522  // st1b {z2.b}, p5, [x9, x4]
    .inst 0xe400f5f9  // st1b {z25.b}, p5, [x15]
    .inst 0xe40455e3  // st1b {z3.b}, p5, [x15, x4]
    .inst 0xe400f504  // st1b {z4.b}, p5, [x8]
    .inst 0xe4045505  // st1b {z5.b}, p5, [x8, x4]
    .inst 0xe400f57a  // st1b {z26.b}, p5, [x11]
    .inst 0xe4045566  // st1b {z6.b}, p5, [x11, x4]
    .inst 0xe400f5a7  // st1b {z7.b}, p5, [x13]
    .inst 0xe40455a8  // st1b {z8.b}, p5, [x13, x4]
    .inst 0xe400f6fb  // st1b {z27.b}, p5, [x23]
    .inst 0xe40456e9  // st1b {z9.b}, p5, [x23, x4]
    .inst 0xe400f4aa  // st1b {z10.b}, p5, [x5]
    .inst 0xe40454ab  // st1b {z11.b}, p5, [x5, x4]

    add x9, x6, x4, lsl #4
    add x15, x13, x4, lsl #3

    .inst 0xe400f53c  // st1b {z28.b}, p5, [x9]
    .inst 0xe404552c  // st1b {z12.b}, p5, [x9, x4]
    .inst 0xe400f5ed  // st1b {z13.b}, p5, [x15]
    b TILE1_Dz_End

TILE1_STORE80:
    .inst 0x05702300  // dup z0.q, z24.q[1]
    .inst 0x05b02301  // dup z1.q, z24.q[2]
    .inst 0x05f02302  // dup z2.q, z24.q[3]
    .inst 0x05702323  // dup z3.q, z25.q[1]
    .inst 0x05b02324  // dup z4.q, z25.q[2]
    .inst 0x05f02325  // dup z5.q, z25.q[3]
    .inst 0x05702346  // dup z6.q, z26.q[1]
    .inst 0x05b02347  // dup z7.q, z26.q[2]
    .inst 0x05f02348  // dup z8.q, z26.q[3]
    .inst 0x05702369  // dup z9.q, z27.q[1]
    .inst 0x05b0236a  // dup z10.q, z27.q[2]
    .inst 0x05f0236b  // dup z11.q, z27.q[3]
    .inst 0x0570238c  // dup z12.q, z28.q[1]

    add x9, x6, x4, lsl #1
    add x15, x6, x4, lsl #2
    add x8, x9, x4, lsl #2
    add x11, x6, x4, lsl #3
    add x13, x9, x4, lsl #3
    add x23, x15, x4, lsl #3
    add x5, x8, x4, lsl #3

    .inst 0xe400f4d8  // st1b {z24.b}, p5, [x6]
    .inst 0xe40454c0  // st1b {z0.b}, p5, [x6, x4]
    .inst 0xe400f521  // st1b {z1.b}, p5, [x9]
    .inst 0xe4045522  // st1b {z2.b}, p5, [x9, x4]
    .inst 0xe400f5f9  // st1b {z25.b}, p5, [x15]
    .inst 0xe40455e3  // st1b {z3.b}, p5, [x15, x4]
    .inst 0xe400f504  // st1b {z4.b}, p5, [x8]
    .inst 0xe4045505  // st1b {z5.b}, p5, [x8, x4]
    .inst 0xe400f57a  // st1b {z26.b}, p5, [x11]
    .inst 0xe4045566  // st1b {z6.b}, p5, [x11, x4]
    .inst 0xe400f5a7  // st1b {z7.b}, p5, [x13]
    .inst 0xe40455a8  // st1b {z8.b}, p5, [x13, x4]
    .inst 0xe400f6fb  // st1b {z27.b}, p5, [x23]
    .inst 0xe40456e9  // st1b {z9.b}, p5, [x23, x4]
    .inst 0xe400f4aa  // st1b {z10.b}, p5, [x5]
    .inst 0xe40454ab  // st1b {z11.b}, p5, [x5, x4]

    add x9, x6, x4, lsl #4
    add x15, x13, x4, lsl #3

    .inst 0xe400f53c  // st1b {z28.b}, p5, [x9]
    .inst 0xe404552c  // st1b {z12.b}, p5, [x9, x4]
    .inst 0xe400f5ed  // st1b {z13.b}, p5, [x15]
    b TILE1_Dz_End

TILE1_STORE84:
    .inst 0x05702300  // dup z0.q, z24.q[1]
    .inst 0x05b02301  // dup z1.q, z24.q[2]
    .inst 0x05f02302  // dup z2.q, z24.q[3]
    .inst 0x05702323  // dup z3.q, z25.q[1]
    .inst 0x05b02324  // dup z4.q, z25.q[2]
    .inst 0x05f02325  // dup z5.q, z25.q[3]
    .inst 0x05702346  // dup z6.q, z26.q[1]
    .inst 0x05b02347  // dup z7.q, z26.q[2]
    .inst 0x05f02348  // dup z8.q, z26.q[3]
    .inst 0x05702369  // dup z9.q, z27.q[1]
    .inst 0x05b0236a  // dup z10.q, z27.q[2]
    .inst 0x05f0236b  // dup z11.q, z27.q[3]
    .inst 0x0570238c  // dup z12.q, z28.q[1]

    add x9, x6, x4, lsl #1
    add x15, x6, x4, lsl #2
    add x8, x9, x4, lsl #2
    add x11, x6, x4, lsl #3
    add x13, x9, x4, lsl #3
    add x23, x15, x4, lsl #3
    add x5, x8, x4, lsl #3

    .inst 0xe400f4d8  // st1b {z24.b}, p5, [x6]
    .inst 0xe40454c0  // st1b {z0.b}, p5, [x6, x4]
    .inst 0xe400f521  // st1b {z1.b}, p5, [x9]
    .inst 0xe4045522  // st1b {z2.b}, p5, [x9, x4]
    .inst 0xe400f5f9  // st1b {z25.b}, p5, [x15]
    .inst 0xe40455e3  // st1b {z3.b}, p5, [x15, x4]
    .inst 0xe400f504  // st1b {z4.b}, p5, [x8]
    .inst 0xe4045505  // st1b {z5.b}, p5, [x8, x4]
    .inst 0xe400f57a  // st1b {z26.b}, p5, [x11]
    .inst 0xe4045566  // st1b {z6.b}, p5, [x11, x4]
    .inst 0xe400f5a7  // st1b {z7.b}, p5, [x13]
    .inst 0xe40455a8  // st1b {z8.b}, p5, [x13, x4]
    .inst 0xe400f6fb  // st1b {z27.b}, p5, [x23]
    .inst 0xe40456e9  // st1b {z9.b}, p5, [x23, x4]
    .inst 0xe400f4aa  // st1b {z10.b}, p5, [x5]
    .inst 0xe40454ab  // st1b {z11.b}, p5, [x5, x4]

    add x9, x6, x4, lsl #4
    add x15, x13, x4, lsl #3

    .inst 0xe400f53c  // st1b {z28.b}, p5, [x9]
    .inst 0xe404552c  // st1b {z12.b}, p5, [x9, x4]
    .inst 0xe400f5ed  // st1b {z13.b}, p5, [x15]
    b TILE1_Dz_End

TILE1_STORE88:
    .inst 0x05702300  // dup z0.q, z24.q[1]
    .inst 0x05b02301  // dup z1.q, z24.q[2]
    .inst 0x05f02302  // dup z2.q, z24.q[3]
    .inst 0x05702323  // dup z3.q, z25.q[1]
    .inst 0x05b02324  // dup z4.q, z25.q[2]
    .inst 0x05f02325  // dup z5.q, z25.q[3]
    .inst 0x05702346  // dup z6.q, z26.q[1]
    .inst 0x05b02347  // dup z7.q, z26.q[2]
    .inst 0x05f02348  // dup z8.q, z26.q[3]
    .inst 0x05702369  // dup z9.q, z27.q[1]
    .inst 0x05b0236a  // dup z10.q, z27.q[2]
    .inst 0x05f0236b  // dup z11.q, z27.q[3]
    .inst 0x0570238c  // dup z12.q, z28.q[1]
    .inst 0x05b0238d  // dup z13.q, z28.q[2]

    add x9, x6, x4, lsl #1
    add x15, x6, x4, lsl #2
    add x8, x9, x4, lsl #2
    add x11, x6, x4, lsl #3
    add x13, x9, x4, lsl #3
    add x23, x15, x4, lsl #3
    add x5, x8, x4, lsl #3

    .inst 0xe400f4d8  // st1b {z24.b}, p5, [x6]
    .inst 0xe40454c0  // st1b {z0.b}, p5, [x6, x4]
    .inst 0xe400f521  // st1b {z1.b}, p5, [x9]
    .inst 0xe4045522  // st1b {z2.b}, p5, [x9, x4]
    .inst 0xe400f5f9  // st1b {z25.b}, p5, [x15]
    .inst 0xe40455e3  // st1b {z3.b}, p5, [x15, x4]
    .inst 0xe400f504  // st1b {z4.b}, p5, [x8]
    .inst 0xe4045505  // st1b {z5.b}, p5, [x8, x4]
    .inst 0xe400f57a  // st1b {z26.b}, p5, [x11]
    .inst 0xe4045566  // st1b {z6.b}, p5, [x11, x4]
    .inst 0xe400f5a7  // st1b {z7.b}, p5, [x13]
    .inst 0xe40455a8  // st1b {z8.b}, p5, [x13, x4]
    .inst 0xe400f6fb  // st1b {z27.b}, p5, [x23]
    .inst 0xe40456e9  // st1b {z9.b}, p5, [x23, x4]
    .inst 0xe400f4aa  // st1b {z10.b}, p5, [x5]
    .inst 0xe40454ab  // st1b {z11.b}, p5, [x5, x4]

    add x9, x6, x4, lsl #4
    add x15, x13, x4, lsl #3

    .inst 0xe400f53c  // st1b {z28.b}, p5, [x9]
    .inst 0xe404552c  // st1b {z12.b}, p5, [x9, x4]
    .inst 0xe400f5ed  // st1b {z13.b}, p5, [x15]
    b TILE1_Dz_End

TILE1_STORE92:
    .inst 0x05702300  // dup z0.q, z24.q[1]
    .inst 0x05b02301  // dup z1.q, z24.q[2]
    .inst 0x05f02302  // dup z2.q, z24.q[3]
    .inst 0x05702323  // dup z3.q, z25.q[1]
    .inst 0x05b02324  // dup z4.q, z25.q[2]
    .inst 0x05f02325  // dup z5.q, z25.q[3]
    .inst 0x05702346  // dup z6.q, z26.q[1]
    .inst 0x05b02347  // dup z7.q, z26.q[2]
    .inst 0x05f02348  // dup z8.q, z26.q[3]
    .inst 0x05702369  // dup z9.q, z27.q[1]
    .inst 0x05b0236a  // dup z10.q, z27.q[2]
    .inst 0x05f0236b  // dup z11.q, z27.q[3]
    .inst 0x0570238c  // dup z12.q, z28.q[1]
    .inst 0x05b0238d  // dup z13.q, z28.q[2]
    .inst 0x05f0238e  // dup z14.q, z28.q[3]

    add x9, x6, x4, lsl #1
    add x15, x6, x4, lsl #2
    add x8, x9, x4, lsl #2
    add x11, x6, x4, lsl #3
    add x13, x9, x4, lsl #3
    add x23, x15, x4, lsl #3
    add x5, x8, x4, lsl #3

    .inst 0xe400f4d8  // st1b {z24.b}, p5, [x6]
    .inst 0xe40454c0  // st1b {z0.b}, p5, [x6, x4]
    .inst 0xe400f521  // st1b {z1.b}, p5, [x9]
    .inst 0xe4045522  // st1b {z2.b}, p5, [x9, x4]
    .inst 0xe400f5f9  // st1b {z25.b}, p5, [x15]
    .inst 0xe40455e3  // st1b {z3.b}, p5, [x15, x4]
    .inst 0xe400f504  // st1b {z4.b}, p5, [x8]
    .inst 0xe4045505  // st1b {z5.b}, p5, [x8, x4]
    .inst 0xe400f57a  // st1b {z26.b}, p5, [x11]
    .inst 0xe4045566  // st1b {z6.b}, p5, [x11, x4]
    .inst 0xe400f5a7  // st1b {z7.b}, p5, [x13]
    .inst 0xe40455a8  // st1b {z8.b}, p5, [x13, x4]
    .inst 0xe400f6fb  // st1b {z27.b}, p5, [x23]
    .inst 0xe40456e9  // st1b {z9.b}, p5, [x23, x4]
    .inst 0xe400f4aa  // st1b {z10.b}, p5, [x5]
    .inst 0xe40454ab  // st1b {z11.b}, p5, [x5, x4]

    add x9, x6, x4, lsl #4
    add x15, x13, x4, lsl #3
    add x8, x23, x4, lsl #3

    .inst 0xe400f53c  // st1b {z28.b}, p5, [x9]
    .inst 0xe404552c  // st1b {z12.b}, p5, [x9, x4]
    .inst 0xe400f5ed  // st1b {z13.b}, p5, [x15]
    .inst 0xe40455ee  // st1b {z14.b}, p5, [x15, x4]
    .inst 0xe400f51d  // st1b {z29.b}, p5, [x8]
    b TILE1_Dz_End

TILE1_STORE96:
    .inst 0x05702300  // dup z0.q, z24.q[1]
    .inst 0x05b02301  // dup z1.q, z24.q[2]
    .inst 0x05f02302  // dup z2.q, z24.q[3]
    .inst 0x05702323  // dup z3.q, z25.q[1]
    .inst 0x05b02324  // dup z4.q, z25.q[2]
    .inst 0x05f02325  // dup z5.q, z25.q[3]
    .inst 0x05702346  // dup z6.q, z26.q[1]
    .inst 0x05b02347  // dup z7.q, z26.q[2]
    .inst 0x05f02348  // dup z8.q, z26.q[3]
    .inst 0x05702369  // dup z9.q, z27.q[1]
    .inst 0x05b0236a  // dup z10.q, z27.q[2]
    .inst 0x05f0236b  // dup z11.q, z27.q[3]
    .inst 0x0570238c  // dup z12.q, z28.q[1]
    .inst 0x05b0238d  // dup z13.q, z28.q[2]
    .inst 0x05f0238e  // dup z14.q, z28.q[3]
    .inst 0x057023af  // dup z15.q, z29.q[1]

    add x9, x6, x4, lsl #1
    add x15, x6, x4, lsl #2
    add x8, x9, x4, lsl #2
    add x11, x6, x4, lsl #3
    add x13, x9, x4, lsl #3
    add x23, x15, x4, lsl #3
    add x5, x8, x4, lsl #3

    .inst 0xe400f4d8  // st1b {z24.b}, p5, [x6]
    .inst 0xe40454c0  // st1b {z0.b}, p5, [x6, x4]
    .inst 0xe400f521  // st1b {z1.b}, p5, [x9]
    .inst 0xe4045522  // st1b {z2.b}, p5, [x9, x4]
    .inst 0xe400f5f9  // st1b {z25.b}, p5, [x15]
    .inst 0xe40455e3  // st1b {z3.b}, p5, [x15, x4]
    .inst 0xe400f504  // st1b {z4.b}, p5, [x8]
    .inst 0xe4045505  // st1b {z5.b}, p5, [x8, x4]
    .inst 0xe400f57a  // st1b {z26.b}, p5, [x11]
    .inst 0xe4045566  // st1b {z6.b}, p5, [x11, x4]
    .inst 0xe400f5a7  // st1b {z7.b}, p5, [x13]
    .inst 0xe40455a8  // st1b {z8.b}, p5, [x13, x4]
    .inst 0xe400f6fb  // st1b {z27.b}, p5, [x23]
    .inst 0xe40456e9  // st1b {z9.b}, p5, [x23, x4]
    .inst 0xe400f4aa  // st1b {z10.b}, p5, [x5]
    .inst 0xe40454ab  // st1b {z11.b}, p5, [x5, x4]

    add x9, x6, x4, lsl #4
    add x15, x13, x4, lsl #3
    add x8, x23, x4, lsl #3

    .inst 0xe400f53c  // st1b {z28.b}, p5, [x9]
    .inst 0xe404552c  // st1b {z12.b}, p5, [x9, x4]
    .inst 0xe400f5ed  // st1b {z13.b}, p5, [x15]
    .inst 0xe40455ee  // st1b {z14.b}, p5, [x15, x4]
    .inst 0xe400f51d  // st1b {z29.b}, p5, [x8]
    .inst 0xe404550f  // st1b {z15.b}, p5, [x8, x4]
    b TILE1_Dz_End

TILE1_STORE100:
    .inst 0x05702300  // dup z0.q, z24.q[1]
    .inst 0x05b02301  // dup z1.q, z24.q[2]
    .inst 0x05f02302  // dup z2.q, z24.q[3]
    .inst 0x05702323  // dup z3.q, z25.q[1]
    .inst 0x05b02324  // dup z4.q, z25.q[2]
    .inst 0x05f02325  // dup z5.q, z25.q[3]
    .inst 0x05702346  // dup z6.q, z26.q[1]
    .inst 0x05b02347  // dup z7.q, z26.q[2]
    .inst 0x05f02348  // dup z8.q, z26.q[3]
    .inst 0x05702369  // dup z9.q, z27.q[1]
    .inst 0x05b0236a  // dup z10.q, z27.q[2]
    .inst 0x05f0236b  // dup z11.q, z27.q[3]
    .inst 0x0570238c  // dup z12.q, z28.q[1]
    .inst 0x05b0238d  // dup z13.q, z28.q[2]
    .inst 0x05f0238e  // dup z14.q, z28.q[3]
    .inst 0x057023af  // dup z15.q, z29.q[1]
    .inst 0x05b023b0  // dup z16.q, z29.q[2]

    add x9, x6, x4, lsl #1
    add x15, x6, x4, lsl #2
    add x8, x9, x4, lsl #2
    add x11, x6, x4, lsl #3
    add x13, x9, x4, lsl #3
    add x23, x15, x4, lsl #3
    add x5, x8, x4, lsl #3

    .inst 0xe400f4d8  // st1b {z24.b}, p5, [x6]
    .inst 0xe40454c0  // st1b {z0.b}, p5, [x6, x4]
    .inst 0xe400f521  // st1b {z1.b}, p5, [x9]
    .inst 0xe4045522  // st1b {z2.b}, p5, [x9, x4]
    .inst 0xe400f5f9  // st1b {z25.b}, p5, [x15]
    .inst 0xe40455e3  // st1b {z3.b}, p5, [x15, x4]
    .inst 0xe400f504  // st1b {z4.b}, p5, [x8]
    .inst 0xe4045505  // st1b {z5.b}, p5, [x8, x4]
    .inst 0xe400f57a  // st1b {z26.b}, p5, [x11]
    .inst 0xe4045566  // st1b {z6.b}, p5, [x11, x4]
    .inst 0xe400f5a7  // st1b {z7.b}, p5, [x13]
    .inst 0xe40455a8  // st1b {z8.b}, p5, [x13, x4]
    .inst 0xe400f6fb  // st1b {z27.b}, p5, [x23]
    .inst 0xe40456e9  // st1b {z9.b}, p5, [x23, x4]
    .inst 0xe400f4aa  // st1b {z10.b}, p5, [x5]
    .inst 0xe40454ab  // st1b {z11.b}, p5, [x5, x4]

    add x9, x6, x4, lsl #4
    add x15, x13, x4, lsl #3
    add x8, x23, x4, lsl #3
    add x11, x5, x4, lsl #3

    .inst 0xe400f53c  // st1b {z28.b}, p5, [x9]
    .inst 0xe404552c  // st1b {z12.b}, p5, [x9, x4]
    .inst 0xe400f5ed  // st1b {z13.b}, p5, [x15]
    .inst 0xe40455ee  // st1b {z14.b}, p5, [x15, x4]
    .inst 0xe400f51d  // st1b {z29.b}, p5, [x8]
    .inst 0xe404550f  // st1b {z15.b}, p5, [x8, x4]
    .inst 0xe400f570  // st1b {z16.b}, p5, [x11]
    b TILE1_Dz_End

TILE1_STORE104:
    .inst 0x05702300  // dup z0.q, z24.q[1]
    .inst 0x05b02301  // dup z1.q, z24.q[2]
    .inst 0x05f02302  // dup z2.q, z24.q[3]
    .inst 0x05702323  // dup z3.q, z25.q[1]
    .inst 0x05b02324  // dup z4.q, z25.q[2]
    .inst 0x05f02325  // dup z5.q, z25.q[3]
    .inst 0x05702346  // dup z6.q, z26.q[1]
    .inst 0x05b02347  // dup z7.q, z26.q[2]
    .inst 0x05f02348  // dup z8.q, z26.q[3]
    .inst 0x05702369  // dup z9.q, z27.q[1]
    .inst 0x05b0236a  // dup z10.q, z27.q[2]
    .inst 0x05f0236b  // dup z11.q, z27.q[3]
    .inst 0x0570238c  // dup z12.q, z28.q[1]
    .inst 0x05b0238d  // dup z13.q, z28.q[2]
    .inst 0x05f0238e  // dup z14.q, z28.q[3]
    .inst 0x057023af  // dup z15.q, z29.q[1]
    .inst 0x05b023b0  // dup z16.q, z29.q[2]
    .inst 0x05f023b1  // dup z17.q, z29.q[3]

    add x9, x6, x4, lsl #1
    add x15, x6, x4, lsl #2
    add x8, x9, x4, lsl #2
    add x11, x6, x4, lsl #3
    add x13, x9, x4, lsl #3
    add x23, x15, x4, lsl #3
    add x5, x8, x4, lsl #3

    .inst 0xe400f4d8  // st1b {z24.b}, p5, [x6]
    .inst 0xe40454c0  // st1b {z0.b}, p5, [x6, x4]
    .inst 0xe400f521  // st1b {z1.b}, p5, [x9]
    .inst 0xe4045522  // st1b {z2.b}, p5, [x9, x4]
    .inst 0xe400f5f9  // st1b {z25.b}, p5, [x15]
    .inst 0xe40455e3  // st1b {z3.b}, p5, [x15, x4]
    .inst 0xe400f504  // st1b {z4.b}, p5, [x8]
    .inst 0xe4045505  // st1b {z5.b}, p5, [x8, x4]
    .inst 0xe400f57a  // st1b {z26.b}, p5, [x11]
    .inst 0xe4045566  // st1b {z6.b}, p5, [x11, x4]
    .inst 0xe400f5a7  // st1b {z7.b}, p5, [x13]
    .inst 0xe40455a8  // st1b {z8.b}, p5, [x13, x4]
    .inst 0xe400f6fb  // st1b {z27.b}, p5, [x23]
    .inst 0xe40456e9  // st1b {z9.b}, p5, [x23, x4]
    .inst 0xe400f4aa  // st1b {z10.b}, p5, [x5]
    .inst 0xe40454ab  // st1b {z11.b}, p5, [x5, x4]

    add x9, x6, x4, lsl #4
    add x15, x13, x4, lsl #3
    add x8, x23, x4, lsl #3
    add x11, x5, x4, lsl #3

    .inst 0xe400f53c  // st1b {z28.b}, p5, [x9]
    .inst 0xe404552c  // st1b {z12.b}, p5, [x9, x4]
    .inst 0xe400f5ed  // st1b {z13.b}, p5, [x15]
    .inst 0xe40455ee  // st1b {z14.b}, p5, [x15, x4]
    .inst 0xe400f51d  // st1b {z29.b}, p5, [x8]
    .inst 0xe404550f  // st1b {z15.b}, p5, [x8, x4]
    .inst 0xe400f570  // st1b {z16.b}, p5, [x11]
    .inst 0xe4045571  // st1b {z17.b}, p5, [x11, x4]
    b TILE1_Dz_End

TILE1_STORE108:
    .inst 0x05702300  // dup z0.q, z24.q[1]
    .inst 0x05b02301  // dup z1.q, z24.q[2]
    .inst 0x05f02302  // dup z2.q, z24.q[3]
    .inst 0x05702323  // dup z3.q, z25.q[1]
    .inst 0x05b02324  // dup z4.q, z25.q[2]
    .inst 0x05f02325  // dup z5.q, z25.q[3]
    .inst 0x05702346  // dup z6.q, z26.q[1]
    .inst 0x05b02347  // dup z7.q, z26.q[2]
    .inst 0x05f02348  // dup z8.q, z26.q[3]
    .inst 0x05702369  // dup z9.q, z27.q[1]
    .inst 0x05b0236a  // dup z10.q, z27.q[2]
    .inst 0x05f0236b  // dup z11.q, z27.q[3]
    .inst 0x0570238c  // dup z12.q, z28.q[1]
    .inst 0x05b0238d  // dup z13.q, z28.q[2]
    .inst 0x05f0238e  // dup z14.q, z28.q[3]
    .inst 0x057023af  // dup z15.q, z29.q[1]
    .inst 0x05b023b0  // dup z16.q, z29.q[2]
    .inst 0x05f023b1  // dup z17.q, z29.q[3]
    .inst 0x057023d2  // dup z18.q, z30.q[1]

    add x9, x6, x4, lsl #1
    add x15, x6, x4, lsl #2
    add x8, x9, x4, lsl #2
    add x11, x6, x4, lsl #3
    add x13, x9, x4, lsl #3
    add x23, x15, x4, lsl #3
    add x5, x8, x4, lsl #3

    .inst 0xe400f4d8  // st1b {z24.b}, p5, [x6]
    .inst 0xe40454c0  // st1b {z0.b}, p5, [x6, x4]
    .inst 0xe400f521  // st1b {z1.b}, p5, [x9]
    .inst 0xe4045522  // st1b {z2.b}, p5, [x9, x4]
    .inst 0xe400f5f9  // st1b {z25.b}, p5, [x15]
    .inst 0xe40455e3  // st1b {z3.b}, p5, [x15, x4]
    .inst 0xe400f504  // st1b {z4.b}, p5, [x8]
    .inst 0xe4045505  // st1b {z5.b}, p5, [x8, x4]
    .inst 0xe400f57a  // st1b {z26.b}, p5, [x11]
    .inst 0xe4045566  // st1b {z6.b}, p5, [x11, x4]
    .inst 0xe400f5a7  // st1b {z7.b}, p5, [x13]
    .inst 0xe40455a8  // st1b {z8.b}, p5, [x13, x4]
    .inst 0xe400f6fb  // st1b {z27.b}, p5, [x23]
    .inst 0xe40456e9  // st1b {z9.b}, p5, [x23, x4]
    .inst 0xe400f4aa  // st1b {z10.b}, p5, [x5]
    .inst 0xe40454ab  // st1b {z11.b}, p5, [x5, x4]

    add x9, x6, x4, lsl #4
    add x15, x13, x4, lsl #3
    add x8, x23, x4, lsl #3
    add x11, x5, x4, lsl #3
    add x13, x9, x4, lsl #3

    .inst 0xe400f53c  // st1b {z28.b}, p5, [x9]
    .inst 0xe404552c  // st1b {z12.b}, p5, [x9, x4]
    .inst 0xe400f5ed  // st1b {z13.b}, p5, [x15]
    .inst 0xe40455ee  // st1b {z14.b}, p5, [x15, x4]
    .inst 0xe400f51d  // st1b {z29.b}, p5, [x8]
    .inst 0xe404550f  // st1b {z15.b}, p5, [x8, x4]
    .inst 0xe400f570  // st1b {z16.b}, p5, [x11]
    .inst 0xe4045571  // st1b {z17.b}, p5, [x11, x4]
    .inst 0xe400f5be  // st1b {z30.b}, p5, [x13]
    .inst 0xe40455b2  // st1b {z18.b}, p5, [x13, x4]
    b TILE1_Dz_End

    TILE1_STORE112:
    .inst 0x05702300  // dup z0.q, z24.q[1]
    .inst 0x05b02301  // dup z1.q, z24.q[2]
    .inst 0x05f02302  // dup z2.q, z24.q[3]
    .inst 0x05702323  // dup z3.q, z25.q[1]
    .inst 0x05b02324  // dup z4.q, z25.q[2]
    .inst 0x05f02325  // dup z5.q, z25.q[3]
    .inst 0x05702346  // dup z6.q, z26.q[1]
    .inst 0x05b02347  // dup z7.q, z26.q[2]
    .inst 0x05f02348  // dup z8.q, z26.q[3]
    .inst 0x05702369  // dup z9.q, z27.q[1]
    .inst 0x05b0236a  // dup z10.q, z27.q[2]
    .inst 0x05f0236b  // dup z11.q, z27.q[3]
    .inst 0x0570238c  // dup z12.q, z28.q[1]
    .inst 0x05b0238d  // dup z13.q, z28.q[2]
    .inst 0x05f0238e  // dup z14.q, z28.q[3]
    .inst 0x057023af  // dup z15.q, z29.q[1]
    .inst 0x05b023b0  // dup z16.q, z29.q[2]
    .inst 0x05f023b1  // dup z17.q, z29.q[3]
    .inst 0x057023d2  // dup z18.q, z30.q[1]
    .inst 0x05b023d3  // dup z19.q, z30.q[2]

    add x9, x6, x4, lsl #1
    add x15, x6, x4, lsl #2
    add x8, x9, x4, lsl #2
    add x11, x6, x4, lsl #3
    add x13, x9, x4, lsl #3
    add x23, x15, x4, lsl #3
    add x5, x8, x4, lsl #3

    .inst 0xe400f4d8  // st1b {z24.b}, p5, [x6]
    .inst 0xe40454c0  // st1b {z0.b}, p5, [x6, x4]
    .inst 0xe400f521  // st1b {z1.b}, p5, [x9]
    .inst 0xe4045522  // st1b {z2.b}, p5, [x9, x4]
    .inst 0xe400f5f9  // st1b {z25.b}, p5, [x15]
    .inst 0xe40455e3  // st1b {z3.b}, p5, [x15, x4]
    .inst 0xe400f504  // st1b {z4.b}, p5, [x8]
    .inst 0xe4045505  // st1b {z5.b}, p5, [x8, x4]
    .inst 0xe400f57a  // st1b {z26.b}, p5, [x11]
    .inst 0xe4045566  // st1b {z6.b}, p5, [x11, x4]
    .inst 0xe400f5a7  // st1b {z7.b}, p5, [x13]
    .inst 0xe40455a8  // st1b {z8.b}, p5, [x13, x4]
    .inst 0xe400f6fb  // st1b {z27.b}, p5, [x23]
    .inst 0xe40456e9  // st1b {z9.b}, p5, [x23, x4]
    .inst 0xe400f4aa  // st1b {z10.b}, p5, [x5]
    .inst 0xe40454ab  // st1b {z11.b}, p5, [x5, x4]

    add x9, x6, x4, lsl #4
    add x15, x13, x4, lsl #3
    add x8, x23, x4, lsl #3
    add x11, x5, x4, lsl #3
    add x13, x9, x4, lsl #3
    add x23, x15, x4, lsl #3

    .inst 0xe400f53c  // st1b {z28.b}, p5, [x9]
    .inst 0xe404552c  // st1b {z12.b}, p5, [x9, x4]
    .inst 0xe400f5ed  // st1b {z13.b}, p5, [x15]
    .inst 0xe40455ee  // st1b {z14.b}, p5, [x15, x4]
    .inst 0xe400f51d  // st1b {z29.b}, p5, [x8]
    .inst 0xe404550f  // st1b {z15.b}, p5, [x8, x4]
    .inst 0xe400f570  // st1b {z16.b}, p5, [x11]
    .inst 0xe4045571  // st1b {z17.b}, p5, [x11, x4]
    .inst 0xe400f5be  // st1b {z30.b}, p5, [x13]
    .inst 0xe40455b2  // st1b {z18.b}, p5, [x13, x4]
    .inst 0xe400f6f3  // st1b {z19.b}, p5, [x23]
    b TILE1_Dz_End

    TILE1_STORE116:
    .inst 0x05702300  // dup z0.q, z24.q[1]
    .inst 0x05b02301  // dup z1.q, z24.q[2]
    .inst 0x05f02302  // dup z2.q, z24.q[3]
    .inst 0x05702323  // dup z3.q, z25.q[1]
    .inst 0x05b02324  // dup z4.q, z25.q[2]
    .inst 0x05f02325  // dup z5.q, z25.q[3]
    .inst 0x05702346  // dup z6.q, z26.q[1]
    .inst 0x05b02347  // dup z7.q, z26.q[2]
    .inst 0x05f02348  // dup z8.q, z26.q[3]
    .inst 0x05702369  // dup z9.q, z27.q[1]
    .inst 0x05b0236a  // dup z10.q, z27.q[2]
    .inst 0x05f0236b  // dup z11.q, z27.q[3]
    .inst 0x0570238c  // dup z12.q, z28.q[1]
    .inst 0x05b0238d  // dup z13.q, z28.q[2]
    .inst 0x05f0238e  // dup z14.q, z28.q[3]
    .inst 0x057023af  // dup z15.q, z29.q[1]
    .inst 0x05b023b0  // dup z16.q, z29.q[2]
    .inst 0x05f023b1  // dup z17.q, z29.q[3]
    .inst 0x057023d2  // dup z18.q, z30.q[1]
    .inst 0x05b023d3  // dup z19.q, z30.q[2]
    .inst 0x05f023d4  // dup z20.q, z30.q[3]

    add x9, x6, x4, lsl #1
    add x15, x6, x4, lsl #2
    add x8, x9, x4, lsl #2
    add x11, x6, x4, lsl #3
    add x13, x9, x4, lsl #3
    add x23, x15, x4, lsl #3
    add x5, x8, x4, lsl #3

    .inst 0xe400f4d8  // st1b {z24.b}, p5, [x6]
    .inst 0xe40454c0  // st1b {z0.b}, p5, [x6, x4]
    .inst 0xe400f521  // st1b {z1.b}, p5, [x9]
    .inst 0xe4045522  // st1b {z2.b}, p5, [x9, x4]
    .inst 0xe400f5f9  // st1b {z25.b}, p5, [x15]
    .inst 0xe40455e3  // st1b {z3.b}, p5, [x15, x4]
    .inst 0xe400f504  // st1b {z4.b}, p5, [x8]
    .inst 0xe4045505  // st1b {z5.b}, p5, [x8, x4]
    .inst 0xe400f57a  // st1b {z26.b}, p5, [x11]
    .inst 0xe4045566  // st1b {z6.b}, p5, [x11, x4]
    .inst 0xe400f5a7  // st1b {z7.b}, p5, [x13]
    .inst 0xe40455a8  // st1b {z8.b}, p5, [x13, x4]
    .inst 0xe400f6fb  // st1b {z27.b}, p5, [x23]
    .inst 0xe40456e9  // st1b {z9.b}, p5, [x23, x4]
    .inst 0xe400f4aa  // st1b {z10.b}, p5, [x5]
    .inst 0xe40454ab  // st1b {z11.b}, p5, [x5, x4]

    add x9, x6, x4, lsl #4
    add x15, x13, x4, lsl #3
    add x8, x23, x4, lsl #3
    add x11, x5, x4, lsl #3
    add x13, x9, x4, lsl #3
    add x23, x15, x4, lsl #3
    add x5, x8, x4, lsl #3

    .inst 0xe400f53c  // st1b {z28.b}, p5, [x9]
    .inst 0xe404552c  // st1b {z12.b}, p5, [x9, x4]
    .inst 0xe400f5ed  // st1b {z13.b}, p5, [x15]
    .inst 0xe40455ee  // st1b {z14.b}, p5, [x15, x4]
    .inst 0xe400f51d  // st1b {z29.b}, p5, [x8]
    .inst 0xe404550f  // st1b {z15.b}, p5, [x8, x4]
    .inst 0xe400f570  // st1b {z16.b}, p5, [x11]
    .inst 0xe4045571  // st1b {z17.b}, p5, [x11, x4]
    .inst 0xe400f5be  // st1b {z30.b}, p5, [x13]
    .inst 0xe40455b2  // st1b {z18.b}, p5, [x13, x4]
    .inst 0xe400f6f3  // st1b {z19.b}, p5, [x23]
    .inst 0xe40456f4  // st1b {z20.b}, p5, [x23, x4]
    .inst 0xe400f4bf  // st1b {z31.b}, p5, [x5]
    b TILE1_Dz_End

    TILE1_STORE120:
    .inst 0x05702300  // dup z0.q, z24.q[1]
    .inst 0x05b02301  // dup z1.q, z24.q[2]
    .inst 0x05f02302  // dup z2.q, z24.q[3]
    .inst 0x05702323  // dup z3.q, z25.q[1]
    .inst 0x05b02324  // dup z4.q, z25.q[2]
    .inst 0x05f02325  // dup z5.q, z25.q[3]
    .inst 0x05702346  // dup z6.q, z26.q[1]
    .inst 0x05b02347  // dup z7.q, z26.q[2]
    .inst 0x05f02348  // dup z8.q, z26.q[3]
    .inst 0x05702369  // dup z9.q, z27.q[1]
    .inst 0x05b0236a  // dup z10.q, z27.q[2]
    .inst 0x05f0236b  // dup z11.q, z27.q[3]
    .inst 0x0570238c  // dup z12.q, z28.q[1]
    .inst 0x05b0238d  // dup z13.q, z28.q[2]
    .inst 0x05f0238e  // dup z14.q, z28.q[3]
    .inst 0x057023af  // dup z15.q, z29.q[1]
    .inst 0x05b023b0  // dup z16.q, z29.q[2]
    .inst 0x05f023b1  // dup z17.q, z29.q[3]
    .inst 0x057023d2  // dup z18.q, z30.q[1]
    .inst 0x05b023d3  // dup z19.q, z30.q[2]
    .inst 0x05f023d4  // dup z20.q, z30.q[3]
    .inst 0x057023f5  // dup z21.q, z31.q[1]

    add x9, x6, x4, lsl #1
    add x15, x6, x4, lsl #2
    add x8, x9, x4, lsl #2
    add x11, x6, x4, lsl #3
    add x13, x9, x4, lsl #3
    add x23, x15, x4, lsl #3
    add x5, x8, x4, lsl #3

    .inst 0xe400f4d8  // st1b {z24.b}, p5, [x6]
    .inst 0xe40454c0  // st1b {z0.b}, p5, [x6, x4]
    .inst 0xe400f521  // st1b {z1.b}, p5, [x9]
    .inst 0xe4045522  // st1b {z2.b}, p5, [x9, x4]
    .inst 0xe400f5f9  // st1b {z25.b}, p5, [x15]
    .inst 0xe40455e3  // st1b {z3.b}, p5, [x15, x4]
    .inst 0xe400f504  // st1b {z4.b}, p5, [x8]
    .inst 0xe4045505  // st1b {z5.b}, p5, [x8, x4]
    .inst 0xe400f57a  // st1b {z26.b}, p5, [x11]
    .inst 0xe4045566  // st1b {z6.b}, p5, [x11, x4]
    .inst 0xe400f5a7  // st1b {z7.b}, p5, [x13]
    .inst 0xe40455a8  // st1b {z8.b}, p5, [x13, x4]
    .inst 0xe400f6fb  // st1b {z27.b}, p5, [x23]
    .inst 0xe40456e9  // st1b {z9.b}, p5, [x23, x4]
    .inst 0xe400f4aa  // st1b {z10.b}, p5, [x5]
    .inst 0xe40454ab  // st1b {z11.b}, p5, [x5, x4]

    add x9, x6, x4, lsl #4
    add x15, x13, x4, lsl #3
    add x8, x23, x4, lsl #3
    add x11, x5, x4, lsl #3
    add x13, x9, x4, lsl #3
    add x23, x15, x4, lsl #3
    add x5, x8, x4, lsl #3

    .inst 0xe400f53c  // st1b {z28.b}, p5, [x9]
    .inst 0xe404552c  // st1b {z12.b}, p5, [x9, x4]
    .inst 0xe400f5ed  // st1b {z13.b}, p5, [x15]
    .inst 0xe40455ee  // st1b {z14.b}, p5, [x15, x4]
    .inst 0xe400f51d  // st1b {z29.b}, p5, [x8]
    .inst 0xe404550f  // st1b {z15.b}, p5, [x8, x4]
    .inst 0xe400f570  // st1b {z16.b}, p5, [x11]
    .inst 0xe4045571  // st1b {z17.b}, p5, [x11, x4]
    .inst 0xe400f5be  // st1b {z30.b}, p5, [x13]
    .inst 0xe40455b2  // st1b {z18.b}, p5, [x13, x4]
    .inst 0xe400f6f3  // st1b {z19.b}, p5, [x23]
    .inst 0xe40456f4  // st1b {z20.b}, p5, [x23, x4]
    .inst 0xe400f4bf  // st1b {z31.b}, p5, [x5]
    .inst 0xe40454b5  // st1b {z21.b}, p5, [x5, x4]
    b TILE1_Dz_End

    TILE1_STORE124:
    .inst 0x05702300  // dup z0.q, z24.q[1]
    .inst 0x05b02301  // dup z1.q, z24.q[2]
    .inst 0x05f02302  // dup z2.q, z24.q[3]
    .inst 0x05702323  // dup z3.q, z25.q[1]
    .inst 0x05b02324  // dup z4.q, z25.q[2]
    .inst 0x05f02325  // dup z5.q, z25.q[3]
    .inst 0x05702346  // dup z6.q, z26.q[1]
    .inst 0x05b02347  // dup z7.q, z26.q[2]
    .inst 0x05f02348  // dup z8.q, z26.q[3]
    .inst 0x05702369  // dup z9.q, z27.q[1]
    .inst 0x05b0236a  // dup z10.q, z27.q[2]
    .inst 0x05f0236b  // dup z11.q, z27.q[3]
    .inst 0x0570238c  // dup z12.q, z28.q[1]
    .inst 0x05b0238d  // dup z13.q, z28.q[2]
    .inst 0x05f0238e  // dup z14.q, z28.q[3]
    .inst 0x057023af  // dup z15.q, z29.q[1]
    .inst 0x05b023b0  // dup z16.q, z29.q[2]
    .inst 0x05f023b1  // dup z17.q, z29.q[3]
    .inst 0x057023d2  // dup z18.q, z30.q[1]
    .inst 0x05b023d3  // dup z19.q, z30.q[2]
    .inst 0x05f023d4  // dup z20.q, z30.q[3]
    .inst 0x057023f5  // dup z21.q, z31.q[1]
    .inst 0x05b023f6  // dup z22.q, z31.q[2]

    add x9, x6, x4, lsl #1 // +2
    add x15, x6, x4, lsl #2 // +4
    add x8, x9, x4, lsl #2 // +6
    add x11, x6, x4, lsl #3 // +8
    add x13, x9, x4, lsl #3 // +10
    add x23, x15, x4, lsl #3 // +12
    add x5, x8, x4, lsl #3   // +14

    .inst 0xe400f4d8  // st1b {z24.b}, p5, [x6]
    .inst 0xe40454c0  // st1b {z0.b}, p5, [x6, x4]
    .inst 0xe400f521  // st1b {z1.b}, p5, [x9]
    .inst 0xe4045522  // st1b {z2.b}, p5, [x9, x4]
    .inst 0xe400f5f9  // st1b {z25.b}, p5, [x15]
    .inst 0xe40455e3  // st1b {z3.b}, p5, [x15, x4]
    .inst 0xe400f504  // st1b {z4.b}, p5, [x8]
    .inst 0xe4045505  // st1b {z5.b}, p5, [x8, x4]
    .inst 0xe400f57a  // st1b {z26.b}, p5, [x11]
    .inst 0xe4045566  // st1b {z6.b}, p5, [x11, x4]
    .inst 0xe400f5a7  // st1b {z7.b}, p5, [x13]
    .inst 0xe40455a8  // st1b {z8.b}, p5, [x13, x4]
    .inst 0xe400f6fb  // st1b {z27.b}, p5, [x23]
    .inst 0xe40456e9  // st1b {z9.b}, p5, [x23, x4]
    .inst 0xe400f4aa  // st1b {z10.b}, p5, [x5]
    .inst 0xe40454ab  // st1b {z11.b}, p5, [x5, x4]

    add x9, x6, x4, lsl #4 // +16
    add x15, x13, x4, lsl #3 // +18
    add x8, x23, x4, lsl #3 // +20
    add x11, x5, x4, lsl #3 // +22
    add x13, x9, x4, lsl #3 // +24
    add x23, x15, x4, lsl #3 // +26
    add x5, x8, x4, lsl #3   // +28
    add x12, x11, x4, lsl #3 // +30

    .inst 0xe400f53c  // st1b {z28.b}, p5, [x9]
    .inst 0xe404552c  // st1b {z12.b}, p5, [x9, x4]
    .inst 0xe400f5ed  // st1b {z13.b}, p5, [x15]
    .inst 0xe40455ee  // st1b {z14.b}, p5, [x15, x4]
    .inst 0xe400f51d  // st1b {z29.b}, p5, [x8]
    .inst 0xe404550f  // st1b {z15.b}, p5, [x8, x4]
    .inst 0xe400f570  // st1b {z16.b}, p5, [x11]
    .inst 0xe4045571  // st1b {z17.b}, p5, [x11, x4]
    .inst 0xe400f5be  // st1b {z30.b}, p5, [x13]
    .inst 0xe40455b2  // st1b {z18.b}, p5, [x13, x4]
    .inst 0xe400f6f3  // st1b {z19.b}, p5, [x23]
    .inst 0xe40456f4  // st1b {z20.b}, p5, [x23, x4]
    .inst 0xe400f4bf  // st1b {z31.b}, p5, [x5]
    .inst 0xe40454b5  // st1b {z21.b}, p5, [x5, x4]
    .inst 0xe400f596  // st1b {z22.b}, p5, [x12]
    b TILE1_Dz_End

    TILE1_STORE128:
    .inst 0x05702300  // dup z0.q, z24.q[1]
    .inst 0x05b02301  // dup z1.q, z24.q[2]
    .inst 0x05f02302  // dup z2.q, z24.q[3]
    .inst 0x05702323  // dup z3.q, z25.q[1]
    .inst 0x05b02324  // dup z4.q, z25.q[2]
    .inst 0x05f02325  // dup z5.q, z25.q[3]
    .inst 0x05702346  // dup z6.q, z26.q[1]
    .inst 0x05b02347  // dup z7.q, z26.q[2]
    .inst 0x05f02348  // dup z8.q, z26.q[3]
    .inst 0x05702369  // dup z9.q, z27.q[1]
    .inst 0x05b0236a  // dup z10.q, z27.q[2]
    .inst 0x05f0236b  // dup z11.q, z27.q[3]
    .inst 0x0570238c  // dup z12.q, z28.q[1]
    .inst 0x05b0238d  // dup z13.q, z28.q[2]
    .inst 0x05f0238e  // dup z14.q, z28.q[3]
    .inst 0x057023af  // dup z15.q, z29.q[1]
    .inst 0x05b023b0  // dup z16.q, z29.q[2]
    .inst 0x05f023b1  // dup z17.q, z29.q[3]
    .inst 0x057023d2  // dup z18.q, z30.q[1]
    .inst 0x05b023d3  // dup z19.q, z30.q[2]
    .inst 0x05f023d4  // dup z20.q, z30.q[3]
    .inst 0x057023f5  // dup z21.q, z31.q[1]
    .inst 0x05b023f6  // dup z22.q, z31.q[2]
    .inst 0x05f023f7  // dup z23.q, z31.q[3]

    add x9, x6, x4, lsl #1 // +2
    add x15, x6, x4, lsl #2 // +4
    add x8, x9, x4, lsl #2 // +6
    add x11, x6, x4, lsl #3 // +8
    add x13, x9, x4, lsl #3 // +10
    add x23, x15, x4, lsl #3 // +12
    add x5, x8, x4, lsl #3   // +14

    .inst 0xe400f4d8  // st1b {z24.b}, p5, [x6]
    .inst 0xe40454c0  // st1b {z0.b}, p5, [x6, x4]
    .inst 0xe400f521  // st1b {z1.b}, p5, [x9]
    .inst 0xe4045522  // st1b {z2.b}, p5, [x9, x4]
    .inst 0xe400f5f9  // st1b {z25.b}, p5, [x15]
    .inst 0xe40455e3  // st1b {z3.b}, p5, [x15, x4]
    .inst 0xe400f504  // st1b {z4.b}, p5, [x8]
    .inst 0xe4045505  // st1b {z5.b}, p5, [x8, x4]
    .inst 0xe400f57a  // st1b {z26.b}, p5, [x11]
    .inst 0xe4045566  // st1b {z6.b}, p5, [x11, x4]
    .inst 0xe400f5a7  // st1b {z7.b}, p5, [x13]
    .inst 0xe40455a8  // st1b {z8.b}, p5, [x13, x4]
    .inst 0xe400f6fb  // st1b {z27.b}, p5, [x23]
    .inst 0xe40456e9  // st1b {z9.b}, p5, [x23, x4]
    .inst 0xe400f4aa  // st1b {z10.b}, p5, [x5]
    .inst 0xe40454ab  // st1b {z11.b}, p5, [x5, x4]

    add x9, x6, x4, lsl #4 // +16
    add x15, x13, x4, lsl #3 // +18
    add x8, x23, x4, lsl #3 // +20
    add x11, x5, x4, lsl #3 // +22
    add x13, x9, x4, lsl #3 // +24
    add x23, x15, x4, lsl #3 // +26
    add x5, x8, x4, lsl #3   // +28
    add x12, x11, x4, lsl #3 // +30

    .inst 0xe400f53c  // st1b {z28.b}, p5, [x9]
    .inst 0xe404552c  // st1b {z12.b}, p5, [x9, x4]
    .inst 0xe400f5ed  // st1b {z13.b}, p5, [x15]
    .inst 0xe40455ee  // st1b {z14.b}, p5, [x15, x4]
    .inst 0xe400f51d  // st1b {z29.b}, p5, [x8]
    .inst 0xe404550f  // st1b {z15.b}, p5, [x8, x4]
    .inst 0xe400f570  // st1b {z16.b}, p5, [x11]
    .inst 0xe4045571  // st1b {z17.b}, p5, [x11, x4]
    .inst 0xe400f5be  // st1b {z30.b}, p5, [x13]
    .inst 0xe40455b2  // st1b {z18.b}, p5, [x13, x4]
    .inst 0xe400f6f3  // st1b {z19.b}, p5, [x23]
    .inst 0xe40456f4  // st1b {z20.b}, p5, [x23, x4]
    .inst 0xe400f4bf  // st1b {z31.b}, p5, [x5]
    .inst 0xe40454b5  // st1b {z21.b}, p5, [x5, x4]
    .inst 0xe400f596  // st1b {z22.b}, p5, [x12]
    .inst 0xe4045597  // st1b {z23.b}, p5, [x12, x4]
    b TILE1_Dz_End


    TILE1_Dz_End:
    sub x24, x24, #32
    cmp x24, #0
    ble End
    add x6, x6, x4, LSL #5
    mov x13, x19
    mov x23, x21
    cbz x27, LoopDz_TILE1
    REVERT_INPUT_DEQUANT_BIAS x27, x15, x26, x25
    b LoopDz_TILE1

 
End:
.inst 0xd503467f  // smstop
 
ldp x19, x20, [sp, #224]
ldp x21, x22, [sp, #208]
ldp x23, x24, [sp, #192]
ldp x25, x26, [sp, #176]
ldp x27, x28, [sp, #160]
ldp d8, d9,   [sp, #80]
ldp d10, d11, [sp, #64]
ldp d12, d13, [sp, #48]
ldp d14, d15, [sp, #32]
ldp x29, x30, [sp], #320
ret
 
#endif // __aarch64__
