//  MNNGemmInt8AddBiasScaleHp128_SME2_w8_Fp32.S
//  Created by MNN on 2022/09/26.
//  Copyright © 2018, Alibaba Group Holding Limited

#if defined(__aarch64__)
#include "MNNAsmGlobal.h"

.text

.macro REVERT_INPUT_DEQUANT_BIAS rg0, rg1, rg2, rg3
mul \rg1, \rg2, \rg3
sub \rg0, \rg0, \rg1
.endm


asm_function MNNGemmInt8AddBiasScaleHp128_SME2_w8_Fp32
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
//void MNNGemmInt8AddBiasScaleHp128_SME2_w8_Fp32(int8_t* dst, const int8_t* src,
//    const int8_t* weight, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad,
// const QuanPostTreatParameters* parameters, size_t realDstCount);

//Auto: x0:dst, x1:src, x2:weight, x3:src_depth_quad, x4:dst_step x6:dst_depth_quad, x0: parameters, x7: realDstCount
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

/* initialize predicates */
mov x22, #48
.inst 0x2518e080  // ptrue p0.b, #4            // first 4 bytes
.inst 0x2518e125  // ptrue p5.b, vl16          // first 16 bytes
.inst 0x25207810  // ptrue pn8.b
.inst 0x2598e3e1  // ptrue p1.s


lsl x22, x7, #2 // eDest * GEMM_INT8_SRC_UNIT


TILE_1:
    cmp x7, #1
    blt End

    mov x20, x9 // bias
    mov x19, x13      // input kernel sum
    mov x21, x23     // input dequant scale
    mov x24, x27

LoopDz_TILE1:
    // load bias
    lsl x10, x5, #2
    mov x11, #64
    .inst 0x25aa67f1  // whilelt pn9.s, xzr, x10, vlx4
    .inst 0x25aa6572  // whilelt pn10.s, x11, x10, vlx4
    .inst 0xa040c698  // ld1w {z24.s-z27.s}, pn9/z, [x20]
    .inst 0xa041ca9c  // ld1w {z28.s-z31.s}, pn10/z, [x20, #4, MUL VL]
    .inst 0x04345114  // addvl x20, x20, #8

    mov x11, x1             // src
    mov x15, x26
TILE1_BLOCKNUM:
    mov x10, x3             // src_depth_quad

    .inst 0xc00800ff  // zero {za}
    mov w8, #0
    LoopSz_TILE_1_lu1:
        .inst 0xa400216c  // ld1rqb {z12.b}, p0/z, [x11]        // src
        .inst 0xa0408044  // ld1b {z4.b-z7.b}, pn8/z, [x2]        // weight
        .inst 0xa0418048  // ld1b {z8.b-z11.b}, pn8/z, [x2, #4, MUL VL]
        .inst 0xc15c90a0  // sdot za.s[w8, 0, VGx4], {z4.b-z7.b}, z12.b[0]
        .inst 0xc15c9124  // sdot za.s[w8, 4, VGx4], {z8.b-z11.b}, z12.b[0]
        subs x10, x10, #1
        add x11, x11, x22
        .inst 0x04225102  // addvl x2, x2, #8

    bne LoopSz_TILE_1_lu1


LoopSzEnd_TILE_1:
    .inst 0xa0408040  // ld1b {z0.b-z3.b}, pn8/z, [x2]     // weight scale: 64*sizeof(float)
    .inst 0xa0418050  // ld1b {z16.b-z19.b}, pn8/z, [x2, #4, MUL VL]    // weight scale
    .inst 0xa0428048  // ld1b {z8.b-z11.b}, pn8/z, [x2, #8, MUL VL]    // weight bias
    .inst 0xa043804c  // ld1b {z12.b-z15.b}, pn8/z, [x2, #12, MUL VL]    // weight bias

    .inst 0x8540c5a4  // ld1rw {z4.s}, p1/z, [x13]     // x kernel sum
    .inst 0x8540c6f7  // ld1rw {z23.s}, p1/z, [x23]    // x scale

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

    .inst 0x6594a508  // scvtf z8.s, p1/m, z8.s
    .inst 0x6594a529  // scvtf z9.s, p1/m, z9.s
    .inst 0x6594a54a  // scvtf z10.s, p1/m, z10.s
    .inst 0x6594a56b  // scvtf z11.s, p1/m, z11.s
    .inst 0x6594a58c  // scvtf z12.s, p1/m, z12.s
    .inst 0x6594a5ad  // scvtf z13.s, p1/m, z13.s
    .inst 0x6594a5ce  // scvtf z14.s, p1/m, z14.s
    .inst 0x6594a5ef  // scvtf z15.s, p1/m, z15.s

    // accumulate
    .inst 0x65a00518  // fmla z24.s, p1/m, z8.s, z0.s
    .inst 0x65a10539  // fmla z25.s, p1/m, z9.s, z1.s
    .inst 0x65a2055a  // fmla z26.s, p1/m, z10.s, z2.s
    .inst 0x65a3057b  // fmla z27.s, p1/m, z11.s, z3.s
    .inst 0x65b0059c  // fmla z28.s, p1/m, z12.s, z16.s
    .inst 0x65b105bd  // fmla z29.s, p1/m, z13.s, z17.s
    .inst 0x65b205de  // fmla z30.s, p1/m, z14.s, z18.s
    .inst 0x65b305ff  // fmla z31.s, p1/m, z15.s, z19.s

    .inst 0x04225202  // addvl x2, x2, #16  // weight ptr
    add x13, x13, x22

    cbz x27, TILE1_ADD_DSTV
    .inst 0x8540c764  // ld1rw {z4.s}, p1/z, [x27]    // input dequant bias
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
    add x27, x27, x22
    add x23, x23, x22
    .inst 0x043c511c  // addvl x28, x28, #8

    TILE1_ADD_DSTV:
    subs x15, x15, #1
    bne TILE1_BLOCKNUM

    TILE1_STORE:
    .inst 0x8540c5c8  // ld1rw {z8.s}, p1/z, [x14]
    .inst 0x8541c5c9  // ld1rw {z9.s}, p1/z, [x14, #4]
    .inst 0xc1a9c918  // fclamp {z24.s-z27.s}, z8.s, z9.s
    .inst 0xc1a9c91c  // fclamp {z28.s-z31.s}, z8.s, z9.s

    cmp x5, #32
    bge TILE1_STORE128
    cmp x5, #31
    beq TILE1_STORE124
    cmp x5, #30
    beq TILE1_STORE120
    cmp x5, #29
    beq TILE1_STORE116
    cmp x5, #28
    beq TILE1_STORE112
    cmp x5, #27
    beq TILE1_STORE108
    cmp x5, #26
    beq TILE1_STORE104
    cmp x5, #25
    beq TILE1_STORE100
    cmp x5, #24
    beq TILE1_STORE96
    cmp x5, #23
    beq TILE1_STORE92
    cmp x5, #22
    beq TILE1_STORE88
    cmp x5, #21
    beq TILE1_STORE84
    cmp x5, #20
    beq TILE1_STORE80
    cmp x5, #19
    beq TILE1_STORE76
    cmp x5, #18
    beq TILE1_STORE72
    cmp x5, #17
    beq TILE1_STORE68
    cmp x5, #16
    beq TILE1_STORE64
    cmp x5, #15
    beq TILE1_STORE60
    cmp x5, #14
    beq TILE1_STORE56
    cmp x5, #13
    beq TILE1_STORE52
    cmp x5, #12
    beq TILE1_STORE48
    cmp x5, #11
    beq TILE1_STORE44
    cmp x5, #10
    beq TILE1_STORE40
    cmp x5, #9
    beq TILE1_STORE36
    cmp x5, #8
    beq TILE1_STORE32
    cmp x5, #7
    beq TILE1_STORE28
    cmp x5, #6
    beq TILE1_STORE24
    cmp x5, #5
    beq TILE1_STORE20
    cmp x5, #4
    beq TILE1_STORE16
    cmp x5, #3
    beq TILE1_STORE12
    cmp x5, #2
    beq TILE1_STORE8

    TILE1_STORE4:
    .inst 0xe400f418  // st1b {z24.b}, p5, [x0]
    b End

    TILE1_STORE8:
    .inst 0x05702300  // dup z0.q, z24.q[1]

    .inst 0xe400f418  // st1b {z24.b}, p5, [x0]
    .inst 0xe4045400  // st1b {z0.b}, p5, [x0, x4]
    b End

    TILE1_STORE12:
    .inst 0x05702300  // dup z0.q, z24.q[1]
    .inst 0x05b02301  // dup z1.q, z24.q[2]

    add x9, x0, x4, lsl #1

    .inst 0xe400f418  // st1b {z24.b}, p5, [x0]
    .inst 0xe4045400  // st1b {z0.b}, p5, [x0, x4]
    .inst 0xe400f521  // st1b {z1.b}, p5, [x9]
    b End

    TILE1_STORE16:
    .inst 0x05702300  // dup z0.q, z24.q[1]
    .inst 0x05b02301  // dup z1.q, z24.q[2]
    .inst 0x05f02302  // dup z2.q, z24.q[3]

    add x9, x0, x4, lsl #1

    .inst 0xe400f418  // st1b {z24.b}, p5, [x0]
    .inst 0xe4045400  // st1b {z0.b}, p5, [x0, x4]
    .inst 0xe400f521  // st1b {z1.b}, p5, [x9]
    .inst 0xe4045522  // st1b {z2.b}, p5, [x9, x4]
    b End

    TILE1_STORE20:
    .inst 0x05702300  // dup z0.q, z24.q[1]
    .inst 0x05b02301  // dup z1.q, z24.q[2]
    .inst 0x05f02302  // dup z2.q, z24.q[3]

    add x9, x0, x4, lsl #1
    add x15, x0, x4, lsl #2

    .inst 0xe400f418  // st1b {z24.b}, p5, [x0]
    .inst 0xe4045400  // st1b {z0.b}, p5, [x0, x4]
    .inst 0xe400f521  // st1b {z1.b}, p5, [x9]
    .inst 0xe4045522  // st1b {z2.b}, p5, [x9, x4]
    .inst 0xe400f5f9  // st1b {z25.b}, p5, [x15]
    b End

    TILE1_STORE24:
    .inst 0x05702300  // dup z0.q, z24.q[1]
    .inst 0x05b02301  // dup z1.q, z24.q[2]
    .inst 0x05f02302  // dup z2.q, z24.q[3]
    .inst 0x05702323  // dup z3.q, z25.q[1]

    add x9, x0, x4, lsl #1
    add x15, x0, x4, lsl #2

    .inst 0xe400f418  // st1b {z24.b}, p5, [x0]
    .inst 0xe4045400  // st1b {z0.b}, p5, [x0, x4]
    .inst 0xe400f521  // st1b {z1.b}, p5, [x9]
    .inst 0xe4045522  // st1b {z2.b}, p5, [x9, x4]
    .inst 0xe400f5f9  // st1b {z25.b}, p5, [x15]
    .inst 0xe40455e3  // st1b {z3.b}, p5, [x15, x4]
    b End

    TILE1_STORE28:
    .inst 0x05702300  // dup z0.q, z24.q[1]
    .inst 0x05b02301  // dup z1.q, z24.q[2]
    .inst 0x05f02302  // dup z2.q, z24.q[3]
    .inst 0x05702323  // dup z3.q, z25.q[1]
    .inst 0x05b02324  // dup z4.q, z25.q[2]

    add x9, x0, x4, lsl #1
    add x15, x0, x4, lsl #2
    add x8, x9, x4, lsl #2

    .inst 0xe400f418  // st1b {z24.b}, p5, [x0]
    .inst 0xe4045400  // st1b {z0.b}, p5, [x0, x4]
    .inst 0xe400f521  // st1b {z1.b}, p5, [x9]
    .inst 0xe4045522  // st1b {z2.b}, p5, [x9, x4]
    .inst 0xe400f5f9  // st1b {z25.b}, p5, [x15]
    .inst 0xe40455e3  // st1b {z3.b}, p5, [x15, x4]
    .inst 0xe400f504  // st1b {z4.b}, p5, [x8]
    b End

    TILE1_STORE32:
    .inst 0x05702300  // dup z0.q, z24.q[1]
    .inst 0x05b02301  // dup z1.q, z24.q[2]
    .inst 0x05f02302  // dup z2.q, z24.q[3]
    .inst 0x05702323  // dup z3.q, z25.q[1]
    .inst 0x05b02324  // dup z4.q, z25.q[2]
    .inst 0x05f02325  // dup z5.q, z25.q[3]

    add x9, x0, x4, lsl #1
    add x15, x0, x4, lsl #2
    add x8, x9, x4, lsl #2

    .inst 0xe400f418  // st1b {z24.b}, p5, [x0]
    .inst 0xe4045400  // st1b {z0.b}, p5, [x0, x4]
    .inst 0xe400f521  // st1b {z1.b}, p5, [x9]
    .inst 0xe4045522  // st1b {z2.b}, p5, [x9, x4]
    .inst 0xe400f5f9  // st1b {z25.b}, p5, [x15]
    .inst 0xe40455e3  // st1b {z3.b}, p5, [x15, x4]
    .inst 0xe400f504  // st1b {z4.b}, p5, [x8]
    .inst 0xe4045505  // st1b {z5.b}, p5, [x8, x4]
    b End

    TILE1_STORE36:
    .inst 0x05702300  // dup z0.q, z24.q[1]
    .inst 0x05b02301  // dup z1.q, z24.q[2]
    .inst 0x05f02302  // dup z2.q, z24.q[3]
    .inst 0x05702323  // dup z3.q, z25.q[1]
    .inst 0x05b02324  // dup z4.q, z25.q[2]
    .inst 0x05f02325  // dup z5.q, z25.q[3]

    add x9, x0, x4, lsl #1 // +2
    add x15, x0, x4, lsl #2 // +4
    add x8, x9, x4, lsl #2 // +6
    add x11, x0, x4, lsl #3 // +8

    .inst 0xe400f418  // st1b {z24.b}, p5, [x0]
    .inst 0xe4045400  // st1b {z0.b}, p5, [x0, x4]
    .inst 0xe400f521  // st1b {z1.b}, p5, [x9]
    .inst 0xe4045522  // st1b {z2.b}, p5, [x9, x4]
    .inst 0xe400f5f9  // st1b {z25.b}, p5, [x15]
    .inst 0xe40455e3  // st1b {z3.b}, p5, [x15, x4]
    .inst 0xe400f504  // st1b {z4.b}, p5, [x8]
    .inst 0xe4045505  // st1b {z5.b}, p5, [x8, x4]
    .inst 0xe400f57a  // st1b {z26.b}, p5, [x11]
    b End

    TILE1_STORE40:
    .inst 0x05702300  // dup z0.q, z24.q[1]
    .inst 0x05b02301  // dup z1.q, z24.q[2]
    .inst 0x05f02302  // dup z2.q, z24.q[3]
    .inst 0x05702323  // dup z3.q, z25.q[1]
    .inst 0x05b02324  // dup z4.q, z25.q[2]
    .inst 0x05f02325  // dup z5.q, z25.q[3]
    .inst 0x05702346  // dup z6.q, z26.q[1]

    add x9, x0, x4, lsl #1 // +2
    add x15, x0, x4, lsl #2 // +4
    add x8, x9, x4, lsl #2 // +6
    add x11, x0, x4, lsl #3 // +8

    .inst 0xe400f418  // st1b {z24.b}, p5, [x0]
    .inst 0xe4045400  // st1b {z0.b}, p5, [x0, x4]
    .inst 0xe400f521  // st1b {z1.b}, p5, [x9]
    .inst 0xe4045522  // st1b {z2.b}, p5, [x9, x4]
    .inst 0xe400f5f9  // st1b {z25.b}, p5, [x15]
    .inst 0xe40455e3  // st1b {z3.b}, p5, [x15, x4]
    .inst 0xe400f504  // st1b {z4.b}, p5, [x8]
    .inst 0xe4045505  // st1b {z5.b}, p5, [x8, x4]
    .inst 0xe400f57a  // st1b {z26.b}, p5, [x11]
    .inst 0xe4045566  // st1b {z6.b}, p5, [x11, x4]
    b End

    TILE1_STORE44:
    .inst 0x05702300  // dup z0.q, z24.q[1]
    .inst 0x05b02301  // dup z1.q, z24.q[2]
    .inst 0x05f02302  // dup z2.q, z24.q[3]
    .inst 0x05702323  // dup z3.q, z25.q[1]
    .inst 0x05b02324  // dup z4.q, z25.q[2]
    .inst 0x05f02325  // dup z5.q, z25.q[3]
    .inst 0x05702346  // dup z6.q, z26.q[1]
    .inst 0x05b02347  // dup z7.q, z26.q[2]

    add x9, x0, x4, lsl #1 // +2
    add x15, x0, x4, lsl #2 // +4
    add x8, x9, x4, lsl #2 // +6
    add x11, x0, x4, lsl #3 // +8
    add x13, x9, x4, lsl #3 // +10

    .inst 0xe400f418  // st1b {z24.b}, p5, [x0]
    .inst 0xe4045400  // st1b {z0.b}, p5, [x0, x4]
    .inst 0xe400f521  // st1b {z1.b}, p5, [x9]
    .inst 0xe4045522  // st1b {z2.b}, p5, [x9, x4]
    .inst 0xe400f5f9  // st1b {z25.b}, p5, [x15]
    .inst 0xe40455e3  // st1b {z3.b}, p5, [x15, x4]
    .inst 0xe400f504  // st1b {z4.b}, p5, [x8]
    .inst 0xe4045505  // st1b {z5.b}, p5, [x8, x4]
    .inst 0xe400f57a  // st1b {z26.b}, p5, [x11]
    .inst 0xe4045566  // st1b {z6.b}, p5, [x11, x4]
    .inst 0xe400f5a7  // st1b {z7.b}, p5, [x13]
    b End

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

    add x9, x0, x4, lsl #1 // +2
    add x15, x0, x4, lsl #2 // +4
    add x8, x9, x4, lsl #2 // +6
    add x11, x0, x4, lsl #3 // +8
    add x13, x9, x4, lsl #3 // +10

    .inst 0xe400f418  // st1b {z24.b}, p5, [x0]
    .inst 0xe4045400  // st1b {z0.b}, p5, [x0, x4]
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
    b End

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

    add x9, x0, x4, lsl #1 // +2
    add x15, x0, x4, lsl #2 // +4
    add x8, x9, x4, lsl #2 // +6
    add x11, x0, x4, lsl #3 // +8
    add x13, x9, x4, lsl #3 // +10
    add x23, x15, x4, lsl #3 // +12

    .inst 0xe400f418  // st1b {z24.b}, p5, [x0]
    .inst 0xe4045400  // st1b {z0.b}, p5, [x0, x4]
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
    b End

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

    add x9, x0, x4, lsl #1 // +2
    add x15, x0, x4, lsl #2 // +4
    add x8, x9, x4, lsl #2 // +6
    add x11, x0, x4, lsl #3 // +8
    add x13, x9, x4, lsl #3 // +10
    add x23, x15, x4, lsl #3 // +12

    .inst 0xe400f418  // st1b {z24.b}, p5, [x0]
    .inst 0xe4045400  // st1b {z0.b}, p5, [x0, x4]
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
    b End

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

    add x9, x0, x4, lsl #1 // +2
    add x15, x0, x4, lsl #2 // +4
    add x8, x9, x4, lsl #2 // +6
    add x11, x0, x4, lsl #3 // +8
    add x13, x9, x4, lsl #3 // +10
    add x23, x15, x4, lsl #3 // +12
    add x6, x8, x4, lsl #3   // +14

    .inst 0xe400f418  // st1b {z24.b}, p5, [x0]
    .inst 0xe4045400  // st1b {z0.b}, p5, [x0, x4]
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
    .inst 0xe400f4ca  // st1b {z10.b}, p5, [x6]
    b End

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

    add x9, x0, x4, lsl #1 // +2
    add x15, x0, x4, lsl #2 // +4
    add x8, x9, x4, lsl #2 // +6
    add x11, x0, x4, lsl #3 // +8
    add x13, x9, x4, lsl #3 // +10
    add x23, x15, x4, lsl #3 // +12
    add x6, x8, x4, lsl #3   // +14

    .inst 0xe400f418  // st1b {z24.b}, p5, [x0]
    .inst 0xe4045400  // st1b {z0.b}, p5, [x0, x4]
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
    .inst 0xe400f4ca  // st1b {z10.b}, p5, [x6]
    .inst 0xe40454cb  // st1b {z11.b}, p5, [x6, x4]
    b End

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

    add x9, x0, x4, lsl #1 // +2
    add x15, x0, x4, lsl #2 // +4
    add x8, x9, x4, lsl #2 // +6
    add x11, x0, x4, lsl #3 // +8
    add x13, x9, x4, lsl #3 // +10
    add x23, x15, x4, lsl #3 // +12
    add x6, x8, x4, lsl #3   // +14

    .inst 0xe400f418  // st1b {z24.b}, p5, [x0]
    .inst 0xe4045400  // st1b {z0.b}, p5, [x0, x4]
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
    .inst 0xe400f4ca  // st1b {z10.b}, p5, [x6]
    .inst 0xe40454cb  // st1b {z11.b}, p5, [x6, x4]

    add x9, x0, x4, lsl #4 // +16

    .inst 0xe400f53c  // st1b {z28.b}, p5, [x9]
    b End

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
    .inst 0x0570238c  // dup z12.q, z28.q[1]

    add x9, x0, x4, lsl #1 // +2
    add x15, x0, x4, lsl #2 // +4
    add x8, x9, x4, lsl #2 // +6
    add x11, x0, x4, lsl #3 // +8
    add x13, x9, x4, lsl #3 // +10
    add x23, x15, x4, lsl #3 // +12
    add x6, x8, x4, lsl #3   // +14

    .inst 0xe400f418  // st1b {z24.b}, p5, [x0]
    .inst 0xe4045400  // st1b {z0.b}, p5, [x0, x4]
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
    .inst 0xe400f4ca  // st1b {z10.b}, p5, [x6]
    .inst 0xe40454cb  // st1b {z11.b}, p5, [x6, x4]

    add x9, x0, x4, lsl #4 // +16

    .inst 0xe400f53c  // st1b {z28.b}, p5, [x9]
    .inst 0xe404552c  // st1b {z12.b}, p5, [x9, x4]
    b End

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
    .inst 0x05b0238d  // dup z13.q, z28.q[2]

    add x9, x0, x4, lsl #1 // +2
    add x15, x0, x4, lsl #2 // +4
    add x8, x9, x4, lsl #2 // +6
    add x11, x0, x4, lsl #3 // +8
    add x13, x9, x4, lsl #3 // +10
    add x23, x15, x4, lsl #3 // +12
    add x6, x8, x4, lsl #3   // +14

    .inst 0xe400f418  // st1b {z24.b}, p5, [x0]
    .inst 0xe4045400  // st1b {z0.b}, p5, [x0, x4]
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
    .inst 0xe400f4ca  // st1b {z10.b}, p5, [x6]
    .inst 0xe40454cb  // st1b {z11.b}, p5, [x6, x4]

    add x9, x0, x4, lsl #4 // +16
    add x15, x13, x4, lsl #3 // +18

    .inst 0xe400f53c  // st1b {z28.b}, p5, [x9]
    .inst 0xe404552c  // st1b {z12.b}, p5, [x9, x4]
    .inst 0xe400f5ed  // st1b {z13.b}, p5, [x15]
    b End

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
    .inst 0x05b0238d  // dup z13.q, z28.q[2]
    .inst 0x05f0238e  // dup z14.q, z28.q[3]

    add x9, x0, x4, lsl #1 // +2
    add x15, x0, x4, lsl #2 // +4
    add x8, x9, x4, lsl #2 // +6
    add x11, x0, x4, lsl #3 // +8
    add x13, x9, x4, lsl #3 // +10
    add x23, x15, x4, lsl #3 // +12
    add x6, x8, x4, lsl #3   // +14

    .inst 0xe400f418  // st1b {z24.b}, p5, [x0]
    .inst 0xe4045400  // st1b {z0.b}, p5, [x0, x4]
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
    .inst 0xe400f4ca  // st1b {z10.b}, p5, [x6]
    .inst 0xe40454cb  // st1b {z11.b}, p5, [x6, x4]

    add x9, x0, x4, lsl #4 // +16
    add x15, x13, x4, lsl #3 // +18

    .inst 0xe400f53c  // st1b {z28.b}, p5, [x9]
    .inst 0xe404552c  // st1b {z12.b}, p5, [x9, x4]
    .inst 0xe400f5ed  // st1b {z13.b}, p5, [x15]
    .inst 0xe40455ee  // st1b {z14.b}, p5, [x15, x4]
    b End

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
    .inst 0x05b0238d  // dup z13.q, z28.q[2]
    .inst 0x05f0238e  // dup z14.q, z28.q[3]

    add x9, x0, x4, lsl #1 // +2
    add x15, x0, x4, lsl #2 // +4
    add x8, x9, x4, lsl #2 // +6
    add x11, x0, x4, lsl #3 // +8
    add x13, x9, x4, lsl #3 // +10
    add x23, x15, x4, lsl #3 // +12
    add x6, x8, x4, lsl #3   // +14

    .inst 0xe400f418  // st1b {z24.b}, p5, [x0]
    .inst 0xe4045400  // st1b {z0.b}, p5, [x0, x4]
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
    .inst 0xe400f4ca  // st1b {z10.b}, p5, [x6]
    .inst 0xe40454cb  // st1b {z11.b}, p5, [x6, x4]

    add x9, x0, x4, lsl #4 // +16
    add x15, x13, x4, lsl #3 // +18
    add x8, x23, x4, lsl #3 // +20

    .inst 0xe400f53c  // st1b {z28.b}, p5, [x9]
    .inst 0xe404552c  // st1b {z12.b}, p5, [x9, x4]
    .inst 0xe400f5ed  // st1b {z13.b}, p5, [x15]
    .inst 0xe40455ee  // st1b {z14.b}, p5, [x15, x4]
    .inst 0xe400f51d  // st1b {z29.b}, p5, [x8]
    b End

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
    .inst 0x05f0238e  // dup z14.q, z28.q[3]
    .inst 0x057023af  // dup z15.q, z29.q[1]

    add x9, x0, x4, lsl #1 // +2
    add x15, x0, x4, lsl #2 // +4
    add x8, x9, x4, lsl #2 // +6
    add x11, x0, x4, lsl #3 // +8
    add x13, x9, x4, lsl #3 // +10
    add x23, x15, x4, lsl #3 // +12
    add x6, x8, x4, lsl #3   // +14

    .inst 0xe400f418  // st1b {z24.b}, p5, [x0]
    .inst 0xe4045400  // st1b {z0.b}, p5, [x0, x4]
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
    .inst 0xe400f4ca  // st1b {z10.b}, p5, [x6]
    .inst 0xe40454cb  // st1b {z11.b}, p5, [x6, x4]

    add x9, x0, x4, lsl #4 // +16
    add x15, x13, x4, lsl #3 // +18
    add x8, x23, x4, lsl #3 // +20

    .inst 0xe400f53c  // st1b {z28.b}, p5, [x9]
    .inst 0xe404552c  // st1b {z12.b}, p5, [x9, x4]
    .inst 0xe400f5ed  // st1b {z13.b}, p5, [x15]
    .inst 0xe40455ee  // st1b {z14.b}, p5, [x15, x4]
    .inst 0xe400f51d  // st1b {z29.b}, p5, [x8]
    .inst 0xe404550f  // st1b {z15.b}, p5, [x8, x4]
    b End

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
    .inst 0x057023af  // dup z15.q, z29.q[1]
    .inst 0x05b023b0  // dup z16.q, z29.q[2]
    .inst 0x05f023b1  // dup z17.q, z29.q[3]

    add x9, x0, x4, lsl #1 // +2
    add x15, x0, x4, lsl #2 // +4
    add x8, x9, x4, lsl #2 // +6
    add x11, x0, x4, lsl #3 // +8
    add x13, x9, x4, lsl #3 // +10
    add x23, x15, x4, lsl #3 // +12
    add x6, x8, x4, lsl #3   // +14

    .inst 0xe400f418  // st1b {z24.b}, p5, [x0]
    .inst 0xe4045400  // st1b {z0.b}, p5, [x0, x4]
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
    .inst 0xe400f4ca  // st1b {z10.b}, p5, [x6]
    .inst 0xe40454cb  // st1b {z11.b}, p5, [x6, x4]

    add x9, x0, x4, lsl #4 // +16
    add x15, x13, x4, lsl #3 // +18
    add x8, x23, x4, lsl #3 // +20
    add x11, x6, x4, lsl #3 // +22

    .inst 0xe400f53c  // st1b {z28.b}, p5, [x9]
    .inst 0xe404552c  // st1b {z12.b}, p5, [x9, x4]
    .inst 0xe400f5ed  // st1b {z13.b}, p5, [x15]
    .inst 0xe40455ee  // st1b {z14.b}, p5, [x15, x4]
    .inst 0xe400f51d  // st1b {z29.b}, p5, [x8]
    .inst 0xe404550f  // st1b {z15.b}, p5, [x8, x4]
    .inst 0xe400f570  // st1b {z16.b}, p5, [x11]
    b End

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
    .inst 0x05b023b0  // dup z16.q, z29.q[2]
    .inst 0x05f023b1  // dup z17.q, z29.q[3]

    add x9, x0, x4, lsl #1 // +2
    add x15, x0, x4, lsl #2 // +4
    add x8, x9, x4, lsl #2 // +6
    add x11, x0, x4, lsl #3 // +8
    add x13, x9, x4, lsl #3 // +10
    add x23, x15, x4, lsl #3 // +12
    add x6, x8, x4, lsl #3   // +14

    .inst 0xe400f418  // st1b {z24.b}, p5, [x0]
    .inst 0xe4045400  // st1b {z0.b}, p5, [x0, x4]
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
    .inst 0xe400f4ca  // st1b {z10.b}, p5, [x6]
    .inst 0xe40454cb  // st1b {z11.b}, p5, [x6, x4]

    add x9, x0, x4, lsl #4 // +16
    add x15, x13, x4, lsl #3 // +18
    add x8, x23, x4, lsl #3 // +20
    add x11, x6, x4, lsl #3 // +22

    .inst 0xe400f53c  // st1b {z28.b}, p5, [x9]
    .inst 0xe404552c  // st1b {z12.b}, p5, [x9, x4]
    .inst 0xe400f5ed  // st1b {z13.b}, p5, [x15]
    .inst 0xe40455ee  // st1b {z14.b}, p5, [x15, x4]
    .inst 0xe400f51d  // st1b {z29.b}, p5, [x8]
    .inst 0xe404550f  // st1b {z15.b}, p5, [x8, x4]
    .inst 0xe400f570  // st1b {z16.b}, p5, [x11]
    .inst 0xe4045571  // st1b {z17.b}, p5, [x11, x4]
    b End

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
    .inst 0x05f023b1  // dup z17.q, z29.q[3]

    add x9, x0, x4, lsl #1 // +2
    add x15, x0, x4, lsl #2 // +4
    add x8, x9, x4, lsl #2 // +6
    add x11, x0, x4, lsl #3 // +8
    add x13, x9, x4, lsl #3 // +10
    add x23, x15, x4, lsl #3 // +12
    add x6, x8, x4, lsl #3   // +14

    .inst 0xe400f418  // st1b {z24.b}, p5, [x0]
    .inst 0xe4045400  // st1b {z0.b}, p5, [x0, x4]
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
    .inst 0xe400f4ca  // st1b {z10.b}, p5, [x6]
    .inst 0xe40454cb  // st1b {z11.b}, p5, [x6, x4]

    add x9, x0, x4, lsl #4 // +16
    add x15, x13, x4, lsl #3 // +18
    add x8, x23, x4, lsl #3 // +20
    add x11, x6, x4, lsl #3 // +22
    add x13, x9, x4, lsl #3 // +24

    .inst 0xe400f53c  // st1b {z28.b}, p5, [x9]
    .inst 0xe404552c  // st1b {z12.b}, p5, [x9, x4]
    .inst 0xe400f5ed  // st1b {z13.b}, p5, [x15]
    .inst 0xe40455ee  // st1b {z14.b}, p5, [x15, x4]
    .inst 0xe400f51d  // st1b {z29.b}, p5, [x8]
    .inst 0xe404550f  // st1b {z15.b}, p5, [x8, x4]
    .inst 0xe400f570  // st1b {z16.b}, p5, [x11]
    .inst 0xe4045571  // st1b {z17.b}, p5, [x11, x4]
    .inst 0xe400f5be  // st1b {z30.b}, p5, [x13]
    b End

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
    .inst 0x057023d2  // dup z18.q, z30.q[1]

    add x9, x0, x4, lsl #1 // +2
    add x15, x0, x4, lsl #2 // +4
    add x8, x9, x4, lsl #2 // +6
    add x11, x0, x4, lsl #3 // +8
    add x13, x9, x4, lsl #3 // +10
    add x23, x15, x4, lsl #3 // +12
    add x6, x8, x4, lsl #3   // +14

    .inst 0xe400f418  // st1b {z24.b}, p5, [x0]
    .inst 0xe4045400  // st1b {z0.b}, p5, [x0, x4]
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
    .inst 0xe400f4ca  // st1b {z10.b}, p5, [x6]
    .inst 0xe40454cb  // st1b {z11.b}, p5, [x6, x4]

    add x9, x0, x4, lsl #4 // +16
    add x15, x13, x4, lsl #3 // +18
    add x8, x23, x4, lsl #3 // +20
    add x11, x6, x4, lsl #3 // +22
    add x13, x9, x4, lsl #3 // +24

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
    b End

    /* oc=108 */
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
    .inst 0x05b023d3  // dup z19.q, z30.q[2]

    add x9, x0, x4, lsl #1 // +2
    add x15, x0, x4, lsl #2 // +4
    add x8, x9, x4, lsl #2 // +6
    add x11, x0, x4, lsl #3 // +8
    add x13, x9, x4, lsl #3 // +10
    add x23, x15, x4, lsl #3 // +12
    add x6, x8, x4, lsl #3   // +14

    .inst 0xe400f418  // st1b {z24.b}, p5, [x0]
    .inst 0xe4045400  // st1b {z0.b}, p5, [x0, x4]
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
    .inst 0xe400f4ca  // st1b {z10.b}, p5, [x6]
    .inst 0xe40454cb  // st1b {z11.b}, p5, [x6, x4]

    add x9, x0, x4, lsl #4 // +16
    add x15, x13, x4, lsl #3 // +18
    add x8, x23, x4, lsl #3 // +20
    add x11, x6, x4, lsl #3 // +22
    add x13, x9, x4, lsl #3 // +24
    add x23, x15, x4, lsl #3 // +26

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
    b End

    /* oc=112 */
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
    .inst 0x05f023d4  // dup z20.q, z30.q[3]

    add x9, x0, x4, lsl #1 // +2
    add x15, x0, x4, lsl #2 // +4
    add x8, x9, x4, lsl #2 // +6
    add x11, x0, x4, lsl #3 // +8
    add x13, x9, x4, lsl #3 // +10
    add x23, x15, x4, lsl #3 // +12
    add x6, x8, x4, lsl #3   // +14

    .inst 0xe400f418  // st1b {z24.b}, p5, [x0]
    .inst 0xe4045400  // st1b {z0.b}, p5, [x0, x4]
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
    .inst 0xe400f4ca  // st1b {z10.b}, p5, [x6]
    .inst 0xe40454cb  // st1b {z11.b}, p5, [x6, x4]

    add x9, x0, x4, lsl #4 // +16
    add x15, x13, x4, lsl #3 // +18
    add x8, x23, x4, lsl #3 // +20
    add x11, x6, x4, lsl #3 // +22
    add x13, x9, x4, lsl #3 // +24
    add x23, x15, x4, lsl #3 // +26

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
    b End

    /* oc=116 */
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

    add x9, x0, x4, lsl #1 // +2
    add x15, x0, x4, lsl #2 // +4
    add x8, x9, x4, lsl #2 // +6
    add x11, x0, x4, lsl #3 // +8
    add x13, x9, x4, lsl #3 // +10
    add x23, x15, x4, lsl #3 // +12
    add x6, x8, x4, lsl #3   // +14

    .inst 0xe400f418  // st1b {z24.b}, p5, [x0]
    .inst 0xe4045400  // st1b {z0.b}, p5, [x0, x4]
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
    .inst 0xe400f4ca  // st1b {z10.b}, p5, [x6]
    .inst 0xe40454cb  // st1b {z11.b}, p5, [x6, x4]

    add x9, x0, x4, lsl #4 // +16
    add x15, x13, x4, lsl #3 // +18
    add x8, x23, x4, lsl #3 // +20
    add x11, x6, x4, lsl #3 // +22
    add x13, x9, x4, lsl #3 // +24
    add x23, x15, x4, lsl #3 // +26
    add x6, x8, x4, lsl #3   // +28

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
    .inst 0xe400f4df  // st1b {z31.b}, p5, [x6]
    b End

    /* oc=120 */
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

    add x9, x0, x4, lsl #1 // +2
    add x15, x0, x4, lsl #2 // +4
    add x8, x9, x4, lsl #2 // +6
    add x11, x0, x4, lsl #3 // +8
    add x13, x9, x4, lsl #3 // +10
    add x23, x15, x4, lsl #3 // +12
    add x6, x8, x4, lsl #3   // +14

    .inst 0xe400f418  // st1b {z24.b}, p5, [x0]
    .inst 0xe4045400  // st1b {z0.b}, p5, [x0, x4]
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
    .inst 0xe400f4ca  // st1b {z10.b}, p5, [x6]
    .inst 0xe40454cb  // st1b {z11.b}, p5, [x6, x4]

    add x9, x0, x4, lsl #4 // +16
    add x15, x13, x4, lsl #3 // +18
    add x8, x23, x4, lsl #3 // +20
    add x11, x6, x4, lsl #3 // +22
    add x13, x9, x4, lsl #3 // +24
    add x23, x15, x4, lsl #3 // +26
    add x6, x8, x4, lsl #3   // +28

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
    .inst 0xe400f4df  // st1b {z31.b}, p5, [x6]
    .inst 0xe40454d5  // st1b {z21.b}, p5, [x6, x4]
    b End

    /* oc=124 */
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

    add x9, x0, x4, lsl #1 // +2
    add x15, x0, x4, lsl #2 // +4
    add x8, x9, x4, lsl #2 // +6
    add x11, x0, x4, lsl #3 // +8
    add x13, x9, x4, lsl #3 // +10
    add x23, x15, x4, lsl #3 // +12
    add x6, x8, x4, lsl #3   // +14

    .inst 0xe400f418  // st1b {z24.b}, p5, [x0]
    .inst 0xe4045400  // st1b {z0.b}, p5, [x0, x4]
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
    .inst 0xe400f4ca  // st1b {z10.b}, p5, [x6]
    .inst 0xe40454cb  // st1b {z11.b}, p5, [x6, x4]

    add x9, x0, x4, lsl #4 // +16
    add x15, x13, x4, lsl #3 // +18
    add x8, x23, x4, lsl #3 // +20
    add x11, x6, x4, lsl #3 // +22
    add x13, x9, x4, lsl #3 // +24
    add x23, x15, x4, lsl #3 // +26
    add x6, x8, x4, lsl #3   // +28
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
    .inst 0xe400f4df  // st1b {z31.b}, p5, [x6]
    .inst 0xe40454d5  // st1b {z21.b}, p5, [x6, x4]
    .inst 0xe400f596  // st1b {z22.b}, p5, [x12]
    b End

    /* oc=128 */
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

    add x9, x0, x4, lsl #1 // +2
    add x15, x0, x4, lsl #2 // +4
    add x8, x9, x4, lsl #2 // +6
    add x11, x0, x4, lsl #3 // +8
    add x13, x9, x4, lsl #3 // +10
    add x23, x15, x4, lsl #3 // +12
    add x6, x8, x4, lsl #3   // +14

    .inst 0xe400f418  // st1b {z24.b}, p5, [x0]
    .inst 0xe4045400  // st1b {z0.b}, p5, [x0, x4]
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
    .inst 0xe400f4ca  // st1b {z10.b}, p5, [x6]
    .inst 0xe40454cb  // st1b {z11.b}, p5, [x6, x4]

    add x9, x0, x4, lsl #4 // +16
    add x15, x13, x4, lsl #3 // +18
    add x8, x23, x4, lsl #3 // +20
    add x11, x6, x4, lsl #3 // +22
    add x13, x9, x4, lsl #3 // +24
    add x23, x15, x4, lsl #3 // +26
    add x6, x8, x4, lsl #3   // +28
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
    .inst 0xe400f4df  // st1b {z31.b}, p5, [x6]
    .inst 0xe40454d5  // st1b {z21.b}, p5, [x6, x4]
    .inst 0xe400f596  // st1b {z22.b}, p5, [x12]
    .inst 0xe4045597  // st1b {z23.b}, p5, [x12, x4]

    subs x5, x5, #32
    add x0, x0, x4, LSL #5
    beq End
    mov x13, x19
    mov x23, x21
    mov x27, x24
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
