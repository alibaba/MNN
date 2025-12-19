//  MNNGemmInt8AddBiasScale_SME2_w4_Fp16.S
//  Created by MNN on 2022/09/26.
//  Copyright © 2018, Alibaba Group Holding Limited

#if defined(__aarch64__)
#include "MNNAsmGlobal.h"

.text

.macro REVERT_INPUT_DEQUANT_BIAS rg0, rg1, rg2, rg3
mul \rg1, \rg2, \rg3
sub \rg0, \rg0, \rg1
.endm

.macro REVERT_WEIGHT_KERNEL_SUM rg0, rg1, rg2, rg3
// blocknum * up_div(ocDiv4, 8) * sizeof(float) * 16 * 2
// rg2: blocknum, rg3:ocDiv4, rg0: address of weightKernelSum
add \rg1, \rg3, #7
lsr \rg1, \rg1, #3
mul \rg1, \rg2, \rg1
sub \rg0, \rg0, \rg1, LSL #7 // revert weight kernel sum
.endm

asm_function MNNGemmInt8AddBiasScale16x32_SME2_w4_Fp32
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
    float* weightKernelSum;
    float* fp32minmax;
    ssize_t blockNum = 1;
    const int32_t* bias = nullptr;
    const float* inputScale = nullptr;
    const float* inputBias = nullptr;
    float* accumBuffer = nullptr;
    int32_t* indices = nullptr;
};
*/
//void MNNGemmInt8AddBiasScale16x32_SME2_w4_Fp32(int8_t* dst, const int8_t* src,
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
lsl x22, x7, #2 // eSize * GEMM_INT8_SRC_UNIT
lsl x21, x7, #4 // eSize * pack * sizeof (float)

/* initialize predicates */
mov x19, #32             // HP=32
.inst 0x2598e3e0  // ptrue p0.s               // all float32 valid
.inst 0x25a717e1  // whilelt p1.s, xzr, x7    // eSize float32 valid
.inst 0x253567f3  // whilelt pn11.b, xzr, x21, vlx4   // eSize * pack float32 valid
.inst 0x2518e3e3  // ptrue p3.b               // all int8 valid
.inst 0x25207810  // ptrue pn8.b              // all int8 valid
.inst 0x253617e7  // whilelt p7.b, xzr, x22   // eSize * LP int8 valid
.inst 0x25b347f2  // whilelt pn10.s, xzr, x19, vlx2 // 32 float valid
.inst 0x2518e084  // ptrue p4.b, #4                 // 4 int8_t valid
.inst 0x2518e125  // ptrue p5.b, vl16               // 16 int8_t valid

.inst 0x8540c1dc  // ld1rw {z28.s}, p0/z, [x14]  // float min
.inst 0x8541c1dd  // ld1rw {z29.s}, p0/z, [x14, #4]

ESIZE:
    cmp x7, #2
    ble TILE_2
    mov x19, x13      // input kernel sum
    mov x21, x23     // input dequant scale
    mov x20, x27     // input dequant bias

LoopH:
    mov x11, x1             // src
    mov x15, #0             // blockid
.inst 0xc00800ff  // zero {za}
LoopBlockNum:
    mov x10, x3             // src_depth_quad

    .inst 0xc0080033  // zero {za0.s, za1.s}

LoopL:
    .inst 0xa400bd60  // ld1b {z0.b}, p7/z, [x11]       // src
    .inst 0xa400ac41  // ld1b {z1.b}, p3/z, [x2]       // weight
    // int4->int8
    .inst 0xc08a4022  // luti4 {z2.b-z3.b}, zt0, z1[0]
    // matmul
    .inst 0xa0827c00  // smopa za0.s, p7/m, p3/m, z0.b, z2.b
    .inst 0xa0837c01  // smopa za1.s, p7/m, p3/m, z0.b, z3.b
    subs x10, x10, #1
    add x11, x11, x22
    .inst 0x04225022  // addvl x2, x2, #1

bne LoopL

    .inst 0xa0408040  // ld1b {z0.b-z3.b}, pn8/z, [x2]  // weight scale&bias
    .inst 0xa540a5be  // ld1w {z30.s}, p1/z, [x13]    // input kernel sum
    .inst 0xa540a6ff  // ld1w {z31.s}, p1/z, [x23]   // input scale
    .inst 0x04225082  // addvl x2, x2, #4
    add x13, x13, x22

    // extract int32_t vectors from za0.s
    mov w8, #0
    mov w10, #8
    .inst 0xc0060c04  // mova {z4.s-z7.s}, za.s[w8, 0, VGx4]   // z4: e=0(za0h.s[0]), z5: e=4(za0h.s[4], z6: e=8(za0h.s[8]), z7: e=12(za0h.s[12), VG=512bit/32bit
    .inst 0xc0060c88  // mova {z8.s-z11.s}, za.s[w8, 4, VGx4]   // z8: e=1, z9: e=5, z10: e=9, z11: e=13
    .inst 0xc0064c0c  // mova {z12.s-z15.s}, za.s[w10, 0, VGx4]   // z12: e=2, z13: e=6, z14: e=10, z15: e=14
    .inst 0xc0064c90  // mova {z16.s-z19.s}, za.s[w10, 4, VGx4]   // z16: e=3, z17: e=7, z18: e=11, z19: e=15

    .inst 0xc132e084  // scvtf {z4.s-z7.s}, {z4.s-z7.s}
    .inst 0xc132e108  // scvtf {z8.s-z11.s}, {z8.s-z11.s}
    .inst 0xc132e18c  // scvtf {z12.s-z15.s}, {z12.s-z15.s}
    .inst 0xc132e210  // scvtf {z16.s-z19.s}, {z16.s-z19.s}
    .inst 0xc0080011  // zero {za0.s}

    // inputKernelSum x weightBias -> [16,16]
    .inst 0x808207c2  // fmopa za2.s, p1/m, p0/m, z30.s, z2.s
    .inst 0x808307c3  // fmopa za3.s, p1/m, p0/m, z30.s, z3.s

    // inputScale x weightScale -> [16,16]
    .inst 0x808007e0  // fmopa za0.s, p1/m, p0/m, z31.s, z0.s

    mov w10, #2
    mov w8, #0
    add x15, x15, #1       // block++
    .inst 0x053ecc3e  // mov z30.b, p3/m, z1.b  // copy 16 weight scale

    cbz x27, HP_DEQUANT
    .inst 0xa540a762  // ld1w {z2.s}, p1/z, [x27]            // input dequant bias
    .inst 0xa0404b80  // ld1w {z0.s, z1.s}, pn10/z, [x28]    // weight kernel sum
    .inst 0x80800442  // fmopa za2.s, p1/m, p0/m, z2.s, z0.s
    .inst 0x80810443  // fmopa za3.s, p1/m, p0/m, z2.s, z1.s
    add x27, x27, x22
    add x23, x23, x22
    .inst 0x043c505c  // addvl x28, x28, #2

    HP_DEQUANT:
    // extract scale from za0.s
    .inst 0xc0060c14  // mova {z20.s-z23.s}, za.s[w8, 0, VGx4]   // z20-z23: e=0, e=4, e=8, e=12
    .inst 0xc0060c98  // mova {z24.s-z27.s}, za.s[w8, 4, VGx4]   // z24-z27: e=1, e=5, e=9, e=13
    mov w8, #8
    .inst 0xc0060c00  // mova {z0.s-z3.s}, za.s[w8, 0, VGx4]     // z0-z3: e=2, e=6, e=10, e=14

    // accumulate to za2.s
    .inst 0xc1b55880  // fmla za.s[w10, 0, VGx4], {z4.s-z7.s}, {z20.s-z23.s}   // za, row:1,17,33,49
    .inst 0xc1b95904  // fmla za.s[w10, 4, VGx4], {z8.s-z11.s}, {z24.s-z27.s}  // za, row: 5,21,37,53
    mov w10, #10
    .inst 0xc0060c94  // mova {z20.s-z23.s}, za.s[w8, 4, VGx4]                 // z20-z23: e=3, e=7, e=11, e=15
    .inst 0xc1a15980  // fmla za.s[w10, 0, VGx4], {z12.s-z15.s}, {z0.s-z3.s}
    .inst 0xc1b55a04  // fmla za.s[w10, 4, VGx4], {z16.s-z19.s}, {z20.s-z23.s}

    // oc:16-31: extract int32_t vectors from za1.s
    mov w8, #1
    mov w10, #9
    .inst 0xc0060c04  // mova {z4.s-z7.s}, za.s[w8, 0, VGx4]    // z4: e=0(za0h.s[0]), z5: e=4(za0h.s[4], z6: e=8(za0h.s[8]), z7: e=12(za0h.s[12), VG=512bit/32bit
    .inst 0xc0060c88  // mova {z8.s-z11.s}, za.s[w8, 4, VGx4]   // z8: e=1, z9: e=5, z10: e=9, z11: e=13
    .inst 0xc0064c0c  // mova {z12.s-z15.s}, za.s[w10, 0, VGx4] // z12: e=2, z13: e=6, z14: e=10, z15: e=14
    .inst 0xc0064c90  // mova {z16.s-z19.s}, za.s[w10, 4, VGx4] // z16: e=3, z17: e=7, z18: e=11, z19: e=15

    .inst 0xc132e084  // scvtf {z4.s-z7.s}, {z4.s-z7.s}
    .inst 0xc132e108  // scvtf {z8.s-z11.s}, {z8.s-z11.s}
    .inst 0xc132e18c  // scvtf {z12.s-z15.s}, {z12.s-z15.s}
    .inst 0xc132e210  // scvtf {z16.s-z19.s}, {z16.s-z19.s}
    .inst 0xc0080022  // zero {za1.s}

    // inputScale x weightScale -> [16,16]
    .inst 0x809e67e1  // fmopa za1.s, p1/m, p3/m, z31.s, z30.s
    mov w8, #1
    mov w10, #3
    // extract scale from za1.s
    .inst 0xc0060c14  // mova {z20.s-z23.s}, za.s[w8, 0, VGx4]   // z20-z23: e=0, e=4, e=8, e=12
    .inst 0xc0060c98  // mova {z24.s-z27.s}, za.s[w8, 4, VGx4]   // z24-z27: e=1, e=5, e=9, e=13
    mov w8, #9
    .inst 0xc0060c00  // mova {z0.s-z3.s}, za.s[w8, 0, VGx4]   // z0-z3: e=2, e=6, e=10, e=14

    // accumulate to za3.s
    .inst 0xc1b55880  // fmla za.s[w10, 0, VGx4], {z4.s-z7.s}, {z20.s-z23.s}   // za, row:1,17,33,49
    .inst 0xc1b95904  // fmla za.s[w10, 4, VGx4], {z8.s-z11.s}, {z24.s-z27.s}   // za, row: 5,21,37,53
    mov w10, #11
    .inst 0xc0060c94  // mova {z20.s-z23.s}, za.s[w8, 4, VGx4]   // z20-z23: e=3, e=7, e=11, e=15
    .inst 0xc1a15980  // fmla za.s[w10, 0, VGx4], {z12.s-z15.s}, {z0.s-z3.s}
    .inst 0xc1b55a04  // fmla za.s[w10, 4, VGx4], {z16.s-z19.s}, {z20.s-z23.s}


    /* next block */
    cmp x15, x26
    beq HP_POST
    b LoopBlockNum

    HP_POST:
    cbz x9, HP_FLOAT_STORE
    lsl x15, x5, #2 // ocRemain
    .inst 0x25af47f1  // whilelt pn9.s, xzr, x15, vlx2
    .inst 0xa0404520  // ld1w {z0.s, z1.s}, pn9/z, [x9]
    .inst 0x25b9ce02  // fmov z2.s, #1
    .inst 0x04295049  // addvl x9, x9, #2
    .inst 0x80800042  // fmopa za2.s, p0/m, p0/m, z2.s, z0.s
    .inst 0x80810043  // fmopa za3.s, p0/m, p0/m, z2.s, z1.s

    HP_FLOAT_STORE:
    /* oc:0~15 */
    mov w13, #0
    mov w15, #4
    .inst 0xc086a440  // mova {z0.s-z3.s}, za2v.s[w13, 0:3]
    .inst 0xc086e444  // mova {z4.s-z7.s}, za2v.s[w15, 0:3]
    mov w13, #8
    mov w15, #12
    .inst 0xc086a448  // mova {z8.s-z11.s}, za2v.s[w13, 0:3]
    .inst 0xc086e44c  // mova {z12.s-z15.s}, za2v.s[w15, 0:3]

    .inst 0xc1b6e010  // zip {z16.s-z19.s}, {z0.s-z3.s}
    .inst 0xc1b6e094  // zip {z20.s-z23.s}, {z4.s-z7.s}
    .inst 0xc1b6e118  // zip {z24.s-z27.s}, {z8.s-z11.s}
    .inst 0xc1b6e180  // zip {z0.s-z3.s}, {z12.s-z15.s}

    .inst 0xc1bdcb90  // fclamp {z16.s-z19.s}, z28.s, z29.s
    .inst 0xc1bdcb94  // fclamp {z20.s-z23.s}, z28.s, z29.s
    .inst 0xc1bdcb98  // fclamp {z24.s-z27.s}, z28.s, z29.s
    .inst 0xc1bdcb80  // fclamp {z0.s-z3.s}, z28.s, z29.s

    cmp x5, #4
    bge STORE16
    cmp x5, #1
    beq STORE4
    cmp x5, #2
    beq STORE8

    STORE12:
    add x10, x0, x4, LSL #1 // + 2*x4
    .inst 0xa0608c10  // st1b {z16.b-z19.b}, pn11, [x0]
    .inst 0xa0248c14  // st1b {z20.b-z23.b}, pn11, [x0, x4]
    .inst 0xa0608d58  // st1b {z24.b-z27.b}, pn11, [x10]
    b End

    STORE8:
    .inst 0xa0608c10  // st1b {z16.b-z19.b}, pn11, [x0]
    .inst 0xa0248c14  // st1b {z20.b-z23.b}, pn11, [x0, x4]
    b End

    STORE4:
    .inst 0xa0608c10  // st1b {z16.b-z19.b}, pn11, [x0]
    b End

    STORE16:
    add x10, x0, x4, LSL #1 // + 2*x4
    subs x5, x5, #4
    .inst 0xa0608c10  // st1b {z16.b-z19.b}, pn11, [x0]
    .inst 0xa0248c14  // st1b {z20.b-z23.b}, pn11, [x0, x4]
    .inst 0xa0608d58  // st1b {z24.b-z27.b}, pn11, [x10]
    .inst 0xa0248d40  // st1b {z0.b-z3.b}, pn11, [x10, x4]
    add x0, x0, x4, LSL #2
    beq End

    /* oc:16~31 */
    mov w13, #0
    mov w15, #4
    .inst 0xc086a460  // mova {z0.s-z3.s}, za3v.s[w13, 0:3]
    .inst 0xc086e464  // mova {z4.s-z7.s}, za3v.s[w15, 0:3]
    mov w13, #8
    mov w15, #12
    .inst 0xc086a468  // mova {z8.s-z11.s}, za3v.s[w13, 0:3]
    .inst 0xc086e46c  // mova {z12.s-z15.s}, za3v.s[w15, 0:3]

    .inst 0xc1b6e010  // zip {z16.s-z19.s}, {z0.s-z3.s}
    .inst 0xc1b6e094  // zip {z20.s-z23.s}, {z4.s-z7.s}
    .inst 0xc1b6e118  // zip {z24.s-z27.s}, {z8.s-z11.s}
    .inst 0xc1b6e180  // zip {z0.s-z3.s}, {z12.s-z15.s}

    .inst 0xc1bdcb90  // fclamp {z16.s-z19.s}, z28.s, z29.s
    .inst 0xc1bdcb94  // fclamp {z20.s-z23.s}, z28.s, z29.s
    .inst 0xc1bdcb98  // fclamp {z24.s-z27.s}, z28.s, z29.s
    .inst 0xc1bdcb80  // fclamp {z0.s-z3.s}, z28.s, z29.s

    cmp x5, #4
    bge STORE32
    cmp x5, #3
    beq STORE28
    cmp x5, #2
    beq STORE24

    STORE20:
    .inst 0xa0608c10  // st1b {z16.b-z19.b}, pn11, [x0]
    b End

    STORE28:
    add x10, x0, x4, LSL #1 // + 2*x4
    .inst 0xa0608c10  // st1b {z16.b-z19.b}, pn11, [x0]
    .inst 0xa0248c14  // st1b {z20.b-z23.b}, pn11, [x0, x4]
    .inst 0xa0608d58  // st1b {z24.b-z27.b}, pn11, [x10]
    b End

    STORE24:
    .inst 0xa0608c10  // st1b {z16.b-z19.b}, pn11, [x0]
    .inst 0xa0248c14  // st1b {z20.b-z23.b}, pn11, [x0, x4]
    b End

    STORE32:
    add x10, x0, x4, LSL #1 // + 2*x4
    subs x5, x5, #4
    .inst 0xa0608c10  // st1b {z16.b-z19.b}, pn11, [x0]
    .inst 0xa0248c14  // st1b {z20.b-z23.b}, pn11, [x0, x4]
    .inst 0xa0608d58  // st1b {z24.b-z27.b}, pn11, [x10]
    .inst 0xa0248d40  // st1b {z0.b-z3.b}, pn11, [x10, x4]
    add x0, x0, x4, LSL #2
    beq End

    // revert input scale/kernelSum
    mov x13, x19
    mov x23, x21
    mov x27, x20
    b LoopH

TILE_2:
    cmp x7, #1
    beq TILE_1
    mov x19, x13      // input kernel sum
    mov x21, x23      // input dequant scale
    mov x20, x27      // input dequant bias

    mov x15, #32
    .inst 0x252f17e5  // whilelt p5.b, xzr, x15

LoopDz_TILE2:
    .inst 0x25b8c01a  // mov z26.s, #0
    .inst 0x25b8c01b  // mov z27.s, #0
    .inst 0x25b8c01e  // mov z30.s, #0
    .inst 0x25b8c01f  // mov z31.s, #0
    mov w8, #0
    mov x11, x1             // src
    mov x15, x26

TILE2_BLOCKNUM:
    mov x10, x3             // src_depth_quad

.inst 0xc00800ff  // zero {za}

    LoopSz_TILE2:
        .inst 0xa4003d62  // ld1rqb {z2.b}, p7/z, [x11]       // src
        .inst 0xa400ac41  // ld1b {z1.b}, p3/z, [x2]       // weight
        // int4->int8
        .inst 0xc08a4024  // luti4 {z4.b-z5.b}, zt0, z1[0]
        // matmul
        .inst 0xc15210a0  // sdot za.s[w8, 0, VGx2], {z4.b-z5.b}, z2.b[0]
        .inst 0xc15214a4  // sdot za.s[w8, 4, VGx2], {z4.b-z5.b}, z2.b[1]
        subs x10, x10, #1
        add x11, x11, x22
        .inst 0x04225022  // addvl x2, x2, #1

    bne LoopSz_TILE2

    sub x15, x15, #1
    .inst 0xc0060808  // mova {z8.s-z9.s}, za.s[w8, 0, VGx2]
    .inst 0xc006088a  // mova {z10.s-z11.s}, za.s[w8, 4, VGx2]
    .inst 0xc132e108  // scvtf {z8.s-z11.s}, {z8.s-z11.s}

    .inst 0xa0408040  // ld1b {z0.b-z3.b}, pn8/z, [x2]    // weight scale&bias
    .inst 0xa4003da4  // ld1rqb {z4.b}, p7/z, [x13]    // input kernel sum
    .inst 0xa4003ee5  // ld1rqb {z5.b}, p7/z, [x23]   // input kernel scale

    .inst 0x64a52006  // fmul z6.s, z0.s, z5.s[0] // e0
    .inst 0x64a52027  // fmul z7.s, z1.s, z5.s[0]
    .inst 0x64ad200c  // fmul z12.s, z0.s, z5.s[1]  // e1
    .inst 0x64ad202d  // fmul z13.s, z1.s, z5.s[1]

    .inst 0x64a4005a  // fmla z26.s, z2.s, z4.s[0]  // e0
    .inst 0x64a4007b  // fmla z27.s, z3.s, z4.s[0]
    .inst 0x64ac005e  // fmla z30.s, z2.s, z4.s[1]  // e1
    .inst 0x64ac007f  // fmla z31.s, z3.s, z4.s[1]

    .inst 0x65a60d1a  // fmla z26.s, p3/m, z8.s, z6.s
    .inst 0x65a70d3b  // fmla z27.s, p3/m, z9.s, z7.s
    .inst 0x65ac0d5e  // fmla z30.s, p3/m, z10.s, z12.s
    .inst 0x65ad0d7f  // fmla z31.s, p3/m, z11.s, z13.s
    .inst 0x04225082  // addvl x2, x2, #4
    add x13, x13, x22

    cbz x27, TILE2_ADD_DSTV
    .inst 0xa4003f65  // ld1rqb {z5.b}, p7/z, [x27]   // input dequant bias
    .inst 0xa0404b88  // ld1w {z8.s, z9.s}, pn10/z, [x28]    // weight kernel sum
    .inst 0x64a5011a  // fmla z26.s, z8.s, z5.s[0]
    .inst 0x64a5013b  // fmla z27.s, z9.s, z5.s[0]
    .inst 0x64ad011e  // fmla z30.s, z8.s, z5.s[1]
    .inst 0x64ad013f  // fmla z31.s, z9.s, z5.s[1]
    add x27, x27, x22
    add x23, x23, x22
    .inst 0x043c505c  // addvl x28, x28, #2

    TILE2_ADD_DSTV:
    cbnz x15, TILE2_BLOCKNUM

    TILE2_STORE:
    lsl x15, x5, #2 // ocRemain
    .inst 0x25af47f1  // whilelt pn9.s, xzr, x15, vlx2
    .inst 0xa0404520  // ld1w {z0.s, z1.s}, pn9/z, [x9]   // bias
    .inst 0x6580035a  // fadd z26.s, z26.s, z0.s
    .inst 0x6581037b  // fadd z27.s, z27.s, z1.s
    .inst 0x658003de  // fadd z30.s, z30.s, z0.s
    .inst 0x658103ff  // fadd z31.s, z31.s, z1.s
    .inst 0x04295049  // addvl x9, x9, #2

    .inst 0xc1bdc39a  // fclamp {z26.s-z27.s}, z28.s, z29.s
    .inst 0xc1bdc39e  // fclamp {z30.s-z31.s}, z28.s, z29.s
    // z0: 0~3,4_7 z1:8~11,12~15
    .inst 0xc13ed740  // zip {z0.q-z1.q}, z26.q, z30.q  // (0,0)(0,1)(0,2)(0,3)(1,0)(1,1)(1,2)(1,3)...(1,12)(1,13)(1,14)(1,15)
    .inst 0xc13fd762  // zip {z2.q-z3.q}, z27.q, z31.q  // (0,16)(0,17)(0,18)(0,19)(1,16)(1,17)(1,18)(1,19)...(1,28)(1,29)(1,30)(1,31)

    cmp x5, #8
    bge TILE2_STORE32
    cmp x5, #7
    beq TILE2_STORE28
    cmp x5, #6
    beq TILE2_STORE24
    cmp x5, #5
    beq TILE2_STORE20
    cmp x5, #4
    beq TILE2_STORE16
    cmp x5, #3
    beq TILE2_STORE12
    cmp x5, #2
    beq TILE2_STORE8

    TILE2_STORE4:
    .inst 0xe400f400  // st1b {z0.b}, p5, [x0]
    b End

    TILE2_STORE32:
    add x10, x0, x4, LSL #1
    add x8, x0, x4, LSL #2
    add x13, x10, x4, LSL #2
    .inst 0x05640005  // ext z5.b, {z0.b, z1.b}, #32
    .inst 0x05640027  // ext z7.b, {z1.b, z2.b}, #32
    .inst 0x05640049  // ext z9.b, {z2.b, z3.b}, #32
    .inst 0x0564006b  // ext z11.b, {z3.b, z4.b}, #32
    .inst 0xe400f400  // st1b {z0.b}, p5, [x0]
    .inst 0xe4045405  // st1b {z5.b}, p5, [x0, x4]
    .inst 0xe400f541  // st1b {z1.b}, p5, [x10]
    .inst 0xe4045547  // st1b {z7.b}, p5, [x10, x4]
    .inst 0xe400f502  // st1b {z2.b}, p5, [x8]
    .inst 0xe4045509  // st1b {z9.b}, p5, [x8, x4]
    .inst 0xe400f5a3  // st1b {z3.b}, p5, [x13]
    .inst 0xe40455ab  // st1b {z11.b}, p5, [x13, x4]
    b TILE2_Dz_End

    TILE2_STORE28:
    add x10, x0, x4, LSL #1
    add x8, x0, x4, LSL #2
    add x13, x10, x4, LSL #2
    .inst 0x05640005  // ext z5.b, {z0.b, z1.b}, #32
    .inst 0x05640027  // ext z7.b, {z1.b, z2.b}, #32
    .inst 0x05640049  // ext z9.b, {z2.b, z3.b}, #32
    .inst 0xe400f400  // st1b {z0.b}, p5, [x0]
    .inst 0xe4045405  // st1b {z5.b}, p5, [x0, x4]
    .inst 0xe400f541  // st1b {z1.b}, p5, [x10]
    .inst 0xe4045547  // st1b {z7.b}, p5, [x10, x4]
    .inst 0xe400f502  // st1b {z2.b}, p5, [x8]
    .inst 0xe4045509  // st1b {z9.b}, p5, [x8, x4]
    .inst 0xe400f5a3  // st1b {z3.b}, p5, [x13]
    b End

    TILE2_STORE24:
    add x10, x0, x4, LSL #1
    add x8, x0, x4, LSL #2
    .inst 0x05640005  // ext z5.b, {z0.b, z1.b}, #32
    .inst 0x05640027  // ext z7.b, {z1.b, z2.b}, #32
    .inst 0x05640049  // ext z9.b, {z2.b, z3.b}, #32
    .inst 0xe400f400  // st1b {z0.b}, p5, [x0]
    .inst 0xe4045405  // st1b {z5.b}, p5, [x0, x4]
    .inst 0xe400f541  // st1b {z1.b}, p5, [x10]
    .inst 0xe4045547  // st1b {z7.b}, p5, [x10, x4]
    .inst 0xe400f502  // st1b {z2.b}, p5, [x8]
    .inst 0xe4045509  // st1b {z9.b}, p5, [x8, x4]
    b End

    TILE2_STORE20:
    add x10, x0, x4, LSL #1
    add x8, x0, x4, LSL #2
    .inst 0x05640005  // ext z5.b, {z0.b, z1.b}, #32
    .inst 0x05640027  // ext z7.b, {z1.b, z2.b}, #32
    .inst 0xe400f400  // st1b {z0.b}, p5, [x0]
    .inst 0xe4045405  // st1b {z5.b}, p5, [x0, x4]
    .inst 0xe400f541  // st1b {z1.b}, p5, [x10]
    .inst 0xe4045547  // st1b {z7.b}, p5, [x10, x4]
    .inst 0xe400f502  // st1b {z2.b}, p5, [x8]
    b End

    TILE2_STORE16:
    add x10, x0, x4, LSL #1
    .inst 0x05640005  // ext z5.b, {z0.b, z1.b}, #32
    .inst 0x05640027  // ext z7.b, {z1.b, z2.b}, #32
    .inst 0xe400f400  // st1b {z0.b}, p5, [x0]
    .inst 0xe4045405  // st1b {z5.b}, p5, [x0, x4]
    .inst 0xe400f541  // st1b {z1.b}, p5, [x10]
    .inst 0xe4045547  // st1b {z7.b}, p5, [x10, x4]
    b End

    TILE2_STORE12:
    add x10, x0, x4, LSL #1
    .inst 0x05640005  // ext z5.b, {z0.b, z1.b}, #32
    .inst 0xe400f400  // st1b {z0.b}, p5, [x0]
    .inst 0xe4045405  // st1b {z5.b}, p5, [x0, x4]
    .inst 0xe400f541  // st1b {z1.b}, p5, [x10]
    b End

    TILE2_STORE8:
    .inst 0x05640005  // ext z5.b, {z0.b, z1.b}, #32
    .inst 0xe400f400  // st1b {z0.b}, p5, [x0]
    .inst 0xe4045405  // st1b {z5.b}, p5, [x0, x4]
    b End


    TILE2_Dz_End:
    subs x5, x5, #8
    add x0, x0, x4, LSL #3
    mov x13, x19
    mov x23, x21
    mov x27, x20
    beq End
    b LoopDz_TILE2

TILE_1:
    cmp x7, #1
    blt End
    mov x19, x13      // input kernel sum
    mov x21, x23      // input dequant scale
    mov x20, x27      // input dequant bias

LoopDz_TILE1:
.inst 0x25b8c01e  // mov z30.s, #0
.inst 0x25b8c01f  // mov z31.s, #0
    mov w8, #0
    mov x11, x1             // src
    mov x15, x26            // blockid
TILE1_BLOCKNUM:
    mov x10, x3             // src_depth_quad

.inst 0xc00800ff  // zero {za}

    LoopSz_TILE_1:
        .inst 0xa4003162  // ld1rqb {z2.b}, p4/z, [x11]       // src
        .inst 0xa400ac40  // ld1b {z0.b}, p3/z, [x2]       // weight
        // int4->int8
        .inst 0xc08a4004  // luti4 {z4.b-z5.b}, zt0, z0[0]
        // matmul
        .inst 0xc15210a0  // sdot za.s[w8, 0, VGx2], {z4.b-z5.b}, z2.b[0]
        subs x10, x10, #1
        add x11, x11, x22
        .inst 0x04225022  // addvl x2, x2, #1

    bne LoopSz_TILE_1

LoopSzEnd_TILE_1:
    sub x15, x15, #1
    .inst 0xc0060808  // mova {z8.s-z9.s}, za.s[w8, 0, VGx2]
    .inst 0x6594a108  // scvtf z8.s, p0/m, z8.s
    .inst 0x6594a129  // scvtf z9.s, p0/m, z9.s

    .inst 0xa0408040  // ld1b {z0.b-z3.b}, pn8/z, [x2]    // weight scale&bias
    .inst 0x8540c1a4  // ld1rw {z4.s}, p0/z, [x13]    // input kernel sum
    .inst 0x8540c2e5  // ld1rw {z5.s}, p0/z, [x23]   // input kernel scale

    .inst 0x65850800  // fmul z0.s, z0.s, z5.s
    .inst 0x65850821  // fmul z1.s, z1.s, z5.s
    .inst 0x64a4005e  // fmla z30.s, z2.s, z4.s[0]
    .inst 0x64a4007f  // fmla z31.s, z3.s, z4.s[0]
    .inst 0x65a0011e  // fmla z30.s, p0/m, z8.s, z0.s
    .inst 0x65a1013f  // fmla z31.s, p0/m, z9.s, z1.s
    .inst 0x04225082  // addvl x2, x2, #4
    add x13, x13, x22

    cbz x27, TILE1_ADD_DSTV
    .inst 0x8540c365  // ld1rw {z5.s}, p0/z, [x27]   // input dequant bias
    .inst 0xa0404b88  // ld1w {z8.s, z9.s}, pn10/z, [x28]    // weight kernel sum
    .inst 0x64a5011e  // fmla z30.s, z8.s, z5.s[0]
    .inst 0x64a5013f  // fmla z31.s, z9.s, z5.s[0]
    add x27, x27, x22
    add x23, x23, x22
    .inst 0x043c505c  // addvl x28, x28, #2

    TILE1_ADD_DSTV:
    cmp x15, #0
    bne TILE1_BLOCKNUM

    TILE1_STORE:
    lsl x15, x5, #2 // ocRemain
    .inst 0x25af47f1  // whilelt pn9.s, xzr, x15, vlx2
    .inst 0xa0404520  // ld1w {z0.s, z1.s}, pn9/z, [x9]   // bias
    .inst 0x04295049  // addvl x9, x9, #2
    .inst 0x658003de  // fadd z30.s, z30.s, z0.s
    .inst 0x658103ff  // fadd z31.s, z31.s, z1.s
    .inst 0xc1bdc39e  // fclamp {z30.s-z31.s}, z28.s, z29.s

    cmp x5, #8
    bge TILE1_STORE32
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
    .inst 0xe400f41e  // st1b {z30.b}, p5, [x0]
    b End

    TILE1_STORE28:
    add x10, x0, x4, LSL #1
    add x8, x0, x4, LSL #2
    add x13, x10, x4, LSL #2
    .inst 0x057023c4  // dup z4.q, z30.q[1]
    .inst 0x05b023c5  // dup z5.q, z30.q[2]
    .inst 0x05f023c6  // dup z6.q, z30.q[3]
    .inst 0x057023e7  // dup z7.q, z31.q[1]
    .inst 0x05b023e8  // dup z8.q, z31.q[2]
    .inst 0xe400f41e  // st1b {z30.b}, p5, [x0]
    .inst 0xe4045404  // st1b {z4.b}, p5, [x0, x4]
    .inst 0xe400f545  // st1b {z5.b}, p5, [x10]
    .inst 0xe4045546  // st1b {z6.b}, p5, [x10, x4]
    .inst 0xe400f51f  // st1b {z31.b}, p5, [x8]
    .inst 0xe4045507  // st1b {z7.b}, p5, [x8, x4]
    .inst 0xe400f5a8  // st1b {z8.b}, p5, [x13]
    b End

    TILE1_STORE24:
    add x10, x0, x4, LSL #1
    add x8, x0, x4, LSL #2
    .inst 0x057023c4  // dup z4.q, z30.q[1]
    .inst 0x05b023c5  // dup z5.q, z30.q[2]
    .inst 0x05f023c6  // dup z6.q, z30.q[3]
    .inst 0x057023e7  // dup z7.q, z31.q[1]
    .inst 0xe400f41e  // st1b {z30.b}, p5, [x0]
    .inst 0xe4045404  // st1b {z4.b}, p5, [x0, x4]
    .inst 0xe400f545  // st1b {z5.b}, p5, [x10]
    .inst 0xe4045546  // st1b {z6.b}, p5, [x10, x4]
    .inst 0xe400f51f  // st1b {z31.b}, p5, [x8]
    .inst 0xe4045507  // st1b {z7.b}, p5, [x8, x4]
    b End

    TILE1_STORE20:
    add x10, x0, x4, LSL #1
    add x8, x0, x4, LSL #2
    .inst 0x057023c4  // dup z4.q, z30.q[1]
    .inst 0x05b023c5  // dup z5.q, z30.q[2]
    .inst 0x05f023c6  // dup z6.q, z30.q[3]
    .inst 0xe400f41e  // st1b {z30.b}, p5, [x0]
    .inst 0xe4045404  // st1b {z4.b}, p5, [x0, x4]
    .inst 0xe400f545  // st1b {z5.b}, p5, [x10]
    .inst 0xe4045546  // st1b {z6.b}, p5, [x10, x4]
    .inst 0xe400f51f  // st1b {z31.b}, p5, [x8]
    b End

    TILE1_STORE16:
    add x10, x0, x4, LSL #1
    .inst 0x057023c4  // dup z4.q, z30.q[1]
    .inst 0x05b023c5  // dup z5.q, z30.q[2]
    .inst 0x05f023c6  // dup z6.q, z30.q[3]
    .inst 0xe400f41e  // st1b {z30.b}, p5, [x0]
    .inst 0xe4045404  // st1b {z4.b}, p5, [x0, x4]
    .inst 0xe400f545  // st1b {z5.b}, p5, [x10]
    .inst 0xe4045546  // st1b {z6.b}, p5, [x10, x4]
    b End

    TILE1_STORE12:
    add x10, x0, x4, LSL #1
    .inst 0x057023c4  // dup z4.q, z30.q[1]
    .inst 0x05b023c5  // dup z5.q, z30.q[2]
    .inst 0xe400f41e  // st1b {z30.b}, p5, [x0]
    .inst 0xe4045404  // st1b {z4.b}, p5, [x0, x4]
    .inst 0xe400f545  // st1b {z5.b}, p5, [x10]
    b End

    TILE1_STORE8:
    .inst 0x057023c4  // dup z4.q, z30.q[1]
    .inst 0xe400f41e  // st1b {z30.b}, p5, [x0]
    .inst 0xe4045404  // st1b {z4.b}, p5, [x0, x4]
    b End


    TILE1_STORE32:
    subs x5, x5, #8
    add x10, x0, x4, LSL #1
    add x8, x0, x4, LSL #2
    add x13, x10, x4, LSL #2
    .inst 0x057023c4  // dup z4.q, z30.q[1]
    .inst 0x05b023c5  // dup z5.q, z30.q[2]
    .inst 0x05f023c6  // dup z6.q, z30.q[3]
    .inst 0x057023e7  // dup z7.q, z31.q[1]
    .inst 0x05b023e8  // dup z8.q, z31.q[2]
    .inst 0x05f023e9  // dup z9.q, z31.q[3]
    .inst 0xe400f41e  // st1b {z30.b}, p5, [x0]
    .inst 0xe4045404  // st1b {z4.b}, p5, [x0, x4]
    .inst 0xe400f545  // st1b {z5.b}, p5, [x10]
    .inst 0xe4045546  // st1b {z6.b}, p5, [x10, x4]
    .inst 0xe400f51f  // st1b {z31.b}, p5, [x8]
    .inst 0xe4045507  // st1b {z7.b}, p5, [x8, x4]
    .inst 0xe400f5a8  // st1b {z8.b}, p5, [x13]
    .inst 0xe40455a9  // st1b {z9.b}, p5, [x13, x4]
    beq End

    add x0, x0, x4, LSL #3
    mov x13, x19
    mov x23, x21
    mov x27, x20
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
