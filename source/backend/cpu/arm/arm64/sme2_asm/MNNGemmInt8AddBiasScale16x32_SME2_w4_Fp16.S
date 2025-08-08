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
// blocknum * up_div(ocDiv4, 4) * sizeof(float) * 16 * 2
// rg2: blocknum, rg3:ocDiv4, rg0: address of weightKernelSum
add \rg1, \rg3, #3
lsr \rg1, \rg1, #2
mul \rg1, \rg2, \rg1
sub \rg0, \rg0, \rg1, LSL #7 // revert weight kernel sum
.endm

asm_function MNNGemmInt8AddBiasScale16x32_SME2_w4_Fp16
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
//void MNNGemmInt8AddBiasScale16x32_SME2_w4_Fp16(int8_t* dst, const int8_t* src,
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
mov x22, #12
.inst 0x2518e080  // ptrue p0.b, #4           // first 4 bytes
.inst 0x0460e3f8  // cnth x24
.inst 0x2518e125  // ptrue p5.b, vl16
.inst 0x253817e2  // whilelt p2.b, xzr, x24   // previous 32 bytes valid
.inst 0x2518e3e3  // ptrue p3.b               // 64 bytes valid
.inst 0x25207810  // ptrue pn8.b
.inst 0x2598e084  // ptrue p4.s, #4


lsl x22, x7, #2 // eDest * GEMM_INT8_SRC_UNIT
mov x25, 0       // inputBlockNum=1
cbz x27, TILE_16
mov x25, x22         // input block quant: realDstCount * sizeof(float)

TILE_16:
cmp x7, #16
blt TILE_12

    mov x24, x5      // dst_depth_quad
    mov x6, x0       // dst
    mov x12, x2      // weight
    mov x20, x9      // bias
    mov x19, x13      // input kernel sum
    mov x21, x23     // input dequant scale

LoopDz_TILE16:
    mov x11, x1             // src
    mov x15, #0             // blockid
.inst 0xc00800ff  // zero {za}
TILE16_BLOCKNUM:
    mov x10, x3             // src_depth_quad

    .inst 0xc0080033  // zero {za0.s, za1.s}

LoopSz_TILE16:
    .inst 0xa400ad60  // ld1b {z0.b}, p3/z, [x11]       // src
    .inst 0xa400ad81  // ld1b {z1.b}, p3/z, [x12]       // weight
    // int4->int8
    .inst 0xc08a4022  // luti4 {z2.b-z3.b}, zt0, z1[0]
    // matmul
    .inst 0xa0826c00  // smopa za0.s, p3/m, p3/m, z0.b, z2.b
    .inst 0xa0836c01  // smopa za1.s, p3/m, p3/m, z0.b, z3.b
    subs x10, x10, #1
    add x11, x11, x22
    .inst 0x042c502c  // addvl x12, x12, #1

bne LoopSz_TILE16

    .inst 0xa040c180  // ld1w {z0.s-z3.s}, pn8/z, [x12]  // weight scale&bias
    .inst 0xa540adbe  // ld1w {z30.s}, p3/z, [x13]    // input kernel sum
    .inst 0xa540aeff  // ld1w {z31.s}, p3/z, [x23]   // input scale
    .inst 0x042c508c  // addvl x12, x12, #4
    add x13, x13, x22
    add x23, x23, x25

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
    .inst 0x80826fc2  // fmopa za2.s, p3/m, p3/m, z30.s, z2.s
    .inst 0x80836fc3  // fmopa za3.s, p3/m, p3/m, z30.s, z3.s

    // inputScale x weightScale -> [16,16]
    .inst 0x80806fe0  // fmopa za0.s, p3/m, p3/m, z31.s, z0.s

    mov w10, #2
    mov w8, #0
    add x15, x15, #1 // block++
    .inst 0x053ecc3e  // mov z30.b, p3/m, z1.b  // copy 16 weight scale

    cbz x27, TILE16_DEQUANT
    .inst 0xa540af62  // ld1w {z2.s}, p3/z, [x27]    // input dequant bias
    .inst 0xa0404380  // ld1w {z0.s, z1.s}, pn8/z, [x28]    // weight kernel sum
    .inst 0x80806c42  // fmopa za2.s, p3/m, p3/m, z2.s, z0.s
    .inst 0x80816c43  // fmopa za3.s, p3/m, p3/m, z2.s, z1.s
    add x27, x27, x25
    .inst 0x043c505c  // addvl x28, x28, #2

    TILE16_DEQUANT:
    // extract scale from za0.s
    .inst 0xc0060c14  // mova {z20.s-z23.s}, za.s[w8, 0, VGx4]   // z20-z23: e=0, e=4, e=8, e=12
    .inst 0xc0060c98  // mova {z24.s-z27.s}, za.s[w8, 4, VGx4]   // z24-z27: e=1, e=5, e=9, e=13
    mov w8, #8
    .inst 0xc0060c00  // mova {z0.s-z3.s}, za.s[w8, 0, VGx4]   // z0-z3: e=2, e=6, e=10, e=14

    // accumulate to za2.s
    .inst 0xc1b55880  // fmla za.s[w10, 0, VGx4], {z4.s-z7.s}, {z20.s-z23.s}   // za, row:1,17,33,49
    .inst 0xc1b95904  // fmla za.s[w10, 4, VGx4], {z8.s-z11.s}, {z24.s-z27.s}   // za, row: 5,21,37,53
    mov w10, #10
    .inst 0xc0060c94  // mova {z20.s-z23.s}, za.s[w8, 4, VGx4]   // z20-z23: e=3, e=7, e=11, e=15
    .inst 0xc1a15980  // fmla za.s[w10, 0, VGx4], {z12.s-z15.s}, {z0.s-z3.s}
    .inst 0xc1b55a04  // fmla za.s[w10, 4, VGx4], {z16.s-z19.s}, {z20.s-z23.s}

    // oc:16-31: extract int32_t vectors from za1.s
    mov w8, #1
    mov w10, #9
    .inst 0xc0060c04  // mova {z4.s-z7.s}, za.s[w8, 0, VGx4]   // z4: e=0(za0h.s[0]), z5: e=4(za0h.s[4], z6: e=8(za0h.s[8]), z7: e=12(za0h.s[12), VG=512bit/32bit
    .inst 0xc0060c88  // mova {z8.s-z11.s}, za.s[w8, 4, VGx4]   // z8: e=1, z9: e=5, z10: e=9, z11: e=13
    .inst 0xc0064c0c  // mova {z12.s-z15.s}, za.s[w10, 0, VGx4]   // z12: e=2, z13: e=6, z14: e=10, z15: e=14
    .inst 0xc0064c90  // mova {z16.s-z19.s}, za.s[w10, 4, VGx4]   // z16: e=3, z17: e=7, z18: e=11, z19: e=15

    .inst 0xc132e084  // scvtf {z4.s-z7.s}, {z4.s-z7.s}
    .inst 0xc132e108  // scvtf {z8.s-z11.s}, {z8.s-z11.s}
    .inst 0xc132e18c  // scvtf {z12.s-z15.s}, {z12.s-z15.s}
    .inst 0xc132e210  // scvtf {z16.s-z19.s}, {z16.s-z19.s}
    .inst 0xc0080022  // zero {za1.s}

    // inputScale x weightScale -> [16,16]
    .inst 0x809e6fe1  // fmopa za1.s, p3/m, p3/m, z31.s, z30.s
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
    beq TILE16_POST
    b TILE16_BLOCKNUM

    TILE16_POST:
    cbz x9, TILE16_STORE
    .inst 0xa0404280  // ld1w {z0.s, z1.s}, pn8/z, [x20]   // bias
    .inst 0x25b9ce02  // fmov z2.s, #1
    .inst 0x04345054  // addvl x20, x20, #2
    .inst 0x80806c42  // fmopa za2.s, p3/m, p3/m, z2.s, z0.s
    .inst 0x80816c43  // fmopa za3.s, p3/m, p3/m, z2.s, z1.s

    TILE16_STORE:
    /* oc:0~15 */
    mov w13, #0
    mov w15, #4
    .inst 0xc086a440  // mova {z0.s-z3.s}, za2v.s[w13, 0:3]
    .inst 0xc086e444  // mova {z4.s-z7.s}, za2v.s[w15, 0:3]
    mov w13, #8
    mov w15, #12
    .inst 0xc086a448  // mova {z8.s-z11.s}, za2v.s[w13, 0:3]
    .inst 0xc086e44c  // mova {z12.s-z15.s}, za2v.s[w15, 0:3]
 
    .inst 0xc120e030  // fcvtn z16.h, {z0.s-z1.s}   // (0,0)(0,1)(1,0)(1,1)...(15,0)(15,1)
    .inst 0xc120e071  // fcvtn z17.h, {z2.s-z3.s}   // (0,2)(0,3)(1,2)(1,3)...(15,2)(15,3)
    .inst 0xc120e0b2  // fcvtn z18.h, {z4.s-z5.s}   // (0,4)(0,5)(1,4)(1,5)...(15,4)(15,5)
    .inst 0xc120e0f3  // fcvtn z19.h, {z6.s-z7.s}   // (0,6)(0,7)(1,6)(1,7)...(15,6)(15,7)

    .inst 0xc120e134  // fcvtn z20.h, {z8.s-z9.s}
    .inst 0xc120e175  // fcvtn z21.h, {z10.s-z11.s}
    .inst 0xc120e1b6  // fcvtn z22.h, {z12.s-z13.s}
    .inst 0xc120e1f7  // fcvtn z23.h, {z14.s-z15.s}

    /* oc:16~31 */
    mov w13, #0
    mov w15, #4
    .inst 0xc086a460  // mova {z0.s-z3.s}, za3v.s[w13, 0:3]
    .inst 0xc086e464  // mova {z4.s-z7.s}, za3v.s[w15, 0:3]
    mov w13, #8
    mov w15, #12
    .inst 0xc086a468  // mova {z8.s-z11.s}, za3v.s[w13, 0:3]
    .inst 0xc086e46c  // mova {z12.s-z15.s}, za3v.s[w15, 0:3]

    .inst 0xc1b6e218  // zip {z24.s-z27.s}, {z16.s-z19.s}
    .inst 0xc1b6e29c  // zip {z28.s-z31.s}, {z20.s-z23.s}

    .inst 0xc120e030  // fcvtn z16.h, {z0.s-z1.s}   // (0,0)(0,1)(1,0)(1,1)...(15,0)(15,1)
    .inst 0xc120e071  // fcvtn z17.h, {z2.s-z3.s}   // (0,2)(0,3)(1,2)(1,3)...(15,2)(15,3)
    .inst 0xc120e0b2  // fcvtn z18.h, {z4.s-z5.s}   // (0,4)(0,5)(1,4)(1,5)...(15,4)(15,5)
    .inst 0xc120e0f3  // fcvtn z19.h, {z6.s-z7.s}   // (0,6)(0,7)(1,6)(1,7)...(15,6)(15,7)

    .inst 0xc120e134  // fcvtn z20.h, {z8.s-z9.s}
    .inst 0xc120e175  // fcvtn z21.h, {z10.s-z11.s}
    .inst 0xc120e1b6  // fcvtn z22.h, {z12.s-z13.s}
    .inst 0xc120e1f7  // fcvtn z23.h, {z14.s-z15.s}

    .inst 0x84c0adc8  // ld1rh {z8.h}, p3/z, [x14]
    .inst 0x84c1adc9  // ld1rh {z9.h}, p3/z, [x14, #2]

    .inst 0xc1b6e200  // zip {z0.s-z3.s}, {z16.s-z19.s}
    .inst 0xc1b6e284  // zip {z4.s-z7.s}, {z20.s-z23.s}

    .inst 0xc169c918  // fclamp {z24.h-z27.h}, z8.h, z9.h
    .inst 0xc169c91c  // fclamp {z28.h-z31.h}, z8.h, z9.h
    .inst 0xc169c900  // fclamp {z0.h-z3.h}, z8.h, z9.h
    .inst 0xc169c904  // fclamp {z4.h-z7.h}, z8.h, z9.h

    cmp x24, #1
    beq TILE16_STORE8
    cmp x24, #2
    beq TILE16_STORE16
    cmp x24, #3
    beq TILE16_STORE24

    TILE16_STORE32:
    add x11, x6, x4, LSL #1
    .inst 0xa06080d8  // st1b {z24.b-z27.b}, pn8, [x6]
    .inst 0xa02480dc  // st1b {z28.b-z31.b}, pn8, [x6, x4]
    .inst 0xa0608160  // st1b {z0.b-z3.b}, pn8, [x11]
    .inst 0xa0248164  // st1b {z4.b-z7.b}, pn8, [x11, x4]
    b TILE16_Dz_End

    TILE16_STORE24:
    add x11, x6, x4, LSL #1
    .inst 0xa06080d8  // st1b {z24.b-z27.b}, pn8, [x6]
    .inst 0xa02480dc  // st1b {z28.b-z31.b}, pn8, [x6, x4]
    .inst 0xa0608160  // st1b {z0.b-z3.b}, pn8, [x11]
    b TILE16_Dz_End

    TILE16_STORE16:
    .inst 0xa06080d8  // st1b {z24.b-z27.b}, pn8, [x6]
    .inst 0xa02480dc  // st1b {z28.b-z31.b}, pn8, [x6, x4]
    b TILE16_Dz_End

    TILE16_STORE8:
    .inst 0xa06080d8  // st1b {z24.b-z27.b}, pn8, [x6]
    b TILE16_Dz_End
    

    TILE16_Dz_End:
    sub x24, x24, #4
    add x6, x6, x4, LSL #2

    cmp x24, #0
    ble End
    // revert input scale/kernelSum
    mov x13, x19
    mov x23, x21
    cbz x27, LoopDz_TILE16
    REVERT_INPUT_DEQUANT_BIAS x27, x15, x26, x25
    b LoopDz_TILE16

TILE_12:
cmp x7, #12
blt TILE_8

    mov x21, #48
.inst 0x253517e7  // whilelt p7.b, xzr, x21

    mov x24, x5      // dst_depth_quad
    mov x6, x0       // dst
    mov x12, x2      // weight
    mov x20, x9      // bias
    mov x19, x13      // input kernel sum
    mov x21, x23     // input dequant scale

LoopDz_TILE12:
    mov x11, x1             // src
    mov x15, #0             // blockid
.inst 0xc00800ff  // zero {za}
TILE12_BLOCKNUM:
    mov x10, x3             // src_depth_quad

    .inst 0xc0080033  // zero {za0.s, za1.s}

LoopSz_TILE12:
    .inst 0xa400bd60  // ld1b {z0.b}, p7/z, [x11]       // src
    .inst 0xa400ad81  // ld1b {z1.b}, p3/z, [x12]       // weight
    // int4->int8
    .inst 0xc08a4022  // luti4 {z2.b-z3.b}, zt0, z1[0]
    // matmul
    .inst 0xa0827c00  // smopa za0.s, p7/m, p3/m, z0.b, z2.b
    .inst 0xa0837c01  // smopa za1.s, p7/m, p3/m, z0.b, z3.b
    subs x10, x10, #1
    add x11, x11, x22
    .inst 0x042c502c  // addvl x12, x12, #1

bne LoopSz_TILE12

    .inst 0xa040c180  // ld1w {z0.s-z3.s}, pn8/z, [x12]  // weight scale&bias
    .inst 0xa400bdbe  // ld1b {z30.b}, p7/z, [x13]    // input kernel sum
    .inst 0xa400beff  // ld1b {z31.b}, p7/z, [x23]   // input scale
    .inst 0x042c508c  // addvl x12, x12, #4
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
    .inst 0x80827fc2  // fmopa za2.s, p7/m, p3/m, z30.s, z2.s
    .inst 0x80837fc3  // fmopa za3.s, p7/m, p3/m, z30.s, z3.s

    // inputScale x weightScale -> [16,16]
    .inst 0x80807fe0  // fmopa za0.s, p7/m, p3/m, z31.s, z0.s

    mov w10, #2
    mov w8, #0
    add x15, x15, #1 // block++
    .inst 0x053ecc3e  // mov z30.b, p3/m, z1.b  // copy 16 weight scale

    cbz x27, TILE12_DEQUANT
    .inst 0xa400bf62  // ld1b {z2.b}, p7/z, [x27]    // input dequant bias
    .inst 0xa0404380  // ld1w {z0.s, z1.s}, pn8/z, [x28]    // weight kernel sum
    .inst 0x80807c42  // fmopa za2.s, p7/m, p3/m, z2.s, z0.s
    .inst 0x80817c43  // fmopa za3.s, p7/m, p3/m, z2.s, z1.s
    add x27, x27, x25
    add x23, x23, x25 // input scale
    .inst 0x043c505c  // addvl x28, x28, #2

    TILE12_DEQUANT:
    // extract scale from za0.s
    .inst 0xc0060c14  // mova {z20.s-z23.s}, za.s[w8, 0, VGx4]   // z20-z23: e=0, e=4, e=8, e=12
    .inst 0xc0060c98  // mova {z24.s-z27.s}, za.s[w8, 4, VGx4]   // z24-z27: e=1, e=5, e=9, e=13
    mov w8, #8
    .inst 0xc0060c00  // mova {z0.s-z3.s}, za.s[w8, 0, VGx4]   // z0-z3: e=2, e=6, e=10, e=14

    // accumulate to za2.s
    .inst 0xc1b55880  // fmla za.s[w10, 0, VGx4], {z4.s-z7.s}, {z20.s-z23.s}   // za, row:1,17,33,49
    .inst 0xc1b95904  // fmla za.s[w10, 4, VGx4], {z8.s-z11.s}, {z24.s-z27.s}   // za, row: 5,21,37,53
    mov w10, #10
    .inst 0xc0060c94  // mova {z20.s-z23.s}, za.s[w8, 4, VGx4]   // z20-z23: e=3, e=7, e=11, e=15
    .inst 0xc1a15980  // fmla za.s[w10, 0, VGx4], {z12.s-z15.s}, {z0.s-z3.s}
    .inst 0xc1b55a04  // fmla za.s[w10, 4, VGx4], {z16.s-z19.s}, {z20.s-z23.s}

    // oc:16-31: extract int32_t vectors from za1.s
    mov w8, #1
    mov w10, #9
    .inst 0xc0060c04  // mova {z4.s-z7.s}, za.s[w8, 0, VGx4]   // z4: e=0(za0h.s[0]), z5: e=4(za0h.s[4], z6: e=8(za0h.s[8]), z7: e=12(za0h.s[12), VG=512bit/32bit
    .inst 0xc0060c88  // mova {z8.s-z11.s}, za.s[w8, 4, VGx4]   // z8: e=1, z9: e=5, z10: e=9, z11: e=13
    .inst 0xc0064c0c  // mova {z12.s-z15.s}, za.s[w10, 0, VGx4]   // z12: e=2, z13: e=6, z14: e=10, z15: e=14
    .inst 0xc0064c90  // mova {z16.s-z19.s}, za.s[w10, 4, VGx4]   // z16: e=3, z17: e=7, z18: e=11, z19: e=15

    .inst 0xc132e084  // scvtf {z4.s-z7.s}, {z4.s-z7.s}
    .inst 0xc132e108  // scvtf {z8.s-z11.s}, {z8.s-z11.s}
    .inst 0xc132e18c  // scvtf {z12.s-z15.s}, {z12.s-z15.s}
    .inst 0xc132e210  // scvtf {z16.s-z19.s}, {z16.s-z19.s}
    .inst 0xc0080022  // zero {za1.s}

    // inputScale x weightScale -> [16,16]
    .inst 0x809e6fe1  // fmopa za1.s, p3/m, p3/m, z31.s, z30.s
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
    beq TILE12_POST
    b TILE12_BLOCKNUM

    TILE12_POST:
    cbz x9, TILE12_STORE
    .inst 0xa0404280  // ld1w {z0.s, z1.s}, pn8/z, [x20]   // bias
    .inst 0x25b9ce02  // fmov z2.s, #1
    .inst 0x04345054  // addvl x20, x20, #2
    .inst 0x80806c42  // fmopa za2.s, p3/m, p3/m, z2.s, z0.s
    .inst 0x80816c43  // fmopa za3.s, p3/m, p3/m, z2.s, z1.s

    TILE12_STORE:
    /* oc:0~15 */
    mov w13, #0
    mov w15, #4
    .inst 0xc086a440  // mova {z0.s-z3.s}, za2v.s[w13, 0:3]
    .inst 0xc086e444  // mova {z4.s-z7.s}, za2v.s[w15, 0:3]
    mov w13, #8
    mov w15, #12
    .inst 0xc086a448  // mova {z8.s-z11.s}, za2v.s[w13, 0:3]
    .inst 0xc086e44c  // mova {z12.s-z15.s}, za2v.s[w15, 0:3]
 
    .inst 0xc120e030  // fcvtn z16.h, {z0.s-z1.s}   // (0,0)(0,1)(1,0)(1,1)...(15,0)(15,1)
    .inst 0xc120e071  // fcvtn z17.h, {z2.s-z3.s}   // (0,2)(0,3)(1,2)(1,3)...(15,2)(15,3)
    .inst 0xc120e0b2  // fcvtn z18.h, {z4.s-z5.s}   // (0,4)(0,5)(1,4)(1,5)...(15,4)(15,5)
    .inst 0xc120e0f3  // fcvtn z19.h, {z6.s-z7.s}   // (0,6)(0,7)(1,6)(1,7)...(15,6)(15,7)

    .inst 0xc120e134  // fcvtn z20.h, {z8.s-z9.s}
    .inst 0xc120e175  // fcvtn z21.h, {z10.s-z11.s}
    .inst 0xc120e1b6  // fcvtn z22.h, {z12.s-z13.s}
    .inst 0xc120e1f7  // fcvtn z23.h, {z14.s-z15.s}

    /* oc:16~31 */
    mov w13, #0
    mov w15, #4
    .inst 0xc086a460  // mova {z0.s-z3.s}, za3v.s[w13, 0:3]
    .inst 0xc086e464  // mova {z4.s-z7.s}, za3v.s[w15, 0:3]
    mov w13, #8
    mov w15, #12
    .inst 0xc086a468  // mova {z8.s-z11.s}, za3v.s[w13, 0:3]
    .inst 0xc086e46c  // mova {z12.s-z15.s}, za3v.s[w15, 0:3]

    .inst 0xc1b6e218  // zip {z24.s-z27.s}, {z16.s-z19.s}
    .inst 0xc1b6e29c  // zip {z28.s-z31.s}, {z20.s-z23.s}

    .inst 0xc120e030  // fcvtn z16.h, {z0.s-z1.s}   // (0,0)(0,1)(1,0)(1,1)...(15,0)(15,1)
    .inst 0xc120e071  // fcvtn z17.h, {z2.s-z3.s}   // (0,2)(0,3)(1,2)(1,3)...(15,2)(15,3)
    .inst 0xc120e0b2  // fcvtn z18.h, {z4.s-z5.s}   // (0,4)(0,5)(1,4)(1,5)...(15,4)(15,5)
    .inst 0xc120e0f3  // fcvtn z19.h, {z6.s-z7.s}   // (0,6)(0,7)(1,6)(1,7)...(15,6)(15,7)

    .inst 0xc120e134  // fcvtn z20.h, {z8.s-z9.s}
    .inst 0xc120e175  // fcvtn z21.h, {z10.s-z11.s}
    .inst 0xc120e1b6  // fcvtn z22.h, {z12.s-z13.s}
    .inst 0xc120e1f7  // fcvtn z23.h, {z14.s-z15.s}

    .inst 0x84c0adc8  // ld1rh {z8.h}, p3/z, [x14]
    .inst 0x84c1adc9  // ld1rh {z9.h}, p3/z, [x14, #2]

    .inst 0xc1b6e200  // zip {z0.s-z3.s}, {z16.s-z19.s}
    .inst 0xc1b6e284  // zip {z4.s-z7.s}, {z20.s-z23.s}

    .inst 0xc169c918  // fclamp {z24.h-z27.h}, z8.h, z9.h
    .inst 0xc169c91c  // fclamp {z28.h-z31.h}, z8.h, z9.h
    .inst 0xc169c900  // fclamp {z0.h-z3.h}, z8.h, z9.h
    .inst 0xc169c904  // fclamp {z4.h-z7.h}, z8.h, z9.h

    cmp x24, #1
    beq TILE12_STORE8
    cmp x24, #2
    beq TILE12_STORE16
    cmp x24, #3
    beq TILE12_STORE24

    TILE12_STORE32:
    add x10, x6, x4
    add x11, x6, x4, LSL #1
    add x13, x11, x4
    .inst 0xa06000d8  // st1b {z24.b-z25.b}, pn8, [x6]
    .inst 0xe402ecda  // st1b {z26.b}, p3, [x6, #2, MUL VL]
    .inst 0xa060015c  // st1b {z28.b-z29.b}, pn8, [x10]
    .inst 0xe402ed5e  // st1b {z30.b}, p3, [x10, #2, MUL VL]
    .inst 0xa0600160  // st1b {z0.b-z1.b}, pn8, [x11]
    .inst 0xe402ed62  // st1b {z2.b}, p3, [x11, #2, MUL VL]
    .inst 0xa06001a4  // st1b {z4.b-z5.b}, pn8, [x13]
    .inst 0xe402eda6  // st1b {z6.b}, p3, [x13, #2, MUL VL]
    b TILE12_Dz_End

    TILE12_STORE24:
    add x10, x6, x4
    add x11, x6, x4, LSL #1
    .inst 0xa06000d8  // st1b {z24.b-z25.b}, pn8, [x6]
    .inst 0xe402ecda  // st1b {z26.b}, p3, [x6, #2, MUL VL]
    .inst 0xa060015c  // st1b {z28.b-z29.b}, pn8, [x10]
    .inst 0xe402ed5e  // st1b {z30.b}, p3, [x10, #2, MUL VL]
    .inst 0xa0600160  // st1b {z0.b-z1.b}, pn8, [x11]
    .inst 0xe402ed62  // st1b {z2.b}, p3, [x11, #2, MUL VL]
    b TILE12_Dz_End

    TILE12_STORE16:
    add x10, x6, x4
    .inst 0xa06000d8  // st1b {z24.b-z25.b}, pn8, [x6]
    .inst 0xe402ecda  // st1b {z26.b}, p3, [x6, #2, MUL VL]
    .inst 0xa060015c  // st1b {z28.b-z29.b}, pn8, [x10]
    .inst 0xe402ed5e  // st1b {z30.b}, p3, [x10, #2, MUL VL]
    b TILE12_Dz_End

    TILE12_STORE8:
    .inst 0xa06000d8  // st1b {z24.b-z25.b}, pn8, [x6]
    .inst 0xe402ecda  // st1b {z26.b}, p3, [x6, #2, MUL VL]
    b TILE12_Dz_End
    

    TILE12_Dz_End:
    sub x24, x24, #4
    add x6, x6, x4, LSL #2

    cmp x24, #0
    ble TILE12_End
    // revert input scale/kernelSum
    mov x13, x19
    mov x23, x21
    cbz x27, LoopDz_TILE12
    REVERT_INPUT_DEQUANT_BIAS x27, x15, x26, x25
    b LoopDz_TILE12

    TILE12_End:
    // update realE/dst/src/inputScale/inputKernelSum
    sub x7, x7, #12
    cbz x7, End
    add x0, x0, #192
    add x1, x1, #48
    add x13, x19, #48
    add x23, x21, #48
    // update inputBias/weightKernelSum
    cbz x27, TILE_8
    REVERT_INPUT_DEQUANT_BIAS x27, x15, x26, x25
    REVERT_WEIGHT_KERNEL_SUM x28, x11, x26, x5
    add x27, x27, #48

TILE_8:
cmp x7, #8
blt TILE_4

    mov x24, x5      // dst_depth_quad
    mov x6, x0       // dst
    mov x12, x2      // weight
    mov x20, x9      // bias
    mov x19, x13      // input kernel sum
    mov x21, x23     // input dequant scale

LoopDz_TILE8:
    mov x11, x1             // src
    mov x15, #0             // blockid
.inst 0xc00800ff  // zero {za}
TILE8_BLOCKNUM:
    mov x10, x3             // src_depth_quad

    .inst 0xc0080033  // zero {za0.s, za1.s}

LoopSz_TILE8:
    .inst 0xa400a960  // ld1b {z0.b}, p2/z, [x11]       // src
    .inst 0xa400ad81  // ld1b {z1.b}, p3/z, [x12]       // weight
    // int4->int8
    .inst 0xc08a4022  // luti4 {z2.b-z3.b}, zt0, z1[0]
    // matmul
    .inst 0xa0826800  // smopa za0.s, p2/m, p3/m, z0.b, z2.b
    .inst 0xa0836801  // smopa za1.s, p2/m, p3/m, z0.b, z3.b
    subs x10, x10, #1
    add x11, x11, x22
    .inst 0x042c502c  // addvl x12, x12, #1

bne LoopSz_TILE8

    .inst 0xa040c180  // ld1w {z0.s-z3.s}, pn8/z, [x12]  // weight scale&bias
    .inst 0xa400a9be  // ld1b {z30.b}, p2/z, [x13]    // input kernel sum
    .inst 0xa400aaff  // ld1b {z31.b}, p2/z, [x23]   // input scale
    .inst 0x042c508c  // addvl x12, x12, #4
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
    .inst 0x80826bc2  // fmopa za2.s, p2/m, p3/m, z30.s, z2.s
    .inst 0x80836bc3  // fmopa za3.s, p2/m, p3/m, z30.s, z3.s

    // inputScale x weightScale -> [16,16]
    .inst 0x80806be0  // fmopa za0.s, p2/m, p3/m, z31.s, z0.s

    mov w10, #2
    mov w8, #0
    add x15, x15, #1 // block++
    .inst 0x053ecc3e  // mov z30.b, p3/m, z1.b  // copy 16 weight scale

    cbz x27, TILE8_DEQUANT
    .inst 0xa400ab62  // ld1b {z2.b}, p2/z, [x27]    // input dequant bias
    .inst 0xa0404380  // ld1w {z0.s, z1.s}, pn8/z, [x28]    // weight kernel sum
    .inst 0x80806842  // fmopa za2.s, p2/m, p3/m, z2.s, z0.s
    .inst 0x80816843  // fmopa za3.s, p2/m, p3/m, z2.s, z1.s
    add x27, x27, x25
    add x23, x23, x25 // input scale
    .inst 0x043c505c  // addvl x28, x28, #2

    TILE8_DEQUANT:
    // extract scale from za0.s
    .inst 0xc0060c14  // mova {z20.s-z23.s}, za.s[w8, 0, VGx4]   // z20-z23: e=0, e=4, e=8, e=12
    .inst 0xc0060c98  // mova {z24.s-z27.s}, za.s[w8, 4, VGx4]   // z24-z27: e=1, e=5, e=9, e=13
    mov w8, #8
    .inst 0xc0060c00  // mova {z0.s-z3.s}, za.s[w8, 0, VGx4]   // z0-z3: e=2, e=6, e=10, e=14

    // accumulate to za2.s
    .inst 0xc1b55880  // fmla za.s[w10, 0, VGx4], {z4.s-z7.s}, {z20.s-z23.s}   // za, row:1,17,33,49
    .inst 0xc1b95904  // fmla za.s[w10, 4, VGx4], {z8.s-z11.s}, {z24.s-z27.s}   // za, row: 5,21,37,53
    mov w10, #10
    .inst 0xc0060c94  // mova {z20.s-z23.s}, za.s[w8, 4, VGx4]   // z20-z23: e=3, e=7, e=11, e=15
    .inst 0xc1a15980  // fmla za.s[w10, 0, VGx4], {z12.s-z15.s}, {z0.s-z3.s}
    .inst 0xc1b55a04  // fmla za.s[w10, 4, VGx4], {z16.s-z19.s}, {z20.s-z23.s}

    // oc:16-31: extract int32_t vectors from za1.s
    mov w8, #1
    mov w10, #9
    .inst 0xc0060c04  // mova {z4.s-z7.s}, za.s[w8, 0, VGx4]   // z4: e=0(za0h.s[0]), z5: e=4(za0h.s[4], z6: e=8(za0h.s[8]), z7: e=12(za0h.s[12), VG=512bit/32bit
    .inst 0xc0060c88  // mova {z8.s-z11.s}, za.s[w8, 4, VGx4]   // z8: e=1, z9: e=5, z10: e=9, z11: e=13
    .inst 0xc0064c0c  // mova {z12.s-z15.s}, za.s[w10, 0, VGx4]   // z12: e=2, z13: e=6, z14: e=10, z15: e=14
    .inst 0xc0064c90  // mova {z16.s-z19.s}, za.s[w10, 4, VGx4]   // z16: e=3, z17: e=7, z18: e=11, z19: e=15

    .inst 0xc132e084  // scvtf {z4.s-z7.s}, {z4.s-z7.s}
    .inst 0xc132e108  // scvtf {z8.s-z11.s}, {z8.s-z11.s}
    .inst 0xc132e18c  // scvtf {z12.s-z15.s}, {z12.s-z15.s}
    .inst 0xc132e210  // scvtf {z16.s-z19.s}, {z16.s-z19.s}
    .inst 0xc0080022  // zero {za1.s}

    // inputScale x weightScale -> [16,16]
    .inst 0x809e6fe1  // fmopa za1.s, p3/m, p3/m, z31.s, z30.s
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
    beq TILE8_POST
    b TILE8_BLOCKNUM

    TILE8_POST:
    cbz x9, TILE8_STORE
    .inst 0xa0404280  // ld1w {z0.s, z1.s}, pn8/z, [x20]   // bias
    .inst 0x25b9ce02  // fmov z2.s, #1
    .inst 0x04345054  // addvl x20, x20, #2
    .inst 0x80806c42  // fmopa za2.s, p3/m, p3/m, z2.s, z0.s
    .inst 0x80816c43  // fmopa za3.s, p3/m, p3/m, z2.s, z1.s

    TILE8_STORE:
    /* oc:0~15 */
    mov w13, #0
    mov w15, #4
    .inst 0xc086a440  // mova {z0.s-z3.s}, za2v.s[w13, 0:3]
    .inst 0xc086e444  // mova {z4.s-z7.s}, za2v.s[w15, 0:3]
    mov w13, #8
    mov w15, #12
    .inst 0xc086a448  // mova {z8.s-z11.s}, za2v.s[w13, 0:3]
    .inst 0xc086e44c  // mova {z12.s-z15.s}, za2v.s[w15, 0:3]
 
    .inst 0xc120e030  // fcvtn z16.h, {z0.s-z1.s}   // (0,0)(0,1)(1,0)(1,1)...(15,0)(15,1)
    .inst 0xc120e071  // fcvtn z17.h, {z2.s-z3.s}   // (0,2)(0,3)(1,2)(1,3)...(15,2)(15,3)
    .inst 0xc120e0b2  // fcvtn z18.h, {z4.s-z5.s}   // (0,4)(0,5)(1,4)(1,5)...(15,4)(15,5)
    .inst 0xc120e0f3  // fcvtn z19.h, {z6.s-z7.s}   // (0,6)(0,7)(1,6)(1,7)...(15,6)(15,7)

    .inst 0xc120e134  // fcvtn z20.h, {z8.s-z9.s}
    .inst 0xc120e175  // fcvtn z21.h, {z10.s-z11.s}
    .inst 0xc120e1b6  // fcvtn z22.h, {z12.s-z13.s}
    .inst 0xc120e1f7  // fcvtn z23.h, {z14.s-z15.s}

    /* oc:16~31 */
    mov w13, #0
    mov w15, #4
    .inst 0xc086a460  // mova {z0.s-z3.s}, za3v.s[w13, 0:3]
    .inst 0xc086e464  // mova {z4.s-z7.s}, za3v.s[w15, 0:3]
    mov w13, #8
    mov w15, #12
    .inst 0xc086a468  // mova {z8.s-z11.s}, za3v.s[w13, 0:3]
    .inst 0xc086e46c  // mova {z12.s-z15.s}, za3v.s[w15, 0:3]

    .inst 0xc1b6e218  // zip {z24.s-z27.s}, {z16.s-z19.s}
    .inst 0xc1b6e29c  // zip {z28.s-z31.s}, {z20.s-z23.s}

    .inst 0xc120e030  // fcvtn z16.h, {z0.s-z1.s}   // (0,0)(0,1)(1,0)(1,1)...(15,0)(15,1)
    .inst 0xc120e071  // fcvtn z17.h, {z2.s-z3.s}   // (0,2)(0,3)(1,2)(1,3)...(15,2)(15,3)
    .inst 0xc120e0b2  // fcvtn z18.h, {z4.s-z5.s}   // (0,4)(0,5)(1,4)(1,5)...(15,4)(15,5)
    .inst 0xc120e0f3  // fcvtn z19.h, {z6.s-z7.s}   // (0,6)(0,7)(1,6)(1,7)...(15,6)(15,7)

    .inst 0xc120e134  // fcvtn z20.h, {z8.s-z9.s}
    .inst 0xc120e175  // fcvtn z21.h, {z10.s-z11.s}
    .inst 0xc120e1b6  // fcvtn z22.h, {z12.s-z13.s}
    .inst 0xc120e1f7  // fcvtn z23.h, {z14.s-z15.s}

    .inst 0x84c0adc8  // ld1rh {z8.h}, p3/z, [x14]
    .inst 0x84c1adc9  // ld1rh {z9.h}, p3/z, [x14, #2]

    .inst 0xc1b6e200  // zip {z0.s-z3.s}, {z16.s-z19.s}
    .inst 0xc1b6e284  // zip {z4.s-z7.s}, {z20.s-z23.s}

    .inst 0xc169c918  // fclamp {z24.h-z27.h}, z8.h, z9.h
    .inst 0xc169c91c  // fclamp {z28.h-z31.h}, z8.h, z9.h
    .inst 0xc169c900  // fclamp {z0.h-z3.h}, z8.h, z9.h
    .inst 0xc169c904  // fclamp {z4.h-z7.h}, z8.h, z9.h

    cmp x24, #4
    bge TILE8_STORE32
    cmp x24, #1
    beq TILE8_STORE8
    cmp x24, #2
    beq TILE8_STORE16
    cmp x24, #3
    beq TILE8_STORE24

    TILE8_STORE32:
    add x11, x6, x4, LSL #1
    .inst 0xa06000d8  // st1b {z24.b-z25.b}, pn8, [x6]
    .inst 0xa02400dc  // st1b {z28.b-z29.b}, pn8, [x6, x4]
    .inst 0xa0600160  // st1b {z0.b-z1.b}, pn8, [x11]
    .inst 0xa0240164  // st1b {z4.b-z5.b}, pn8, [x11, x4]
    b TILE8_Dz_End

    TILE8_STORE24:
    add x11, x6, x4, LSL #1
    .inst 0xa06000d8  // st1b {z24.b-z25.b}, pn8, [x6]
    .inst 0xa02400dc  // st1b {z28.b-z29.b}, pn8, [x6, x4]
    .inst 0xa0600160  // st1b {z0.b-z1.b}, pn8, [x11]
    b TILE8_Dz_End

    TILE8_STORE16:
    .inst 0xa06000d8  // st1b {z24.b-z25.b}, pn8, [x6]
    .inst 0xa02400dc  // st1b {z28.b-z29.b}, pn8, [x6, x4]
    b TILE8_Dz_End

    TILE8_STORE8:
    .inst 0xa06000d8  // st1b {z24.b-z25.b}, pn8, [x6]
    b TILE8_Dz_End
    

    TILE8_Dz_End:
    sub x24, x24, #4
    add x6, x6, x4, LSL #2

    cmp x24, #0
    ble TILE8_End
    // revert input scale/kernelSum
    mov x13, x19
    mov x23, x21
    cbz x27, LoopDz_TILE8
    REVERT_INPUT_DEQUANT_BIAS x27, x15, x26, x25
    b LoopDz_TILE8

    TILE8_End:
    // update realE/dst/src/inputScale/inputKernelSum
    sub x7, x7, #8
    cbz x7, End
    add x0, x0, #128
    add x1, x1, #32
    add x13, x19, #32
    add x23, x21, #32
    // update inputBias/weightKernelSum
    cbz x27, TILE_4
    REVERT_INPUT_DEQUANT_BIAS x27, x15, x26, x25
    REVERT_WEIGHT_KERNEL_SUM x28, x11, x26, x5
    add x27, x27, #32

TILE_4:
cmp x7, #4
blt TILE_2

    mov x24, x5      // dst_depth_quad
    mov x6, x0       // dst
    mov x12, x2      // weight
    mov x20, x9      // bias
    mov x19, x13      // input kernel sum
    mov x21, x23     // input dequant scale

LoopDz_TILE4:
    mov x11, x1             // src
    mov x15, #0             // blockid
.inst 0xc00800ff  // zero {za}
TILE4_BLOCKNUM:
    mov x10, x3             // src_depth_quad

    .inst 0xc0080033  // zero {za0.s, za1.s}

LoopSz_TILE4:
    .inst 0xa400b560  // ld1b {z0.b}, p5/z, [x11]       // src
    .inst 0xa400ad81  // ld1b {z1.b}, p3/z, [x12]       // weight
    // int4->int8
    .inst 0xc08a4022  // luti4 {z2.b-z3.b}, zt0, z1[0]
    // matmul
    .inst 0xa0827400  // smopa za0.s, p5/m, p3/m, z0.b, z2.b
    .inst 0xa0837401  // smopa za1.s, p5/m, p3/m, z0.b, z3.b
    subs x10, x10, #1
    add x11, x11, x22
    .inst 0x042c502c  // addvl x12, x12, #1

bne LoopSz_TILE4

    .inst 0xa040c180  // ld1w {z0.s-z3.s}, pn8/z, [x12]  // weight scale&bias
    .inst 0xa400b5be  // ld1b {z30.b}, p5/z, [x13]    // input kernel sum
    .inst 0xa400b6ff  // ld1b {z31.b}, p5/z, [x23]   // input scale
    .inst 0x042c508c  // addvl x12, x12, #4
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
    .inst 0x808277c2  // fmopa za2.s, p5/m, p3/m, z30.s, z2.s
    .inst 0x808377c3  // fmopa za3.s, p5/m, p3/m, z30.s, z3.s

    // inputScale x weightScale -> [16,16]
    .inst 0x808077e0  // fmopa za0.s, p5/m, p3/m, z31.s, z0.s

    mov w10, #2
    mov w8, #0
    add x15, x15, #1 // block++
    .inst 0x053ecc3e  // mov z30.b, p3/m, z1.b  // copy 16 weight scale

    cbz x27, TILE4_DEQUANT
    .inst 0xa400b762  // ld1b {z2.b}, p5/z, [x27]    // input dequant bias
    .inst 0xa0404380  // ld1w {z0.s, z1.s}, pn8/z, [x28]    // weight kernel sum
    .inst 0x80807442  // fmopa za2.s, p5/m, p3/m, z2.s, z0.s
    .inst 0x80817443  // fmopa za3.s, p5/m, p3/m, z2.s, z1.s
    add x27, x27, x25
    add x23, x23, x25 // input scale
    .inst 0x043c505c  // addvl x28, x28, #2

    TILE4_DEQUANT:
    // extract scale from za0.s
    .inst 0xc0060c14  // mova {z20.s-z23.s}, za.s[w8, 0, VGx4]   // z20-z23: e=0, e=4, e=8, e=12
    .inst 0xc0060c98  // mova {z24.s-z27.s}, za.s[w8, 4, VGx4]   // z24-z27: e=1, e=5, e=9, e=13
    mov w8, #8
    .inst 0xc0060c00  // mova {z0.s-z3.s}, za.s[w8, 0, VGx4]   // z0-z3: e=2, e=6, e=10, e=14

    // accumulate to za2.s
    .inst 0xc1b55880  // fmla za.s[w10, 0, VGx4], {z4.s-z7.s}, {z20.s-z23.s}   // za, row:1,17,33,49
    .inst 0xc1b95904  // fmla za.s[w10, 4, VGx4], {z8.s-z11.s}, {z24.s-z27.s}   // za, row: 5,21,37,53
    mov w10, #10
    .inst 0xc0060c94  // mova {z20.s-z23.s}, za.s[w8, 4, VGx4]   // z20-z23: e=3, e=7, e=11, e=15
    .inst 0xc1a15980  // fmla za.s[w10, 0, VGx4], {z12.s-z15.s}, {z0.s-z3.s}
    .inst 0xc1b55a04  // fmla za.s[w10, 4, VGx4], {z16.s-z19.s}, {z20.s-z23.s}

    // oc:16-31: extract int32_t vectors from za1.s
    mov w8, #1
    mov w10, #9
    .inst 0xc0060c04  // mova {z4.s-z7.s}, za.s[w8, 0, VGx4]   // z4: e=0(za0h.s[0]), z5: e=4(za0h.s[4], z6: e=8(za0h.s[8]), z7: e=12(za0h.s[12), VG=512bit/32bit
    .inst 0xc0060c88  // mova {z8.s-z11.s}, za.s[w8, 4, VGx4]   // z8: e=1, z9: e=5, z10: e=9, z11: e=13
    .inst 0xc0064c0c  // mova {z12.s-z15.s}, za.s[w10, 0, VGx4]   // z12: e=2, z13: e=6, z14: e=10, z15: e=14
    .inst 0xc0064c90  // mova {z16.s-z19.s}, za.s[w10, 4, VGx4]   // z16: e=3, z17: e=7, z18: e=11, z19: e=15

    .inst 0xc132e084  // scvtf {z4.s-z7.s}, {z4.s-z7.s}
    .inst 0xc132e108  // scvtf {z8.s-z11.s}, {z8.s-z11.s}
    .inst 0xc132e18c  // scvtf {z12.s-z15.s}, {z12.s-z15.s}
    .inst 0xc132e210  // scvtf {z16.s-z19.s}, {z16.s-z19.s}
    .inst 0xc0080022  // zero {za1.s}

    // inputScale x weightScale -> [16,16]
    .inst 0x809e6fe1  // fmopa za1.s, p3/m, p3/m, z31.s, z30.s
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
    beq TILE4_POST
    b TILE4_BLOCKNUM

    TILE4_POST:
    cbz x9, TILE4_STORE
    .inst 0xa0404280  // ld1w {z0.s, z1.s}, pn8/z, [x20]   // bias
    .inst 0x25b9ce02  // fmov z2.s, #1
    .inst 0x04345054  // addvl x20, x20, #2
    .inst 0x80806c42  // fmopa za2.s, p3/m, p3/m, z2.s, z0.s
    .inst 0x80816c43  // fmopa za3.s, p3/m, p3/m, z2.s, z1.s

    TILE4_STORE:
    /* oc:0~15 */
    mov w13, #0
    mov w15, #4
    .inst 0xc086a440  // mova {z0.s-z3.s}, za2v.s[w13, 0:3]
    .inst 0xc086e444  // mova {z4.s-z7.s}, za2v.s[w15, 0:3]
    mov w13, #8
    mov w15, #12
    .inst 0xc086a448  // mova {z8.s-z11.s}, za2v.s[w13, 0:3]
    .inst 0xc086e44c  // mova {z12.s-z15.s}, za2v.s[w15, 0:3]
 
    .inst 0xc120e030  // fcvtn z16.h, {z0.s-z1.s}   // (0,0)(0,1)(1,0)(1,1)...(15,0)(15,1)
    .inst 0xc120e071  // fcvtn z17.h, {z2.s-z3.s}   // (0,2)(0,3)(1,2)(1,3)...(15,2)(15,3)
    .inst 0xc120e0b2  // fcvtn z18.h, {z4.s-z5.s}   // (0,4)(0,5)(1,4)(1,5)...(15,4)(15,5)
    .inst 0xc120e0f3  // fcvtn z19.h, {z6.s-z7.s}   // (0,6)(0,7)(1,6)(1,7)...(15,6)(15,7)

    .inst 0xc120e134  // fcvtn z20.h, {z8.s-z9.s}
    .inst 0xc120e175  // fcvtn z21.h, {z10.s-z11.s}
    .inst 0xc120e1b6  // fcvtn z22.h, {z12.s-z13.s}
    .inst 0xc120e1f7  // fcvtn z23.h, {z14.s-z15.s}

    /* oc:16~31 */
    mov w13, #0
    mov w15, #4
    .inst 0xc086a460  // mova {z0.s-z3.s}, za3v.s[w13, 0:3]
    .inst 0xc086e464  // mova {z4.s-z7.s}, za3v.s[w15, 0:3]
    mov w13, #8
    mov w15, #12
    .inst 0xc086a468  // mova {z8.s-z11.s}, za3v.s[w13, 0:3]
    .inst 0xc086e46c  // mova {z12.s-z15.s}, za3v.s[w15, 0:3]

    .inst 0xc1b6e218  // zip {z24.s-z27.s}, {z16.s-z19.s}
    .inst 0xc1b6e29c  // zip {z28.s-z31.s}, {z20.s-z23.s}

    .inst 0xc120e030  // fcvtn z16.h, {z0.s-z1.s}   // (0,0)(0,1)(1,0)(1,1)...(15,0)(15,1)
    .inst 0xc120e071  // fcvtn z17.h, {z2.s-z3.s}   // (0,2)(0,3)(1,2)(1,3)...(15,2)(15,3)
    .inst 0xc120e0b2  // fcvtn z18.h, {z4.s-z5.s}   // (0,4)(0,5)(1,4)(1,5)...(15,4)(15,5)
    .inst 0xc120e0f3  // fcvtn z19.h, {z6.s-z7.s}   // (0,6)(0,7)(1,6)(1,7)...(15,6)(15,7)

    .inst 0xc120e134  // fcvtn z20.h, {z8.s-z9.s}
    .inst 0xc120e175  // fcvtn z21.h, {z10.s-z11.s}
    .inst 0xc120e1b6  // fcvtn z22.h, {z12.s-z13.s}
    .inst 0xc120e1f7  // fcvtn z23.h, {z14.s-z15.s}

    .inst 0x84c0adc8  // ld1rh {z8.h}, p3/z, [x14]
    .inst 0x84c1adc9  // ld1rh {z9.h}, p3/z, [x14, #2]

    .inst 0xc1b6e200  // zip {z0.s-z3.s}, {z16.s-z19.s}
    .inst 0xc1b6e284  // zip {z4.s-z7.s}, {z20.s-z23.s}

    .inst 0xc169c918  // fclamp {z24.h-z27.h}, z8.h, z9.h
    .inst 0xc169c91c  // fclamp {z28.h-z31.h}, z8.h, z9.h
    .inst 0xc169c900  // fclamp {z0.h-z3.h}, z8.h, z9.h
    .inst 0xc169c904  // fclamp {z4.h-z7.h}, z8.h, z9.h

    cmp x24, #4
    bge TILE4_STORE32
    cmp x24, #1
    beq TILE4_STORE8
    cmp x24, #2
    beq TILE4_STORE16
    cmp x24, #3
    beq TILE4_STORE24

    TILE4_STORE32:
    add x11, x6, x4, LSL #1
    .inst 0xe400ecd8  // st1b {z24.b}, p3, [x6]
    .inst 0xe4044cdc  // st1b {z28.b}, p3, [x6, x4]
    .inst 0xe400ed60  // st1b {z0.b}, p3, [x11]
    .inst 0xe4044d64  // st1b {z4.b}, p3, [x11, x4]
    b TILE4_Dz_End

    TILE4_STORE24:
    add x11, x6, x4, LSL #1
    .inst 0xe400ecd8  // st1b {z24.b}, p3, [x6]
    .inst 0xe4044cdc  // st1b {z28.b}, p3, [x6, x4]
    .inst 0xe400ed60  // st1b {z0.b}, p3, [x11]
    b TILE4_Dz_End

    TILE4_STORE16:
    .inst 0xe400ecd8  // st1b {z24.b}, p3, [x6]
    .inst 0xe4044cdc  // st1b {z28.b}, p3, [x6, x4]
    b TILE4_Dz_End

    TILE4_STORE8:
    .inst 0xe400ecd8  // st1b {z24.b}, p3, [x6]
    b TILE4_Dz_End
    

    TILE4_Dz_End:
    sub x24, x24, #4
    add x6, x6, x4, LSL #2

    cmp x24, #0
    ble TILE4_End
    // revert input scale/kernelSum
    mov x13, x19
    mov x23, x21
    cbz x27, LoopDz_TILE4
    REVERT_INPUT_DEQUANT_BIAS x27, x15, x26, x25
    b LoopDz_TILE4

    TILE4_End:
    // update realE/dst/src/inputScale/inputKernelSum
    sub x7, x7, #4
    cbz x7, End
    add x0, x0, #64
    add x1, x1, #16
    add x13, x19, #16
    add x23, x21, #16
    // update inputBias/weightKernelSum
    cbz x27, TILE_2
    REVERT_INPUT_DEQUANT_BIAS x27, x15, x26, x25
    REVERT_WEIGHT_KERNEL_SUM x28, x11, x26, x5
    add x27, x27, #16

TILE_2:
    .inst 0x84c0adde  // ld1rh {z30.h}, p3/z, [x14]
    .inst 0x84c1addf  // ld1rh {z31.h}, p3/z, [x14, #2]

    cmp x7, #2
    blt TILE_1
    mov x24, x5 // dst_depth_quad
    mov x6, x0 // dst
    mov x12, x2 // weight
    mov x20, x9 // bias
    mov x19, x13      // input kernel sum
    mov x21, x23     // input dequant scale

    .inst 0x2518e107  // ptrue p7.b, #8

LoopDz_TILE2:
    mov x11, x1             // src
    mov x15, x26
    .inst 0xa040429c  // ld1w {z28.s, z29.s}, pn8/z, [x20]   // bias
    .inst 0x04345054  // addvl x20, x20, #2
    mov w8, #0
    .inst 0x053acf9a  // mov z26.b, p3/m, z28.b
    .inst 0x053bcfbb  // mov z27.b, p3/m, z29.b
TILE2_BLOCKNUM:
    mov x10, x3             // src_depth_quad

.inst 0xc00800ff  // zero {za}

    LoopSz_TILE2:
        .inst 0xa4003d62  // ld1rqb {z2.b}, p7/z, [x11]       // src
        .inst 0xa400ad80  // ld1b {z0.b}, p3/z, [x12]       // weight
        // int4->int8
        .inst 0xc08a4004  // luti4 {z4.b-z5.b}, zt0, z0[0]
        // matmul
        .inst 0xc15210a0  // sdot za.s[w8, 0, VGx2], {z4.b-z5.b}, z2.b[0]
        .inst 0xc15214a4  // sdot za.s[w8, 4, VGx2], {z4.b-z5.b}, z2.b[1]
        subs x10, x10, #1
        add x11, x11, x22
        .inst 0x042c502c  // addvl x12, x12, #1

    bne LoopSz_TILE2

    sub x15, x15, #1
    .inst 0xc0060808  // mova {z8.s-z9.s}, za.s[w8, 0, VGx2]
    .inst 0xc006088a  // mova {z10.s-z11.s}, za.s[w8, 4, VGx2]
    .inst 0xc132e108  // scvtf {z8.s-z11.s}, {z8.s-z11.s}

    .inst 0xa040c180  // ld1w {z0.s-z3.s}, pn8/z, [x12]    // weight scale&bias
    .inst 0xa4003da4  // ld1rqb {z4.b}, p7/z, [x13]    // input kernel sum
    .inst 0x8540cee5  // ld1rw {z5.s}, p3/z, [x23]   // input kernel scale
    .inst 0x8541ceee  // ld1rw {z14.s}, p3/z, [x23, #4]   // input kernel scale

    .inst 0x65850806  // fmul z6.s, z0.s, z5.s  // e0
    .inst 0x65850827  // fmul z7.s, z1.s, z5.s
    .inst 0x658e080c  // fmul z12.s, z0.s, z14.s  // e1
    .inst 0x658e082d  // fmul z13.s, z1.s, z14.s

    .inst 0x64a4005a  // fmla z26.s, z2.s, z4.s[0]  // e0
    .inst 0x64a4007b  // fmla z27.s, z3.s, z4.s[0]
    .inst 0x64ac005c  // fmla z28.s, z2.s, z4.s[1]  // e1
    .inst 0x64ac007d  // fmla z29.s, z3.s, z4.s[1]

    .inst 0x65a60d1a  // fmla z26.s, p3/m, z8.s, z6.s
    .inst 0x65a70d3b  // fmla z27.s, p3/m, z9.s, z7.s
    .inst 0x65ac0d5c  // fmla z28.s, p3/m, z10.s, z12.s
    .inst 0x65ad0d7d  // fmla z29.s, p3/m, z11.s, z13.s
    .inst 0x042c508c  // addvl x12, x12, #4
    add x13, x13, x22

    cbz x27, TILE2_ADD_DSTV
    .inst 0xa4003f65  // ld1rqb {z5.b}, p7/z, [x27]   // input dequant bias
    .inst 0xa0404388  // ld1w {z8.s, z9.s}, pn8/z, [x28]    // weight kernel sum
    .inst 0x64a5011a  // fmla z26.s, z8.s, z5.s[0]
    .inst 0x64a5013b  // fmla z27.s, z9.s, z5.s[0]
    .inst 0x64ad011c  // fmla z28.s, z8.s, z5.s[1]
    .inst 0x64ad013d  // fmla z29.s, z9.s, z5.s[1]
    add x27, x27, x25
    add x23, x23, x25
    .inst 0x043c505c  // addvl x28, x28, #2

    TILE2_ADD_DSTV:
    cbnz x15, TILE2_BLOCKNUM

    TILE2_STORE:
    .inst 0xc120e342  // fcvt z2.h, {z26.s-z27.s}
    .inst 0xc120e383  // fcvt z3.h, {z28.s-z29.s}
    .inst 0xc17fc3c2  // fclamp {z2.h-z3.h}, z30.h, z31.h
    .inst 0xc123d440  // zip {z0.q, z1.q}, z2.q, z3.q

    cmp x24, #4
    bge TILE2_STORE32
    cmp x24, #3
    beq TILE2_STORE24
    cmp x24, #2
    beq TILE2_STORE16

    .inst 0xe400e8c0  // st1b {z0.b}, p2, [x6]
    b TILE2_Dz_End

    TILE2_STORE16:
    add x11, x6, x4
    .inst 0x05b02002  // dup z2.q, z0.q[2]
    .inst 0x05f02003  // dup z3.q, z0.q[3]
    .inst 0xe400e8c0  // st1b {z0.b}, p2, [x6]
    .inst 0xe400f562  // st1b {z2.b}, p5, [x11]
    add x11, x11, #16
    .inst 0xe400f563  // st1b {z3.b}, p5, [x11]
    b TILE2_Dz_End

    TILE2_STORE24:
    add x11, x6, x4
    add x10, x6, x4, LSL #1
    .inst 0x05b02002  // dup z2.q, z0.q[2]
    .inst 0x05f02003  // dup z3.q, z0.q[3]
    .inst 0xe400e8c0  // st1b {z0.b}, p2, [x6]
    .inst 0xe400f562  // st1b {z2.b}, p5, [x11]
    add x11, x11, #16
    .inst 0xe400f563  // st1b {z3.b}, p5, [x11]
    .inst 0xe400e941  // st1b {z1.b}, p2, [x10]
    b TILE2_Dz_End

    TILE2_STORE32:
    add x11, x6, x4
    add x10, x6, x4, LSL #1
    add x13, x11, x4, LSL #1
    .inst 0x05b02002  // dup z2.q, z0.q[2]
    .inst 0x05f02003  // dup z3.q, z0.q[3]
    .inst 0x05b02024  // dup z4.q, z1.q[2]
    .inst 0x05f02025  // dup z5.q, z1.q[3]
    .inst 0xe400e8c0  // st1b {z0.b}, p2, [x6]
    .inst 0xe400f562  // st1b {z2.b}, p5, [x11]
    add x11, x11, #16
    .inst 0xe400f563  // st1b {z3.b}, p5, [x11]
    .inst 0xe400e941  // st1b {z1.b}, p2, [x10]
    .inst 0xe400f5a4  // st1b {z4.b}, p5, [x13]
    add x13, x13, #16
    .inst 0xe400f5a5  // st1b {z5.b}, p5, [x13]

    TILE2_Dz_End:
    sub x24, x24, #4
    cmp x24, #0
    ble TILE2_End
    add x6, x6, x4, LSL #2
    mov x13, x19
    mov x23, x21
    cbz x27, LoopDz_TILE2
    REVERT_INPUT_DEQUANT_BIAS x27, x15, x26, x25
    b LoopDz_TILE2

    TILE2_End:
    // update realE/dst/src/inputScale/inputKernelSum
    sub x7, x7, #2
    cbz x7, End
    add x0, x0, #32
    add x1, x1, #8
    add x13, x19, #8
    add x23, x21, #8
    // update inputBias/weightKernelSum
    cbz x27, TILE_1
    REVERT_INPUT_DEQUANT_BIAS x27, x15, x26, x25
    REVERT_WEIGHT_KERNEL_SUM x28, x11, x26, x5
    add x27, x27, #8

TILE_1:
    cmp x7, #1
    blt End
    mov x24, x5 // dst_depth_quad
    mov x6, x0 // dst
    mov x12, x2 // weight
    mov x20, x9 // bias
    mov x19, x13      // input kernel sum
    mov x21, x23     // input dequant scale

LoopDz_TILE1:
    mov x11, x1             // src
    mov x15, x26
    .inst 0xa040429c  // ld1w {z28.s, z29.s}, pn8/z, [x20]   // bias
    .inst 0x04345054  // addvl x20, x20, #2
    mov w8, #0
TILE1_BLOCKNUM:
    mov x10, x3             // src_depth_quad

.inst 0xc00800ff  // zero {za}

    LoopSz_TILE_1:
        .inst 0xa4002162  // ld1rqb {z2.b}, p0/z, [x11]       // src
        .inst 0xa400ad80  // ld1b {z0.b}, p3/z, [x12]       // weight
        // int4->int8
        .inst 0xc08a4004  // luti4 {z4.b-z5.b}, zt0, z0[0]
        // matmul
        .inst 0xc15210a0  // sdot za.s[w8, 0, VGx2], {z4.b-z5.b}, z2.b[0]
        subs x10, x10, #1
        add x11, x11, x22
        .inst 0x042c502c  // addvl x12, x12, #1

    bne LoopSz_TILE_1

LoopSzEnd_TILE_1:
    sub x15, x15, #1
    .inst 0xc0060808  // mova {z8.s-z9.s}, za.s[w8, 0, VGx2]
    .inst 0x6594ad08  // scvtf z8.s, p3/m, z8.s
    .inst 0x6594ad29  // scvtf z9.s, p3/m, z9.s

    .inst 0xa040c180  // ld1w {z0.s-z3.s}, pn8/z, [x12]    // weight scale&bias
    .inst 0x8540cda4  // ld1rw {z4.s}, p3/z, [x13]    // input kernel sum
    .inst 0x8540cee5  // ld1rw {z5.s}, p3/z, [x23]   // input kernel scale

    .inst 0x65850800  // fmul z0.s, z0.s, z5.s
    .inst 0x65850821  // fmul z1.s, z1.s, z5.s
    .inst 0x64a4005c  // fmla z28.s, z2.s, z4.s[0]
    .inst 0x64a4007d  // fmla z29.s, z3.s, z4.s[0]
    .inst 0x65a00d1c  // fmla z28.s, p3/m, z8.s, z0.s
    .inst 0x65a10d3d  // fmla z29.s, p3/m, z9.s, z1.s
    .inst 0x042c508c  // addvl x12, x12, #4
    add x13, x13, x22

    cbz x27, TILE1_ADD_DSTV
    .inst 0x8540cf65  // ld1rw {z5.s}, p3/z, [x27]   // input dequant bias
    .inst 0xa0404388  // ld1w {z8.s, z9.s}, pn8/z, [x28]    // weight kernel sum
    .inst 0x64a5011c  // fmla z28.s, z8.s, z5.s[0]
    .inst 0x64a5013d  // fmla z29.s, z9.s, z5.s[0]
    add x27, x27, x25
    add x23, x23, x25
    .inst 0x043c505c  // addvl x28, x28, #2

    TILE1_ADD_DSTV:
    cmp x15, #0
    bne TILE1_BLOCKNUM

    TILE1_STORE:
    .inst 0xc120e382  // fcvt z2.h, {z28.s-z29.s}
    .inst 0x647f27c2  // fclamp z2.h, z30.h, z31.h

    cmp x24, #4
    bge TILE1_STORE32
    cmp x24, #3
    beq TILE1_STORE24
    cmp x24, #2
    beq TILE1_STORE16

    .inst 0xe400f4c2  // st1b {z2.b}, p5, [x6]
    b Tile1LoopEnd

    TILE1_STORE16:
    .inst 0x05702043  // dup z3.q, z2.q[1]
    .inst 0xe400f4c2  // st1b {z2.b}, p5, [x6]
    .inst 0xe40454c3  // st1b {z3.b}, p5, [x6, x4]
    b Tile1LoopEnd

    TILE1_STORE24:
    add x11, x6, x4, LSL #1
    .inst 0x05702043  // dup z3.q, z2.q[1]
    .inst 0x05b02044  // dup z4.q, z2.q[2]
    .inst 0xe400f4c2  // st1b {z2.b}, p5, [x6]
    .inst 0xe40454c3  // st1b {z3.b}, p5, [x6, x4]
    .inst 0xe400f564  // st1b {z4.b}, p5, [x11]
    b Tile1LoopEnd

    TILE1_STORE32:
    add x11, x6, x4, LSL #1
    .inst 0x05702043  // dup z3.q, z2.q[1]
    .inst 0x05b02044  // dup z4.q, z2.q[2]
    .inst 0x05f02045  // dup z5.q, z2.q[3]
    .inst 0xe400f4c2  // st1b {z2.b}, p5, [x6]
    .inst 0xe40454c3  // st1b {z3.b}, p5, [x6, x4]
    .inst 0xe400f564  // st1b {z4.b}, p5, [x11]
    .inst 0xe4045565  // st1b {z5.b}, p5, [x11, x4]

    Tile1LoopEnd:
    sub x24, x24, #4
    cmp x24, #0
    ble TILE1_End
    add x6, x6, x4, LSL #2
    mov x13, x19
    mov x23, x21
    cbz x27, LoopDz_TILE1
    REVERT_INPUT_DEQUANT_BIAS x27, x15, x26, x25
    b LoopDz_TILE1

    TILE1_End:
    // update realE/dst/src/inputScale/inputKernelSum
    sub x7, x7, #1
    cbz x7, End
    add x0, x0, #16
    add x1, x1, #4
    add x13, x19, #4
    add x23, x21, #4
    // update inputBias/weightKernelSum
    cbz x27, TILE_1
    REVERT_INPUT_DEQUANT_BIAS x27, x15, x26, x25
    REVERT_WEIGHT_KERNEL_SUM x28, x11, x26, x5
    add x27, x27, #4
    b TILE_1
    

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
