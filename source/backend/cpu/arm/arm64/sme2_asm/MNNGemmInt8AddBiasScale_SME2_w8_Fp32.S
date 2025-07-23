//  MNNGemmInt8AddBiasScale_SME2_w8_Fp32.S
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
// blocknum * up_div(ocDiv4, 4) * sizeof(float) * 16
// rg2: blocknum, rg3:ocDiv4, rg0: address of weightKernelSum
add \rg1, \rg3, #3
lsr \rg1, \rg1, #2
mul \rg1, \rg2, \rg1
sub \rg0, \rg0, \rg1, LSL #6 // revert weight kernel sum
.endm

.macro ROUND_TO_NEAREST_INT
.inst 0x65912c04 // fcmlt	p4.s, p3/z, z0.s, #0.0
.inst 0x65999000 // fsub	z0.s, p4/m, z0.s, #0.5
.inst 0x65912c24 // fcmlt	p4.s, p3/z, z1.s, #0.0
.inst 0x65999001 // fsub	z1.s, p4/m, z1.s, #0.5
.inst 0x65912c44 // fcmlt	p4.s, p3/z, z2.s, #0.0
.inst 0x65999002 // fsub	z2.s, p4/m, z2.s, #0.5
.inst 0x65912c64 // fcmlt	p4.s, p3/z, z3.s, #0.0
.inst 0x65999003 // fsub	z3.s, p4/m, z3.s, #0.5
.inst 0x65912c84 // fcmlt	p4.s, p3/z, z4.s, #0.0
.inst 0x65999004 // fsub	z4.s, p4/m, z4.s, #0.5
.inst 0x65912ca4 // fcmlt	p4.s, p3/z, z5.s, #0.0
.inst 0x65999005 // fsub	z5.s, p4/m, z5.s, #0.5
.inst 0x65912cc4 // fcmlt	p4.s, p3/z, z6.s, #0.0
.inst 0x65999006 // fsub	z6.s, p4/m, z6.s, #0.5
.inst 0x65912ce4 // fcmlt	p4.s, p3/z, z7.s, #0.0
.inst 0x65999007 // fsub	z7.s, p4/m, z7.s, #0.5
.inst 0x65912d04 // fcmlt	p4.s, p3/z, z8.s, #0.0
.inst 0x65999008 // fsub	z8.s, p4/m, z8.s, #0.5
.inst 0x65912d24 // fcmlt	p4.s, p3/z, z9.s, #0.0
.inst 0x65999009 // fsub	z9.s, p4/m, z9.s, #0.5
.inst 0x65912d44 // fcmlt	p4.s, p3/z, z10.s, #0.0
.inst 0x6599900a // fsub	z10.s, p4/m, z10.s, #0.5
.inst 0x65912d64 // fcmlt	p4.s, p3/z, z11.s, #0.0
.inst 0x6599900b // fsub	z11.s, p4/m, z11.s, #0.5
.inst 0x65912d84 // fcmlt	p4.s, p3/z, z12.s, #0.0
.inst 0x6599900c // fsub	z12.s, p4/m, z12.s, #0.5
.inst 0x65912da4 // fcmlt	p4.s, p3/z, z13.s, #0.0
.inst 0x6599900d // fsub	z13.s, p4/m, z13.s, #0.5
.inst 0x65912dc4 // fcmlt	p4.s, p3/z, z14.s, #0.0
.inst 0x6599900e // fsub	z14.s, p4/m, z14.s, #0.5
.inst 0x65912de4 // fcmlt	p4.s, p3/z, z15.s, #0.0
.inst 0x6599900f // fsub	z15.s, p4/m, z15.s, #0.5
.inst 0x65902c04 // fcmge	p4.s, p3/z, z0.s, #0.0
.inst 0x65989000 // fadd	z0.s, p4/m, z0.s, #0.5
.inst 0x65902c24 // fcmge	p4.s, p3/z, z1.s, #0.0
.inst 0x65989001 // fadd	z1.s, p4/m, z1.s, #0.5
.inst 0x65902c44 // fcmge	p4.s, p3/z, z2.s, #0.0
.inst 0x65989002 // fadd	z2.s, p4/m, z2.s, #0.5
.inst 0x65902c64 // fcmge	p4.s, p3/z, z3.s, #0.0
.inst 0x65989003 // fadd	z3.s, p4/m, z3.s, #0.5
.inst 0x65902c84 // fcmge	p4.s, p3/z, z4.s, #0.0
.inst 0x65989004 // fadd	z4.s, p4/m, z4.s, #0.5
.inst 0x65902ca4 // fcmge	p4.s, p3/z, z5.s, #0.0
.inst 0x65989005 // fadd	z5.s, p4/m, z5.s, #0.5
.inst 0x65902cc4 // fcmge	p4.s, p3/z, z6.s, #0.0
.inst 0x65989006 // fadd	z6.s, p4/m, z6.s, #0.5
.inst 0x65902ce4 // fcmge	p4.s, p3/z, z7.s, #0.0
.inst 0x65989007 // fadd	z7.s, p4/m, z7.s, #0.5
.inst 0x65902d04 // fcmge	p4.s, p3/z, z8.s, #0.0
.inst 0x65989008 // fadd	z8.s, p4/m, z8.s, #0.5
.inst 0x65902d24 // fcmge	p4.s, p3/z, z9.s, #0.0
.inst 0x65989009 // fadd	z9.s, p4/m, z9.s, #0.5
.inst 0x65902d44 // fcmge	p4.s, p3/z, z10.s, #0.0
.inst 0x6598900a // fadd	z10.s, p4/m, z10.s, #0.5
.inst 0x65902d64 // fcmge	p4.s, p3/z, z11.s, #0.0
.inst 0x6598900b // fadd	z11.s, p4/m, z11.s, #0.5
.inst 0x65902d84 // fcmge	p4.s, p3/z, z12.s, #0.0
.inst 0x6598900c // fadd	z12.s, p4/m, z12.s, #0.5
.inst 0x65902da4 // fcmge	p4.s, p3/z, z13.s, #0.0
.inst 0x6598900d // fadd	z13.s, p4/m, z13.s, #0.5
.inst 0x65902dc4 // fcmge	p4.s, p3/z, z14.s, #0.0
.inst 0x6598900e // fadd	z14.s, p4/m, z14.s, #0.5
.inst 0x65902de4 // fcmge	p4.s, p3/z, z15.s, #0.0
.inst 0x6598900f // fadd	z15.s, p4/m, z15.s, #0.5
.endm


asm_function MNNGemmInt8AddBiasScale_SME2_w8_Fp32
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
//void MNNGemmInt8AddBiasScale_SME2_w8_Fp32(int8_t* dst, const int8_t* src,
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


/* initialize predicates */
mov x22, #12
.inst 0x2518e080  // ptrue p0.b, #4          // first 4 bytes
.inst 0x2518e106  // ptrue p6.b, #8          // first 8 bytes
.inst 0x253617e1  // whilelt p1.b, xzr, x22
.inst 0x2518e127  // ptrue p7.b, vl16
.inst 0x0460e3f8  // cnth x24
.inst 0x2518e125  // ptrue p5.b, vl16
.inst 0x253817e2  // whilelt p2.b, xzr, x24  // previous 32 bytes valid
.inst 0x25015ee7  // eor p7.b, p7/z, p7.b, p1.b  // forth 4 bytes valid
.inst 0x25064621  // eor p1.b, p1/z, p1.b, p6.b  // third 4 bytes valid
.inst 0x2518e3e3  // ptrue p3.b              // 64 bytes valid
.inst 0x25207810  // ptrue pn8.b
.inst 0x25005ac6  // eor p6.b, p6/z, p6.b, p0.b  // second 4 bytes
.inst 0x2598e084  // ptrue p4.s, #4


lsl x22, x7, #2 // eDest * GEMM_INT8_SRC_UNIT
mov x25, 0       // inputBlockNum=1
cbz x27, reluRead
mov x25, x22         // input block quant: realDstCount * sizeof(float)

reluRead:
/* relu min/max*/
cbz x28, Int8Relu
.inst 0x8540cddd  // ld1rw {z29.s}, p3/z, [x14]
.inst 0x8541cdde  // ld1rw {z30.s}, p3/z, [x14, #4]
b TILE_16

Int8Relu:
add x14, x6, #16
.inst 0x84408dde  // ld1rb {z30.b}, p3/z, [x14]
.inst 0x84448ddd  // ld1rb {z29.b}, p3/z, [x14, #4]


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

    .inst 0xc0080055  // zero {za0.s, za2.s}

LoopSz_TILE16:
    .inst 0xa400ad60  // ld1b {z0.b}, p3/z, [x11]      // src
    .inst 0xa400ad81  // ld1b {z1.b}, p3/z, [x12]      // weight
    // matmul
    .inst 0xa0816c00 // smopa za0.s, p3/m, p3/m, z0.b, z1.b
    subs x10, x10, #1
    add x11, x11, x22
    .inst 0x042c502c  // addvl x12, x12, #1

bne LoopSz_TILE16

    // extract int32_t vectors from za0.s
    mov w8, #0
    .inst 0xc0060c04  // mova {z4.s-z7.s}, za.s[w8, 0, VGx4]  // z4: e=0(za0h.s[0]), z5: e=4(za0h.s[4], z6: e=8(za0h.s[8]), z7: e=12(za0h.s[12), VG=512bit/32bit
    .inst 0xc0060c88  // mova {z8.s-z11.s}, za.s[w8, 4, VGx4]  // z8: e=1, z9: e=5, z10: e=9, z11: e=13
    mov w8, #8
    .inst 0xc0060c0c  // mova {z12.s-z15.s}, za.s[w8, 0, VGx4]  // z12: e=2, z13: e=6, z14: e=10, z15: e=14
    .inst 0xc0060c90  // mova {z16.s-z19.s}, za.s[w8, 4, VGx4]  // z16: e=3, z17: e=7, z18: e=11, z19: e=15

    // weight scale&bias
    .inst 0xa0404180  // ld1w {z0.s-z1.s}, pn8/z, [x12]
    .inst 0xa540ada2  // ld1w {z2.s}, p3/z, [x13]   // input kernel sum
    .inst 0xa540aee3  // ld1w {z3.s}, p3/z, [x23]  // input scale
    .inst 0x042c504c  // addvl x12, x12, #2
    add x13, x13, x22
    add x23, x23, x25

    .inst 0xc132e084  // scvtf {z4.s-z7.s}, {z4.s-z7.s}
    .inst 0xc132e108  // scvtf {z8.s-z11.s}, {z8.s-z11.s}
    .inst 0xc132e18c  // scvtf {z12.s-z15.s}, {z12.s-z15.s}
    .inst 0xc132e210  // scvtf {z16.s-z19.s}, {z16.s-z19.s}

    // inputKernelSum x weightBias -> [16,16]
    .inst 0x80816c41 // fmopa za1.s, p3/m, p3/m, z2.s, z1.s
    // inputScale x weightScale -> [16,16]
    .inst 0x80806c62 // fmopa za2.s, p3/m, p3/m, z3.s, z0.s

    mov w10, #1
    mov w8, #2
    add x15, x15, #1

    cbz x27, TILE16_DEQUANT
    .inst 0xa540af60  // ld1w {z0.s}, p3/z, [x27]   // input dequant bias
    .inst 0xa540af81  // ld1w {z1.s}, p3/z, [x28]   // weight kernel sum
    .inst 0x80816c01 // fmopa za1.s, p3/m, p3/m, z0.s, z1.s
    add x27, x27, x25
    .inst 0x043c503c  // addvl x28, x28, #1

    TILE16_DEQUANT:
    // extract vector from za2.s
    .inst 0xc0060c14  // mova {z20.s-z23.s}, za.s[w8, 0, VGx4]  // z20-z23: e=0, e=4, e=8, e=12
    .inst 0xc0060c98  // mova {z24.s-z27.s}, za.s[w8, 4, VGx4]  // z24-z27: e=1, e=5, e=9, e=13
    mov w8, #10
    .inst 0xc0060c00  // mova {z0.s-z3.s}, za.s[w8, 0, VGx4]  // z0-z3: e=2, e=6, e=10, e=14

    // accumulate to za1.s
    .inst 0xc1b55880  // fmla za.s[w10, 0, VGx4], {z4.s-z7.s}, {z20.s-z23.s}  // za, row:1,17,33,49
    .inst 0xc1b95904  // fmla za.s[w10, 4, VGx4], {z8.s-z11.s}, {z24.s-z27.s}  // za, row: 5,21,37,53
    mov w10, #9
    .inst 0xc0060c94  // mova {z20.s-z23.s}, za.s[w8, 4, VGx4]  // z20-z23: e=3, e=7, e=11, e=15
    .inst 0xc1a15980  // fmla za.s[w10, 0, VGx4], {z12.s-z15.s}, {z0.s-z3.s}
    .inst 0xc1b55a04  // fmla za.s[w10, 4, VGx4], {z16.s-z19.s}, {z20.s-z23.s}

    cmp x15, x26
    beq TILE16_POST
    b TILE16_BLOCKNUM

    TILE16_POST:
    .inst 0xa540ae80  // ld1w {z0.s}, p3/z, [x20]  // bias
    .inst 0x25b9ce01  // fmov z1.s, #1
    add x20, x20, #64
    .inst 0x80806c21 // fmopa za1.s, p3/m, p3/m, z1.s, z0.s
    mov w14, #0
    mov w15, #4

    // (e,oc)
    // z0.s: (0,0)(1,0)(2,0)...(15,0)
    // z1.s: (0,1)(1,1)(2,1)...(15,1)
    // ...
    // z15.s:(0,15)(1,15)(2,15)...(15,15)
    .inst 0xc086c420  // mova {z0.s-z3.s}, za1v.s[w14, 0:3]
    .inst 0xc086e424  // mova {z4.s-z7.s}, za1v.s[w15, 0:3]
    mov w14, #8
    mov w15, #12
    .inst 0xc086c428  // mova {z8.s-z11.s}, za1v.s[w14, 0:3]
    .inst 0xc086e42c  // mova {z12.s-z15.s}, za1v.s[w15, 0:3]

    cbz x28, TILE16_Int8_Output

    TILE16_RELU:
    .inst 0xc1becba0  // fclamp {z0.s-z3.s}, z29.s, z30.s
    .inst 0xc1becba4  // fclamp {z4.s-z7.s}, z29.s, z30.s
    .inst 0xc1becba8  // fclamp {z8.s-z11.s}, z29.s, z30.s
    .inst 0xc1becbac  // fclamp {z12.s-z15.s}, z29.s, z30.s

    TILE16_STORE:
    .inst 0xc1b6e010  // zip {z16.s-z19.s}, {z0.s-z3.s}
    .inst 0xc1b6e094  // zip {z20.s-z23.s}, {z4.s-z7.s}
    .inst 0xc1b6e118  // zip {z24.s-z27.s}, {z8.s-z11.s}
    .inst 0xc1b6e180  // zip {z0.s-z3.s}, {z12.s-z15.s}

    cmp x24, #1
    beq TILE16_STORE4

    cmp x24, #2
    beq TILE16_STORE8

    cmp x24, #3
    beq TILE16_STORE12

    TILE16_STORE16:
    add x10, x6, x4 // +x4
    add x8, x6, x4, LSL #1 // + 2*x4
    add x14, x10, x4, LSL #1 // + 3*x4
    .inst 0xa06080d0  // st1b {z16.b-z19.b}, pn8, [x6]
    .inst 0xa0608154  // st1b {z20.b-z23.b}, pn8, [x10]
    .inst 0xa0608118  // st1b {z24.b-z27.b}, pn8, [x8]
    .inst 0xa06081c0  // st1b {z0.b-z3.b}, pn8, [x14]
    b TILE16_Dz_End

    TILE16_STORE4:
    .inst 0xa06080d0  // st1b {z16.b-z19.b}, pn8, [x6]
    b TILE16_Dz_End

    TILE16_STORE8:
    add x10, x6, x4 // +x4
    .inst 0xa06080d0  // st1b {z16.b-z19.b}, pn8, [x6]
    .inst 0xa0608154  // st1b {z20.b-z23.b}, pn8, [x10]
    b TILE16_Dz_End

    TILE16_STORE12:
    add x10, x6, x4 // +x4
    add x8, x6, x4, LSL #1 // + 2*x4
    .inst 0xa06080d0  // st1b {z16.b-z19.b}, pn8, [x6]
    .inst 0xa0608154  // st1b {z20.b-z23.b}, pn8, [x10]
    .inst 0xa0608118  // st1b {z24.b-z27.b}, pn8, [x8]
    b TILE16_Dz_End

    TILE16_Int8_Output:
    
    ROUND_TO_NEAREST_INT

    .inst 0xc131e000  // fcvtzs {z0.s-z3.s}, {z0.s-z3.s}
    .inst 0xc131e084  // fcvtzs {z4.s-z7.s}, {z4.s-z7.s}
    .inst 0xc131e108  // fcvtzs {z8.s-z11.s}, {z8.s-z11.s}
    .inst 0xc131e18c  // fcvtzs {z12.s-z15.s}, {z12.s-z15.s}

    .inst 0xc133e050  // sqcvtn z16.b, {z0.s-z3.s}
    .inst 0xc133e0d1  // sqcvtn z17.b, {z4.s-z7.s}
    .inst 0xc133e152  // sqcvtn z18.b, {z8.s-z11.s}
    .inst 0xc133e1d3  // sqcvtn z19.b, {z12.s-z15.s}
    .inst 0xc13ecfb0  // sclamp {z16.b-z19.b}, z29.b, z30.b
    
    cmp x24, #1
    beq TILE16_4

    cmp x24, #2
    beq TILE16_8

    cmp x24, #3
    beq TILE16_12

    add x10, x6, x4 // +x4
    add x8, x6, x4, LSL #1 // + 2*x4
    add x14, x10, x4, LSL #1 // + 3*x4
    .inst 0xe400ecd0  // st1b {z16.b}, p3, [x6]
    .inst 0xe400ed51  // st1b {z17.b}, p3, [x10]
    .inst 0xe400ed12  // st1b {z18.b}, p3, [x8]
    .inst 0xe400edd3  // st1b {z19.b}, p3, [x14]
    b TILE16_Dz_End

    TILE16_4:
    .inst 0xe400ecd0  // st1b {z16.b}, p3, [x6]
    b TILE16_Dz_End

    TILE16_8:
    add x10, x6, x4 // +x4
    .inst 0xe400ecd0  // st1b {z16.b}, p3, [x6]
    .inst 0xe400ed51  // st1b {z17.b}, p3, [x10]
    b TILE16_Dz_End

    TILE16_12:
    add x10, x6, x4 // +x4
    add x8, x6, x4, LSL #1 // + 2*x4
    .inst 0xe400ecd0  // st1b {z16.b}, p3, [x6]
    .inst 0xe400ed51  // st1b {z17.b}, p3, [x10]
    .inst 0xe400ed12  // st1b {z18.b}, p3, [x8]
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

    .inst 0xc0080055  // zero {za0.s, za2.s}

LoopSz_TILE12:
    .inst 0xa400bd60  // ld1b {z0.b}, p7/z, [x11]      // src
    .inst 0xa400ad81  // ld1b {z1.b}, p3/z, [x12]      // weight
    // matmul
    .inst 0xa0817c00 // smopa za0.s, p7/m, p3/m, z0.b, z1.b
    subs x10, x10, #1
    add x11, x11, x22
    .inst 0x042c502c  // addvl x12, x12, #1

bne LoopSz_TILE12

    // extract int32_t vectors from za0.s
    mov w8, #0
    .inst 0xc0060c04  // mova {z4.s-z7.s}, za.s[w8, 0, VGx4]  // z4: e=0(za0h.s[0]), z5: e=4(za0h.s[4], z6: e=8(za0h.s[8]), z7: e=12(za0h.s[12), VG=512bit/32bit
    .inst 0xc0060c88  // mova {z8.s-z11.s}, za.s[w8, 4, VGx4]  // z8: e=1, z9: e=5, z10: e=9, z11: e=13
    mov w8, #8
    .inst 0xc0060c0c  // mova {z12.s-z15.s}, za.s[w8, 0, VGx4]  // z12: e=2, z13: e=6, z14: e=10, z15: e=14
    .inst 0xc0060c90  // mova {z16.s-z19.s}, za.s[w8, 4, VGx4]  // z16: e=3, z17: e=7, z18: e=11, z19: e=15

    // weight scale&bias
    .inst 0xa0404180  // ld1w {z0.s-z1.s}, pn8/z, [x12]
    .inst 0xa400bda2  // ld1b {z2.b}, p7/z, [x13]   // input kernel sum
    .inst 0xa400bee3  // ld1b {z3.b}, p7/z, [x23]  // input scale
    .inst 0x042c504c  // addvl x12, x12, #2
    add x13, x13, x22
    add x23, x23, x25

    .inst 0xc132e084  // scvtf {z4.s-z7.s}, {z4.s-z7.s}
    .inst 0xc132e108  // scvtf {z8.s-z11.s}, {z8.s-z11.s}
    .inst 0xc132e18c  // scvtf {z12.s-z15.s}, {z12.s-z15.s}
    .inst 0xc132e210  // scvtf {z16.s-z19.s}, {z16.s-z19.s}

    // inputKernelSum x weightBias -> [16,16]
    .inst 0x80817c41 // fmopa za1.s, p7/m, p3/m, z2.s, z1.s
    // inputScale x weightScale -> [16,16]
    .inst 0x80807c62 // fmopa za2.s, p7/m, p3/m, z3.s, z0.s

    mov w10, #1
    mov w8, #2
    add x15, x15, #1

    cbz x27, TILE12_DEQUANT
    .inst 0xa400bf60  // ld1b {z0.b}, p7/z, [x27]   // input dequant bias
    .inst 0xa540af81  // ld1w {z1.s}, p3/z, [x28]   // weight kernel sum
    .inst 0x80817c01 // fmopa za1.s, p7/m, p3/m, z0.s, z1.s
    add x27, x27, x25
    .inst 0x043c503c  // addvl x28, x28, #1

    TILE12_DEQUANT:
    // extract vector from za2.s
    .inst 0xc0060c14  // mova {z20.s-z23.s}, za.s[w8, 0, VGx4]  // z20-z23: e=0, e=4, e=8, e=12
    .inst 0xc0060c98  // mova {z24.s-z27.s}, za.s[w8, 4, VGx4]  // z24-z27: e=1, e=5, e=9, e=13
    mov w8, #10
    .inst 0xc0060c00  // mova {z0.s-z3.s}, za.s[w8, 0, VGx4]  // z0-z3: e=2, e=6, e=10, e=14

    // accumulate to za1.s
    .inst 0xc1b55880  // fmla za.s[w10, 0, VGx4], {z4.s-z7.s}, {z20.s-z23.s}  // za, row:1,17,33,49
    .inst 0xc1b95904  // fmla za.s[w10, 4, VGx4], {z8.s-z11.s}, {z24.s-z27.s}  // za, row: 5,21,37,53
    mov w10, #9
    .inst 0xc0060c94  // mova {z20.s-z23.s}, za.s[w8, 4, VGx4]  // z20-z23: e=3, e=7, e=11, e=15
    .inst 0xc1a15980  // fmla za.s[w10, 0, VGx4], {z12.s-z15.s}, {z0.s-z3.s}
    .inst 0xc1b55a04  // fmla za.s[w10, 4, VGx4], {z16.s-z19.s}, {z20.s-z23.s}

    cmp x15, x26
    beq TILE12_POST
    b TILE12_BLOCKNUM

    TILE12_POST:
    .inst 0xa540ae80  // ld1w {z0.s}, p3/z, [x20]  // bias
    .inst 0x25b9ce01  // fmov z1.s, #1
    add x20, x20, #64
    .inst 0x80806c21 // fmopa za1.s, p3/m, p3/m, z1.s, z0.s
    mov w14, #0
    mov w15, #4

    .inst 0xc086c420  // mova {z0.s-z3.s}, za1v.s[w14, 0:3]
    .inst 0xc086e424  // mova {z4.s-z7.s}, za1v.s[w15, 0:3]
    mov w14, #8
    mov w15, #12
    .inst 0xc086c428  // mova {z8.s-z11.s}, za1v.s[w14, 0:3]
    .inst 0xc086e42c  // mova {z12.s-z15.s}, za1v.s[w15, 0:3]

    cbz x28, TILE12_Int8_Output
    mov x11, #192 // fp32 output: 12*4*sizeof(float)

    TILE12_RELU:
    .inst 0xc1becba0  // fclamp {z0.s-z3.s}, z29.s, z30.s
    .inst 0xc1becba4  // fclamp {z4.s-z7.s}, z29.s, z30.s
    .inst 0xc1becba8  // fclamp {z8.s-z11.s}, z29.s, z30.s
    .inst 0xc1becbac  // fclamp {z12.s-z15.s}, z29.s, z30.s

    TILE12_STORE:
    .inst 0xc1b6e010  // zip {z16.s-z19.s}, {z0.s-z3.s}
    .inst 0xc1b6e094  // zip {z20.s-z23.s}, {z4.s-z7.s}
    .inst 0xc1b6e118  // zip {z24.s-z27.s}, {z8.s-z11.s}
    .inst 0xc1b6e180  // zip {z0.s-z3.s}, {z12.s-z15.s}

    cmp x24, #1
    beq TILE12_STORE4

    cmp x24, #2
    beq TILE12_STORE8

    cmp x24, #3
    beq TILE12_STORE12

    TILE12_STORE16:
    add x10, x6, x4 // +x4
    add x8, x6, x4, LSL #1 // + 2*x4
    add x14, x8, x4 // + 3*x4
    .inst 0xa06000d0  // st1b {z16.b, z17.b}, pn8, [x6]
    .inst 0xe402ecd2  // st1b {z18.b}, p3, [x6, #2, MUL VL]
    .inst 0xa0600154  // st1b {z20.b, z21.b}, pn8, [x10]
    .inst 0xe402ed56  // st1b {z22.b}, p3, [x10, #2, MUL VL]
    .inst 0xa0600118  // st1b {z24.b, z25.b}, pn8, [x8]
    .inst 0xe402ed1a  // st1b {z26.b}, p3, [x8, #2, MUL VL]
    .inst 0xa06001c0  // st1b {z0.b, z1.b}, pn8, [x14]
    .inst 0xe402edc2  // st1b {z2.b}, p3, [x14, #2, MUL VL]
    b TILE12_Dz_End

    TILE12_STORE4:
    .inst 0xa06000d0  // st1b {z16.b, z17.b}, pn8, [x6]
    .inst 0xe402ecd2  // st1b {z18.b}, p3, [x6, #2, MUL VL]
    b TILE12_Dz_End

    TILE12_STORE8:
    add x10, x6, x4 // +x4
    .inst 0xa06000d0  // st1b {z16.b, z17.b}, pn8, [x6]
    .inst 0xe402ecd2  // st1b {z18.b}, p3, [x6, #2, MUL VL]
    .inst 0xa0600154  // st1b {z20.b, z21.b}, pn8, [x10]
    .inst 0xe402ed56  // st1b {z22.b}, p3, [x10, #2, MUL VL]
    b TILE12_Dz_End

    TILE12_STORE12:
    add x10, x6, x4 // +x4
    add x8, x6, x4, LSL #1 // + 2*x4
    .inst 0xa06000d0  // st1b {z16.b, z17.b}, pn8, [x6]
    .inst 0xe402ecd2  // st1b {z18.b}, p3, [x6, #2, MUL VL]
    .inst 0xa0600154  // st1b {z20.b, z21.b}, pn8, [x10]
    .inst 0xe402ed56  // st1b {z22.b}, p3, [x10, #2, MUL VL]
    .inst 0xa0600118  // st1b {z24.b, z25.b}, pn8, [x8]
    .inst 0xe402ed1a  // st1b {z26.b}, p3, [x8, #2, MUL VL]
    b TILE12_Dz_End


    TILE12_Int8_Output:
    mov x11, #48 // int8_t output: 12*4*sizeof(int8_t)
    
    ROUND_TO_NEAREST_INT

    .inst 0xc131e000  // fcvtzs {z0.s-z3.s}, {z0.s-z3.s}
    .inst 0xc131e084  // fcvtzs {z4.s-z7.s}, {z4.s-z7.s}
    .inst 0xc131e108  // fcvtzs {z8.s-z11.s}, {z8.s-z11.s}
    .inst 0xc131e18c  // fcvtzs {z12.s-z15.s}, {z12.s-z15.s}

    .inst 0xc133e050  // sqcvtn z16.b, {z0.s-z3.s}
    .inst 0xc133e0d1  // sqcvtn z17.b, {z4.s-z7.s}
    .inst 0xc133e152  // sqcvtn z18.b, {z8.s-z11.s}
    .inst 0xc133e1d3  // sqcvtn z19.b, {z12.s-z15.s}
    .inst 0xc13ecfb0  // sclamp {z16.b-z19.b}, z29.b, z30.b
    
    cmp x24, #1
    beq TILE12_4

    cmp x24, #2
    beq TILE12_8

    cmp x24, #3
    beq TILE12_12

    add x10, x6, x4 // +x4
    add x8, x6, x4, LSL #1 // + 2*x4
    add x14, x10, x4, LSL #1 // + 3*x4
    .inst 0xe400fcd0  // st1b {z16.b}, p7, [x6]
    .inst 0xe400fd51  // st1b {z17.b}, p7, [x10]
    .inst 0xe400fd12  // st1b {z18.b}, p7, [x8]
    .inst 0xe400fdd3  // st1b {z19.b}, p7, [x14]
    b TILE12_Dz_End

    TILE12_4:
    .inst 0xe400fcd0  // st1b {z16.b}, p7, [x6]
    b TILE12_Dz_End

    TILE12_8:
    add x10, x6, x4 // +x4
    .inst 0xe400fcd0  // st1b {z16.b}, p7, [x6]
    .inst 0xe400fd51  // st1b {z17.b}, p7, [x10]
    b TILE12_Dz_End

    TILE12_12:
    add x10, x6, x4 // +x4
    add x8, x6, x4, LSL #1 // + 2*x4
    .inst 0xe400fcd0  // st1b {z16.b}, p7, [x6]
    .inst 0xe400fd51  // st1b {z17.b}, p7, [x10]
    .inst 0xe400fd12  // st1b {z18.b}, p7, [x8]
    b TILE12_Dz_End

    TILE12_Dz_End:
    sub x24, x24, #4
    cmp x24, #0
    ble TILE12_End
    add x6, x6, x4, LSL #2
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
    add x0, x0, x11
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

    .inst 0xc0080055  // zero {za0.s, za2.s}

LoopSz_TILE8:
    .inst 0xa400a960  // ld1b {z0.b}, p2/z, [x11]      // src
    .inst 0xa400ad81  // ld1b {z1.b}, p3/z, [x12]      // weight
    // matmul
    .inst 0xa0816800 // smopa za0.s, p2/m, p3/m, z0.b, z1.b
    subs x10, x10, #1
    add x11, x11, x22
    .inst 0x042c502c  // addvl x12, x12, #1

bne LoopSz_TILE8

    // extract int32_t vectors from za0.s
    mov w8, #0
    .inst 0xc0060c04  // mova {z4.s-z7.s}, za.s[w8, 0, VGx4]  // z4: e=0(za0h.s[0]), z5: e=4(za0h.s[4], z6: e=8(za0h.s[8]), z7: e=12(za0h.s[12), VG=512bit/32bit
    .inst 0xc0060c88  // mova {z8.s-z11.s}, za.s[w8, 4, VGx4]  // z8: e=1, z9: e=5, z10: e=9, z11: e=13
    mov w8, #8
    .inst 0xc0060c0c  // mova {z12.s-z15.s}, za.s[w8, 0, VGx4]  // z12: e=2, z13: e=6, z14: e=10, z15: e=14
    .inst 0xc0060c90  // mova {z16.s-z19.s}, za.s[w8, 4, VGx4]  // z16: e=3, z17: e=7, z18: e=11, z19: e=15

    // weight scale&bias
    .inst 0xa0404180  // ld1w {z0.s-z1.s}, pn8/z, [x12]
    .inst 0xa400a9a2  // ld1b {z2.b}, p2/z, [x13]   // input kernel sum
    .inst 0xa400aae3  // ld1b {z3.b}, p2/z, [x23]  // input scale
    .inst 0x042c504c  // addvl x12, x12, #2
    add x13, x13, x22
    add x23, x23, x25

    .inst 0xc132e084  // scvtf {z4.s-z7.s}, {z4.s-z7.s}
    .inst 0xc132e108  // scvtf {z8.s-z11.s}, {z8.s-z11.s}
    .inst 0xc132e18c  // scvtf {z12.s-z15.s}, {z12.s-z15.s}
    .inst 0xc132e210  // scvtf {z16.s-z19.s}, {z16.s-z19.s}

    // inputKernelSum x weightBias -> [16,16]
    .inst 0x80816841 // fmopa za1.s, p2/m, p3/m, z2.s, z1.s
    // inputScale x weightScale -> [16,16]
    .inst 0x80806862 // fmopa za2.s, p2/m, p3/m, z3.s, z0.s

    mov w10, #1
    mov w8, #2
    add x15, x15, #1

    cbz x27, TILE8_DEQUANT
    .inst 0xa400ab60  // ld1b {z0.b}, p2/z, [x27]   // input dequant bias
    .inst 0xa540af81  // ld1w {z1.s}, p3/z, [x28]   // weight kernel sum
    .inst 0x80816801 // fmopa za1.s, p2/m, p3/m, z0.s, z1.s
    add x27, x27, x25
    .inst 0x043c503c  // addvl x28, x28, #1

    TILE8_DEQUANT:
    // extract vector from za2.s
    .inst 0xc0060c14  // mova {z20.s-z23.s}, za.s[w8, 0, VGx4]  // z20-z23: e=0, e=4, e=8, e=12
    .inst 0xc0060c98  // mova {z24.s-z27.s}, za.s[w8, 4, VGx4]  // z24-z27: e=1, e=5, e=9, e=13
    mov w8, #10
    .inst 0xc0060c00  // mova {z0.s-z3.s}, za.s[w8, 0, VGx4]  // z0-z3: e=2, e=6, e=10, e=14

    // accumulate to za1.s
    .inst 0xc1b55880  // fmla za.s[w10, 0, VGx4], {z4.s-z7.s}, {z20.s-z23.s}  // za, row:1,17,33,49
    .inst 0xc1b95904  // fmla za.s[w10, 4, VGx4], {z8.s-z11.s}, {z24.s-z27.s}  // za, row: 5,21,37,53
    mov w10, #9
    .inst 0xc0060c94  // mova {z20.s-z23.s}, za.s[w8, 4, VGx4]  // z20-z23: e=3, e=7, e=11, e=15
    .inst 0xc1a15980  // fmla za.s[w10, 0, VGx4], {z12.s-z15.s}, {z0.s-z3.s}
    .inst 0xc1b55a04  // fmla za.s[w10, 4, VGx4], {z16.s-z19.s}, {z20.s-z23.s}

    cmp x15, x26
    beq TILE8_POST
    b TILE8_BLOCKNUM

    TILE8_POST:
    .inst 0xa540ae80  // ld1w {z0.s}, p3/z, [x20]  // bias
    .inst 0x25b9ce01  // fmov z1.s, #1
    add x20, x20, #64
    .inst 0x80806c21 // fmopa za1.s, p3/m, p3/m, z1.s, z0.s
    mov w14, #0
    mov w15, #4

    .inst 0xc086c420  // mova {z0.s-z3.s}, za1v.s[w14, 0:3]
    .inst 0xc086e424  // mova {z4.s-z7.s}, za1v.s[w15, 0:3]
    mov w14, #8
    mov w15, #12
    .inst 0xc086c428  // mova {z8.s-z11.s}, za1v.s[w14, 0:3]
    .inst 0xc086e42c  // mova {z12.s-z15.s}, za1v.s[w15, 0:3]

    cbz x28, TILE8_Int8_Output
    mov x11, #128 // fp32 output: 8*4*sizeof(float) 

    TILE8_RELU:
    .inst 0xc1becba0  // fclamp {z0.s-z3.s}, z29.s, z30.s
    .inst 0xc1becba4  // fclamp {z4.s-z7.s}, z29.s, z30.s
    .inst 0xc1becba8  // fclamp {z8.s-z11.s}, z29.s, z30.s
    .inst 0xc1becbac  // fclamp {z12.s-z15.s}, z29.s, z30.s

    TILE8_STORE:
    .inst 0xc1b6e010  // zip {z16.s-z19.s}, {z0.s-z3.s}
    .inst 0xc1b6e094  // zip {z20.s-z23.s}, {z4.s-z7.s}
    .inst 0xc1b6e118  // zip {z24.s-z27.s}, {z8.s-z11.s}
    .inst 0xc1b6e180  // zip {z0.s-z3.s}, {z12.s-z15.s}

    cmp x24, #1
    beq TILE8_STORE4

    cmp x24, #2
    beq TILE8_STORE8

    cmp x24, #3
    beq TILE8_STORE12

    TILE8_STORE16:
    add x10, x6, x4 // +x4
    add x8, x6, x4, LSL #1 // + 2*x4
    add x14, x8, x4 // + 3*x4
    .inst 0xa06000d0  // st1b {z16.b, z17.b}, pn8, [x6]
    .inst 0xa0600154  // st1b {z20.b, z21.b}, pn8, [x10]
    .inst 0xa0600118  // st1b {z24.b, z25.b}, pn8, [x8]
    .inst 0xa06001c0  // st1b {z0.b, z1.b}, pn8, [x14]
    b TILE8_Dz_End

    TILE8_STORE4:
    .inst 0xa06000d0  // st1b {z16.b, z17.b}, pn8, [x6]
    b TILE8_Dz_End

    TILE8_STORE8:
    add x10, x6, x4 // +x4
    .inst 0xa06000d0  // st1b {z16.b, z17.b}, pn8, [x6]
    .inst 0xa0600154  // st1b {z20.b, z21.b}, pn8, [x10]
    b TILE8_Dz_End

    TILE8_STORE12:
    add x10, x6, x4 // +x4
    add x8, x6, x4, LSL #1 // + 2*x4
    .inst 0xa06000d0  // st1b {z16.b, z17.b}, pn8, [x6]
    .inst 0xa0600154  // st1b {z20.b, z21.b}, pn8, [x10]
    .inst 0xa0600118  // st1b {z24.b, z25.b}, pn8, [x8]
    b TILE8_Dz_End

    TILE8_Int8_Output:
    mov x11, #32 // int8 output: 8*4*sizeof(int8_t) 

    ROUND_TO_NEAREST_INT

    .inst 0xc131e000  // fcvtzs {z0.s-z3.s}, {z0.s-z3.s}
    .inst 0xc131e084  // fcvtzs {z4.s-z7.s}, {z4.s-z7.s}
    .inst 0xc131e108  // fcvtzs {z8.s-z11.s}, {z8.s-z11.s}
    .inst 0xc131e18c  // fcvtzs {z12.s-z15.s}, {z12.s-z15.s}

    .inst 0xc133e050  // sqcvtn z16.b, {z0.s-z3.s}
    .inst 0xc133e0d1  // sqcvtn z17.b, {z4.s-z7.s}
    .inst 0xc133e152  // sqcvtn z18.b, {z8.s-z11.s}
    .inst 0xc133e1d3  // sqcvtn z19.b, {z12.s-z15.s}
    .inst 0xc13ecfb0  // sclamp {z16.b-z19.b}, z29.b, z30.b
    
    cmp x24, #1
    beq TILE8_4

    cmp x24, #2
    beq TILE8_8

    cmp x24, #3
    beq TILE8_12

    add x10, x6, x4 // +x4
    add x8, x6, x4, LSL #1 // + 2*x4
    add x14, x10, x4, LSL #1 // + 3*x4
    .inst 0xe400e8d0  // st1b {z16.b}, p2, [x6]
    .inst 0xe400e951  // st1b {z17.b}, p2, [x10]
    .inst 0xe400e912  // st1b {z18.b}, p2, [x8]
    .inst 0xe400e9d3  // st1b {z19.b}, p2, [x14]
    b TILE8_Dz_End

    TILE8_4:
    .inst 0xe400e8d0  // st1b {z16.b}, p2, [x6]
    b TILE8_Dz_End

    TILE8_8:
    add x10, x6, x4 // +x4
    .inst 0xe400e8d0  // st1b {z16.b}, p2, [x6]
    .inst 0xe400e951  // st1b {z17.b}, p2, [x10]
    b TILE8_Dz_End

    TILE8_12:
    add x10, x6, x4 // +x4
    add x8, x6, x4, LSL #1 // + 2*x4
    .inst 0xe400e8d0  // st1b {z16.b}, p2, [x6]
    .inst 0xe400e951  // st1b {z17.b}, p2, [x10]
    .inst 0xe400e912  // st1b {z18.b}, p2, [x8]
    b TILE8_Dz_End

    TILE8_Dz_End:
    sub x24, x24, #4
    cmp x24, #0
    ble TILE8_End
    add x6, x6, x4, LSL #2
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
    add x0, x0, x11
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

    .inst 0xc0080055  // zero {za0.s, za2.s}

LoopSz_TILE4:
    .inst 0xa400b560  // ld1b {z0.b}, p5/z, [x11]      // src
    .inst 0xa400ad81  // ld1b {z1.b}, p3/z, [x12]      // weight
    // matmul
    .inst 0xa0817400 // smopa za0.s, p5/m, p3/m, z0.b, z1.b
    subs x10, x10, #1
    add x11, x11, x22
    .inst 0x042c502c  // addvl x12, x12, #1

bne LoopSz_TILE4

    // extract int32_t vectors from za0.s
    mov w8, #0
    .inst 0xc0060c04  // mova {z4.s-z7.s}, za.s[w8, 0, VGx4]  // z4: e=0(za0h.s[0]), z5: e=4(za0h.s[4], z6: e=8(za0h.s[8]), z7: e=12(za0h.s[12), VG=512bit/32bit
    .inst 0xc0060c88  // mova {z8.s-z11.s}, za.s[w8, 4, VGx4]  // z8: e=1, z9: e=5, z10: e=9, z11: e=13
    mov w8, #8
    .inst 0xc0060c0c  // mova {z12.s-z15.s}, za.s[w8, 0, VGx4]  // z12: e=2, z13: e=6, z14: e=10, z15: e=14
    .inst 0xc0060c90  // mova {z16.s-z19.s}, za.s[w8, 4, VGx4]  // z16: e=3, z17: e=7, z18: e=11, z19: e=15

    // weight scale&bias
    .inst 0xa0404180  // ld1w {z0.s-z1.s}, pn8/z, [x12]
    .inst 0xa400b5a2  // ld1b {z2.b}, p5/z, [x13]   // input kernel sum
    .inst 0xa400b6e3  // ld1b {z3.b}, p5/z, [x23]  // input scale
    .inst 0x042c504c  // addvl x12, x12, #2
    add x13, x13, x22
    add x23, x23, x25

    .inst 0xc132e084  // scvtf {z4.s-z7.s}, {z4.s-z7.s}
    .inst 0xc132e108  // scvtf {z8.s-z11.s}, {z8.s-z11.s}
    .inst 0xc132e18c  // scvtf {z12.s-z15.s}, {z12.s-z15.s}
    .inst 0xc132e210  // scvtf {z16.s-z19.s}, {z16.s-z19.s}

    // inputKernelSum x weightBias -> [16,16]
    .inst 0x80817441 // fmopa za1.s, p5/m, p3/m, z2.s, z1.s
    // inputScale x weightScale -> [16,16]
    .inst 0x80807462 // fmopa za2.s, p5/m, p3/m, z3.s, z0.s

    mov w10, #1
    mov w8, #2
    add x15, x15, #1

    cbz x27, TILE4_DEQUANT
    .inst 0xa400b760  // ld1b {z0.b}, p5/z, [x27]   // input dequant bias
    .inst 0xa540af81  // ld1w {z1.s}, p3/z, [x28]   // weight kernel sum
    .inst 0x80817401 // fmopa za1.s, p5/m, p3/m, z0.s, z1.s
    add x27, x27, x25
    .inst 0x043c503c  // addvl x28, x28, #1

    TILE4_DEQUANT:
    // extract vector from za2.s
    .inst 0xc0060c14  // mova {z20.s-z23.s}, za.s[w8, 0, VGx4]  // z20-z23: e=0, e=4, e=8, e=12
    .inst 0xc0060c98  // mova {z24.s-z27.s}, za.s[w8, 4, VGx4]  // z24-z27: e=1, e=5, e=9, e=13
    mov w8, #10
    .inst 0xc0060c00  // mova {z0.s-z3.s}, za.s[w8, 0, VGx4]  // z0-z3: e=2, e=6, e=10, e=14

    // accumulate to za1.s
    .inst 0xc1b55880  // fmla za.s[w10, 0, VGx4], {z4.s-z7.s}, {z20.s-z23.s}  // za, row:1,17,33,49
    .inst 0xc1b95904  // fmla za.s[w10, 4, VGx4], {z8.s-z11.s}, {z24.s-z27.s}  // za, row: 5,21,37,53
    mov w10, #9
    .inst 0xc0060c94  // mova {z20.s-z23.s}, za.s[w8, 4, VGx4]  // z20-z23: e=3, e=7, e=11, e=15
    .inst 0xc1a15980  // fmla za.s[w10, 0, VGx4], {z12.s-z15.s}, {z0.s-z3.s}
    .inst 0xc1b55a04  // fmla za.s[w10, 4, VGx4], {z16.s-z19.s}, {z20.s-z23.s}

    cmp x15, x26
    beq TILE4_POST
    b TILE4_BLOCKNUM

    TILE4_POST:
    .inst 0xa540ae80  // ld1w {z0.s}, p3/z, [x20]  // bias
    .inst 0x25b9ce01  // fmov z1.s, #1
    add x20, x20, #64
    .inst 0x80806c21 // fmopa za1.s, p3/m, p3/m, z1.s, z0.s
    mov w14, #0
    mov w15, #4

    .inst 0xc086c420  // mova {z0.s-z3.s}, za1v.s[w14, 0:3]
    .inst 0xc086e424  // mova {z4.s-z7.s}, za1v.s[w15, 0:3]
    mov w14, #8
    mov w15, #12
    .inst 0xc086c428  // mova {z8.s-z11.s}, za1v.s[w14, 0:3]
    .inst 0xc086e42c  // mova {z12.s-z15.s}, za1v.s[w15, 0:3]

    cbz x28, TILE4_Int8_Output
    mov x11, #64 // fp32 output: 4*4*sizeof(float)

    TILE4_RELU:
    .inst 0xc1becba0  // fclamp {z0.s-z3.s}, z29.s, z30.s
    .inst 0xc1becba4  // fclamp {z4.s-z7.s}, z29.s, z30.s
    .inst 0xc1becba8  // fclamp {z8.s-z11.s}, z29.s, z30.s
    .inst 0xc1becbac  // fclamp {z12.s-z15.s}, z29.s, z30.s

    TILE4_STORE:
    .inst 0xc1b6e010  // zip {z16.s-z19.s}, {z0.s-z3.s}
    .inst 0xc1b6e094  // zip {z20.s-z23.s}, {z4.s-z7.s}
    .inst 0xc1b6e118  // zip {z24.s-z27.s}, {z8.s-z11.s}
    .inst 0xc1b6e180  // zip {z0.s-z3.s}, {z12.s-z15.s}

    cmp x24, #1
    beq TILE4_STORE4

    cmp x24, #2
    beq TILE4_STORE8

    cmp x24, #3
    beq TILE4_STORE12

    TILE4_STORE16:
    add x10, x6, x4 // +x4
    add x8, x6, x4, LSL #1 // + 2*x4
    add x14, x8, x4 // + 3*x4
    .inst 0xe400ecd0  // st1b {z16.b}, p3, [x6]
    .inst 0xe400ed54  // st1b {z20.b}, p3, [x10]
    .inst 0xe400ed18  // st1b {z24.b}, p3, [x8]
    .inst 0xe400edc0  // st1b {z0.b}, p3, [x14]
    b TILE4_Dz_End

    TILE4_STORE4:
    .inst 0xe400ecd0  // st1b {z16.b}, p3, [x6]
    b TILE4_Dz_End

    TILE4_STORE8:
    add x10, x6, x4 // +x4
    .inst 0xe400ecd0  // st1b {z16.b}, p3, [x6]
    .inst 0xe400ed54  // st1b {z20.b}, p3, [x10]
    b TILE4_Dz_End

    TILE4_STORE12:
    add x10, x6, x4 // +x4
    add x8, x6, x4, LSL #1 // + 2*x4
    .inst 0xe400ecd0  // st1b {z16.b}, p3, [x6]
    .inst 0xe400ed54  // st1b {z20.b}, p3, [x10]
    .inst 0xe400ed18  // st1b {z24.b}, p3, [x8]
    b TILE4_Dz_End


    TILE4_Int8_Output:
    mov x11, #16 // int8_t output: 4*4*sizeof(int8_t)

    ROUND_TO_NEAREST_INT

    .inst 0xc131e000  // fcvtzs {z0.s-z3.s}, {z0.s-z3.s}
    .inst 0xc131e084  // fcvtzs {z4.s-z7.s}, {z4.s-z7.s}
    .inst 0xc131e108  // fcvtzs {z8.s-z11.s}, {z8.s-z11.s}
    .inst 0xc131e18c  // fcvtzs {z12.s-z15.s}, {z12.s-z15.s}

    .inst 0xc133e050  // sqcvtn z16.b, {z0.s-z3.s}
    .inst 0xc133e0d1  // sqcvtn z17.b, {z4.s-z7.s}
    .inst 0xc133e152  // sqcvtn z18.b, {z8.s-z11.s}
    .inst 0xc133e1d3  // sqcvtn z19.b, {z12.s-z15.s}
    .inst 0xc13ecfb0  // sclamp {z16.b-z19.b}, z29.b, z30.b
    
    cmp x24, #1
    beq TILE4_4

    cmp x24, #2
    beq TILE4_8

    cmp x24, #3
    beq TILE4_12

    add x10, x6, x4 // +x4
    add x8, x6, x4, LSL #1 // + 2*x4
    add x14, x10, x4, LSL #1 // + 3*x4
    .inst 0xe400f4d0  // st1b {z16.b}, p5, [x6]
    .inst 0xe400f551  // st1b {z17.b}, p5, [x10]
    .inst 0xe400f512  // st1b {z18.b}, p5, [x8]
    .inst 0xe400f5d3  // st1b {z19.b}, p5, [x14]
    b TILE4_Dz_End

    TILE4_4:
    .inst 0xe400f4d0  // st1b {z16.b}, p5, [x6]
    b TILE4_Dz_End

    TILE4_8:
    add x10, x6, x4 // +x4
    .inst 0xe400f4d0  // st1b {z16.b}, p5, [x6]
    .inst 0xe400f551  // st1b {z17.b}, p5, [x10]
    b TILE4_Dz_End

    TILE4_12:
    add x10, x6, x4 // +x4
    add x8, x6, x4, LSL #1 // + 2*x4
    .inst 0xe400f4d0  // st1b {z16.b}, p5, [x6]
    .inst 0xe400f551  // st1b {z17.b}, p5, [x10]
    .inst 0xe400f512  // st1b {z18.b}, p5, [x8]
    b TILE4_Dz_End

    TILE4_Dz_End:
    sub x24, x24, #4
    cmp x24, #0
    ble TILE4_End
    add x6, x6, x4, LSL #2
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
    add x0, x0, x11
    add x1, x1, #16
    add x13, x19, #16
    add x23, x21, #16
    // update inputBias/weightKernelSum
    cbz x27, TILE_2
    REVERT_INPUT_DEQUANT_BIAS x27, x15, x26, x25
    REVERT_WEIGHT_KERNEL_SUM x28, x11, x26, x5
    add x27, x27, #16

TILE_2:
cmp x7, #2
blt TILE_1

    .inst 0x2518e107  // ptrue p7.b, #8

    mov x24, x5      // dst_depth_quad
    mov x6, x0       // dst
    mov x12, x2      // weight
    mov x20, x9      // bias
    mov x19, x13      // input kernel sum
    mov x21, x23     // input dequant scale

LoopDz_TILE2:
    mov x11, x1             // src
    mov x15, #0             // blockid
    .inst 0xc00800ff  // zero {za}
TILE2_BLOCKNUM:
    mov x10, x3             // src_depth_quad

    .inst 0xc0080055  // zero {za0.s, za2.s}

LoopSz_TILE2:
    .inst 0xa400bd60  // ld1b {z0.b}, p7/z, [x11]      // src
    .inst 0xa400ad81  // ld1b {z1.b}, p3/z, [x12]      // weight
    // matmul
    .inst 0xa0817c00 // smopa za0.s, p7/m, p3/m, z0.b, z1.b
    subs x10, x10, #1
    add x11, x11, x22
    .inst 0x042c502c  // addvl x12, x12, #1

bne LoopSz_TILE2

    // extract int32_t vectors from za0.s
    mov w8, #0
    .inst 0xc0060c04  // mova {z4.s-z7.s}, za.s[w8, 0, VGx4]  // z4: e=0(za0h.s[0]), z5: e=4(za0h.s[4], z6: e=8(za0h.s[8]), z7: e=12(za0h.s[12), VG=512bit/32bit
    .inst 0xc0060c88  // mova {z8.s-z11.s}, za.s[w8, 4, VGx4]  // z8: e=1, z9: e=5, z10: e=9, z11: e=13
    mov w8, #8
    .inst 0xc0060c0c  // mova {z12.s-z15.s}, za.s[w8, 0, VGx4]  // z12: e=2, z13: e=6, z14: e=10, z15: e=14
    .inst 0xc0060c90  // mova {z16.s-z19.s}, za.s[w8, 4, VGx4]  // z16: e=3, z17: e=7, z18: e=11, z19: e=15

    // weight scale&bias
    .inst 0xa0404180  // ld1w {z0.s-z1.s}, pn8/z, [x12]
    .inst 0xa400bda2  // ld1b {z2.b}, p7/z, [x13]   // input kernel sum
    .inst 0xa400bee3  // ld1b {z3.b}, p7/z, [x23]  // input scale
    .inst 0x042c504c  // addvl x12, x12, #2
    add x13, x13, x22
    add x23, x23, x25

    .inst 0xc132e084  // scvtf {z4.s-z7.s}, {z4.s-z7.s}
    .inst 0xc132e108  // scvtf {z8.s-z11.s}, {z8.s-z11.s}
    .inst 0xc132e18c  // scvtf {z12.s-z15.s}, {z12.s-z15.s}
    .inst 0xc132e210  // scvtf {z16.s-z19.s}, {z16.s-z19.s}

    // inputKernelSum x weightBias -> [16,16]
    .inst 0x80817c41 // fmopa za1.s, p7/m, p3/m, z2.s, z1.s
    // inputScale x weightScale -> [16,16]
    .inst 0x80807c62 // fmopa za2.s, p7/m, p3/m, z3.s, z0.s

    mov w10, #1
    mov w8, #2
    add x15, x15, #1

    cbz x27, TILE2_DEQUANT
    .inst 0xa400bf60  // ld1b {z0.b}, p7/z, [x27]   // input dequant bias
    .inst 0xa540af81  // ld1w {z1.s}, p3/z, [x28]   // weight kernel sum
    .inst 0x80817c01 // fmopa za1.s, p7/m, p3/m, z0.s, z1.s
    add x27, x27, x25
    .inst 0x043c503c  // addvl x28, x28, #1

    TILE2_DEQUANT:
    // extract vector from za2.s
    .inst 0xc0060c14  // mova {z20.s-z23.s}, za.s[w8, 0, VGx4]  // z20-z23: e=0, e=4, e=8, e=12
    .inst 0xc0060c98  // mova {z24.s-z27.s}, za.s[w8, 4, VGx4]  // z24-z27: e=1, e=5, e=9, e=13
    mov w8, #10
    .inst 0xc0060c00  // mova {z0.s-z3.s}, za.s[w8, 0, VGx4]  // z0-z3: e=2, e=6, e=10, e=14

    // accumulate to za1.s
    .inst 0xc1b55880  // fmla za.s[w10, 0, VGx4], {z4.s-z7.s}, {z20.s-z23.s}  // za, row:1,17,33,49
    .inst 0xc1b95904  // fmla za.s[w10, 4, VGx4], {z8.s-z11.s}, {z24.s-z27.s}  // za, row: 5,21,37,53
    mov w10, #9
    .inst 0xc0060c94  // mova {z20.s-z23.s}, za.s[w8, 4, VGx4]  // z20-z23: e=3, e=7, e=11, e=15
    .inst 0xc1a15980  // fmla za.s[w10, 0, VGx4], {z12.s-z15.s}, {z0.s-z3.s}
    .inst 0xc1b55a04  // fmla za.s[w10, 4, VGx4], {z16.s-z19.s}, {z20.s-z23.s}

    cmp x15, x26
    beq TILE2_POST
    b TILE2_BLOCKNUM

    TILE2_POST:
    .inst 0xa540ae80  // ld1w {z0.s}, p3/z, [x20]  // bias
    .inst 0x25b9ce01  // fmov z1.s, #1
    add x20, x20, #64
    .inst 0x80806c21 // fmopa za1.s, p3/m, p3/m, z1.s, z0.s
    mov w14, #0
    mov w15, #4

    .inst 0xc086c420  // mova {z0.s-z3.s}, za1v.s[w14, 0:3]
    .inst 0xc086e424  // mova {z4.s-z7.s}, za1v.s[w15, 0:3]
    mov w14, #8
    mov w15, #12
    .inst 0xc086c428  // mova {z8.s-z11.s}, za1v.s[w14, 0:3]
    .inst 0xc086e42c  // mova {z12.s-z15.s}, za1v.s[w15, 0:3]

    cbz x28, TILE2_Int8_Output
    mov x11, #32 // fp32 output: 2*4*sizeof(float)

    TILE2_RELU:
    .inst 0xc1becba0  // fclamp {z0.s-z3.s}, z29.s, z30.s
    .inst 0xc1becba4  // fclamp {z4.s-z7.s}, z29.s, z30.s
    .inst 0xc1becba8  // fclamp {z8.s-z11.s}, z29.s, z30.s
    .inst 0xc1becbac  // fclamp {z12.s-z15.s}, z29.s, z30.s

    TILE2_STORE:
    .inst 0xc1b6e010  // zip {z16.s-z19.s}, {z0.s-z3.s}
    .inst 0xc1b6e094  // zip {z20.s-z23.s}, {z4.s-z7.s}
    .inst 0xc1b6e118  // zip {z24.s-z27.s}, {z8.s-z11.s}
    .inst 0xc1b6e180  // zip {z0.s-z3.s}, {z12.s-z15.s}

    cmp x24, #1
    beq TILE2_STORE4

    cmp x24, #2
    beq TILE2_STORE8

    cmp x24, #3
    beq TILE2_STORE12

    TILE2_STORE16:
    add x10, x6, x4 // +x4
    add x8, x6, x4, LSL #1 // + 2*x4
    add x14, x8, x4 // + 3*x4
    .inst 0xe400e8d0  // st1b {z16.b}, p2, [x6]
    .inst 0xe400e954  // st1b {z20.b}, p2, [x10]
    .inst 0xe400e918  // st1b {z24.b}, p2, [x8]
    .inst 0xe400e9c0  // st1b {z0.b}, p2, [x14]
    b TILE2_Dz_End

    TILE2_STORE4:
    .inst 0xe400e8d0  // st1b {z16.b}, p2, [x6]
    b TILE2_Dz_End

    TILE2_STORE8:
    add x10, x6, x4 // +x4
    .inst 0xe400e8d0  // st1b {z16.b}, p2, [x6]
    .inst 0xe400e954  // st1b {z20.b}, p2, [x10]
    b TILE2_Dz_End

    TILE2_STORE12:
    add x10, x6, x4 // +x4
    add x8, x6, x4, LSL #1 // + 2*x4
    .inst 0xe400e8d0  // st1b {z16.b}, p2, [x6]
    .inst 0xe400e954  // st1b {z20.b}, p2, [x10]
    .inst 0xe400e918  // st1b {z24.b}, p2, [x8]
    b TILE2_Dz_End


    TILE2_Int8_Output:
    mov x11, #8 // int8_t output: 2*4*sizeof(int8_t)

    ROUND_TO_NEAREST_INT

    .inst 0xc131e000  // fcvtzs {z0.s-z3.s}, {z0.s-z3.s}
    .inst 0xc131e084  // fcvtzs {z4.s-z7.s}, {z4.s-z7.s}
    .inst 0xc131e108  // fcvtzs {z8.s-z11.s}, {z8.s-z11.s}
    .inst 0xc131e18c  // fcvtzs {z12.s-z15.s}, {z12.s-z15.s}

    .inst 0xc133e050  // sqcvtn z16.b, {z0.s-z3.s}
    .inst 0xc133e0d1  // sqcvtn z17.b, {z4.s-z7.s}
    .inst 0xc133e152  // sqcvtn z18.b, {z8.s-z11.s}
    .inst 0xc133e1d3  // sqcvtn z19.b, {z12.s-z15.s}
    .inst 0xc13ecfb0  // sclamp {z16.b-z19.b}, z29.b, z30.b
    
    cmp x24, #1
    beq TILE2_4

    cmp x24, #2
    beq TILE2_8

    cmp x24, #3
    beq TILE2_12

    add x10, x6, x4 // +x4
    add x8, x6, x4, LSL #1 // + 2*x4
    add x14, x10, x4, LSL #1 // + 3*x4
    .inst 0xe400fcd0  // st1b {z16.b}, p7, [x6]
    .inst 0xe400fd51  // st1b {z17.b}, p7, [x10]
    .inst 0xe400fd12  // st1b {z18.b}, p7, [x8]
    .inst 0xe400fdd3  // st1b {z19.b}, p7, [x14]
    b TILE2_Dz_End

    TILE2_4:
    .inst 0xe400fcd0  // st1b {z16.b}, p7, [x6]
    b TILE2_Dz_End

    TILE2_8:
    add x10, x6, x4 // +x4
    .inst 0xe400fcd0  // st1b {z16.b}, p7, [x6]
    .inst 0xe400fd51  // st1b {z17.b}, p7, [x10]
    b TILE2_Dz_End

    TILE2_12:
    add x10, x6, x4 // +x4
    add x8, x6, x4, LSL #1 // + 2*x4
    .inst 0xe400fcd0  // st1b {z16.b}, p7, [x6]
    .inst 0xe400fd51  // st1b {z17.b}, p7, [x10]
    .inst 0xe400fd12  // st1b {z18.b}, p7, [x8]
    b TILE2_Dz_End

    TILE2_Dz_End:
    sub x24, x24, #4
    cmp x24, #0
    ble TILE2_End
    add x6, x6, x4, LSL #2
    // revert input scale/kernelSum
    mov x13, x19
    mov x23, x21
    cbz x27, LoopDz_TILE2
    REVERT_INPUT_DEQUANT_BIAS x27, x15, x26, x25
    b LoopDz_TILE2

    TILE2_End:
    // update realE/dst/src/inputScale/inputKernelSum
    sub x7, x7, #2
    cbz x7, End
    add x0, x0, x11
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
    .inst 0xa540ae9c  // ld1w {z28.s}, p3/z, [x20]  // bias
    add x20, x20, #64
TILE1_BLOCKNUM:
    mov x10, x3             // src_depth_quad

    .inst 0xc00800ff  // zero {za}

    cmp x10, #4
    blt LoopSz_TILE_1_lu1
    cmp x22, #4
    bne LoopSz_TILE_1_lu1

    LoopSz_TILE_1_lu4:
        .inst 0xa400b562  // ld1b {z2.b}, p5/z, [x11]      // src
        .inst 0xa0408184  // ld1b {z4.b-z7.b}, pn8/z, [x12]      // weight
        // matmul
        .inst 0xa0846040 // smopa za0.s, p0/m, p3/m, z2.b, z4.b
        .inst 0xa0857840 // smopa za0.s, p6/m, p3/m, z2.b, z5.b
        .inst 0xa0866440 // smopa za0.s, p1/m, p3/m, z2.b, z6.b
        .inst 0xa0877c40 // smopa za0.s, p7/m, p3/m, z2.b, z7.b
        sub x10, x10, #4
        add x11, x11, #16
        .inst 0x042c508c  // addvl x12, x12, #4

        cmp x10, #4
        bge LoopSz_TILE_1_lu4
    cbz x10, LoopSzEnd_TILE_1

    LoopSz_TILE_1_lu1:
        .inst 0xa400a162  // ld1b {z2.b}, p0/z, [x11]      // src
        .inst 0xa400ad80  // ld1b {z0.b}, p3/z, [x12]      // weight
        // matmul
        .inst 0xa0806040 // smopa za0.s, p0/m, p3/m, z2.b, z0.b
        subs x10, x10, #1
        add x11, x11, x22
        .inst 0x042c502c  // addvl x12, x12, #1

    bne LoopSz_TILE_1_lu1




LoopSzEnd_TILE_1:
    mov w14, #0
    sub x15, x15, #1
    .inst 0xc0864408  // mova {z8.s-z11.s}, za0h.s[w14, 0:3]
    .inst 0x04a90108  // add z8.s, z8.s, z9.s
    .inst 0x04ab014a  // add z10.s, z10.s, z11.s
    .inst 0x04aa010c  // add z12.s, z8.s, z10.s
    .inst 0x6594ad8c  // scvtf z12.s, p3/m, z12.s

    .inst 0xa0404180  // ld1w {z0.s-z1.s}, pn8/z, [x12]   // weight scale: 16*sizeof(float)
    .inst 0x8540cda3  // ld1rw {z3.s}, p3/z, [x13]   // x kernel sum
    .inst 0x8540cefb  // ld1rw {z27.s}, p3/z, [x23]

    .inst 0x659b0800  // fmul z0.s, z0.s, z27.s
    .inst 0x64a3003c  // fmla z28.s, z1.s, z3.s[0]
    .inst 0x65a00d9c  // fmla z28.s, p3/m, z12.s, z0.s
    .inst 0x042c504c  // addvl x12, x12, #2
    add x13, x13, x22

    cbz x27, TILE1_ADD_DSTV
    .inst 0x8540cf65  // ld1rw {z5.s}, p3/z, [x27]  // input dequant bias
    .inst 0xa540af86  // ld1w {z6.s}, p3/z, [x28]   // weight kernel sum
    .inst 0x64a500dc  // fmla z28.s, z6.s, z5.s[0]
    add x27, x27, x25
    add x23, x23, x25
    .inst 0x043c503c  // addvl x28, x28, #1

    TILE1_ADD_DSTV:
    cmp x15, #0
    bne TILE1_BLOCKNUM

    cbz x28, TILE1_Int8_Output

    TILE1_POST:
    .inst 0x64be27bc  // fclamp z28.s, z29.s, z30.s

    TILE1_STORE:
    .inst 0x05702383  // dup z3.q, z28.q[1]
    .inst 0x05b02384  // dup z4.q, z28.q[2]
    .inst 0x05f02385  // dup z5.q, z28.q[3]
    cmp x24, #4
    bge Tile1_Store_16
    cmp x24, #3
    beq Tile1_Store_12
    cmp x24, #2
    beq Tile1_Store_8

    // x24==1
    .inst 0xe400f4dc  // st1b {z28.b}, p5, [x6]
    b TILE1_Dz_End

    Tile1_Store_16:
    add x10, x6, x4 // +x4
    add x8, x6, x4, LSL #1 // + 2*x4
    add x14, x10, x4, LSL #1 // + 3*x4
    .inst 0xe400f4dc  // st1b {z28.b}, p5, [x6]
    .inst 0xe400f543  // st1b {z3.b}, p5, [x10]
    .inst 0xe400f504  // st1b {z4.b}, p5, [x8]
    .inst 0xe400f5c5  // st1b {z5.b}, p5, [x14]
    b TILE1_Dz_End

    Tile1_Store_12:
    add x10, x6, x4 // +x4
    add x8, x6, x4, LSL #1 // + 2*x4
    .inst 0xe400f4dc  // st1b {z28.b}, p5, [x6]
    .inst 0xe400f543  // st1b {z3.b}, p5, [x10]
    .inst 0xe400f504  // st1b {z4.b}, p5, [x8]
    b TILE1_Dz_End

    Tile1_Store_8:
    add x10, x6, x4 // +x4
    .inst 0xe400f4dc  // st1b {z28.b}, p5, [x6]
    .inst 0xe400f543  // st1b {z3.b}, p5, [x10]
    b TILE1_Dz_End

    TILE1_Int8_Output:
    .inst 0x65912f84  // fcmlt p4.s, p3/z, z28.s, #0.0
    .inst 0x6599901c  // fsub z28.s, p4/m, z28.s, #0.5
    .inst 0x65902f84  // fcmge p4.s, p3/z, z28.s, #0.0
    .inst 0x6598901c  // fadd z28.s, p4/m, z28.s, #0.5
    .inst 0x659caf9c  // fcvtzs z28.s, p3/m, z28.s
    .inst 0x4530439b  // sqxtnb z27.h, z28.s
    .inst 0x057b6b60  // uzp1 z0.h, z27.h, z27.h
    .inst 0x4528401a  // sqxtnb z26.b, z0.h
    .inst 0x053a6b50  // uzp1 z16.b, z26.b, z26.b
    .inst 0x441ec3b0  // sclamp z16.b, z29.b, z30.b
    .inst 0x052c2211  // dup z17.s, z16.s[1]
    .inst 0x05342212  // dup z18.s, z16.s[2]
    .inst 0x053c2213  // dup z19.s, z16.s[3]
    
    cmp x24, #1
    beq TILE1_4

    cmp x24, #2
    beq TILE1_8

    cmp x24, #3
    beq TILE1_12

    add x10, x6, x4 // +x4
    add x8, x6, x4, LSL #1 // + 2*x4
    add x14, x10, x4, LSL #1 // + 3*x4
    .inst 0xe400e0d0  // st1b {z16.b}, p0, [x6]
    .inst 0xe400e151  // st1b {z17.b}, p0, [x10]
    .inst 0xe400e112  // st1b {z18.b}, p0, [x8]
    .inst 0xe400e1d3  // st1b {z19.b}, p0, [x14]
    b TILE1_Dz_End

    TILE1_4:
    .inst 0xe400e0d0  // st1b {z16.b}, p0, [x6]
    b TILE1_Dz_End

    TILE1_8:
    add x10, x6, x4 // +x4
    .inst 0xe400e0d0  // st1b {z16.b}, p0, [x6]
    .inst 0xe400e151  // st1b {z17.b}, p0, [x10]
    b TILE1_Dz_End

    TILE1_12:
    add x10, x6, x4 // +x4
    add x8, x6, x4, LSL #1 // + 2*x4
    .inst 0xe400e0d0  // st1b {z16.b}, p0, [x6]
    .inst 0xe400e151  // st1b {z17.b}, p0, [x10]
    .inst 0xe400e112  // st1b {z18.b}, p0, [x8]
    b TILE1_Dz_End

    TILE1_Dz_End:
    sub x24, x24, #4
    cmp x24, #0
    ble End
    add x6, x6, x4, LSL #2
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
