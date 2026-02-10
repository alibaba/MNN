//
//  MNNPackedMatMulRemainFP16_SME2.S
//  MNN
//
//  Created by MNN on 2020/06/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5


asm_function MNNPackedMatMulRemainFP16_SME2
//void MNNPackedMatMulRemainFP16_SME2(FLOAT16* C, const FLOAT16* A, const FLOAT16* B, size_t eSize, const size_t* parameter, const FLOAT16* postParameters, const FLOAT16* bias);
//Auto x0: C, x1:A, x2:B, x3:eSize, x4:parameter, x5:postParameters, x6:bias
// parameter: {aStride, l, h, cStride, bExtraStride}
stp d14, d15, [sp, #(-16 * 6)]!
stp d12, d13, [sp, #(16 * 1)]
stp d10, d11, [sp, #(16 * 2)]
stp d8,  d9,  [sp, #(16 * 3)]
stp x21, x22, [sp, #(16 * 4)]
stp x19, x20, [sp, #(16 * 5)]
.inst 0xd503477f  // smstart

// EP=16 LP=2 HP=64
ldr x22, [x4, #0] // aStride
ldr x9, [x4, #8] // l
ldr x10, [x4, #16] // h

ldr x7, [x4, #24] // cStride
ldr x19, [x4, #40] // bExtraStride

lsl x21, x3, #1   // eSize * lP

mov w12, #0
mov w13, #4
mov w14, #8
mov w15, #12

// x10: ocDiv8, x9: lDiv2
add x10, x10, #7
add x9, x9, #1
lsr x10, x10, #3
lsr x9, x9, #1
lsl x20, x3, #3 // x20: eSize * pack

// predicates
.inst 0x2598e3e0  // ptrue p0.s
.inst 0x2558e3e1  // ptrue p1.h
.inst 0x257517e3  // whilelt p3.h, xzr, x21  // eSize * lP valid
.inst 0x25607810  // ptrue pn8.h
.inst 0x2518e124  // ptrue p4.b, vl16        // first 16 bytes valid
.inst 0x257467f2  // whilelt pn10.h, xzr, x20, vlx4 // eSize * pack valid

// Relu parameters
cbz x5, ESIZE
.inst 0x8542c0be  // ld1rw {z30.s}, p0/z, [x5, #8]  // min
.inst 0x8543c0bf  // ld1rw {z31.s}, p0/z, [x5, #12]  // max
.inst 0x6588a3dc  // fcvt z28.h, p0/m, z30.s
.inst 0x6588a3fd  // fcvt z29.h, p0/m, z31.s
.inst 0x0522239e  // dup z30.h, z28.h[0]
.inst 0x052223bf  // dup z31.h, z29.h[0]

ESIZE: // x3 <= eP
cmp x3, #1
beq E1

lsr x7, x7, #1    // cStride / sizeof(float16_t)
cmp x3, #16
blt LoopOcDiv8

mov x22, #64 // aStride = 64 if eSize==16

LoopOcDiv8:
mov x8, x1 // A
mov x21, x9 // LU

.inst 0xc00800ff  // zero {za}

cbz x6, E16LoopL
// add bias
lsl x4, x10, #3
.inst 0x256447f1  // whilelt pn9.h, xzr, x4, vlx2
.inst 0xa04024c8  // ld1h {z8.h-z9.h}, pn9/z, [x6]
.inst 0x04265046  // addvl x6, x6, #2
.inst 0x25b9ce05  // fmov z5.s, #1

.inst 0x6589a50c  // fcvt z12.s, p1/m, z8.h
.inst 0x6489a50d  // fcvtlt z13.s, p1/m, z8.h
.inst 0x6589a52e  // fcvt z14.s, p1/m, z9.h
.inst 0x6489a52f  // fcvtlt z15.s, p1/m, z9.h

.inst 0x05ad6194  // zip1 z20.s, z12.s, z13.s
.inst 0x05ad6595  // zip2 z21.s, z12.s, z13.s
.inst 0x05af61d6  // zip1 z22.s, z14.s, z15.s
.inst 0x05af65d7  // zip2 z23.s, z14.s, z15.s

.inst 0x809400a0  // fmopa za0.s, p0/m, p0/m, z5.s, z20.s
.inst 0x809500a1  // fmopa za1.s, p0/m, p0/m, z5.s, z21.s
.inst 0x809600a2  // fmopa za2.s, p0/m, p0/m, z5.s, z22.s
.inst 0x809700a3  // fmopa za3.s, p0/m, p0/m, z5.s, z23.s

E16LoopL:
.inst 0xa4a0ad04  // ld1h {z4.h}, p3/z, [x8]  // A
.inst 0xa040a040  // ld1h {z0.h-z3.h}, pn8/z, [x2]  // B
// [EP,LP] x [HP,LP] -> [EP,HP]
.inst 0x81a02480  // fmopa za0.s, p1/m, p1/m, z4.h, z0.h
.inst 0x81a12481  // fmopa za1.s, p1/m, p1/m, z4.h, z1.h
.inst 0x81a22482  // fmopa za2.s, p1/m, p1/m, z4.h, z2.h
.inst 0x81a32483  // fmopa za3.s, p1/m, p1/m, z4.h, z3.h

subs x21, x21, #1
add x8, x8, x22
.inst 0x04225082  // addvl x2, x2, #4
bne E16LoopL

add x2, x2, x19 // bExtraStride

// 1. Extract oc=0...15 from za0
.inst 0xc0868400  // mova {z0.s-z3.s}, za0v.s[w12, 0:3]
.inst 0xc086a404  // mova {z4.s-z7.s}, za0v.s[w13, 0:3]
.inst 0xc086c408  // mova {z8.s-z11.s}, za0v.s[w14, 0:3]
.inst 0xc086e40c  // mova {z12.s-z15.s}, za0v.s[w15, 0:3]

.inst 0xc120e030  // fcvtn z16.h, {z0.s-z1.s}
.inst 0xc120e071  // fcvtn z17.h, {z2.s-z3.s}
.inst 0xc120e0b2  // fcvtn z18.h, {z4.s-z5.s}
.inst 0xc120e0f3  // fcvtn z19.h, {z6.s-z7.s}

.inst 0xc120e134  // fcvtn z20.h, {z8.s-z9.s}
.inst 0xc120e175  // fcvtn z21.h, {z10.s-z11.s}
.inst 0xc120e1b6  // fcvtn z22.h, {z12.s-z13.s}
.inst 0xc120e1f7  // fcvtn z23.h, {z14.s-z15.s}

.inst 0xc1b6e218  // zip {z24.s-z27.s}, {z16.s-z19.s}
.inst 0xc1b6e280  // zip {z0.s-z3.s}, {z20.s-z23.s}

cbz x5, StoreOc_0_15
.inst 0xc17fcbd8  // fclamp {z24.h-z27.h}, z30.h, z31.h
.inst 0xc17fcbc0  // fclamp {z0.h-z3.h}, z30.h, z31.h


StoreOc_0_15:
cmp x10, #2
bge StoreOc16

StoreOc8:
.inst 0xa060a818  // st1h {z24.h-z27.h}, pn10, [x0]
b End


StoreOc16:
.inst 0xa060a818  // st1h {z24.h-z27.h}, pn10, [x0]
.inst 0xa027a800  // st1h {z0.h-z3.h}, pn10, [x0, x7, lsl #1]
subs x10, x10, #2
add x0, x0, x7, LSL #2
beq End

// 2. Extract oc=16...31 from za1
.inst 0xc0868420  // mova {z0.s-z3.s}, za1v.s[w12, 0:3]
.inst 0xc086a424  // mova {z4.s-z7.s}, za1v.s[w13, 0:3]
.inst 0xc086c428  // mova {z8.s-z11.s}, za1v.s[w14, 0:3]
.inst 0xc086e42c  // mova {z12.s-z15.s}, za1v.s[w15, 0:3]

.inst 0xc120e030  // fcvtn z16.h, {z0.s-z1.s}
.inst 0xc120e071  // fcvtn z17.h, {z2.s-z3.s}
.inst 0xc120e0b2  // fcvtn z18.h, {z4.s-z5.s}
.inst 0xc120e0f3  // fcvtn z19.h, {z6.s-z7.s}

.inst 0xc120e134  // fcvtn z20.h, {z8.s-z9.s}
.inst 0xc120e175  // fcvtn z21.h, {z10.s-z11.s}
.inst 0xc120e1b6  // fcvtn z22.h, {z12.s-z13.s}
.inst 0xc120e1f7  // fcvtn z23.h, {z14.s-z15.s}

.inst 0xc1b6e218  // zip {z24.s-z27.s}, {z16.s-z19.s}
.inst 0xc1b6e280  // zip {z0.s-z3.s}, {z20.s-z23.s}

cbz x5, StoreOc_16_31
.inst 0xc17fcbd8  // fclamp {z24.h-z27.h}, z30.h, z31.h
.inst 0xc17fcbc0  // fclamp {z0.h-z3.h}, z30.h, z31.h


StoreOc_16_31:
cmp x10, #2
bge StoreOc32

StoreOc24:
.inst 0xa060a818  // st1h {z24.h-z27.h}, pn10, [x0]
b End


StoreOc32:
.inst 0xa060a818  // st1h {z24.h-z27.h}, pn10, [x0]
.inst 0xa027a800  // st1h {z0.h-z3.h}, pn10, [x0, x7, lsl #1]
subs x10, x10, #2
add x0, x0, x7, LSL #2
beq End

// 3. Extract oc=32...47 from za2
.inst 0xc0868440  // mova {z0.s-z3.s}, za2v.s[w12, 0:3]
.inst 0xc086a444  // mova {z4.s-z7.s}, za2v.s[w13, 0:3]
.inst 0xc086c448  // mova {z8.s-z11.s}, za2v.s[w14, 0:3]
.inst 0xc086e44c  // mova {z12.s-z15.s}, za2v.s[w15, 0:3]

.inst 0xc120e030  // fcvtn z16.h, {z0.s-z1.s}
.inst 0xc120e071  // fcvtn z17.h, {z2.s-z3.s}
.inst 0xc120e0b2  // fcvtn z18.h, {z4.s-z5.s}
.inst 0xc120e0f3  // fcvtn z19.h, {z6.s-z7.s}

.inst 0xc120e134  // fcvtn z20.h, {z8.s-z9.s}
.inst 0xc120e175  // fcvtn z21.h, {z10.s-z11.s}
.inst 0xc120e1b6  // fcvtn z22.h, {z12.s-z13.s}
.inst 0xc120e1f7  // fcvtn z23.h, {z14.s-z15.s}

.inst 0xc1b6e218  // zip {z24.s-z27.s}, {z16.s-z19.s}
.inst 0xc1b6e280  // zip {z0.s-z3.s}, {z20.s-z23.s}

cbz x5, StoreOc_32_47
.inst 0xc17fcbd8  // fclamp {z24.h-z27.h}, z30.h, z31.h
.inst 0xc17fcbc0  // fclamp {z0.h-z3.h}, z30.h, z31.h


StoreOc_32_47:
cmp x10, #2
bge StoreOc48

StoreOc40:
.inst 0xa060a818  // st1h {z24.h-z27.h}, pn10, [x0]
b End


StoreOc48:
.inst 0xa060a818  // st1h {z24.h-z27.h}, pn10, [x0]
.inst 0xa027a800  // st1h {z0.h-z3.h}, pn10, [x0, x7, lsl #1]
subs x10, x10, #2
add x0, x0, x7, LSL #2
beq End

// 4. Extract oc=48...63 from za3
.inst 0xc0868460  // mova {z0.s-z3.s}, za3v.s[w12, 0:3]
.inst 0xc086a464  // mova {z4.s-z7.s}, za3v.s[w13, 0:3]
.inst 0xc086c468  // mova {z8.s-z11.s}, za3v.s[w14, 0:3]
.inst 0xc086e46c  // mova {z12.s-z15.s}, za3v.s[w15, 0:3]

.inst 0xc120e030  // fcvtn z16.h, {z0.s-z1.s}
.inst 0xc120e071  // fcvtn z17.h, {z2.s-z3.s}
.inst 0xc120e0b2  // fcvtn z18.h, {z4.s-z5.s}
.inst 0xc120e0f3  // fcvtn z19.h, {z6.s-z7.s}

.inst 0xc120e134  // fcvtn z20.h, {z8.s-z9.s}
.inst 0xc120e175  // fcvtn z21.h, {z10.s-z11.s}
.inst 0xc120e1b6  // fcvtn z22.h, {z12.s-z13.s}
.inst 0xc120e1f7  // fcvtn z23.h, {z14.s-z15.s}

.inst 0xc1b6e218  // zip {z24.s-z27.s}, {z16.s-z19.s}
.inst 0xc1b6e280  // zip {z0.s-z3.s}, {z20.s-z23.s}

cbz x5, StoreOc_48_63
.inst 0xc17fcbd8  // fclamp {z24.h-z27.h}, z30.h, z31.h
.inst 0xc17fcbc0  // fclamp {z0.h-z3.h}, z30.h, z31.h


StoreOc_48_63:
cmp x10, #2
bge StoreOc64

StoreOc56:
.inst 0xa060a818  // st1h {z24.h-z27.h}, pn10, [x0]
b End


StoreOc64:
.inst 0xa060a818  // st1h {z24.h-z27.h}, pn10, [x0]
.inst 0xa027a800  // st1h {z0.h-z3.h}, pn10, [x0, x7, lsl #1]
subs x10, x10, #2
add x0, x0, x7, LSL #2
beq End

b LoopOcDiv8 // continue next ocDiv8

E1:
cmp x3, #1
blt End

E1LoopH:
mov x8, x1 // A
mov x21, x9 // LU

.inst 0xc00800ff  // zero {za}
mov w11, #0

cbz x6, E1LoopL
// bias
lsl x4, x10, #3
.inst 0x256447f1  // whilelt pn9.h, xzr, x4, vlx2 // oc to process, maximum is 64
.inst 0xa04024c8  // ld1h {z8.h-z9.h}, pn9/z, [x6]
.inst 0x04265046  // addvl x6, x6, #2

.inst 0x6589a50c  // fcvt z12.s, p1/m, z8.h
.inst 0x6489a50d  // fcvtlt z13.s, p1/m, z8.h
.inst 0x6589a52e  // fcvt z14.s, p1/m, z9.h
.inst 0x6489a52f  // fcvtlt z15.s, p1/m, z9.h

.inst 0x05ad6194  // zip1 z20.s, z12.s, z13.s
.inst 0x05ad6595  // zip2 z21.s, z12.s, z13.s
.inst 0x05af61d6  // zip1 z22.s, z14.s, z15.s
.inst 0x05af65d7  // zip2 z23.s, z14.s, z15.s
.inst 0xc1a17e80  // fadd za.s[w11, 0, VGx4], {z20.s-z23.s}

E1LoopL:
.inst 0x8540c504  // ld1rw {z4.s}, p1/z, [x8]  // A
.inst 0xa040a040  // ld1h {z0.h-z3.h}, pn8/z, [x2]  // B
// [EP,LP] x [HP,LP] -> [EP,HP]
.inst 0xc1347000  // fdot za.s[w11, 0, VGx4], {z0.h-z3.h}, z4.h

subs x21, x21, #1
add x8, x8, x22
.inst 0x04225082  // addvl x2, x2, #4
bne E1LoopL

add x2, x2, x19 // bExtraStride

E1_To_FP16:
.inst 0xc0820000  // mova z0.s, p0/m, za0h.s[w12, 0]
.inst 0xc0822001  // mova z1.s, p0/m, za0h.s[w13, 0]
.inst 0xc0824002  // mova z2.s, p0/m, za0h.s[w14, 0]
.inst 0xc0826003  // mova z3.s, p0/m, za0h.s[w15, 0]

.inst 0xc120e010  // fcvt z16.h, {z0.s-z1.s}
.inst 0xc120e051  // fcvt z17.h, {z2.s-z3.s}
cbz x5, E1Store
.inst 0xc17fc3d0  // fclamp {z16.h-z17.h}, z30.h, z31.h

E1Store:
cmp x10, #8
bge E1StoreH64

cmp x10, #1
beq E1StoreH8

cmp x10, #2
beq E1StoreH16

cmp x10, #3
beq E1StoreH24

cmp x10, #4
beq E1StoreH32

cmp x10, #5
beq E1StoreH40

cmp x10, #6
beq E1StoreH48

cmp x10, #7
beq E1StoreH56

E1StoreH64:
add x21, x0, x7, LSL #1 // x0+2*x7
add x11, x0, x7, LSL #2 // x0+4*x7
add x20, x11, x7, LSL #1 // x20+6*x7
.inst 0x05702212  // dup z18.q, z16.q[1]
.inst 0x05b02213  // dup z19.q, z16.q[2]
.inst 0x05f02214  // dup z20.q, z16.q[3]
.inst 0x05702235  // dup z21.q, z17.q[1]
.inst 0x05b02236  // dup z22.q, z17.q[2]
.inst 0x05f02237  // dup z23.q, z17.q[3]
.inst 0xe400f010  // st1b {z16.b}, p4, [x0]
.inst 0xe4075012  // st1b {z18.b}, p4, [x0, x7]
.inst 0xe400f2b3  // st1b {z19.b}, p4, [x21]
.inst 0xe40752b4  // st1b {z20.b}, p4, [x21, x7]
.inst 0xe400f171  // st1b {z17.b}, p4, [x11]
.inst 0xe4075175  // st1b {z21.b}, p4, [x11, x7]
.inst 0xe400f296  // st1b {z22.b}, p4, [x20]
.inst 0xe4075297  // st1b {z23.b}, p4, [x20, x7]
b E1H16_End

E1StoreH56:
add x21, x0, x7, LSL #1
add x11, x0, x7, LSL #2
add x20, x11, x7, LSL #1
.inst 0x05702212  // dup z18.q, z16.q[1]
.inst 0x05b02213  // dup z19.q, z16.q[2]
.inst 0x05f02214  // dup z20.q, z16.q[3]
.inst 0x05702235  // dup z21.q, z17.q[1]
.inst 0x05b02236  // dup z22.q, z17.q[2]
.inst 0xe400f010  // st1b {z16.b}, p4, [x0]
.inst 0xe4075012  // st1b {z18.b}, p4, [x0, x7]
.inst 0xe400f2b3  // st1b {z19.b}, p4, [x21]
.inst 0xe40752b4  // st1b {z20.b}, p4, [x21, x7]
.inst 0xe400f171  // st1b {z17.b}, p4, [x11]
.inst 0xe4075175  // st1b {z21.b}, p4, [x11, x7]
.inst 0xe400f296  // st1b {z22.b}, p4, [x20]
b End

E1StoreH48:
add x21, x0, x7, LSL #1
add x11, x0, x7, LSL #2
.inst 0x05702212  // dup z18.q, z16.q[1]
.inst 0x05b02213  // dup z19.q, z16.q[2]
.inst 0x05f02214  // dup z20.q, z16.q[3]
.inst 0x05702235  // dup z21.q, z17.q[1]
.inst 0xe400f010  // st1b {z16.b}, p4, [x0]
.inst 0xe4075012  // st1b {z18.b}, p4, [x0, x7]
.inst 0xe400f2b3  // st1b {z19.b}, p4, [x21]
.inst 0xe40752b4  // st1b {z20.b}, p4, [x21, x7]
.inst 0xe400f171  // st1b {z17.b}, p4, [x11]
.inst 0xe4075175  // st1b {z21.b}, p4, [x11, x7]
b End

E1StoreH40:
add x21, x0, x7, LSL #1
add x11, x0, x7, LSL #2
.inst 0x05702212  // dup z18.q, z16.q[1]
.inst 0x05b02213  // dup z19.q, z16.q[2]
.inst 0x05f02214  // dup z20.q, z16.q[3]
.inst 0xe400f010  // st1b {z16.b}, p4, [x0]
.inst 0xe4075012  // st1b {z18.b}, p4, [x0, x7]
.inst 0xe400f2b3  // st1b {z19.b}, p4, [x21]
.inst 0xe40752b4  // st1b {z20.b}, p4, [x21, x7]
.inst 0xe400f171  // st1b {z17.b}, p4, [x11]
b End

E1StoreH32:
add x21, x0, x7, LSL #1
.inst 0x05702212  // dup z18.q, z16.q[1]
.inst 0x05b02213  // dup z19.q, z16.q[2]
.inst 0x05f02214  // dup z20.q, z16.q[3]
.inst 0xe400f010  // st1b {z16.b}, p4, [x0]
.inst 0xe4075012  // st1b {z18.b}, p4, [x0, x7]
.inst 0xe400f2b3  // st1b {z19.b}, p4, [x21]
.inst 0xe40752b4  // st1b {z20.b}, p4, [x21, x7]
b End

E1StoreH24:
add x21, x0, x7, LSL #1
.inst 0x05702212  // dup z18.q, z16.q[1]
.inst 0x05b02213  // dup z19.q, z16.q[2]
.inst 0xe400f010  // st1b {z16.b}, p4, [x0]
.inst 0xe4075012  // st1b {z18.b}, p4, [x0, x7]
.inst 0xe400f2b3  // st1b {z19.b}, p4, [x21]
b End

E1StoreH16:
.inst 0x05702212  // dup z18.q, z16.q[1]
.inst 0xe400f010  // st1b {z16.b}, p4, [x0]
.inst 0xe4075012  // st1b {z18.b}, p4, [x0, x7]
b End

E1StoreH8:
.inst 0xe400f010  // st1b {z16.b}, p4, [x0]
b End

E1H16_End:
subs x10, x10, #8
add x0, x0, x7, LSL #3
bne E1LoopH

End:
.inst 0xd503467f  // smstop

ldp x19, x20, [sp, #80]
ldp x21, x22, [sp, #64]
ldp d8,  d9,  [sp, #48]
ldp d10, d11, [sp, #32]
ldp d12, d13, [sp, #16]
ldp d14, d15, [sp], #96

ret

#endif
