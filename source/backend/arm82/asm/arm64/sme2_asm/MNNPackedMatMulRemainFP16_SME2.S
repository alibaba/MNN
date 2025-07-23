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


ldr x9, [x4, #8] // l
ldr x10, [x4, #16] // h

ldr x7, [x4, #24] // cStride
ldr x19, [x4, #40] // bExtraStride

/*
EP=16
LP=2
HP=16
*/
// x10: ocDiv8, x9: lDiv2
add x10, x10, #7
add x9, x9, #1
lsr x10, x10, #3
lsr x9, x9, #1

/* predicates */
.inst 0x0460e3eb  // cnth x11
.inst 0x252b17e1  // whilelt p1.b, xzr, x11  // previous 32 bytes valid
.inst 0x2518e3e3  // ptrue p3.b
.inst 0x25207810  // ptrue pn8.b
.inst 0x2518e124  // ptrue p4.b, vl16        // first 16 bytes valid
.inst 0x25034e22  // eor p2.b, p3/z, p1.b, p3.b  // last 32 bytes valid
mov x11, #48
.inst 0x252b17e5  // whilelt p5.b, xzr, x11  // previous 48 bytes valid
/* const register */
mov w12, #0
mov w13, #4
mov w14, #8
mov w15, #12

cbz x5, E16
add x5, x5, #8
.inst 0x8540ccbe  // ld1rw {z30.s}, p3/z, [x5]  // min
add x5, x5, #4
.inst 0x8540ccbf  // ld1rw {z31.s}, p3/z, [x5]  // max
.inst 0x6588afdd  // fcvt z29.h, p3/m, z30.s
.inst 0x6488affd  // fcvtnt z29.h, p3/m, z31.s

.inst 0x052223be  // dup z30.h, z29.h[0]
.inst 0x052623bf  // dup z31.h, z29.h[1]

E16:
cmp x3, #16
blt E12

mov x20, x10 // ocDiv8
mov x11, x6 // bias

E16LoopH:
mov x8, x1 // A
mov x21, x9 // LU

.inst 0xc00800ff  // zero {za}

cbz x11, E16LoopL
// add bias
.inst 0xa0400168  // ld1b {z8.b-z9.b}, pn8/z, [x11]
.inst 0x042b504b  // addvl x11, x11, #2
.inst 0x25b9ce05  // fmov z5.s, #1

.inst 0x6589ad0c  // fcvt z12.s, p3/m, z8.h
.inst 0x6489ad0d  // fcvtlt z13.s, p3/m, z8.h
.inst 0x6589ad2e  // fcvt z14.s, p3/m, z9.h
.inst 0x6489ad2f  // fcvtlt z15.s, p3/m, z9.h

.inst 0x05ad6194  // zip1 z20.s, z12.s, z13.s
.inst 0x05ad6595  // zip2 z21.s, z12.s, z13.s
.inst 0x05af61d6  // zip1 z22.s, z14.s, z15.s
.inst 0x05af65d7  // zip2 z23.s, z14.s, z15.s

.inst 0x80946ca0 // fmopa za0.s, p3/m, p3/m, z5.s, z20.s
.inst 0x80956ca1 // fmopa za1.s, p3/m, p3/m, z5.s, z21.s
.inst 0x80966ca2 // fmopa za2.s, p3/m, p3/m, z5.s, z22.s
.inst 0x80976ca3 // fmopa za3.s, p3/m, p3/m, z5.s, z23.s

E16LoopL:
.inst 0xa400ad04  // ld1b {z4.b}, p3/z, [x8]  // A
.inst 0xa0408040  // ld1b {z0.b-z3.b}, pn8/z, [x2]  // B
// [EP,LP] x [HP,LP] -> [EP,HP]
.inst 0x81a06c80 // fmopa za0.s, p3/m, p3/m, z4.h, z0.h
.inst 0x81a16c81 // fmopa za1.s, p3/m, p3/m, z4.h, z1.h
.inst 0x81a26c82 // fmopa za2.s, p3/m, p3/m, z4.h, z2.h
.inst 0x81a36c83 // fmopa za3.s, p3/m, p3/m, z4.h, z3.h

subs x21, x21, #1
.inst 0x04285028  // addvl x8, x8, #1
.inst 0x04225082  // addvl x2, x2, #4
bne E16LoopL

add x2, x2, x19 // bExtraStride

//  E16_Store_FP16 za0, 0
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

cbz x5, E16Store_0
.inst 0xc17fcbd8  // fclamp {z24.h-z27.h}, z30.h, z31.h
.inst 0xc17fcbc0  // fclamp {z0.h-z3.h}, z30.h, z31.h

E16Store_0:
cmp x20, #1
beq E16StoreH8_0

.inst 0xa0608018  // st1b {z24.b-z27.b}, pn8, [x0]
.inst 0xa0278000  // st1b {z0.b-z3.b}, pn8, [x0, x7]
b E16H16_End_0

E16StoreH8_0:
.inst 0xa0608018  // st1b {z24.b-z27.b}, pn8, [x0]

E16H16_End_0:
sub x20, x20, #2
add x0, x0, x7, LSL #1
cmp x20, #0
ble E16H16_End

//  E16_Store_FP16 za1, 1
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

cbz x5, E16Store_1
.inst 0xc17fcbd8  // fclamp {z24.h-z27.h}, z30.h, z31.h
.inst 0xc17fcbc0  // fclamp {z0.h-z3.h}, z30.h, z31.h

E16Store_1:
cmp x20, #1
beq E16StoreH8_1

.inst 0xa0608018  // st1b {z24.b-z27.b}, pn8, [x0]
.inst 0xa0278000  // st1b {z0.b-z3.b}, pn8, [x0, x7]
b E16H16_End_1

E16StoreH8_1:
.inst 0xa0608018  // st1b {z24.b-z27.b}, pn8, [x0]

E16H16_End_1:
sub x20, x20, #2
add x0, x0, x7, LSL #1
cmp x20, #0
ble E16H16_End

//  E16_Store_FP16 za2, 2
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

cbz x5, E16Store_2
.inst 0xc17fcbd8  // fclamp {z24.h-z27.h}, z30.h, z31.h
.inst 0xc17fcbc0  // fclamp {z0.h-z3.h}, z30.h, z31.h

E16Store_2:
cmp x20, #1
beq E16StoreH8_2

.inst 0xa0608018  // st1b {z24.b-z27.b}, pn8, [x0]
.inst 0xa0278000  // st1b {z0.b-z3.b}, pn8, [x0, x7]
b E16H16_End_2

E16StoreH8_2:
.inst 0xa0608018  // st1b {z24.b-z27.b}, pn8, [x0]

E16H16_End_2:
sub x20, x20, #2
add x0, x0, x7, LSL #1
cmp x20, #0
ble E16H16_End

//  E16_Store_FP16 za3, 3
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

cbz x5, E16Store_3
.inst 0xc17fcbd8  // fclamp {z24.h-z27.h}, z30.h, z31.h
.inst 0xc17fcbc0  // fclamp {z0.h-z3.h}, z30.h, z31.h

E16Store_3:
cmp x20, #1
beq E16StoreH8_3

.inst 0xa0608018  // st1b {z24.b-z27.b}, pn8, [x0]
.inst 0xa0278000  // st1b {z0.b-z3.b}, pn8, [x0, x7]
b E16H16_End_3

E16StoreH8_3:
.inst 0xa0608018  // st1b {z24.b-z27.b}, pn8, [x0]

E16H16_End_3:
sub x20, x20, #2
add x0, x0, x7, LSL #1


E16H16_End:
cmp x20, #0
bgt E16LoopH
b End

E12:
cmp x3, #12
blt E8

mov x20, x10 // ocDiv8
mov x11, x6 // bias
mov x22, x0 // dst
mov x23, x2 // B

E12LoopH:
mov x8, x1 // A
mov x21, x9 // LU

.inst 0xc00800ff  // zero {za}

cbz x11, E12LoopL
// add bias
.inst 0xa0400168  // ld1b {z8.b-z9.b}, pn8/z, [x11]
.inst 0x042b504b  // addvl x11, x11, #2
.inst 0x25b9ce05  // fmov z5.s, #1

.inst 0x6589ad0c  // fcvt z12.s, p3/m, z8.h
.inst 0x6489ad0d  // fcvtlt z13.s, p3/m, z8.h
.inst 0x6589ad2e  // fcvt z14.s, p3/m, z9.h
.inst 0x6489ad2f  // fcvtlt z15.s, p3/m, z9.h

.inst 0x05ad6194  // zip1 z20.s, z12.s, z13.s
.inst 0x05ad6595  // zip2 z21.s, z12.s, z13.s
.inst 0x05af61d6  // zip1 z22.s, z14.s, z15.s
.inst 0x05af65d7  // zip2 z23.s, z14.s, z15.s

.inst 0x80946ca0 // fmopa za0.s, p3/m, p3/m, z5.s, z20.s
.inst 0x80956ca1 // fmopa za1.s, p3/m, p3/m, z5.s, z21.s
.inst 0x80966ca2 // fmopa za2.s, p3/m, p3/m, z5.s, z22.s
.inst 0x80976ca3 // fmopa za3.s, p3/m, p3/m, z5.s, z23.s

E12LoopL:
.inst 0xa400ad04  // ld1b {z4.b}, p3/z, [x8]  // A
.inst 0xa04082e0  // ld1b {z0.b-z3.b}, pn8/z, [x23]  // B
// [EP,LP] x [HP,LP] -> [EP,HP]
.inst 0x81a07480 // fmopa za0.s, p5/m, p3/m, z4.h, z0.h
.inst 0x81a17481 // fmopa za1.s, p5/m, p3/m, z4.h, z1.h
.inst 0x81a27482 // fmopa za2.s, p5/m, p3/m, z4.h, z2.h
.inst 0x81a37483 // fmopa za3.s, p5/m, p3/m, z4.h, z3.h

subs x21, x21, #1
.inst 0x04285028  // addvl x8, x8, #1
.inst 0x04375097  // addvl x23, x23, #4
bne E12LoopL

add x23, x23, x19 // bExtraStride

//  E12_Store_FP16 za0, 0
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

cbz x5, E12Store_0
.inst 0xc17fcbd8  // fclamp {z24.h-z27.h}, z30.h, z31.h
.inst 0xc17fcbc0  // fclamp {z0.h-z3.h}, z30.h, z31.h

E12Store_0:
cmp x20, #1
beq E12StoreH8_0

add x21, x22, x7
.inst 0xa06002d8  // st1b {z24.b-z25.b}, pn8, [x22]
.inst 0xe402eeda  // st1b {z26.b}, p3, [x22, #2, MUL VL]
.inst 0xa06002a0  // st1b {z0.b-z1.b}, pn8, [x21]
.inst 0xe402eea2  // st1b {z2.b}, p3, [x21, #2, MUL VL]
b E12H16_End_0

E12StoreH8_0:
.inst 0xa06002d8  // st1b {z24.b-z25.b}, pn8, [x22]
.inst 0xe402eeda  // st1b {z26.b}, p3, [x22, #2, MUL VL]

E12H16_End_0:
sub x20, x20, #2
add x22, x22, x7, LSL #1
cmp x20, #0
ble E12H16_End

//  E12_Store_FP16 za1, 1
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

cbz x5, E12Store_1
.inst 0xc17fcbd8  // fclamp {z24.h-z27.h}, z30.h, z31.h
.inst 0xc17fcbc0  // fclamp {z0.h-z3.h}, z30.h, z31.h

E12Store_1:
cmp x20, #1
beq E12StoreH8_1

add x21, x22, x7
.inst 0xa06002d8  // st1b {z24.b-z25.b}, pn8, [x22]
.inst 0xe402eeda  // st1b {z26.b}, p3, [x22, #2, MUL VL]
.inst 0xa06002a0  // st1b {z0.b-z1.b}, pn8, [x21]
.inst 0xe402eea2  // st1b {z2.b}, p3, [x21, #2, MUL VL]
b E12H16_End_1

E12StoreH8_1:
.inst 0xa06002d8  // st1b {z24.b-z25.b}, pn8, [x22]
.inst 0xe402eeda  // st1b {z26.b}, p3, [x22, #2, MUL VL]

E12H16_End_1:
sub x20, x20, #2
add x22, x22, x7, LSL #1
cmp x20, #0
ble E12H16_End

//  E12_Store_FP16 za2, 2
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

cbz x5, E12Store_2
.inst 0xc17fcbd8  // fclamp {z24.h-z27.h}, z30.h, z31.h
.inst 0xc17fcbc0  // fclamp {z0.h-z3.h}, z30.h, z31.h

E12Store_2:
cmp x20, #1
beq E12StoreH8_2

add x21, x22, x7
.inst 0xa06002d8  // st1b {z24.b-z25.b}, pn8, [x22]
.inst 0xe402eeda  // st1b {z26.b}, p3, [x22, #2, MUL VL]
.inst 0xa06002a0  // st1b {z0.b-z1.b}, pn8, [x21]
.inst 0xe402eea2  // st1b {z2.b}, p3, [x21, #2, MUL VL]
b E12H16_End_2

E12StoreH8_2:
.inst 0xa06002d8  // st1b {z24.b-z25.b}, pn8, [x22]
.inst 0xe402eeda  // st1b {z26.b}, p3, [x22, #2, MUL VL]

E12H16_End_2:
sub x20, x20, #2
add x22, x22, x7, LSL #1
cmp x20, #0
ble E12H16_End

//  E12_Store_FP16 za3, 3
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

cbz x5, E12Store_3
.inst 0xc17fcbd8  // fclamp {z24.h-z27.h}, z30.h, z31.h
.inst 0xc17fcbc0  // fclamp {z0.h-z3.h}, z30.h, z31.h

E12Store_3:
cmp x20, #1
beq E12StoreH8_3

add x21, x22, x7
.inst 0xa06002d8  // st1b {z24.b-z25.b}, pn8, [x22]
.inst 0xe402eeda  // st1b {z26.b}, p3, [x22, #2, MUL VL]
.inst 0xa06002a0  // st1b {z0.b-z1.b}, pn8, [x21]
.inst 0xe402eea2  // st1b {z2.b}, p3, [x21, #2, MUL VL]
b E12H16_End_3

E12StoreH8_3:
.inst 0xa06002d8  // st1b {z24.b-z25.b}, pn8, [x22]
.inst 0xe402eeda  // st1b {z26.b}, p3, [x22, #2, MUL VL]

E12H16_End_3:
sub x20, x20, #2
add x22, x22, x7, LSL #1

E12H16_End:
cmp x20, #0
bgt E12LoopH

E12End:
sub x3, x3, #12
add x0, x0, #192 // 12*pack*sizeof(float16)
add x1, x1, #48  // 12*LP*sizeof(float16)


E8:
cmp x3, #8
blt E4

mov x20, x10 // ocDiv8
mov x11, x6 // bias
mov x22, x0 // dst
mov x23, x2 // B

E8LoopH:
mov x8, x1 // A
mov x21, x9 // LU

.inst 0xc00800ff  // zero {za}

cbz x11, E8LoopL
// add bias
.inst 0xa0400168  // ld1b {z8.b-z9.b}, pn8/z, [x11]
.inst 0x042b504b  // addvl x11, x11, #2
.inst 0x25b9ce05  // fmov z5.s, #1

.inst 0x6589ad0c  // fcvt z12.s, p3/m, z8.h
.inst 0x6489ad0d  // fcvtlt z13.s, p3/m, z8.h
.inst 0x6589ad2e  // fcvt z14.s, p3/m, z9.h
.inst 0x6489ad2f  // fcvtlt z15.s, p3/m, z9.h

.inst 0x05ad6194  // zip1 z20.s, z12.s, z13.s
.inst 0x05ad6595  // zip2 z21.s, z12.s, z13.s
.inst 0x05af61d6  // zip1 z22.s, z14.s, z15.s
.inst 0x05af65d7  // zip2 z23.s, z14.s, z15.s

.inst 0x80946ca0 // fmopa za0.s, p3/m, p3/m, z5.s, z20.s
.inst 0x80956ca1 // fmopa za1.s, p3/m, p3/m, z5.s, z21.s
.inst 0x80966ca2 // fmopa za2.s, p3/m, p3/m, z5.s, z22.s
.inst 0x80976ca3 // fmopa za3.s, p3/m, p3/m, z5.s, z23.s

E8LoopL:
.inst 0xa400ad04  // ld1b {z4.b}, p3/z, [x8]  // A
.inst 0xa04082e0  // ld1b {z0.b-z3.b}, pn8/z, [x23]  // B
// [EP,LP] x [HP,LP] -> [EP,HP]
.inst 0x81a06480 // fmopa za0.s, p1/m, p3/m, z4.h, z0.h
.inst 0x81a16481 // fmopa za1.s, p1/m, p3/m, z4.h, z1.h
.inst 0x81a26482 // fmopa za2.s, p1/m, p3/m, z4.h, z2.h
.inst 0x81a36483 // fmopa za3.s, p1/m, p3/m, z4.h, z3.h

subs x21, x21, #1
.inst 0x04285028  // addvl x8, x8, #1
.inst 0x04375097  // addvl x23, x23, #4
bne E8LoopL

add x23, x23, x19 // bExtraStride

//  E8_Store_FP16 za0, 0
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

cbz x5, E8Store_0
.inst 0xc17fcbd8  // fclamp {z24.h-z27.h}, z30.h, z31.h
.inst 0xc17fcbc0  // fclamp {z0.h-z3.h}, z30.h, z31.h

E8Store_0:
cmp x20, #1
beq E8StoreH8_0

.inst 0xa06002d8  // st1b {z24.b-z25.b}, pn8, [x22]
.inst 0xa02702c0  // st1b {z0.b-z1.b}, pn8, [x22, x7]
b E8H16_End_0

E8StoreH8_0:
.inst 0xa06002d8  // st1b {z24.b-z25.b}, pn8, [x22]

E8H16_End_0:
sub x20, x20, #2
add x22, x22, x7, LSL #1
cmp x20, #0
ble E8H16_End

//  E8_Store_FP16 za1, 1
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

cbz x5, E8Store_1
.inst 0xc17fcbd8  // fclamp {z24.h-z27.h}, z30.h, z31.h
.inst 0xc17fcbc0  // fclamp {z0.h-z3.h}, z30.h, z31.h

E8Store_1:
cmp x20, #1
beq E8StoreH8_1

.inst 0xa06002d8  // st1b {z24.b-z25.b}, pn8, [x22]
.inst 0xa02702c0  // st1b {z0.b-z1.b}, pn8, [x22, x7]
b E8H16_End_1

E8StoreH8_1:
.inst 0xa06002d8  // st1b {z24.b-z25.b}, pn8, [x22]

E8H16_End_1:
sub x20, x20, #2
add x22, x22, x7, LSL #1
cmp x20, #0
ble E8H16_End

//  E8_Store_FP16 za2, 2
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

cbz x5, E8Store_2
.inst 0xc17fcbd8  // fclamp {z24.h-z27.h}, z30.h, z31.h
.inst 0xc17fcbc0  // fclamp {z0.h-z3.h}, z30.h, z31.h

E8Store_2:
cmp x20, #1
beq E8StoreH8_2

.inst 0xa06002d8  // st1b {z24.b-z25.b}, pn8, [x22]
.inst 0xa02702c0  // st1b {z0.b-z1.b}, pn8, [x22, x7]
b E8H16_End_2

E8StoreH8_2:
.inst 0xa06002d8  // st1b {z24.b-z25.b}, pn8, [x22]

E8H16_End_2:
sub x20, x20, #2
add x22, x22, x7, LSL #1
cmp x20, #0
ble E8H16_End

//  E8_Store_FP16 za3, 3
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

cbz x5, E8Store_3
.inst 0xc17fcbd8  // fclamp {z24.h-z27.h}, z30.h, z31.h
.inst 0xc17fcbc0  // fclamp {z0.h-z3.h}, z30.h, z31.h

E8Store_3:
cmp x20, #1
beq E8StoreH8_3

.inst 0xa06002d8  // st1b {z24.b-z25.b}, pn8, [x22]
.inst 0xa02702c0  // st1b {z0.b-z1.b}, pn8, [x22, x7]
b E8H16_End_3

E8StoreH8_3:
.inst 0xa06002d8  // st1b {z24.b-z25.b}, pn8, [x22]

E8H16_End_3:
sub x20, x20, #2
add x22, x22, x7, LSL #1

E8H16_End:
cmp x20, #0
bgt E8LoopH

E8End:
sub x3, x3, #8
add x0, x0, #128 // 8*pack*sizeof(float16)
add x1, x1, #32  // 8*LP*sizeof(float16)

E4:
cmp x3, #4
blt E2

mov x20, x10 // ocDiv8
mov x11, x6 // bias
mov x22, x0 // dst
mov x23, x2 // B

E4LoopH:
mov x8, x1 // A
mov x21, x9 // LU

.inst 0xc00800ff  // zero {za}

cbz x11, E4LoopL
// add bias
.inst 0xa0400168  // ld1b {z8.b-z9.b}, pn8/z, [x11]
.inst 0x042b504b  // addvl x11, x11, #2
.inst 0x25b9ce05  // fmov z5.s, #1

.inst 0x6589ad0c  // fcvt z12.s, p3/m, z8.h
.inst 0x6489ad0d  // fcvtlt z13.s, p3/m, z8.h
.inst 0x6589ad2e  // fcvt z14.s, p3/m, z9.h
.inst 0x6489ad2f  // fcvtlt z15.s, p3/m, z9.h

.inst 0x05ad6194  // zip1 z20.s, z12.s, z13.s
.inst 0x05ad6595  // zip2 z21.s, z12.s, z13.s
.inst 0x05af61d6  // zip1 z22.s, z14.s, z15.s
.inst 0x05af65d7  // zip2 z23.s, z14.s, z15.s

.inst 0x80946ca0 // fmopa za0.s, p3/m, p3/m, z5.s, z20.s
.inst 0x80956ca1 // fmopa za1.s, p3/m, p3/m, z5.s, z21.s
.inst 0x80966ca2 // fmopa za2.s, p3/m, p3/m, z5.s, z22.s
.inst 0x80976ca3 // fmopa za3.s, p3/m, p3/m, z5.s, z23.s

E4LoopL:
.inst 0xa400ad04  // ld1b {z4.b}, p3/z, [x8]  // A
.inst 0xa04082e0  // ld1b {z0.b-z3.b}, pn8/z, [x23]  // B
// [EP,LP] x [HP,LP] -> [EP,HP]
.inst 0x81a06480 // fmopa za0.s, p1/m, p3/m, z4.h, z0.h
.inst 0x81a16481 // fmopa za1.s, p1/m, p3/m, z4.h, z1.h
.inst 0x81a26482 // fmopa za2.s, p1/m, p3/m, z4.h, z2.h
.inst 0x81a36483 // fmopa za3.s, p1/m, p3/m, z4.h, z3.h

subs x21, x21, #1
.inst 0x04285028  // addvl x8, x8, #1
.inst 0x04375097  // addvl x23, x23, #4
bne E4LoopL

add x23, x23, x19 // bExtraStride

//  E4_Store_FP16 za0, 0
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

cbz x5, E4Store_0
.inst 0xc17fcbd8  // fclamp {z24.h-z27.h}, z30.h, z31.h
.inst 0xc17fcbc0  // fclamp {z0.h-z3.h}, z30.h, z31.h

E4Store_0:
cmp x20, #1
beq E4StoreH8_0

.inst 0xe400eed8  // st1b {z24.b}, p3, [x22]
.inst 0xe4074ec0  // st1b {z0.b}, p3, [x22, x7]
b E4H16_End_0

E4StoreH8_0:
.inst 0xe400eed8  // st1b {z24.b}, p3, [x22]

E4H16_End_0:
sub x20, x20, #2
add x22, x22, x7, LSL #1
cmp x20, #0
ble E4H16_End

//  E4_Store_FP16 za1, 1
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

cbz x5, E4Store_1
.inst 0xc17fcbd8  // fclamp {z24.h-z27.h}, z30.h, z31.h
.inst 0xc17fcbc0  // fclamp {z0.h-z3.h}, z30.h, z31.h

E4Store_1:
cmp x20, #1
beq E4StoreH8_1

.inst 0xe400eed8  // st1b {z24.b}, p3, [x22]
.inst 0xe4074ec0  // st1b {z0.b}, p3, [x22, x7]
b E4H16_End_1

E4StoreH8_1:
.inst 0xe400eed8  // st1b {z24.b}, p3, [x22]

E4H16_End_1:
sub x20, x20, #2
add x22, x22, x7, LSL #1
cmp x20, #0
ble E4H16_End

//  E4_Store_FP16 za2, 2
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

cbz x5, E4Store_2
.inst 0xc17fcbd8  // fclamp {z24.h-z27.h}, z30.h, z31.h
.inst 0xc17fcbc0  // fclamp {z0.h-z3.h}, z30.h, z31.h

E4Store_2:
cmp x20, #1
beq E4StoreH8_2

.inst 0xe400eed8  // st1b {z24.b}, p3, [x22]
.inst 0xe4074ec0  // st1b {z0.b}, p3, [x22, x7]
b E4H16_End_2

E4StoreH8_2:
.inst 0xe400eed8  // st1b {z24.b}, p3, [x22]

E4H16_End_2:
sub x20, x20, #2
add x22, x22, x7, LSL #1
cmp x20, #0
ble E4H16_End

//  E4_Store_FP16 za3, 3
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

cbz x5, E4Store_3
.inst 0xc17fcbd8  // fclamp {z24.h-z27.h}, z30.h, z31.h
.inst 0xc17fcbc0  // fclamp {z0.h-z3.h}, z30.h, z31.h

E4Store_3:
cmp x20, #1
beq E4StoreH8_3

.inst 0xe400eed8  // st1b {z24.b}, p3, [x22]
.inst 0xe4074ec0  // st1b {z0.b}, p3, [x22, x7]
b E4H16_End_3

E4StoreH8_3:
.inst 0xe400eed8  // st1b {z24.b}, p3, [x22]

E4H16_End_3:
sub x20, x20, #2
add x22, x22, x7, LSL #1

E4H16_End:
cmp x20, #0
bgt E4LoopH

E4End:
sub x3, x3, #4
add x0, x0, #64 // 4*pack*sizeof(float16)
add x1, x1, #16  // 4*LP*sizeof(float16)

E2:
cmp x3, #2
blt E1

mov x20, x10 // ocDiv8
mov x11, x6 // bias
mov x22, x0 // dst
mov x23, x2 // B

E2LoopH:
mov x8, x1 // A
mov x21, x9 // LU

.inst 0xc00800ff  // zero {za}

cbz x11, E2LoopL
// add bias
.inst 0xa0400168  // ld1b {z8.b-z9.b}, pn8/z, [x11]
.inst 0x042b504b  // addvl x11, x11, #2
.inst 0x25b9ce05  // fmov z5.s, #1

.inst 0x6589ad0c  // fcvt z12.s, p3/m, z8.h
.inst 0x6489ad0d  // fcvtlt z13.s, p3/m, z8.h
.inst 0x6589ad2e  // fcvt z14.s, p3/m, z9.h
.inst 0x6489ad2f  // fcvtlt z15.s, p3/m, z9.h

.inst 0x05ad6194  // zip1 z20.s, z12.s, z13.s
.inst 0x05ad6595  // zip2 z21.s, z12.s, z13.s
.inst 0x05af61d6  // zip1 z22.s, z14.s, z15.s
.inst 0x05af65d7  // zip2 z23.s, z14.s, z15.s

.inst 0x80946ca0 // fmopa za0.s, p3/m, p3/m, z5.s, z20.s
.inst 0x80956ca1 // fmopa za1.s, p3/m, p3/m, z5.s, z21.s
.inst 0x80966ca2 // fmopa za2.s, p3/m, p3/m, z5.s, z22.s
.inst 0x80976ca3 // fmopa za3.s, p3/m, p3/m, z5.s, z23.s

E2LoopL:
.inst 0xa400ad04  // ld1b {z4.b}, p3/z, [x8]  // A
.inst 0xa04082e0  // ld1b {z0.b-z3.b}, pn8/z, [x23]  // B
// [EP,LP] x [HP,LP] -> [EP,HP]
.inst 0x81a06480 // fmopa za0.s, p1/m, p3/m, z4.h, z0.h
.inst 0x81a16481 // fmopa za1.s, p1/m, p3/m, z4.h, z1.h
.inst 0x81a26482 // fmopa za2.s, p1/m, p3/m, z4.h, z2.h
.inst 0x81a36483 // fmopa za3.s, p1/m, p3/m, z4.h, z3.h

subs x21, x21, #1
.inst 0x04285028  // addvl x8, x8, #1
.inst 0x04375097  // addvl x23, x23, #4
bne E2LoopL

add x23, x23, x19 // bExtraStride

//  E2_Store_FP16 za0, 0
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

cbz x5, E2Store_0
.inst 0xc17fcbd8  // fclamp {z24.h-z27.h}, z30.h, z31.h
.inst 0xc17fcbc0  // fclamp {z0.h-z3.h}, z30.h, z31.h

E2Store_0:
cmp x20, #1
beq E2StoreH8_0

.inst 0xe400e6d8  // st1b {z24.b}, p1, [x22]
.inst 0xe40746c0  // st1b {z0.b}, p1, [x22, x7]
b E2H16_End_0

E2StoreH8_0:
.inst 0xe400e6d8  // st1b {z24.b}, p1, [x22]

E2H16_End_0:
sub x20, x20, #2
add x22, x22, x7, LSL #1
cmp x20, #0
ble E2H16_End

//  E2_Store_FP16 za1, 1
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

cbz x5, E2Store_1
.inst 0xc17fcbd8  // fclamp {z24.h-z27.h}, z30.h, z31.h
.inst 0xc17fcbc0  // fclamp {z0.h-z3.h}, z30.h, z31.h

E2Store_1:
cmp x20, #1
beq E2StoreH8_1

.inst 0xe400e6d8  // st1b {z24.b}, p1, [x22]
.inst 0xe40746c0  // st1b {z0.b}, p1, [x22, x7]
b E2H16_End_1

E2StoreH8_1:
.inst 0xe400e6d8  // st1b {z24.b}, p1, [x22]

E2H16_End_1:
sub x20, x20, #2
add x22, x22, x7, LSL #1
cmp x20, #0
ble E2H16_End

//  E2_Store_FP16 za2, 2
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

cbz x5, E2Store_2
.inst 0xc17fcbd8  // fclamp {z24.h-z27.h}, z30.h, z31.h
.inst 0xc17fcbc0  // fclamp {z0.h-z3.h}, z30.h, z31.h

E2Store_2:
cmp x20, #1
beq E2StoreH8_2

.inst 0xe400e6d8  // st1b {z24.b}, p1, [x22]
.inst 0xe40746c0  // st1b {z0.b}, p1, [x22, x7]
b E2H16_End_2

E2StoreH8_2:
.inst 0xe400e6d8  // st1b {z24.b}, p1, [x22]

E2H16_End_2:
sub x20, x20, #2
add x22, x22, x7, LSL #1
cmp x20, #0
ble E2H16_End

//  E2_Store_FP16 za3, 3
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

cbz x5, E2Store_3
.inst 0xc17fcbd8  // fclamp {z24.h-z27.h}, z30.h, z31.h
.inst 0xc17fcbc0  // fclamp {z0.h-z3.h}, z30.h, z31.h

E2Store_3:
cmp x20, #1
beq E2StoreH8_3

.inst 0xe400e6d8  // st1b {z24.b}, p1, [x22]
.inst 0xe40746c0  // st1b {z0.b}, p1, [x22, x7]
b E2H16_End_3

E2StoreH8_3:
.inst 0xe400e6d8  // st1b {z24.b}, p1, [x22]

E2H16_End_3:
sub x20, x20, #2
add x22, x22, x7, LSL #1


E2H16_End:
cmp x20, #0
bgt E2LoopH

E2End:
sub x3, x3, #2
add x0, x0, #32 // 2*pack*sizeof(float16)
add x1, x1, #8  // 2*LP*sizeof(float16)

E1:
cmp x3, #1
blt End

mov x20, x10 // ocDiv8
mov x11, x6 // bias
mov x22, x0 // dst
mov x23, x2 // B

E1LoopH:
mov x8, x1 // A
mov x21, x9 // LU

.inst 0xc00800ff  // zero {za}

cbz x11, E1LoopL
// add bias
.inst 0xa0400168  // ld1b {z8.b-z9.b}, pn8/z, [x11]
.inst 0x042b504b  // addvl x11, x11, #2
.inst 0x25b9ce05  // fmov z5.s, #1

.inst 0x6589ad0c  // fcvt z12.s, p3/m, z8.h
.inst 0x6489ad0d  // fcvtlt z13.s, p3/m, z8.h
.inst 0x6589ad2e  // fcvt z14.s, p3/m, z9.h
.inst 0x6489ad2f  // fcvtlt z15.s, p3/m, z9.h

.inst 0x05ad6194  // zip1 z20.s, z12.s, z13.s
.inst 0x05ad6595  // zip2 z21.s, z12.s, z13.s
.inst 0x05af61d6  // zip1 z22.s, z14.s, z15.s
.inst 0x05af65d7  // zip2 z23.s, z14.s, z15.s

.inst 0x80946ca0 // fmopa za0.s, p3/m, p3/m, z5.s, z20.s
.inst 0x80956ca1 // fmopa za1.s, p3/m, p3/m, z5.s, z21.s
.inst 0x80966ca2 // fmopa za2.s, p3/m, p3/m, z5.s, z22.s
.inst 0x80976ca3 // fmopa za3.s, p3/m, p3/m, z5.s, z23.s

E1LoopL:
.inst 0xa400ad04  // ld1b {z4.b}, p3/z, [x8]  // A
.inst 0xa04082e0  // ld1b {z0.b-z3.b}, pn8/z, [x23]  // B
// [EP,LP] x [HP,LP] -> [EP,HP]
.inst 0x81a06c80 // fmopa za0.s, p3/m, p3/m, z4.h, z0.h
.inst 0x81a16c81 // fmopa za1.s, p3/m, p3/m, z4.h, z1.h
.inst 0x81a26c82 // fmopa za2.s, p3/m, p3/m, z4.h, z2.h
.inst 0x81a36c83 // fmopa za3.s, p3/m, p3/m, z4.h, z3.h

subs x21, x21, #1
.inst 0x04285028  // addvl x8, x8, #1
.inst 0x04375097  // addvl x23, x23, #4
bne E1LoopL

add x23, x23, x19 // bExtraStride

mov w12, #0
.inst 0xc0820c00  // mova z0.s, p3/m, za0h.s[w12, 0]
.inst 0xc0820c81  // mova z1.s, p3/m, za1h.s[w12, 0]
.inst 0xc0820d02  // mova z2.s, p3/m, za2h.s[w12, 0]
.inst 0xc0820d83  // mova z3.s, p3/m, za3h.s[w12, 0]

.inst 0xc120e010  // fcvt z16.h, {z0.s-z1.s}
.inst 0xc120e051  // fcvt z17.h, {z2.s-z3.s}
cbz x5, E1Store
.inst 0xc17fc3d0  // fclamp {z16.h-z17.h}, z30.h, z31.h

E1Store:
cmp x20, #1
beq E1StoreH8

cmp x20, #2
beq E1StoreH16

cmp x20, #3
beq E1StoreH24

cmp x20, #4
beq E1StoreH32

cmp x20, #5
beq E1StoreH40

cmp x20, #6
beq E1StoreH48

cmp x20, #7
beq E1StoreH56

E1StoreH64:
add x21, x22, x7, LSL #1 // x22+2*x7
add x24, x22, x7, LSL #2 // x22+4*x7
add x25, x24, x7, LSL #1 // x25+6*x7
.inst 0x05702212  // dup z18.q, z16.q[1]
.inst 0x05b02213  // dup z19.q, z16.q[2]
.inst 0x05f02214  // dup z20.q, z16.q[3]
.inst 0x05702235  // dup z21.q, z17.q[1]
.inst 0x05b02236  // dup z22.q, z17.q[2]
.inst 0x05f02237  // dup z23.q, z17.q[3]
.inst 0xe400f2d0  // st1b {z16.b}, p4, [x22]
.inst 0xe40752d2  // st1b {z18.b}, p4, [x22, x7]
.inst 0xe400f2b3  // st1b {z19.b}, p4, [x21]
.inst 0xe40752b4  // st1b {z20.b}, p4, [x21, x7]
.inst 0xe400f311  // st1b {z17.b}, p4, [x24]
.inst 0xe4075315  // st1b {z21.b}, p4, [x24, x7]
.inst 0xe400f336  // st1b {z22.b}, p4, [x25]
.inst 0xe4075337  // st1b {z23.b}, p4, [x25, x7]
b E1H16_End

E1StoreH56:
add x21, x22, x7, LSL #1
add x24, x22, x7, LSL #2
add x25, x24, x7, LSL #1
.inst 0x05702212  // dup z18.q, z16.q[1]
.inst 0x05b02213  // dup z19.q, z16.q[2]
.inst 0x05f02214  // dup z20.q, z16.q[3]
.inst 0x05702235  // dup z21.q, z17.q[1]
.inst 0x05b02236  // dup z22.q, z17.q[2]
.inst 0xe400f2d0  // st1b {z16.b}, p4, [x22]
.inst 0xe40752d2  // st1b {z18.b}, p4, [x22, x7]
.inst 0xe400f2b3  // st1b {z19.b}, p4, [x21]
.inst 0xe40752b4  // st1b {z20.b}, p4, [x21, x7]
.inst 0xe400f311  // st1b {z17.b}, p4, [x24]
.inst 0xe4075315  // st1b {z21.b}, p4, [x24, x7]
.inst 0xe400f336  // st1b {z22.b}, p4, [x25]
b E1H16_End

E1StoreH48:
add x21, x22, x7, LSL #1
add x24, x22, x7, LSL #2
.inst 0x05702212  // dup z18.q, z16.q[1]
.inst 0x05b02213  // dup z19.q, z16.q[2]
.inst 0x05f02214  // dup z20.q, z16.q[3]
.inst 0x05702235  // dup z21.q, z17.q[1]
.inst 0xe400f2d0  // st1b {z16.b}, p4, [x22]
.inst 0xe40752d2  // st1b {z18.b}, p4, [x22, x7]
.inst 0xe400f2b3  // st1b {z19.b}, p4, [x21]
.inst 0xe40752b4  // st1b {z20.b}, p4, [x21, x7]
.inst 0xe400f311  // st1b {z17.b}, p4, [x24]
.inst 0xe4075315  // st1b {z21.b}, p4, [x24, x7]
b E1H16_End

E1StoreH40:
add x21, x22, x7, LSL #1
add x24, x22, x7, LSL #2
.inst 0x05702212  // dup z18.q, z16.q[1]
.inst 0x05b02213  // dup z19.q, z16.q[2]
.inst 0x05f02214  // dup z20.q, z16.q[3]
.inst 0xe400f2d0  // st1b {z16.b}, p4, [x22]
.inst 0xe40752d2  // st1b {z18.b}, p4, [x22, x7]
.inst 0xe400f2b3  // st1b {z19.b}, p4, [x21]
.inst 0xe40752b4  // st1b {z20.b}, p4, [x21, x7]
.inst 0xe400f311  // st1b {z17.b}, p4, [x24]
b E1H16_End

E1StoreH32:
add x21, x22, x7, LSL #1
.inst 0x05702212  // dup z18.q, z16.q[1]
.inst 0x05b02213  // dup z19.q, z16.q[2]
.inst 0x05f02214  // dup z20.q, z16.q[3]
.inst 0xe400f2d0  // st1b {z16.b}, p4, [x22]
.inst 0xe40752d2  // st1b {z18.b}, p4, [x22, x7]
.inst 0xe400f2b3  // st1b {z19.b}, p4, [x21]
.inst 0xe40752b4  // st1b {z20.b}, p4, [x21, x7]
b E1H16_End

E1StoreH24:
add x21, x22, x7, LSL #1
.inst 0x05702212  // dup z18.q, z16.q[1]
.inst 0x05b02213  // dup z19.q, z16.q[2]
.inst 0xe400f2d0  // st1b {z16.b}, p4, [x22]
.inst 0xe40752d2  // st1b {z18.b}, p4, [x22, x7]
.inst 0xe400f2b3  // st1b {z19.b}, p4, [x21]
b E1H16_End

E1StoreH16:
.inst 0x05702212  // dup z18.q, z16.q[1]
.inst 0xe400f2d0  // st1b {z16.b}, p4, [x22]
.inst 0xe40752d2  // st1b {z18.b}, p4, [x22, x7]
b E1H16_End

E1StoreH8:
.inst 0xe400f2d0  // st1b {z16.b}, p4, [x22]

E1H16_End:
sub x20, x20, #8
add x22, x22, x7, LSL #3
cmp x20, #0
bgt E1LoopH

E1End:
subs x3, x3, #1
add x0, x0, #16 // 1*pack*sizeof(float16)
add x1, x1, #4  // 1*LP*sizeof(float16)
bne E1

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

#endif
