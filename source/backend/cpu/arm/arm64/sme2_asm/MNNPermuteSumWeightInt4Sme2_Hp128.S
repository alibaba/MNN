#ifdef __aarch64__
#include "MNNAsmGlobal.h"

.text
.align 5

.macro SET_0 s0, s1, s2, s3
    movi \s0\().4s, #0
    movi \s1\().4s, #0
    movi \s2\().4s, #0
    movi \s3\().4s, #0
.endm

asm_function MNNPermuteSumWeightInt4Sme2_Hp128
// void MNNPermuteSumWeightInt4Sme2_Hp128(uint8_t* dest, uint8_t* source, size_t outside, size_t inside, float* kernelSum, int32_t* table);
// auto load: x0: dest, x1: source, x2: outside, x3: inside, x4: kernelSum, x5: table

// inside = lu
// outside = blocknum*hu
// kernelSum shape: [hu, blockNum, hp]


stp d14, d15, [sp, #-64]!
stp d12, d13, [sp, #16]
stp d10, d11, [sp, #32]
stp d8,  d9,  [sp, #48]

.inst 0xd503477f  // smstart
.inst 0x2518e3e3  // ptrue p3.b
.inst 0x25207810  // ptrue pn8.b
mov w8, #0
mov w9, #4
mov w10, #8
mov w11, #12
.inst 0x2538c02f  // mov z15.b, #1

.inst 0xe11f80a0  // ldr zt0, [x5]      // table

Loop: // blocknum*hu
mov x6, x3     // lu

.inst 0xc00800ff  // zero {za}

cmp x6, #2
blt LoopLU

LoopLU2:
.inst 0xa0408020  // ld1b {z0.b-z3.b}, pn8/z, [x1]
.inst 0xa0418030  // ld1b {z16.b-z19.b}, pn8/z, [x1, #4, MUL VL]

.inst 0xc08a4004  // luti4 {z4.b-z5.b}, zt0, z0[0]
.inst 0xc08a4026  // luti4 {z6.b-z7.b}, zt0, z1[0]
.inst 0xc08a4048  // luti4 {z8.b-z9.b}, zt0, z2[0]
.inst 0xc08a406a  // luti4 {z10.b-z11.b}, zt0, z3[0]

.inst 0xc08a4214  // luti4 {z20.b-z21.b}, zt0, z16[0]
.inst 0xc08a4236  // luti4 {z22.b-z23.b}, zt0, z17[0]
.inst 0xc08a4258  // luti4 {z24.b-z25.b}, zt0, z18[0]
.inst 0xc08a427a  // luti4 {z26.b-z27.b}, zt0, z19[0]

.inst 0x04215101  // addvl x1, x1, #8

.inst 0xc15f90b0  // udot za.s[w8, 0, VGx4], {z4.b-z7.b}, z15.b[0]
.inst 0xc15fb130  // udot za.s[w9, 0, VGx4], {z8.b-z11.b}, z15.b[0]
.inst 0xc15fd2b0  // udot za.s[w10, 0, VGx4], {z20.b-z23.b}, z15.b[0]
.inst 0xc15ff330  // udot za.s[w11, 0, VGx4], {z24.b-z27.b}, z15.b[0]

.inst 0x05648c8c  // revb z12.h, p3/m, z4.h
.inst 0x05648cad  // revb z13.h, p3/m, z5.h
.inst 0x05648cce  // revb z14.h, p3/m, z6.h
.inst 0x05648cf0  // revb z16.h, p3/m, z7.h
.inst 0x05648d11  // revb z17.h, p3/m, z8.h
.inst 0x05648d32  // revb z18.h, p3/m, z9.h
.inst 0x05648d53  // revb z19.h, p3/m, z10.h
.inst 0x05648d7c  // revb z28.h, p3/m, z11.h


.inst 0x05648e9d  // revb z29.h, p3/m, z20.h
.inst 0x05648ebe  // revb z30.h, p3/m, z21.h
.inst 0x05648edf  // revb z31.h, p3/m, z22.h
.inst 0x05648ee0  // revb z0.h, p3/m, z23.h
.inst 0x05648f01  // revb z1.h, p3/m, z24.h
.inst 0x05648f22  // revb z2.h, p3/m, z25.h
.inst 0x05648f43  // revb z3.h, p3/m, z26.h
.inst 0x05648f64  // revb z4.h, p3/m, z27.h

.inst 0xc12dd187  // uzp {z6.b, z7.b}, z12.b, z13.b
.inst 0xc130d1c9  // uzp {z8.b, z9.b}, z14.b, z16.b
.inst 0xc132d22b  // uzp {z10.b, z11.b}, z17.b, z18.b
.inst 0xc13cd26d  // uzp {z12.b, z13.b}, z19.b, z28.b
.inst 0xc13ed3b7  // uzp {z22.b, z23.b}, z29.b, z30.b
.inst 0xc120d3f1  // uzp {z16.b, z17.b}, z31.b, z0.b
.inst 0xc122d033  // uzp {z18.b, z19.b}, z1.b, z2.b
.inst 0xc124d075  // uzp {z20.b, z21.b}, z3.b, z4.b


.inst 0x04078d87  // uqshl z7.b, p3/m, z7.b, #4
.inst 0x04078d89  // uqshl z9.b, p3/m, z9.b, #4
.inst 0x04078d8b  // uqshl z11.b, p3/m, z11.b, #4
.inst 0x04078d8d  // uqshl z13.b, p3/m, z13.b, #4
.inst 0x04078d97  // uqshl z23.b, p3/m, z23.b, #4
.inst 0x04078d91  // uqshl z17.b, p3/m, z17.b, #4
.inst 0x04078d93  // uqshl z19.b, p3/m, z19.b, #4
.inst 0x04078d95  // uqshl z21.b, p3/m, z21.b, #4

sub x6, x6, #2

.inst 0x04180cc7  // orr z7.b, p3/m, z7.b, z6.b
.inst 0x04180d09  // orr z9.b, p3/m, z9.b, z8.b
.inst 0x04180d4b  // orr z11.b, p3/m, z11.b, z10.b
.inst 0x04180d8d  // orr z13.b, p3/m, z13.b, z12.b
.inst 0x04180ed7  // orr z23.b, p3/m, z23.b, z22.b
.inst 0x04180e11  // orr z17.b, p3/m, z17.b, z16.b
.inst 0x04180e53  // orr z19.b, p3/m, z19.b, z18.b
.inst 0x04180e95  // orr z21.b, p3/m, z21.b, z20.b

cmp x6, #2

.inst 0xe400ec07  // st1b {z7.b}, p3, [x0]
.inst 0xe401ec09  // st1b {z9.b}, p3, [x0, #1, MUL VL]
.inst 0xe402ec0b  // st1b {z11.b}, p3, [x0, #2, MUL VL]
.inst 0xe403ec0d  // st1b {z13.b}, p3, [x0, #3, MUL VL]
.inst 0xe404ec17  // st1b {z23.b}, p3, [x0, #4, MUL VL]
.inst 0xe405ec11  // st1b {z17.b}, p3, [x0, #5, MUL VL]
.inst 0xe406ec13  // st1b {z19.b}, p3, [x0, #6, MUL VL]
.inst 0xe407ec15  // st1b {z21.b}, p3, [x0, #7, MUL VL]
.inst 0x04205100  // addvl x0, x0, #8

bge LoopLU2
cbz x6, LUEnd

LoopLU:

.inst 0xa0408020  // ld1b {z0.b-z3.b}, pn8/z, [x1]
.inst 0x04215081  // addvl x1, x1, #4

.inst 0xc08a4004  // luti4 {z4.b-z5.b}, zt0, z0[0]
.inst 0xc08a4026  // luti4 {z6.b-z7.b}, zt0, z1[0]
.inst 0xc08a4048  // luti4 {z8.b-z9.b}, zt0, z2[0]
.inst 0xc08a406a  // luti4 {z10.b-z11.b}, zt0, z3[0]

.inst 0xc15f90b0  // udot za.s[w8, 0, VGx4], {z4.b-z7.b}, z15.b[0]
.inst 0xc15fb130  // udot za.s[w9, 0, VGx4], {z8.b-z11.b}, z15.b[0]


.inst 0x05648c8c  // revb z12.h, p3/m, z4.h
.inst 0x05648cad  // revb z13.h, p3/m, z5.h
.inst 0x05648cce  // revb z14.h, p3/m, z6.h
.inst 0x05648cf0  // revb z16.h, p3/m, z7.h

.inst 0x05648d00  // revb z0.h, p3/m, z8.h
.inst 0x05648d21  // revb z1.h, p3/m, z9.h
.inst 0x05648d42  // revb z2.h, p3/m, z10.h
.inst 0x05648d63  // revb z3.h, p3/m, z11.h

.inst 0xc12dd195  // uzp {z20.b, z21.b}, z12.b, z13.b
.inst 0xc130d1d7  // uzp {z22.b, z23.b}, z14.b, z16.b
.inst 0xc121d019  // uzp {z24.b, z25.b}, z0.b, z1.b
.inst 0xc123d05b  // uzp {z26.b, z27.b}, z2.b, z3.b

.inst 0x04078d95  // uqshl z21.b, p3/m, z21.b, #4
.inst 0x04078d97  // uqshl z23.b, p3/m, z23.b, #4
.inst 0x04078d99  // uqshl z25.b, p3/m, z25.b, #4
.inst 0x04078d9b  // uqshl z27.b, p3/m, z27.b, #4

.inst 0x04180e95  // orr z21.b, p3/m, z21.b, z20.b
.inst 0x04180ed7  // orr z23.b, p3/m, z23.b, z22.b
.inst 0x04180f19  // orr z25.b, p3/m, z25.b, z24.b
.inst 0x04180f5b  // orr z27.b, p3/m, z27.b, z26.b

subs x6, x6, #1
.inst 0xe400ec15  // st1b {z21.b}, p3, [x0]
.inst 0xe401ec17  // st1b {z23.b}, p3, [x0, #1, MUL VL]
.inst 0xe402ec19  // st1b {z25.b}, p3, [x0, #2, MUL VL]
.inst 0xe403ec1b  // st1b {z27.b}, p3, [x0, #3, MUL VL]
.inst 0x04205080  // addvl x0, x0, #4
bne LoopLU


LUEnd:
.inst 0xc0060c00  // mova {z0.s-z3.s}, za.s[w8, 0, VGx4]
.inst 0xc0062c04  // mova {z4.s-z7.s}, za.s[w9, 0, VGx4]
.inst 0xc0064c08  // mova {z8.s-z11.s}, za.s[w10, 0, VGx4]
.inst 0xc0066c10  // mova {z16.s-z19.s}, za.s[w11, 0, VGx4]

.inst 0x04a80000  // add z0.s, z0.s, z8.s
.inst 0x04a90021  // add z1.s, z1.s, z9.s
.inst 0x04aa0042  // add z2.s, z2.s, z10.s
.inst 0x04ab0063  // add z3.s, z3.s, z11.s
.inst 0x04b00084  // add z4.s, z4.s, z16.s
.inst 0x04b100a5  // add z5.s, z5.s, z17.s
.inst 0x04b200c6  // add z6.s, z6.s, z18.s
.inst 0x04b300e7  // add z7.s, z7.s, z19.s

.inst 0xc132e000  // scvtf {z0.s-z3.s}, {z0.s-z3.s}
.inst 0xc132e084  // scvtf {z4.s-z7.s}, {z4.s-z7.s}

.inst 0xa0608080  // st1b {z0.b-z3.b}, pn8, [x4]
.inst 0xa0618084  // st1b {z4.b-z7.b}, pn8, [x4, #4, MUL VL]
.inst 0x04245104  // addvl x4, x4, #8

subs x2, x2, #1 // outside--
bne Loop


End:
.inst 0xd503467f  // smstop
    ldp d8,  d9,  [sp, #48]
    ldp d10, d11, [sp, #32]
    ldp d12, d13, [sp, #16]
    ldp d14, d15, [sp], #64
    ret

#endif
