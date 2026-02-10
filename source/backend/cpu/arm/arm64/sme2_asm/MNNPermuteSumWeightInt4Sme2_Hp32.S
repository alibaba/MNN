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

asm_function MNNPermuteSumWeightInt4Sme2_Hp32
// void MNNPermuteSumWeightInt4Sme2_Hp32(uint8_t* dest, uint8_t* source, size_t outside, size_t inside, float* kernelSum, int32_t* table);
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
.inst 0x2538c02f  // mov z15.b, #1

.inst 0xe11f80a0  // ldr zt0, [x5]    // table

Loop: // blocknum*hu
mov x6, x3     // lu

.inst 0xc00800ff  // zero {za}
.inst 0x2538c01e  // mov z30.b, #0
.inst 0x2538c01f  // mov z31.b, #0
cmp x6, #4
blt LoopLU

LoopLU4:
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
.inst 0x05648cfc  // revb z28.h, p3/m, z7.h
.inst 0x05648d10  // revb z16.h, p3/m, z8.h
.inst 0x05648d31  // revb z17.h, p3/m, z9.h
.inst 0x05648d52  // revb z18.h, p3/m, z10.h
.inst 0x05648d73  // revb z19.h, p3/m, z11.h

.inst 0xc12dd195  // uzp {z20.b, z21.b}, z12.b, z13.b
.inst 0xc13cd1d7  // uzp {z22.b, z23.b}, z14.b, z28.b
.inst 0xc131d219  // uzp {z24.b, z25.b}, z16.b, z17.b
.inst 0xc133d25b  // uzp {z26.b, z27.b}, z18.b, z19.b

.inst 0x04078d95  // uqshl z21.b, p3/m, z21.b, #4
.inst 0x04078d97  // uqshl z23.b, p3/m, z23.b, #4
.inst 0x04078d99  // uqshl z25.b, p3/m, z25.b, #4
.inst 0x04078d9b  // uqshl z27.b, p3/m, z27.b, #4

.inst 0x04180e95  // orr z21.b, p3/m, z21.b, z20.b
.inst 0x04180ed7  // orr z23.b, p3/m, z23.b, z22.b
.inst 0x04180f19  // orr z25.b, p3/m, z25.b, z24.b
.inst 0x04180f5b  // orr z27.b, p3/m, z27.b, z26.b

sub x6, x6, #4

.inst 0xe400ec15  // st1b {z21.b}, p3, [x0]
.inst 0xe401ec17  // st1b {z23.b}, p3, [x0, #1, MUL VL]
.inst 0xe402ec19  // st1b {z25.b}, p3, [x0, #2, MUL VL]
.inst 0xe403ec1b  // st1b {z27.b}, p3, [x0, #3, MUL VL]
.inst 0x04205080  // addvl x0, x0, #4

cmp x6, #4
bge LoopLU4

.inst 0xc0060c00  // mova {z0.s-z3.s}, za.s[w8, 0, VGx4]
.inst 0xc0062c04  // mova {z4.s-z7.s}, za.s[w9, 0, VGx4]
.inst 0x04a20000  // add z0.s, z0.s, z2.s
.inst 0x04a30021  // add z1.s, z1.s, z3.s
.inst 0x04a60084  // add z4.s, z4.s, z6.s
.inst 0x04a700a5  // add z5.s, z5.s, z7.s
.inst 0x04a4001e  // add z30.s, z0.s, z4.s
.inst 0x04a5003f  // add z31.s, z1.s, z5.s
cbz x6, LUEnd

LoopLU:

.inst 0xa400ac20  // ld1b {z0.b}, p3/z, [x1]
.inst 0x04215021  // addvl x1, x1, #1

.inst 0xc08a4004  // luti4 {z4.b-z5.b}, zt0, z0[0]

.inst 0xc15f50b0  // udot za.s[w10, 0, VGx2], {z4.b-z5.b}, z15.b[0]

.inst 0x05648c8c  // revb z12.h, p3/m, z4.h
.inst 0x05648cad  // revb z13.h, p3/m, z5.h

.inst 0xc12dd195  // uzp {z20.b, z21.b}, z12.b, z13.b

.inst 0x04078d95  // uqshl z21.b, p3/m, z21.b, #4

.inst 0x04180e95  // orr z21.b, p3/m, z21.b, z20.b

subs x6, x6, #1
.inst 0xe400ec15  // st1b {z21.b}, p3, [x0]
.inst 0x04205020  // addvl x0, x0, #1
bne LoopLU

.inst 0xc0064808  // mova {z8.s-z9.s}, za.s[w10, 0, VGx2]
.inst 0x04a803de  // add z30.s, z30.s, z8.s
.inst 0x04a903ff  // add z31.s, z31.s, z9.s

LUEnd:
.inst 0xc122e3de  // scvtf {z30.s-z31.s}, {z30.s-z31.s}


.inst 0xa060009e  // st1b {z30.b-z31.b}, pn8, [x4]
.inst 0x04245044  // addvl x4, x4, #2

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
