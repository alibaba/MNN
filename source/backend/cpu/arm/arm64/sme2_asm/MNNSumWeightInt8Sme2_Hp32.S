#ifdef __aarch64__
#include "MNNAsmGlobal.h"

.text
.align 5


asm_function MNNSumWeightInt8Sme2_Hp32
// void MNNSumWeightInt8Sme2_Hp32(float* kernelsum, int8_t* source, size_t outside, size_t reduceAxis, size_t hP, size_t lP)
// auto load: x0: dest, x1: source, x2: outside, x3: reduceAxis, x4: hP, x5: lP

// weight shape: [outside, reduceAxis, hP, lP]
// outside    = blocknum * hU
// reduceAxis = kernelCount * lU


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


Loop: // blocknum*hu
mov x6, x3     // lu

.inst 0xc00800ff  // zero {za}
.inst 0x2538c01e  // mov z30.b, #0
.inst 0x2538c01f  // mov z31.b, #0
cmp x6, #4
blt LoopLU

LoopLU4:
.inst 0xa0408024  // ld1b {z4.b-z7.b}, pn8/z, [x1]
.inst 0xa0418028  // ld1b {z8.b-z11.b}, pn8/z, [x1, #4, MUL VL]
.inst 0x04215101  // addvl x1, x1, #8
sub x6, x6, #4
.inst 0xc15f90a0  // sdot za.s[w8, 0, VGx4], {z4.b-z7.b}, z15.b[0]
.inst 0xc15fb120  // sdot za.s[w9, 0, VGx4], {z8.b-z11.b}, z15.b[0]

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

.inst 0xa0400024  // ld1b {z4.b-z5.b}, pn8/z, [x1]
.inst 0x04215041  // addvl x1, x1, #2
.inst 0xc15f50a0  // sdot za.s[w10, 0, VGx2], {z4.b-z5.b}, z15.b[0]
subs x6, x6, #1
bne LoopLU

.inst 0xc0064808  // mova {z8.s-z9.s}, za.s[w10, 0, VGx2]
.inst 0x04a803de  // add z30.s, z30.s, z8.s
.inst 0x04a903ff  // add z31.s, z31.s, z9.s

LUEnd:
.inst 0xc122e3de  // scvtf {z30.s-z31.s}, {z30.s-z31.s}
.inst 0xa060001e  // st1b {z30.b-z31.b}, pn8, [x0]
.inst 0x04205040  // addvl x0, x0, #2

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
