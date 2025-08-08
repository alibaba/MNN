#ifdef __aarch64__
#include "MNNAsmGlobal.h"

.text
.align 5


asm_function MNNSumWeightInt8Sme2_Hp128
// void MNNSumWeightInt8Sme2_Hp128(float* kernelsum, int8_t* source, size_t outside, size_t reduceAxis, size_t hP, size_t lP)
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
mov w11, #12
.inst 0x2538c02f  // mov z15.b, #1


Loop: // blocknum*hu
mov x6, x3     // lu

.inst 0xc00800ff  // zero {za}
cmp x6, #2
blt LoopLU


LoopLU2:
.inst 0xa0408020  // ld1b {z0.b-z3.b}, pn8/z, [x1]
.inst 0xa0418024  // ld1b {z4.b-z7.b}, pn8/z, [x1, #4, MUL VL]
.inst 0xa0428028  // ld1b {z8.b-z11.b}, pn8/z, [x1, #8, MUL VL]
.inst 0xa0438030  // ld1b {z16.b-z19.b}, pn8/z, [x1, #12, MUL VL]
.inst 0x04215201  // addvl x1, x1, #16
sub x6, x6, #2
.inst 0xc15f9020  // sdot za.s[w8, 0, VGx4], {z0.b-z3.b}, z15.b[0]
.inst 0xc15fb0a0  // sdot za.s[w9, 0, VGx4], {z4.b-z7.b}, z15.b[0]
.inst 0xc15fd120  // sdot za.s[w10, 0, VGx4], {z8.b-z11.b}, z15.b[0]
.inst 0xc15ff220  // sdot za.s[w11, 0, VGx4], {z16.b-z19.b}, z15.b[0]

cmp x6, #2
bge LoopLU2

cbz x6, LUEnd

LoopLU:

.inst 0xa0408020  // ld1b {z0.b-z3.b}, pn8/z, [x1]
.inst 0xa0418024  // ld1b {z4.b-z7.b}, pn8/z, [x1, #4, MUL VL]
.inst 0x04215101  // addvl x1, x1, #8
.inst 0xc15f9020  // sdot za.s[w8, 0, VGx4], {z0.b-z3.b}, z15.b[0]
.inst 0xc15fb0a0  // sdot za.s[w9, 0, VGx4], {z4.b-z7.b}, z15.b[0]
subs x6, x6, #1
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

.inst 0xa0608000  // st1b {z0.b-z3.b}, pn8, [x0]
.inst 0xa0618004  // st1b {z4.b-z7.b}, pn8, [x0, #4, MUL VL]
.inst 0x04205100  // addvl x0, x0, #8

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
