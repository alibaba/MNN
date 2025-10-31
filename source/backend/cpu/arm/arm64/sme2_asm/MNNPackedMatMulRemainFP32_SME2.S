//
//  MNNPackedMatMulRemainFP32_SME2.S
//  MNN
//
//  Created by MNN on 2020/06/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5


asm_function MNNPackedMatMulRemainFP32_SME2
//void MNNPackedMatMulRemainFP32_SME2(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter, const float* postParameters, const float* bias, const float* k, const float* b);
//Auto x0: C, x1:A, x2:B, x3:eSize, x4:parameter, x5:postParameters, x6:bias
// parameter: {aStride, l, h, cStride, bExtraStride}
stp d14, d15, [sp, #(-16 * 6)]!
stp d12, d13, [sp, #(16 * 1)]
stp d10, d11, [sp, #(16 * 2)]
stp d8,  d9,  [sp, #(16 * 3)]
stp x21, x22, [sp, #(16 * 4)]
stp x19, x20, [sp, #(16 * 5)]
.inst 0xd503477f  // smstart


// EP=16 LP=1 HP=64
ldr x22, [x4, #0] // aStride
ldr x9, [x4, #8] // l
ldr x10, [x4, #16] // h

ldr x7, [x4, #24] // cStride
ldr x19, [x4, #40] // bExtraStride

lsr x7, x7, #2 // cStride/sizeof(float)

mov w12, #0
mov w13, #4
mov w14, #8
mov w15, #12

// x10: ocDiv4
add x10, x10, #3
lsr x10, x10, #2
lsl x20, x3, #2 // x20: eSize * pack

// predicates
.inst 0x2598e3e0  // ptrue p0.s
.inst 0x25a317e1  // whilelt p1.s, xzr, x3
.inst 0x25a07810  // ptrue pn8.s
.inst 0x25b467f2  // whilelt pn10.s, xzr, x20, vlx4 // eSize * pack valid


// Relu parameters
cbz x5, ESIZE
.inst 0x8542c0be  // ld1rw {z30.s}, p0/z, [x5, #8]
.inst 0x8543c0bf  // ld1rw {z31.s}, p0/z, [x5, #12]

ESIZE: // x3 <= eP

cmp x3, #16
blt LoopOcDiv4

mov x22, #64 // aStride = 64 if eSize==16


LoopOcDiv4:
mov x8, x1 // A
mov x21, x9 // LU

.inst 0xc00800ff  // zero {za}

cbz x6, LoopL
// add bias
lsl x4, x10, #2
.inst 0x25a467f1  // whilelt pn9.s, xzr, x4, vlx4
.inst 0xa040c4d4  // ld1w {z20.s-z23.s}, pn9/z, [x6]
.inst 0x04265086  // addvl x6, x6, #4
.inst 0x25b9ce05  // fmov z5.s, #1

.inst 0x809400a0  // fmopa za0.s, p0/m, p0/m, z5.s, z20.s
.inst 0x809500a1  // fmopa za1.s, p0/m, p0/m, z5.s, z21.s
.inst 0x809600a2  // fmopa za2.s, p0/m, p0/m, z5.s, z22.s
.inst 0x809700a3  // fmopa za3.s, p0/m, p0/m, z5.s, z23.s

LoopL:
.inst 0xa540a504  // ld1w {z4.s}, p1/z, [x8]  // A
.inst 0xa040c040  // ld1w {z0.s-z3.s}, pn8/z, [x2]  // B
// [EP,LP] x [HP,LP] -> [EP,HP]
.inst 0x80800080  // fmopa za0.s, p0/m, p0/m, z4.s, z0.s
.inst 0x80810081  // fmopa za1.s, p0/m, p0/m, z4.s, z1.s
.inst 0x80820082  // fmopa za2.s, p0/m, p0/m, z4.s, z2.s
.inst 0x80830083  // fmopa za3.s, p0/m, p0/m, z4.s, z3.s

subs x21, x21, #1
// addvl x8, x8, #1
add x8, x8, x22
.inst 0x04225082  // addvl x2, x2, #4
bne LoopL

add x2, x2, x19 // bExtraStride

// 1. Extract oc=0...15 from za0
.inst 0xc0868400  // mova {z0.s-z3.s}, za0v.s[w12, 0:3]
.inst 0xc086a404  // mova {z4.s-z7.s}, za0v.s[w13, 0:3]
.inst 0xc086c408  // mova {z8.s-z11.s}, za0v.s[w14, 0:3]
.inst 0xc086e40c  // mova {z12.s-z15.s}, za0v.s[w15, 0:3]

.inst 0xc1b6e010  // zip {z16.s-z19.s}, {z0.s-z3.s}
.inst 0xc1b6e094  // zip {z20.s-z23.s}, {z4.s-z7.s}
.inst 0xc1b6e118  // zip {z24.s-z27.s}, {z8.s-z11.s}
.inst 0xc1b6e180  // zip {z0.s-z3.s}, {z12.s-z15.s}

cbz x5, StoreOc_0_15
.inst 0xc1bfcbd0  // fclamp {z16.s-z19.s}, z30.s, z31.s
.inst 0xc1bfcbd4  // fclamp {z20.s-z23.s}, z30.s, z31.s
.inst 0xc1bfcbd8  // fclamp {z24.s-z27.s}, z30.s, z31.s
.inst 0xc1bfcbc0  // fclamp {z0.s-z3.s}, z30.s, z31.s

StoreOc_0_15:

cmp x10, #4
bge StoreOc16

cmp x10, #3
beq StoreOc12

cmp x10, #2
beq StoreOc8

cmp x10, #1
beq StoreOc4

StoreOc4:
.inst 0xa060c810  // st1w {z16.s-z19.s}, pn10, [x0]
b End

StoreOc8:
.inst 0xa060c810  // st1w {z16.s-z19.s}, pn10, [x0]
.inst 0xa027c814  // st1w {z20.s-z23.s}, pn10, [x0, x7, lsl #2]
b End

StoreOc12:
add x21, x0, x7, LSL #3
.inst 0xa060c810  // st1w {z16.s-z19.s}, pn10, [x0]
.inst 0xa027c814  // st1w {z20.s-z23.s}, pn10, [x0, x7, lsl #2]
.inst 0xa060cab8  // st1w {z24.s-z27.s}, pn10, [x21]
b End

StoreOc16:
add x21, x0, x7, LSL #3
.inst 0xa060c810  // st1w {z16.s-z19.s}, pn10, [x0]
.inst 0xa027c814  // st1w {z20.s-z23.s}, pn10, [x0, x7, lsl #2]
.inst 0xa060cab8  // st1w {z24.s-z27.s}, pn10, [x21]
.inst 0xa027caa0  // st1w {z0.s-z3.s}, pn10, [x21, x7, lsl #2]
sub x10, x10, #4
add x0, x0, x7, LSL #4
cmp x10, #0
ble End

// 2. Extract oc=16...31 from za1
.inst 0xc0868420  // mova {z0.s-z3.s}, za1v.s[w12, 0:3]
.inst 0xc086a424  // mova {z4.s-z7.s}, za1v.s[w13, 0:3]
.inst 0xc086c428  // mova {z8.s-z11.s}, za1v.s[w14, 0:3]
.inst 0xc086e42c  // mova {z12.s-z15.s}, za1v.s[w15, 0:3]

.inst 0xc1b6e010  // zip {z16.s-z19.s}, {z0.s-z3.s}
.inst 0xc1b6e094  // zip {z20.s-z23.s}, {z4.s-z7.s}
.inst 0xc1b6e118  // zip {z24.s-z27.s}, {z8.s-z11.s}
.inst 0xc1b6e180  // zip {z0.s-z3.s}, {z12.s-z15.s}

cbz x5, StoreOc_16_31
.inst 0xc1bfcbd0  // fclamp {z16.s-z19.s}, z30.s, z31.s
.inst 0xc1bfcbd4  // fclamp {z20.s-z23.s}, z30.s, z31.s
.inst 0xc1bfcbd8  // fclamp {z24.s-z27.s}, z30.s, z31.s
.inst 0xc1bfcbc0  // fclamp {z0.s-z3.s}, z30.s, z31.s

StoreOc_16_31:

cmp x10, #4
bge StoreOc32

cmp x10, #3
beq StoreOc28

cmp x10, #2
beq StoreOc24

cmp x10, #1
beq StoreOc20

StoreOc20:
.inst 0xa060c810  // st1w {z16.s-z19.s}, pn10, [x0]
b End

StoreOc24:
.inst 0xa060c810  // st1w {z16.s-z19.s}, pn10, [x0]
.inst 0xa027c814  // st1w {z20.s-z23.s}, pn10, [x0, x7, lsl #2]
b End

StoreOc28:
add x21, x0, x7, LSL #3
.inst 0xa060c810  // st1w {z16.s-z19.s}, pn10, [x0]
.inst 0xa027c814  // st1w {z20.s-z23.s}, pn10, [x0, x7, lsl #2]
.inst 0xa060cab8  // st1w {z24.s-z27.s}, pn10, [x21]
b End

StoreOc32:
add x21, x0, x7, LSL #3
.inst 0xa060c810  // st1w {z16.s-z19.s}, pn10, [x0]
.inst 0xa027c814  // st1w {z20.s-z23.s}, pn10, [x0, x7, lsl #2]
.inst 0xa060cab8  // st1w {z24.s-z27.s}, pn10, [x21]
.inst 0xa027caa0  // st1w {z0.s-z3.s}, pn10, [x21, x7, lsl #2]
sub x10, x10, #4
add x0, x0, x7, LSL #4
cmp x10, #0
ble End

// 3. Extract oc=32...47 from za2
.inst 0xc0868440  // mova {z0.s-z3.s}, za2v.s[w12, 0:3]
.inst 0xc086a444  // mova {z4.s-z7.s}, za2v.s[w13, 0:3]
.inst 0xc086c448  // mova {z8.s-z11.s}, za2v.s[w14, 0:3]
.inst 0xc086e44c  // mova {z12.s-z15.s}, za2v.s[w15, 0:3]

.inst 0xc1b6e010  // zip {z16.s-z19.s}, {z0.s-z3.s}
.inst 0xc1b6e094  // zip {z20.s-z23.s}, {z4.s-z7.s}
.inst 0xc1b6e118  // zip {z24.s-z27.s}, {z8.s-z11.s}
.inst 0xc1b6e180  // zip {z0.s-z3.s}, {z12.s-z15.s}

cbz x5, StoreOc_32_47
.inst 0xc1bfcbd0  // fclamp {z16.s-z19.s}, z30.s, z31.s
.inst 0xc1bfcbd4  // fclamp {z20.s-z23.s}, z30.s, z31.s
.inst 0xc1bfcbd8  // fclamp {z24.s-z27.s}, z30.s, z31.s
.inst 0xc1bfcbc0  // fclamp {z0.s-z3.s}, z30.s, z31.s

StoreOc_32_47:

cmp x10, #4
bge StoreOc48

cmp x10, #3
beq StoreOc44

cmp x10, #2
beq StoreOc40

cmp x10, #1
beq StoreOc36

StoreOc36:
.inst 0xa060c810  // st1w {z16.s-z19.s}, pn10, [x0]
b End

StoreOc40:
.inst 0xa060c810  // st1w {z16.s-z19.s}, pn10, [x0]
.inst 0xa027c814  // st1w {z20.s-z23.s}, pn10, [x0, x7, lsl #2]
b End

StoreOc44:
add x21, x0, x7, LSL #3
.inst 0xa060c810  // st1w {z16.s-z19.s}, pn10, [x0]
.inst 0xa027c814  // st1w {z20.s-z23.s}, pn10, [x0, x7, lsl #2]
.inst 0xa060cab8  // st1w {z24.s-z27.s}, pn10, [x21]
b End

StoreOc48:
add x21, x0, x7, LSL #3
.inst 0xa060c810  // st1w {z16.s-z19.s}, pn10, [x0]
.inst 0xa027c814  // st1w {z20.s-z23.s}, pn10, [x0, x7, lsl #2]
.inst 0xa060cab8  // st1w {z24.s-z27.s}, pn10, [x21]
.inst 0xa027caa0  // st1w {z0.s-z3.s}, pn10, [x21, x7, lsl #2]
sub x10, x10, #4
add x0, x0, x7, LSL #4
cmp x10, #0
ble End


// 4. Extract oc=48...63 from za3
.inst 0xc0868460  // mova {z0.s-z3.s}, za3v.s[w12, 0:3]
.inst 0xc086a464  // mova {z4.s-z7.s}, za3v.s[w13, 0:3]
.inst 0xc086c468  // mova {z8.s-z11.s}, za3v.s[w14, 0:3]
.inst 0xc086e46c  // mova {z12.s-z15.s}, za3v.s[w15, 0:3]

.inst 0xc1b6e010  // zip {z16.s-z19.s}, {z0.s-z3.s}
.inst 0xc1b6e094  // zip {z20.s-z23.s}, {z4.s-z7.s}
.inst 0xc1b6e118  // zip {z24.s-z27.s}, {z8.s-z11.s}
.inst 0xc1b6e180  // zip {z0.s-z3.s}, {z12.s-z15.s}

cbz x5, StoreOc_48_63
.inst 0xc1bfcbd0  // fclamp {z16.s-z19.s}, z30.s, z31.s
.inst 0xc1bfcbd4  // fclamp {z20.s-z23.s}, z30.s, z31.s
.inst 0xc1bfcbd8  // fclamp {z24.s-z27.s}, z30.s, z31.s
.inst 0xc1bfcbc0  // fclamp {z0.s-z3.s}, z30.s, z31.s

StoreOc_48_63:

cmp x10, #4
bge StoreOc64

cmp x10, #3
beq StoreOc60

cmp x10, #2
beq StoreOc56

cmp x10, #1
beq StoreOc52

StoreOc52:
.inst 0xa060c810  // st1w {z16.s-z19.s}, pn10, [x0]
b End

StoreOc56:
.inst 0xa060c810  // st1w {z16.s-z19.s}, pn10, [x0]
.inst 0xa027c814  // st1w {z20.s-z23.s}, pn10, [x0, x7, lsl #2]
b End

StoreOc60:
add x21, x0, x7, LSL #3
.inst 0xa060c810  // st1w {z16.s-z19.s}, pn10, [x0]
.inst 0xa027c814  // st1w {z20.s-z23.s}, pn10, [x0, x7, lsl #2]
.inst 0xa060cab8  // st1w {z24.s-z27.s}, pn10, [x21]
b End

StoreOc64:
add x21, x0, x7, LSL #3
.inst 0xa060c810  // st1w {z16.s-z19.s}, pn10, [x0]
.inst 0xa027c814  // st1w {z20.s-z23.s}, pn10, [x0, x7, lsl #2]
.inst 0xa060cab8  // st1w {z24.s-z27.s}, pn10, [x21]
.inst 0xa027caa0  // st1w {z0.s-z3.s}, pn10, [x21, x7, lsl #2]
sub x10, x10, #4
add x0, x0, x7, LSL #4
cmp x10, #0
ble End

b LoopOcDiv4


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
