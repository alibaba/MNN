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

.macro Int32_To_Float32 s0, s1, s2, s3
    scvtf \s0\().4s, \s0\().4s
    scvtf \s1\().4s, \s1\().4s
    scvtf \s2\().4s, \s2\().4s
    scvtf \s3\().4s, \s3\().4s
.endm

asm_function MNNPermuteSumWeightInt4Arm86
// void MNNPermuteSumWeightInt4Arm86(uint8_t* dest, uint8_t* source, size_t outside, size_t inside, float* kernelSum);
// auto load: x0: dest, x1: source, x2: outside, x3: inside, x4: kernelSum

// inside = lu
// outside = blocknum*hu


stp d14, d15, [sp, #-64]!
stp d12, d13, [sp, #16]
stp d10, d11, [sp, #32]
stp d8,  d9,  [sp, #48]

movi v31.16b, #15
movi v30.16b, #4
movi v29.16b, #1

Loop: // blocknum*hu
mov x6, x3     // lu

SET_0 v4, v5, v6, v7
SET_0 v16, v17, v18, v19
cmp x6, #2
blt LoopLU

LoopLU2:
ld1 {v0.16b, v1.16b, v2.16b, v3.16b}, [x1], #64
ushr v8.16b, v0.16b, #4
and v9.16b, v0.16b, v31.16b
ushr v10.16b, v1.16b, #4
and v11.16b, v1.16b, v31.16b

ushr v27.16b, v2.16b, #4
and v28.16b, v2.16b, v31.16b
ushr v20.16b, v3.16b, #4
and v21.16b, v3.16b, v31.16b

// weight kernel sum
.inst 0x6e8897a4 // udot v4.4s, v29.16b, v8.16b
.inst 0x6e8997a5 // udot v5.4s, v29.16b, v9.16b
.inst 0x6e8a97a6 // udot v6.4s, v29.16b, v10.16b
.inst 0x6e8b97a7 // udot v7.4s, v29.16b, v11.16b

.inst 0x6e9b97b0 // udot v16.4s, v29.16b, v27.16b
.inst 0x6e9c97b1 // udot v17.4s, v29.16b, v28.16b
.inst 0x6e9497b2 // udot v18.4s, v29.16b, v20.16b
.inst 0x6e9597b3 // udot v19.4s, v29.16b, v21.16b

sub x6, x6, #2
// transpose
ushl v12.16b, v8.16b, v30.16b
ushl v13.16b, v9.16b, v30.16b
ushl v22.16b, v27.16b, v30.16b
ushl v23.16b, v28.16b, v30.16b

orr v14.16b, v12.16b, v10.16b
orr v15.16b, v13.16b, v11.16b
orr v24.16b, v22.16b, v20.16b
orr v25.16b, v23.16b, v21.16b

zip1 v0.16b, v14.16b, v15.16b
zip2 v1.16b, v14.16b, v15.16b
zip1 v2.16b, v24.16b, v25.16b
zip2 v3.16b, v24.16b, v25.16b
st1 {v0.16b, v1.16b, v2.16b, v3.16b}, [x0], #64

cmp x6, #2
bge LoopLU2
cbz x6, LUEnd

LoopLU:
cbz x6, LUEnd
ld1 {v0.16b, v1.16b}, [x1], #32
ushr v8.16b, v0.16b, #4
and v9.16b, v0.16b, v31.16b
ushr v10.16b, v1.16b, #4
and v11.16b, v1.16b, v31.16b

// weight kernel sum
.inst 0x6e8897a4 // udot v4.4s, v29.16b, v8.16b
.inst 0x6e8997a5 // udot v5.4s, v29.16b, v9.16b
.inst 0x6e8a97a6 // udot v6.4s, v29.16b, v10.16b
.inst 0x6e8b97a7 // udot v7.4s, v29.16b, v11.16b

// transpose
ushl v12.16b, v8.16b, v30.16b
ushl v13.16b, v9.16b, v30.16b
orr v14.16b, v12.16b, v10.16b
orr v15.16b, v13.16b, v11.16b
zip1 v0.16b, v14.16b, v15.16b
zip2 v1.16b, v14.16b, v15.16b
st1 {v0.16b, v1.16b}, [x0], #32

LUEnd:
add v8.4s, v4.4s, v5.4s
add v9.4s, v6.4s, v7.4s
add v10.4s, v16.4s, v17.4s
add v11.4s, v18.4s, v19.4s

add v8.4s, v8.4s, v10.4s
add v9.4s, v9.4s, v11.4s
scvtf v8.4s, v8.4s
scvtf v9.4s, v9.4s
st1 {v8.4s, v9.4s}, [x4], #32


subs x2, x2, #1 // outside--
bne Loop


End:
    ldp d8,  d9,  [sp, #48]
    ldp d10, d11, [sp, #32]
    ldp d12, d13, [sp, #16]
    ldp d14, d15, [sp], #64
    ret

#endif
