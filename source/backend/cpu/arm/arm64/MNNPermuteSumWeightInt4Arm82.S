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

asm_function MNNPermuteSumWeightInt4Arm82
// void MNNPermuteSumWeightInt4Arm82(uint8_t* dest, uint8_t* source, size_t outside, size_t inside, float* kernelSum);
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

Loop:
mov x5, x3
SET_0 v16, v17, v18, v19

cmp x5, #4
blt LU2

LU4:
ld1 {v0.16b, v1.16b, v2.16b, v3.16b}, [x1], #64
//int4->int8
ushr v4.16b, v0.16b, #4
and v5.16b, v0.16b, v31.16b
ushr v6.16b, v1.16b, #4
and v7.16b, v1.16b, v31.16b

ushr v12.16b, v2.16b, #4
and v13.16b, v2.16b, v31.16b
ushr v14.16b, v3.16b, #4
and v15.16b, v3.16b, v31.16b

// [0 2 4 6] x [1 3 5 7] -> [0 1 2 3] x [4 5 6 7]
zip1 v8.16b, v4.16b, v5.16b
zip2 v9.16b, v4.16b, v5.16b
zip1 v10.16b, v6.16b, v7.16b
zip2 v11.16b, v6.16b, v7.16b

zip1 v20.16b, v12.16b, v13.16b
zip2 v21.16b, v12.16b, v13.16b
zip1 v22.16b, v14.16b, v15.16b
zip2 v23.16b, v14.16b, v15.16b

// kernel sum
.inst 0x6e8897b0 // udot v16.4s, v29.16b, v8.16b
.inst 0x6e8997b1 // udot v17.4s, v29.16b, v9.16b
.inst 0x6e8a97b2 // udot v18.4s, v29.16b, v10.16b
.inst 0x6e8b97b3 // udot v19.4s, v29.16b, v11.16b

.inst 0x6e9497b0 // udot v16.4s, v29.16b, v20.16b
.inst 0x6e9597b1 // udot v17.4s, v29.16b, v21.16b
.inst 0x6e9697b2 // udot v18.4s, v29.16b, v22.16b
.inst 0x6e9797b3 // udot v19.4s, v29.16b, v23.16b

sub x5, x5, #4
// int8->int4
ushl v24.16b, v8.16b, v30.16b
ushl v25.16b, v10.16b, v30.16b
orr v24.16b, v24.16b, v9.16b
orr v25.16b, v25.16b, v11.16b

ushl v26.16b, v20.16b, v30.16b
ushl v27.16b, v22.16b, v30.16b
orr v26.16b, v26.16b, v21.16b
orr v27.16b, v27.16b, v23.16b
st1 {v24.16b, v25.16b, v26.16b, v27.16b}, [x0], #64

cmp x5, #4
bge LU4
cbz x5, LUEnd

LU2:
ld1 {v0.16b, v1.16b}, [x1], #32
ushr v2.16b, v0.16b, #4
and v3.16b, v0.16b, v31.16b
ushr v8.16b, v1.16b, #4
and v9.16b, v1.16b, v31.16b

zip1 v4.16b, v2.16b, v3.16b
zip2 v5.16b, v2.16b, v3.16b
zip1 v10.16b, v8.16b, v9.16b
zip2 v11.16b, v8.16b, v9.16b

// kernel sum
.inst 0x6e8497b0 // udot v16.4s, v29.16b, v4.16b
.inst 0x6e8597b1 // udot v17.4s, v29.16b, v5.16b
.inst 0x6e8a97b2 // udot v18.4s, v29.16b, v10.16b
.inst 0x6e8b97b3 // udot v19.4s, v29.16b, v11.16b

sub x5, x5, #2
// transpose
ushl v6.16b, v4.16b, v30.16b
ushl v7.16b, v10.16b, v30.16b
orr v6.16b, v6.16b, v5.16b
orr v7.16b, v7.16b, v11.16b
st1 {v6.16b, v7.16b}, [x0], #32

cmp x5, #2
bge LU2
cbz x5, LUEnd

LU1: // outside
cbz x5, LUEnd
ld1 {v0.16b}, [x1], #16
ushr v2.16b, v0.16b, #4
and v3.16b, v0.16b, v31.16b
zip1 v4.16b, v2.16b, v3.16b
zip2 v5.16b, v2.16b, v3.16b
// kernel sum
.inst 0x6e8497b0 // udot v16.4s, v29.16b, v4.16b
.inst 0x6e8597b1 // udot v17.4s, v29.16b, v5.16b
// transpose
ushl v10.16b, v4.16b, v30.16b
orr v6.16b, v10.16b, v5.16b
st1 {v6.16b}, [x0], #16

LUEnd:
add v16.4s, v16.4s, v18.4s
add v17.4s, v17.4s, v19.4s
scvtf v16.4s, v16.4s
scvtf v17.4s, v17.4s
st1 {v16.4s, v17.4s}, [x4], #32

subs x2, x2, #1 // outside--
bne Loop


End:
    ldp d8,  d9,  [sp, #48]
    ldp d10, d11, [sp, #32]
    ldp d12, d13, [sp, #16]
    ldp d14, d15, [sp], #64
    ret

#endif
