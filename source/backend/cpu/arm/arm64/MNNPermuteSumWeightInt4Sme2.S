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

asm_function MNNPermuteSumWeightInt4Sme2
// void MNNPermuteSumWeightInt4Sme2(uint8_t* dest, uint8_t* source, size_t outside, size_t inside, float* kernelSum);
// auto load: x0: dest, x1: source, x2: outside, x3: inside, x4: kernelSum

// inside = lu
// outside = blocknum*hu
// kernelSum shape: [hu, blockNum, hp]


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
SET_0 v20, v21, v22, v23
cmp x6, #2
blt LoopLU

LoopLU2:
ld1 {v0.16b, v1.16b, v2.16b, v3.16b}, [x1], #64
ushr v8.16b, v0.16b, #4 // v8: 0 2 ... 30
and v9.16b, v0.16b, v31.16b // v9: 1 3 ... 31
ushr v10.16b, v1.16b, #4
and v11.16b, v1.16b, v31.16b

ushr v27.16b, v2.16b, #4
and v28.16b, v2.16b, v31.16b
ushr v24.16b, v3.16b, #4
and v25.16b, v3.16b, v31.16b

zip1 v12.16b, v8.16b, v9.16b // v12: 0 1 2 3 ... 14 15
zip2 v13.16b, v8.16b, v9.16b // v13: 16 17 18 19 ... 30 31
zip1 v14.16b, v10.16b, v11.16b // v14: 32 33 34 35 ... 46 47
zip2 v15.16b, v10.16b, v11.16b // v15: 48 49 50 51 ... 62 63

zip1 v16.16b, v27.16b, v28.16b
zip2 v17.16b, v27.16b, v28.16b
zip1 v18.16b, v24.16b, v25.16b
zip2 v19.16b, v24.16b, v25.16b

// weight kernel sum
.inst 0x6e8c97a4 // udot v4.4s, v29.16b, v12.16b
.inst 0x6e8d97a5 // udot v5.4s, v29.16b, v13.16b
.inst 0x6e8e97a6 // udot v6.4s, v29.16b, v14.16b
.inst 0x6e8f97a7 // udot v7.4s, v29.16b, v15.16b

.inst 0x6e9097b4 // udot v20.4s, v29.16b, v16.16b
.inst 0x6e9197b5 // udot v21.4s, v29.16b, v17.16b
.inst 0x6e9297b6 // udot v22.4s, v29.16b, v18.16b
.inst 0x6e9397b7 // udot v23.4s, v29.16b, v19.16b

sub x6, x6, #2
// transpose
ushl v9.16b, v9.16b, v30.16b
ushl v11.16b, v11.16b, v30.16b
ushl v28.16b, v28.16b, v30.16b
ushl v25.16b, v25.16b, v30.16b

orr v0.16b, v8.16b, v9.16b
orr v1.16b, v10.16b, v11.16b
orr v2.16b, v27.16b, v28.16b
orr v3.16b, v24.16b, v25.16b

st1 {v0.16b, v1.16b, v2.16b, v3.16b}, [x0], #64

cmp x6, #2
bge LoopLU2
cbz x6, LUEnd

LoopLU:
cbz x6, LUEnd

ld1 {v0.16b, v1.16b}, [x1], #32
ushr v8.16b, v0.16b, #4 // v8: 0 2 ... 30
and v9.16b, v0.16b, v31.16b // v9: 1 3 ... 31
ushr v10.16b, v1.16b, #4
and v11.16b, v1.16b, v31.16b

zip1 v12.16b, v8.16b, v9.16b // v12: 0 1 2 3 ... 14 15
zip2 v13.16b, v8.16b, v9.16b // v13: 16 17 18 19 ... 30 31
zip1 v14.16b, v10.16b, v11.16b // v14: 32 33 34 35 ... 46 47
zip2 v15.16b, v10.16b, v11.16b // v15: 48 49 50 51 ... 62 63


// weight kernel sum
.inst 0x6e8c97a4 // udot v4.4s, v29.16b, v12.16b
.inst 0x6e8d97a5 // udot v5.4s, v29.16b, v13.16b
.inst 0x6e8e97a6 // udot v6.4s, v29.16b, v14.16b
.inst 0x6e8f97a7 // udot v7.4s, v29.16b, v15.16b

// <<4
ushl v9.16b, v9.16b, v30.16b
ushl v11.16b, v11.16b, v30.16b

orr v0.16b, v8.16b, v9.16b
orr v1.16b, v10.16b, v11.16b

st1 {v0.16b, v1.16b}, [x0], #32

LUEnd:

add v4.4s, v4.4s, v20.4s
add v5.4s, v5.4s, v21.4s
add v6.4s, v6.4s, v22.4s
add v7.4s, v7.4s, v23.4s
scvtf v4.4s, v4.4s
scvtf v5.4s, v5.4s
scvtf v6.4s, v6.4s
scvtf v7.4s, v7.4s
st1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x4], #64

subs x2, x2, #1 // outside--
bne Loop


End:
    ldp d8,  d9,  [sp, #48]
    ldp d10, d11, [sp, #32]
    ldp d12, d13, [sp, #16]
    ldp d14, d15, [sp], #64
    ret

#endif
