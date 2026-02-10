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

asm_function MNNSumWeightInt8Arm86
// void MNNSumWeightInt8Arm86(float* kernelsum, int8_t* source, size_t outside, size_t reduceAxis, size_t hP, size_t lP)
// auto load: x0: dest, x1: source, x2: outside, x3: reduceAxis, x4: hP, x5: lP

// weight shape: [outside, axis, hP, lP]
// outside    = blocknum * hU
// reduceAxis = kernelCount * lU

stp d14, d15, [sp, #-64]!
stp d12, d13, [sp, #16]
stp d10, d11, [sp, #32]
stp d8,  d9,  [sp, #48]

movi v31.16b, #1

Loop:
mov x5, x3
SET_0 v16, v17, v18, v19
SET_0 v20, v21, v22, v23


LU4:
cmp x5, #4
blt LU2
ld1 {v0.16b, v1.16b, v2.16b, v3.16b}, [x1], #64
ld1 {v4.16b, v5.16b, v6.16b, v7.16b}, [x1], #64
ld1 {v8.16b, v9.16b, v10.16b, v11.16b}, [x1], #64
ld1 {v12.16b, v13.16b, v14.16b, v15.16b}, [x1], #64

// kernel sum
.inst 0x4e80a7f0 // smmla v16.4s, v31.16b, v0.16b
.inst 0x4e81a7f1 // smmla v17.4s, v31.16b, v1.16b
.inst 0x4e82a7f2 // smmla v18.4s, v31.16b, v2.16b
.inst 0x4e83a7f3 // smmla v19.4s, v31.16b, v3.16b

.inst 0x4e84a7f4 // smmla v20.4s, v31.16b, v4.16b
.inst 0x4e85a7f5 // smmla v21.4s, v31.16b, v5.16b
.inst 0x4e86a7f6 // smmla v22.4s, v31.16b, v6.16b
.inst 0x4e87a7f7 // smmla v23.4s, v31.16b, v7.16b

.inst 0x4e88a7f0 // smmla v16.4s, v31.16b, v8.16b
.inst 0x4e89a7f1 // smmla v17.4s, v31.16b, v9.16b
.inst 0x4e8aa7f2 // smmla v18.4s, v31.16b, v10.16b
.inst 0x4e8ba7f3 // smmla v19.4s, v31.16b, v11.16b

.inst 0x4e8ca7f4 // smmla v20.4s, v31.16b, v12.16b
.inst 0x4e8da7f5 // smmla v21.4s, v31.16b, v13.16b
.inst 0x4e8ea7f6 // smmla v22.4s, v31.16b, v14.16b
.inst 0x4e8fa7f7 // smmla v23.4s, v31.16b, v15.16b

sub x5, x5, #4
cmp x5, #4
bge LU4
cbz x5, LUEnd
b LU4


LU2:
cmp x5, #2
blt LU1
ld1 {v0.16b, v1.16b, v2.16b, v3.16b}, [x1], #64
ld1 {v4.16b, v5.16b, v6.16b, v7.16b}, [x1], #64

// kernel sum
.inst 0x4e80a7f0 // smmla v16.4s, v31.16b, v0.16b
.inst 0x4e81a7f1 // smmla v17.4s, v31.16b, v1.16b
.inst 0x4e82a7f2 // smmla v18.4s, v31.16b, v2.16b
.inst 0x4e83a7f3 // smmla v19.4s, v31.16b, v3.16b

.inst 0x4e84a7f4 // smmla v20.4s, v31.16b, v4.16b
.inst 0x4e85a7f5 // smmla v21.4s, v31.16b, v5.16b
.inst 0x4e86a7f6 // smmla v22.4s, v31.16b, v6.16b
.inst 0x4e87a7f7 // smmla v23.4s, v31.16b, v7.16b

sub x5, x5, #2
cbz x5, LUEnd
b LU2

LU1: // outside
cbz x5, LUEnd
ld1 {v0.16b, v1.16b, v2.16b, v3.16b}, [x1], #64
.inst 0x4e80a7f0 // smmla v16.4s, v31.16b, v0.16b
.inst 0x4e81a7f1 // smmla v17.4s, v31.16b, v1.16b
.inst 0x4e82a7f2 // smmla v18.4s, v31.16b, v2.16b
.inst 0x4e83a7f3 // smmla v19.4s, v31.16b, v3.16b

LUEnd:
add v16.4s, v16.4s, v20.4s
add v17.4s, v17.4s, v21.4s
add v18.4s, v18.4s, v22.4s
add v19.4s, v19.4s, v23.4s
zip1 v0.2d, v16.2d, v17.2d
zip1 v1.2d, v18.2d, v19.2d
scvtf v0.4s, v0.4s
scvtf v1.4s, v1.4s
st1 {v0.4s, v1.4s}, [x0], #32

subs x2, x2, #1 // outside--
bne Loop


End:
    ldp d8,  d9,  [sp, #48]
    ldp d10, d11, [sp, #32]
    ldp d12, d13, [sp, #16]
    ldp d14, d15, [sp], #64
    ret

#endif
