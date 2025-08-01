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

asm_function MNNSumWeightInt8Sme2
// void MNNSumWeightInt8Sme2(float* kernlesum, int8_t* source, size_t outside, size_t reduceAxis, size_t hP, size_t lP)
// auto load: x0: dest, x1: source, x2: outside, x3: reduceAxis, x4: hP, x5: lP

// weight shape: [outside, reduceAxis, hP, lP]
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
SET_0 v24, v25, v26, v27

LU3:
cmp x5, #3
blt LU2
ld1 {v0.16b, v1.16b, v2.16b, v3.16b}, [x1], #64
ld1 {v4.16b, v5.16b, v6.16b, v7.16b}, [x1], #64
ld1 {v8.16b, v9.16b, v10.16b, v11.16b}, [x1], #64

// kernel sum
.inst 0x4e8097f0 // sdot v16.4s, v31.16b, v0.16b
.inst 0x4e8197f1 // sdot v17.4s, v31.16b, v1.16b
.inst 0x4e8297f2 // sdot v18.4s, v31.16b, v2.16b
.inst 0x4e8397f3 // sdot v19.4s, v31.16b, v3.16b

.inst 0x4e8497f4 // sdot v20.4s, v31.16b, v4.16b
.inst 0x4e8597f5 // sdot v21.4s, v31.16b, v5.16b
.inst 0x4e8697f6 // sdot v22.4s, v31.16b, v6.16b
.inst 0x4e8797f7 // sdot v23.4s, v31.16b, v7.16b

.inst 0x4e8897f8 // sdot v24.4s, v31.16b, v8.16b
.inst 0x4e8997f9 // sdot v25.4s, v31.16b, v9.16b
.inst 0x4e8a97fa // sdot v26.4s, v31.16b, v10.16b
.inst 0x4e8b97fb // sdot v27.4s, v31.16b, v11.16b


subs x5, x5, #3
beq LUEnd
b LU3


LU2:
cmp x5, #2
blt LU1
ld1 {v0.16b, v1.16b, v2.16b, v3.16b}, [x1], #64
ld1 {v4.16b, v5.16b, v6.16b, v7.16b}, [x1], #64

// kernel sum
.inst 0x4e8097f0 // sdot v16.4s, v31.16b, v0.16b
.inst 0x4e8197f1 // sdot v17.4s, v31.16b, v1.16b
.inst 0x4e8297f2 // sdot v18.4s, v31.16b, v2.16b
.inst 0x4e8397f3 // sdot v19.4s, v31.16b, v3.16b

.inst 0x4e8497f4 // sdot v20.4s, v31.16b, v4.16b
.inst 0x4e8597f5 // sdot v21.4s, v31.16b, v5.16b
.inst 0x4e8697f6 // sdot v22.4s, v31.16b, v6.16b
.inst 0x4e8797f7 // sdot v23.4s, v31.16b, v7.16b

subs x5, x5, #2
beq LUEnd
b LU2

LU1: // outside
cbz x5, LUEnd
ld1 {v0.16b, v1.16b, v2.16b, v3.16b}, [x1], #64
// kernel sum
.inst 0x4e8097f0 // sdot v16.4s, v31.16b, v0.16b
.inst 0x4e8197f1 // sdot v17.4s, v31.16b, v1.16b
.inst 0x4e8297f2 // sdot v18.4s, v31.16b, v2.16b
.inst 0x4e8397f3 // sdot v19.4s, v31.16b, v3.16b

LUEnd:
add v16.4s, v16.4s, v20.4s
add v17.4s, v17.4s, v21.4s
add v18.4s, v18.4s, v22.4s
add v19.4s, v19.4s, v23.4s
add v16.4s, v16.4s, v24.4s
add v17.4s, v17.4s, v25.4s
add v18.4s, v18.4s, v26.4s
add v19.4s, v19.4s, v27.4s
scvtf v16.4s, v16.4s
scvtf v17.4s, v17.4s
scvtf v18.4s, v18.4s
scvtf v19.4s, v19.4s
st1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x0], #64

subs x2, x2, #1 // outside--
bne Loop


End:
    ldp d8,  d9,  [sp, #48]
    ldp d10, d11, [sp, #32]
    ldp d12, d13, [sp, #16]
    ldp d14, d15, [sp], #64
    ret

#endif
