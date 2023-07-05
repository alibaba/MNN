//  MNNBilinearLineC8.S
//  MNN
//
//  Created by MNN on 2019/01/18.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"
.text
.align 5
asm_function MNNBilinearLineC8
// void MNNBilinearLineC8(int8_t* dst, const int16_t* A, const int16_t* B, const float* t, int8_t* zeroPoint, size_t number)
// Auto load:
// x0: dst, x1: src0, x2: src1, x3: factor, x4: zeroPoint, x5: number

stp d14, d15, [sp, #-64]!
stp d12, d13, [sp, #16]
stp d10, d11, [sp, #32]
stp d8,  d9,  [sp, #48]

cmp x5, #0
beq END

ld1 {v31.s}[0], [x3]
dup v31.4s, v31.s[0]     // v31: df
fmov s30, #1.0     // v30: sf=1-df
fsub s30, s30, s31
movi v1.4s, #128   // s1=128
fmul s31, s31, s1
fmul s30, s30, s1
dup v31.8h, v31.h[0]
dup v30.8h, v30.h[0]

cmp x5, #0
beq END
cmp x5, #2
blt L1Loop
cmp x5, #4
blt L2Loop
cmp x5, #8
blt L4Loop

L8Loop:

ld1 {v0.8h, v1.8h, v2.8h, v3.8h}, [x1], #64
ld1 {v4.8h, v5.8h, v6.8h, v7.8h}, [x2], #64


smull v8.4s, v0.4h, v30.4h
smull2 v9.4s, v0.8h, v30.8h
smlal  v8.4s, v4.4h, v31.4h
smlal2 v9.4s, v4.8h, v31.8h

smull  v10.4s, v1.4h, v30.4h
smull2 v11.4s, v1.8h, v30.8h
smlal  v10.4s, v5.4h, v31.4h
smlal2 v11.4s, v5.8h, v31.8h

smull  v12.4s, v2.4h, v30.4h
smull2 v13.4s, v2.8h, v30.8h
smlal  v12.4s, v6.4h, v31.4h
smlal2 v13.4s, v6.8h, v31.8h

smull  v14.4s, v3.4h, v30.4h
smull2 v15.4s, v3.8h, v30.8h
smlal  v14.4s, v7.4h, v31.4h
smlal2 v15.4s, v7.8h, v31.8h

///
ld1 {v16.8h, v17.8h, v18.8h, v19.8h}, [x1], #64
ld1 {v20.8h, v21.8h, v22.8h, v23.8h}, [x2], #64


smull  v24.4s, v16.4h, v30.4h
smull2 v25.4s, v16.8h, v30.8h
smlal  v24.4s, v20.4h, v31.4h
smlal2 v25.4s, v20.8h, v31.8h

smull  v26.4s, v17.4h, v30.4h
smull2 v27.4s, v17.8h, v30.8h
smlal  v26.4s, v21.4h, v31.4h
smlal2 v27.4s, v21.8h, v31.8h

smull  v28.4s, v18.4h, v30.4h
smull2 v29.4s, v18.8h, v30.8h
smlal  v28.4s, v22.4h, v31.4h
smlal2 v29.4s, v22.8h, v31.8h

smull  v0.4s, v19.4h, v30.4h
smull2 v1.4s, v19.8h, v30.8h
smlal  v0.4s, v23.4h, v31.4h
smlal2 v1.4s, v23.8h, v31.8h

shrn  v8.4h, v8.4s, #14
shrn2 v8.8h, v9.4s, #14

shrn  v10.4h, v10.4s, #14
shrn2 v10.8h, v11.4s, #14

shrn v12.4h,  v12.4s, #14
shrn2 v12.8h, v13.4s, #14

shrn  v14.4h, v14.4s, #14
shrn2 v14.8h, v15.4s, #14
////
shrn  v24.4h, v24.4s, #14
shrn2 v24.8h, v25.4s, #14

shrn  v26.4h, v26.4s, #14
shrn2 v26.8h, v27.4s, #14

shrn  v28.4h, v28.4s, #14
shrn2 v28.8h, v29.4s, #14

shrn  v0.4h, v0.4s, #14
shrn2 v0.8h, v1.4s, #14

sqxtn v2.8b, v8.8h
sqxtn2 v2.16b, v10.8h
sqxtn v3.8b, v12.8h
sqxtn2 v3.16b, v14.8h

sqxtn v4.8b, v24.8h
sqxtn2 v4.16b, v26.8h
sqxtn v5.8b, v28.8h
sqxtn2 v5.16b, v0.8h

st1 {v2.16b, v3.16b, v4.16b, v5.16b}, [x0], #64

sub x5, x5, #8
cmp x5, #8
bge L8Loop
cmp x5, #0
beq END
cmp x5, #2
blt L1Loop
cmp x5, #4
blt L2Loop

L4Loop:

ld1 {v0.8h, v1.8h, v2.8h, v3.8h}, [x1], #64
ld1 {v4.8h, v5.8h, v6.8h, v7.8h}, [x2], #64

smull v8.4s, v0.4h, v30.4h
smull2 v9.4s, v0.8h, v30.8h
smlal  v8.4s, v4.4h, v31.4h
smlal2 v9.4s, v4.8h, v31.8h

smull  v10.4s, v1.4h, v30.4h
smull2 v11.4s, v1.8h, v30.8h
smlal  v10.4s, v5.4h, v31.4h
smlal2 v11.4s, v5.8h, v31.8h

smull  v12.4s, v2.4h, v30.4h
smull2 v13.4s, v2.8h, v30.8h
smlal  v12.4s, v6.4h, v31.4h
smlal2 v13.4s, v6.8h, v31.8h

smull  v14.4s, v3.4h, v30.4h
smull2 v15.4s, v3.8h, v30.8h
smlal  v14.4s, v7.4h, v31.4h
smlal2 v15.4s, v7.8h, v31.8h

shrn  v8.4h, v8.4s, #14
shrn2 v8.8h, v9.4s, #14

shrn  v10.4h, v10.4s, #14
shrn2 v10.8h, v11.4s, #14

shrn v12.4h,  v12.4s, #14
shrn2 v12.8h, v13.4s, #14

shrn  v14.4h, v14.4s, #14
shrn2 v14.8h, v15.4s, #14

sqxtn v8.8b, v8.8h
sqxtn2 v8.16b, v10.8h
sqxtn v9.8b, v12.8h
sqxtn2 v9.16b, v14.8h

st1 {v8.16b, v9.16b}, [x0], #32

sub x5, x5, #4
cmp x5, #4
bge L4Loop
cmp x5, #0
beq END
cmp x5, #2
blt L1Loop

L2Loop:

ld1 {v0.8h, v1.8h}, [x1], #32
ld1 {v2.8h, v3.8h}, [x2], #32

smull v8.4s, v0.4h, v30.4h
smull2 v9.4s, v0.8h, v30.8h
smlal  v8.4s, v2.4h, v31.4h
smlal2 v9.4s, v2.8h, v31.8h

smull  v10.4s, v1.4h, v30.4h
smull2 v11.4s, v1.8h, v30.8h
smlal  v10.4s, v3.4h, v31.4h
smlal2 v11.4s, v3.8h, v31.8h

shrn  v8.4h, v8.4s, #14
shrn2 v8.8h, v9.4s, #14

shrn  v10.4h, v10.4s, #14
shrn2 v10.8h, v11.4s, #14

sqxtn v8.8b, v8.8h
sqxtn2 v8.16b, v10.8h
st1 {v8.16b}, [x0], #16

sub x5, x5, #2
cmp x5, #2
bge L2Loop
cmp x5, #0
beq END

L1Loop:
ld1 {v0.8h}, [x1], #16
ld1 {v1.8h}, [x2], #16

smull v8.4s, v0.4h, v30.4h
smull2 v9.4s, v0.8h, v30.8h
smlal  v8.4s, v1.4h, v31.4h
smlal2 v9.4s, v1.8h, v31.8h

shrn  v8.4h, v8.4s, #14
shrn2 v8.8h, v9.4s, #14
sqxtn v8.8b, v8.8h

st1 {v8.8b}, [x0], #8

sub x5, x5, #1
cmp x5, #1
bge L1Loop

END:
ldp d8,  d9,  [sp, #48]
ldp d10, d11, [sp, #32]
ldp d12, d13, [sp, #16]
ldp d14, d15, [sp], #64
ret

#endif
