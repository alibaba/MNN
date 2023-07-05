//  MNNBilinearSampleC8.S
//  MNN
//
//  Created by MNN on 2019/01/18.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"
.text
.align 5
asm_function MNNBilinearSampleC8
// void MNNBilinearSampleC8(const int8_t* src, int16_t* dst, const int32_t* position, const float* factor, int8_t* zeroPoint, size_t number);

// Auto load:
// x0: src, x1: dst, x2: position, x3: factor, x4:zeroPoint, x5: number

stp d14, d15, [sp, #(-16 * 7)]!
stp d12, d13, [sp, #16]
stp d10, d11, [sp, #32]
stp d8,  d9,  [sp, #48]
stp x23, x24, [sp, #(16 * 4)]
stp x21, x22, [sp, #(16 * 5)]
stp x19, x20, [sp, #(16 * 6)]

mov w15, #8       // w15: pack
uxtw x15, w15
movi v14.4s, #128

cmp x5, #0
beq END
cmp x5, #2
blt L1Loop
cmp x5, #4
blt L2Loop


L4Loop:

ld1 {v22.4s}, [x3], #16       // v22: factor
fmov v23.4s, #1.0
fsub v23.4s, v23.4s, v22.4s   // v23: 1-factor
fmul v23.4s, v23.4s, v14.s[0]
fmul v22.4s, v22.4s, v14.s[0]

dup v30.8b, v23.b[0]   // v30: sf0
dup v31.8b, v22.b[0]   // v31: df0
dup v28.8b, v23.b[4]  //  v28: sf1
dup v29.8b, v22.b[4]  //  v29: df1
dup v26.8b, v23.b[8]  //  v26: sf2
dup v27.8b, v22.b[8]  //  v27: df2
dup v24.8b, v23.b[12]  //  v24:sf3
dup v25.8b, v22.b[12]  //  v25:df3

/* src offset */

ldr w7, [x2, #0]  // w7: position[2i]
ldr w8, [x2, #4] // w8: position[2i+1]
uxtw x7, w7
uxtw x8, w8
mul x7, x15, x7
mul x8, x15, x8

ldr w11, [x2, #8] // w11: position[2i+2]
ldr w12, [x2, #12]  // w12: position[2i+3]
uxtw x11, w11
uxtw x12, w12
mul x11, x15, x11
mul x12, x15, x12

ldr w9, [x2, #16] // w9: position[2i+4]
ldr w10, [x2, #20]  // w10: position[2i+5]
uxtw x9,  w9
uxtw x10, w10
mul x9, x15, x9
mul x10, x15, x10

ldr w13, [x2, #24] // w13: position[2i+6]
ldr w14, [x2, #28]  // w14: position[2i+8]
add x2, x2, #32
uxtw x13, w13
uxtw x14, w14
mul x13, x15, x13
mul x14, x15, x14

add x7, x0, x7
add x8, x0, x8
add x11, x0, x11
add x12, x0, x12

add x9, x0, x9
add x10, x0, x10
add x13, x0, x13
add x14, x0, x14

ld1 {v0.8b}, [x7]
ld1 {v1.8b}, [x8]
ld1 {v2.8b}, [x11]
ld1 {v3.8b}, [x12]

ld1 {v4.8b}, [x9]
ld1 {v5.8b}, [x10]
ld1 {v6.8b}, [x13]
ld1 {v7.8b}, [x14]

cmp w4, #0
beq L4COMPUTE

L4COMPUTE:
smull v8.8h, v0.8b,  v30.8b
smlal v8.8h, v1.8b,  v31.8b
smull v9.8h, v2.8b,  v28.8b
smlal v9.8h, v3.8b,  v29.8b
smull v10.8h, v4.8b, v26.8b
smlal v10.8h, v5.8b, v27.8b
smull v11.8h, v6.8b, v24.8b
smlal v11.8h, v7.8b, v25.8b

st1 {v8.8h, v9.8h, v10.8h, v11.8h}, [x1], #64

sub x5, x5, #4
cmp x5, #4
bge L4Loop
cmp x5, #0
beq END
cmp x5, #2
blt L1Loop

L2Loop:
ld1 {v22.2s}, [x3], #8        // v22: factor
fmov v23.2s, #1.0
fsub v23.2s, v23.2s, v22.2s   // v23: 1-factor
fmul v23.2s, v23.2s, v14.s[0]
fmul v22.2s, v22.2s, v14.s[0]

dup v30.8b, v23.b[0]   // v30: sf0
dup v31.8b, v22.b[0]   // v31: df0
dup v28.8b, v23.b[4]  //  v28: sf1
dup v29.8b, v22.b[4]  //  v29: df1

/* src offset */
ldr w7, [x2, #0]  // w7: position[2i]
ldr w8, [x2, #4] // w8: position[2i+1]
uxtw x7, w7
uxtw x8, w8
mul x7, x15, x7
mul x8, x15, x8
ldr w11, [x2, #8] // w11: position[2i+2]
ldr w12, [x2, #12]  // w12: position[2i+3]
add x2, x2, #16
uxtw x11, w11
uxtw x12, w12
mul x11, x15, x11
mul x12, x15, x12

add x7, x0, x7
add x8, x0, x8
add x11, x0, x11
add x12, x0, x12

ld1 {v0.8b}, [x7]
ld1 {v1.8b}, [x8]
ld1 {v2.8b}, [x11]
ld1 {v3.8b}, [x12]

cmp w4, #0
beq L2COMPUTE
L2COMPUTE:
smull v4.8h, v0.8b, v30.8b
smlal v4.8h, v1.8b, v31.8b

smull v5.8h, v2.8b, v28.8b
smlal v5.8h, v3.8b, v29.8b

st1 {v4.8h, v5.8h}, [x1], #32

sub x5, x5, #2
cmp x5, #2
bge L2Loop
cmp x5, #0
beq END

L1Loop:
ld1 {v31.s}[0], [x3], #4
dup v31.4s, v31.s[0]          
fmov s30, #1.0
fsub s30, s30, s31
fmul s30, s30, s14     // (float)t -> (int16)t
fmul s31, s31, s14
dup v31.16b, v31.b[0]   // v31: df0
dup v30.16b, v30.b[0]   // v30: sf0

/* src offset */
ldr w7, [x2, #0]  // w7: position[2i]
ldr w8, [x2, #4] // w8: position[2i+1]
uxtw x7, w7
uxtw x8, w8
mul x7, x15, x7
mul x8, x15, x8
add x2, x2, #8

add x9, x0, x7
add x10, x0, x8

ld1 {v0.8b}, [x9]
ld1 {v8.8b}, [x10]
smull v1.8h, v0.8b, v30.8b
smlal v1.8h, v8.8b, v31.8b

st1 {v1.8h}, [x1], #16

sub x5, x5, #1
cmp x5, #1
bge L1Loop

END:
ldp x19, x20, [sp, #(16 * 6)]
ldp x21, x22, [sp, #(16 * 5)]
ldp x23, x24, [sp, #(16 * 4)]
ldp d8,  d9,  [sp, #48]
ldp d10, d11, [sp, #32]
ldp d12, d13, [sp, #16]
ldp d14, d15, [sp], #(16 * 7)
ret

#endif
