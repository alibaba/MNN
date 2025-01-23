//
//  MNNGeneralIm2col_Arm82.S
//  MNN
//
//  Created by MNN on 2024/12/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"
.text
.align 5

//void MNNGeneralIm2col_Arm82(float* destOrigin, float const** sourceGroup, const int32_t* info, const int32_t* el, int32_t LP, int32_t pack)
asm_function MNNGeneralIm2col_Arm82

// x0:destOrigin, x1:sourceGroup, x2:info, x3:el, x4:LP, x5:pack
stp d14, d15, [sp, #(-16 * 5)]!
stp d12, d13, [sp, #(16 * 1)]
stp d10, d11, [sp, #(16 * 2)]
stp d8,  d9,  [sp, #(16 * 3)]
stp x19, x20, [sp, #(16 * 4)]

// load el info
ldr w6, [x2, #0]  // number
ldr w7, [x2, #4]  // eReal
ldr w15, [x2, #8]  // eDest (< EP)
ldr w9, [x2, #12] // offset (stride)
ldr x14, [x1, #0] // src start
lsl x9, x9, #4    // pack*offset*sizeof(float16_t)
// stride
lsl x19, x15, #3 // eDest*LP*sizeof(float16_t)
lsl x7, x7, #4  // eReal*pack*sizeof(float16_t)
mov x20, #3      // Arm82,LP=4

LoopNum:

ldr w10, [x3], #4 // e
ldr w11, [x3], #4 // l
ldr w12, [x3], #4 // eOffset
ldr w13, [x3], #4 // lOffset
// dst address: x2
and x2, x13, x20 // lR
sub x13, x13, x2 // lOffset-lR
mul x13, x13, x15 // (lOffset-lR)*(eDest)
add x13, x13, x2 // (lOffset-lR)*(eDest)+lR
add x13, x13, x12, LSL #2 // + eoffset*lp
add x2, x0, x13, LSL #1 // *sizeof(float16_t)

lsl x8, x19, #1  // 2*(eDest*LP*sizeof(float16_t))
cmp x11, #8
blt LoopL4

LoopL8:
mov x5, x2
mov x4, x14
mov x13, x10
add x12, x2, x19 // eDest*LP*sizeof(float16_t)

cmp x13, #12
blt LoopL8E8

sub x8, x8, #64
LoopL8E12:
sub x13, x13, #12
ld1 {v0.8h}, [x14], x9
ld1 {v1.8h}, [x14], x9
ld1 {v2.8h}, [x14], x9
ld1 {v3.8h}, [x14], x9
ld1 {v4.8h}, [x14], x9
ld1 {v5.8h}, [x14], x9
ld1 {v6.8h}, [x14], x9
ld1 {v7.8h}, [x14], x9
ld1 {v8.8h}, [x14], x9
ld1 {v9.8h}, [x14], x9
ld1 {v10.8h}, [x14], x9
ld1 {v11.8h}, [x14], x9
zip1 v12.2d, v0.2d, v1.2d
zip1 v13.2d, v2.2d, v3.2d
zip1 v14.2d, v4.2d, v5.2d
zip1 v15.2d, v6.2d, v7.2d
zip1 v16.2d, v8.2d, v9.2d
zip1 v17.2d, v10.2d, v11.2d
zip2 v18.2d, v0.2d, v1.2d
zip2 v19.2d, v2.2d, v3.2d
zip2 v20.2d, v4.2d, v5.2d
zip2 v21.2d, v6.2d, v7.2d
zip2 v22.2d, v8.2d, v9.2d
zip2 v23.2d, v10.2d, v11.2d

st1 {v12.8h, v13.8h, v14.8h, v15.8h}, [x2], #64
st1 {v16.8h, v17.8h}, [x2], x8
st1 {v18.8h, v19.8h, v20.8h, v21.8h}, [x12], #64
st1 {v22.8h, v23.8h}, [x12], x8
cmp x13, #12
bge LoopL8E12
add x8, x8, #64

LoopL8E8:
cmp x13, #8
blt LoopL8E4
ld1 {v0.8h}, [x14], x9
ld1 {v1.8h}, [x14], x9
ld1 {v2.8h}, [x14], x9
ld1 {v3.8h}, [x14], x9
ld1 {v4.8h}, [x14], x9
ld1 {v5.8h}, [x14], x9
ld1 {v6.8h}, [x14], x9
ld1 {v7.8h}, [x14], x9
zip1 v12.2d, v0.2d, v1.2d
zip1 v13.2d, v2.2d, v3.2d
zip1 v14.2d, v4.2d, v5.2d
zip1 v15.2d, v6.2d, v7.2d
zip2 v18.2d, v0.2d, v1.2d
zip2 v19.2d, v2.2d, v3.2d
zip2 v20.2d, v4.2d, v5.2d
zip2 v21.2d, v6.2d, v7.2d
st1 {v12.8h, v13.8h, v14.8h, v15.8h}, [x2], #64
st1 {v18.8h, v19.8h, v20.8h, v21.8h}, [x12], #64
sub x13, x13, #8

LoopL8E4:
cmp x13, #4
blt LoopL8E2
ld1 {v0.8h}, [x14], x9
ld1 {v1.8h}, [x14], x9
ld1 {v2.8h}, [x14], x9
ld1 {v3.8h}, [x14], x9
zip1 v12.2d, v0.2d, v1.2d
zip1 v13.2d, v2.2d, v3.2d
zip2 v18.2d, v0.2d, v1.2d
zip2 v19.2d, v2.2d, v3.2d
st1 {v12.8h, v13.8h}, [x2], #32
st1 {v18.8h, v19.8h}, [x12], #32
sub x13, x13, #4

LoopL8E2:
cmp x13, #2
blt LoopL8E1
ld1 {v0.8h}, [x14], x9
ld1 {v1.8h}, [x14], x9
zip1 v12.2d, v0.2d, v1.2d
zip2 v18.2d, v0.2d, v1.2d
st1 {v12.8h}, [x2], #16
st1 {v18.8h}, [x12], #16
sub x13, x13, #2

LoopL8E1:
cmp x13, #1
blt EndL8LoopE
ld1 {v0.8h}, [x14], x9
st1 {v0.d}[0], [x2], #8
st1 {v0.d}[1], [x12], #8

EndL8LoopE:
sub x11, x11, #8
cmp x11, #8
add x2, x5, x8 // eDest*LP*2*sizeof(float16_t)
add x14, x4, x7
bge LoopL8
cbz x11, EndLoopL

LoopL4:
mov x5, x2
mov x4, x14
mov x13, x10

cmp x13, #12
blt LoopL4E8

sub x8, x19, #64
LoopL4E12:
sub x13, x13, #12
ld1 {v0.8h}, [x14], x9
ld1 {v1.8h}, [x14], x9
ld1 {v2.8h}, [x14], x9
ld1 {v3.8h}, [x14], x9
ld1 {v4.8h}, [x14], x9
ld1 {v5.8h}, [x14], x9
ld1 {v6.8h}, [x14], x9
ld1 {v7.8h}, [x14], x9
ld1 {v8.8h}, [x14], x9
ld1 {v9.8h}, [x14], x9
ld1 {v10.8h}, [x14], x9
ld1 {v11.8h}, [x14], x9
zip1 v12.2d, v0.2d, v1.2d
zip1 v13.2d, v2.2d, v3.2d
zip1 v14.2d, v4.2d, v5.2d
zip1 v15.2d, v6.2d, v7.2d
zip1 v16.2d, v8.2d, v9.2d
zip1 v17.2d, v10.2d, v11.2d

st1 {v12.8h, v13.8h, v14.8h, v15.8h}, [x2], #64
st1 {v16.8h, v17.8h}, [x2], x8
cmp x13, #12
bge LoopL4E12

LoopL4E8:
cmp x13, #8
blt LoopL4E4
ld1 {v0.8h}, [x14], x9
ld1 {v1.8h}, [x14], x9
ld1 {v2.8h}, [x14], x9
ld1 {v3.8h}, [x14], x9
ld1 {v4.8h}, [x14], x9
ld1 {v5.8h}, [x14], x9
ld1 {v6.8h}, [x14], x9
ld1 {v7.8h}, [x14], x9
zip1 v12.2d, v0.2d, v1.2d
zip1 v13.2d, v2.2d, v3.2d
zip1 v14.2d, v4.2d, v5.2d
zip1 v15.2d, v6.2d, v7.2d
st1 {v12.8h, v13.8h, v14.8h, v15.8h}, [x2], #64
sub x13, x13, #8

LoopL4E4:
cmp x13, #4
blt LoopL4E2
ld1 {v0.8h}, [x14], x9
ld1 {v1.8h}, [x14], x9
ld1 {v2.8h}, [x14], x9
ld1 {v3.8h}, [x14], x9
zip1 v12.2d, v0.2d, v1.2d
zip1 v13.2d, v2.2d, v3.2d
st1 {v12.8h, v13.8h}, [x2], #32
sub x13, x13, #4

LoopL4E2:
cmp x13, #2
blt LoopL4E1
ld1 {v0.8h}, [x14], x9
ld1 {v1.8h}, [x14], x9
zip1 v12.2d, v0.2d, v1.2d
st1 {v12.8h}, [x2], #16
sub x13, x13, #2

LoopL4E1:
cmp x13, #1
blt EndLoopL
ld1 {v0.8h}, [x14], x9
st1 {v0.d}[0], [x2], #8

EndLoopL:
subs x6, x6, #1
add x1, x1, #8
ldr x14, [x1, #0]
bne LoopNum


End:
ldp x19, x20, [sp, #(16 * 4)]
ldp d8,  d9,  [sp, #(16 * 3)]
ldp d10, d11, [sp, #(16 * 2)]
ldp d12, d13, [sp, #(16 * 1)]
ldp d14, d15, [sp], #(16 * 5)
ret

#endif

