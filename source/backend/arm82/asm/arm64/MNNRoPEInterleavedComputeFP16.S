//
//  MNNRoPEInterleavedComputeFP16.S
//  MNN
//
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNRoPEInterleavedComputeFP16
//void MNNRoPEInterleavedComputeFP16(void* dst, const void* src, const void* cosRow, const void* sinRow,
//                                   int numHead, int headDim, int ropeCutHeadDim);
//x0:dst, x1:src, x2:cosRow, x3:sinRow, w4:numHead, w5:headDim, w6:ropeCutHeadDim

cbz w4, .Lexit

cmp w6, #0
b.gt .Lcheck_max
mov w6, w5
b .Lrope_even
.Lcheck_max:
cmp w6, w5
b.le .Lrope_even
mov w6, w5
.Lrope_even:
bic w6, w6, #1
lsr w7, w6, #1
sub w8, w5, w6
lsl w9, w5, #1
sub x5, x9, #64

mov x11, x2
mov x12, x3
bic w10, w7, #15
cbz w10, .LphaseB

.Lblock_loop:
sub x13, x11, x2
add x14, x13, x0
add x13, x13, x1
ld2 {v4.8h, v5.8h}, [x11], #32
ld2 {v6.8h, v7.8h}, [x11], #32
fcvtl2 v5.4s, v4.8h
fcvtl v4.4s, v4.4h
fcvtl2 v7.4s, v6.8h
fcvtl v6.4s, v6.4h
ld2 {v20.8h, v21.8h}, [x12], #32
ld2 {v22.8h, v23.8h}, [x12], #32
fcvtl2 v21.4s, v20.8h
fcvtl v20.4s, v20.4h
fcvtl2 v23.4s, v22.8h
fcvtl v22.4s, v22.4h

mov w15, w4
.Lhead_a:
ld2 {v0.8h, v1.8h}, [x13], #32
ld2 {v2.8h, v3.8h}, [x13], #32
fcvtl2 v16.4s, v0.8h
fcvtl v0.4s, v0.4h
fcvtl2 v17.4s, v1.8h
fcvtl v1.4s, v1.4h
fcvtl2 v18.4s, v2.8h
fcvtl v2.4s, v2.4h
fcvtl2 v19.4s, v3.8h
fcvtl v3.4s, v3.4h
fmul v24.4s, v0.4s, v4.4s
fmul v25.4s, v1.4s, v4.4s
fmul v26.4s, v16.4s, v5.4s
fmul v27.4s, v17.4s, v5.4s
fmul v28.4s, v2.4s, v6.4s
fmul v29.4s, v3.4s, v6.4s
fmul v30.4s, v18.4s, v7.4s
fmul v31.4s, v19.4s, v7.4s
fmls v24.4s, v1.4s, v20.4s
fmla v25.4s, v0.4s, v20.4s
fmls v26.4s, v17.4s, v21.4s
fmla v27.4s, v16.4s, v21.4s
fmls v28.4s, v3.4s, v22.4s
fmla v29.4s, v2.4s, v22.4s
fmls v30.4s, v19.4s, v23.4s
fmla v31.4s, v18.4s, v23.4s
fcvtn v0.4h, v24.4s
fcvtn v1.4h, v25.4s
fcvtn v2.4h, v28.4s
fcvtn v3.4h, v29.4s
fcvtn2 v0.8h, v26.4s
fcvtn2 v1.8h, v27.4s
fcvtn2 v2.8h, v30.4s
fcvtn2 v3.8h, v31.4s
st2 {v0.8h, v1.8h}, [x14], #32
st2 {v2.8h, v3.8h}, [x14], #32
add x13, x13, x5
add x14, x14, x5
subs w15, w15, #1
b.ne .Lhead_a
subs w10, w10, #16
b.ne .Lblock_loop

.LphaseB:
mov x2, x11
mov x3, x12
and w10, w7, #15
sub w11, w7, w10
lsl w11, w11, #2
mov w15, w4
.Lhead_b:
sub w12, w4, w15
madd x13, x12, x9, x1
add x13, x13, x11
madd x14, x12, x9, x0
add x14, x14, x11
mov x5, x2
mov x6, x3
mov w12, w10

cmp w12, #8
b.lt .Lpairs4
.Lpairs8:
ld2 {v0.8h, v1.8h}, [x13], #32
ld2 {v2.8h, v3.8h}, [x5], #32
ld2 {v4.8h, v5.8h}, [x6], #32
fcvtl v16.4s, v0.4h
fcvtl2 v17.4s, v0.8h
fcvtl v18.4s, v1.4h
fcvtl2 v19.4s, v1.8h
fcvtl v20.4s, v2.4h
fcvtl2 v21.4s, v2.8h
fcvtl v22.4s, v4.4h
fcvtl2 v23.4s, v4.8h
fmul v24.4s, v16.4s, v20.4s
fmul v25.4s, v18.4s, v20.4s
fmul v26.4s, v17.4s, v21.4s
fmul v27.4s, v19.4s, v21.4s
fmls v24.4s, v18.4s, v22.4s
fmla v25.4s, v16.4s, v22.4s
fmls v26.4s, v19.4s, v23.4s
fmla v27.4s, v17.4s, v23.4s
fcvtn v0.4h, v24.4s
fcvtn v1.4h, v25.4s
fcvtn2 v0.8h, v26.4s
fcvtn2 v1.8h, v27.4s
st2 {v0.8h, v1.8h}, [x14], #32
sub w12, w12, #8
cmp w12, #8
b.ge .Lpairs8

.Lpairs4:
cmp w12, #4
b.lt .Lpairs1
ld2 {v0.4h, v1.4h}, [x13], #16
ld2 {v2.4h, v3.4h}, [x5], #16
ld2 {v4.4h, v5.4h}, [x6], #16
fcvtl v16.4s, v0.4h
fcvtl v18.4s, v1.4h
fcvtl v20.4s, v2.4h
fcvtl v22.4s, v4.4h
fmul v24.4s, v16.4s, v20.4s
fmul v25.4s, v18.4s, v20.4s
fmls v24.4s, v18.4s, v22.4s
fmla v25.4s, v16.4s, v22.4s
fcvtn v0.4h, v24.4s
fcvtn v1.4h, v25.4s
st2 {v0.4h, v1.4h}, [x14], #16
sub w12, w12, #4

.Lpairs1:
cbz w12, .Ltail_copy
.Lpair1_loop:
ldr h0, [x13], #2
ldr h1, [x13], #2
ldr h2, [x5], #4
ldr h3, [x6], #4
fcvt s0, h0
fcvt s1, h1
fcvt s2, h2
fcvt s3, h3
fmul s4, s0, s2
fmul s5, s1, s2
fmsub s4, s1, s3, s4
fmadd s5, s0, s3, s5
fcvt h4, s4
fcvt h5, s5
str h4, [x14], #2
str h5, [x14], #2
subs w12, w12, #1
b.ne .Lpair1_loop

.Ltail_copy:
cbz w8, .Lnext_head
mov w12, w8
cmp w12, #8
b.lt .Ltail1
.Ltail8:
ld1 {v0.8h}, [x13], #16
st1 {v0.8h}, [x14], #16
sub w12, w12, #8
cmp w12, #8
b.ge .Ltail8
.Ltail1:
cbz w12, .Lnext_head
ldr h0, [x13], #2
str h0, [x14], #2
subs w12, w12, #1
b.ne .Ltail1

.Lnext_head:
subs w15, w15, #1
b.ne .Lhead_b

.Lexit:
ret

#endif