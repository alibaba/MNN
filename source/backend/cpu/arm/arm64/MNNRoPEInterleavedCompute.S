//
//  MNNRoPEInterleavedCompute.S
//  MNN
//
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNRoPEInterleavedCompute
//void MNNRoPEInterleavedCompute(void* dst, const void* src, const void* cosRow, const void* sinRow,
//                               int numHead, int headDim, int ropeCutHeadDim);
//x0:dst, x1:src, x2:cosRow, x3:sinRow, w4:numHead, w5:headDim, w6:ropeCutHeadDim

cbz w4, .Lexit_raw
stp d8, d9, [sp, #-64]!
stp d10, d11, [sp, #16]
stp d12, d13, [sp, #32]
stp d14, d15, [sp, #48]

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
lsl w9, w5, #2
sub x5, x9, #128

mov x11, x2
mov x12, x3
bic w10, w7, #15
cbz w10, .LphaseB

.Lblock_loop:
sub x13, x11, x2
add x14, x13, x0
add x13, x13, x1
ld2 {v0.4s, v1.4s}, [x11], #32
ld2 {v2.4s, v3.4s}, [x11], #32
ld2 {v4.4s, v5.4s}, [x11], #32
ld2 {v6.4s, v7.4s}, [x11], #32
ld2 {v8.4s, v9.4s}, [x12], #32
ld2 {v10.4s, v11.4s}, [x12], #32
ld2 {v12.4s, v13.4s}, [x12], #32
ld2 {v14.4s, v15.4s}, [x12], #32

mov w15, w4
.Lhead_a:
ld2 {v16.4s, v17.4s}, [x13], #32
ld2 {v18.4s, v19.4s}, [x13], #32
ld2 {v20.4s, v21.4s}, [x13], #32
ld2 {v22.4s, v23.4s}, [x13], #32
fmul v24.4s, v16.4s, v0.4s
fmul v25.4s, v17.4s, v0.4s
fmul v26.4s, v18.4s, v2.4s
fmul v27.4s, v19.4s, v2.4s
fmul v28.4s, v20.4s, v4.4s
fmul v29.4s, v21.4s, v4.4s
fmul v30.4s, v22.4s, v6.4s
fmul v31.4s, v23.4s, v6.4s
fmls v24.4s, v17.4s, v8.4s
fmla v25.4s, v16.4s, v8.4s
fmls v26.4s, v19.4s, v10.4s
fmla v27.4s, v18.4s, v10.4s
fmls v28.4s, v21.4s, v12.4s
fmla v29.4s, v20.4s, v12.4s
fmls v30.4s, v23.4s, v14.4s
fmla v31.4s, v22.4s, v14.4s
st2 {v24.4s, v25.4s}, [x14], #32
st2 {v26.4s, v27.4s}, [x14], #32
st2 {v28.4s, v29.4s}, [x14], #32
st2 {v30.4s, v31.4s}, [x14], #32
add x13, x13, x5
add x14, x14, x5
subs w15, w15, #1
b.ne .Lhead_a
sub w10, w10, #16
cbnz w10, .Lblock_loop

.LphaseB:
mov x2, x11
mov x3, x12
and w10, w7, #15
sub w11, w7, w10
lsl w11, w11, #3
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
ld2 {v0.4s, v1.4s}, [x13], #32
ld2 {v2.4s, v3.4s}, [x13], #32
ld2 {v4.4s, v5.4s}, [x5], #32
ld2 {v6.4s, v7.4s}, [x5], #32
ld2 {v16.4s, v17.4s}, [x6], #32
ld2 {v18.4s, v19.4s}, [x6], #32
fmul v20.4s, v0.4s, v4.4s
fmul v21.4s, v1.4s, v4.4s
fmul v22.4s, v2.4s, v6.4s
fmul v23.4s, v3.4s, v6.4s
fmls v20.4s, v1.4s, v16.4s
fmla v21.4s, v0.4s, v16.4s
fmls v22.4s, v3.4s, v18.4s
fmla v23.4s, v2.4s, v18.4s
st2 {v20.4s, v21.4s}, [x14], #32
st2 {v22.4s, v23.4s}, [x14], #32
sub w12, w12, #8
cmp w12, #8
b.ge .Lpairs8

.Lpairs4:
cmp w12, #4
b.lt .Lpairs1
ld2 {v0.4s, v1.4s}, [x13], #32
ld2 {v4.4s, v5.4s}, [x5], #32
ld2 {v16.4s, v17.4s}, [x6], #32
fmul v20.4s, v0.4s, v4.4s
fmul v21.4s, v1.4s, v4.4s
fmls v20.4s, v1.4s, v16.4s
fmla v21.4s, v0.4s, v16.4s
st2 {v20.4s, v21.4s}, [x14], #32
sub w12, w12, #4

.Lpairs1:
cbz w12, .Ltail_copy
.Lpair1_loop:
ldr s0, [x13], #4
ldr s1, [x13], #4
ldr s2, [x5], #8
ldr s3, [x6], #8
fmul s4, s0, s2
fmsub s4, s1, s3, s4
fmul s5, s1, s2
fmadd s5, s0, s3, s5
str s4, [x14], #4
str s5, [x14], #4
subs w12, w12, #1
b.ne .Lpair1_loop

.Ltail_copy:
cbz w8, .Lnext_head
mov w12, w8
cmp w12, #4
b.lt .Ltail1
.Ltail4:
ld1 {v0.4s}, [x13], #16
st1 {v0.4s}, [x14], #16
sub w12, w12, #4
cmp w12, #4
b.ge .Ltail4
.Ltail1:
cbz w12, .Lnext_head
ldr s0, [x13], #4
str s0, [x14], #4
subs w12, w12, #1
b.ne .Ltail1

.Lnext_head:
subs w15, w15, #1
b.ne .Lhead_b

.Lexit:
ldp d14, d15, [sp, #48]
ldp d12, d13, [sp, #32]
ldp d10, d11, [sp, #16]
ldp d8, d9, [sp], #64
.Lexit_raw:
ret

#endif