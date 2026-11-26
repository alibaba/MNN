//
//  MNNRoPEComputeFP16.S
//  MNN
//
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNRoPEComputeFP16
//void MNNRoPEComputeFP16(void* dst, const void* src, const void* cosEven, const void* cosOdd,
//                        const void* sinEven, const void* sinOdd, int numHead, int headDim, int ropeCutHeadDim);
//x0:dst, x1:src, x2:cosEven, x3:cosOdd, x4:sinEven, x5:sinOdd, w6:numHead, w7:headDim,
//ropeCutHeadDim is the 9th argument and arrives at [sp].

cbz w6, .Lexit_raw
ldr w8, [sp]
sub sp, sp, #112
stp d8, d9, [sp, #48]
stp d10, d11, [sp, #64]
stp d12, d13, [sp, #80]
stp d14, d15, [sp, #96]

cmp w8, #0
b.gt .Lcheck_max
mov w8, w7
b .Lrope_even
.Lcheck_max:
cmp w8, w7
b.le .Lrope_even
mov w8, w7
.Lrope_even:
bic w8, w8, #1
lsr w9, w8, #1
uxtw x6, w6
str w9, [sp]
str x2, [sp, #8]
str x1, [sp, #16]
str x0, [sp, #24]
lsl w7, w7, #1
lsl w10, w9, #1
add x8, x7, x7
add x9, x7, x8
mov x11, x1
mov x12, x0
add x13, x1, x10
add x14, x0, x10
mul x15, x6, x7
sub x15, x15, #16
mov w0, #3
udiv w1, w6, w0
msub w0, w1, w0, w6
str w1, [sp, #40]
str w0, [sp, #44]
bic w0, w10, #15
cbz w0, .LphaseB

.Lblock_loop:
ld1 {v0.8h}, [x2], #16
fcvtl2 v1.4s, v0.8h
fcvtl v0.4s, v0.4h
ld1 {v2.8h}, [x3], #16
fcvtl2 v3.4s, v2.8h
fcvtl v2.4s, v2.4h
ld1 {v4.8h}, [x4], #16
fcvtl2 v5.4s, v4.8h
fcvtl v4.4s, v4.4h
ld1 {v6.8h}, [x5], #16
fcvtl2 v7.4s, v6.8h
fcvtl v6.4s, v6.4h
ldr w1, [sp, #40]
cbz w1, .Lhead_two
.Lhead_tri:
ldr q16, [x11]
ldr q17, [x13]
ldr q24, [x11, x7]
ldr q25, [x13, x7]
ldr q8, [x11, x8]
ldr q9, [x13, x8]
fcvtl2 v18.4s, v16.8h
fcvtl v16.4s, v16.4h
fcvtl2 v19.4s, v17.8h
fcvtl v17.4s, v17.4h
fcvtl2 v26.4s, v24.8h
fcvtl v24.4s, v24.4h
fcvtl2 v27.4s, v25.8h
fcvtl v25.4s, v25.4h
fcvtl2 v10.4s, v8.8h
fcvtl v8.4s, v8.4h
fcvtl2 v11.4s, v9.8h
fcvtl v9.4s, v9.4h
fmul v20.4s, v16.4s, v0.4s
fmul v21.4s, v18.4s, v1.4s
fmul v22.4s, v17.4s, v2.4s
fmul v23.4s, v19.4s, v3.4s
fmul v28.4s, v24.4s, v0.4s
fmul v29.4s, v26.4s, v1.4s
fmul v30.4s, v25.4s, v2.4s
fmul v31.4s, v27.4s, v3.4s
fmul v12.4s, v8.4s, v0.4s
fmul v13.4s, v10.4s, v1.4s
fmul v14.4s, v9.4s, v2.4s
fmul v15.4s, v11.4s, v3.4s
fmls v20.4s, v17.4s, v4.4s
fmls v21.4s, v19.4s, v5.4s
fmla v22.4s, v16.4s, v6.4s
fmla v23.4s, v18.4s, v7.4s
fmls v28.4s, v25.4s, v4.4s
fmls v29.4s, v27.4s, v5.4s
fmla v30.4s, v24.4s, v6.4s
fmla v31.4s, v26.4s, v7.4s
fmls v12.4s, v9.4s, v4.4s
fmls v13.4s, v11.4s, v5.4s
fmla v14.4s, v8.4s, v6.4s
fmla v15.4s, v10.4s, v7.4s
fcvtn v16.4h, v20.4s
fcvtn v17.4h, v22.4s
fcvtn v24.4h, v28.4s
fcvtn v25.4h, v30.4s
fcvtn v8.4h, v12.4s
fcvtn v9.4h, v14.4s
fcvtn2 v16.8h, v21.4s
fcvtn2 v17.8h, v23.4s
fcvtn2 v24.8h, v29.4s
fcvtn2 v25.8h, v31.4s
fcvtn2 v8.8h, v13.4s
fcvtn2 v9.8h, v15.4s
str q16, [x12]
str q17, [x14]
str q24, [x12, x7]
str q25, [x14, x7]
str q8, [x12, x8]
str q9, [x14, x8]
add x11, x11, x9
add x12, x12, x9
add x13, x13, x9
add x14, x14, x9
subs w1, w1, #1
b.ne .Lhead_tri

.Lhead_two:
ldr w1, [sp, #44]
cmp w1, #2
b.lt .Lhead_one
ldr q16, [x11]
ldr q17, [x13]
ldr q24, [x11, x7]
ldr q25, [x13, x7]
fcvtl2 v18.4s, v16.8h
fcvtl v16.4s, v16.4h
fcvtl2 v19.4s, v17.8h
fcvtl v17.4s, v17.4h
fcvtl2 v26.4s, v24.8h
fcvtl v24.4s, v24.4h
fcvtl2 v27.4s, v25.8h
fcvtl v25.4s, v25.4h
fmul v20.4s, v16.4s, v0.4s
fmul v21.4s, v18.4s, v1.4s
fmul v22.4s, v17.4s, v2.4s
fmul v23.4s, v19.4s, v3.4s
fmul v28.4s, v24.4s, v0.4s
fmul v29.4s, v26.4s, v1.4s
fmul v30.4s, v25.4s, v2.4s
fmul v31.4s, v27.4s, v3.4s
fmls v20.4s, v17.4s, v4.4s
fmls v21.4s, v19.4s, v5.4s
fmla v22.4s, v16.4s, v6.4s
fmla v23.4s, v18.4s, v7.4s
fmls v28.4s, v25.4s, v4.4s
fmls v29.4s, v27.4s, v5.4s
fmla v30.4s, v24.4s, v6.4s
fmla v31.4s, v26.4s, v7.4s
fcvtn v16.4h, v20.4s
fcvtn v17.4h, v22.4s
fcvtn v24.4h, v28.4s
fcvtn v25.4h, v30.4s
fcvtn2 v16.8h, v21.4s
fcvtn2 v17.8h, v23.4s
fcvtn2 v24.8h, v29.4s
fcvtn2 v25.8h, v31.4s
str q16, [x12]
str q17, [x14]
str q24, [x12, x7]
str q25, [x14, x7]
add x11, x11, x8
add x12, x12, x8
add x13, x13, x8
add x14, x14, x8
b .Lblock_next

.Lhead_one:
cbz w1, .Lblock_next
ldr q16, [x11]
ldr q17, [x13]
fcvtl2 v18.4s, v16.8h
fcvtl v16.4s, v16.4h
fcvtl2 v19.4s, v17.8h
fcvtl v17.4s, v17.4h
fmul v20.4s, v16.4s, v0.4s
fmul v21.4s, v18.4s, v1.4s
fmul v22.4s, v17.4s, v2.4s
fmul v23.4s, v19.4s, v3.4s
fmls v20.4s, v17.4s, v4.4s
fmls v21.4s, v19.4s, v5.4s
fmla v22.4s, v16.4s, v6.4s
fmla v23.4s, v18.4s, v7.4s
fcvtn v16.4h, v20.4s
fcvtn v17.4h, v22.4s
fcvtn2 v16.8h, v21.4s
fcvtn2 v17.8h, v23.4s
str q16, [x12]
str q17, [x14]
add x11, x11, x7
add x12, x12, x7
add x13, x13, x7
add x14, x14, x7

.Lblock_next:
sub x11, x11, x15
sub x12, x12, x15
sub x13, x13, x15
sub x14, x14, x15
subs w0, w0, #16
b.ne .Lblock_loop

.LphaseB:
ldr w9, [sp]
ldr x15, [sp, #8]
sub x15, x2, x15
ldr x1, [sp, #16]
ldr x0, [sp, #24]
add x1, x1, x15
add x0, x0, x15
and w8, w9, #7
mov w9, w6
.Lhead_b:
sub w15, w6, w9
madd x11, x15, x7, x1
madd x12, x15, x7, x0
add x13, x11, x10
add x14, x12, x10
mov w15, w8

cmp w15, #4
b.lt .Lk1
ld1 {v16.4h}, [x11], #8
ld1 {v17.4h}, [x13], #8
ld1 {v0.4h}, [x2], #8
ld1 {v2.4h}, [x3], #8
ld1 {v4.4h}, [x4], #8
ld1 {v6.4h}, [x5], #8
fcvtl v16.4s, v16.4h
fcvtl v17.4s, v17.4h
fcvtl v0.4s, v0.4h
fcvtl v2.4s, v2.4h
fcvtl v4.4s, v4.4h
fcvtl v6.4s, v6.4h
fmul v20.4s, v16.4s, v0.4s
fmul v22.4s, v17.4s, v2.4s
fmls v20.4s, v17.4s, v4.4s
fmla v22.4s, v16.4s, v6.4s
fcvtn v16.4h, v20.4s
fcvtn v17.4h, v22.4s
st1 {v16.4h}, [x12], #8
st1 {v17.4h}, [x14], #8
sub w15, w15, #4

.Lk1:
cbz w15, .Lk_done
.Lk1_loop:
ldr h16, [x11], #2
ldr h17, [x13], #2
ldr h0, [x2], #2
ldr h2, [x3], #2
ldr h4, [x4], #2
ldr h6, [x5], #2
fcvt s16, h16
fcvt s17, h17
fcvt s0, h0
fcvt s2, h2
fcvt s4, h4
fcvt s6, h6
fmul s20, s16, s0
fmul s22, s17, s2
fmsub s20, s17, s4, s20
fmadd s22, s16, s6, s22
fcvt h20, s20
fcvt h22, s22
str h20, [x12], #2
str h22, [x14], #2
subs w15, w15, #1
b.ne .Lk1_loop

.Lk_done:
lsl w15, w8, #1
sub x2, x2, x15
sub x3, x3, x15
sub x4, x4, x15
sub x5, x5, x15
lsl w15, w10, #1
sub w15, w7, w15
cbz w15, .Lnext_head
cmp w15, #16
b.lt .Ltail2
.Ltail16:
ldr q0, [x13], #16
str q0, [x14], #16
sub w15, w15, #16
cmp w15, #16
b.ge .Ltail16
.Ltail2:
cbz w15, .Lnext_head
ldr h0, [x13], #2
str h0, [x14], #2
subs w15, w15, #2
b.ne .Ltail2

.Lnext_head:
subs w9, w9, #1
b.ne .Lhead_b

ldp d8, d9, [sp, #48]
ldp d10, d11, [sp, #64]
ldp d12, d13, [sp, #80]
ldp d14, d15, [sp, #96]
add sp, sp, #112
.Lexit_raw:
ret

#endif