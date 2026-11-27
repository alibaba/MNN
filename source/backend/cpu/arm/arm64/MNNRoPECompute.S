//
//  MNNRoPECompute.S
//  MNN
//
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNRoPECompute
//void MNNRoPECompute(void* dst, const void* src, const void* cosEven, const void* cosOdd,
//                    const void* sinEven, const void* sinOdd, int numHead, int headDim, int ropeCutHeadDim);
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
lsl w7, w7, #2
lsl w10, w9, #2
add x8, x7, x7
add x9, x7, #16
add x15, x8, #16
mov x11, x1
mov x12, x0
add x13, x1, x10
add x14, x0, x10
mov w0, #3
udiv w1, w6, w0
msub w0, w1, w0, w6
str w1, [sp, #40]
str w0, [sp, #44]
mul x0, x6, x7
sub x0, x0, #32
str x0, [sp, #32]
bic w0, w10, #31
add x10, x7, x8
cbz w0, .LphaseB

.Lblock_loop:
ld1 {v0.4s, v1.4s}, [x2], #32
ld1 {v2.4s, v3.4s}, [x3], #32
ld1 {v4.4s, v5.4s}, [x4], #32
ld1 {v6.4s, v7.4s}, [x5], #32
ldr w1, [sp, #40]
cbz w1, .Lhead_two
.Lhead_tri:
ldr q16, [x11]
ldr q17, [x11, #16]
ldr q18, [x13]
ldr q19, [x13, #16]
ldr q20, [x11, x7]
ldr q21, [x11, x9]
ldr q22, [x13, x7]
ldr q23, [x13, x9]
ldr q8, [x11, x8]
ldr q9, [x11, x15]
ldr q10, [x13, x8]
ldr q11, [x13, x15]
fmul v24.4s, v16.4s, v0.4s
fmul v25.4s, v17.4s, v1.4s
fmul v26.4s, v18.4s, v2.4s
fmul v27.4s, v19.4s, v3.4s
fmul v28.4s, v20.4s, v0.4s
fmul v29.4s, v21.4s, v1.4s
fmul v30.4s, v22.4s, v2.4s
fmul v31.4s, v23.4s, v3.4s
fmul v12.4s, v8.4s, v0.4s
fmul v13.4s, v9.4s, v1.4s
fmul v14.4s, v10.4s, v2.4s
fmul v15.4s, v11.4s, v3.4s
fmls v24.4s, v18.4s, v4.4s
fmls v25.4s, v19.4s, v5.4s
fmla v26.4s, v16.4s, v6.4s
fmla v27.4s, v17.4s, v7.4s
fmls v28.4s, v22.4s, v4.4s
fmls v29.4s, v23.4s, v5.4s
fmla v30.4s, v20.4s, v6.4s
fmla v31.4s, v21.4s, v7.4s
fmls v12.4s, v10.4s, v4.4s
fmls v13.4s, v11.4s, v5.4s
fmla v14.4s, v8.4s, v6.4s
fmla v15.4s, v9.4s, v7.4s
str q24, [x12]
str q25, [x12, #16]
str q26, [x14]
str q27, [x14, #16]
str q28, [x12, x7]
str q29, [x12, x9]
str q30, [x14, x7]
str q31, [x14, x9]
str q12, [x12, x8]
str q13, [x12, x15]
str q14, [x14, x8]
str q15, [x14, x15]
add x11, x11, x10
add x12, x12, x10
add x13, x13, x10
add x14, x14, x10
subs w1, w1, #1
b.ne .Lhead_tri

.Lhead_two:
ldr w1, [sp, #44]
cmp w1, #2
b.lt .Lhead_one
ldr q16, [x11]
ldr q17, [x11, #16]
ldr q18, [x13]
ldr q19, [x13, #16]
ldr q24, [x11, x7]
ldr q25, [x11, x9]
ldr q26, [x13, x7]
ldr q27, [x13, x9]
fmul v20.4s, v16.4s, v0.4s
fmul v21.4s, v17.4s, v1.4s
fmul v22.4s, v18.4s, v2.4s
fmul v23.4s, v19.4s, v3.4s
fmul v28.4s, v24.4s, v0.4s
fmul v29.4s, v25.4s, v1.4s
fmul v30.4s, v26.4s, v2.4s
fmul v31.4s, v27.4s, v3.4s
fmls v20.4s, v18.4s, v4.4s
fmls v21.4s, v19.4s, v5.4s
fmla v22.4s, v16.4s, v6.4s
fmla v23.4s, v17.4s, v7.4s
fmls v28.4s, v26.4s, v4.4s
fmls v29.4s, v27.4s, v5.4s
fmla v30.4s, v24.4s, v6.4s
fmla v31.4s, v25.4s, v7.4s
str q20, [x12]
str q21, [x12, #16]
str q22, [x14]
str q23, [x14, #16]
str q28, [x12, x7]
str q29, [x12, x9]
str q30, [x14, x7]
str q31, [x14, x9]
add x11, x11, x8
add x12, x12, x8
add x13, x13, x8
add x14, x14, x8
b .Lblock_next

.Lhead_one:
cbz w1, .Lblock_next
ldr q16, [x11]
ldr q17, [x11, #16]
ldr q18, [x13]
ldr q19, [x13, #16]
fmul v20.4s, v16.4s, v0.4s
fmul v21.4s, v17.4s, v1.4s
fmul v22.4s, v18.4s, v2.4s
fmul v23.4s, v19.4s, v3.4s
fmls v20.4s, v18.4s, v4.4s
fmls v21.4s, v19.4s, v5.4s
fmla v22.4s, v16.4s, v6.4s
fmla v23.4s, v17.4s, v7.4s
str q20, [x12]
str q21, [x12, #16]
str q22, [x14]
str q23, [x14, #16]
add x11, x11, x7
add x12, x12, x7
add x13, x13, x7
add x14, x14, x7

.Lblock_next:
ldr x1, [sp, #32]
sub x11, x11, x1
sub x12, x12, x1
sub x13, x13, x1
sub x14, x14, x1
subs w0, w0, #32
b.ne .Lblock_loop

.LphaseB:
ldr w9, [sp]
lsl w10, w9, #2
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
ld1 {v16.4s}, [x11], #16
ld1 {v18.4s}, [x13], #16
ld1 {v0.4s}, [x2], #16
ld1 {v2.4s}, [x3], #16
ld1 {v4.4s}, [x4], #16
ld1 {v6.4s}, [x5], #16
fmul v20.4s, v16.4s, v0.4s
fmul v22.4s, v18.4s, v2.4s
fmls v20.4s, v18.4s, v4.4s
fmla v22.4s, v16.4s, v6.4s
st1 {v20.4s}, [x12], #16
st1 {v22.4s}, [x14], #16
sub w15, w15, #4

.Lk1:
cbz w15, .Lk_done
.Lk1_loop:
ldr s16, [x11], #4
ldr s18, [x13], #4
ldr s0, [x2], #4
ldr s2, [x3], #4
ldr s4, [x4], #4
ldr s6, [x5], #4
fmul s20, s16, s0
fmul s22, s18, s2
fmsub s20, s18, s4, s20
fmadd s22, s16, s6, s22
str s20, [x12], #4
str s22, [x14], #4
subs w15, w15, #1
b.ne .Lk1_loop

.Lk_done:
lsl w15, w8, #2
sub x2, x2, x15
sub x3, x3, x15
sub x4, x4, x15
sub x5, x5, x15
lsl w15, w10, #1
sub w15, w7, w15
cbz w15, .Lnext_head
cmp w15, #16
b.lt .Ltail4
.Ltail16:
ld1 {v0.4s}, [x13], #16
st1 {v0.4s}, [x14], #16
sub w15, w15, #16
cmp w15, #16
b.ge .Ltail16
.Ltail4:
cbz w15, .Lnext_head
ldr s0, [x13], #4
str s0, [x14], #4
subs w15, w15, #4
b.ne .Ltail4

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