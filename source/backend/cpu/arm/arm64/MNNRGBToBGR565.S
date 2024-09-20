#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

// void MNNRGBToBGR565Fast(const unsigned char* source, unsigned char* dest, size_t count);
asm_function MNNRGBToBGR565Fast
// x0: source, x1: dest, x2: count, x3: c
stp d14, d15, [sp, #(-16 * 4)]!
stp d12, d13, [sp, #(16 * 1)]
stp d10, d11, [sp, #(16 * 2)]
stp d8,  d9,  [sp, #(16 * 3)]

movi v31.16b, #8
neg v31.16b, v31.16b

L6:
cmp x2, #6
blt L4

movi v30.16b, #4
neg v30.16b, v30.16b

ld3 {v0.16b, v1.16b, v2.16b}, [x0], #48
ld3 {v11.16b, v12.16b, v13.16b}, [x0], #48
ld3 {v24.16b, v25.16b, v26.16b}, [x0], #48
and v0.16b, v0.16b, v31.16b // r & ~7
and v1.16b, v1.16b, v30.16b // g & ~3
ushr v2.16b, v2.16b, #3  // b >> 3
and v11.16b, v11.16b, v31.16b // r & ~7
and v12.16b, v12.16b, v30.16b // g & ~3
ushr v13.16b, v13.16b, #3  // b >> 3
and v24.16b, v24.16b, v31.16b // r & ~7
and v25.16b, v25.16b, v30.16b // g & ~3
ushr v26.16b, v26.16b, #3  // b >> 3
sub x2, x2, #6

ushll v3.8h, v0.8b, #7
shl v3.8h, v3.8h, #1
ushll v4.8h, v1.8b, #3
uxtl v5.8h, v2.8b
ushll2 v8.8h, v0.16b, #7
shl v8.8h, v8.8h, #1
ushll2 v9.8h, v1.16b, #3
uxtl2 v10.8h, v2.16b

ushll v14.8h, v11.8b, #7
shl v14.8h, v14.8h, #1
ushll v15.8h, v12.8b, #3
uxtl v16.8h, v13.8b
ushll2 v17.8h, v11.16b, #7
shl v17.8h, v17.8h, #1
ushll2 v18.8h, v12.16b, #3
uxtl2 v19.8h, v13.16b

ushll v6.8h, v24.8b, #7
shl v6.8h, v6.8h, #1
ushll v7.8h, v25.8b, #3
uxtl v27.8h, v26.8b
ushll2 v28.8h, v24.16b, #7
shl v28.8h, v28.8h, #1
ushll2 v29.8h, v25.16b, #3
uxtl2 v30.8h, v26.16b

orr v0.16b, v3.16b, v4.16b
orr v0.16b, v0.16b, v5.16b
orr v1.16b, v8.16b, v9.16b
orr v1.16b, v1.16b, v10.16b

orr v2.16b, v14.16b, v15.16b
orr v2.16b, v2.16b, v16.16b
orr v3.16b, v17.16b, v18.16b
orr v3.16b, v3.16b, v19.16b

orr v4.16b, v6.16b, v7.16b
orr v4.16b, v4.16b, v27.16b
orr v5.16b, v28.16b, v29.16b
orr v5.16b, v5.16b, v30.16b

st1 {v0.8h, v1.8h, v2.8h, v3.8h}, [x1], #64
st1 {v4.8h, v5.8h}, [x1], #32

b L6

L4:
movi v30.16b, #4
neg v30.16b, v30.16b
cmp x2, #4
blt L2

ld3 {v0.16b, v1.16b, v2.16b}, [x0], #48
ld3 {v11.16b, v12.16b, v13.16b}, [x0], #48
and v0.16b, v0.16b, v31.16b // r & ~7
and v1.16b, v1.16b, v30.16b // g & ~3
ushr v2.16b, v2.16b, #3  // b >> 3
and v11.16b, v11.16b, v31.16b // r & ~7
and v12.16b, v12.16b, v30.16b // g & ~3
ushr v13.16b, v13.16b, #3  // b >> 3
sub x2, x2, #4

ushll v3.8h, v0.8b, #7
shl v3.8h, v3.8h, #1
ushll v4.8h, v1.8b, #3
uxtl v5.8h, v2.8b
ushll2 v8.8h, v0.16b, #7
shl v8.8h, v8.8h, #1
ushll2 v9.8h, v1.16b, #3
uxtl2 v10.8h, v2.16b

ushll v14.8h, v11.8b, #7
shl v14.8h, v14.8h, #1
ushll v15.8h, v12.8b, #3
uxtl v16.8h, v13.8b
ushll2 v17.8h, v11.16b, #7
shl v17.8h, v17.8h, #1
ushll2 v18.8h, v12.16b, #3
uxtl2 v19.8h, v13.16b


orr v20.16b, v3.16b, v4.16b
orr v20.16b, v20.16b, v5.16b
orr v21.16b, v8.16b, v9.16b
orr v21.16b, v21.16b, v10.16b

orr v22.16b, v14.16b, v15.16b
orr v22.16b, v22.16b, v16.16b
orr v23.16b, v17.16b, v18.16b
orr v23.16b, v23.16b, v19.16b

st1 {v20.8h, v21.8h, v22.8h, v23.8h}, [x1], #64

b L4

L2:
cmp x2, #2
blt L1

ld3 {v0.16b, v1.16b, v2.16b}, [x0], #48
and v0.16b, v0.16b, v31.16b // r & ~7
and v1.16b, v1.16b, v30.16b // g & ~7
sub x2, x2, #2
ushr v2.16b, v2.16b, #3  // b >> 3

ushll v3.8h, v0.8b, #7
shl v3.8h, v3.8h, #1
ushll v4.8h, v1.8b, #3
uxtl v5.8h, v2.8b
ushll2 v8.8h, v0.16b, #7
shl v8.8h, v8.8h, #1
ushll2 v9.8h, v1.16b, #3
uxtl2 v10.8h, v2.16b

orr v6.16b, v3.16b, v4.16b
orr v6.16b, v6.16b, v5.16b
orr v7.16b, v8.16b, v9.16b
orr v7.16b, v7.16b, v10.16b

st1 {v6.8h, v7.8h}, [x1], #32

b L2

L1:
cmp x2, #1
blt End

ld3 {v0.8b, v1.8b, v2.8b}, [x0], #24
and v0.8b, v0.8b, v31.8b // r & ~7
and v1.8b, v1.8b, v30.8b // g & ~7
ushr v2.8b, v2.8b, #3  // b >> 3
ushll v0.8h, v0.8b, #7
shl v0.8h, v0.8h, #1
ushll v1.8h, v1.8b, #3
uxtl v2.8h, v2.8b
orr v0.16b, v0.16b, v1.16b
orr v0.16b, v0.16b, v2.16b

st1 {v0.8h}, [x1], #16

End:
ldp d8,  d9,  [sp, #(16 * 3)]
ldp d10, d11, [sp, #(16 * 2)]
ldp d12, d13, [sp, #(16 * 1)]
ldp d14, d15, [sp], #(16 * 4)
ret
#endif
