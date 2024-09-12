#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5
// void MNNGRAYToC4Fast(const unsigned char* source, unsigned char* dest, size_t count);
asm_function MNNGRAYToC4Fast
// x0: source, x1: dest, x2: count
stp d14, d15, [sp, #(-16 * 4)]!
stp d12, d13, [sp, #(16 * 1)]
stp d10, d11, [sp, #(16 * 2)]
stp d8,  d9,  [sp, #(16 * 3)]
movi v31.16b, #255

L12:
cmp x2, #12
blt L8
ld1 {v0.16b, v1.16b, v2.16b, v3.16b}, [x0], #64
ld1 {v21.16b, v22.16b}, [x0], #32
sub x2, x2, #12
mov v5.16b, v0.16b
mov v6.16b, v0.16b
mov v7.16b, v0.16b
mov v8.16b, v31.16b

mov v9.16b, v1.16b
mov v10.16b, v1.16b
mov v11.16b, v1.16b
mov v12.16b, v31.16b

mov v13.16b, v2.16b
mov v14.16b, v2.16b
mov v15.16b, v2.16b
mov v16.16b, v31.16b

mov v17.16b, v3.16b
mov v18.16b, v3.16b
mov v19.16b, v3.16b
mov v20.16b, v31.16b

mov v23.16b, v21.16b
mov v24.16b, v21.16b
mov v25.16b, v21.16b
mov v26.16b, v31.16b

mov v27.16b, v22.16b
mov v28.16b, v22.16b
mov v29.16b, v22.16b
mov v30.16b, v31.16b

st4 {v5.16b, v6.16b, v7.16b, v8.16b}, [x1], #64
st4 {v9.16b, v10.16b, v11.16b, v12.16b}, [x1], #64
st4 {v13.16b, v14.16b, v15.16b, v16.16b}, [x1], #64
st4 {v17.16b, v18.16b, v19.16b, v20.16b}, [x1], #64
st4 {v23.16b, v24.16b, v25.16b, v26.16b}, [x1], #64
st4 {v27.16b, v28.16b, v29.16b, v30.16b}, [x1], #64
b L12


L8:
cmp x2, #8
blt L4
ld1 {v0.16b, v1.16b, v2.16b, v3.16b}, [x0], #64
sub x2, x2, #8
mov v5.16b, v0.16b
mov v6.16b, v0.16b
mov v7.16b, v0.16b
mov v8.16b, v31.16b

mov v9.16b, v1.16b
mov v10.16b, v1.16b
mov v11.16b, v1.16b
mov v12.16b, v31.16b

mov v13.16b, v2.16b
mov v14.16b, v2.16b
mov v15.16b, v2.16b
mov v16.16b, v31.16b

mov v17.16b, v3.16b
mov v18.16b, v3.16b
mov v19.16b, v3.16b
mov v20.16b, v31.16b

st4 {v5.16b, v6.16b, v7.16b, v8.16b}, [x1], #64
st4 {v9.16b, v10.16b, v11.16b, v12.16b}, [x1], #64
st4 {v13.16b, v14.16b, v15.16b, v16.16b}, [x1], #64
st4 {v17.16b, v18.16b, v19.16b, v20.16b}, [x1], #64
b L8

L4:
cmp x2, #4
blt L2
ld1 {v0.16b, v1.16b}, [x0], #32
sub x2, x2, #4
mov v5.16b, v0.16b
mov v6.16b, v0.16b
mov v7.16b, v0.16b
mov v8.16b, v31.16b

mov v9.16b, v1.16b
mov v10.16b, v1.16b
mov v11.16b, v1.16b
mov v12.16b, v31.16b

st4 {v5.16b, v6.16b, v7.16b, v8.16b}, [x1], #64
st4 {v9.16b, v10.16b, v11.16b, v12.16b}, [x1], #64
b L4

L2:
cmp x2, #2
blt L1
ld1 {v0.16b}, [x0], #16
mov v5.16b, v0.16b
mov v6.16b, v0.16b
mov v7.16b, v0.16b
mov v8.16b, v31.16b
sub x2, x2, #2
st4 {v5.16b, v6.16b, v7.16b, v8.16b}, [x1], #64
b L2

L1:
cmp x2, #1
blt End
ld1 {v0.8b}, [x0], #8
mov v5.8b, v0.8b
mov v6.8b, v0.8b
mov v7.8b, v0.8b
mov v8.8b, v31.8b
st4 {v5.8b, v6.8b, v7.8b, v8.8b}, [x1], #32

End:
ldp d8,  d9,  [sp, #(16 * 3)]
ldp d10, d11, [sp, #(16 * 2)]
ldp d12, d13, [sp, #(16 * 1)]
ldp d14, d15, [sp], #(16 * 4)
ret
#endif
