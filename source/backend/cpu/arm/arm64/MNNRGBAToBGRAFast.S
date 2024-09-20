#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5
// void MNNRGBAToBGRAFast(const unsigned char* source, unsigned char* dest, size_t count);
asm_function MNNRGBAToBGRAFast
// x0: source, x1: dest, x2: count
stp d14, d15, [sp, #(-16 * 4)]!
stp d12, d13, [sp, #(16 * 1)]
stp d10, d11, [sp, #(16 * 2)]
stp d8,  d9,  [sp, #(16 * 3)]

L10:
cmp x2, #10
blt L8
ld4 {v0.16b, v1.16b, v2.16b, v3.16b}, [x0], #64
ld4 {v4.16b, v5.16b, v6.16b, v7.16b}, [x0], #64
ld4 {v8.16b, v9.16b, v10.16b, v11.16b}, [x0], #64
ld4 {v12.16b, v13.16b, v14.16b, v15.16b}, [x0], #64
ld4 {v28.16b, v29.16b, v30.16b, v31.16b}, [x0], #64
sub x2, x2, #10

mov v16.16b, v2.16b
mov v17.16b, v1.16b
mov v18.16b, v0.16b
mov v19.16b, v3.16b

mov v20.16b, v6.16b
mov v21.16b, v5.16b
mov v22.16b, v4.16b
mov v23.16b, v7.16b

mov v24.16b, v10.16b
mov v25.16b, v9.16b
mov v26.16b, v8.16b
mov v27.16b, v11.16b

mov v0.16b, v14.16b
mov v1.16b, v13.16b
mov v2.16b, v12.16b
mov v3.16b, v15.16b

mov v4.16b, v30.16b
mov v5.16b, v29.16b
mov v6.16b, v28.16b
mov v7.16b, v31.16b

st4 {v16.16b, v17.16b, v18.16b, v19.16b}, [x1], #64
st4 {v20.16b, v21.16b, v22.16b, v23.16b}, [x1], #64
st4 {v24.16b, v25.16b, v26.16b, v27.16b}, [x1], #64
st4 {v0.16b, v1.16b, v2.16b, v3.16b}, [x1], #64
st4 {v4.16b, v5.16b, v6.16b, v7.16b}, [x1], #64

b L10


L8:
cmp x2, #8
blt L4
ld4 {v0.16b, v1.16b, v2.16b, v3.16b}, [x0], #64
ld4 {v4.16b, v5.16b, v6.16b, v7.16b}, [x0], #64
ld4 {v8.16b, v9.16b, v10.16b, v11.16b}, [x0], #64
ld4 {v12.16b, v13.16b, v14.16b, v15.16b}, [x0], #64
sub x2, x2, #8

mov v16.16b, v2.16b
mov v17.16b, v1.16b
mov v18.16b, v0.16b
mov v19.16b, v3.16b

mov v20.16b, v6.16b
mov v21.16b, v5.16b
mov v22.16b, v4.16b
mov v23.16b, v7.16b

mov v24.16b, v10.16b
mov v25.16b, v9.16b
mov v26.16b, v8.16b
mov v27.16b, v11.16b

mov v28.16b, v14.16b
mov v29.16b, v13.16b
mov v30.16b, v12.16b
mov v31.16b, v15.16b

st4 {v16.16b, v17.16b, v18.16b, v19.16b}, [x1], #64
st4 {v20.16b, v21.16b, v22.16b, v23.16b}, [x1], #64
st4 {v24.16b, v25.16b, v26.16b, v27.16b}, [x1], #64
st4 {v28.16b, v29.16b, v30.16b, v31.16b}, [x1], #64
b L8

L4:
cmp x2, #4
blt L2
ld4 {v0.16b, v1.16b, v2.16b, v3.16b}, [x0], #64
ld4 {v4.16b, v5.16b, v6.16b, v7.16b}, [x0], #64
sub x2, x2, #4

mov v16.16b, v2.16b
mov v17.16b, v1.16b
mov v18.16b, v0.16b
mov v19.16b, v3.16b

mov v20.16b, v6.16b
mov v21.16b, v5.16b
mov v22.16b, v4.16b
mov v23.16b, v7.16b

st4 {v16.16b, v17.16b, v18.16b, v19.16b}, [x1], #64
st4 {v20.16b, v21.16b, v22.16b, v23.16b}, [x1], #64
b L4

L2:
cmp x2, #2
blt L1
ld4 {v0.16b, v1.16b, v2.16b, v3.16b}, [x0], #64
sub x2, x2, #2

mov v16.16b, v2.16b
mov v17.16b, v1.16b
mov v18.16b, v0.16b
mov v19.16b, v3.16b

st4 {v16.16b, v17.16b, v18.16b, v19.16b}, [x1], #64
b L2

L1:
cmp x2, #1
blt End
ld4 {v0.8b, v1.8b, v2.8b, v3.8b}, [x0], #32

mov v16.8b, v2.8b
mov v17.8b, v1.8b
mov v18.8b, v0.8b
mov v19.8b, v3.8b

st4 {v16.8b, v17.8b, v18.8b, v19.8b}, [x1], #32

End:
ldp d8,  d9,  [sp, #(16 * 3)]
ldp d10, d11, [sp, #(16 * 2)]
ldp d12, d13, [sp, #(16 * 1)]
ldp d14, d15, [sp], #(16 * 4)
ret
#endif
