//
//  MNNAvgPoolInt8.s
//  ALL_BUILD
//
//  Created by MNN on 2023/1/9.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__
#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNAvgPoolInt8
// void MNNAvgPoolInt8(int8_t* dst, int8_t* src, size_t outputWidth, size_t inputWidth, size_t kernelx, size_t kernely,
//                  size_t stridesx, ssize_t paddingx, ssize_t factor);
// Auto load: x0: dst, x1: src, x2: outputWidth, x3: inputWidth, x4: kernelx, x5: kernely, x6: stridesx, x7: paddingx

// Load from sp:
// w8: factor
ldr x8, [sp, #0]
dup v24.4s, w8

sub sp, sp, #32
str x19, [sp, #0]
str x20, [sp, #8]
str x21, [sp, #16]

cmp x4, #0
ble END
cmp x5, #0
ble END

mov x8, #16
mul x8, x8, x6      // x8: 16*stridesx

mov x9, #16
mul x9, x9, x3        // x9: 16*inputWidth
cmp x2, #2
blt L1Loop


/*L2Loop */
L2Loop:
movi v0.4s, #0
movi v1.4s, #0
movi v2.4s, #0
movi v3.4s, #0

movi v7.4s, #0
movi v8.4s, #0
movi v9.4s, #0
movi v10.4s, #0

mov x10, x5             // x5: kernely
mov x13, x1
add x14, x13, x8

Loop2Y:
mov x7, x4              // x4: kernelx
mov x3, x13
mov x11, x14

Loop2X:
ld1 {v4.4s}, [x3], #16
ld1 {v11.4s}, [x11], #16

sxtl v5.8h, v4.8b
sxtl2 v6.8h, v4.16b
sxtl v12.8h, v11.8b
sxtl2 v13.8h, v11.16b

saddw v0.4s, v0.4s, v5.4h
saddw2 v1.4s, v1.4s, v5.8h
saddw v2.4s, v2.4s, v6.4h
saddw2 v3.4s, v3.4s, v6.8h
saddw v7.4s, v7.4s, v12.4h
saddw2 v8.4s, v8.4s, v12.8h
saddw v9.4s, v9.4s, v13.4h
saddw2 v10.4s, v10.4s, v13.8h

sub x7, x7, #1          // x7: kernelx
cmp x7, #0
bne Loop2X

EndLoop2X:
add x13, x13, x9
add x14, x14, x9

sub x10, x10, #1        // x10: kernely 
cmp x10, #0
bne Loop2Y

EndLoop2Y:
mul v0.4s, v0.4s, v24.4s
mul v1.4s, v1.4s, v24.4s
mul v2.4s, v2.4s, v24.4s
mul v3.4s, v3.4s, v24.4s
mul v7.4s, v7.4s, v24.4s
mul v8.4s, v8.4s, v24.4s
mul v9.4s, v9.4s, v24.4s
mul v10.4s, v10.4s, v24.4s

sshr v0.4s, v0.4s, #24       // shift right
sshr v1.4s, v1.4s, #24
sshr v2.4s, v2.4s, #24
sshr v3.4s, v3.4s, #24
sshr v7.4s, v7.4s, #24
sshr v8.4s, v8.4s, #24
sshr v9.4s, v9.4s, #24
sshr v10.4s, v10.4s, #24

sqxtn v0.4h, v0.4s
sqxtn2 v0.8h, v1.4s
sqxtn v1.4h, v2.4s
sqxtn2 v1.8h, v3.4s
sqxtn v2.4h, v7.4s
sqxtn2 v2.8h, v8.4s
sqxtn v3.4h, v9.4s
sqxtn2 v3.8h, v10.4s

sqxtn v0.8b, v0.8h
sqxtn2 v0.16b, v1.8h
sqxtn v1.8b, v2.8h
sqxtn2 v1.16b, v3.8h

st1 {v0.16b, v1.16b}, [x0], #32

mov x3, #2
mul x3, x3, x8         // x3: 2* 16* strides
add x1, x1, x3       
       
sub x2, x2, #2          // x2: OutputWidth
cmp x2, #2
bge L2Loop

cmp x2, #0
beq END

/*L1Loop */
L1Loop:
movi v0.4s, #0
movi v1.4s, #0
movi v2.4s, #0
movi v3.4s, #0

mov x10, x5             // x5: kernely
mov x13, x1

Loop1Y:
mov x7, x4              // x4: kernelx
mov x3, x13

Loop1X:
ld1 {v4.4s}, [x3], #16
sxtl v5.8h, v4.8b
sxtl2 v6.8h, v4.16b

saddw v0.4s, v0.4s, v5.4h
saddw2 v1.4s, v1.4s, v5.8h
saddw v2.4s, v2.4s, v6.4h
saddw2 v3.4s, v3.4s, v6.8h

sub x7, x7, #1          // x7: kernelx
cmp x7, #0
bne Loop1X

EndLoop1X:
add x13, x13, x9

sub x10, x10, #1        // x10: kernely 
cmp x10, #0
bne Loop1Y

EndLoop1Y:
mul v0.4s, v0.4s, v24.4s
mul v1.4s, v1.4s, v24.4s
mul v2.4s, v2.4s, v24.4s
mul v3.4s, v3.4s, v24.4s

sshr v0.4s, v0.4s, #24       // shift right
sshr v1.4s, v1.4s, #24
sshr v2.4s, v2.4s, #24
sshr v3.4s, v3.4s, #24

sqxtn v0.4h, v0.4s
sqxtn2 v0.8h, v1.4s
sqxtn v1.4h, v2.4s
sqxtn2 v1.8h, v3.4s

sqxtn v0.8b, v0.8h
sqxtn v1.8b, v1.8h

st1 {v0.8b, v1.8b}, [x0], #16

mov x3, #1
mul x3, x3, x8         // x3: 16* strides
add x1, x1, x3       
       
sub x2, x2, #1          // x2: OutputWidth
cmp x2, #1
bge L1Loop

cmp x2, #0
beq END

END:
ldr x19, [sp, #0]
ldr x20, [sp, #8]
ldr x21, [sp, #16]
add sp, sp, #32

ret

#endif