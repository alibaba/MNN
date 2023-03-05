//
//  MNNMaxPoolInt8.s
//  ALL_BUILD
//
//  Created by MNN on 2023/1/9.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__
#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNMaxPoolInt8
// void MNNMaxPoolInt8(int8_t* dst, int8_t* src, size_t outputWidth, size_t inputWidth, size_t kernelx, size_t kernely,
//                  size_t stridesx);
// Auto load: x0: dst, x1: src, x2: outputWidth, x3: inputWidth, x4: kernelx, x5: kernely, x6: stridesx

sub sp, sp, #32
str x19, [sp, #0]
str x20, [sp, #8]
str x21, [sp, #16]

cmp x4, #0
ble END
cmp x5, #0
ble END

mov x9, #16
mul x9, x9, x3        // x9: 16*inputWidth

mov x8, #16
mul x8, x8, x6      // x8: 16*stridesx


cmp x2, #4
blt L1Loop

cmp x2, #8
blt L4Loop

/*L8Loop */
L8Loop:
mov x14, #4
mul x14, x14, x8     // x14: 64*stridesx
mov w7, #-0x80
dup v0.16b, w7
dup v1.16b, w7
dup v2.16b, w7
dup v3.16b, w7
dup v16.16b, w7
dup v17.16b, w7
dup v18.16b, w7
dup v19.16b, w7

mov x10, x5             // x5: kernely
mov x13, x1

Loop8Y:
mov x7, x4              // x4: kernelx
mov x3, x13
add x15, x13, x14
add x11, x3, x8
add x19, x15, x8
add x6, x11, x8
add x20, x19, x8
add x12, x6, x8
add x21, x20, x8


Loop8X:
ld1 {v4.16b}, [x3], #16
ld1 {v5.16b}, [x11], #16
ld1 {v6.16b}, [x6], #16
ld1 {v7.16b}, [x12], #16
ld1 {v20.16b}, [x15], #16
ld1 {v21.16b}, [x19], #16
ld1 {v22.16b}, [x20], #16
ld1 {v23.16b}, [x21], #16

smax v0.16b, v0.16b, v4.16b
smax v1.16b, v1.16b, v5.16b
smax v2.16b, v2.16b, v6.16b
smax v3.16b, v3.16b, v7.16b
smax v16.16b, v16.16b, v20.16b
smax v17.16b, v17.16b, v21.16b
smax v18.16b, v18.16b, v22.16b
smax v19.16b, v19.16b, v23.16b

sub x7, x7, #1          // x7: kernelx
cmp x7, #0
bne Loop8X

EndLoop8X:
add x13, x13, x9

sub x10, x10, #1        // x10: kernely 
cmp x10, #0
bne Loop8Y

EndLoop8Y:
st1 {v0.16b, v1.16b, v2.16b, v3.16b}, [x0], #64
st1 {v16.16b, v17.16b, v18.16b, v19.16b}, [x0], #64

mov x3, #8
mul x3, x3, x8         // x3: 128* strides
add x1, x1, x3       
       
sub x2, x2, #8          // x2: OutputWidth
cmp x2, #8
bge L8Loop
cmp x2, #4
bge L4Loop
cmp x2, #1
bge L1Loop
cmp x2, #0
beq END


/*L4Loop */
L4Loop:
mov w7, #-0x80
dup v0.16b, w7
dup v1.16b, w7
dup v2.16b, w7
dup v3.16b, w7

mov x10, x5             // x5: kernely
mov x13, x1

Loop4Y:
mov x7, x4              // x4: kernelx
mov x3, x13
add x11, x3, x8          
add x6, x11, x8         
add x12, x6, x8       

Loop4X:
ld1 {v4.16b}, [x3], #16
ld1 {v5.16b}, [x11], #16
ld1 {v6.16b}, [x6], #16
ld1 {v7.16b}, [x12], #16

smax v0.16b, v0.16b, v4.16b
smax v1.16b, v1.16b, v5.16b
smax v2.16b, v2.16b, v6.16b
smax v3.16b, v3.16b, v7.16b

sub x7, x7, #1          // x7: kernelx
cmp x7, #0
bne Loop4X

EndLoop4X:
add x13, x13, x9

sub x10, x10, #1        // x10: kernely 
cmp x10, #0
bne Loop4Y

EndLoop4Y:
st1 {v0.16b, v1.16b, v2.16b, v3.16b}, [x0], #64

mov x3, #4
mul x3, x3, x8         // x3: 64* strides
add x1, x1, x3       
       
sub x2, x2, #4          // x2: OutputWidth
cmp x2, #4
bge L4Loop
cmp x2, #0
beq END

/* L1Loop */
L1Loop:
mov w7, #-0x80
dup v0.16b, w7
mov x10, x5              // x10: kernely
mov x12, x1              // x1: src

L1LoopY:   
mov x7, x4               // x7: kernelx
mov x3, x12              // x12: src

L1LoopX:
ld1 {v1.16b}, [x3], #16
smax v0.16b, v0.16b, v1.16b
sub x7, x7, #1
cmp x7, #0
bne L1LoopX

add x12, x12, x9
sub x10, x10, #1
cmp x10, #0
bne L1LoopY

st1 {v0.16b}, [x0], #16

add x1, x1, x8
sub x2, x2, #1         // x2: OutputWidth
cmp x2, #0
bgt L1Loop
beq END


END:
ldr x19, [sp, #0]
ldr x20, [sp, #8]
ldr x21, [sp, #16]
add sp, sp, #32

ret

#endif