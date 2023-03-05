//
//  MNNAvgPoolInt8.s
//  ALL_BUILD
//
//  Created by MNN on 2023/1/9.
//

#ifdef __arm__
#ifndef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNAvgPoolInt8
// void MNNAvgPoolInt8(int8_t* dst, int8_t* src, size_t outputWidth, size_t inputWidth, size_t kernelx, size_t kernely,
//                  size_t stridesx, ssize_t paddingx, ssize_t factor);
// Auto load: r0: dst, r1: src, r2: outputWidth, r3: inputWidth
// r4: kernelx, r5: kernely, r7: stridesx, r8: paddingx, lr: factor

push {r4-r8, r10-r12, lr}

ldr r4, [sp, #36]
ldr r5, [sp, #40]
ldr r7, [sp, #44]
ldr r8, [sp, #48]
ldr lr, [sp, #52]       // lr: factor

vpush {q4-q6}
vdup.32 q4, lr

cmp r4, #0
ble END
cmp r5, #0
ble END

mov r11, #16
mul r11, r11, r7      // r11: 16*stridesx

mov lr, #16
mul lr, lr, r3        // lr: 16*inputWidth (16: channel pack)

L1:
/*L1Loop */
L1Loop:
vmov.i32 q0, #0
vmov.i32 q1, #0
vmov.i32 q2, #0
vmov.i32 q3, #0

mov r12, r5             // r5: kernely
mov r6, r1
Loop1Y:
mov r8, r4              // r4: kernelx
mov r3, r6    

Loop1X:
vld1.8 {q12}, [r3]!     //q12: int8x16
vmovl.s8 q13, d24       //q13: int16x8, d24: int8x8
vmovl.s8 q14, d25

vaddw.s16 q0, q0, d26   //d26: int16x4
vaddw.s16 q1, q1, d27
vaddw.s16 q2, q2, d28
vaddw.s16 q3, q3, d29

sub r8, r8, #1          // r8: kernelx
cmp r8, #0
bne Loop1X

EndLoop1X:
add r6, r6, lr          // lr: 16*inputWidth

sub r12, r12, #1        // r12: kernely 
cmp r12, #0
bne Loop1Y

EndLoop1Y:
vmul.s32 q0, q0, q4
vmul.s32 q1, q1, q4
vmul.s32 q2, q2, q4
vmul.s32 q3, q3, q4

vshr.s32 q0, q0, #24
vshr.s32 q1, q1, #24
vshr.s32 q2, q2, #24
vshr.s32 q3, q3, #24

vqmovn.s32 d0, q0
vqmovn.s32 d1, q1
vqmovn.s32 d2, q2
vqmovn.s32 d3, q3

vqmovn.s16 d0, q0
vqmovn.s16 d1, q1

vst1.8 {q0}, [r0]!

mov r3, #1               // #1: Computer only one width_point each time.
mul r3, r3, r11          // r3: 16* stridesx
add r1, r1, r3

sub r2, r2, #1           // r2: OutputWidth

cmp r2, #0
beq END
cmp r2, #1
bge L1Loop

END:
vpop {q4-q6}
pop {r4-r8, r10-r12, pc}

#endif
#endif