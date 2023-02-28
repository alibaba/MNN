//
//  MNNMaxPoolInt8.s
//  ALL_BUILD
//
//  Created by MNN on 2023/1/9.
//

#ifdef __arm__
#ifndef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNMaxPoolInt8
// void MNNMaxPoolInt8(int8_t* dst, int8_t* src, size_t outputWidth, size_t inputWidth, size_t kernelx, size_t kernely,
//                  size_t stridesx);
// Auto load: r0: dst, r1: src, r2: outputWidth, r3: inputWidth
// r4: kernelx, r5: kernely, r7: stridesx

push {r4-r8, r10-r12, lr}

ldr r4, [sp, #36]
ldr r5, [sp, #40]
ldr r7, [sp, #44]

cmp r4, #0
ble END
cmp r5, #0
ble END

mov lr, #16
mul lr, lr, r3        // lr: 16*inputWidth

L4:
mov r11, #16
mul r11, r11, r7      // r11: 16*strides

cmp r2, #4
blt L1Loop

/*L4Loop */
L4Loop:
mov r8, #-0x80
vdup.8 q0, r8
vdup.8 q1, r8
vdup.8 q2, r8
vdup.8 q3, r8

mov r12, r5         

LoopY:
mov r8, r4              // kernelx
mov r3, r1
add r6, r3, r11          
add r7, r6, r11         
add r10, r7, r11       

LoopX:
vld1.8 {q8}, [r3]!
vld1.8 {q9}, [r6]!
vld1.8 {q10}, [r7]!
vld1.8 {q11}, [r10]!

vmax.s8 q0, q0, q8
vmax.s8 q1, q1, q9
vmax.s8 q2, q2, q10
vmax.s8 q3, q3, q11

sub r8, r8, #1          // r8: kernelx
cmp r8, #0
bne LoopX

EndLoopX:
add r1, r1, lr          // lr: 16*inputWidth

sub r12, r12, #1        // r12: kernely 
cmp r12, #0
bne LoopY

EndLoopY:
vst1.8 {q0, q1}, [r0]!
vst1.8 {q2, q3}, [r0]!

mul r3, r5, lr          // r5: kernely
sub r1, r1, r3

mov r3, #4
mul r3, r3, r11         // r3: 64* stridesx
add r1, r1, r3

sub r2, r2, #4          // r2: OutputWidth
cmp r2, #4
bge L4Loop
cmp r2, #0
beq END

/* L1Loop */
L1Loop:
mov r8, #-0x80
vdup.8 q0, r8
mov r12, r5            // r12: kernely
  
L1LoopY:  
mov r8, r4             // r8: kernelx
mov r6, r1             // r6: src

L1LoopX:
vld1.8 {q1}, [r6]!
vmax.s8 q0, q0, q1
sub r8, r8, #1
cmp r8, #0
bne L1LoopX

add r1, r1, lr
sub r12, r12, #1
cmp r12, #0
bne L1LoopY

vst1.8 {q0}, [r0]!

mul r6, r5, lr          // r5: kernely
sub r1, r1, r6
add r1, r1, r11        // r11: 16*stridesx

sub r2, r2, #1         // r2: OutputWidth
cmp r2, #0
bgt L1Loop
beq END

END:
pop {r4-r8, r10-r12, pc}

#endif
#endif