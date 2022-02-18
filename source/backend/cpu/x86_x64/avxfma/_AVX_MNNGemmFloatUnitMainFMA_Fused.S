//
//  _AVX_MNNGemmFloatUnitMainFMA_Fused.S
//  MNN
//
//  Created by MNN on 2021/05/18.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "../MNNAsmGlobal.h"
.text
.align 4

asm_function _AVX_MNNGemmFloatUnitMainFMA_Fused
//void _AVX_MNNGemmFloatUnitMainFMA_Fused(float* C, const float* A, const float* B, const size_t* parameter, const float* p, const float* bias)

// SystemV Auto: rdi: C, rsi:A, rdx:B, rcx:parameter, r8: postParameters, r9:bias
// Microsoft x64 Auto: rcx:C, rdx:A, r8:B, r9:parameter
pushq   %rbp
movq    %rsp, %rbp

#ifdef WIN32
pushq   %rdi
pushq   %rsi
movq    %rcx, %rdi
movq    %rdx, %rsi
movq    %r8, %rdx
movq    %r9, %rcx
#define push_registers_bytes ((3 + 1) * 8 + 32) // pushq + callq + shadow_space
movq (push_registers_bytes)(%rsp), %r8
movq (push_registers_bytes + 8)(%rsp), %r9
pushq   %r12
pushq   %r13
pushq   %r14
pushq   %r15
movq %r8, %r14
movq %r9, %r15
leaq (-1280)(%rsp), %rsp
vmovdqu %xmm6,  (128*0)(%rsp)
vmovdqu %xmm7,  (128*1)(%rsp)
vmovdqu %xmm8,  (128*2)(%rsp)
vmovdqu %xmm9,  (128*3)(%rsp)
vmovdqu %xmm10, (128*4)(%rsp)
vmovdqu %xmm11, (128*5)(%rsp)
vmovdqu %xmm12, (128*6)(%rsp)
vmovdqu %xmm13, (128*7)(%rsp)
vmovdqu %xmm14, (128*8)(%rsp)
vmovdqu %xmm15, (128*9)(%rsp)
#else
pushq   %r12
pushq   %r13
pushq   %r14
pushq   %r15
movq %r8, %r14
movq %r9, %r15
#endif

movq 40(%rcx), %r10 // bExtraStride
movq 24(%rcx), %r8 // cStride
movq 16(%rcx), %r9 // h
movq 8(%rcx), %rcx // l

// h -> UP_DIV(h, 4)
addq $3, %r9
shrq $2, %r9

// ymm4-ymm15: Dst
// ymm0-ymm2: Src
// ymm3: W

movq $0, %r12
cmpq $0, %r9
je End

movq %rsi, %r13
LoopDz:
    vbroadcastss (%r15), %ymm4
    vbroadcastss 4(%r15), %ymm7
    vbroadcastss 8(%r15), %ymm10
    vbroadcastss 12(%r15), %ymm13

    addq $16, %r15
    vmovups %ymm4, %ymm5
    vmovups %ymm4, %ymm6
    vmovups %ymm7, %ymm8
    vmovups %ymm7, %ymm9
    vmovups %ymm10, %ymm11
    vmovups %ymm10, %ymm12
    vmovups %ymm13, %ymm14
    vmovups %ymm13, %ymm15

    movq %rcx, %r11
    movq %r13, %rsi

    cmpq $2, %r11
    jl Remain
    
    LoopSz2:
        vmovups (%rsi), %ymm0
        vmovups 32(%rsi), %ymm1
        vmovups 64(%rsi), %ymm2

        vbroadcastss (%rdx), %ymm3
        vfmadd231ps %ymm3, %ymm0, %ymm4
        addq $96, %rsi
        vfmadd231ps %ymm3, %ymm1, %ymm5
        vfmadd231ps %ymm3, %ymm2, %ymm6

        vbroadcastss 4(%rdx), %ymm3
        vfmadd231ps %ymm3, %ymm0, %ymm7
        vfmadd231ps %ymm3, %ymm1, %ymm8
        prefetcht0 512(%rsi)
        vfmadd231ps %ymm3, %ymm2, %ymm9

        vbroadcastss 8(%rdx), %ymm3
        vfmadd231ps %ymm3, %ymm0, %ymm10
        vfmadd231ps %ymm3, %ymm1, %ymm11
        vfmadd231ps %ymm3, %ymm2, %ymm12
        vbroadcastss 12(%rdx), %ymm3
        vfmadd231ps %ymm3, %ymm0, %ymm13
        vfmadd231ps %ymm3, %ymm1, %ymm14
        addq $16, %rdx
        vmovups (%rsi), %ymm0
        vfmadd231ps %ymm3, %ymm2, %ymm15

        vmovups 32(%rsi), %ymm1
        vmovups 64(%rsi), %ymm2

        vbroadcastss (%rdx), %ymm3
        vfmadd231ps %ymm3, %ymm0, %ymm4
        vfmadd231ps %ymm3, %ymm1, %ymm5
        vfmadd231ps %ymm3, %ymm2, %ymm6

        vbroadcastss 4(%rdx), %ymm3
        vfmadd231ps %ymm3, %ymm0, %ymm7
        vfmadd231ps %ymm3, %ymm1, %ymm8
        vfmadd231ps %ymm3, %ymm2, %ymm9

        vbroadcastss 8(%rdx), %ymm3
        vfmadd231ps %ymm3, %ymm0, %ymm10
        vfmadd231ps %ymm3, %ymm1, %ymm11
        vfmadd231ps %ymm3, %ymm2, %ymm12
        vbroadcastss 12(%rdx), %ymm3
        prefetcht0 512(%rsi)
        vfmadd231ps %ymm3, %ymm0, %ymm13
        vfmadd231ps %ymm3, %ymm1, %ymm14
        vfmadd231ps %ymm3, %ymm2, %ymm15
        addq $16, %rdx
        addq $96, %rsi

        subq $2, %r11
        cmpq $2, %r11
        jge LoopSz2

    cmpq $0, %r11
    je Last

    Remain:
        vmovups (%rsi), %ymm0
        vmovups 32(%rsi), %ymm1
        vmovups 64(%rsi), %ymm2

        vbroadcastss (%rdx), %ymm3
        vfmadd231ps %ymm3, %ymm0, %ymm4
        vfmadd231ps %ymm3, %ymm1, %ymm5
        vfmadd231ps %ymm3, %ymm2, %ymm6

        vbroadcastss 4(%rdx), %ymm3
        vfmadd231ps %ymm3, %ymm0, %ymm7
        vfmadd231ps %ymm3, %ymm1, %ymm8
        vfmadd231ps %ymm3, %ymm2, %ymm9

        vbroadcastss 8(%rdx), %ymm3
        vfmadd231ps %ymm3, %ymm0, %ymm10
        vfmadd231ps %ymm3, %ymm1, %ymm11
        vfmadd231ps %ymm3, %ymm2, %ymm12
        vbroadcastss 12(%rdx), %ymm3
        prefetcht0 512(%rsi)
        vfmadd231ps %ymm3, %ymm0, %ymm13
        vfmadd231ps %ymm3, %ymm1, %ymm14
        vfmadd231ps %ymm3, %ymm2, %ymm15
        addq $16, %rdx
        addq $96, %rsi
        addq $1, %r11
    Last:

.macro TRANSPOSE_SAVE x0, x1, x2, x3
    vbroadcastss 8(%r14), %ymm0 // minV
    vbroadcastss 12(%r14), %ymm1 // maxV

    vmaxps \x0, %ymm0, \x0
    vmaxps \x1, %ymm0, \x1
    vmaxps \x2, %ymm0, \x2
    vmaxps \x3, %ymm0, \x3

    vminps \x0, %ymm1, \x0
    vminps \x1, %ymm1, \x1
    vminps \x2, %ymm1, \x2
    vminps \x3, %ymm1, \x3

    vpunpckldq \x1, \x0, %ymm0
    vpunpckldq \x3, \x2, %ymm2
    vpunpckhdq \x1, \x0, %ymm1
    vpunpckhdq \x3, \x2, %ymm3

    vpunpcklqdq %ymm2, %ymm0, \x0
    vpunpckhqdq %ymm2, %ymm0, \x1
    vpunpcklqdq %ymm3, %ymm1, \x2
    vpunpckhqdq %ymm3, %ymm1, \x3

    vextractf128 $0, \x0, %xmm0
    vextractf128 $0, \x1, %xmm1
    vextractf128 $0, \x2, %xmm2
    vextractf128 $0, \x3, %xmm3

    vmovups %xmm0, (%r11)
    vmovups %xmm1, 32(%r11)
    vmovups %xmm2, 64(%r11)
    vmovups %xmm3, 96(%r11)

    vextractf128 $1, \x0, %xmm0
    vextractf128 $1, \x1, %xmm1
    vextractf128 $1, \x2, %xmm2
    vextractf128 $1, \x3, %xmm3

    vmovups %xmm0, 128(%r11)
    vmovups %xmm1, 160(%r11)
    vmovups %xmm2, 192(%r11)
    vmovups %xmm3, 224(%r11)

.endm
    movq %rdi, %r11

    TRANSPOSE_SAVE %ymm4, %ymm7, %ymm10, %ymm13

    addq $256, %r11

    TRANSPOSE_SAVE %ymm5, %ymm8, %ymm11, %ymm14

    addq $256, %r11
    TRANSPOSE_SAVE %ymm6, %ymm9, %ymm12, %ymm15

    testq %r12, %r12
    je EndAdd4
    subq $16, %rdi
    addq %r8, %rdi
    jmp EndLoop
    EndAdd4:
    addq $16, %rdi
    
    EndLoop:

    addq %r10, %rdx

    addq $1, %r12
    andq $1, %r12

    subq $1, %r9
    testq %r9, %r9
    jne LoopDz


End:

#ifdef WIN32
vmovdqu (128*0)(%rsp), %xmm6
vmovdqu (128*1)(%rsp), %xmm7
vmovdqu (128*2)(%rsp), %xmm8
vmovdqu (128*3)(%rsp), %xmm9
vmovdqu (128*4)(%rsp), %xmm10
vmovdqu (128*5)(%rsp), %xmm11
vmovdqu (128*6)(%rsp), %xmm12
vmovdqu (128*7)(%rsp), %xmm13
vmovdqu (128*8)(%rsp), %xmm14
vmovdqu (128*9)(%rsp), %xmm15
leaq (1280)(%rsp), %rsp
popq    %r15
popq    %r14
popq    %r13
popq    %r12
popq    %rsi
popq    %rdi
#else
popq    %r15
popq    %r14
popq    %r13
popq    %r12
#endif
popq    %rbp

retq

