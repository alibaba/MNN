//
//  _AVX_MNNGemmFloatUnitMainFMA6x16.S
//  MNN
//
//  Created by MNN on 2021/05/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "../MNNAsmGlobal.h"
.text
.align 4

asm_function _AVX_MNNGemmFloatUnitMainFMA6x16
//void _AVX_MNNGemmFloatUnitMainFMA(float* C, const float* A, const float* B, const size_t* parameter, size_t hC4)

// SystemV Auto: rdi: C, rsi:A, rdx:B, rcx:parameter, r8: hC4
// Microsoft x64 Auto: rcx:C, rdx:A, r8:B, r9:parameter
pushq   %rbp
movq    %rsp, %rbp

#ifdef WIN32
#define push_registers_bytes ((1 + 1) * 8 + 32)
movq (push_registers_bytes)(%rsp), %r10
pushq %rdi
pushq %rsi
pushq %r12
pushq %r13
movq %rcx, %rdi
movq %rdx, %rsi
movq %r8, %rdx
movq %r9, %rcx
movq %r10, %r9
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
movq %r8, %r9
#endif

movq 40(%rcx), %r10 // bExtraStride
movq 24(%rcx), %r8 // cStride
movq 8(%rcx), %rcx // l

// ymm4-ymm15: Dst
// ymm0-ymm2: Src
// ymm3: W

cmpq $0, %r9
je End

movq %rsi, %r13
LoopDz:
    vzeroall
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
    vpunpckldq \x1, \x0, %ymm0
    vpunpckldq \x3, \x2, %ymm2
    vpunpckhdq \x1, \x0, %ymm1
    vpunpckhdq \x3, \x2, %ymm3

    vpunpcklqdq %ymm2, %ymm0, \x0
    vpunpckhqdq %ymm2, %ymm0, \x1
    vpunpcklqdq %ymm3, %ymm1, \x2
    vpunpckhqdq %ymm3, %ymm1, \x3

    // 32 = 0 + 16 * 2: frist 128 x0_lo, second 128 x1_lo
    // 49 = 1 + 16 * 3: frist 128 x0_hi, second 128 x1_hi
    vperm2f128 $32, \x1, \x0, %ymm0
    vperm2f128 $49, \x1, \x0, %ymm2
    vperm2f128 $32, \x3, \x2, %ymm1
    vperm2f128 $49, \x3, \x2, %ymm3

    vmovups %ymm0, (%r11)
    vmovups %ymm1, 32(%r11)
    vmovups %ymm2, 64(%r11)
    vmovups %ymm3, 96(%r11)

.endm
    movq %rdi, %r11

    TRANSPOSE_SAVE %ymm4, %ymm7, %ymm10, %ymm13

    addq $128, %r11

    TRANSPOSE_SAVE %ymm5, %ymm8, %ymm11, %ymm14

    addq $128, %r11
    TRANSPOSE_SAVE %ymm6, %ymm9, %ymm12, %ymm15


    addq %r8, %rdi
    addq %r10, %rdx

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
popq    %r13
popq    %r12
popq    %rsi
popq    %rdi
popq    %rbp
#else
popq    %r13
popq    %r12
popq    %rbp
#endif

retq

