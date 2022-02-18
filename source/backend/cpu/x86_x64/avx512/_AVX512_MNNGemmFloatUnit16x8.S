//
//  _AVX512_MNNGemmFloatUnit16x8.S
//  MNN
//
//  Created by MNN on 2020/05/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "../MNNAsmGlobal.h"
.text
.align 4

asm_function _AVX512_MNNGemmFloatUnit16x8
//void _AVX512_MNNGemmFloatUnit16x8(float* C, const float* A, const float* B, const size_t* parameter)

// SystemV Auto: rdi: C, rsi:A, rdx:B, rcx:parameter
// Microsoft x64 Auto: rcx:C, rdx:A, r8:B, r9:parameter
pushq   %rbp
movq    %rsp, %rbp
pushq   %rbx

#ifdef WIN32
pushq %rdi
pushq %rsi
pushq %r12
pushq %r13
pushq %r14
movq %rcx, %rdi
movq %rdx, %rsi
movq %r8, %rdx
movq %r9, %rcx
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
#endif

movq 40(%rcx), %r10 // bExtraStride
movq 24(%rcx), %r8 // cStride
movq 16(%rcx), %r9 // h
movq (%rcx), %r14 // aStride
movq 8(%rcx), %rcx // l

// h -> UP_DIV(h, 8)
addq $7, %r9
shrq $3, %r9

// zmm8-zmm31: Dst
// zmm0-zmm2: Src
// zmm3-zmm7: W

cmpq $0, %r9
je End

movq $0, %r12
movq %rsi, %r13
LoopDz:
    movq %rcx, %r11
    shlq $5, %r11
    movq %r13, %rsi
    subq $32,  %r11

    Init:
        vmovups (%rsi), %zmm0

        vbroadcastss (%rdx), %zmm4
        vbroadcastss 4(%rdx), %zmm5
        vbroadcastss 8(%rdx), %zmm6
        vbroadcastss 12(%rdx), %zmm7
        movq $32, %rbx
        vmulps %zmm4, %zmm0, %zmm8
        vmulps %zmm5, %zmm0, %zmm11
        vmulps %zmm6, %zmm0, %zmm14
        vmulps %zmm7, %zmm0, %zmm17

        vbroadcastss 16(%rdx), %zmm4
        vbroadcastss 20(%rdx), %zmm5
        vbroadcastss 24(%rdx), %zmm6
        vbroadcastss 28(%rdx), %zmm7

        vmulps %zmm4, %zmm0, %zmm20
        vmulps %zmm5, %zmm0, %zmm23
        vmulps %zmm6, %zmm0, %zmm26
        vmulps %zmm7, %zmm0, %zmm29

        addq %r14, %rsi

    cmpq %rbx, %r11
    jl Last
        vmovups (%rsi), %zmm1
        addq %r14, %rsi

    cmpq %rbx, %r11
    je LastLoop

    LoopSz:
        vmovups %zmm1, %zmm0

        vbroadcastss (%rdx, %rbx), %zmm4
        vbroadcastss 4(%rdx, %rbx), %zmm5
        vbroadcastss 8(%rdx, %rbx), %zmm6
        vbroadcastss 12(%rdx, %rbx), %zmm7

        vfmadd231ps %zmm4, %zmm1, %zmm8

        vfmadd231ps %zmm5, %zmm1, %zmm11

        vfmadd231ps %zmm6, %zmm1, %zmm14

        vfmadd231ps %zmm7, %zmm1, %zmm17

        vbroadcastss 16(%rdx, %rbx), %zmm4
        vbroadcastss 20(%rdx, %rbx), %zmm5
        vbroadcastss 24(%rdx, %rbx), %zmm6
        vbroadcastss 28(%rdx, %rbx), %zmm7

        vfmadd231ps %zmm4, %zmm0, %zmm20
        vmovups (%rsi), %zmm1
        vfmadd231ps %zmm5, %zmm0, %zmm23

        vfmadd231ps %zmm6, %zmm0, %zmm26

        vfmadd231ps %zmm7, %zmm0, %zmm29

        addq %r14, %rsi // cannot be eliminated by rbx
        addq $32, %rbx
        cmpq %rbx, %r11
        jne LoopSz

    LastLoop:
        vbroadcastss (%rdx, %rbx), %zmm4
        vbroadcastss 4(%rdx, %rbx), %zmm5
        vbroadcastss 8(%rdx, %rbx), %zmm6
        vbroadcastss 12(%rdx, %rbx), %zmm7

        vfmadd231ps %zmm4, %zmm1, %zmm8
        vfmadd231ps %zmm5, %zmm1, %zmm11
        vfmadd231ps %zmm6, %zmm1, %zmm14
        vfmadd231ps %zmm7, %zmm1, %zmm17

        vbroadcastss 16(%rdx, %rbx), %zmm4
        vbroadcastss 20(%rdx, %rbx), %zmm5
        vbroadcastss 24(%rdx, %rbx), %zmm6
        vbroadcastss 28(%rdx, %rbx), %zmm7

        vfmadd231ps %zmm4, %zmm1, %zmm20
        vfmadd231ps %zmm5, %zmm1, %zmm23
        vfmadd231ps %zmm6, %zmm1, %zmm26
        vfmadd231ps %zmm7, %zmm1, %zmm29

    Last:
    addq $32,  %r11
    addq %r11, %rdx

.macro TRANSPOSE_SAVE x0, x1, x2, x3, x4, x5, x6, x7
    vpunpckldq \x1, \x0, %zmm0
    vpunpckhdq \x1, \x0, %zmm1
    vpunpckldq \x3, \x2, %zmm2
    vpunpckhdq \x3, \x2, %zmm3
    vpunpckldq \x5, \x4, %zmm4
    vpunpckhdq \x5, \x4, %zmm5
    vpunpckldq \x7, \x6, %zmm6
    vpunpckhdq \x7, \x6, %zmm7

    vpunpcklqdq %zmm2, %zmm0, \x0
    vpunpckhqdq %zmm2, %zmm0, \x1
    vpunpcklqdq %zmm3, %zmm1, \x2
    vpunpckhqdq %zmm3, %zmm1, \x3

    vpunpcklqdq %zmm6, %zmm4, \x4
    vpunpckhqdq %zmm6, %zmm4, \x5
    vpunpcklqdq %zmm7, %zmm5, \x6
    vpunpckhqdq %zmm7, %zmm5, \x7

    VEXTRACTF64x4 $0, \x0, %ymm0
    VEXTRACTF64x4 $0, \x4, %ymm1
    VEXTRACTF64x4 $0, \x1, %ymm2
    VEXTRACTF64x4 $0, \x5, %ymm3

    vshufi64x2 $0, %ymm1, %ymm0, %ymm4
    vshufi64x2 $3, %ymm1, %ymm0, %ymm5
    vshufi64x2 $0, %ymm3, %ymm2, %ymm6
    vshufi64x2 $3, %ymm3, %ymm2, %ymm7

    vmovups %ymm4, (%r11)
    vmovups %ymm6, 64(%r11)
    vmovups %ymm5, 256(%r11)
    vmovups %ymm7, 320(%r11)

    VEXTRACTF64x4 $0, \x2, %ymm0
    VEXTRACTF64x4 $0, \x6, %ymm1
    VEXTRACTF64x4 $0, \x3, %ymm2
    VEXTRACTF64x4 $0, \x7, %ymm3

    vshufi64x2 $0, %ymm1, %ymm0, %ymm4
    vshufi64x2 $3, %ymm1, %ymm0, %ymm5
    vshufi64x2 $0, %ymm3, %ymm2, %ymm6
    vshufi64x2 $3, %ymm3, %ymm2, %ymm7

    vmovups %ymm4, 128(%r11)
    vmovups %ymm6, 192(%r11)
    vmovups %ymm5, 384(%r11)
    vmovups %ymm7, 448(%r11)

    addq $512, %r11

    VEXTRACTF64x4 $1, \x0, %ymm0
    VEXTRACTF64x4 $1, \x4, %ymm1
    VEXTRACTF64x4 $1, \x1, %ymm2
    VEXTRACTF64x4 $1, \x5, %ymm3

    vshufi64x2 $0, %ymm1, %ymm0, %ymm4
    vshufi64x2 $3, %ymm1, %ymm0, %ymm5
    vshufi64x2 $0, %ymm3, %ymm2, %ymm6
    vshufi64x2 $3, %ymm3, %ymm2, %ymm7

    vmovups %ymm4, (%r11)
    vmovups %ymm6, 64(%r11)
    vmovups %ymm5, 256(%r11)
    vmovups %ymm7, 320(%r11)

    VEXTRACTF64x4 $1, \x2, %ymm0
    VEXTRACTF64x4 $1, \x6, %ymm1
    VEXTRACTF64x4 $1, \x3, %ymm2
    VEXTRACTF64x4 $1, \x7, %ymm3

    vshufi64x2 $0, %ymm1, %ymm0, %ymm4
    vshufi64x2 $3, %ymm1, %ymm0, %ymm5
    vshufi64x2 $0, %ymm3, %ymm2, %ymm6
    vshufi64x2 $3, %ymm3, %ymm2, %ymm7

    vmovups %ymm4, 128(%r11)
    vmovups %ymm6, 192(%r11)
    vmovups %ymm5, 384(%r11)
    vmovups %ymm7, 448(%r11)

    addq $512, %r11

.endm
    movq %rdi, %r11

    TRANSPOSE_SAVE %zmm8, %zmm11, %zmm14, %zmm17, %zmm20, %zmm23, %zmm26, %zmm29

    cmpq $0, %r12
    je EndAdd8
    subq $32, %rdi
    addq %r8, %rdi
    jmp EndLoop
    EndAdd8:
    addq $32, %rdi

    EndLoop:

    addq %r10, %rdx

    addq $1, %r12
    andq $1, %r12

    subq $1, %r9
    cmpq $0, %r9
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
popq    %r14
popq    %r13
popq    %r12
popq    %rsi
popq    %rdi
#else
popq    %r14
popq    %r13
popq    %r12
#endif

popq    %rbx
popq    %rbp

retq

