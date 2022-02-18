//
//  _AVX512_MNNGemmFloatUnit48x8Fused.S
//  MNN
//
//  Created by MNN on 2020/05/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "../MNNAsmGlobal.h"
.text
.align 4

asm_function _AVX512_MNNGemmFloatUnit48x8Fused
//void _AVX512_MNNGemmFloatUnit48x8Fused(float* C, const float* A, const float* B, const size_t* parameter, const float* p, const float* bias)

// SystemV Auto: rdi: C, rsi:A, rdx:B, rcx:parameter, r8: postParameters, r9:bias

// Microsoft x64 Auto: rcx:C, rdx:A, r8:B, r9:parameter
// stack: postParameters, bias
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
movq (push_registers_bytes)(%rsp), %r8 // postParameters
movq (push_registers_bytes + 8)(%rsp), %r9 // bias
pushq   %rbx
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
pushq   %rbx
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
        vbroadcastss (%r15), %zmm8
        vbroadcastss 4(%r15), %zmm11
        vbroadcastss 8(%r15), %zmm14
        vbroadcastss 12(%r15), %zmm17
        vbroadcastss 16(%r15), %zmm20
        vbroadcastss 20(%r15), %zmm23
        vbroadcastss 24(%r15), %zmm26
        vbroadcastss 28(%r15), %zmm29

        vmovups %zmm8, %zmm9
        vmovups %zmm8, %zmm10
        vmovups %zmm11, %zmm12
        vmovups %zmm11, %zmm13
        vmovups %zmm14, %zmm15
        vmovups %zmm14, %zmm16
        vmovups %zmm17, %zmm18
        vmovups %zmm17, %zmm19
        vmovups %zmm20, %zmm21
        vmovups %zmm20, %zmm22
        vmovups %zmm23, %zmm24
        vmovups %zmm23, %zmm25
        vmovups %zmm26, %zmm27
        vmovups %zmm26, %zmm28
        vmovups %zmm29, %zmm30
        vmovups %zmm29, %zmm31

        addq $32, %r15
        movq $32, %rbx

        vmovups (%rsi), %zmm0
        vmovups 64(%rsi), %zmm1
        vmovups 128(%rsi), %zmm2
        vbroadcastss (%rdx), %zmm4
        vbroadcastss 4(%rdx), %zmm5
        vbroadcastss 8(%rdx), %zmm6
        vbroadcastss 12(%rdx), %zmm7

        vfmadd231ps %zmm4, %zmm0, %zmm8
        vfmadd231ps %zmm4, %zmm1, %zmm9
        vfmadd231ps %zmm4, %zmm2, %zmm10
        vfmadd231ps %zmm5, %zmm0, %zmm11
        vfmadd231ps %zmm5, %zmm1, %zmm12
        vfmadd231ps %zmm5, %zmm2, %zmm13
        vfmadd231ps %zmm6, %zmm0, %zmm14
        vfmadd231ps %zmm6, %zmm1, %zmm15
        vfmadd231ps %zmm6, %zmm2, %zmm16
        vfmadd231ps %zmm7, %zmm0, %zmm17
        vfmadd231ps %zmm7, %zmm1, %zmm18
        vfmadd231ps %zmm7, %zmm2, %zmm19

        vbroadcastss 16(%rdx), %zmm4
        vbroadcastss 20(%rdx), %zmm5
        vbroadcastss 24(%rdx), %zmm6
        vbroadcastss 28(%rdx), %zmm7

        vfmadd231ps %zmm4, %zmm0, %zmm20
        vfmadd231ps %zmm4, %zmm1, %zmm21
        vfmadd231ps %zmm4, %zmm2, %zmm22
        vfmadd231ps %zmm5, %zmm0, %zmm23
        vfmadd231ps %zmm5, %zmm1, %zmm24
        vfmadd231ps %zmm5, %zmm2, %zmm25
        vfmadd231ps %zmm6, %zmm0, %zmm26
        vfmadd231ps %zmm6, %zmm1, %zmm27
        vfmadd231ps %zmm6, %zmm2, %zmm28
        vfmadd231ps %zmm7, %zmm0, %zmm29
        vfmadd231ps %zmm7, %zmm1, %zmm30
        vfmadd231ps %zmm7, %zmm2, %zmm31

        addq $192, %rsi
        //addq $32, %rdx

    cmpq %rbx, %r11
    jl Last

        vmovups (%rsi), %zmm3
        vmovups 64(%rsi), %zmm4
        vmovups 128(%rsi), %zmm5
        addq $192, %rsi

    cmpq %rbx, %r11
    je LastLoop

    LoopSz:
        vmovups %zmm3, %zmm0
        vmovups %zmm4, %zmm1
        vmovups %zmm5, %zmm2
        vbroadcastss (%rdx, %rbx), %zmm4
        vbroadcastss 4(%rdx, %rbx), %zmm5
        vbroadcastss 8(%rdx, %rbx), %zmm6
        vbroadcastss 12(%rdx, %rbx), %zmm7

        vfmadd231ps %zmm4, %zmm3, %zmm8
        vfmadd231ps %zmm4, %zmm1, %zmm9
        vfmadd231ps %zmm4, %zmm2, %zmm10

        vfmadd231ps %zmm5, %zmm3, %zmm11
        vfmadd231ps %zmm5, %zmm1, %zmm12
        vfmadd231ps %zmm5, %zmm2, %zmm13

        vfmadd231ps %zmm6, %zmm3, %zmm14
        vfmadd231ps %zmm6, %zmm1, %zmm15
        vfmadd231ps %zmm6, %zmm2, %zmm16

        vfmadd231ps %zmm7, %zmm3, %zmm17
        vfmadd231ps %zmm7, %zmm1, %zmm18
        vfmadd231ps %zmm7, %zmm2, %zmm19

        vbroadcastss 16(%rdx, %rbx), %zmm4
        vbroadcastss 20(%rdx, %rbx), %zmm5
        vbroadcastss 24(%rdx, %rbx), %zmm6
        vbroadcastss 28(%rdx, %rbx), %zmm7

        vfmadd231ps %zmm4, %zmm0, %zmm20
        vfmadd231ps %zmm4, %zmm1, %zmm21
        vfmadd231ps %zmm4, %zmm2, %zmm22
        vmovups (%rsi), %zmm3
        // prefetcht0 (192)(%rsi) // to be observed

        vfmadd231ps %zmm5, %zmm0, %zmm23
        vfmadd231ps %zmm5, %zmm1, %zmm24
        vfmadd231ps %zmm5, %zmm2, %zmm25
        vmovups 64(%rsi), %zmm4
        // prefetcht0 (256)(%rsi)
        vfmadd231ps %zmm6, %zmm0, %zmm26
        vfmadd231ps %zmm6, %zmm1, %zmm27
        vfmadd231ps %zmm6, %zmm2, %zmm28

        vmovups 128(%rsi), %zmm5
        // prefetcht0 (320)(%rsi)
        vfmadd231ps %zmm7, %zmm0, %zmm29
        vfmadd231ps %zmm7, %zmm1, %zmm30
        vfmadd231ps %zmm7, %zmm2, %zmm31

        addq $192, %rsi // cannot be eliminated by rbx
        addq $32, %rbx
        cmpq %rbx, %r11

        jne LoopSz

    LastLoop:
        vbroadcastss (%rdx, %rbx), %zmm1
        vbroadcastss 4(%rdx, %rbx), %zmm2
        vbroadcastss 8(%rdx, %rbx), %zmm6
        vbroadcastss 12(%rdx, %rbx), %zmm7

        vfmadd231ps %zmm1, %zmm3, %zmm8
        vfmadd231ps %zmm1, %zmm4, %zmm9
        vfmadd231ps %zmm1, %zmm5, %zmm10

        vfmadd231ps %zmm2, %zmm3, %zmm11
        vfmadd231ps %zmm2, %zmm4, %zmm12
        vfmadd231ps %zmm2, %zmm5, %zmm13

        vfmadd231ps %zmm6, %zmm3, %zmm14
        vfmadd231ps %zmm6, %zmm4, %zmm15
        vfmadd231ps %zmm6, %zmm5, %zmm16

        vfmadd231ps %zmm7, %zmm3, %zmm17
        vfmadd231ps %zmm7, %zmm4, %zmm18
        vfmadd231ps %zmm7, %zmm5, %zmm19

        vbroadcastss 16(%rdx, %rbx), %zmm1
        vbroadcastss 20(%rdx, %rbx), %zmm2
        vbroadcastss 24(%rdx, %rbx), %zmm6
        vbroadcastss 28(%rdx, %rbx), %zmm7

        vfmadd231ps %zmm1, %zmm3, %zmm20
        vfmadd231ps %zmm1, %zmm4, %zmm21
        vfmadd231ps %zmm1, %zmm5, %zmm22

        vfmadd231ps %zmm2, %zmm3, %zmm23
        vfmadd231ps %zmm2, %zmm4, %zmm24
        vfmadd231ps %zmm2, %zmm5, %zmm25

        vfmadd231ps %zmm6, %zmm3, %zmm26
        vfmadd231ps %zmm6, %zmm4, %zmm27
        vfmadd231ps %zmm6, %zmm5, %zmm28

        vfmadd231ps %zmm7, %zmm3, %zmm29
        vfmadd231ps %zmm7, %zmm4, %zmm30
        vfmadd231ps %zmm7, %zmm5, %zmm31

    Last:
    addq $32,  %r11
    vbroadcastss 8(%r14), %zmm0 // minV
    vbroadcastss 12(%r14), %zmm1 // maxV
    addq %r11, %rdx

    vmaxps %zmm8, %zmm0, %zmm8
    vmaxps %zmm9, %zmm0, %zmm9
    vmaxps %zmm10, %zmm0, %zmm10
    vmaxps %zmm11, %zmm0, %zmm11
    vmaxps %zmm12, %zmm0, %zmm12
    vmaxps %zmm13, %zmm0, %zmm13
    vmaxps %zmm14, %zmm0, %zmm14
    vmaxps %zmm15, %zmm0, %zmm15
    vmaxps %zmm16, %zmm0, %zmm16
    vmaxps %zmm17, %zmm0, %zmm17
    vmaxps %zmm18, %zmm0, %zmm18
    vmaxps %zmm19, %zmm0, %zmm19
    vmaxps %zmm20, %zmm0, %zmm20
    vmaxps %zmm21, %zmm0, %zmm21
    vmaxps %zmm22, %zmm0, %zmm22
    vmaxps %zmm23, %zmm0, %zmm23
    vmaxps %zmm24, %zmm0, %zmm24
    vmaxps %zmm25, %zmm0, %zmm25
    vmaxps %zmm26, %zmm0, %zmm26
    vmaxps %zmm27, %zmm0, %zmm27
    vmaxps %zmm28, %zmm0, %zmm28
    vmaxps %zmm29, %zmm0, %zmm29
    vmaxps %zmm30, %zmm0, %zmm30
    vmaxps %zmm31, %zmm0, %zmm31

    vminps %zmm8,  %zmm1, %zmm8
    vminps %zmm9,  %zmm1, %zmm9
    vminps %zmm10, %zmm1, %zmm10
    vminps %zmm11, %zmm1, %zmm11
    vminps %zmm12, %zmm1, %zmm12
    vminps %zmm13, %zmm1, %zmm13
    vminps %zmm14, %zmm1, %zmm14
    vminps %zmm15, %zmm1, %zmm15
    vminps %zmm16, %zmm1, %zmm16
    vminps %zmm17, %zmm1, %zmm17
    vminps %zmm18, %zmm1, %zmm18
    vminps %zmm19, %zmm1, %zmm19
    vminps %zmm20, %zmm1, %zmm20
    vminps %zmm21, %zmm1, %zmm21
    vminps %zmm22, %zmm1, %zmm22
    vminps %zmm23, %zmm1, %zmm23
    vminps %zmm24, %zmm1, %zmm24
    vminps %zmm25, %zmm1, %zmm25
    vminps %zmm26, %zmm1, %zmm26
    vminps %zmm27, %zmm1, %zmm27
    vminps %zmm28, %zmm1, %zmm28
    vminps %zmm29, %zmm1, %zmm29
    vminps %zmm30, %zmm1, %zmm30
    vminps %zmm31, %zmm1, %zmm31

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

    TRANSPOSE_SAVE %zmm9, %zmm12, %zmm15, %zmm18, %zmm21, %zmm24, %zmm27, %zmm30

    TRANSPOSE_SAVE %zmm10, %zmm13, %zmm16, %zmm19, %zmm22, %zmm25, %zmm28, %zmm31

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
popq    %r15
popq    %r14
popq    %r13
popq    %r12
popq    %rbx
popq    %rsi
popq    %rdi
#else
popq    %r15
popq    %r14
popq    %r13
popq    %r12
popq    %rbx
#endif

popq    %rbp
retq

