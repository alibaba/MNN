//
//  _AVX_MNNPackedSparseMatMulEpx4NFMA_ASM.S
//  MNN
//
//  Created by MNN on 2021/07/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "../MNNAsmGlobal.h"
.text
.align 4

asm_function _AVX_MNNPackedSparseMatMulEpx4NFMA_ASM

// struct SparseMatMulParas
// {
//     float* C;
//     const float* A;
//     const float* B;
//     unsigned int* NNZMap;
//     int* dataOffsetMap;
// };

// void _AVX_MNNPackedSparseMatMulEpx4NFMA_ASM(SparseMatMulParas* packedParas, const float* bias, const size_t* parameter, const float* postParameters);


// SystemV Auto: rdi: packedParas, rsi: bias, rdx: parameter, rcx: postParameters
// Microsoft x64 Auto: rcx:packedParas, rdx:bias, r8:parameter, r9:postParameters

// all callee save regs:
// %rbx, %rbp, %r12~%r15
// unused para regs: %r8, %r9
// can use regs: %r8~%r15, %rdi, %rsi, %rdx, %rcx, %rbx, %rax
pushq   %rbp
movq    %rsp, %rbp

#ifdef WIN32
pushq   %rdi
pushq   %rsi
movq    %rcx, %rdi
movq    %rdx, %rsi
movq    %r8, %rdx
movq    %r9, %rcx
pushq   %rbx
pushq   %r12
pushq   %r13
pushq   %r14
pushq   %r15
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
pushq   %rax
pushq   %rbx
pushq   %r8
pushq   %r9
pushq   %r12
pushq   %r13
pushq   %r14
pushq   %r15
#endif

movq (%rdi),    %rax        // %rax C
movq 8(%rdi),   %rbx        // %rbx A
movq 16(%rdi),  %r8         // %r8 B
movq 24(%rdi),  %r9         // %r9 NNZMap
movq 32(%rdi),  %r10        // %r10 dataOffsetMap
movq 16(%rdx),  %r11        // %r11 h

//  %rax: C, %rbx: A, %r8: B, %rsi: bias, %rcx: postParameters
//  %r9: NNZMap, %r10: dataOffsetMap, %r11: h 
//  free: %r12~%r15, %rdx

// %ymm4 ~ %ymm15:cVecs
// %ymm0 ~ %ymm2: aVecs
// %ymm3: bVecs
// %ymm0 ~ %ymm3 will be resued for other actions

.macro TRANSPOSE_SAVE x0, x1, x2, x3
    vbroadcastss 8(%rcx), %ymm0 // minV
    vbroadcastss 12(%rcx), %ymm1 // maxV

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

    vmovups %xmm0, (%r15)
    vmovups %xmm1, 32(%r15)
    vmovups %xmm2, 64(%r15)
    vmovups %xmm3, 96(%r15)

    vextractf128 $1, \x0, %xmm0
    vextractf128 $1, \x1, %xmm1
    vextractf128 $1, \x2, %xmm2
    vextractf128 $1, \x3, %xmm3

    vmovups %xmm0, 128(%r15)
    vmovups %xmm1, 160(%r15)
    vmovups %xmm2, 192(%r15)
    vmovups %xmm3, 224(%r15)

.endm

movq    $0, %rdi
movq    %rbx, %r14      // %r14: tempA
LoopE24H4:
    cmpq    $0, %r11
    je  End
    movslq  (%r9), %r12     // %r12: nonZeroCnt
    addq    $4, %r9
    subq    $4, %r11
    addq    $1, %rdi

    // Load bias to CVecs
    vzeroall
    cmpq    $0, %rsi
    je  LoopE24H4L1
        vbroadcastss    (%rsi), %ymm4
        addq    $4, %rsi
        vbroadcastss    (%rsi), %ymm7
        addq    $4, %rsi
        vbroadcastss    (%rsi), %ymm10
        addq    $4, %rsi
        vbroadcastss    (%rsi), %ymm13
        addq    $4, %rsi
        vmovups %ymm4, %ymm5
        vmovups %ymm4, %ymm6
        vmovups %ymm7, %ymm8
        vmovups %ymm7, %ymm9
        vmovups %ymm10, %ymm11
        vmovups %ymm10, %ymm12
        vmovups %ymm13, %ymm14
        vmovups %ymm13, %ymm15
    
    LoopE24H4L1:
        cmpq    $0, %r12
        je  LoopE24H4End
        vbroadcastss (%r8), %ymm3
        subq    $1, %r12
        movslq  (%r10), %r15
        salq    $2, %r15
        addq     %r15, %r14      // tempA += *dataOffsetMap
        addq    $4, %r10
        vmovups (%r14), %ymm0
        vmovups 32(%r14), %ymm1
        vmovups 64(%r14), %ymm2
        
        addq    $4, %r8
        vfmadd231ps %ymm3, %ymm0, %ymm4
        vfmadd231ps %ymm3, %ymm1, %ymm5
        vfmadd231ps %ymm3, %ymm2, %ymm6

        vbroadcastss (%r8), %ymm3
        addq    $4, %r8
        vfmadd231ps %ymm3, %ymm0, %ymm7
        vfmadd231ps %ymm3, %ymm1, %ymm8
        vfmadd231ps %ymm3, %ymm2, %ymm9

        vbroadcastss (%r8), %ymm3
        addq    $4, %r8
        vfmadd231ps %ymm3, %ymm0, %ymm10
        vfmadd231ps %ymm3, %ymm1, %ymm11
        vfmadd231ps %ymm3, %ymm2, %ymm12

        vbroadcastss (%r8), %ymm3
        addq    $4, %r8
        vfmadd231ps %ymm3, %ymm0, %ymm13
        vfmadd231ps %ymm3, %ymm1, %ymm14
        vfmadd231ps %ymm3, %ymm2, %ymm15

        jmp LoopE24H4L1
    
    LoopE24H4End:
        movq    %rax, %r15

        TRANSPOSE_SAVE  %ymm4, %ymm7, %ymm10, %ymm13
        addq    $256, %r15
        TRANSPOSE_SAVE  %ymm5, %ymm8, %ymm11, %ymm14
        addq    $256, %r15
        TRANSPOSE_SAVE  %ymm6, %ymm9, %ymm12, %ymm15

        movq    %rdi, %r15
        andq    $1, %r15
        cmpq    $0, %r15
        je  FullC
            addq    $16, %rax
            jmp LoopE24H4
        FullC:
            subq    $16, %rax
            addq    24(%rdx), %rax
            jmp LoopE24H4
    
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
popq    %r9
popq    %r8
popq    %rbx
popq    %rax
#endif

popq    %rbp
retq

