//
//  _AVX512_MNNPackedMatMul.S
//  MNN
//
//  Created by MNN on b'2020/09/22'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "../MNNAsmGlobal.h"
#ifdef MNN_FMA_ENABLE
.text
.align 4

asm_function _AVX512_MNNPackedMatMul
//void _AVX512_MNNPackedMatMul(float* C, const float* A, const float* B, const size_t* parameter)

//Auto: rdi: C, rsi:A, rdx:B, rcx: parameter
//parameter: e, l, h, CStride, aStride, bStride
pushq   %rbp
movq    %rsp, %rbp
pushq   %r12
pushq   %r13
pushq   %r14
pushq   %r15

movq $0, %r11
movq 8(%rcx), %r8
movq 16(%rcx), %r9
LoopE:
    movq $0, %r12
    movq %rdx, %r15
    movq %rdi, %r10
    LoopH:
        vzeroall

        // zmm15 -> zmm10 : C
        // ymm0, ymm1: A
        // ymm2, ymm3: B Temp
        movq $0, %r13
        movq %rsi, %r14
        LoopL:
            vmovups (%r14), %zmm0
            vbroadcastss (%r15), %zmm2
            vbroadcastss 4(%r15), %zmm3
            vbroadcastss 8(%r15), %zmm4
            vbroadcastss 12(%r15), %zmm5
            vbroadcastss 16(%r15), %zmm6
            vbroadcastss 20(%r15), %zmm7
            prefetcht0 512(%r14)
            vfmadd231ps %zmm2, %zmm0, %zmm10
            vfmadd231ps %zmm3, %zmm0, %zmm11

            vfmadd231ps %zmm4, %zmm0, %zmm12
            vfmadd231ps %zmm5, %zmm0, %zmm13

            vfmadd231ps %zmm6, %zmm0, %zmm14
            vfmadd231ps %zmm7, %zmm0, %zmm15

            addq $64, %r14
            addq $24, %r15

            incq %r13
            cmpq %r8, %r13
            jne LoopL
        vmovups %zmm10, (%r10)
        vmovups %zmm11, 64(%r10)
        vmovups %zmm12, 128(%r10)
        vmovups %zmm13, 192(%r10)
        vmovups %zmm14, 256(%r10)
        vmovups %zmm15, 320(%r10)
        addq $384, %r10
        addq 40(%rcx), %r15

        incq %r12
        cmpq %r9, %r12
        jne LoopH
    addq 24(%rcx), %rdi
    addq 32(%rcx), %rsi
    incq %r11
    cmpq (%rcx), %r11
    jne LoopE

End:
popq    %r15
popq    %r14
popq    %r13
popq    %r12
popq    %rbp

retq



#endif
