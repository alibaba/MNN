#include "../MNNAsmGlobal.h"
#ifdef MNN_FMA_ENABLE
.text
.align 5

asm_function _AVX_MNNGemm16x4
//void _AVX_MNNGemm16x4(float* C, const float* A, const float* B, size_t e, size_t l, size_t h)

//Auto: rdi: C, rsi:A, rdx:B, rcx:e , r8:l, r9:h
pushq   %rbp
movq    %rsp, %rbp
pushq   %r12
pushq   %r13
pushq   %r14
pushq   %r15

movq $0, %r11    
LoopE:
    movq $0, %r12
    movq %rdx, %r15
    LoopH:
        vzeroall

        // ymm15 -> ymm4 : C
        // ymm0, ymm1: A
        // ymm2, ymm3: B Temp
        movq $0, %r13
        movq %rsi, %r14
        LoopL:
            vmovups (%r14), %ymm0
            vmovups 32(%r14), %ymm1
            vbroadcastss (%r15), %ymm2
            vbroadcastss 4(%r15), %ymm3
            prefetcht0 512(%r14)
            vfmadd231ps %ymm2, %ymm0, %ymm4
            vfmadd231ps %ymm2, %ymm1, %ymm5
            vfmadd231ps %ymm3, %ymm0, %ymm6
            vfmadd231ps %ymm3, %ymm1, %ymm7

            vbroadcastss 8(%r15), %ymm2
            vbroadcastss 12(%r15), %ymm3
            vfmadd231ps %ymm2, %ymm0, %ymm8
            vfmadd231ps %ymm2, %ymm1, %ymm9
            vfmadd231ps %ymm3, %ymm0, %ymm10
            vfmadd231ps %ymm3, %ymm1, %ymm11

            vbroadcastss 16(%r15), %ymm2
            vbroadcastss 20(%r15), %ymm3
            vfmadd231ps %ymm2, %ymm0, %ymm12
            vfmadd231ps %ymm2, %ymm1, %ymm13
            vfmadd231ps %ymm3, %ymm0, %ymm14
            vfmadd231ps %ymm3, %ymm1, %ymm15

            addq $64, %r14
            addq $24, %r15

            incq %r13
            cmpq %r8, %r13
            jne LoopL
        vmovups %ymm4, (%rdi)
        vmovups %ymm5, 32(%rdi)
        vmovups %ymm6, 64(%rdi)
        vmovups %ymm7, 96(%rdi)
        vmovups %ymm8, 128(%rdi)
        vmovups %ymm9, 160(%rdi)
        vmovups %ymm10, 192(%rdi)
        vmovups %ymm11, 224(%rdi)
        vmovups %ymm12, 256(%rdi)
        vmovups %ymm13, 288(%rdi)
        vmovups %ymm14, 320(%rdi)
        vmovups %ymm15, 352(%rdi)
        addq $384, %rdi

        incq %r12
        cmpq %r9, %r12
        jne LoopH
    movq $64, %r10
    imulq %r8, %r10
    addq %r10, %rsi
    incq %r11
    cmpq %rcx, %r11
    jne LoopE

End:
popq    %r15
popq    %r14
popq    %r13
popq    %r12
popq    %rbp

retq



#endif
