#include "../MNNAsmGlobal.h"
.text
.align 4

asm_function _AVX_MNNGemmFloatUnitFMA_4
//void _AVX_MNNGemmFloatUnitFMA_4(float* dst, const float* src, const float* weight, size_t src_depth_quad, size_t dst_step,
//                             size_t dst_depth_quad, size_t weight_depth_offset) {

//Auto: rdi: dst, rsi:src, rdx:weight, rcx:src_depth_quad , r8:dst_step, r9:dst_depth_quad
// r10:weight_depth_offset
pushq   %rbp
movq    %rsp, %rbp
pushq   %r12
pushq   %r13
cmpq $0, %r9
je End

movq 16(%rbp), %r10
imulq $4, %r10
imulq $4, %r8

movq %rsi, %r13
LoopDz:
    vzeroall
    movq $0, %r11
    movq %rdx, %r12
    movq %r13, %rsi
    
    LoopSz:
        vbroadcastf128 (%rdx), %ymm0
        vbroadcastf128 16(%rdx), %ymm1
        vbroadcastf128 32(%rdx), %ymm2
        vbroadcastf128 48(%rdx), %ymm3

        vmovups (%rsi), %ymm4

        vfmadd231ps %ymm4, %ymm0, %ymm8
        vfmadd231ps %ymm4, %ymm1, %ymm9
        vfmadd231ps %ymm4, %ymm2, %ymm10
        vmovups 32(%rsi), %ymm5
        vfmadd231ps %ymm4, %ymm3, %ymm11

        vfmadd231ps %ymm5, %ymm0, %ymm12
        vfmadd231ps %ymm5, %ymm1, %ymm13
        prefetcht0 512(%rdx)
        prefetcht0 512(%rsi)
        vfmadd231ps %ymm5, %ymm2, %ymm14
        vfmadd231ps %ymm5, %ymm3, %ymm15

        addq $64, %rdx
        addq $128, %rsi
        addq $1, %r11
        cmpq %rcx, %r11
        jne LoopSz
    vhaddps %ymm9, %ymm8, %ymm8
    vhaddps %ymm11, %ymm10, %ymm10
    vhaddps %ymm13, %ymm12, %ymm12
    vhaddps %ymm15, %ymm14, %ymm14
    vhaddps %ymm10, %ymm8, %ymm8
    vhaddps %ymm14, %ymm12, %ymm12
    vmovups %ymm8, (%rdi)
    vmovups %ymm12, 32(%rdi)

    vzeroall
    movq $0, %r11
    movq %r13, %rsi

    LoopSz2:
        vbroadcastf128 (%r12), %ymm0
        vbroadcastf128 16(%r12), %ymm1
        vbroadcastf128 32(%r12), %ymm2
        vbroadcastf128 48(%r12), %ymm3

        vmovups 64(%rsi), %ymm4

        vfmadd231ps %ymm4, %ymm0, %ymm8
        vfmadd231ps %ymm4, %ymm1, %ymm9
        vmovups 96(%rsi), %ymm5
        vfmadd231ps %ymm4, %ymm2, %ymm10
        vfmadd231ps %ymm4, %ymm3, %ymm11

        vfmadd231ps %ymm5, %ymm0, %ymm12
        prefetcht0 512(%r12)
        prefetcht0 512(%rsi)
        vfmadd231ps %ymm5, %ymm1, %ymm13
        vfmadd231ps %ymm5, %ymm2, %ymm14
        vfmadd231ps %ymm5, %ymm3, %ymm15

        addq $64, %r12
        addq $128, %rsi
        addq $1, %r11
        cmpq %rcx, %r11
        jne LoopSz2
    vhaddps %ymm9, %ymm8, %ymm8
    vhaddps %ymm11, %ymm10, %ymm10
    vhaddps %ymm13, %ymm12, %ymm12
    vhaddps %ymm15, %ymm14, %ymm14
    vhaddps %ymm10, %ymm8, %ymm8
    vhaddps %ymm14, %ymm12, %ymm12
    vmovups %ymm8, 64(%rdi)
    vmovups %ymm12, 96(%rdi)
    
    addq %r8, %rdi
    addq %r10, %rdx

    subq $1, %r9
    testq %r9, %r9
    jne LoopDz


End:
popq    %r13
popq    %r12
popq    %rbp

retq

