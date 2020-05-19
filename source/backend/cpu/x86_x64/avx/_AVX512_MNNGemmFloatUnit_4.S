#include "../MNNAsmGlobal.h"
#ifdef MNN_FMA_ENABLE
.text
.align 4

asm_function _AVX512_MNNGemmFloatUnit_4
//void _AVX512_MNNGemmFloatUnit_4(float* dst, const float* src, const float* weight, size_t src_depth_quad, size_t dst_step,
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
    movq $1, %r11
    movq %rdx, %r12
    movq %r13, %rsi
    vbroadcastf32x4 (%rdx), %zmm0
    vbroadcastf32x4 16(%rdx), %zmm1
    vbroadcastf32x4 32(%rdx), %zmm2
    vbroadcastf32x4 48(%rdx), %zmm3

    vmovups (%rsi), %zmm4
    vmovups 64(%rsi), %zmm5

    vmulps %zmm4, %zmm0, %zmm8
    vmulps %zmm4, %zmm1, %zmm9
    vmulps %zmm4, %zmm2, %zmm10
    vmulps %zmm4, %zmm3, %zmm11

    vmulps %zmm5, %zmm0, %zmm12
    vmulps %zmm5, %zmm1, %zmm13
    vmulps %zmm5, %zmm2, %zmm14
    vmulps %zmm5, %zmm3, %zmm15

    addq $64, %rdx
    addq $128, %rsi
    cmpq %rcx, %r11
    je LoopSzEnd

    LoopSz:
        vbroadcastf32x4 (%rdx), %zmm0
        vbroadcastf32x4 16(%rdx), %zmm1
        vbroadcastf32x4 32(%rdx), %zmm2
        vbroadcastf32x4 48(%rdx), %zmm3

        vmovups (%rsi), %zmm4
        vmovups 64(%rsi), %zmm5

        vfmadd231ps %zmm4, %zmm0, %zmm8
        vfmadd231ps %zmm4, %zmm1, %zmm9
        vfmadd231ps %zmm4, %zmm2, %zmm10
        vfmadd231ps %zmm4, %zmm3, %zmm11

        vfmadd231ps %zmm5, %zmm0, %zmm12
        vfmadd231ps %zmm5, %zmm1, %zmm13
        vfmadd231ps %zmm5, %zmm2, %zmm14
        vfmadd231ps %zmm5, %zmm3, %zmm15

        addq $64, %rdx
        addq $128, %rsi
        addq $1, %r11
        cmpq %rcx, %r11
        jne LoopSz
    LoopSzEnd:
    vextractf32x8 $1, %zmm8, %ymm0
    vextractf32x8 $1, %zmm9, %ymm1
    vextractf32x8 $1, %zmm10, %ymm2
    vextractf32x8 $1, %zmm11, %ymm3
    vhaddps %ymm0, %ymm8, %ymm8
    vhaddps %ymm1, %ymm9, %ymm9
    vhaddps %ymm2, %ymm10, %ymm10
    vhaddps %ymm3, %ymm11, %ymm11

    vextractf32x8 $1, %zmm12, %ymm0
    vextractf32x8 $1, %zmm13, %ymm1
    vextractf32x8 $1, %zmm14, %ymm2
    vextractf32x8 $1, %zmm15, %ymm3
    vhaddps %ymm0, %ymm12, %ymm12
    vhaddps %ymm1, %ymm13, %ymm13
    vhaddps %ymm2, %ymm14, %ymm14
    vhaddps %ymm3, %ymm15, %ymm15
    
    vhaddps %ymm9, %ymm8, %ymm8
    vhaddps %ymm11, %ymm10, %ymm10

    vhaddps %ymm12, %ymm11, %ymm12
    vhaddps %ymm14, %ymm13, %ymm14

    vmovups %ymm8, (%rdi)
    vmovups %ymm10, 32(%rdi)
    vmovups %ymm12, 64(%rdi)
    vmovups %ymm14, 96(%rdi)
    
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

#endif
