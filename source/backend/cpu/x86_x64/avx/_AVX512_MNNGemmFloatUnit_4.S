#include "../MNNAsmGlobal.h"
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
    movq $1, %r11
    movq %r13, %rsi
    vbroadcastf32x4 (%rdx), %zmm0
    vbroadcastf32x4 16(%rdx), %zmm1
    vbroadcastf32x4 32(%rdx), %zmm2
    vbroadcastf32x4 48(%rdx), %zmm3

    vmovups (%rsi), %zmm4
    vmovups 64(%rsi), %zmm5
    vmovups 128(%rsi), %zmm6
    vmovups 192(%rsi), %zmm7

    vmulps %zmm4, %zmm0, %zmm8
    vmulps %zmm4, %zmm1, %zmm9
    vmulps %zmm4, %zmm2, %zmm10
    vmulps %zmm4, %zmm3, %zmm11

    vmulps %zmm5, %zmm0, %zmm12
    prefetcht0 512(%rdx)
    prefetcht0 512(%rsi)
    vmulps %zmm5, %zmm1, %zmm13
    vmulps %zmm5, %zmm2, %zmm14
    vmulps %zmm5, %zmm3, %zmm15

    vmulps %zmm6, %zmm0, %zmm16
    vmulps %zmm6, %zmm1, %zmm17
    vmulps %zmm6, %zmm2, %zmm18
    vmulps %zmm6, %zmm3, %zmm19

    vmulps %zmm7, %zmm0, %zmm20
    vmulps %zmm7, %zmm1, %zmm21
    vmulps %zmm7, %zmm2, %zmm22
    vmulps %zmm7, %zmm3, %zmm23
    addq $64, %rdx
    addq $256, %rsi

    cmpq %rcx, %r11
    je LoopSzEnd

    LoopSz:
        vbroadcastf32x4 (%rdx), %zmm0
        vbroadcastf32x4 16(%rdx), %zmm1
        vbroadcastf32x4 32(%rdx), %zmm2
        vbroadcastf32x4 48(%rdx), %zmm3

        vmovups (%rsi), %zmm4
        vmovups 64(%rsi), %zmm5
        vmovups 128(%rsi), %zmm6
        vmovups 192(%rsi), %zmm7

        vfmadd231ps %zmm4, %zmm0, %zmm8
        vfmadd231ps %zmm4, %zmm1, %zmm9
        vfmadd231ps %zmm4, %zmm2, %zmm10
        vfmadd231ps %zmm4, %zmm3, %zmm11
        prefetcht0 512(%rdx)
        prefetcht0 512(%rsi)

        vfmadd231ps %zmm5, %zmm0, %zmm12
        vfmadd231ps %zmm5, %zmm1, %zmm13
        vfmadd231ps %zmm5, %zmm2, %zmm14
        vfmadd231ps %zmm5, %zmm3, %zmm15

        vfmadd231ps %zmm6, %zmm0, %zmm16
        vfmadd231ps %zmm6, %zmm1, %zmm17
        vfmadd231ps %zmm6, %zmm2, %zmm18
        vfmadd231ps %zmm6, %zmm3, %zmm19

        vfmadd231ps %zmm7, %zmm0, %zmm20
        vfmadd231ps %zmm7, %zmm1, %zmm21
        vfmadd231ps %zmm7, %zmm2, %zmm22
        vfmadd231ps %zmm7, %zmm3, %zmm23

        addq $64, %rdx
        addq $256, %rsi
        addq $1, %r11
        cmpq %rcx, %r11
        jne LoopSz
    LoopSzEnd:
    vextractf32x8 $1, %zmm8, %ymm0
    vextractf32x8 $1, %zmm9, %ymm1
    vextractf32x8 $1, %zmm10, %ymm2
    vextractf32x8 $1, %zmm11, %ymm3

    vextractf32x8 $1, %zmm12, %ymm4
    vextractf32x8 $1, %zmm13, %ymm5
    vextractf32x8 $1, %zmm14, %ymm6
    vextractf32x8 $1, %zmm15, %ymm7

    vhaddps %ymm9, %ymm8, %ymm8
    vhaddps %ymm11, %ymm10, %ymm10
    vhaddps %ymm13, %ymm12, %ymm12
    vhaddps %ymm15, %ymm14, %ymm14

    vhaddps %ymm1, %ymm0, %ymm0
    vhaddps %ymm3, %ymm2, %ymm2
    vhaddps %ymm5, %ymm4, %ymm4
    vhaddps %ymm7, %ymm6, %ymm6

    vhaddps %ymm10, %ymm8, %ymm8
    vhaddps %ymm14, %ymm12, %ymm12

    vhaddps %ymm2, %ymm0, %ymm0
    vhaddps %ymm6, %ymm4, %ymm4

    vmovups %ymm8, (%rdi)
    vmovups %ymm0, 32(%rdi)
    vmovups %ymm12, 64(%rdi)
    vmovups %ymm4, 96(%rdi)

    vextractf32x8 $0, %zmm16, %ymm8
    vextractf32x8 $0, %zmm17, %ymm9
    vextractf32x8 $0, %zmm18, %ymm10
    vextractf32x8 $0, %zmm19, %ymm11

    vextractf32x8 $1, %zmm16, %ymm0
    vextractf32x8 $1, %zmm17, %ymm1
    vextractf32x8 $1, %zmm18, %ymm2
    vextractf32x8 $1, %zmm19, %ymm3

    vextractf32x8 $1, %zmm20, %ymm4
    vextractf32x8 $1, %zmm21, %ymm5
    vextractf32x8 $1, %zmm22, %ymm6
    vextractf32x8 $1, %zmm23, %ymm7

    vextractf32x8 $0, %zmm20, %ymm12
    vextractf32x8 $0, %zmm21, %ymm13
    vextractf32x8 $0, %zmm22, %ymm14
    vextractf32x8 $0, %zmm23, %ymm15

    vhaddps %ymm9, %ymm8, %ymm8
    vhaddps %ymm11, %ymm10, %ymm10
    vhaddps %ymm13, %ymm12, %ymm12
    vhaddps %ymm15, %ymm14, %ymm14

    vhaddps %ymm1, %ymm0, %ymm0
    vhaddps %ymm3, %ymm2, %ymm2
    vhaddps %ymm5, %ymm4, %ymm4
    vhaddps %ymm7, %ymm6, %ymm6

    vhaddps %ymm10, %ymm8, %ymm8
    vhaddps %ymm14, %ymm12, %ymm12

    vhaddps %ymm2, %ymm0, %ymm0
    vhaddps %ymm6, %ymm4, %ymm4

    vmovups %ymm8, 128(%rdi)
    vmovups %ymm0, 160(%rdi)
    vmovups %ymm12,192(%rdi)
    vmovups %ymm4, 224(%rdi)

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

