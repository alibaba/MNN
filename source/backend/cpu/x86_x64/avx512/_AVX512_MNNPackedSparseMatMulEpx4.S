
#include "../MNNAsmGlobal.h"
.text
.align 4

#define sizeof_value 4
#define sizeof_value_lg2 2
#define sparse_blockoc 4
#define sparse_blockoc_log 2
#define packC_unit 16
#define packC_unit_log 4

#define AVX512F32 16

// caution: asm version is a sub-loop of _AVX512_MNNPackedSparseMatMulEpx4()
// void _AVX512_MNNPackedSparseMatMulEpx4(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter,
//                                     const float* postParameters, const float* bias, unsigned int* NNZMap,
//                                     int* dataOffsetMap) {
asm_function _AVX512_MNNPackedSparseMatMulEpx4_ASM
// SystemV Auto: rdi: C, rsi: A, rdx:B,  rcx: eSize, r8: parameter, r9: postparameter,
// stack: bias, unsigned int* NNZMap, int* dataOffsetMap

// Microsoft x64 Auto: rcx:C, rdx:A, r8:B, r9:eSize
// stack: parameter, postParameters, bias, unsigned int* NNZMap, int* dataOffsetMap

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
#define push_registers_bytes_ ((8 + 1) * 8 + 32) // pushq + callq + shadow_space
movq (push_registers_bytes_)(%rsp), %r8 // parameter
movq (push_registers_bytes_ + 8)(%rsp), %r9 // postparameter
#define push_registers_bytes (push_registers_bytes_ + 2 * 8) // pushq + callq + shadow_space + extra
#else
pushq   %rax
pushq   %rbx
pushq   %r8
pushq   %r9
pushq   %r12
pushq   %r13
pushq   %r14
pushq   %r15
#define push_registers_bytes ((9 + 1) * 8) // pushq + callq
#endif

movq (%r8), %r10 // eP * sizeof
shrq  $(sizeof_value_lg2), %r10
cmpq %rcx, %r10 // eSize == eP
jne loop_end

//rcx:C,
//rax:A,
//rbx:B,
//rdx:bias

// rdi, r8: unsigned int* NNZMap,
// rsi, r9: int* dataOffsetMap
// r10: h,
// r11: cStride with sizeof

// z0-z2: A, z3-z5: swap
// z6-z9:  B
// z10: minValue
// z11: maxValue
// z12-v23: C

movq %rsi, %rax // A
movq %rdx, %rbx // B
movq %rdi, %rcx // C
movq 16(%r8), %r10 // h
movq 24(%r8), %r11 // cStride

vbroadcastss 8(%r9), %zmm10
vbroadcastss 12(%r9), %zmm11

movq %r10, %r14
shrq $(sparse_blockoc_log), %r14
shlq $(sparse_blockoc_log), %r14 // h even divid sparse_blockoc

movq (push_registers_bytes)(%rsp), %rdx // bias
movq (push_registers_bytes + 8)(%rsp), %rdi // unsigned int* NNZMap,
movq (push_registers_bytes + 16)(%rsp), %rsi // int* dataOffsetMap


// r8 as ih
// r9 as il

// movq %r8, %rdi
// movq %r9, %rsi

#ifdef WIN32
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
#endif

movslq (%rsi), %r15
leaq (%rax, %r15, 4), %rax // a = a + diff;
addq $4, %rsi // dataOffsetMap++

movq $0, %r8 //ih
cmp $0, %r14
je loop_e48h4_end

loop_e48h4:
    movq %r8, %r9
    movq %r8, %r12
    shrq $(packC_unit_log), %r9
    andq $15, %r12 // ih % packC_unit
    leaq (%rcx, %r12, sizeof_value), %r12
    imulq %r11, %r9 // (ih >> packC_unit_log) * cStride
    addq %r9, %r12 // r12 = c_address;

    cmp $0, %rdx
    je load_e48h4_zero
        vbroadcastss (%rdx), %zmm12
        vbroadcastss 4(%rdx), %zmm15
        vbroadcastss 8(%rdx), %zmm18
        vbroadcastss 12(%rdx), %zmm21
        addq $(sparse_blockoc * 4), %rdx // always 32-bit
        jmp load_e48h4_zero_end
    load_e48h4_zero:
      vxorps %zmm12, %zmm12, %zmm12
      vxorps %zmm15, %zmm15, %zmm15
      vxorps %zmm18, %zmm18, %zmm18
      vxorps %zmm21, %zmm21, %zmm21

    load_e48h4_zero_end:
        movl (%rdi), %r9d
        vmovaps %zmm12, %zmm13
        vmovaps %zmm12, %zmm14
        vmovaps %zmm15, %zmm16
        vmovaps %zmm15, %zmm17
        vmovaps %zmm18, %zmm19
        vmovaps %zmm18, %zmm20
        vmovaps %zmm21, %zmm22
        vmovaps %zmm21, %zmm23
        cmpl $0, %r9d
        je loop_e48h4l1_end

    movslq (%rsi), %r15
    vmovups (%rax), %zmm3
    vmovups 64(%rax), %zmm4
    vmovups 128(%rax), %zmm5
    leaq (%rax, %r15, sizeof_value), %rax // a = a + diff;
    addq $4, %rsi // dataOffsetMap++

    loop_e48h4l1:

         movslq (%rsi), %r15
         decl %r9d
         vbroadcastss (%rbx), %zmm6
         vbroadcastss 4(%rbx), %zmm7
         vbroadcastss 8(%rbx), %zmm8
         vbroadcastss 12(%rbx), %zmm9
         vmovaps %zmm3, %zmm0
         vmovaps %zmm4, %zmm1
         vmovaps %zmm5, %zmm2

         vfmadd231ps %zmm3, %zmm6, %zmm12
         vfmadd231ps %zmm4, %zmm6, %zmm13
         vmovups (%rax), %zmm3
         vmovups 64(%rax), %zmm4
         vfmadd231ps %zmm5, %zmm6, %zmm14
         vmovups 128(%rax), %zmm5

         vfmadd231ps %zmm0, %zmm7, %zmm15
         vfmadd231ps %zmm1, %zmm7, %zmm16
         vfmadd231ps %zmm2, %zmm7, %zmm17
         vfmadd231ps %zmm0, %zmm8, %zmm18
         vfmadd231ps %zmm1, %zmm8, %zmm19
         vfmadd231ps %zmm2, %zmm8, %zmm20
         vfmadd231ps %zmm0, %zmm9, %zmm21
         vfmadd231ps %zmm1, %zmm9, %zmm22
         vfmadd231ps %zmm2, %zmm9, %zmm23

         leaq (%rax, %r15, sizeof_value), %rax // a = a + diff; // 求证：skylake lea占用浮点计算流水线
         // shlq $sizeof_value_lg2, %r15
         addq $(sparse_blockoc * sizeof_value), %rbx
         addq $4, %rsi // dataOffsetMap++
         // addq %r15, %rax


        // vmovaps %zmm3, %zmm0
        // vmovaps %zmm4, %zmm1
        // vmovaps %zmm5, %zmm2
        // vbroadcastss (%rbx), %zmm6
        // vbroadcastss 4(%rbx), %zmm7
        // vbroadcastss 8(%rbx), %zmm8
        // vbroadcastss 12(%rbx), %zmm9
//
        // vfmadd231ps %zmm0, %zmm6, %zmm12
        // vfmadd231ps %zmm1, %zmm6, %zmm13
        // vmovups (%rax), %zmm3
        // vmovups 64(%rax), %zmm4
        // vmovups 128(%rax), %zmm5
        // vfmadd231ps %zmm2, %zmm6, %zmm14
        // vfmadd231ps %zmm0, %zmm7, %zmm15
        // vfmadd231ps %zmm1, %zmm7, %zmm16
        // vfmadd231ps %zmm2, %zmm7, %zmm17
        // movslq (%rsi), %r15
        // decl %r9d
        // addq $(sparse_blockoc * sizeof_value), %rbx
        // addq $4, %rsi // dataOffsetMap++
        // leaq (%rax, %r15, sizeof_value), %rax // a = a + diff; // 求证：skylake lea占用浮点计算流水线
        // vfmadd231ps %zmm0, %zmm8, %zmm18
        // vfmadd231ps %zmm1, %zmm8, %zmm19
        // vfmadd231ps %zmm2, %zmm8, %zmm20
        // vfmadd231ps %zmm0, %zmm9, %zmm21
        // vfmadd231ps %zmm1, %zmm9, %zmm22
        // vfmadd231ps %zmm2, %zmm9, %zmm23

        cmpl $0, %r9d
        jne loop_e48h4l1

    loop_e48h4l1_end:

    vminps %zmm11, %zmm12, %zmm12
    vminps %zmm11, %zmm13, %zmm13
    vminps %zmm11, %zmm14, %zmm14
    vminps %zmm11, %zmm15, %zmm15
    vminps %zmm11, %zmm16, %zmm16
    vminps %zmm11, %zmm17, %zmm17
    vminps %zmm11, %zmm18, %zmm18
    vminps %zmm11, %zmm19, %zmm19
    vminps %zmm11, %zmm20, %zmm20
    vminps %zmm11, %zmm21, %zmm21
    vminps %zmm11, %zmm22, %zmm22
    vminps %zmm11, %zmm23, %zmm23
    vmaxps %zmm10, %zmm12, %zmm12
    vmaxps %zmm10, %zmm13, %zmm13
    vmaxps %zmm10, %zmm14, %zmm14
    vmaxps %zmm10, %zmm15, %zmm15
    vmaxps %zmm10, %zmm16, %zmm16
    vmaxps %zmm10, %zmm17, %zmm17
    vmaxps %zmm10, %zmm18, %zmm18
    vmaxps %zmm10, %zmm19, %zmm19
    vmaxps %zmm10, %zmm20, %zmm20
    vmaxps %zmm10, %zmm21, %zmm21
    vmaxps %zmm10, %zmm22, %zmm22
    vmaxps %zmm10, %zmm23, %zmm23

.macro TRANSPOSE4x4_STORE dest, ablock, aSegment, packCUnit, acc0, acc1, acc2, acc3
    vextractf32x4 $\aSegment, \acc0, %xmm0
    vextractf32x4 $\aSegment, \acc1, %xmm1
    vextractf32x4 $\aSegment, \acc2, %xmm2
    vextractf32x4 $\aSegment, \acc3, %xmm3
    vunpcklps %xmm1, %xmm0, %xmm4
    vunpcklps %xmm3, %xmm2, %xmm5
    vunpckhps %xmm1, %xmm0, %xmm0
    vunpckhps %xmm3, %xmm2, %xmm1
    vmovlhps  %xmm5, %xmm4, %xmm2
    vunpckhpd %xmm5, %xmm4, %xmm3
    vmovlhps  %xmm1, %xmm0, %xmm4
    vunpckhpd %xmm1, %xmm0, %xmm0
    vmovaps %xmm2, ((\ablock * AVX512F32 * \packCUnit + 4 * \aSegment * \packCUnit) * sizeof_value)(\dest)
    vmovaps %xmm3, ((\ablock * AVX512F32 * \packCUnit + 4 * \aSegment * \packCUnit + \packCUnit) * sizeof_value)(\dest)
    vmovaps %xmm4, ((\ablock * AVX512F32 * \packCUnit + 4 * \aSegment * \packCUnit + \packCUnit * 2) * sizeof_value)(\dest)
    vmovaps %xmm0, ((\ablock * AVX512F32 * \packCUnit + 4 * \aSegment * \packCUnit + \packCUnit * 3) * sizeof_value)(\dest)
.endm

    subq $4, %rsi // dataOffsetMap--
    movslq (%rsi), %r15
    addq $(sparse_blockoc), %r8
    addq $4, %rdi
    negq %r15
    leaq (%rax, %r15, sizeof_value), %rax // a = a - diff;

    TRANSPOSE4x4_STORE %r12, 0, 0, packC_unit, %zmm12, %zmm15, %zmm18, %zmm21
    TRANSPOSE4x4_STORE %r12, 0, 1, packC_unit, %zmm12, %zmm15, %zmm18, %zmm21
    TRANSPOSE4x4_STORE %r12, 0, 2, packC_unit, %zmm12, %zmm15, %zmm18, %zmm21
    TRANSPOSE4x4_STORE %r12, 0, 3, packC_unit, %zmm12, %zmm15, %zmm18, %zmm21
    TRANSPOSE4x4_STORE %r12, 1, 0, packC_unit, %zmm13, %zmm16, %zmm19, %zmm22
    TRANSPOSE4x4_STORE %r12, 1, 1, packC_unit, %zmm13, %zmm16, %zmm19, %zmm22
    TRANSPOSE4x4_STORE %r12, 1, 2, packC_unit, %zmm13, %zmm16, %zmm19, %zmm22
    TRANSPOSE4x4_STORE %r12, 1, 3, packC_unit, %zmm13, %zmm16, %zmm19, %zmm22
    TRANSPOSE4x4_STORE %r12, 2, 0, packC_unit, %zmm14, %zmm17, %zmm20, %zmm23
    TRANSPOSE4x4_STORE %r12, 2, 1, packC_unit, %zmm14, %zmm17, %zmm20, %zmm23
    TRANSPOSE4x4_STORE %r12, 2, 2, packC_unit, %zmm14, %zmm17, %zmm20, %zmm23
    TRANSPOSE4x4_STORE %r12, 2, 3, packC_unit, %zmm14, %zmm17, %zmm20, %zmm23



    // movq %r12, %r15
    // subq %rcx, %r15
    // movl $10, (%rcx, %r8, 4)
    // movl $0, 4(%rcx, %r8, 4)
    // movl %r15d, 8(%rcx, %r8, 4) // c_offset
    // movl %r8d, 12(%rcx, %r8, 4) // ih
    // movl %r9d, 16(%rcx, %r8, 4) // il
    cmpq %r14, %r8
    jl loop_e48h4  // r8 < r14

loop_e48h4_end:

cmpq %r10, %r8
je loop_end

loop_e48h1:
    movq %r8, %r9
    movq %r8, %r12
    shrq $(packC_unit_log), %r9
    andq $15, %r12 // ih % packC_unit
    leaq (%rcx, %r12, sizeof_value), %r12
    imulq %r11, %r9 // (ih >> packC_unit_log) * cStride
    addq %r9, %r12 // r12 = c_address;

    cmp $0, %rdx
    je load_e48h1_zero
        vbroadcastss (%rdx), %zmm12
        addq $(4), %rdx // always 32-bit
        jmp load_e48h1_zero_end
    load_e48h1_zero:
      vxorps %zmm12, %zmm12, %zmm12

    load_e48h1_zero_end:
        movl (%rdi), %r9d
        vmovaps %zmm12, %zmm13
        vmovaps %zmm12, %zmm14
        cmpl $0, %r9d
        je loop_e48h1l1_end

    movslq (%rsi), %r15
    vmovups (%rax), %zmm3
    vmovups 64(%rax), %zmm4
    vmovups 128(%rax), %zmm5
    leaq (%rax, %r15, sizeof_value), %rax // a = a + diff;
    addq $4, %rsi // dataOffsetMap++

    loop_e48h1l1:

         movslq (%rsi), %r15
         decl %r9d
         vbroadcastss (%rbx), %zmm6

         vfmadd231ps %zmm3, %zmm6, %zmm12
         vfmadd231ps %zmm4, %zmm6, %zmm13
         vmovups (%rax), %zmm3
         vmovups 64(%rax), %zmm4
         vfmadd231ps %zmm5, %zmm6, %zmm14
         vmovups 128(%rax), %zmm5

         leaq (%rax, %r15, sizeof_value), %rax // a = a + diff; // 求证：skylake lea占用浮点计算流水线
         // shlq $sizeof_value_lg2, %r15
         addq $(sizeof_value), %rbx
         addq $4, %rsi // dataOffsetMap++
         // addq %r15, %rax

        cmpl $0, %r9d
        jne loop_e48h1l1

    loop_e48h1l1_end:

    vminps %zmm11, %zmm12, %zmm12
    vminps %zmm11, %zmm13, %zmm13
    vminps %zmm11, %zmm14, %zmm14
    vmaxps %zmm10, %zmm12, %zmm12
    vmaxps %zmm10, %zmm13, %zmm13
    vmaxps %zmm10, %zmm14, %zmm14

    subq $4, %rsi // dataOffsetMap--
    movslq (%rsi), %r15
    addq $1, %r8
    addq $4, %rdi
    negq %r15
    leaq (%rax, %r15, sizeof_value), %rax // a = a - diff;


    vextractf128 $0x1,%ymm12, %xmm0
    vmovss %xmm12, (%rdx)
    vextractps $0x1, %xmm12, 0x40(%rdx)
    vextractps $0x2, %xmm12, 0x80(%rdx)
    vextractps $0x3, %xmm12, 0xc0(%rdx)

    vextractf32x8 $0x1, %zmm12, %ymm1
    vmovss %xmm0, 0x100(%rdx)
    vextractps $0x1, %xmm0, 0x140(%rdx)
    vextractps $0x2, %xmm0, 0x180(%rdx)
    vextractps $0x3, %xmm0, 0x1c0(%rdx)

    vextractf128 $0x1, %ymm1, %xmm2
    vmovss %xmm1, 0x200(%rdx)
    vextractps $0x1, %xmm1, 0x240(%rdx)
    vextractps $0x2, %xmm1, 0x280(%rdx)
    vextractps $0x3, %xmm1, 0x2c0(%rdx)

    vextractf32x8 $0x1, %zmm13, %ymm0
    vmovss %xmm2, 0x300(%rdx)
    vextractps $0x1, %xmm2, 0x340(%rdx)
    vextractps $0x2, %xmm2, 0x380(%rdx)
    vextractps $0x3, %xmm2, 0x3c0(%rdx)


    vextractf128 $0x1,%ymm13, %xmm0
    vmovss %xmm13, 0x400(%rdx)
    vextractps $0x1, %xmm13, 0x440(%rdx)
    vextractps $0x2, %xmm13, 0x480(%rdx)
    vextractps $0x3, %xmm13, 0x4c0(%rdx)

    vextractf32x8 $0x1, %zmm12, %ymm1
    vmovss %xmm0, 0x500(%rdx)
    vextractps $0x1, %xmm0, 0x540(%rdx)
    vextractps $0x2, %xmm0, 0x580(%rdx)
    vextractps $0x3, %xmm0, 0x5c0(%rdx)

    vextractf128 $0x1, %ymm1, %xmm2
    vmovss %xmm1, 0x600(%rdx)
    vextractps $0x1, %xmm1, 0x640(%rdx)
    vextractps $0x2, %xmm1, 0x680(%rdx)
    vextractps $0x3, %xmm1, 0x6c0(%rdx)

    vextractf32x8 $0x1, %zmm14, %ymm0
    vmovss %xmm2, 0x700(%rdx)
    vextractps $0x1, %xmm2, 0x740(%rdx)
    vextractps $0x2, %xmm2, 0x780(%rdx)
    vextractps $0x3, %xmm2, 0x7c0(%rdx)


    vextractf128 $0x1,%ymm12, %xmm0
    vmovss %xmm12, 0x800(%rdx)
    vextractps $0x1, %xmm12,0x840(%rdx)
    vextractps $0x2, %xmm12,0x880(%rdx)
    vextractps $0x3, %xmm12,0x8c0(%rdx)

    vextractf32x8 $0x1, %zmm12, %ymm1
    vmovss %xmm0, 0x900(%rdx)
    vextractps $0x1, %xmm0, 0x940(%rdx)
    vextractps $0x2, %xmm0, 0x980(%rdx)
    vextractps $0x3, %xmm0, 0x9c0(%rdx)

    vextractf128 $0x1, %ymm1, %xmm2
    vmovss %xmm1, 0xa00(%rdx)
    vextractps $0x1, %xmm1, 0xa40(%rdx)
    vextractps $0x2, %xmm1, 0xa80(%rdx)
    vextractps $0x3, %xmm1, 0xac0(%rdx)

    vmovss %xmm2, 0xb00(%rdx)
    vextractps $0x1, %xmm2, 0xb40(%rdx)
    vextractps $0x2, %xmm2, 0xb80(%rdx)
    vextractps $0x3, %xmm2, 0xbc0(%rdx)

    cmpq %r10, %r8
    jl loop_e48h1  // r8 < r14


loop_e48h1_end:


loop_end:

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

#undef sizeof_value
#undef sizeof_value_lg2
#undef sparse_blockoc
#undef packC_unit
#undef packC_unit_log
#undef AVX512F32
#undef push_registers_bytes

