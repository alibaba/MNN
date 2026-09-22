#if defined(__aarch64__)

    .macro MNN_HIAI_TRAMPOLINE symbol
    .text
    .balign 4
    .global \symbol
    .hidden \symbol
    .type \symbol, %function
\symbol:
    adrp x16, mnn_hiai_dynamic_function_table
    add x16, x16, :lo12:mnn_hiai_dynamic_function_table
    ldr x16, [x16, #(mnn_hiai_dynamic_function_index * 8)]
    br x16
    .size \symbol, .-\symbol
    .set mnn_hiai_dynamic_function_index, mnn_hiai_dynamic_function_index + 1
    .endm

#elif defined(__arm__)

    .syntax unified
    .arm
    .macro MNN_HIAI_TRAMPOLINE symbol
    .text
    .balign 4
    .global \symbol
    .hidden \symbol
    .type \symbol, %function
\symbol:
    ldr ip, 1f
0:
    add ip, pc, ip
    ldr ip, [ip, #(mnn_hiai_dynamic_function_index * 4)]
    bx ip
    .balign 4
1:
    .word mnn_hiai_dynamic_function_table - (0b + 8)
    .size \symbol, .-\symbol
    .set mnn_hiai_dynamic_function_index, mnn_hiai_dynamic_function_index + 1
    .endm

#else
#error "The HiAI dynamic ABI bridge supports AArch64 and AArch32 only"
#endif

    .hidden mnn_hiai_dynamic_function_table
    .set mnn_hiai_dynamic_function_index, 0

#define MNN_HIAI_FUNCTION(symbol) MNN_HIAI_TRAMPOLINE symbol
#define MNN_HIAI_OBJECT(symbol, size64, size32)
#include "HiAIDynamicSymbols.inc"
#undef MNN_HIAI_OBJECT
#undef MNN_HIAI_FUNCTION

    .pushsection .bss.mnn_hiai_dynamic_objects, "aw", %nobits
#if defined(__aarch64__)
    .balign 8
#define MNN_HIAI_OBJECT_SIZE(size64, size32) size64
#else
    .balign 4
#define MNN_HIAI_OBJECT_SIZE(size64, size32) size32
#endif

    .macro MNN_HIAI_DYNAMIC_OBJECT symbol, object_size
    .global \symbol
    .hidden \symbol
    .type \symbol, %object
\symbol:
    .zero \object_size
    .size \symbol, .-\symbol
    .endm

#define MNN_HIAI_FUNCTION(symbol)
#define MNN_HIAI_OBJECT(symbol, size64, size32) \
    MNN_HIAI_DYNAMIC_OBJECT symbol, MNN_HIAI_OBJECT_SIZE(size64, size32)
#include "HiAIDynamicSymbols.inc"
#undef MNN_HIAI_OBJECT
#undef MNN_HIAI_FUNCTION
#undef MNN_HIAI_OBJECT_SIZE

    .popsection

    .section .note.GNU-stack,"",%progbits
