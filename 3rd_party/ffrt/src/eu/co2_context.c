/*
 * Copyright (c) 2023 Huawei Device Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "eu/co2_context.h"
#include "c/fiber.h"

#include <errno.h>
#define API_ATTRIBUTE(attr) __attribute__(attr)

void context_entry(void);

#if defined(__aarch64__)
asm(".global context_entry; .type context_entry, %function; context_entry:\n"
    "ldp x0, x1, [sp], #0x10\n"
    "mov lr, xzr\n"
    "br  x1\n"
    ".size context_entry, . - context_entry\n"
    ".global co2_save_context; .type co2_save_context, %function; co2_save_context:\n"
    "stp x19, x20, [x0,#0]\n"
    "stp x21, x22, [x0,#16]\n"
    "stp x23, x24, [x0,#32]\n"
    "stp x25, x26, [x0,#48]\n"
    "stp x27, x28, [x0,#64]\n"
    "stp x29, x30, [x0,#80]\n"
    "mov x2, sp\n"
    "str x2, [x0,#104]\n"
    "stp  d8,  d9, [x0,#112]\n"
    "stp d10, d11, [x0,#128]\n"
    "stp d12, d13, [x0,#144]\n"
    "stp d14, d15, [x0,#160]\n"
    "mov x0, #0\n"
    "ret\n"
    ".size co2_save_context, . - co2_save_context\n"
    ".global co2_restore_context; .type co2_restore_context, %function; co2_restore_context:\n"
    "ldp x19, x20, [x0,#0]\n"
    "ldp x21, x22, [x0,#16]\n"
    "ldp x23, x24, [x0,#32]\n"
    "ldp x25, x26, [x0,#48]\n"
    "ldp x27, x28, [x0,#64]\n"
    "ldp x29, x30, [x0,#80]\n"
    "ldr x2, [x0,#104]\n"
    "mov sp, x2\n"
    "ldp  d8,  d9, [x0,#112]\n"
    "ldp d10, d11, [x0,#128]\n"
    "ldp d12, d13, [x0,#144]\n"
    "ldp d14, d15, [x0,#160]\n"
    "mov x0, #1\n"
    "ret\n"
    ".size co2_restore_context, . - co2_restore_context\n");
#elif defined(__arm__)
asm(".global context_entry; .type context_entry, %function; context_entry:\n"
    "pop {r0, r1}\n"
    "mov lr, #0\n"
    "bx  r1\n"
    ".size context_entry, . - context_entry\n"
    ".global co2_save_context; .type co2_save_context, %function; co2_save_context:\n"
    "mov ip, r0\n"
    "str sp, [ip], #4\n"
    "str lr, [ip], #4\n"
    "stmia ip!, {v1-v6, sl, fp}\n"
    ".fpu vfp\n"
    "vstmia ip!, {d8-d15}\n"
    "mov r0, #0\n"
    "bx lr\n"
    ".size co2_save_context, . - co2_save_context\n"
    ".global co2_restore_context; .type co2_restore_context, %function; co2_restore_context:\n"
    "mov ip, r0\n"
    "ldr a4, [ip], #4\n"
    "ldr r4, [ip], #4\n"
    "mov sp, a4\n"
    "mov lr, r4\n"
    "ldmia ip!, {v1-v6, sl, fp}\n"
    ".fpu vfp\n"
    "vldmia ip!, {d8-d15}\n"
    "mov r0, #1\n"
    "bx lr\n"
    ".size co2_restore_context, . - co2_restore_context\n");
#elif (defined(__x86_64__) && !defined _MSC_VER)
asm(".global context_entry; .type context_entry, %function; context_entry:\n"
    "pop %rdi\n"
    "jmp *(%rsp)\n"
    ".size context_entry, . - context_entry\n"
    ".global co2_save_context; .type co2_save_context, %function; co2_save_context:\n"
    "mov %rbx, (%rdi)\n"
    "mov %rbp, 8(%rdi)\n"
    "mov %r12, 16(%rdi)\n"
    "mov %r13, 24(%rdi)\n"
    "mov %r14, 32(%rdi)\n"
    "mov %r15, 40(%rdi)\n"
    "lea 8(%rsp), %rdx\n"
    "mov %rdx, 48(%rdi)\n"
    "mov (%rsp), %rdx\n"
    "mov %rdx, 56(%rdi)\n"
    "xor %rax, %rax\n"
    "ret\n"
    ".size co2_save_context, . - co2_save_context\n"
    ".global co2_restore_context; .type co2_restore_context, %function; co2_restore_context:\n"
    "xor %rax, %rax\n"
    "inc %rax\n"
    "mov (%rdi), %rbx\n"
    "mov 8(%rdi), %rbp\n"
    "mov 16(%rdi), %r12\n"
    "mov 24(%rdi), %r13\n"
    "mov 32(%rdi), %r14\n"
    "mov 40(%rdi), %r15\n"
    "mov 48(%rdi), %rdx\n"
    "mov %rdx, %rsp\n"
    "jmp *56(%rdi)\n"
    ".size co2_restore_context, . - co2_restore_context\n");
#elif defined(__riscv) && __riscv_xlen == 64
asm(".global context_entry; .type context_entry, %function; context_entry:\n"
    "ld a0, 0(sp)\n"
    "ld a1, 8(sp)\n"
    "jalr a1\n"
    ".size context_entry, . - context_entry\n"
#ifndef __riscv_float_abi_soft
    ".attribute arch,\"rv64gc\" // LLVM Bug: https://github.com/llvm/llvm-project/issues/61991\n"
#endif
    ".global co2_save_context; .type co2_save_context, %function; co2_save_context:\n"
    "sd ra,    0(a0)\n"
    "sd s0,    8(a0)\n"
    "sd s1,    16(a0)\n"
    "sd s2,    24(a0)\n"
    "sd s3,    32(a0)\n"
    "sd s4,    40(a0)\n"
    "sd s5,    48(a0)\n"
    "sd s6,    56(a0)\n"
    "sd s7,    64(a0)\n"
    "sd s8,    72(a0)\n"
    "sd s9,    80(a0)\n"
    "sd s10,   88(a0)\n"
    "sd s11,   96(a0)\n"
    "sd sp,    104(a0)\n"
#ifndef __riscv_float_abi_soft
    "fsd fs0,  112(a0)\n"
    "fsd fs1,  120(a0)\n"
    "fsd fs2,  128(a0)\n"
    "fsd fs3,  136(a0)\n"
    "fsd fs4,  144(a0)\n"
    "fsd fs5,  152(a0)\n"
    "fsd fs6,  160(a0)\n"
    "fsd fs7,  168(a0)\n"
    "fsd fs8,  176(a0)\n"
    "fsd fs9,  184(a0)\n"
    "fsd fs10, 192(a0)\n"
    "fsd fs11, 200(a0)\n"
#endif
    "li a0, 0\n"
    "ret\n"
    ".size co2_save_context, . - co2_save_context\n"
    ".global co2_restore_context; .type co2_restore_context, %function; co2_restore_context:\n"
    "ld ra,    0(a0)\n"
    "ld s0,    8(a0)\n"
    "ld s1,    16(a0)\n"
    "ld s2,    24(a0)\n"
    "ld s3,    32(a0)\n"
    "ld s4,    40(a0)\n"
    "ld s5,    48(a0)\n"
    "ld s6,    56(a0)\n"
    "ld s7,    64(a0)\n"
    "ld s8,    72(a0)\n"
    "ld s9,    80(a0)\n"
    "ld s10,   88(a0)\n"
    "ld s11,   96(a0)\n"
    "ld sp,    104(a0)\n"
#ifndef __riscv_float_abi_soft
    "fld fs0,  112(a0)\n"
    "fld fs1,  120(a0)\n"
    "fld fs2,  128(a0)\n"
    "fld fs3,  136(a0)\n"
    "fld fs4,  144(a0)\n"
    "fld fs5,  152(a0)\n"
    "fld fs6,  160(a0)\n"
    "fld fs7,  168(a0)\n"
    "fld fs8,  176(a0)\n"
    "fld fs9,  184(a0)\n"
    "fld fs10, 192(a0)\n"
    "fld fs11, 200(a0)\n"
#endif
    "li a0, 1\n"
    "ret\n"
    ".size co2_restore_context, . - co2_restore_context\n");
#endif

API_ATTRIBUTE((visibility("default")))
int ffrt_fiber_init(ffrt_fiber_t* fiber, void(*func)(void*), void* arg, void* stack, size_t stack_size)
{
    if (stack_size < 0x4 * sizeof(uintptr_t)) {
        return EINVAL;
    }

    uintptr_t stack_top = (uintptr_t)stack + stack_size - 0x2 * sizeof(uintptr_t);
    stack_top -= stack_top % (0x2 * sizeof(uintptr_t));
    uintptr_t* data = (uintptr_t*)stack_top;

    fiber->storage[FFRT_REG_LR] = (uintptr_t)context_entry;
    fiber->storage[FFRT_REG_SP] = stack_top;

    data[0] = (uintptr_t)arg;
    data[1] = (uintptr_t)func;

    return 0;
}

API_ATTRIBUTE((visibility("default")))
void ffrt_fiber_switch(ffrt_fiber_t* from, ffrt_fiber_t* to)
{
    if (co2_save_context(from) == 0) {
        co2_restore_context(to);
    }
}