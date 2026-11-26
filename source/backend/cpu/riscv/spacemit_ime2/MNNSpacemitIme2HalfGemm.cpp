//
//  MNNSpacemitIme2HalfGemm.cpp
//  MNN
//
//  Created by MNN on 2026/11/18.
//  Copyright © 2026, Alibaba Group Holding Limited
//

#include "MNNSpacemitIme2Half.hpp"
#if defined(__riscv_zvfh)
#include <riscv_vector.h>
#include <cassert>

namespace MNNIme2 {
namespace {
// Stitch adjacent eight-float row segments into sixteen- or thirty-two-float segments.
template <int stage>
inline vfloat32m2_t zip(vfloat32m1_t left, vfloat32m1_t right) {
    vfloat32m2_t pair;
    asm("vpack.vv %0, %1, %2, %3" : "=&vr"(pair) : "vr"(left), "vr"(right), "i"(stage));
    return pair;
}

inline void storeRowPair(uint8_t* dst, size_t pitch, vfloat32m1_t left, vfloat32m1_t right) {
    auto rows = zip<3>(left, right);
    __riscv_vse32_v_f32m1(reinterpret_cast<float*>(dst), __riscv_vget_v_f32m2_f32m1(rows, 0), 32);
    __riscv_vse32_v_f32m1(reinterpret_cast<float*>(dst + pitch), __riscv_vget_v_f32m2_f32m1(rows, 1), 32);
}

template <int half>
inline void storeFourRows(uint8_t* dst, size_t pitch, vfloat32m8_t block) {
    auto left = zip<2>(__riscv_vget_v_f32m8_f32m1(block, half), __riscv_vget_v_f32m8_f32m1(block, 2 + half));
    auto right = zip<2>(__riscv_vget_v_f32m8_f32m1(block, 4 + half), __riscv_vget_v_f32m8_f32m1(block, 6 + half));
    storeRowPair(dst, pitch, __riscv_vget_v_f32m2_f32m1(left, 0), __riscv_vget_v_f32m2_f32m1(right, 0));
    storeRowPair(dst + 2 * pitch, pitch, __riscv_vget_v_f32m2_f32m1(left, 1), __riscv_vget_v_f32m2_f32m1(right, 1));
}
} // namespace

// Each load brings in one 8x8 input fragment for the next panel while the
// current panel updates a different accumulator. A and B advance by 16x8 and 32x8 halves.
// clang-format off
#define MNN_IME2_LOAD(reg, ptr, bytes) \
    "vle16.v " #reg ", (%[" #ptr "])\n\t" \
    "addi %[" #ptr "], %[" #ptr "], " #bytes "\n\t"
#define MNN_IME2_DOT(dst, row, col) "vfwmadot " #dst ", " #row ", " #col "\n\t"
#define MNN_IME2_NEXT(a0, a1, b0, b1, b2, b3, na0, na1, nb0, nb1, nb2, nb3) \
    MNN_IME2_LOAD(na0, a, 256)       MNN_IME2_DOT(v0, a0, b0)  \
    MNN_IME2_LOAD(na1, secondA, 256) MNN_IME2_DOT(v2, a0, b1)  \
    MNN_IME2_LOAD(nb0, b, 512)       MNN_IME2_DOT(v4, a0, b2)  \
    MNN_IME2_LOAD(nb1, col1, 512)    MNN_IME2_DOT(v6, a0, b3)  \
    MNN_IME2_LOAD(nb2, col2, 512)    MNN_IME2_DOT(v8, a1, b0)  \
    MNN_IME2_LOAD(nb3, col3, 512)    MNN_IME2_DOT(v10, a1, b1) \
    MNN_IME2_DOT(v12, a1, b2)        MNN_IME2_DOT(v14, a1, b3)
#define MNN_IME2_FINISH(a0, a1, b0, b1, b2, b3) \
    MNN_IME2_DOT(v0, a0, b0)  MNN_IME2_DOT(v2, a0, b1)  \
    MNN_IME2_DOT(v4, a0, b2)  MNN_IME2_DOT(v6, a0, b3)  \
    MNN_IME2_DOT(v8, a1, b0)  MNN_IME2_DOT(v10, a1, b1) \
    MNN_IME2_DOT(v12, a1, b2) MNN_IME2_DOT(v14, a1, b3)
// clang-format on

void matmulTile(size_t depth, const _Float16* b, const _Float16* a, void* out, size_t pitch) {
    assert(depth && depth % 8 == 0 && __riscv_vlenb() == 128);
    register vfloat32m8_t top asm("v0");
    register vfloat32m8_t bottom asm("v8");
    // Prime one bank, consume pairs, then drain the last one or two panels without reading past K.
    size_t panels = depth / 8;
    uintptr_t secondA, col1, col2, col3;
    asm volatile(
        "vsetvli t0, zero, e32, m8, ta, ma\n\t"
        "vmv.v.i v0, 0\n\t"
        "vmv.v.i v8, 0\n\t"
        "beqz %[panels], 4f\n\t"
        "vsetvli t0, zero, e16, m1, ta, ma\n\t"
        "vl4re16.v v20, (%[b])\n\t"
        "vl2re16.v v16, (%[a])\n\t"
        "addi %[b], %[b], 512\n\t"
        "addi %[a], %[a], 256\n\t"
        "addi %[secondA], %[a], 128\n\t"
        "addi %[col1], %[b], 128\n\t"
        "addi %[col2], %[b], 256\n\t"
        "addi %[col3], %[b], 384\n\t"
        "addi %[panels], %[panels], -1\n\t"
        "li t0, 2\n\t"
        "bltu %[panels], t0, 2f\n\t"
        ".p2align 4\n"
        "1:\n\t"
        MNN_IME2_NEXT(v16, v17, v20, v21, v22, v23, v18, v19, v24, v25, v26, v27)
        MNN_IME2_NEXT(v18, v19, v24, v25, v26, v27, v16, v17, v20, v21, v22, v23)
        "addi %[panels], %[panels], -2\n\t"
        "bgeu %[panels], t0, 1b\n"
        "2:\n\t"
        "beqz %[panels], 3f\n\t"
        MNN_IME2_NEXT(v16, v17, v20, v21, v22, v23, v18, v19, v24, v25, v26, v27)
        MNN_IME2_FINISH(v18, v19, v24, v25, v26, v27)
        "j 4f\n"
        "3:\n\t"
        MNN_IME2_FINISH(v16, v17, v20, v21, v22, v23)
        "4:"
        : "=&vr"(top), "=&vr"(bottom), [a] "+&r"(a), [b] "+&r"(b), [panels] "+&r"(panels),
          [secondA] "=&r"(secondA), [col1] "=&r"(col1), [col2] "=&r"(col2), [col3] "=&r"(col3)
        :
        : "t0", "memory", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27");

    if (!pitch) {
        auto dst = static_cast<_Float16*>(out);
        __riscv_vse16_v_f16m4(dst, __riscv_vfncvt_f_f_w_f16m4(top, 256), 256);
        __riscv_vse16_v_f16m4(dst + 256, __riscv_vfncvt_f_f_w_f16m4(bottom, 256), 256);
        return;
    }
    // vpack interprets the blocking unit relative to SEW.
    asm volatile("vsetvli t0, zero, e32, m1, ta, ma" : "+vr"(top), "+vr"(bottom) : : "t0");
    auto dst = static_cast<uint8_t*>(out);
    storeFourRows<0>(dst, pitch, top);
    storeFourRows<1>(dst + 4 * pitch, pitch, top);
    storeFourRows<0>(dst + 8 * pitch, pitch, bottom);
    storeFourRows<1>(dst + 12 * pitch, pitch, bottom);
}
#undef MNN_IME2_NEXT
#undef MNN_IME2_FINISH
#undef MNN_IME2_DOT
#undef MNN_IME2_LOAD
} // namespace MNNIme2
#endif
