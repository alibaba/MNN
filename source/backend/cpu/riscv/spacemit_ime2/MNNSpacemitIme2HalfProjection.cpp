//
//  MNNSpacemitIme2HalfProjection.cpp
//  MNN
//
//  Created by MNN on 2026/11/18.
//  Copyright © 2026, Alibaba Group Holding Limited
//

#include "MNNSpacemitIme2Half.hpp"
#if defined(__riscv_zvfh)
#include <riscv_vector.h>
#include <algorithm>
#include <atomic>
#include <cstring>
#include <limits>

extern "C" size_t MNNSpacemitIme2RunTcmTasks(size_t, void (*)(size_t, void*, size_t, void*), void*);
using namespace MNNIme2;
namespace {
struct Projection {
    const float* src;
    float* dst;
    const float* bias;
    const float* clamp;
    const Block64Weight* weight;
    size_t rows, depth, cols, workers, groups, paddedRows;
    _Float16* partial;
    std::atomic<bool> ok{true};
};
static void packA(_Float16* dst, const float* src, size_t rows, size_t depth, size_t validRows) {
    const size_t vl = validRows * 4;
    const auto zero = __riscv_vfmv_v_f_f16m1(0, 64);
    for (size_t k = 0; k < depth; k += 8) {
        auto lo = __riscv_vfncvt_f_f_w_f16m1_tu(zero, __riscv_vle32_v_f32m2(src + k * rows, vl), vl);
        auto hi = __riscv_vfncvt_f_f_w_f16m1_tu(zero, __riscv_vle32_v_f32m2(src + (k + 4) * rows, vl), vl);
        auto l = __riscv_vreinterpret_v_u16m1_u64m1(__riscv_vreinterpret_v_f16m1_u16m1(lo));
        auto h = __riscv_vreinterpret_v_u16m1_u64m1(__riscv_vreinterpret_v_f16m1_u16m1(hi));
        vuint64m2_t pair;
        asm volatile("vsetvli t0, zero, e64, m1, ta, ma\n\tvpack.vv %0, %1, %2, 0"
                     : "=&vr"(pair)
                     : "vr"(l), "vr"(h)
                     : "t0");
        __riscv_vse64_v_u64m2(reinterpret_cast<uint64_t*>(dst + k * 16), pair, 32);
    }
}
static void packB(_Float16* dst, const Block64Weight* src, size_t depth) {
    constexpr size_t vl = 128;
    const auto idx = __riscv_vsrl_vx_u16m2(__riscv_vid_v_u16m2(vl), 3, vl);
    for (size_t k = 0; k < depth; k += 64, ++src) {
        auto s0 = __riscv_vrgather_vv_f16m2(__riscv_vle16_v_f16m2(src->scale, 16), idx, vl);
        auto s1 = __riscv_vrgather_vv_f16m2(__riscv_vle16_v_f16m2(src->scale + 16, 16), idx, vl);
        auto o0 = __riscv_vrgather_vv_f16m2(__riscv_vle16_v_f16m2(src->offset, 16), idx, vl);
        auto o1 = __riscv_vrgather_vv_f16m2(__riscv_vle16_v_f16m2(src->offset + 16, 16), idx, vl);
        for (size_t sub = 0; sub < 8; ++sub) {
            auto q = __riscv_vle8_v_u8m1(src->quants[sub], vl);
            auto lo = __riscv_vfwcvt_f_xu_v_f16m2(__riscv_vand_vx_u8m1(q, 15, vl), vl);
            auto hi = __riscv_vfwcvt_f_xu_v_f16m2(__riscv_vsrl_vx_u8m1(q, 4, vl), vl);
            __riscv_vse16_v_f16m2(dst, __riscv_vfmacc_vv_f16m2(o0, lo, s0, vl), vl);
            __riscv_vse16_v_f16m2(dst + vl, __riscv_vfmacc_vv_f16m2(o1, hi, s1, vl), vl);
            dst += 256;
        }
    }
}
// Each group owns a disjoint K slice. Workers within it split whole 16-row tiles.
static void run(size_t id, void* tcm, size_t bytes, void* user) {
    auto& p = *static_cast<Projection*>(user);
    if (!tcm || __riscv_vlenb() != 128 || bytes < 512 * sizeof(float)) {
        p.ok.store(false);
        return;
    }
    HalfRangeCheck range;
    const size_t group = id % p.groups;
    const size_t owners = p.workers / p.groups;
    const size_t owner = id / p.groups;
    const size_t k0 = group * p.depth / p.groups;
    const size_t kk = p.depth / p.groups;
    const size_t mb = ((p.paddedRows / 16) * owner / owners) * 16;
    const size_t me = ((p.paddedRows / 16) * (owner + 1) / owners) * 16;
    const size_t capacity = bytes / (sizeof(_Float16) * kk);
    if (capacity < 48) {
        p.ok.store(false);
        return;
    }
    const size_t tileM = std::min(p.paddedRows, (capacity - 32) / 16 * 16);
    auto a = static_cast<_Float16*>(tcm);
    auto b = a + tileM * kk;
    _Float16* result = p.partial + group * 512;
    for (size_t row = mb; row < me; row += tileM) {
        const size_t mm = std::min(tileM, me - row);
        for (size_t m = 0; m < mm; m += 16)
            packA(a + m * kk, p.src + k0 * p.rows + (row + m) * 4, p.rows, kk, std::min<size_t>(16, p.rows - row - m));
        for (size_t i = 0; i < p.cols / 32; ++i) {
            const size_t n = 32 * ((i + id) % (p.cols / 32));
            packB(b, p.weight + (n / 32 * p.depth + k0) / 64, kk);
            for (size_t m = 0; m < mm; m += 16) {
                matmulTile(kk, b, a + m * kk,
                           result + ((n / 32) * (p.paddedRows / 16) + (row + m) / 16) * 512 * p.groups);
            }
        }
    }
    if (range.failed()) p.ok.store(false);
}
static inline void reduceTile(const _Float16* raw, float* out, size_t stride, size_t groups) {
    asm volatile(
        "vsetvli t0,zero,e16,m4,ta,ma\n\t"
        "vle16.v v0,(%[raw])\n\t"
        "addi %[raw],%[raw],512\n\t"
        "vle16.v v4,(%[raw])\n\t"
        "addi %[raw],%[raw],512\n\t"
        "vfwcvt.f.f.v v16,v0\n\t"
        "vfwcvt.f.f.v v24,v4\n\t"
        "addi %[groups],%[groups],-1\n\t"
        "beqz %[groups],.L_reduced_%=\n\t"
        ".L_reduce_%=:\n\t"
        "vle16.v v0,(%[raw])\n\t"
        "addi %[raw],%[raw],512\n\t"
        "vle16.v v4,(%[raw])\n\t"
        "addi %[raw],%[raw],512\n\t"
        "vfwadd.wv v16,v16,v0\n\t"
        "vfwadd.wv v24,v24,v4\n\t"
        "addi %[groups],%[groups],-1\n\t"
        "bnez %[groups],.L_reduce_%=\n\t"
        ".L_reduced_%=:\n\t"
        "vsetvli t0,zero,e32,m1,tu,mu\n\t"
        "vpack.vv v0,v16,v17,1\n\t"
        "vpack.vv v16,v0,v1,1\n\t"
        "vpack.vv v0,v16,v17,1\n\t"
        "vpack.vv v2,v18,v19,1\n\t"
        "vpack.vv v18,v2,v3,1\n\t"
        "vpack.vv v2,v18,v19,1\n\t"
        "vpack.vv v4,v20,v21,1\n\t"
        "vpack.vv v20,v4,v5,1\n\t"
        "vpack.vv v4,v20,v21,1\n\t"
        "vpack.vv v6,v22,v23,1\n\t"
        "vpack.vv v22,v6,v7,1\n\t"
        "vpack.vv v6,v22,v23,1\n\t"
        "vpack.vv v8,v24,v25,1\n\t"
        "vpack.vv v24,v8,v9,1\n\t"
        "vpack.vv v8,v24,v25,1\n\t"
        "vpack.vv v10,v26,v27,1\n\t"
        "vpack.vv v26,v10,v11,1\n\t"
        "vpack.vv v10,v26,v27,1\n\t"
        "vpack.vv v12,v28,v29,1\n\t"
        "vpack.vv v28,v12,v13,1\n\t"
        "vpack.vv v12,v28,v29,1\n\t"
        "vpack.vv v14,v30,v31,1\n\t"
        "vpack.vv v30,v14,v15,1\n\t"
        "vpack.vv v14,v30,v31,1\n\t"
        "vs1r.v v0,(%[out])\n\t"
        "addi t1,%[out],128\n\t"
        "vs1r.v v8,(t1)\n\t"
        "add %[out],%[out],%[stride]\n\t"
        "vs1r.v v1,(%[out])\n\t"
        "addi t1,%[out],128\n\t"
        "vs1r.v v9,(t1)\n\t"
        "add %[out],%[out],%[stride]\n\t"
        "vs1r.v v2,(%[out])\n\t"
        "addi t1,%[out],128\n\t"
        "vs1r.v v10,(t1)\n\t"
        "add %[out],%[out],%[stride]\n\t"
        "vs1r.v v3,(%[out])\n\t"
        "addi t1,%[out],128\n\t"
        "vs1r.v v11,(t1)\n\t"
        "add %[out],%[out],%[stride]\n\t"
        "vs1r.v v4,(%[out])\n\t"
        "addi t1,%[out],128\n\t"
        "vs1r.v v12,(t1)\n\t"
        "add %[out],%[out],%[stride]\n\t"
        "vs1r.v v5,(%[out])\n\t"
        "addi t1,%[out],128\n\t"
        "vs1r.v v13,(t1)\n\t"
        "add %[out],%[out],%[stride]\n\t"
        "vs1r.v v6,(%[out])\n\t"
        "addi t1,%[out],128\n\t"
        "vs1r.v v14,(t1)\n\t"
        "add %[out],%[out],%[stride]\n\t"
        "vs1r.v v7,(%[out])\n\t"
        "addi t1,%[out],128\n\t"
        "vs1r.v v15,(t1)\n\t"
        "add %[out],%[out],%[stride]\n\t"
        : [raw] "+&r"(raw), [out] "+&r"(out), [groups] "+&r"(groups)
        : [stride] "r"(stride)
        : "memory", "t0", "t1", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13",
          "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28",
          "v29", "v30", "v31");
}
static void reduce(size_t id, void* tcm, size_t bytes, void* user) {
    auto& p = *static_cast<Projection*>(user);
    if (!tcm || __riscv_vlenb() != 128 || bytes < 512 * sizeof(float)) {
        p.ok.store(false);
        return;
    }
    float* tile = static_cast<float*>(tcm);
    for (size_t n = id * 32; n < p.cols; n += 32 * p.workers) {
        bool bias = false;
        for (size_t j = 0; j < 32 && p.bias; ++j)
            bias |= p.bias[n + j] != 0;
        for (size_t block = 0; block < p.paddedRows / 16; ++block) {
            const auto m = ((block + id) % (p.paddedRows / 16)) * 16;
            auto base = p.partial + ((n / 32) * (p.paddedRows / 16) + m / 16) * p.groups * 512;
            if (p.rows - m >= 16) {
                reduceTile(base, p.dst + n * p.rows + m * 4, p.rows * 4 * sizeof(float), p.groups);
            } else {
                reduceTile(base, tile, 64 * sizeof(float), p.groups);
                for (size_t j = 0; j < 32; j += 4)
                    std::memcpy(p.dst + (n + j) * p.rows + m * 4, tile + j * 16, 4 * (p.rows - m) * sizeof(float));
            }
        }
        if (bias || (p.clamp && (p.clamp[0] > -std::numeric_limits<float>::max() ||
                                 p.clamp[1] < std::numeric_limits<float>::max()))) {
            for (size_t j = 0; j < 32; j += 4) {
                auto lane = __riscv_vand_vx_u32m4(__riscv_vid_v_u32m4(128), 3, 128);
                auto add = bias ? __riscv_vrgather_vv_f32m4(__riscv_vle32_v_f32m4(p.bias + n + j, 4), lane, 128)
                                : __riscv_vfmv_v_f_f32m4(0, 128);
                for (size_t r = 0; r < 4 * p.rows;) {
                    auto ptr = p.dst + (n + j) * p.rows + r;
                    auto vl = __riscv_vsetvl_e32m4(4 * p.rows - r);
                    auto x = __riscv_vfadd_vv_f32m4(__riscv_vle32_v_f32m4(ptr, vl), add, vl);
                    if (p.clamp)
                        x = __riscv_vfmin_vf_f32m4(__riscv_vfmax_vf_f32m4(x, p.clamp[0], vl), p.clamp[1], vl);
                    __riscv_vse32_v_f32m4(ptr, x, vl);
                    r += vl;
                }
            }
        }
    }
}
} // namespace
static size_t projectionGroups(size_t rows, size_t depth, size_t cols, size_t workers) {
    if (rows <= 1 || rows > 2048 || !depth || depth % 64 || !cols || cols % 32 || !workers || workers > 8)
        return 0;
    size_t groups = cols <= 512 ? workers : std::min(workers, std::max<size_t>(1, depth / 512));
    while (groups > 1 && (workers % groups || depth % (64 * groups)))
        --groups;
    // Gate before allocating a cached weight or split-K workspace for large vocabulary projections.
    const size_t paddedRows = (rows + 15) / 16 * 16;
    return cols <= (64u * 1024 * 1024 / sizeof(_Float16)) / paddedRows / groups ? groups : 0;
}
size_t MNNIme2HalfProjectionBytes(size_t rows, size_t depth, size_t cols, size_t workers) {
    const auto groups = projectionGroups(rows, depth, cols, workers);
    // The CPU planner can place a chunk at a suballocation offset. Allow for
    // the native tile alignment independently of the arena's base alignment.
    return groups ? ((rows + 15) / 16 * 16) * cols * groups * sizeof(_Float16) + 255 : 0;
}
bool MNNIme2HalfProjection(const float* src, float* dst, const float* bias, const float* clamp, const uint8_t* weight,
                           size_t rows, size_t depth, size_t cols, size_t workers, void* workspace) {
    Projection p;
    p.src = src;
    p.dst = dst;
    p.bias = bias;
    p.clamp = clamp;
    p.weight = reinterpret_cast<const Block64Weight*>(weight);
    p.rows = rows;
    p.cols = cols;
    p.depth = depth;
    p.workers = workers;
    p.groups = projectionGroups(rows, depth, cols, workers);
    if (!src || !dst || !weight || !workspace || !p.groups)
        return false;
    p.paddedRows = (rows + 15) / 16 * 16;
    p.partial = reinterpret_cast<_Float16*>((reinterpret_cast<uintptr_t>(workspace) + 255) & ~uintptr_t(255));
    if (MNNSpacemitIme2RunTcmTasks(workers, run, &p) != workers || !p.ok.load())
        return false;
    return MNNSpacemitIme2RunTcmTasks(workers, reduce, &p) == workers && p.ok.load();
}

#endif
