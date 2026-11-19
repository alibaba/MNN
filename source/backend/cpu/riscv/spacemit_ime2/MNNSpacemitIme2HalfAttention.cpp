//
//  MNNSpacemitIme2HalfAttention.cpp
//  MNN
//
//  Created by MNN on 2026/11/18.
//  Copyright © 2026, Alibaba Group Holding Limited
//

#include "MNNSpacemitIme2Half.hpp"
#if defined(__riscv_zvfh)
#include "MNNRvvPrefill.hpp"
#include <riscv_vector.h>
#include <algorithm>
#include <atomic>
#include <cstring>
#include <limits>

extern "C" size_t MNNSpacemitIme2RunTcmTasks(size_t, void (*)(size_t, void*, size_t, void*), void*);
using namespace MNNIme2;
static inline void packDenseA(const float* src, size_t rs, _Float16* lo) {
    asm volatile(
        "vsetvli t0,zero,e32,m2,tu,mu\n\t"
        "vle32.v v16,(%[src])\n\t"
        "add %[src],%[src],%[rs]\n\t"
        "vle32.v v18,(%[src])\n\t"
        "add %[src],%[src],%[rs]\n\t"
        "vle32.v v20,(%[src])\n\t"
        "add %[src],%[src],%[rs]\n\t"
        "vle32.v v22,(%[src])\n\t"
        "add %[src],%[src],%[rs]\n\t"
        "vle32.v v24,(%[src])\n\t"
        "add %[src],%[src],%[rs]\n\t"
        "vle32.v v26,(%[src])\n\t"
        "add %[src],%[src],%[rs]\n\t"
        "vle32.v v28,(%[src])\n\t"
        "add %[src],%[src],%[rs]\n\t"
        "vle32.v v30,(%[src])\n\t"
        "add %[src],%[src],%[rs]\n\t"
        "vsetvli t0,zero,e16,m1,tu,mu\n\t"
        "vfncvt.f.f.w v0,v16\n\t"
        "vfncvt.f.f.w v1,v18\n\t"
        "vfncvt.f.f.w v2,v20\n\t"
        "vfncvt.f.f.w v3,v22\n\t"
        "vfncvt.f.f.w v4,v24\n\t"
        "vfncvt.f.f.w v5,v26\n\t"
        "vfncvt.f.f.w v6,v28\n\t"
        "vfncvt.f.f.w v7,v30\n\t"
        "vpack.vv v8,v0,v1,1\n\t"
        "vpack.vv v10,v2,v3,1\n\t"
        "vpack.vv v12,v4,v5,1\n\t"
        "vpack.vv v14,v6,v7,1\n\t"
        "vpack.vv v0,v8,v10,2\n\t"
        "vpack.vv v2,v9,v11,2\n\t"
        "vpack.vv v4,v12,v14,2\n\t"
        "vpack.vv v6,v13,v15,2\n\t"
        "vpack.vv v8,v0,v4,3\n\t"
        "vpack.vv v10,v1,v5,3\n\t"
        "vpack.vv v12,v2,v6,3\n\t"
        "vpack.vv v14,v3,v7,3\n\t"
        "vs1r.v v8,(%[lo])\n\t"
        "addi %[lo],%[lo],256\n\t"
        "vs1r.v v9,(%[lo])\n\t"
        "addi %[lo],%[lo],256\n\t"
        "vs1r.v v10,(%[lo])\n\t"
        "addi %[lo],%[lo],256\n\t"
        "vs1r.v v11,(%[lo])\n\t"
        "addi %[lo],%[lo],256\n\t"
        "vs1r.v v12,(%[lo])\n\t"
        "addi %[lo],%[lo],256\n\t"
        "vs1r.v v13,(%[lo])\n\t"
        "addi %[lo],%[lo],256\n\t"
        "vs1r.v v14,(%[lo])\n\t"
        "addi %[lo],%[lo],256\n\t"
        "vs1r.v v15,(%[lo])\n\t"
        "addi %[lo],%[lo],256\n\t"
        : [src] "+&r"(src), [lo] "+&r"(lo)
        : [rs] "r"(rs)
        : "memory", "t0", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14",
          "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29",
          "v30", "v31");
}
static inline void packDenseB(const float* src, size_t rs, _Float16* lo, _Float16* hi) {
    asm volatile(
        "vsetvli t0,zero,e32,m2,tu,mu\n\t"
        "vle32.v v16,(%[src])\n\t"
        "add %[src],%[src],%[rs]\n\t"
        "vle32.v v18,(%[src])\n\t"
        "add %[src],%[src],%[rs]\n\t"
        "vle32.v v20,(%[src])\n\t"
        "add %[src],%[src],%[rs]\n\t"
        "vle32.v v22,(%[src])\n\t"
        "add %[src],%[src],%[rs]\n\t"
        "vle32.v v24,(%[src])\n\t"
        "add %[src],%[src],%[rs]\n\t"
        "vle32.v v26,(%[src])\n\t"
        "add %[src],%[src],%[rs]\n\t"
        "vle32.v v28,(%[src])\n\t"
        "add %[src],%[src],%[rs]\n\t"
        "vle32.v v30,(%[src])\n\t"
        "add %[src],%[src],%[rs]\n\t"
        "vsetvli t0,zero,e16,m1,tu,mu\n\t"
        "vfncvt.f.f.w v0,v16\n\t"
        "vfncvt.f.f.w v1,v18\n\t"
        "vfncvt.f.f.w v2,v20\n\t"
        "vfncvt.f.f.w v3,v22\n\t"
        "vfncvt.f.f.w v4,v24\n\t"
        "vfncvt.f.f.w v5,v26\n\t"
        "vfncvt.f.f.w v6,v28\n\t"
        "vfncvt.f.f.w v7,v30\n\t"
        "vpack.vv v8,v0,v1,0\n\t"
        "vpack.vv v10,v2,v3,0\n\t"
        "vpack.vv v12,v4,v5,0\n\t"
        "vpack.vv v14,v6,v7,0\n\t"
        "vsetvli t0,zero,e32,m1,tu,mu\n\t"
        "vpack.vv v0,v8,v10,0\n\t"
        "vpack.vv v2,v9,v11,0\n\t"
        "vpack.vv v4,v12,v14,0\n\t"
        "vpack.vv v6,v13,v15,0\n\t"
        "vsetvli t0,zero,e64,m1,tu,mu\n\t"
        "vpack.vv v8,v0,v4,0\n\t"
        "vpack.vv v10,v1,v5,0\n\t"
        "vpack.vv v12,v2,v6,0\n\t"
        "vpack.vv v14,v3,v7,0\n\t"
        "vs4r.v v8,(%[lo])\n\t"
        "vs4r.v v12,(%[hi])\n\t"
        : [src] "+&r"(src)
        : [lo] "r"(lo), [hi] "r"(hi), [rs] "r"(rs)
        : "memory", "t0", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14",
          "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29",
          "v30", "v31");
}

static bool gatedProjection(const float* state, const float* q, const float* k, float* outQ, float* outK, size_t pitch,
                            void* scratch, size_t bytes) {
    if (!scratch || bytes < 40960 || __riscv_vlenb() != 128)
        return false;
    HalfRangeCheck range;
    auto aq = static_cast<_Float16*>(scratch);
    auto ak = aq + 16 * 128;
    auto b = ak + 16 * 128;
    for (size_t d = 0; d < 128; d += 64)
        for (size_t m = 0; m < 16; m += 8) {
            packDenseA(q + m * 128 + d, 512, aq + d * 16 + m * 8);
            packDenseA(k + m * 128 + d, 512, ak + d * 16 + m * 8);
        }
    for (size_t c = 0; c < 128; c += 64)
        for (size_t d = 0; d < 128; d += 8) {
            packDenseB(state + d * 128 + c, 512, b + c * 128 + d * 32, b + (c + 32) * 128 + d * 32);
        }
    for (size_t c = 0; c < 128; c += 32) {
        matmulTile(128, b + c * 128, aq, outQ + c, pitch * sizeof(float));
        matmulTile(128, b + c * 128, ak, outK + c, 128 * sizeof(float));
    }
    return !range.failed();
}

namespace {
struct CausalAttention {
    const float* q;
    const float* const* key;
    const float* const* value;
    float* output;
    _Float16* b;
    size_t len, dim, heads, kv, threads;
    float scale;
    std::atomic<bool> ready{true};
};

static void packKV(size_t id, void* tcm, size_t bytes, void* data) {
    auto& p = *static_cast<CausalAttention*>(data);
    if (!tcm || __riscv_vlenb() != 128 || bytes < 192 * std::max(p.len, p.dim) + 256 * (p.len + p.dim)) {
        p.ready.store(false);
        return;
    }
    HalfRangeCheck range;
    // Keys are [token/4,dim,4], values [token/64,dim/4,64,4].
    for (size_t block = id; block < p.kv * (p.len / 32 + p.dim / 32); block += p.threads) {
        const size_t per = p.len / 32 + p.dim / 32, h = block / per, t = block % per;
        auto base = p.b + h * p.len * p.dim * 2;
        if (t < p.len / 32) {
            auto to = base + t * 32 * p.dim;
            for (size_t k = 0; k < p.dim; k += 8)
                for (size_t n = 0; n < 32; ++n) {
                    auto from = p.key[h] + ((t * 32 + n) / 4 * p.dim + k) * 4 + n % 4;
                    auto vals = __riscv_vlse32_v_f32m1(from, 16, 8);
                    __riscv_vse16_v_f16mf2(to + k * 32 + n * 8, __riscv_vfncvt_f_f_w_f16mf2(vals, 8), 8);
                }
        } else {
            const size_t d = (t - p.len / 32) * 32;
            auto to = base + p.len * p.dim + d * p.len;
            for (size_t k = 0; k < p.len; k += 8)
                for (size_t n = 0; n < 32; ++n) {
                    auto from = p.value[h] + k / 64 * p.dim * 64 + ((d + n) / 4 * 64 + k % 64) * 4 + n % 4;
                    auto vals = __riscv_vlse32_v_f32m1(from, 16, 8);
                    __riscv_vse16_v_f16mf2(to + k * 32 + n * 8, __riscv_vfncvt_f_f_w_f16mf2(vals, 8), 8);
                }
        }
    }
    if (range.failed()) p.ready.store(false);
}

static void attentionHead(size_t id, void* tcm, size_t bytes, void* data) {
    auto& p = *static_cast<CausalAttention*>(data);
    const size_t maxK = std::max(p.dim, p.len);
    if (!tcm || __riscv_vlenb() != 128 || bytes < 192 * maxK + 256 * (p.len + p.dim)) {
        p.ready.store(false);
        return;
    }
    HalfRangeCheck range;
    auto a = static_cast<_Float16*>(tcm);
    auto b = a + 64 * maxK;
    auto scores = reinterpret_cast<float*>(b + 32 * maxK);
    auto out = scores + 64 * p.len;
    for (size_t head = id; head < p.heads; head += p.threads) {
        auto keys = p.b + (head / (p.heads / p.kv)) * 2 * p.len * p.dim;
        auto values = keys + p.len * p.dim;
        for (size_t start = 0; start < p.len; start += 64) {
            const size_t klen = start + 64;
            for (size_t m = 0; m < 64; m += 16)
                for (size_t d = 0; d < p.dim; d += 64)
                    for (size_t h = 0; h < 16; h += 8) {
                        auto q = p.q + ((start + m + h) * p.heads + head) * p.dim + d;
                        packDenseA(q, p.dim * p.heads * sizeof(float), a + m * p.dim + d * 16 + h * 8);
                    }
            for (size_t n = 0; n < klen; n += 32) {
                std::memcpy(b, keys + n * p.dim, p.dim * 32 * sizeof(_Float16));
                for (size_t m = 0; m < 64; m += 16)
                    matmulTile(p.dim, b, a + m * p.dim, scores + m * klen + n, klen * sizeof(float));
            }
            for (size_t m = 0; m < 64; ++m)
                MNNIme2MaskedSoftmax(scores + m * klen, start + m + 1, klen, p.scale);
            for (size_t m = 0; m < 64; m += 16)
                for (size_t k = 0; k < klen; k += 64)
                    for (size_t h = 0; h < 16; h += 8) {
                        packDenseA(scores + (m + h) * klen + k, klen * sizeof(float), a + m * klen + k * 16 + h * 8);
                    }
            for (size_t n = 0; n < p.dim; n += 32) {
                std::memcpy(b, values + n * p.len, klen * 32 * sizeof(_Float16));
                for (size_t m = 0; m < 64; m += 16)
                    matmulTile(klen, b, a + m * klen, out + m * p.dim + n, p.dim * sizeof(float));
            }
            for (size_t d = 0; d < p.dim; d += 4) {
                auto target = p.output + (head * p.dim + d) * p.len + start * 4;
                for (size_t m = 0; m < 64; ++m)
                    std::memcpy(target + m * 4, out + m * p.dim + d, 4 * sizeof(float));
            }
        }
    }
    if (range.failed()) p.ready.store(false);
}
} // namespace

bool MNNIme2CausalAttention(const float* q, const float* const* keys, const float* const* values, float* dst,
                            size_t len, size_t dim, size_t heads, size_t kv, size_t threads, float scale) {
    CausalAttention context;
    if (!q || !keys || !values || !dst || dim != 256 || len < 64 || len > 512 || len % 64 ||
        !kv || !heads || heads % kv || !threads || threads > 8 ||
        heads > std::numeric_limits<size_t>::max() / (len * dim * sizeof(float))) return false;
    auto packed = halfWorkspace(2 * len * dim * kv, 128);
    if (!packed) return false;
    context.b = packed.get();
    context.q = q;
    context.key = keys;
    context.value = values;
    context.output = dst;
    context.len = len;
    context.dim = dim;
    context.heads = heads;
    context.kv = kv;
    context.threads = threads;
    context.scale = scale;
    if (MNNSpacemitIme2RunTcmTasks(threads, packKV, &context) != threads || !context.ready.load())
        return false;
    return MNNSpacemitIme2RunTcmTasks(threads, attentionHead, &context) == threads && context.ready.load();
}

static void gatedStateUpdate(float* state, const float* k, const float* residual, const float* decay, void* scratch) {
    auto a = static_cast<_Float16*>(scratch);
    auto b = a + 128 * 16;
    auto sh = b + 16 * 128;
    auto scaled = reinterpret_cast<float*>(sh + 512);
    auto tile = scaled + 16 * 128;
    HalfRangeCheck range;
    float product = 1.0f;
    for (size_t t = 16; t--;) {
        auto row = __riscv_vle32_v_f32m4(residual + t * 128, 128);
        __riscv_vse32_v_f32m4(scaled + t * 128, __riscv_vfmul_vf_f32m4(row, product, 128), 128);
        product *= decay[t];
    }
    for (size_t t = 0; t < 16; t += 8)
        for (size_t c = 0; c < 128; c += 64) {
            packDenseB(scaled + t * 128 + c, 512, b + c * 16 + t * 32, b + (c + 32) * 16 + t * 32);
            packDenseB(k + t * 128 + c, 512, sh, sh + 256);
            for (size_t g = 0; g < 4; ++g)
                std::memcpy(a + (c + g * 16) * 16 + t * 16, sh + g * 128, 128 * sizeof(_Float16));
        }
    if (range.failed()) {
        // Residuals have already been solved. Commit exactly one FP32 update
        // per token, before writing any half-derived updates into the live state.
        for (size_t t = 0; t < 16; ++t)
            MNNDecayRankOneUpdate_RVV(state, k + t * 128, residual + t * 128, decay[t], 128, 128);
        return;
    }
    for (size_t m = 0; m < 128; m += 16)
        for (size_t n = 0; n < 128; n += 32) {
            matmulTile(16, b + n * 16, a + m * 16, tile, 32 * sizeof(float));
            for (size_t row = 0; row < 16; ++row) {
                auto dst = state + (m + row) * 128 + n;
                auto delta = __riscv_vle32_v_f32m1(tile + row * 32, 32);
                auto updated = __riscv_vfmacc_vf_f32m1(delta, product, __riscv_vle32_v_f32m1(dst, 32), 32);
                __riscv_vse32_v_f32m1(dst, updated, 32);
            }
        }
}

bool MNNIme2GatedPrefill(float* state, const float* q, const float* k, float* v, float* projection, float* output,
                         const float* decay, const float* beta, size_t stride, void* scratch, size_t scratchBytes) {
    static_assert(MNN::kRvvGatedDeltaChunk == 16 && MNN::kRvvGatedDeltaHeadDim == 128, "IME2 uses 16x128 panels");
    if (!halfPrefillEnabled() || !gatedProjection(state, q, k, output, projection, stride, scratch, scratchBytes))
        return false;
    MNNGatedDeltaChunk_RVV(state, q, k, v, projection, output, decay, beta, stride, true);
    gatedStateUpdate(state, k, v, decay, scratch);
    return true;
}
#else
bool MNNIme2GatedPrefill(float*, const float*, const float*, float*, float*, float*, const float*, const float*,
                         size_t, void*, size_t) {
    return false;
}
#endif
