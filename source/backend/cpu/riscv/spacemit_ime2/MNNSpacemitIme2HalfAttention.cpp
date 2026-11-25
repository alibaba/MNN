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
    size_t len, padLen, dim, heads, kv, threads;
    float scale;
    std::atomic<bool> ready{true};
};

// Constant TCM requirement for FlashAttention (Bq=64, Bkv=64, D=256): ~142 KB < 384 KB.
static constexpr size_t kFlashAttnTcmBytes = 152 * 1024;

static void packKV(size_t id, void* tcm, size_t bytes, void* data) {
    auto& p = *static_cast<CausalAttention*>(data);
    if (!tcm || __riscv_vlenb() != 128 || bytes < kFlashAttnTcmBytes) {
        p.ready.store(false);
        return;
    }
    HalfRangeCheck range;
    // Keys are [UP_DIV(len,4), dim, 4], values [UP_DIV(len,64), dim/4, 64, 4].
    const size_t kTiles = p.padLen / 32;
    const size_t vTiles = p.dim / 32;
    const size_t per = kTiles + vTiles;
    for (size_t block = id; block < p.kv * per; block += p.threads) {
        const size_t h = block / per, t = block % per;
        auto base = p.b + h * p.padLen * p.dim * 2;
        if (t < kTiles) {
            auto to = base + t * 32 * p.dim;
            for (size_t k = 0; k < p.dim; k += 8) {
                for (size_t n = 0; n < 32; ++n) {
                    const size_t tok = t * 32 + n;
                    auto dst = to + k * 32 + n * 8;
                    if (tok < p.len) {
                        auto from = p.key[h] + (tok / 4 * p.dim + k) * 4 + tok % 4;
                        auto vals = __riscv_vlse32_v_f32m1(from, 16, 8);
                        __riscv_vse16_v_f16mf2(dst, __riscv_vfncvt_f_f_w_f16mf2(vals, 8), 8);
                    } else {
                        std::memset(dst, 0, 8 * sizeof(_Float16));
                    }
                }
            }
        } else {
            const size_t d = (t - kTiles) * 32;
            auto to = base + p.padLen * p.dim + d * p.padLen;
            for (size_t k = 0; k < p.padLen; k += 8) {
                if (k + 8 <= p.len) {
                    for (size_t n = 0; n < 32; ++n) {
                        auto from = p.value[h] + k / 64 * p.dim * 64 + ((d + n) / 4 * 64 + k % 64) * 4 + n % 4;
                        auto vals = __riscv_vlse32_v_f32m1(from, 16, 8);
                        __riscv_vse16_v_f16mf2(to + k * 32 + n * 8, __riscv_vfncvt_f_f_w_f16mf2(vals, 8), 8);
                    }
                } else if (k < p.len) {
                    const size_t validK = p.len - k;
                    for (size_t n = 0; n < 32; ++n) {
                        auto dst = to + k * 32 + n * 8;
                        auto from = p.value[h] + k / 64 * p.dim * 64 + ((d + n) / 4 * 64 + k % 64) * 4 + n % 4;
                        auto vals = __riscv_vlse32_v_f32m1(from, 16, validK);
                        __riscv_vse16_v_f16mf2(dst, __riscv_vfncvt_f_f_w_f16mf2(vals, validK), validK);
                        std::memset(dst + validK, 0, (8 - validK) * sizeof(_Float16));
                    }
                } else {
                    std::memset(to + k * 32, 0, 32 * 8 * sizeof(_Float16));
                }
            }
        }
    }
    if (range.failed()) p.ready.store(false);
}

// Online-softmax FlashAttention (Bq=64, Bkv=64, D=256) in constant 142 KB TCM,
// supporting arbitrary sequence lengths without 64-alignment or 512-length cliffs.
static void attentionHead(size_t id, void* tcm, size_t bytes, void* data) {
    auto& p = *static_cast<CausalAttention*>(data);
    if (!tcm || __riscv_vlenb() != 128 || bytes < kFlashAttnTcmBytes) {
        p.ready.store(false);
        return;
    }
    HalfRangeCheck range;
    constexpr size_t Bq = 64, Bkv = 64;
    auto aQ = static_cast<_Float16*>(tcm);                // 64 * 256 f16 = 32 KB
    auto aP = aQ + Bq * p.dim;                            // 64 * 64  f16 =  8 KB
    auto bKV = aP + Bq * Bkv;                             // 32 * 256 f16 = 16 KB
    auto scores = reinterpret_cast<float*>(bKV + 32 * p.dim); // 64 * 64  f32 = 16 KB
    auto out = scores + Bq * Bkv;                         // 64 * 256 f32 = 64 KB
    auto tile = out + Bq * p.dim;                         // 16 * 32  f32 =  2 KB
    float rowMax[Bq], rowSum[Bq];

    for (size_t head = id; head < p.heads; head += p.threads) {
        auto keys = p.b + (head / (p.heads / p.kv)) * 2 * p.padLen * p.dim;
        auto values = keys + p.padLen * p.dim;
        for (size_t start = 0; start < p.len; start += Bq) {
            const size_t qRows = std::min<size_t>(Bq, p.len - start);
            if (qRows == Bq) {
                for (size_t m = 0; m < Bq; m += 16)
                    for (size_t d = 0; d < p.dim; d += 64)
                        for (size_t h = 0; h < 16; h += 8) {
                            auto q = p.q + ((start + m + h) * p.heads + head) * p.dim + d;
                            packDenseA(q, p.dim * p.heads * sizeof(float), aQ + m * p.dim + d * 16 + h * 8);
                        }
            } else {
                // Stage partial last Q block into dead `out` buffer with zero padding.
                for (size_t m = 0; m < qRows; ++m) {
                    std::memcpy(out + m * p.dim, p.q + ((start + m) * p.heads + head) * p.dim,
                                p.dim * sizeof(float));
                }
                std::memset(out + qRows * p.dim, 0, (Bq - qRows) * p.dim * sizeof(float));
                for (size_t m = 0; m < Bq; m += 16)
                    for (size_t d = 0; d < p.dim; d += 64)
                        for (size_t h = 0; h < 16; h += 8) {
                            packDenseA(out + (m + h) * p.dim + d, p.dim * sizeof(float),
                                       aQ + m * p.dim + d * 16 + h * 8);
                        }
            }
            for (size_t m = 0; m < Bq; ++m) {
                rowMax[m] = -std::numeric_limits<float>::max();
                rowSum[m] = 0.0f;
            }
            std::memset(out, 0, Bq * p.dim * sizeof(float));

            const size_t kvLimit = start + qRows;
            for (size_t kvStart = 0; kvStart < kvLimit; kvStart += Bkv) {
                for (size_t n = 0; n < Bkv; n += 32) {
                    std::memcpy(bKV, keys + (kvStart + n) * p.dim, p.dim * 32 * sizeof(_Float16));
                    for (size_t m = 0; m < Bq; m += 16) {
                        matmulTile(p.dim, bKV, aQ + m * p.dim, scores + m * Bkv + n, Bkv * sizeof(float));
                    }
                }
                for (size_t m = 0; m < qRows; ++m) {
                    const size_t qPos = start + m;
                    const int valid = qPos >= kvStart ? static_cast<int>(std::min<size_t>(Bkv, qPos - kvStart + 1)) : 0;
                    float alpha = 1.0f;
                    MNNIme2OnlineSoftmaxBlock(scores + m * Bkv, valid, Bkv, p.scale, rowMax[m], rowSum[m], alpha);
                    if (alpha != 1.0f && alpha != 0.0f) {
                        float* oRow = out + m * p.dim;
                        for (size_t d = 0; d < p.dim; d += 128) {
                            auto v = __riscv_vle32_v_f32m4(oRow + d, 128);
                            __riscv_vse32_v_f32m4(oRow + d, __riscv_vfmul_vf_f32m4(v, alpha, 128), 128);
                        }
                    }
                }
                if (qRows < Bq) {
                    std::memset(scores + qRows * Bkv, 0, (Bq - qRows) * Bkv * sizeof(float));
                }
                for (size_t m = 0; m < Bq; m += 16)
                    for (size_t h = 0; h < 16; h += 8) {
                        packDenseA(scores + (m + h) * Bkv, Bkv * sizeof(float), aP + m * Bkv + h * 8);
                    }
                for (size_t n = 0; n < p.dim; n += 32) {
                    std::memcpy(bKV, values + n * p.padLen + kvStart * 32, Bkv * 32 * sizeof(_Float16));
                    for (size_t m = 0; m < Bq; m += 16) {
                        matmulTile(Bkv, bKV, aP + m * Bkv, tile, 32 * sizeof(float));
                        for (size_t r = 0; r < 16; ++r) {
                            float* dst = out + (m + r) * p.dim + n;
                            auto acc = __riscv_vle32_v_f32m1(dst, 32);
                            auto inc = __riscv_vle32_v_f32m1(tile + r * 32, 32);
                            __riscv_vse32_v_f32m1(dst, __riscv_vfadd_vv_f32m1(acc, inc, 32), 32);
                        }
                    }
                }
            }
            for (size_t m = 0; m < qRows; ++m) {
                const float invSum = 1.0f / rowSum[m];
                float* oRow = out + m * p.dim;
                for (size_t d = 0; d < p.dim; d += 128) {
                    auto v = __riscv_vle32_v_f32m4(oRow + d, 128);
                    __riscv_vse32_v_f32m4(oRow + d, __riscv_vfmul_vf_f32m4(v, invSum, 128), 128);
                }
            }
            for (size_t d = 0; d < p.dim; d += 4) {
                auto target = p.output + (head * p.dim + d) * p.len + start * 4;
                for (size_t m = 0; m < qRows; ++m) {
                    std::memcpy(target + m * 4, out + m * p.dim + d, 4 * sizeof(float));
                }
            }
        }
    }
    if (range.failed()) p.ready.store(false);
}
} // namespace

bool MNNIme2CausalAttention(const float* q, const float* const* keys, const float* const* values, float* dst,
                            size_t len, size_t dim, size_t heads, size_t kv, size_t threads, float scale) {
    CausalAttention context;
    if (!q || !keys || !values || !dst || dim != 256 || len < 16 || !kv || !heads || heads % kv || !threads ||
        threads > 8) {
        return false;
    }
    const size_t padLen = (len + 63) & ~size_t(63);
    if (heads > std::numeric_limits<size_t>::max() / (padLen * dim * sizeof(float))) {
        return false;
    }
    auto packed = halfWorkspace(2 * padLen * dim * kv, 128);
    if (!packed) return false;
    context.b = packed.get();
    context.q = q;
    context.key = keys;
    context.value = values;
    context.output = dst;
    context.len = len;
    context.padLen = padLen;
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

// Pack 8 rows x 64 depth of a row-major matrix as transposed B panels (stride 512 B per 8-depth block).
static inline void packDenseBTransposed(const float* src, size_t rs, _Float16* dst) {
    asm volatile(
        "vsetvli t0,zero,e32,m2,tu,mu\n\t"
        "vle32.v v16,(%[src])\n\t" "add %[src],%[src],%[rs]\n\t"
        "vle32.v v18,(%[src])\n\t" "add %[src],%[src],%[rs]\n\t"
        "vle32.v v20,(%[src])\n\t" "add %[src],%[src],%[rs]\n\t"
        "vle32.v v22,(%[src])\n\t" "add %[src],%[src],%[rs]\n\t"
        "vle32.v v24,(%[src])\n\t" "add %[src],%[src],%[rs]\n\t"
        "vle32.v v26,(%[src])\n\t" "add %[src],%[src],%[rs]\n\t"
        "vle32.v v28,(%[src])\n\t" "add %[src],%[src],%[rs]\n\t"
        "vle32.v v30,(%[src])\n\t" "add %[src],%[src],%[rs]\n\t"
        "vsetvli t0,zero,e16,m1,tu,mu\n\t"
        "vfncvt.f.f.w v0,v16\n\t" "vfncvt.f.f.w v1,v18\n\t"
        "vfncvt.f.f.w v2,v20\n\t" "vfncvt.f.f.w v3,v22\n\t"
        "vfncvt.f.f.w v4,v24\n\t" "vfncvt.f.f.w v5,v26\n\t"
        "vfncvt.f.f.w v6,v28\n\t" "vfncvt.f.f.w v7,v30\n\t"
        "vpack.vv v8,v0,v1,1\n\t" "vpack.vv v10,v2,v3,1\n\t"
        "vpack.vv v12,v4,v5,1\n\t" "vpack.vv v14,v6,v7,1\n\t"
        "vpack.vv v0,v8,v10,2\n\t" "vpack.vv v2,v9,v11,2\n\t"
        "vpack.vv v4,v12,v14,2\n\t" "vpack.vv v6,v13,v15,2\n\t"
        "vpack.vv v8,v0,v4,3\n\t" "vpack.vv v10,v1,v5,3\n\t"
        "vpack.vv v12,v2,v6,3\n\t" "vpack.vv v14,v3,v7,3\n\t"
        "vs1r.v v8,(%[dst])\n\t"  "addi %[dst],%[dst],512\n\t"
        "vs1r.v v9,(%[dst])\n\t"  "addi %[dst],%[dst],512\n\t"
        "vs1r.v v10,(%[dst])\n\t" "addi %[dst],%[dst],512\n\t"
        "vs1r.v v11,(%[dst])\n\t" "addi %[dst],%[dst],512\n\t"
        "vs1r.v v12,(%[dst])\n\t" "addi %[dst],%[dst],512\n\t"
        "vs1r.v v13,(%[dst])\n\t" "addi %[dst],%[dst],512\n\t"
        "vs1r.v v14,(%[dst])\n\t" "addi %[dst],%[dst],512\n\t"
        "vs1r.v v15,(%[dst])\n\t" "addi %[dst],%[dst],512\n\t"
        : [src] "+&r"(src), [dst] "+&r"(dst)
        : [rs] "r"(rs)
        : "memory", "t0", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14",
          "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29",
          "v30", "v31");
}

// Chunk-64 WY Gated Delta Rule completely inside TCM (162 KB scratch).
// Keeps state and v untouched until all FP16 conversions pass HalfRangeCheck.
bool MNNIme2GatedPrefill64(float* state, const float* q, const float* k, const float* v, float* projK, float* output,
                           const float* decay, const float* beta, size_t stride, void* scratch, size_t scratchBytes,
                           size_t validCount) {
    constexpr size_t C = 64, D = 128;
    constexpr size_t kNeedBytes =
        (2 * C * D + D * D) * sizeof(_Float16) + (2 * C * C + 2 * C * D + 16 * 32) * sizeof(float);
    if (!halfPrefillEnabled() || !scratch || scratchBytes < kNeedBytes || validCount == 0 || validCount > C ||
        __riscv_vlenb() != 128) {
        return false;
    }
    HalfRangeCheck range;
    auto* aQ = static_cast<_Float16*>(scratch);
    auto* aK = aQ + C * D;
    auto* bBuf = aK + C * D;
    auto* KK = reinterpret_cast<float*>(bBuf + D * D);
    auto* QK = KK + C * C;
    auto* U = QK + C * C;
    auto* projQ = U + C * D;
    auto* tile = projQ + C * D;

    // 1. Pack Q, K, and K^T, then compute KK = K @ K^T and QK = Q @ K^T via vfwmadot.
    for (size_t m = 0; m < C; m += 16) {
        for (size_t d = 0; d < D; d += 64) {
            for (size_t h = 0; h < 16; h += 8) {
                packDenseA(q + (m + h) * D + d, D * sizeof(float), aQ + m * D + d * 16 + h * 8);
                packDenseA(k + (m + h) * D + d, D * sizeof(float), aK + m * D + d * 16 + h * 8);
            }
        }
    }
    for (size_t n = 0; n < C; n += 32) {
        for (size_t d = 0; d < D; d += 64) {
            for (size_t h = 0; h < 32; h += 8) {
                packDenseBTransposed(k + (n + h) * D + d, D * sizeof(float), bBuf + n * D + d * 32 + h * 8);
            }
        }
    }
    for (size_t n = 0; n < C; n += 32) {
        for (size_t m = 0; m < C; m += 16) {
            matmulTile(D, bBuf + n * D, aK + m * D, KK + m * C + n, C * sizeof(float));
            matmulTile(D, bBuf + n * D, aQ + m * D, QK + m * C + n, C * sizeof(float));
        }
    }

    // 2. Pack state S -> bBuf and compute projQ (in TCM) and projK (in TCM).
    for (size_t c = 0; c < D; c += 64) {
        for (size_t d = 0; d < D; d += 8) {
            packDenseB(state + d * D + c, D * sizeof(float), bBuf + c * D + d * 32, bBuf + (c + 32) * D + d * 32);
        }
    }
    for (size_t c = 0; c < D; c += 32) {
        for (size_t m = 0; m < C; m += 16) {
            matmulTile(D, bBuf + c * D, aQ + m * D, projQ + m * D + c, D * sizeof(float));
            matmulTile(D, bBuf + c * D, aK + m * D, projK + m * D + c, D * sizeof(float));
        }
    }

    // 3. Build W = (I + L)^{-1} in-place in KK and lower-triangular M_QK in QK.
    float prefix[C];
    float carry = 1.0f;
    alignas(128) float lRow[C];
    for (size_t t = 0; t < C; ++t) {
        carry *= decay[t];
        prefix[t] = carry;
        float between = decay[t];
        const float bt = beta[t];
        float* kkRow = KK + t * C;
        float* qkRow = QK + t * C;
        for (size_t j = t + 1; j < C; ++j) {
            qkRow[j] = 0.0f;
        }
        lRow[t] = 0.0f;
        for (size_t j = t; j-- > 0;) {
            lRow[j] = bt * between * kkRow[j];
            qkRow[j] *= between;
            between *= decay[j];
        }
        auto wt = __riscv_vfmv_v_f_f32m2(0.0f, C);
        for (size_t m = 0; m < t; ++m) {
            auto wm = __riscv_vle32_v_f32m2(KK + m * C, m + 1);
            wt = __riscv_vfnmsac_vf_f32m2_tu(wt, lRow[m], wm, m + 1);
        }
        __riscv_vse32_v_f32m2(kkRow, wt, C);
        kkRow[t] = 1.0f;
    }

    // 4. rhs = beta * (v - prefix * projK) into projK (keeps v untouched for fallback).
    for (size_t t = 0; t < C; ++t) {
        auto vt = __riscv_vle32_v_f32m4(v + t * D, D);
        auto pk = __riscv_vle32_v_f32m4(projK + t * D, D);
        auto rhs = __riscv_vfmul_vf_f32m4(__riscv_vfnmsac_vf_f32m4(vt, prefix[t], pk, D), beta[t], D);
        __riscv_vse32_v_f32m4(projK + t * D, rhs, D);
    }

    // 5. U = W @ rhs (64x64x128).
    for (size_t m = 0; m < C; m += 16) {
        for (size_t h = 0; h < 16; h += 8) {
            packDenseA(KK + (m + h) * C, C * sizeof(float), aQ + m * C + h * 8);
        }
    }
    for (size_t c = 0; c < D; c += 64) {
        for (size_t t = 0; t < C; t += 8) {
            packDenseB(projK + t * D + c, D * sizeof(float), bBuf + c * C + t * 32, bBuf + (c + 32) * C + t * 32);
        }
    }
    for (size_t c = 0; c < D; c += 32) {
        for (size_t m = 0; m < C; m += 16) {
            matmulTile(C, bBuf + c * C, aQ + m * C, U + m * D + c, D * sizeof(float));
        }
    }

    // 6. output = prefix * projQ + M_QK @ U (64x64x128).
    for (size_t m = 0; m < C; m += 16) {
        for (size_t h = 0; h < 16; h += 8) {
            packDenseA(QK + (m + h) * C, C * sizeof(float), aQ + m * C + h * 8);
        }
    }
    for (size_t c = 0; c < D; c += 64) {
        for (size_t t = 0; t < C; t += 8) {
            packDenseB(U + t * D + c, D * sizeof(float), bBuf + c * C + t * 32, bBuf + (c + 32) * C + t * 32);
        }
    }
    for (size_t c = 0; c < D; c += 32) {
        for (size_t m = 0; m < C; m += 16) {
            if (m >= validCount) {
                break;
            }
            matmulTile(C, bBuf + c * C, aQ + m * C, tile, 32 * sizeof(float));
            const size_t rLimit = std::min<size_t>(16, validCount - m);
            for (size_t r = 0; r < rLimit; ++r) {
                float* dst = output + (m + r) * stride + c;
                auto pq = __riscv_vle32_v_f32m1(projQ + (m + r) * D + c, 32);
                auto intra = __riscv_vle32_v_f32m1(tile + r * 32, 32);
                __riscv_vse32_v_f32m1(dst, __riscv_vfmacc_vf_f32m1(intra, prefix[m + r], pq, 32), 32);
            }
        }
    }

    // 7. Prepare state update S = product * S + K^T @ (suffix * U); abort before mutating state on overflow.
    float product = 1.0f;
    for (size_t t = C; t-- > 0;) {
        auto uRow = __riscv_vle32_v_f32m4(U + t * D, D);
        __riscv_vse32_v_f32m4(U + t * D, __riscv_vfmul_vf_f32m4(uRow, product, D), D);
        product *= decay[t];
    }
    auto* sh = aK;
    for (size_t t = 0; t < C; t += 8) {
        for (size_t c = 0; c < D; c += 64) {
            packDenseB(U + t * D + c, D * sizeof(float), bBuf + c * C + t * 32, bBuf + (c + 32) * C + t * 32);
            packDenseB(k + t * D + c, D * sizeof(float), sh, sh + 256);
            for (size_t g = 0; g < 4; ++g) {
                std::memcpy(aQ + (c + g * 16) * C + t * 16, sh + g * 128, 128 * sizeof(_Float16));
            }
        }
    }
    if (range.failed()) {
        return false;
    }
    for (size_t m = 0; m < D; m += 16) {
        for (size_t n = 0; n < D; n += 32) {
            matmulTile(C, bBuf + n * C, aQ + m * C, tile, 32 * sizeof(float));
            for (size_t r = 0; r < 16; ++r) {
                float* dst = state + (m + r) * D + n;
                auto delta = __riscv_vle32_v_f32m1(tile + r * 32, 32);
                auto s = __riscv_vle32_v_f32m1(dst, 32);
                __riscv_vse32_v_f32m1(dst, __riscv_vfmacc_vf_f32m1(delta, product, s, 32), 32);
            }
        }
    }
    return true;
}
#else
bool MNNIme2GatedPrefill(float*, const float*, const float*, float*, float*, float*, const float*, const float*,
                         size_t, void*, size_t) {
    return false;
}
bool MNNIme2GatedPrefill64(float*, const float*, const float*, const float*, float*, float*, const float*,
                           const float*, size_t, void*, size_t, size_t) {
    return false;
}
#endif
