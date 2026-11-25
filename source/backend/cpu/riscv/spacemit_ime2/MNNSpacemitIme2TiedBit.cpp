//
//  MNNSpacemitIme2TiedBit.cpp
//  MNN
//
//  Created by MNN on 2026/11/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNNSpacemitIme2Half.hpp"

#include <algorithm>
#include <atomic>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#if defined(__riscv_vector)
#include <riscv_vector.h>
#endif

#include "backend/cpu/compute/CommonOptFunction.h"

using MNNSpacemitIme2TcmTask = void (*)(size_t index, void* tcmBuffer, size_t tcmSize, void* context);
extern "C" size_t MNNSpacemitIme2RunDecodeTasks(size_t taskCount, MNNSpacemitIme2TcmTask task, void* context);

#if defined(MNN_USE_SPACEMIT_IME2) && defined(__riscv_zvfh) && defined(__riscv_vector)
namespace {

struct TiedBitCandidate {
    float score;
    uint32_t index;
};

static inline bool candidateGreater(const TiedBitCandidate& a, const TiedBitCandidate& b) {
    return a.score > b.score || (a.score == b.score && a.index < b.index);
}

// Monotonic bit-ordered uint32 key so workers can publish/read a lock-free max threshold.
static inline uint32_t floatToOrderedKey(float v) {
    uint32_t u = 0;
    std::memcpy(&u, &v, sizeof(u));
    return (u & 0x80000000u) ? ~u : (u ^ 0x80000000u);
}

static inline float orderedKeyToFloat(uint32_t k) {
    const uint32_t u = (k & 0x80000000u) ? (k ^ 0x80000000u) : ~k;
    float v = 0.0f;
    std::memcpy(&v, &u, sizeof(v));
    return v;
}

static inline void publishMaxThreshold(std::atomic<uint32_t>& sharedKey, float candidate) {
    const uint32_t key = floatToOrderedKey(candidate);
    uint32_t cur = sharedKey.load(std::memory_order_relaxed);
    while (key > cur && !sharedKey.compare_exchange_weak(cur, key, std::memory_order_relaxed)) {
    }
}

// Vectorized RVV (vlenb=128) activation packer: per-K64 signed 4-bit [-7, 7] + vnpack4.vv.
static bool prepareTiedBitA4(const float* src, size_t blockNum, size_t blocksPerSuper, uint8_t* aPre,
                             _Float16* aRelScale, float* superScale, _Float16* xSum) {
    const size_t superBlocks = blockNum / blocksPerSuper;
    const vfloat32m1_t vZero = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    float blkDa[16];
    for (size_t sb = 0; sb < superBlocks; ++sb) {
        float maxDa = 0.0f;
        for (size_t sub = 0; sub < blocksPerSuper; ++sub) {
            const size_t bk = sb * blocksPerSuper + sub;
            const vfloat32m2_t xv = __riscv_vle32_v_f32m2(src + bk * 64, 64);
            const vfloat32m2_t ax = __riscv_vfabs_v_f32m2(xv, 64);
            const float sum = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m2_f32m1(xv, vZero, 64));
            const float maxAbs = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m2_f32m1(ax, vZero, 64));
            if (!std::isfinite(sum) || !std::isfinite(maxAbs) || std::fabs(sum) > 32768.0f) {
                return false;
            }
            xSum[bk] = static_cast<_Float16>(sum);
            const float da = maxAbs > 1e-12f ? (maxAbs / 7.0f) : 0.0f;
            blkDa[sub] = da;
            if (da > maxDa) {
                maxDa = da;
            }
        }
        // Fold the Q1 span factor (8.0) into the FP32 superblock scale to preserve full FP16 scale range.
        superScale[sb] = 8.0f * maxDa;
        const float invMaxDa = maxDa > 0.0f ? (1.0f / maxDa) : 0.0f;
        for (size_t sub = 0; sub < blocksPerSuper; ++sub) {
            const size_t bk = sb * blocksPerSuper + sub;
            const float da = blkDa[sub];
            aRelScale[bk] = static_cast<_Float16>(da * invMaxDa);
            const float invDa = da > 0.0f ? (1.0f / da) : 0.0f;
            const vfloat32m2_t xv = __riscv_vle32_v_f32m2(src + bk * 64, 64);
            vint32m2_t qi = __riscv_vfcvt_x_f_v_i32m2(__riscv_vfmul_vf_f32m2(xv, invDa, 64), 64);
            qi = __riscv_vmax_vx_i32m2(__riscv_vmin_vx_i32m2(qi, 7, 64), -7, 64);
            const vuint8mf2_t q8 = __riscv_vand_vx_u8mf2(
                __riscv_vncvt_x_x_w_u8mf2(__riscv_vncvt_x_x_w_u16m1(__riscv_vreinterpret_v_i32m2_u32m2(qi), 64), 64),
                0x0f, 64);
            alignas(64) uint8_t q64[64];
            __riscv_vse8_v_u8mf2(q64, q8, 64);
            uint8_t* dst0 = aPre + (bk * 2 + 0) * 128;
            uint8_t* dst1 = aPre + (bk * 2 + 1) * 128;
            asm volatile(
                "vsetvli    t0, x0, e8, mf4     \n\t"
                "vle8.v     v3, 0(%[SRC])       \n\t"
                "addi       t1, %[SRC], 32      \n\t"
                "vle8.v     v4, (t1)            \n\t"
                "vsetvli    t0, x0, e16, m1     \n\t"
                "vnpack4.vv v2, v3, v3, 3       \n\t"
                "vnpack4.vv v5, v4, v4, 3       \n\t"
                "vs1r.v     v2, (%[D0])         \n\t"
                "vs1r.v     v5, (%[D1])         \n\t"
                :
                : [SRC] "r"(q64), [D0] "r"(dst0), [D1] "r"(dst1)
                : "t0", "t1", "v2", "v3", "v4", "v5", "memory");
        }
    }
    return true;
}

// Evaluate one N32 Q1 proxy tile in A100 IME2 (vlenb=128) and compress any lanes >= thresh
// directly into dstScores / dstIdx using RVV vcompress (zero scalar scan on hits).
static inline size_t evalQ1Group32AndCompress(const uint8_t* aPre, const uint8_t* bTile, const _Float16* aRelScale,
                                              const float* superScale, const _Float16* xSum, const float* bias32,
                                              size_t superBlocks, size_t blocksPerSuper, float thresh,
                                              uint32_t ocBase, float* dstScores, uint32_t* dstIdx) {
    const uint8_t* aPtr = aPre;
    const uint8_t* bPtr = bTile;
    const _Float16* daPtr = aRelScale;
    const float* ssPtr = superScale;
    const _Float16* xsPtr = xSum;
    size_t hits = 0;

    asm volatile(
        "li             t2, 0x11                     \n\t"
        "vsetvli        t0, x0, e32, m1              \n\t"
        "vxor.vv        v31, v31, v31                \n\t"
        "mv             t4, %[NSUPER]                \n\t"

        ".align 4                                    \n\t"
        "1:                                          \n\t"
        "flw            ft4, (%[SS])                 \n\t"
        "addi           %[SS], %[SS], 4              \n\t"
        "mv             t5, %[NIN]                   \n\t"
        "vsetvli        t0, x0, e16, m1              \n\t"
        "vxor.vv        v16, v16, v16                \n\t"
        "vxor.vv        v17, v17, v17                \n\t"
        "vxor.vv        v18, v18, v18                \n\t"
        "vxor.vv        v19, v19, v19                \n\t"

        ".align 4                                    \n\t"
        "2:                                          \n\t"
        "addi           s5, %[B], 128                \n\t"
        "vl2r.v         v28, (s5)                    \n\t"
        "vl2r.v         v2, (%[A])                   \n\t"
        "flh            ft0, (%[DA])                 \n\t"
        "flh            fa2, (%[XS])                 \n\t"
        "addi           %[DA], %[DA], 2              \n\t"
        "addi           %[XS], %[XS], 2              \n\t"
        "addi           %[A], %[A], 256              \n\t"

        "vsetvli        t0, x0, e16, mf2             \n\t"
        "vle16.v        v8, (%[B])                   \n\t"
        "addi           s5, %[B], 64                 \n\t"
        "vle16.v        v10, (s5)                    \n\t"
        "addi           %[B], %[B], 384              \n\t"
        "vfmul.vf       v0, v8, ft0                  \n\t"
        "vfwmacc.vf     v31, fa2, v10                \n\t"

        "vsetvli        t0, x0, e8, m1               \n\t"
        "vand.vx        v24, v28, t2                 \n\t"
        "vsrl.vi        v25, v28, 1                  \n\t"
        "vsrl.vi        v26, v28, 2                  \n\t"
        "vsrl.vi        v27, v28, 3                  \n\t"
        "vand.vx        v25, v25, t2                 \n\t"
        "vand.vx        v26, v26, t2                 \n\t"
        "vand.vx        v27, v27, t2                 \n\t"

        "vsetvli        t0, x0, e16, m1              \n\t"
        "vmadot.hp      v16, v2, v24, v0, 0, i4      \n\t"
        "vmadot.hp      v17, v2, v25, v0, 1, i4      \n\t"
        "vmadot.hp      v18, v2, v26, v0, 2, i4      \n\t"
        "vmadot.hp      v19, v2, v27, v0, 3, i4      \n\t"

        "vsetvli        t0, x0, e8, m1               \n\t"
        "vand.vx        v4,  v29, t2                 \n\t"
        "vsrl.vi        v5,  v29, 1                  \n\t"
        "vsrl.vi        v6,  v29, 2                  \n\t"
        "vsrl.vi        v7,  v29, 3                  \n\t"
        "vand.vx        v5,  v5,  t2                 \n\t"
        "vand.vx        v6,  v6,  t2                 \n\t"
        "vand.vx        v7,  v7,  t2                 \n\t"

        "vsetvli        t0, x0, e16, m1              \n\t"
        "vmadot.hp      v16, v3, v4,  v0, 0, i4      \n\t"
        "vmadot.hp      v17, v3, v5,  v0, 1, i4      \n\t"
        "vmadot.hp      v18, v3, v6,  v0, 2, i4      \n\t"
        "vmadot.hp      v19, v3, v7,  v0, 3, i4      \n\t"

        "addi           t5, t5, -1                   \n\t"
        "bgtz           t5, 2b                       \n\t"

        "vsetvli        t0, x0, e16, m1              \n\t"
        "vpack.vv       v8, v16, v17, 1              \n\t"
        "vpack.vv       v12, v18, v19, 1             \n\t"
        "vpack.vv       v20, v8, v12, 2              \n\t"
        "vsetvli        t0, x0, e16, mf2             \n\t"
        "vfwcvt.f.f.v   v22, v20                     \n\t"
        "vsetvli        t0, x0, e32, m1              \n\t"
        "vfmacc.vf      v31, ft4, v22                \n\t"

        "addi           t4, t4, -1                   \n\t"
        "bgtz           t4, 1b                       \n\t"

        "vsetvli        t0, x0, e32, m1              \n\t"
        "beqz           %[BIAS], 3f                  \n\t"
        "vle32.v        v8, (%[BIAS])                \n\t"
        "vfadd.vv       v31, v31, v8                 \n\t"
        "3:                                          \n\t"
        "vmfge.vf       v0, v31, %[TH]               \n\t"
        "vcpop.m        %[HIT], v0                   \n\t"
        "beqz           %[HIT], 4f                   \n\t"
        "vid.v          v8                           \n\t"
        "vadd.vx        v8, v8, %[OC]                \n\t"
        "vcompress.vm   v10, v31, v0                 \n\t"
        "vcompress.vm   v12, v8, v0                  \n\t"
        "vsetvli        t0, %[HIT], e32, m1          \n\t"
        "vse32.v        v10, (%[DOUT])               \n\t"
        "vse32.v        v12, (%[IOUT])               \n\t"
        "4:                                          \n\t"
        : [A] "+r"(aPtr), [B] "+r"(bPtr), [DA] "+r"(daPtr), [SS] "+r"(ssPtr), [XS] "+r"(xsPtr), [HIT] "+r"(hits)
        : [NSUPER] "r"(superBlocks), [NIN] "r"(blocksPerSuper), [BIAS] "r"(bias32), [TH] "f"(thresh),
          [OC] "r"(ocBase), [DOUT] "r"(dstScores), [IOUT] "r"(dstIdx)
        : "cc", "memory", "t0", "t2", "t4", "t5", "s5",
          "v0", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v10", "v12",
          "v16", "v17", "v18", "v19", "v20", "v22", "v24", "v25", "v26", "v27",
          "v28", "v29", "v31", "fa2", "ft0", "ft4");
    return hits;
}

static inline void pruneInclusiveTopM(std::vector<float>& scores, std::vector<uint32_t>& indices, size_t& count,
                                      size_t topM, float& threshold, size_t& pruneLimit,
                                      std::vector<TiedBitCandidate>& tmp) {
    if (count <= topM) {
        return;
    }
    tmp.resize(count);
    for (size_t i = 0; i < count; ++i) {
        tmp[i] = {scores[i], indices[i]};
    }
    std::nth_element(tmp.begin(), tmp.begin() + (topM - 1), tmp.end(), candidateGreater);
    const float newThresh = tmp[topM - 1].score;
    auto it = std::partition(tmp.begin() + topM, tmp.end(),
                             [newThresh](const TiedBitCandidate& c) { return c.score >= newThresh; });
    const size_t kept = static_cast<size_t>(it - tmp.begin());
    for (size_t i = 0; i < kept; ++i) {
        scores[i] = tmp[i].score;
        indices[i] = tmp[i].index;
    }
    count = kept;
    threshold = newThresh;
    if (kept * 2 > pruneLimit) {
        pruneLimit = kept * 2;
        if (scores.size() < pruneLimit + 32) {
            scores.resize(pruneLimit + 32);
            indices.resize(pruneLimit + 32);
        }
    }
}

// Exact FP32-input x W4 dot product for one vocabulary row directly from MNN mWeightInt8.
static inline float refineOneCandidateW4(const float* src, const uint8_t* w4Base, size_t blockNum, uint32_t oc,
                                         const float* bias, const uint8_t* nextRowBase) {
    constexpr size_t srcStride = 4 * 32 + 2 * 4 * sizeof(float); // 160 B per block64
    const size_t dz = oc >> 2;
    const size_t lane = oc & 3u;
    const size_t laneOff = (lane & 1u) * 16;
    const bool upper = (lane < 2);
    const uint8_t* rowBase = w4Base + dz * blockNum * srcStride;
    if (nextRowBase != nullptr) {
        __builtin_prefetch(nextRowBase, 0, 0);
        __builtin_prefetch(nextRowBase + 160, 0, 0);
        __builtin_prefetch(nextRowBase + 320, 0, 0);
        __builtin_prefetch(nextRowBase + 480, 0, 0);
    }
    vfloat32m2_t acc = __riscv_vfmv_v_f_f32m2(0.0f, 64);
    for (size_t bk = 0; bk < blockNum; ++bk) {
        const uint8_t* bp = rowBase + bk * srcStride;
        if (bk + 2 < blockNum) {
            __builtin_prefetch(bp + 2 * srcStride, 0, 0);
            __builtin_prefetch(bp + 2 * srcStride + 64, 0, 0);
            __builtin_prefetch(bp + 2 * srcStride + 128, 0, 0);
        }
        const float* params = reinterpret_cast<const float*>(bp + 128);
        const float scale = params[lane];
        const float offset = params[4 + lane];
        alignas(64) uint8_t raw64[64];
        std::memcpy(raw64 + 0, bp + 0 + laneOff, 16);
        std::memcpy(raw64 + 16, bp + 32 + laneOff, 16);
        std::memcpy(raw64 + 32, bp + 64 + laneOff, 16);
        std::memcpy(raw64 + 48, bp + 96 + laneOff, 16);
        const vuint8mf2_t r = __riscv_vle8_v_u8mf2(raw64, 64);
        const vuint8mf2_t q = upper ? __riscv_vsrl_vx_u8mf2(r, 4, 64) : __riscv_vand_vx_u8mf2(r, 15, 64);
        const vfloat32m2_t qf = __riscv_vfcvt_f_xu_v_f32m2(__riscv_vzext_vf4_u32m2(q, 64), 64);
        const vfloat32m2_t xv = __riscv_vle32_v_f32m2(src + bk * 64, 64);
        const vfloat32m2_t wv = __riscv_vfadd_vf_f32m2(__riscv_vfmul_vf_f32m2(qf, scale, 64), offset, 64);
        acc = __riscv_vfmacc_vv_f32m2(acc, wv, xv, 64);
    }
    const vfloat32m1_t red = __riscv_vfredusum_vs_f32m2_f32m1(acc, __riscv_vfmv_v_f_f32m1(0.0f, 1), 64);
    float val = __riscv_vfmv_f_s_f32m1_f32(red);
    if (bias != nullptr) {
        val += bias[oc];
    }
    return val;
}

} // namespace
#endif // defined(MNN_USE_SPACEMIT_IME2) && defined(__riscv_zvfh) && defined(__riscv_vector)

extern "C" int MNNSpacemitIme2LinearTiedBitDecode(int8_t* dst, const float* src, const int8_t* weight,
                                                  size_t srcDepthQuad, size_t dstStep, size_t dstDepthQuad,
                                                  const QuanPostTreatParameters* post, void* context) {
#if defined(MNN_USE_SPACEMIT_IME2) && defined(__riscv_zvfh) && defined(__riscv_vector)
    if (context == nullptr || dst == nullptr || src == nullptr || weight == nullptr || post == nullptr ||
        post->useInt8 != 0 || post->blockNum <= 0 || srcDepthQuad != 4 || post->blockNum % 4 != 0 ||
        dstDepthQuad < 8 || dstDepthQuad % 8 != 0 || dstStep != 4 * sizeof(float) || post->inputScale != nullptr ||
        post->inputBias != nullptr) {
        return 0;
    }
    if (post->fp32minmax != nullptr && !(post->fp32minmax[0] < -3.0e38f && post->fp32minmax[1] > 3.0e38f)) {
        return 0;
    }
    const size_t blockNum = static_cast<size_t>(post->blockNum);
    const size_t depth = blockNum * 64;
    const size_t countN = dstDepthQuad * 4;
    const size_t topM = MNNIme2::tiedBitTopM();
    if (topM == 0 || topM >= countN) {
        return 0;
    }
    const uint8_t* q1Weight = MNNIme2GetBlock64Q1Weight(context, weight, depth, countN);
    if (q1Weight == nullptr) {
        return 0;
    }

    const size_t groupCount = countN / 32;
    const size_t workers = std::min<size_t>(4, groupCount);
    const size_t blocksPerSuper = (blockNum % 16 == 0) ? 16 : ((blockNum % 8 == 0) ? 8 : 4);
    const size_t superBlocks = blockNum / blocksPerSuper;
    const size_t tileStride = blockNum * sizeof(MNNIme2::Block64Q1Weight);
    const float* bias = post->biasFloat;

    struct Stage1Ctx {
        const float* src;
        const uint8_t* q1Weight;
        const float* bias;
        size_t blockNum;
        size_t blocksPerSuper;
        size_t superBlocks;
        size_t groupCount;
        size_t tileStride;
        size_t workers;
        size_t topM;
        std::atomic<uint32_t> sharedThreshKey;
        const float* wScores[4];
        const uint32_t* wIndices[4];
        size_t wCount[4];
        std::atomic<bool> ok{true};
    };
    Stage1Ctx s1;
    s1.src = src;
    s1.q1Weight = q1Weight;
    s1.bias = bias;
    s1.blockNum = blockNum;
    s1.blocksPerSuper = blocksPerSuper;
    s1.superBlocks = superBlocks;
    s1.groupCount = groupCount;
    s1.tileStride = tileStride;
    s1.workers = workers;
    s1.topM = topM;
    s1.sharedThreshKey.store(floatToOrderedKey(-FLT_MAX), std::memory_order_relaxed);
    for (size_t i = 0; i < 4; ++i) {
        s1.wScores[i] = nullptr;
        s1.wIndices[i] = nullptr;
        s1.wCount[i] = 0;
    }

    auto stage1Task = [](size_t id, void* tcmBuffer, size_t tcmSize, void* user) {
        auto& p = *static_cast<Stage1Ctx*>(user);
        if (__riscv_vlenb() != 128) {
            p.ok.store(false, std::memory_order_relaxed);
            return;
        }
        const size_t aPreBytes = p.blockNum * 2 * 128;
        const size_t aRelBytes = ((p.blockNum * sizeof(_Float16)) + 63) & ~size_t(63);
        const size_t ssBytes = ((p.superBlocks * sizeof(float)) + 63) & ~size_t(63);
        const size_t xsBytes = ((p.blockNum * sizeof(_Float16)) + 63) & ~size_t(63);
        const size_t totalScratch = aPreBytes + aRelBytes + ssBytes + xsBytes;
        thread_local std::vector<uint8_t> fallbackScratch;
        uint8_t* scratch = nullptr;
        if (tcmBuffer != nullptr && tcmSize >= totalScratch) {
            scratch = static_cast<uint8_t*>(tcmBuffer);
        } else {
            if (fallbackScratch.size() < totalScratch + 128) {
                fallbackScratch.resize(totalScratch + 128);
            }
            scratch = reinterpret_cast<uint8_t*>(
                (reinterpret_cast<uintptr_t>(fallbackScratch.data()) + 127) & ~uintptr_t(127));
        }
        uint8_t* aPre = scratch;
        _Float16* aRelScale = reinterpret_cast<_Float16*>(aPre + aPreBytes);
        float* superScale = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(aRelScale) + aRelBytes);
        _Float16* xSum = reinterpret_cast<_Float16*>(reinterpret_cast<uint8_t*>(superScale) + ssBytes);

        if (!prepareTiedBitA4(p.src, p.blockNum, p.blocksPerSuper, aPre, aRelScale, superScale, xSum)) {
            p.ok.store(false, std::memory_order_relaxed);
            return;
        }

        const size_t base = p.groupCount / p.workers;
        const size_t rem = p.groupCount % p.workers;
        const size_t gBegin = id * base + std::min(id, rem);
        const size_t gEnd = gBegin + base + (id < rem ? 1 : 0);

        // Reuse thread-local candidate buffers across tokens to avoid per-token allocations.
        thread_local std::vector<float> scoresBuf;
        thread_local std::vector<uint32_t> indicesBuf;
        thread_local std::vector<TiedBitCandidate> pruneTmp;
        size_t pruneLimit = std::max<size_t>(p.topM * 2, 1024);
        if (scoresBuf.size() < pruneLimit + 32) {
            scoresBuf.resize(pruneLimit + 32);
            indicesBuf.resize(pruneLimit + 32);
        }
        size_t count = 0;
        float thresh = -FLT_MAX;

        for (size_t g = gBegin; g < gEnd; ++g) {
            if ((g & 15u) == 0u) {
                const float shared = orderedKeyToFloat(p.sharedThreshKey.load(std::memory_order_relaxed));
                if (shared > thresh) {
                    thresh = shared;
                }
            }
            const uint8_t* bTile = p.q1Weight + g * p.tileStride;
            const float* bias32 = p.bias ? (p.bias + g * 32) : nullptr;
            const size_t hits = evalQ1Group32AndCompress(aPre, bTile, aRelScale, superScale, xSum, bias32,
                                                         p.superBlocks, p.blocksPerSuper, thresh,
                                                         static_cast<uint32_t>(g * 32), scoresBuf.data() + count,
                                                         indicesBuf.data() + count);
            if (hits > 0) {
                count += hits;
                if (count >= pruneLimit) {
                    pruneInclusiveTopM(scoresBuf, indicesBuf, count, p.topM, thresh, pruneLimit, pruneTmp);
                    publishMaxThreshold(p.sharedThreshKey, thresh);
                }
            }
        }
        if (count > p.topM) {
            pruneInclusiveTopM(scoresBuf, indicesBuf, count, p.topM, thresh, pruneLimit, pruneTmp);
        }
        // Safe to expose thread-local data pointers: workers sleep until next RunDecodeTasks.
        p.wScores[id] = scoresBuf.data();
        p.wIndices[id] = indicesBuf.data();
        p.wCount[id] = count;
    };

    if (MNNSpacemitIme2RunDecodeTasks(workers, stage1Task, &s1) != workers ||
        !s1.ok.load(std::memory_order_relaxed)) {
        return 0;
    }

    // Merge per-worker inclusive Top-M lists into global inclusive Top-M candidate indices.
    thread_local std::vector<TiedBitCandidate> merged;
    merged.clear();
    size_t mergedCap = 0;
    for (size_t w = 0; w < workers; ++w) {
        mergedCap += s1.wCount[w];
    }
    if (merged.capacity() < mergedCap) {
        merged.reserve(mergedCap);
    }
    const float finalShared = orderedKeyToFloat(s1.sharedThreshKey.load(std::memory_order_relaxed));
    for (size_t w = 0; w < workers; ++w) {
        const float* ws = s1.wScores[w];
        const uint32_t* wi = s1.wIndices[w];
        const size_t wc = s1.wCount[w];
        for (size_t i = 0; i < wc; ++i) {
            if (ws[i] >= finalShared) {
                merged.push_back({ws[i], wi[i]});
            }
        }
    }
    if (merged.empty()) {
        return 0;
    }
    if (merged.size() > topM) {
        std::nth_element(merged.begin(), merged.begin() + (topM - 1), merged.end(), candidateGreater);
        const float globalThresh = merged[topM - 1].score;
        auto it = std::partition(merged.begin() + topM, merged.end(),
                                 [globalThresh](const TiedBitCandidate& c) { return c.score >= globalThresh; });
        merged.erase(it, merged.end());
    }
    thread_local std::vector<uint32_t> candIndices;
    thread_local std::vector<float> refinedVals;
    const size_t candCount = merged.size();
    candIndices.resize(candCount);
    refinedVals.resize(candCount);
    for (size_t i = 0; i < candCount; ++i) {
        candIndices[i] = merged[i].index;
    }
    std::sort(candIndices.begin(), candIndices.end());

    // Stage 2: each worker fills its exclusive slice of dst with -FLT_MAX and refines an equal share
    // of candidates into refinedVals[], eliminating vocabulary-skew load imbalance with zero races.
    struct Stage2Ctx {
        float* dst;
        const float* src;
        const uint8_t* w4Base;
        const float* bias;
        const uint32_t* cands;
        float* refinedVals;
        size_t candCount;
        size_t blockNum;
        size_t countN;
        size_t workers;
    };
    Stage2Ctx s2 = {reinterpret_cast<float*>(dst),
                    src,
                    reinterpret_cast<const uint8_t*>(weight),
                    bias,
                    candIndices.data(),
                    refinedVals.data(),
                    candCount,
                    blockNum,
                    countN,
                    workers};

    auto stage2Task = [](size_t id, void*, size_t, void* user) {
        auto& p = *static_cast<Stage2Ctx*>(user);
        const size_t groups = p.countN / 32;
        const size_t base = groups / p.workers;
        const size_t rem = groups % p.workers;
        const size_t ocBegin = (id * base + std::min(id, rem)) * 32;
        const size_t ocEnd = ocBegin + (base + (id < rem ? 1 : 0)) * 32;

        const vfloat32m4_t negMax = __riscv_vfmv_v_f_f32m4(-FLT_MAX, 128);
        size_t pos = ocBegin;
        for (; pos + 128 <= ocEnd; pos += 128) {
            __riscv_vse32_v_f32m4(p.dst + pos, negMax, 128);
        }
        for (; pos < ocEnd; ++pos) {
            p.dst[pos] = -FLT_MAX;
        }

        const size_t cBase = p.candCount / p.workers;
        const size_t cRem = p.candCount % p.workers;
        const size_t cBegin = id * cBase + std::min(id, cRem);
        const size_t cEnd = cBegin + cBase + (id < cRem ? 1 : 0);
        constexpr size_t srcStride = 4 * 32 + 2 * 4 * sizeof(float);
        for (size_t i = cBegin; i < cEnd; ++i) {
            const uint8_t* nextRow =
                (i + 1 < cEnd) ? (p.w4Base + (p.cands[i + 1] >> 2) * p.blockNum * srcStride) : nullptr;
            p.refinedVals[i] = refineOneCandidateW4(p.src, p.w4Base, p.blockNum, p.cands[i], p.bias, nextRow);
        }
    };

    if (MNNSpacemitIme2RunDecodeTasks(workers, stage2Task, &s2) != workers) {
        return 0;
    }
    float* outFp32 = reinterpret_cast<float*>(dst);
    for (size_t i = 0; i < candCount; ++i) {
        outFp32[candIndices[i]] = refinedVals[i];
    }
    return 1;
#else
    (void)dst;
    (void)src;
    (void)weight;
    (void)srcDepthQuad;
    (void)dstStep;
    (void)dstDepthQuad;
    (void)post;
    (void)context;
    return 0;
#endif
}
