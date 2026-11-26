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
#include <cstdlib>
#include <cstring>
#include <limits>

extern "C" size_t MNNSpacemitIme2RunTcmTasks(size_t, void (*)(size_t, void*, size_t, void*), void*);
using namespace MNNIme2;

namespace {
// Packed activation superblock covering M=32 rows and up to K=512 (8 x block64).
// Quantized dynamically per-row-per-K512 to INT8 and split into 4-bit hi/lo nibbles.
struct alignas(128) PackedASuper512 {
    float aScale[32];
    _Float16 aSum[4][64];     // [rg 0..3][(row % 8) * 8 + bk] in vfwmadot A order
    uint8_t q[8][2][4][256];  // [bk][k32][rg][0..127 = a_hi, 128..255 = a_lo]
};

static_assert(sizeof(PackedASuper512) == 17024, "unexpected PackedASuper512 layout");

constexpr size_t kMaxBlocksPerSuper = 8;
constexpr size_t kBScaleBytes = kMaxBlocksPerSuper * 64 * sizeof(_Float16); // 1024B expanded [scale, 16*scale]
constexpr size_t kBScratchBytes = kMaxBlocksPerSuper * sizeof(Block64Weight) + kBScaleBytes + 512; // 9216 + 1024 + 512

static thread_local MNNIme2ActivationCache* gActivationCache = nullptr;

// Lazily allocated per-thread slab reused across FusedLinear calls to avoid
// repeated mmap/munmap page-fault overhead without increasing peak memory.
static void* getReusableActivationSlab(size_t bytes) {
    struct Slab {
        void* ptr = nullptr;
        size_t cap = 0;
        ~Slab() { std::free(ptr); }
    };
    static thread_local Slab slab;
    if (slab.cap < bytes) {
        void* next = nullptr;
        if (posix_memalign(&next, 128, bytes) != 0 || !next) {
            return nullptr;
        }
        std::free(slab.ptr);
        slab.ptr = next;
        slab.cap = bytes;
    }
    return slab.ptr;
}

static void* getReusableHandoffSlab(size_t bytes) {
    struct Slab {
        void* ptr = nullptr;
        size_t cap = 0;
        ~Slab() { std::free(ptr); }
    };
    static thread_local Slab slab;
    if (slab.cap < bytes) {
        void* next = nullptr;
        if (posix_memalign(&next, 128, bytes) != 0 || !next) {
            return nullptr;
        }
        std::free(slab.ptr);
        slab.ptr = next;
        slab.cap = bytes;
    }
    return slab.ptr;
}

static constexpr uint32_t kRowScaleIdx[32] = {
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
    2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
};
static constexpr _Float16 kZero32[32] = {};

static inline void rvvMemcpy(void* dst, const void* src, size_t bytes) {
    auto* d = static_cast<uint8_t*>(dst);
    auto* s = static_cast<const uint8_t*>(src);
    while (bytes >= 1024) {
        asm volatile("vl8r.v v0, (%[s])\n\tvs8r.v v0, (%[d])\n\t"
                     :
                     : [s] "r"(s), [d] "r"(d)
                     : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
        s += 1024;
        d += 1024;
        bytes -= 1024;
    }
    while (bytes >= 128) {
        asm volatile("vl1r.v v0, (%[s])\n\tvs1r.v v0, (%[d])\n\t" : : [s] "r"(s), [d] "r"(d) : "memory", "v0");
        s += 128;
        d += 128;
        bytes -= 128;
    }
    if (bytes) {
        std::memcpy(d, s, bytes);
    }
}

// Quantize, sum per block64, transpose, and nibble-pack an M=32 x K<=512 slice
// already staged in TCM or L2 ([k4][rg][8 rows x 4 chans]) with precomputed per-row absmax.
template <typename T>
static void quantizeStagedToPackedA(PackedASuper512* dst, const T* stagedBuf, const float* amaxBuf, size_t rows,
                                    size_t r0, size_t blocksInSb) {
    const size_t validM = std::min<size_t>(32, rows - r0);
    alignas(128) float invBuf[128];
    for (size_t r = 0; r < 32; ++r) {
        const float am = r < validM ? amaxBuf[r * 4] : 0.0f;
        dst->aScale[r] = am / 127.0f;
        const float inv = am > 0.0f ? (127.0f / am) : 0.0f;
        invBuf[r * 4 + 0] = inv;
        invBuf[r * 4 + 1] = inv;
        invBuf[r * 4 + 2] = inv;
        invBuf[r * 4 + 3] = inv;
    }

    if (blocksInSb < kMaxBlocksPerSuper) {
        std::memset(dst->aSum, 0, sizeof(dst->aSum));
    }

    auto load32 = [](const float* p) -> vfloat32m1_t { return __riscv_vle32_v_f32m1(p, 32); };
    auto load32Half = [](const _Float16* p) -> vfloat32m1_t {
        return __riscv_vfwcvt_f_f_v_f32m1(__riscv_vle16_v_f16mf2(p, 32), 32);
    };
    (void)load32;
    (void)load32Half;
    for (size_t rg = 0; rg < 4; ++rg) {
        if (r0 + rg * 8 >= rows) {
            std::memset(dst->aSum[rg], 0, 64 * sizeof(_Float16));
            for (size_t bk = 0; bk < blocksInSb; ++bk) {
                std::memset(dst->q[bk][0][rg], 0, 256);
                std::memset(dst->q[bk][1][rg], 0, 256);
            }
            continue;
        }
        const size_t validR = std::min<size_t>(8, rows - (r0 + rg * 8));
        const auto vInv = __riscv_vle32_v_f32m1(invBuf + rg * 32, 32);
        const auto vInvM4 = __riscv_vcreate_v_f32m1_f32m4(vInv, vInv, vInv, vInv);
        const T* tcmRg = stagedBuf + rg * 32;
        for (size_t bk = 0; bk < blocksInSb; ++bk) {
            auto sumI32 = __riscv_vmv_v_x_i32m1(0, 32);
            for (size_t k32 = 0; k32 < 2; ++k32) {
                const size_t k4Base = bk * 16 + k32 * 8;
                auto quant16 = [&](size_t k4) -> vuint32m1_t {
                    vfloat32m1_t x0, x1, x2, x3;
                    if (sizeof(T) == 2) {
                        const auto* h = reinterpret_cast<const _Float16*>(tcmRg);
                        x0 = load32Half(h + (k4 + 0) * 128);
                        x1 = load32Half(h + (k4 + 1) * 128);
                        x2 = load32Half(h + (k4 + 2) * 128);
                        x3 = load32Half(h + (k4 + 3) * 128);
                    } else {
                        const auto* f = reinterpret_cast<const float*>(tcmRg);
                        x0 = load32(f + (k4 + 0) * 128);
                        x1 = load32(f + (k4 + 1) * 128);
                        x2 = load32(f + (k4 + 2) * 128);
                        x3 = load32(f + (k4 + 3) * 128);
                    }
                    auto xm4 = __riscv_vcreate_v_f32m1_f32m4(x0, x1, x2, x3);
                    auto qi32 = __riscv_vfcvt_x_f_v_i32m4(__riscv_vfmul_vv_f32m4(xm4, vInvM4, 128), 128);
                    sumI32 = __riscv_vadd_vv_i32m1(sumI32, __riscv_vget_v_i32m4_i32m1(qi32, 0), 32);
                    sumI32 = __riscv_vadd_vv_i32m1(sumI32, __riscv_vget_v_i32m4_i32m1(qi32, 1), 32);
                    sumI32 = __riscv_vadd_vv_i32m1(sumI32, __riscv_vget_v_i32m4_i32m1(qi32, 2), 32);
                    sumI32 = __riscv_vadd_vv_i32m1(sumI32, __riscv_vget_v_i32m4_i32m1(qi32, 3), 32);
                    auto qi16 = __riscv_vnclip_wx_i16m2(qi32, 0, __RISCV_VXRM_RNU, 128);
                    auto qi8 = __riscv_vnclip_wx_i8m1(qi16, 0, __RISCV_VXRM_RNU, 128);
                    return __riscv_vreinterpret_v_u8m1_u32m1(__riscv_vreinterpret_v_i8m1_u8m1(qi8));
                };
                auto q0 = quant16(k4Base);
                auto q1 = quant16(k4Base + 4);
                // 8x8 u32 transpose via 3x vpack + vupack -> [r0,r2,r4,r6] and [r1,r3,r5,r7].
                vuint32m2_t tPair, uPair, wPair, eoPair;
                asm volatile(
                    "vsetvli t0, zero, e32, m1, ta, ma\n\t"
                    "vpack.vv %[t], %[q0], %[q1], 0\n\t"
                    : [t] "=&vr"(tPair)
                    : [q0] "vr"(q0), [q1] "vr"(q1)
                    : "t0");
                auto t0 = __riscv_vget_v_u32m2_u32m1(tPair, 0);
                auto t1 = __riscv_vget_v_u32m2_u32m1(tPair, 1);
                asm volatile(
                    "vsetvli t0, zero, e32, m1, ta, ma\n\t"
                    "vpack.vv %[u], %[t0], %[t1], 0\n\t"
                    : [u] "=&vr"(uPair)
                    : [t0] "vr"(t0), [t1] "vr"(t1)
                    : "t0");
                auto u0 = __riscv_vget_v_u32m2_u32m1(uPair, 0);
                auto u1 = __riscv_vget_v_u32m2_u32m1(uPair, 1);
                asm volatile(
                    "vsetvli t0, zero, e32, m1, ta, ma\n\t"
                    "vpack.vv %[w], %[u0], %[u1], 0\n\t"
                    : [w] "=&vr"(wPair)
                    : [u0] "vr"(u0), [u1] "vr"(u1)
                    : "t0");
                auto w0 = __riscv_vget_v_u32m2_u32m1(wPair, 0);
                auto w1 = __riscv_vget_v_u32m2_u32m1(wPair, 1);
                asm volatile(
                    "vsetvli t0, zero, e32, m1, ta, ma\n\t"
                    "vupack.vv %[eo], %[w0], %[w1], 2\n\t"
                    : [eo] "=&vr"(eoPair)
                    : [w0] "vr"(w0), [w1] "vr"(w1)
                    : "t0");
                auto vEven = __riscv_vreinterpret_v_u32m1_u8m1(__riscv_vget_v_u32m2_u32m1(eoPair, 0));
                auto vOdd = __riscv_vreinterpret_v_u32m1_u8m1(__riscv_vget_v_u32m2_u32m1(eoPair, 1));
                auto vEvenHi = __riscv_vsrl_vx_u8m1(vEven, 4, 128);
                auto vOddHi = __riscv_vsrl_vx_u8m1(vOdd, 4, 128);
                uint8_t* outPtr = dst->q[bk][k32][rg];
                vuint8m1_t packedHi, packedLo;
                asm volatile(
                    "vsetvli t0, zero, e8, m1, ta, ma\n\t"
                    "vnpack4.vv %[hi], %[eh], %[oh], 2\n\t"
                    "vnpack4.vv %[lo], %[el], %[ol], 2\n\t"
                    : [hi] "=&vr"(packedHi), [lo] "=&vr"(packedLo)
                    : [eh] "vr"(vEvenHi), [oh] "vr"(vOddHi), [el] "vr"(vEven), [ol] "vr"(vOdd)
                    : "t0");
                __riscv_vse8_v_u8m1(outPtr, packedHi, 128);
                __riscv_vse8_v_u8m1(outPtr + 128, packedLo, 128);
            }
            sumI32 = __riscv_vadd_vv_i32m1(sumI32, __riscv_vslidedown_vx_i32m1(sumI32, 1, 32), 32);
            sumI32 = __riscv_vadd_vv_i32m1(sumI32, __riscv_vslidedown_vx_i32m1(sumI32, 2, 32), 32);
            alignas(128) int32_t sumBuf[32];
            __riscv_vse32_v_i32m1(sumBuf, sumI32, 32);
            for (size_t r = 0; r < 8; ++r) {
                dst->aSum[rg][r * 8 + bk] = static_cast<_Float16>(r < validR ? sumBuf[r * 4] : 0);
            }
        }
    }
}

// Single DRAM pass: stage M=32 x K<=512 NC4HW4 slice into TCM while computing
// per-row absmax, then quantize, sum per block64, transpose, and nibble-pack from TCM.
static void packAM32Super(PackedASuper512* dst, const float* src, size_t rows, size_t r0, size_t k0, size_t blocksInSb,
                          float* tcmBuf) {
    const size_t kCount = blocksInSb * 64;
    const size_t validM = std::min<size_t>(32, rows - r0);
    const size_t vlM4 = validM * 4;
    const auto zeroM4 = __riscv_vfmv_v_f_f32m4(0.0f, 128);
    auto vmaxM4 = zeroM4;

    const float* baseSrc = src + k0 * rows + r0 * 4;
    for (size_t k = 0; k < kCount; k += 4) {
        auto x = __riscv_vle32_v_f32m4_tu(zeroM4, baseSrc + k * rows, vlM4);
        __riscv_vse32_v_f32m4(tcmBuf + (k / 4) * 128, x, 128);
        vmaxM4 = __riscv_vfmax_vv_f32m4(vmaxM4, __riscv_vfabs_v_f32m4(x, 128), 128);
    }

    vmaxM4 = __riscv_vfmax_vv_f32m4(vmaxM4, __riscv_vslidedown_vx_f32m4(vmaxM4, 1, 128), 128);
    vmaxM4 = __riscv_vfmax_vv_f32m4(vmaxM4, __riscv_vslidedown_vx_f32m4(vmaxM4, 2, 128), 128);
    alignas(128) float amaxBuf[128];
    __riscv_vse32_v_f32m4(amaxBuf, vmaxM4, 128);
    quantizeStagedToPackedA(dst, tcmBuf, amaxBuf, rows, r0, blocksInSb);
}

// Single contiguous DRAM pass over a 32 KB tiled FP16 superblock ([k4][rg][32]),
// staging into 1-cycle TCM while computing per-row absmax in FP16, then quantizing.
static void packAM32SuperFromFp16(PackedASuper512* dst, const _Float16* srcTile, size_t rows, size_t r0,
                                  size_t blocksInSb, _Float16* tcmFp16) {
    const size_t k4Count = blocksInSb * 16;
    auto vmaxH = __riscv_vfmv_v_f_f16m2(static_cast<_Float16>(0.0f), 128);
    for (size_t k4 = 0; k4 < k4Count; ++k4) {
        auto h = __riscv_vle16_v_f16m2(srcTile + k4 * 128, 128);
        __riscv_vse16_v_f16m2(tcmFp16 + k4 * 128, h, 128);
        vmaxH = __riscv_vfmax_vv_f16m2(vmaxH, __riscv_vfabs_v_f16m2(h, 128), 128);
    }
    vmaxH = __riscv_vfmax_vv_f16m2(vmaxH, __riscv_vslidedown_vx_f16m2(vmaxH, 1, 128), 128);
    vmaxH = __riscv_vfmax_vv_f16m2(vmaxH, __riscv_vslidedown_vx_f16m2(vmaxH, 2, 128), 128);
    alignas(128) float amaxBuf[128];
    __riscv_vse32_v_f32m4(amaxBuf, __riscv_vfwcvt_f_f_v_f32m4(vmaxH, 128), 128);
    quantizeStagedToPackedA(dst, tcmFp16, amaxBuf, rows, r0, blocksInSb);
}

// Expand scales to [scale, 16*scale] and transpose centeredOffset[32] into vfwmadot B order [4][64].
static inline void packBScalesAndOffsets(_Float16* bScale, _Float16* bOff, const Block64Weight* b, size_t blocksInSb) {
    const _Float16 s16 = static_cast<_Float16>(16.0f);
    for (size_t i = 0; i < blocksInSb; ++i) {
        auto s = __riscv_vle16_v_f16mf2(b[i].scales, 32);
        __riscv_vse16_v_f16mf2(bScale + i * 64, s, 32);
        __riscv_vse16_v_f16mf2(bScale + i * 64 + 32, __riscv_vfmul_vf_f16mf2(s, s16, 32), 32);
    }
    const _Float16* p[8];
    for (size_t i = 0; i < 8; ++i) {
        p[i] = i < blocksInSb ? b[i].centeredOffset : kZero32;
    }
    asm volatile(
        "vsetvli t0, zero, e16, mf2, ta, ma\n\t"
        "vle16.v v0, (%[p0])\n\t"
        "vle16.v v1, (%[p1])\n\t"
        "vle16.v v2, (%[p2])\n\t"
        "vle16.v v3, (%[p3])\n\t"
        "vle16.v v4, (%[p4])\n\t"
        "vle16.v v5, (%[p5])\n\t"
        "vle16.v v6, (%[p6])\n\t"
        "vle16.v v7, (%[p7])\n\t"
        "vsetvli t0, zero, e16, m1, ta, ma\n\t"
        "vpack.vv v8,  v0, v1, 0\n\t"
        "vpack.vv v10, v2, v3, 0\n\t"
        "vpack.vv v12, v4, v5, 0\n\t"
        "vpack.vv v14, v6, v7, 0\n\t"
        "vsetvli t0, zero, e32, m1, ta, ma\n\t"
        "vpack.vv v0, v8,  v10, 0\n\t"
        "vpack.vv v2, v12, v14, 0\n\t"
        "vsetvli t0, zero, e64, m1, ta, ma\n\t"
        "vpack.vv v4, v0, v2, 0\n\t"
        "vpack.vv v6, v1, v3, 0\n\t"
        "vs4r.v v4, (%[dst])\n\t"
        :
        : [p0] "r"(p[0]), [p1] "r"(p[1]), [p2] "r"(p[2]), [p3] "r"(p[3]),
          [p4] "r"(p[4]), [p5] "r"(p[5]), [p6] "r"(p[6]), [p7] "r"(p[7]), [dst] "r"(bOff)
        : "t0", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13",
          "v14", "v15");
}

// Compute M=32 x N=32 over one K<=512 superblock via native i4 vmadot.hp/vmadotus.hp,
// add centered offsets via vfwmadot, scale by per-row aScale, and accumulate in FP32 TCM.
// clang-format off
#define MNN_IME2_FLUSH_RG(srcReg, lblAcc, lblDone) \
    "vl1r.v v8, (%[asum])\n\t" \
    "addi %[asum], %[asum], 128\n\t" \
    "vsetvli t0, zero, e16, m4, ta, ma\n\t" \
    "vfwcvt.f.f.v v0, " #srcReg "\n\t" \
    "vsetvli t0, zero, e16, m1, ta, ma\n\t" \
    "vfwmadot v0, v8, v12\n\t" \
    "vfwmadot v2, v8, v13\n\t" \
    "vfwmadot v4, v8, v14\n\t" \
    "vfwmadot v6, v8, v15\n\t" \
    "vsetvli t0, zero, e32, m1, ta, ma\n\t" \
    "vrgather.vv v18, v16, v17\n\t" \
    "vadd.vi v17, v17, 4\n\t" \
    "vrgather.vv v19, v16, v17\n\t" \
    "vadd.vi v17, v17, 4\n\t" \
    "beqz %[first], " #lblAcc "f\n\t" \
    "vfmul.vv v0, v0, v18\n\t" "vfmul.vv v1, v1, v19\n\t" \
    "vfmul.vv v2, v2, v18\n\t" "vfmul.vv v3, v3, v19\n\t" \
    "vfmul.vv v4, v4, v18\n\t" "vfmul.vv v5, v5, v19\n\t" \
    "vfmul.vv v6, v6, v18\n\t" "vfmul.vv v7, v7, v19\n\t" \
    "vs8r.v v0, (%[acc])\n\t" \
    "addi %[acc], %[acc], 1024\n\t" \
    "j " #lblDone "f\n" \
    #lblAcc ":\n\t" \
    "vl4r.v v8, (%[acc])\n\t" \
    "vfmacc.vv v8,  v0, v18\n\t" "vfmacc.vv v9,  v1, v19\n\t" \
    "vfmacc.vv v10, v2, v18\n\t" "vfmacc.vv v11, v3, v19\n\t" \
    "vs4r.v v8, (%[acc])\n\t" \
    "addi %[acc], %[acc], 512\n\t" \
    "vl4r.v v8, (%[acc])\n\t" \
    "vfmacc.vv v8,  v4, v18\n\t" "vfmacc.vv v9,  v5, v19\n\t" \
    "vfmacc.vv v10, v6, v18\n\t" "vfmacc.vv v11, v7, v19\n\t" \
    "vs4r.v v8, (%[acc])\n\t" \
    "addi %[acc], %[acc], 512\n\t" \
    #lblDone ":\n\t"

static void gemmM32N32Super(const PackedASuper512* pa, const Block64Weight* b, const _Float16* bScale,
                            const _Float16* bOff, size_t blocksInSb, float* acc, int isFirst) {
    const uint8_t* bPtr = reinterpret_cast<const uint8_t*>(b);
    const uint8_t* bScalePtr = reinterpret_cast<const uint8_t*>(bScale);
    const uint8_t* aPtr = reinterpret_cast<const uint8_t*>(pa->q);
    const uint8_t* aSum = reinterpret_cast<const uint8_t*>(pa->aSum);
    size_t cnt = blocksInSb;
    asm volatile(
        "vsetvli t0, zero, e16, m8, ta, ma\n\t"
        "vmv.v.i v16, 0\n\t"
        "vmv.v.i v24, 0\n\t"
        "vsetvli t0, zero, e16, m1, ta, ma\n\t"
        "vl1r.v v0, (%[bscale])\n\t"
        "addi %[bscale], %[bscale], 128\n\t"
        "addi %[b], %[b], 128\n\t"
        "vl4r.v v4, (%[b])\n\t"
        "addi %[b], %[b], 512\n\t"
        "vl2r.v v2, (%[a])\n\t"
        "addi %[a], %[a], 256\n\t"
        "vl2r.v v12, (%[a])\n\t"
        "addi %[a], %[a], 256\n\t"
        ".p2align 4\n"
        "1:\n\t"
        "vl2r.v v14, (%[a])\n\t"
        "vmadot.hp   v16, v2, v4, v0, 4, i4\n\t"
        "vmadot.hp   v17, v2, v5, v0, 5, i4\n\t"
        "vmadot.hp   v18, v2, v6, v0, 6, i4\n\t"
        "vmadot.hp   v19, v2, v7, v0, 7, i4\n\t"
        "addi %[a], %[a], 256\n\t"
        "vmadot.hp   v20, v12, v4, v0, 4, i4\n\t"
        "vmadot.hp   v21, v12, v5, v0, 5, i4\n\t"
        "vmadot.hp   v22, v12, v6, v0, 6, i4\n\t"
        "vmadot.hp   v23, v12, v7, v0, 7, i4\n\t"
        "vl4r.v v8, (%[b])\n\t"
        "vmadot.hp   v24, v14, v4, v0, 4, i4\n\t"
        "vmadot.hp   v25, v14, v5, v0, 5, i4\n\t"
        "vmadot.hp   v26, v14, v6, v0, 6, i4\n\t"
        "vmadot.hp   v27, v14, v7, v0, 7, i4\n\t"
        "addi %[b], %[b], 512\n\t"
        "vmadotus.hp v16, v3, v4, v0, 0, i4\n\t"
        "vmadotus.hp v17, v3, v5, v0, 1, i4\n\t"
        "vmadotus.hp v18, v3, v6, v0, 2, i4\n\t"
        "vmadotus.hp v19, v3, v7, v0, 3, i4\n\t"
        "vl2r.v v2, (%[a])\n\t"
        "vmadotus.hp v20, v13, v4, v0, 0, i4\n\t"
        "vmadotus.hp v21, v13, v5, v0, 1, i4\n\t"
        "vmadotus.hp v22, v13, v6, v0, 2, i4\n\t"
        "vmadotus.hp v23, v13, v7, v0, 3, i4\n\t"
        "addi %[a], %[a], 256\n\t"
        "vmadotus.hp v24, v15, v4, v0, 0, i4\n\t"
        "vmadotus.hp v25, v15, v5, v0, 1, i4\n\t"
        "vmadotus.hp v26, v15, v6, v0, 2, i4\n\t"
        "vmadotus.hp v27, v15, v7, v0, 3, i4\n\t"
        "vl2r.v v12, (%[a])\n\t"
        "vmadot.hp   v28, v2, v4, v0, 4, i4\n\t"
        "vmadot.hp   v29, v2, v5, v0, 5, i4\n\t"
        "vmadot.hp   v30, v2, v6, v0, 6, i4\n\t"
        "vmadot.hp   v31, v2, v7, v0, 7, i4\n\t"
        "addi %[a], %[a], 256\n\t"
        "vl2r.v v14, (%[a])\n\t"
        "vmadotus.hp v28, v3, v4, v0, 0, i4\n\t"
        "vmadotus.hp v29, v3, v5, v0, 1, i4\n\t"
        "vmadotus.hp v30, v3, v6, v0, 2, i4\n\t"
        "vmadotus.hp v31, v3, v7, v0, 3, i4\n\t"
        "addi %[a], %[a], 256\n\t"

        "vl2r.v v2, (%[a])\n\t"
        "vmadot.hp   v16, v12, v8, v0, 4, i4\n\t"
        "vmadot.hp   v17, v12, v9, v0, 5, i4\n\t"
        "vmadot.hp   v18, v12, v10, v0, 6, i4\n\t"
        "vmadot.hp   v19, v12, v11, v0, 7, i4\n\t"
        "addi %[a], %[a], 256\n\t"
        "vmadot.hp   v20, v14, v8, v0, 4, i4\n\t"
        "vmadot.hp   v21, v14, v9, v0, 5, i4\n\t"
        "vmadot.hp   v22, v14, v10, v0, 6, i4\n\t"
        "vmadot.hp   v23, v14, v11, v0, 7, i4\n\t"
        "vmadot.hp   v24, v2, v8, v0, 4, i4\n\t"
        "vmadot.hp   v25, v2, v9, v0, 5, i4\n\t"
        "vmadot.hp   v26, v2, v10, v0, 6, i4\n\t"
        "vmadot.hp   v27, v2, v11, v0, 7, i4\n\t"
        "vmadotus.hp v16, v13, v8, v0, 0, i4\n\t"
        "vmadotus.hp v17, v13, v9, v0, 1, i4\n\t"
        "vmadotus.hp v18, v13, v10, v0, 2, i4\n\t"
        "vmadotus.hp v19, v13, v11, v0, 3, i4\n\t"
        "vl2r.v v12, (%[a])\n\t"
        "vmadotus.hp v20, v15, v8, v0, 0, i4\n\t"
        "vmadotus.hp v21, v15, v9, v0, 1, i4\n\t"
        "vmadotus.hp v22, v15, v10, v0, 2, i4\n\t"
        "vmadotus.hp v23, v15, v11, v0, 3, i4\n\t"
        "addi %[a], %[a], 256\n\t"
        "vmadotus.hp v24, v3, v8, v0, 0, i4\n\t"
        "vmadotus.hp v25, v3, v9, v0, 1, i4\n\t"
        "vmadotus.hp v26, v3, v10, v0, 2, i4\n\t"
        "vmadotus.hp v27, v3, v11, v0, 3, i4\n\t"
        "addi %[cnt], %[cnt], -1\n\t"
        "vmadot.hp   v28, v12, v8, v0, 4, i4\n\t"
        "vmadot.hp   v29, v12, v9, v0, 5, i4\n\t"
        "vmadot.hp   v30, v12, v10, v0, 6, i4\n\t"
        "vmadot.hp   v31, v12, v11, v0, 7, i4\n\t"
        "vmadotus.hp v28, v13, v8, v0, 0, i4\n\t"
        "vmadotus.hp v29, v13, v9, v0, 1, i4\n\t"
        "vmadotus.hp v30, v13, v10, v0, 2, i4\n\t"
        "vmadotus.hp v31, v13, v11, v0, 3, i4\n\t"
        "beqz %[cnt], 4f\n\t"
        "vl1r.v v0, (%[bscale])\n\t"
        "addi %[bscale], %[bscale], 128\n\t"
        "addi %[b], %[b], 128\n\t"
        "vl4r.v v4, (%[b])\n\t"
        "addi %[b], %[b], 512\n\t"
        "vl2r.v v2, (%[a])\n\t"
        "addi %[a], %[a], 256\n\t"
        "vl2r.v v12, (%[a])\n\t"
        "addi %[a], %[a], 256\n\t"
        "j 1b\n"
        "4:\n\t"
        "vl4r.v v12, (%[boff])\n\t"
        // Row group 0 (v16..v19); load aScale into v16 and ridx into v17 once v16..v19 are widened.
        "vl1r.v v8, (%[asum])\n\t"
        "addi %[asum], %[asum], 128\n\t"
        "vsetvli t0, zero, e16, m4, ta, ma\n\t"
        "vfwcvt.f.f.v v0, v16\n\t"
        "vsetvli t0, zero, e16, m1, ta, ma\n\t"
        "vfwmadot v0, v8, v12\n\t"
        "vfwmadot v2, v8, v13\n\t"
        "vfwmadot v4, v8, v14\n\t"
        "vfwmadot v6, v8, v15\n\t"
        "vsetvli t0, zero, e32, m1, ta, ma\n\t"
        "vle32.v v16, (%[ascale])\n\t"
        "vle32.v v17, (%[ridx])\n\t"
        "vrgather.vv v18, v16, v17\n\t"
        "vadd.vi v17, v17, 4\n\t"
        "vrgather.vv v19, v16, v17\n\t"
        "vadd.vi v17, v17, 4\n\t"
        "beqz %[first], 10f\n\t"
        "vfmul.vv v0, v0, v18\n\t" "vfmul.vv v1, v1, v19\n\t"
        "vfmul.vv v2, v2, v18\n\t" "vfmul.vv v3, v3, v19\n\t"
        "vfmul.vv v4, v4, v18\n\t" "vfmul.vv v5, v5, v19\n\t"
        "vfmul.vv v6, v6, v18\n\t" "vfmul.vv v7, v7, v19\n\t"
        "vs8r.v v0, (%[acc])\n\t"
        "addi %[acc], %[acc], 1024\n\t"
        "j 11f\n"
        "10:\n\t"
        "vl4r.v v8, (%[acc])\n\t"
        "vfmacc.vv v8,  v0, v18\n\t" "vfmacc.vv v9,  v1, v19\n\t"
        "vfmacc.vv v10, v2, v18\n\t" "vfmacc.vv v11, v3, v19\n\t"
        "vs4r.v v8, (%[acc])\n\t"
        "addi %[acc], %[acc], 512\n\t"
        "vl4r.v v8, (%[acc])\n\t"
        "vfmacc.vv v8,  v4, v18\n\t" "vfmacc.vv v9,  v5, v19\n\t"
        "vfmacc.vv v10, v6, v18\n\t" "vfmacc.vv v11, v7, v19\n\t"
        "vs4r.v v8, (%[acc])\n\t"
        "addi %[acc], %[acc], 512\n\t"
        "11:\n\t"
        MNN_IME2_FLUSH_RG(v20, 12, 13)
        MNN_IME2_FLUSH_RG(v24, 14, 15)
        MNN_IME2_FLUSH_RG(v28, 16, 17)
        : [a] "+&r"(aPtr), [b] "+&r"(bPtr), [bscale] "+&r"(bScalePtr), [acc] "+&r"(acc), [cnt] "+&r"(cnt),
          [asum] "+&r"(aSum)
        : [boff] "r"(bOff), [ascale] "r"(pa->aScale), [ridx] "r"(kRowScaleIdx), [first] "r"(isFirst)
        : "t0", "memory",
          "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
          "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30",
          "v31");
}
#undef MNN_IME2_FLUSH_RG
// clang-format on

static inline vfloat32m2_t siluM2Low(vfloat32m2_t x) {
    constexpr size_t vl = 64;
    auto z = __riscv_vfmin_vf_f32m2(__riscv_vfmax_vf_f32m2(__riscv_vfneg_v_f32m2(x, vl), -87.0f, vl), 87.0f, vl);
    auto n = __riscv_vfcvt_x_f_v_i32m2(__riscv_vfmul_vf_f32m2(z, 1.4426950408889634f, vl), vl);
    auto r = __riscv_vfnmsac_vf_f32m2(z, 0.6931471805599453f, __riscv_vfcvt_f_x_v_f32m2(n, vl), vl);
    auto y = __riscv_vfmv_v_f_f32m2(1.f / 720.f, vl);
    y = __riscv_vfmadd_vv_f32m2(y, r, __riscv_vfmv_v_f_f32m2(1.f / 120.f, vl), vl);
    y = __riscv_vfmadd_vv_f32m2(y, r, __riscv_vfmv_v_f_f32m2(1.f / 24.f, vl), vl);
    y = __riscv_vfmadd_vv_f32m2(y, r, __riscv_vfmv_v_f_f32m2(1.f / 6.f, vl), vl);
    y = __riscv_vfmadd_vv_f32m2(y, r, __riscv_vfmv_v_f_f32m2(0.5f, vl), vl);
    const auto one = __riscv_vfmv_v_f_f32m2(1.0f, vl);
    y = __riscv_vfmadd_vv_f32m2(y, r, one, vl);
    y = __riscv_vfmadd_vv_f32m2(y, r, one, vl);
    auto power = __riscv_vreinterpret_v_i32m2_f32m2(__riscv_vsll_vx_i32m2(__riscv_vadd_vx_i32m2(n, 127, vl), 23, vl));
    auto denom = __riscv_vfmacc_vv_f32m2(one, y, power, vl);
    auto inv = __riscv_vfrec7_v_f32m2(denom, vl);
    auto two = __riscv_vfmv_v_f_f32m2(2.0f, vl);
    inv = __riscv_vfmul_vv_f32m2(inv, __riscv_vfnmsac_vv_f32m2(two, denom, inv, vl), vl);
    auto result = __riscv_vfmul_vv_f32m2(x, inv, vl);
    return __riscv_vfmerge_vfm_f32m2(result, -0.0f, __riscv_vmflt_vf_f32m2_b16(x, -87.0f, vl), vl);
}

static inline bool hasNonZeroBias(const float* bias, size_t nOff) {
    if (!bias) {
        return false;
    }
    for (size_t j = 0; j < 32; ++j) {
        if (bias[nOff + j] != 0.0f) {
            return true;
        }
    }
    return false;
}

static void storeC4M32N32(float* dst, const float* acc, size_t rows, size_t r0, size_t n0, const float* bias,
                          bool hasBias, float cmin, float cmax, bool hasClamp) {
    const auto chan = __riscv_vand_vx_u32m1(__riscv_vid_v_u32m1(32), 3, 32);
    for (size_t rg = 0; rg < 4; ++rg) {
        const size_t rowBase = r0 + rg * 8;
        if (rowBase >= rows) {
            break;
        }
        const size_t validR = std::min<size_t>(8, rows - rowBase);
        const size_t vl = validR * 4;
        const float* tile = acc + rg * 256;
        for (size_t cg = 0; cg < 4; ++cg) {
            auto v0 = __riscv_vle32_v_f32m1(tile + cg * 64, 32);
            auto v1 = __riscv_vle32_v_f32m1(tile + cg * 64 + 32, 32);
            vfloat32m2_t pair;
            asm volatile("vsetvli t0, zero, e32, m1, ta, ma\n\tvupack.vv %0, %1, %2, 1\n\t"
                         : "=&vr"(pair)
                         : "vr"(v0), "vr"(v1)
                         : "t0");
            auto out0 = __riscv_vget_v_f32m2_f32m1(pair, 0);
            auto out1 = __riscv_vget_v_f32m2_f32m1(pair, 1);
            if (hasBias) {
                auto b0 = __riscv_vrgather_vv_f32m1(__riscv_vle32_v_f32m1(bias + n0 + cg * 8, 4), chan, 32);
                auto b1 = __riscv_vrgather_vv_f32m1(__riscv_vle32_v_f32m1(bias + n0 + cg * 8 + 4, 4), chan, 32);
                out0 = __riscv_vfadd_vv_f32m1(out0, b0, 32);
                out1 = __riscv_vfadd_vv_f32m1(out1, b1, 32);
            }
            if (hasClamp) {
                out0 = __riscv_vfmin_vf_f32m1(__riscv_vfmax_vf_f32m1(out0, cmin, 32), cmax, 32);
                out1 = __riscv_vfmin_vf_f32m1(__riscv_vfmax_vf_f32m1(out1, cmin, 32), cmax, 32);
            }
            __riscv_vse32_v_f32m1(dst + (n0 + cg * 8 + 0) * rows + rowBase * 4, out0, vl);
            __riscv_vse32_v_f32m1(dst + (n0 + cg * 8 + 4) * rows + rowBase * 4, out1, vl);
        }
    }
}

static void storeC4GateUpM32N32(float* dst, const float* accGate, const float* accUp, size_t rows, size_t r0, size_t n0,
                                const float* biasG, bool hasBiasG, float cminG, float cmaxG, bool hasClampG,
                                const float* biasU, bool hasBiasU, float cminU, float cmaxU, bool hasClampU) {
    const auto chan = __riscv_vand_vx_u32m1(__riscv_vid_v_u32m1(32), 3, 32);
    for (size_t rg = 0; rg < 4; ++rg) {
        const size_t rowBase = r0 + rg * 8;
        if (rowBase >= rows) {
            break;
        }
        const size_t validR = std::min<size_t>(8, rows - rowBase);
        const size_t vl = validR * 4;
        const float* tileG = accGate + rg * 256;
        const float* tileU = accUp + rg * 256;
        for (size_t cg = 0; cg < 4; ++cg) {
            vfloat32m2_t pairG, pairU;
            auto gLo = __riscv_vle32_v_f32m1(tileG + cg * 64, 32);
            auto gHi = __riscv_vle32_v_f32m1(tileG + cg * 64 + 32, 32);
            auto uLo = __riscv_vle32_v_f32m1(tileU + cg * 64, 32);
            auto uHi = __riscv_vle32_v_f32m1(tileU + cg * 64 + 32, 32);
            asm volatile("vsetvli t0, zero, e32, m1, ta, ma\n\tvupack.vv %0, %2, %3, 1\n\tvupack.vv %1, %4, %5, 1\n\t"
                         : "=&vr"(pairG), "=&vr"(pairU)
                         : "vr"(gLo), "vr"(gHi), "vr"(uLo), "vr"(uHi)
                         : "t0");
            if (hasBiasG) {
                auto b0 = __riscv_vrgather_vv_f32m1(__riscv_vle32_v_f32m1(biasG + n0 + cg * 8, 4), chan, 32);
                auto b1 = __riscv_vrgather_vv_f32m1(__riscv_vle32_v_f32m1(biasG + n0 + cg * 8 + 4, 4), chan, 32);
                pairG = __riscv_vfadd_vv_f32m2(pairG, __riscv_vcreate_v_f32m1_f32m2(b0, b1), 64);
            }
            if (hasClampG) {
                pairG = __riscv_vfmin_vf_f32m2(__riscv_vfmax_vf_f32m2(pairG, cminG, 64), cmaxG, 64);
            }
            if (hasBiasU) {
                auto b0 = __riscv_vrgather_vv_f32m1(__riscv_vle32_v_f32m1(biasU + n0 + cg * 8, 4), chan, 32);
                auto b1 = __riscv_vrgather_vv_f32m1(__riscv_vle32_v_f32m1(biasU + n0 + cg * 8 + 4, 4), chan, 32);
                pairU = __riscv_vfadd_vv_f32m2(pairU, __riscv_vcreate_v_f32m1_f32m2(b0, b1), 64);
            }
            if (hasClampU) {
                pairU = __riscv_vfmin_vf_f32m2(__riscv_vfmax_vf_f32m2(pairU, cminU, 64), cmaxU, 64);
            }
            auto outM2 = __riscv_vfmul_vv_f32m2(siluM2Low(pairG), pairU, 64);
            __riscv_vse32_v_f32m1(dst + (n0 + cg * 8 + 0) * rows + rowBase * 4, __riscv_vget_v_f32m2_f32m1(outM2, 0), vl);
            __riscv_vse32_v_f32m1(dst + (n0 + cg * 8 + 4) * rows + rowBase * 4, __riscv_vget_v_f32m2_f32m1(outM2, 1), vl);
        }
    }
}

// Compute silu(gate) * up into a contiguous 32 KB tiled FP16 superblock ([k4][rg][32]),
// halving DRAM write/read traffic and eliminating 8 KB power-of-2 stride conflicts for down_proj.
static void storeGateUpToFp16M32N32(_Float16* siluTile, const float* accGate, const float* accUp, size_t rows,
                                    size_t r0, size_t n0, size_t nLocal, const float* biasG, bool hasBiasG, float cminG,
                                    float cmaxG, bool hasClampG, const float* biasU, bool hasBiasU, float cminU,
                                    float cmaxU, bool hasClampU) {
    const auto vid = __riscv_vid_v_u32m1(32);
    const auto chan = __riscv_vand_vx_u32m1(vid, 3, 32);
    const auto zeroM1 = __riscv_vfmv_v_f_f32m1(0.0f, 32);
    for (size_t rg = 0; rg < 4; ++rg) {
        const size_t rowBase = r0 + rg * 8;
        if (rowBase >= rows) {
            break;
        }
        const size_t validR = std::min<size_t>(8, rows - rowBase);
        const float* tileG = accGate + rg * 256;
        const float* tileU = accUp + rg * 256;
        for (size_t cg = 0; cg < 4; ++cg) {
            vfloat32m2_t pairG, pairU;
            auto gLo = __riscv_vle32_v_f32m1(tileG + cg * 64, 32);
            auto gHi = __riscv_vle32_v_f32m1(tileG + cg * 64 + 32, 32);
            auto uLo = __riscv_vle32_v_f32m1(tileU + cg * 64, 32);
            auto uHi = __riscv_vle32_v_f32m1(tileU + cg * 64 + 32, 32);
            asm volatile("vsetvli t0, zero, e32, m1, ta, ma\n\tvupack.vv %0, %2, %3, 1\n\tvupack.vv %1, %4, %5, 1\n\t"
                         : "=&vr"(pairG), "=&vr"(pairU)
                         : "vr"(gLo), "vr"(gHi), "vr"(uLo), "vr"(uHi)
                         : "t0");
            if (hasBiasG) {
                auto b0 = __riscv_vrgather_vv_f32m1(__riscv_vle32_v_f32m1(biasG + n0 + cg * 8, 4), chan, 32);
                auto b1 = __riscv_vrgather_vv_f32m1(__riscv_vle32_v_f32m1(biasG + n0 + cg * 8 + 4, 4), chan, 32);
                pairG = __riscv_vfadd_vv_f32m2(pairG, __riscv_vcreate_v_f32m1_f32m2(b0, b1), 64);
            }
            if (hasClampG) {
                pairG = __riscv_vfmin_vf_f32m2(__riscv_vfmax_vf_f32m2(pairG, cminG, 64), cmaxG, 64);
            }
            if (hasBiasU) {
                auto b0 = __riscv_vrgather_vv_f32m1(__riscv_vle32_v_f32m1(biasU + n0 + cg * 8, 4), chan, 32);
                auto b1 = __riscv_vrgather_vv_f32m1(__riscv_vle32_v_f32m1(biasU + n0 + cg * 8 + 4, 4), chan, 32);
                pairU = __riscv_vfadd_vv_f32m2(pairU, __riscv_vcreate_v_f32m1_f32m2(b0, b1), 64);
            }
            if (hasClampU) {
                pairU = __riscv_vfmin_vf_f32m2(__riscv_vfmax_vf_f32m2(pairU, cminU, 64), cmaxU, 64);
            }
            auto outM2 = __riscv_vfmul_vv_f32m2(siluM2Low(pairG), pairU, 64);
            auto out0 = __riscv_vget_v_f32m2_f32m1(outM2, 0);
            auto out1 = __riscv_vget_v_f32m2_f32m1(outM2, 1);
            if (validR < 8) {
                const auto keep = __riscv_vmsltu_vx_u32m1_b32(vid, validR * 4, 32);
                out0 = __riscv_vmerge_vvm_f32m1(zeroM1, out0, keep, 32);
                out1 = __riscv_vmerge_vvm_f32m1(zeroM1, out1, keep, 32);
            }
            const size_t k4 = (nLocal + cg * 8) / 4;
            __riscv_vse16_v_f16mf2(siluTile + (k4 + 0) * 128 + rg * 32, __riscv_vfncvt_f_f_w_f16mf2(out0, 32), 32);
            __riscv_vse16_v_f16mf2(siluTile + (k4 + 1) * 128 + rg * 32, __riscv_vfncvt_f_f_w_f16mf2(out1, 32), 32);
        }
    }
}

struct Projection {
    const float* src = nullptr;
    const _Float16* srcFp16 = nullptr;
    float* dst = nullptr;
    _Float16* outFp16 = nullptr;
    const float* bias = nullptr;
    const Block64Weight* weight = nullptr;
    const float* biasUp = nullptr;
    const Block64Weight* weightUp = nullptr;
    PackedASuper512* packedA = nullptr;
    size_t rows = 0, depth = 0, cols = 0, workers = 0, numSb = 0, numM32 = 0, numN32 = 0;
    float clampMin = 0.0f, clampMax = 0.0f;
    bool hasClamp = false;
    float clampMinUp = 0.0f, clampMaxUp = 0.0f;
    bool hasClampUp = false;
    std::atomic<bool> ok{true};
};

static void packATask(size_t id, void* tcm, size_t bytes, void* user) {
    auto& p = *static_cast<Projection*>(user);
    if (!tcm || __riscv_vlenb() != 128 || bytes < 65536) {
        p.ok.store(false);
        return;
    }
    HalfRangeCheck range;
    const size_t total = p.numSb * p.numM32;
    const size_t totalBlocks = p.depth / 64;
    if (p.srcFp16) {
        auto* tcmFp16 = static_cast<_Float16*>(tcm);
        for (size_t idx = id; idx < total; idx += p.workers) {
            const size_t sb = idx / p.numM32;
            const size_t mi = idx % p.numM32;
            const size_t nb = std::min<size_t>(kMaxBlocksPerSuper, totalBlocks - sb * kMaxBlocksPerSuper);
            packAM32SuperFromFp16(&p.packedA[idx], p.srcFp16 + idx * 16384, p.rows, mi * 32, nb, tcmFp16);
        }
    } else {
        auto* tcmBuf = static_cast<float*>(tcm);
        for (size_t idx = id; idx < total; idx += p.workers) {
            const size_t sb = idx / p.numM32;
            const size_t mi = idx % p.numM32;
            const size_t nb = std::min<size_t>(kMaxBlocksPerSuper, totalBlocks - sb * kMaxBlocksPerSuper);
            packAM32Super(&p.packedA[idx], p.src, p.rows, mi * 32, sb * 512, nb, tcmBuf);
        }
    }
    if (range.failed()) {
        p.ok.store(false);
    }
}

static bool preparePackedA(Projection& p, void* workspace) {
    const size_t packedBytes = p.numSb * p.numM32 * sizeof(PackedASuper512);
    auto* cache = gActivationCache;
    if (cache && cache->src == p.src && cache->rows == p.rows && cache->depth == p.depth && cache->data) {
        p.packedA = static_cast<PackedASuper512*>(cache->data);
        return true;
    }
    if (cache) {
        if (void* slab = getReusableActivationSlab(packedBytes)) {
            p.packedA = static_cast<PackedASuper512*>(slab);
            cache->src = nullptr;
            cache->data = slab;
        }
    }
    if (!p.packedA) {
        p.packedA = reinterpret_cast<PackedASuper512*>((reinterpret_cast<uintptr_t>(workspace) + 127) & ~uintptr_t(127));
    }
    if (MNNSpacemitIme2RunTcmTasks(p.workers, packATask, &p) != p.workers || !p.ok.load()) {
        return false;
    }
    if (cache && p.packedA == cache->data) {
        cache->src = p.src;
        cache->rows = p.rows;
        cache->depth = p.depth;
    }
    return true;
}

static void gemmTask(size_t id, void* tcm, size_t bytes, void* user) {
    auto& p = *static_cast<Projection*>(user);
    if (!tcm || __riscv_vlenb() != 128 || bytes < kBScratchBytes + sizeof(PackedASuper512) + 4096) {
        p.ok.store(false);
        return;
    }
    HalfRangeCheck range;
    size_t mBegin = 0, mEnd = p.numM32, nBegin = 0, nEnd = p.numN32;
    if (p.numN32 >= p.workers) {
        nBegin = p.numN32 * id / p.workers;
        nEnd = p.numN32 * (id + 1) / p.workers;
    } else {
        mBegin = p.numM32 * id / p.workers;
        mEnd = p.numM32 * (id + 1) / p.workers;
    }
    if (mBegin >= mEnd || nBegin >= nEnd) {
        return;
    }

    const size_t avail = bytes - kBScratchBytes;
    const size_t mSpan = mEnd - mBegin;
    const size_t nSpan = nEnd - nBegin;
    size_t mBlk = std::min<size_t>(6, mSpan);
    while (mBlk > 1 && mBlk * sizeof(PackedASuper512) + mBlk * 4096 > avail) {
        --mBlk;
    }
    const size_t cAvail = avail - mBlk * sizeof(PackedASuper512);
    const size_t nBlk = std::min<size_t>(nSpan, cAvail / (mBlk * 4096));
    if (nBlk == 0) {
        p.ok.store(false);
        return;
    }

    auto* bTcm = static_cast<Block64Weight*>(tcm);
    auto* bScaleTcm = reinterpret_cast<_Float16*>(static_cast<uint8_t*>(tcm) + kMaxBlocksPerSuper * sizeof(Block64Weight));
    auto* bOffTcm = reinterpret_cast<_Float16*>(reinterpret_cast<uint8_t*>(bScaleTcm) + kBScaleBytes);
    auto* aTcm = reinterpret_cast<PackedASuper512*>(static_cast<uint8_t*>(tcm) + kBScratchBytes);
    auto* cTcm = reinterpret_cast<float*>(static_cast<uint8_t*>(tcm) + kBScratchBytes + mBlk * sizeof(PackedASuper512));

    const size_t totalBlocks = p.depth / 64;
    for (size_t m0 = mBegin; m0 < mEnd; m0 += mBlk) {
        const size_t mc = std::min(mBlk, mEnd - m0);
        for (size_t n0 = nBegin; n0 < nEnd; n0 += nBlk) {
            const size_t nc = std::min(nBlk, nEnd - n0);
            for (size_t sb = 0; sb < p.numSb; ++sb) {
                const size_t nb = std::min<size_t>(kMaxBlocksPerSuper, totalBlocks - sb * kMaxBlocksPerSuper);
                rvvMemcpy(aTcm, p.packedA + sb * p.numM32 + m0, mc * sizeof(PackedASuper512));
                for (size_t dn = 0; dn < nc; ++dn) {
                    const size_t ni = n0 + dn;
                    rvvMemcpy(bTcm, p.weight + ni * totalBlocks + sb * kMaxBlocksPerSuper, nb * sizeof(Block64Weight));
                    packBScalesAndOffsets(bScaleTcm, bOffTcm, bTcm, nb);
                    for (size_t dm = 0; dm < mc; ++dm) {
                        gemmM32N32Super(&aTcm[dm], bTcm, bScaleTcm, bOffTcm, nb, cTcm + (dn * mc + dm) * 1024, sb == 0);
                    }
                }
            }
            if (range.failed()) {
                p.ok.store(false);
                return;
            }
            for (size_t dn = 0; dn < nc; ++dn) {
                const size_t nOff = (n0 + dn) * 32;
                const bool hasBias = hasNonZeroBias(p.bias, nOff);
                for (size_t dm = 0; dm < mc; ++dm) {
                    storeC4M32N32(p.dst, cTcm + (dn * mc + dm) * 1024, p.rows, (m0 + dm) * 32, nOff, p.bias, hasBias,
                                  p.clampMin, p.clampMax, p.hasClamp);
                }
            }
        }
    }
    if (range.failed()) {
        p.ok.store(false);
    }
}

// Compute gate and up projections together in TCM, sharing each staged A block,
// then fuse SiLU(gate) * up in registers before writing the final NC4HW4 output.
static void gemmGateUpTask(size_t id, void* tcm, size_t bytes, void* user) {
    auto& p = *static_cast<Projection*>(user);
    if (!tcm || __riscv_vlenb() != 128 || bytes < kBScratchBytes + sizeof(PackedASuper512) + 8192) {
        p.ok.store(false);
        return;
    }
    HalfRangeCheck range;
    size_t mBegin = 0, mEnd = p.numM32, nBegin = 0, nEnd = p.numN32;
    if (p.numN32 >= p.workers) {
        nBegin = p.numN32 * id / p.workers;
        nEnd = p.numN32 * (id + 1) / p.workers;
    } else {
        mBegin = p.numM32 * id / p.workers;
        mEnd = p.numM32 * (id + 1) / p.workers;
    }
    if (mBegin >= mEnd || nBegin >= nEnd) {
        return;
    }

    const size_t avail = bytes - kBScratchBytes;
    const size_t mSpan = mEnd - mBegin;
    const size_t nSpan = nEnd - nBegin;
    size_t mBlk = std::min<size_t>(6, mSpan);
    while (mBlk > 1 && mBlk * sizeof(PackedASuper512) + mBlk * 8192 > avail) {
        --mBlk;
    }
    const size_t cAvail = avail - mBlk * sizeof(PackedASuper512);
    const size_t nBlk = std::min<size_t>(nSpan, cAvail / (mBlk * 8192));
    if (nBlk == 0) {
        p.ok.store(false);
        return;
    }

    auto* bTcm = static_cast<Block64Weight*>(tcm);
    auto* bScaleTcm = reinterpret_cast<_Float16*>(static_cast<uint8_t*>(tcm) + kMaxBlocksPerSuper * sizeof(Block64Weight));
    auto* bOffTcm = reinterpret_cast<_Float16*>(reinterpret_cast<uint8_t*>(bScaleTcm) + kBScaleBytes);
    auto* aTcm = reinterpret_cast<PackedASuper512*>(static_cast<uint8_t*>(tcm) + kBScratchBytes);
    auto* cGateTcm = reinterpret_cast<float*>(static_cast<uint8_t*>(tcm) + kBScratchBytes + mBlk * sizeof(PackedASuper512));
    auto* cUpTcm = cGateTcm + nBlk * mBlk * 1024;

    const size_t totalBlocks = p.depth / 64;
    for (size_t m0 = mBegin; m0 < mEnd; m0 += mBlk) {
        const size_t mc = std::min(mBlk, mEnd - m0);
        for (size_t n0 = nBegin; n0 < nEnd; n0 += nBlk) {
            const size_t nc = std::min(nBlk, nEnd - n0);
            for (size_t sb = 0; sb < p.numSb; ++sb) {
                const size_t nb = std::min<size_t>(kMaxBlocksPerSuper, totalBlocks - sb * kMaxBlocksPerSuper);
                rvvMemcpy(aTcm, p.packedA + sb * p.numM32 + m0, mc * sizeof(PackedASuper512));
                for (size_t dn = 0; dn < nc; ++dn) {
                    const size_t ni = n0 + dn;
                    rvvMemcpy(bTcm, p.weight + ni * totalBlocks + sb * kMaxBlocksPerSuper, nb * sizeof(Block64Weight));
                    packBScalesAndOffsets(bScaleTcm, bOffTcm, bTcm, nb);
                    for (size_t dm = 0; dm < mc; ++dm) {
                        gemmM32N32Super(&aTcm[dm], bTcm, bScaleTcm, bOffTcm, nb, cGateTcm + (dn * mc + dm) * 1024, sb == 0);
                    }
                    rvvMemcpy(bTcm, p.weightUp + ni * totalBlocks + sb * kMaxBlocksPerSuper, nb * sizeof(Block64Weight));
                    packBScalesAndOffsets(bScaleTcm, bOffTcm, bTcm, nb);
                    for (size_t dm = 0; dm < mc; ++dm) {
                        gemmM32N32Super(&aTcm[dm], bTcm, bScaleTcm, bOffTcm, nb, cUpTcm + (dn * mc + dm) * 1024, sb == 0);
                    }
                }
            }
            if (range.failed()) {
                p.ok.store(false);
                return;
            }
            for (size_t dn = 0; dn < nc; ++dn) {
                const size_t nOff = (n0 + dn) * 32;
                const bool hasBiasG = hasNonZeroBias(p.bias, nOff);
                const bool hasBiasU = hasNonZeroBias(p.biasUp, nOff);
                if (p.outFp16) {
                    const size_t sbOut = nOff / 512;
                    const size_t nLocal = nOff % 512;
                    for (size_t dm = 0; dm < mc; ++dm) {
                        storeGateUpToFp16M32N32(p.outFp16 + (sbOut * p.numM32 + m0 + dm) * 16384,
                                                cGateTcm + (dn * mc + dm) * 1024, cUpTcm + (dn * mc + dm) * 1024,
                                                p.rows, (m0 + dm) * 32, nOff, nLocal, p.bias, hasBiasG, p.clampMin,
                                                p.clampMax, p.hasClamp, p.biasUp, hasBiasU, p.clampMinUp, p.clampMaxUp,
                                                p.hasClampUp);
                    }
                } else {
                    for (size_t dm = 0; dm < mc; ++dm) {
                        storeC4GateUpM32N32(p.dst, cGateTcm + (dn * mc + dm) * 1024, cUpTcm + (dn * mc + dm) * 1024,
                                            p.rows, (m0 + dm) * 32, nOff, p.bias, hasBiasG, p.clampMin, p.clampMax,
                                            p.hasClamp, p.biasUp, hasBiasU, p.clampMinUp, p.clampMaxUp, p.hasClampUp);
                    }
                }
            }
        }
    }
    if (range.failed()) {
        p.ok.store(false);
    }
}

static bool shapeSupported(size_t rows, size_t depth, size_t cols, size_t workers) {
    return rows > 1 && rows <= 2048 && depth >= 256 && depth % 256 == 0 && cols > 0 && cols % 32 == 0 && workers > 0 &&
           workers <= 8;
}
} // namespace

MNNIme2ActivationCache* MNNIme2SwapActivationCache(MNNIme2ActivationCache* next) {
    auto* prev = gActivationCache;
    gActivationCache = next;
    return prev;
}

size_t MNNIme2HalfProjectionBytes(size_t rows, size_t depth, size_t cols, size_t workers) {
    if (!shapeSupported(rows, depth, cols, workers)) {
        return 0;
    }
    const size_t numSb = (depth + 511) / 512;
    const size_t numM32 = (rows + 31) / 32;
    return numSb * numM32 * sizeof(PackedASuper512) + 127;
}

bool MNNIme2HalfProjection(const float* src, float* dst, const float* bias, const float* clamp, const uint8_t* weight,
                           size_t rows, size_t depth, size_t cols, size_t workers, void* workspace) {
    if (!src || !dst || !weight || !workspace || !shapeSupported(rows, depth, cols, workers)) {
        return false;
    }
    Projection p;
    p.src = src;
    p.dst = dst;
    p.bias = bias;
    p.weight = reinterpret_cast<const Block64Weight*>(weight);
    p.rows = rows;
    p.depth = depth;
    p.cols = cols;
    p.workers = workers;
    p.numSb = (depth + 511) / 512;
    p.numM32 = (rows + 31) / 32;
    p.numN32 = cols / 32;
    p.hasClamp = clamp && (clamp[0] > -std::numeric_limits<float>::max() || clamp[1] < std::numeric_limits<float>::max());
    p.clampMin = p.hasClamp ? clamp[0] : 0.0f;
    p.clampMax = p.hasClamp ? clamp[1] : 0.0f;

    if (!preparePackedA(p, workspace)) {
        return false;
    }
    return MNNSpacemitIme2RunTcmTasks(workers, gemmTask, &p) == workers && p.ok.load();
}

bool MNNIme2HalfProjectionFromFp16(const void* srcFp16, float* dst, const float* bias, const float* clamp,
                                   const uint8_t* weight, size_t rows, size_t depth, size_t cols, size_t workers,
                                   void* workspace) {
    if (!srcFp16 || !dst || !weight || !workspace || depth % 512 != 0 || !shapeSupported(rows, depth, cols, workers)) {
        return false;
    }
    Projection p;
    p.srcFp16 = static_cast<const _Float16*>(srcFp16);
    p.dst = dst;
    p.bias = bias;
    p.weight = reinterpret_cast<const Block64Weight*>(weight);
    p.rows = rows;
    p.depth = depth;
    p.cols = cols;
    p.workers = workers;
    p.numSb = (depth + 511) / 512;
    p.numM32 = (rows + 31) / 32;
    p.numN32 = cols / 32;
    p.hasClamp = clamp && (clamp[0] > -std::numeric_limits<float>::max() || clamp[1] < std::numeric_limits<float>::max());
    p.clampMin = p.hasClamp ? clamp[0] : 0.0f;
    p.clampMax = p.hasClamp ? clamp[1] : 0.0f;
    p.packedA = reinterpret_cast<PackedASuper512*>((reinterpret_cast<uintptr_t>(workspace) + 127) & ~uintptr_t(127));
    if (MNNSpacemitIme2RunTcmTasks(workers, packATask, &p) != workers || !p.ok.load()) {
        return false;
    }
    return MNNSpacemitIme2RunTcmTasks(workers, gemmTask, &p) == workers && p.ok.load();
}

const void* MNNIme2HalfFusedGateUpToFp16(const float* src, const float* biasGate, const float* clampGate,
                                         const uint8_t* weightGate, const float* biasUp, const float* clampUp,
                                         const uint8_t* weightUp, size_t rows, size_t depth, size_t cols,
                                         size_t workers, void* workspace) {
    if (!src || !weightGate || !weightUp || !workspace || cols % 512 != 0 || !shapeSupported(rows, depth, cols, workers)) {
        return nullptr;
    }
    const size_t numSbOut = cols / 512;
    const size_t numM32 = (rows + 31) / 32;
    void* outSlab = getReusableHandoffSlab(numSbOut * numM32 * 16384 * sizeof(_Float16));
    if (!outSlab) {
        return nullptr;
    }
    Projection p;
    p.src = src;
    p.outFp16 = static_cast<_Float16*>(outSlab);
    p.bias = biasGate;
    p.weight = reinterpret_cast<const Block64Weight*>(weightGate);
    p.biasUp = biasUp;
    p.weightUp = reinterpret_cast<const Block64Weight*>(weightUp);
    p.rows = rows;
    p.depth = depth;
    p.cols = cols;
    p.workers = workers;
    p.numSb = (depth + 511) / 512;
    p.numM32 = numM32;
    p.numN32 = cols / 32;
    p.hasClamp = clampGate &&
                 (clampGate[0] > -std::numeric_limits<float>::max() || clampGate[1] < std::numeric_limits<float>::max());
    p.clampMin = p.hasClamp ? clampGate[0] : 0.0f;
    p.clampMax = p.hasClamp ? clampGate[1] : 0.0f;
    p.hasClampUp = clampUp &&
                   (clampUp[0] > -std::numeric_limits<float>::max() || clampUp[1] < std::numeric_limits<float>::max());
    p.clampMinUp = p.hasClampUp ? clampUp[0] : 0.0f;
    p.clampMaxUp = p.hasClampUp ? clampUp[1] : 0.0f;

    if (!preparePackedA(p, workspace)) {
        return nullptr;
    }
    if (MNNSpacemitIme2RunTcmTasks(workers, gemmGateUpTask, &p) == workers && p.ok.load()) {
        return outSlab;
    }
    return nullptr;
}

void MNNIme2ConvertTiledFp16ToC4(const void* srcFp16, float* dst, size_t rows, size_t depth) {
    if (!srcFp16 || !dst || rows == 0 || depth % 512 != 0) {
        return;
    }
    const auto* tiles = static_cast<const _Float16*>(srcFp16);
    const size_t numSb = depth / 512;
    const size_t numM32 = (rows + 31) / 32;
    for (size_t sb = 0; sb < numSb; ++sb) {
        for (size_t mi = 0; mi < numM32; ++mi) {
            const _Float16* tile = tiles + (sb * numM32 + mi) * 16384;
            const size_t r0 = mi * 32;
            const size_t validM = std::min<size_t>(32, rows - r0);
            for (size_t k4 = 0; k4 < 128; ++k4) {
                const size_t ch = sb * 512 + k4 * 4;
                float* dstRow = dst + ch * rows + r0 * 4;
                const _Float16* srcRow = tile + k4 * 128;
                for (size_t i = 0; i < validM * 4; ++i) {
                    dstRow[i] = static_cast<float>(srcRow[i]);
                }
            }
        }
    }
}

bool MNNIme2HalfFusedGateUp(const float* src, float* dst, const float* biasGate, const float* clampGate,
                            const uint8_t* weightGate, const float* biasUp, const float* clampUp,
                            const uint8_t* weightUp, size_t rows, size_t depth, size_t cols, size_t workers,
                            void* workspace) {
    if (!src || !dst || !weightGate || !weightUp || !workspace || !shapeSupported(rows, depth, cols, workers)) {
        return false;
    }
    Projection p;
    p.src = src;
    p.dst = dst;
    p.bias = biasGate;
    p.weight = reinterpret_cast<const Block64Weight*>(weightGate);
    p.biasUp = biasUp;
    p.weightUp = reinterpret_cast<const Block64Weight*>(weightUp);
    p.rows = rows;
    p.depth = depth;
    p.cols = cols;
    p.workers = workers;
    p.numSb = (depth + 511) / 512;
    p.numM32 = (rows + 31) / 32;
    p.numN32 = cols / 32;
    p.hasClamp = clampGate &&
                 (clampGate[0] > -std::numeric_limits<float>::max() || clampGate[1] < std::numeric_limits<float>::max());
    p.clampMin = p.hasClamp ? clampGate[0] : 0.0f;
    p.clampMax = p.hasClamp ? clampGate[1] : 0.0f;
    p.hasClampUp = clampUp &&
                   (clampUp[0] > -std::numeric_limits<float>::max() || clampUp[1] < std::numeric_limits<float>::max());
    p.clampMinUp = p.hasClampUp ? clampUp[0] : 0.0f;
    p.clampMaxUp = p.hasClampUp ? clampUp[1] : 0.0f;

    if (!preparePackedA(p, workspace)) {
        return false;
    }
    return MNNSpacemitIme2RunTcmTasks(workers, gemmGateUpTask, &p) == workers && p.ok.load();
}

#endif
