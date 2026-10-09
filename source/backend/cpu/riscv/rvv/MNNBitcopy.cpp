// Copyright © 2026, Alibaba Group Holding Limited
//
// RVV kernels for the CPURaster blit family:
//   MNN{4,2,1}BitcopyWithStride  -- element copies with an arbitrary source
//                                   stride and destination stride
//   MNN{4,2,1}BitcopyFast        -- the restricted form the raster picks when
//                                   the destination is contiguous and the
//                                   source is either contiguous or a broadcast
//
// The generic implementations stay in compute/CommonOptFunction.cpp and are
// never removed: those object files are shared by every riscv64 machine while
// supportRVV is probed at runtime, so the scalar versions remain the build-time
// baseline. These kernels use their own _RVV symbol names and are wired into
// the base function table only when the runtime probe reports RVV.
//
// Contract notes carried over from the generic code:
//   * `stride` and `ds` are counted in ELEMENTS, not bytes. The RVV strided
//     load/store instructions take a byte stride, so every use here multiplies
//     by the element size before handing it to the intrinsic.
//   * `stride == 0` makes the source a broadcast; `ds == 0` makes every element
//     land on the same destination address.
//   * `size <= 0` must touch no memory at all. The generic MNN2BitcopyFast and
//     MNN1BitCopyFast read `*src` unconditionally even then, which is an
//     out-of-range read the vector versions do not reproduce.
//
// Two things were measured on hardware (VLEN=128, single core, m1..m8 sweep)
// and drive the shape of this file; both are easy to get wrong by reasoning
// alone:
//
//   1. A strided vector access costs roughly one memory transaction per lane,
//      so it does not reduce transactions the way a contiguous access does.
//      Against a scalar loop with a fixed stride -- which the hardware
//      prefetcher handles well -- the win has to come from somewhere else: by
//      keeping ONE of the two sides contiguous, the transaction count is
//      halved. That is why this file special-cases `ds == 1` and `stride == 1`
//      instead of always issuing a strided load paired with a strided store.
//      Measured at size 1024, that change alone moved `stride=2, ds=1` from
//      0.51x (a 2x regression) to 2.84x.
//
//   2. The best LMUL differs between the two shapes, so there are two trait
//      sets below rather than one. Contiguous copies get faster with a wider
//      vector (fewer instructions, transactions unchanged): m4 peaks at 8.01x
//      over scalar. Strided copies get slower once the vector is too wide,
//      because a wider vector means more independent addresses in flight per
//      instruction: m2 wins, while m8 collapses to 0.23x on a strided store.
//      Using a single LMUL for both costs most of the strided win.
//
// The kernels are VLEN-agnostic: every loop asks the hardware for its own vl
// through vsetvl, so the same object runs on VLEN 128 .. 1024 without a build
// flag.

#include <riscv_vector.h>

#include <stddef.h>
#include <stdint.h>

namespace {

// Below this the vector setup cannot be amortised: measured at VLEN=128, a
// single-element copy is ~11% slower vectorised, two elements break even, and
// three starts to pay. The raster does reach size 1 in practice (a region whose
// innermost extent is 1 has size[2] == 1), so this guard sits on a hot path
// rather than a theoretical one.
constexpr int kMinVectorSize = 3;

// dst[i * ds] = src[i * stride], one element per iteration.
//
// Kept for the cases a vector access cannot express:
//   * ds == 0 -- every lane writes the same address, and the RVV spec leaves
//     the result of an overlapping vector store undefined. The scalar order
//     ("the last element wins") is observable behaviour, so it is preserved
//     rather than approximated.
//   * a caller that violates the Fast entry point's stride/ds contract.
//   * an input too small for the vector setup to pay for itself.
template <typename T>
inline void bitcopyScalar(T* dst, const T* src, int size, int stride, int ds) {
    for (int i = 0; i < size; ++i) {
        *dst = *src;
        src += stride;
        dst += ds;
    }
}

// One trait set per element width. The RVV intrinsic names spell the operand
// type in two different ways, which is why SEW, the lane type and the LMUL are
// separate parameters here: `vsetvl` concatenates SEW with the LMUL
// (vsetvl_e32m4) while the memory intrinsics insert a `_v_` and the unsigned
// lane type in between (vlse32_v_u32m4).
#define MNN_BITCOPY_TRAITS(NAME, TYPE, SEW, VTYPE, LMUL, VECTOR_TYPE)                                           \
    template <>                                                                                                 \
    struct NAME<TYPE> {                                                                                         \
        using Vec = VECTOR_TYPE;                                                                                \
        static inline size_t setvl(size_t avl) { return __riscv_vsetvl_e##SEW##LMUL(avl); }                     \
        static inline Vec broadcast(TYPE value, size_t vl) { return __riscv_vmv_v_x_##VTYPE##LMUL(value, vl); } \
        static inline Vec loadStrided(const TYPE* base, ptrdiff_t byteStride, size_t vl) {                      \
            return __riscv_vlse##SEW##_v_##VTYPE##LMUL(base, byteStride, vl);                                   \
        }                                                                                                       \
        static inline void storeStrided(TYPE* base, ptrdiff_t byteStride, Vec value, size_t vl) {               \
            __riscv_vsse##SEW##_v_##VTYPE##LMUL(base, byteStride, value, vl);                                   \
        }                                                                                                       \
        static inline Vec loadContiguous(const TYPE* base, size_t vl) {                                         \
            return __riscv_vle##SEW##_v_##VTYPE##LMUL(base, vl);                                                \
        }                                                                                                       \
        static inline void storeContiguous(TYPE* base, Vec value, size_t vl) {                                  \
            __riscv_vse##SEW##_v_##VTYPE##LMUL(base, value, vl);                                                \
        }                                                                                                       \
    };

// Wide form (m4): the contiguous Fast path, where a wider vector only removes
// instructions.
template <typename T>
struct BitcopyWide;

MNN_BITCOPY_TRAITS(BitcopyWide, uint32_t, 32, u32, m4, vuint32m4_t)
MNN_BITCOPY_TRAITS(BitcopyWide, uint16_t, 16, u16, m4, vuint16m4_t)
MNN_BITCOPY_TRAITS(BitcopyWide, uint8_t, 8, u8, m4, vuint8m4_t)

// Narrow form (m2): the strided paths, where a wider vector would put more
// independent addresses in flight per instruction.
template <typename T>
struct BitcopyNarrow;

MNN_BITCOPY_TRAITS(BitcopyNarrow, uint32_t, 32, u32, m2, vuint32m2_t)
MNN_BITCOPY_TRAITS(BitcopyNarrow, uint16_t, 16, u16, m2, vuint16m2_t)
MNN_BITCOPY_TRAITS(BitcopyNarrow, uint8_t, 8, u8, m2, vuint8m2_t)

#undef MNN_BITCOPY_TRAITS

// dst[i * ds] = src[i * stride], vectorised.
//
// Whichever side is already contiguous is kept contiguous, which halves the
// number of memory transactions compared with issuing a strided access on both
// sides -- see the file header.
//
// The two families below deliberately use different trait sets. `ds == 1` has
// exactly one strided side and behaves like the contiguous case (m4 measured
// best across the real size range), while a strided destination or two strided
// sides needs the narrow form. A test-suite trace of 4B WithStride calls found
// 77.5% at stride=2/ds=1 and 19.3% at stride=4/ds=2. Most of these calls came
// from the ConvInt8 winograd test's FP32 reference graph; they are not a
// production-model workload distribution.
template <typename T>
inline void bitcopyWithStrideRvv(uint8_t* dstO, const uint8_t* srcO, int size, int stride, int ds) {
    auto* src = reinterpret_cast<const T*>(srcO);
    auto* dst = reinterpret_cast<T*>(dstO);
    if (size < kMinVectorSize) {
        // Keep each load after the preceding store: the two elements may alias.
        if (size > 0) {
            *dst = *src;
            if (size == 2) {
                dst[ds] = src[stride];
            }
        }
        return;
    }
    if (ds == 0) {
        bitcopyScalar(dst, src, size, stride, ds);
        return;
    }
    if (ds == 1) {
        // Destination contiguous: strided gather, contiguous store.
        if (stride == 0) {
            // Every element reads the same address, so issue one broadcast
            // instead of `size` identical strided loads.
            const T value = src[0];
            for (int i = 0; i < size;) {
                const size_t vl = BitcopyWide<T>::setvl(static_cast<size_t>(size - i));
                BitcopyWide<T>::storeContiguous(dst + i, BitcopyWide<T>::broadcast(value, vl), vl);
                i += static_cast<int>(vl);
            }
            return;
        }
        const ptrdiff_t srcBytes = static_cast<ptrdiff_t>(stride) * static_cast<ptrdiff_t>(sizeof(T));
        for (int i = 0; i < size;) {
            const size_t vl = BitcopyWide<T>::setvl(static_cast<size_t>(size - i));
            BitcopyWide<T>::storeContiguous(dst + i, BitcopyWide<T>::loadStrided(src, srcBytes, vl), vl);
            src += static_cast<ptrdiff_t>(vl) * stride;
            i += static_cast<int>(vl);
        }
        return;
    }
    // Strided destination, or two strided sides.
    using Traits = BitcopyNarrow<T>;
    const ptrdiff_t dstBytes = static_cast<ptrdiff_t>(ds) * static_cast<ptrdiff_t>(sizeof(T));
    if (stride == 0) {
        const T value = src[0];
        for (int i = 0; i < size;) {
            const size_t vl = Traits::setvl(static_cast<size_t>(size - i));
            Traits::storeStrided(dst, dstBytes, Traits::broadcast(value, vl), vl);
            dst += static_cast<ptrdiff_t>(vl) * ds;
            i += static_cast<int>(vl);
        }
        return;
    }
    if (stride == 1) {
        // Source contiguous: contiguous load, strided scatter.
        for (int i = 0; i < size;) {
            const size_t vl = Traits::setvl(static_cast<size_t>(size - i));
            Traits::storeStrided(dst, dstBytes, Traits::loadContiguous(src + i, vl), vl);
            dst += static_cast<ptrdiff_t>(vl) * ds;
            i += static_cast<int>(vl);
        }
        return;
    }
    const ptrdiff_t srcBytes = static_cast<ptrdiff_t>(stride) * static_cast<ptrdiff_t>(sizeof(T));
    for (int i = 0; i < size;) {
        const size_t vl = Traits::setvl(static_cast<size_t>(size - i));
        Traits::storeStrided(dst, dstBytes, Traits::loadStrided(src, srcBytes, vl), vl);
        src += static_cast<ptrdiff_t>(vl) * stride;
        dst += static_cast<ptrdiff_t>(vl) * ds;
        i += static_cast<int>(vl);
    }
}

// Keep the broadcast value in a scalar argument so the compiler emits vmv.v.x
// instead of a zero-stride vector load. Small broadcasts stay in the caller:
// the extra dispatch only pays off above eight elements on the measured core.
__attribute__((noinline)) void broadcastLarge(uint32_t* dst, uint32_t value, int size) {
    for (int i = 0; i < size;) {
        const size_t vl = __riscv_vsetvl_e32m4(size - i);
        const auto v = __riscv_vmv_v_x_u32m4(value, vl);
        __riscv_vse32_v_u32m4(dst + i, v, vl);
        i += static_cast<int>(vl);
    }
}

// The raster only routes here for ds == 1 with stride in {0, 1}. Anything else
// falls back to the generic loop so a caller that breaks that contract still
// gets correct results instead of silently ignoring ds.
template <typename T>
inline void bitcopyFastRvv(uint8_t* dstO, const uint8_t* srcO, int size, int stride, int ds) {
    using Traits = BitcopyWide<T>;
    auto* src = reinterpret_cast<const T*>(srcO);
    auto* dst = reinterpret_cast<T*>(dstO);
    if (size < kMinVectorSize || ds != 1 || stride < 0 || stride > 1) {
        bitcopyScalar(dst, src, size, stride, ds);
        return;
    }
    for (int i = 0; i < size;) {
        const size_t vl = Traits::setvl(static_cast<size_t>(size - i));
        if (stride == 1) {
            Traits::storeContiguous(dst + i, Traits::loadContiguous(src + i, vl), vl);
        } else {
            if (sizeof(T) == 4 && size > 8) {
                return broadcastLarge(reinterpret_cast<uint32_t*>(dst + i), src[0], size - i);
            }
            Traits::storeContiguous(dst + i, Traits::broadcast(src[0], vl), vl);
        }
        i += static_cast<int>(vl);
    }
}

} // namespace

void MNN4BitcopyWithStride_RVV(uint8_t* dstO, const uint8_t* srcO, int size, int stride, int ds) {
    bitcopyWithStrideRvv<uint32_t>(dstO, srcO, size, stride, ds);
}

void MNN2BitcopyWithStride_RVV(uint8_t* dstO, const uint8_t* srcO, int size, int stride, int ds) {
    bitcopyWithStrideRvv<uint16_t>(dstO, srcO, size, stride, ds);
}

void MNN1BitcopyWithStride_RVV(uint8_t* dstO, const uint8_t* srcO, int size, int stride, int ds) {
    bitcopyWithStrideRvv<uint8_t>(dstO, srcO, size, stride, ds);
}

void MNN4BitcopyFast_RVV(uint8_t* dstO, const uint8_t* srcO, int size, int stride, int ds) {
    bitcopyFastRvv<uint32_t>(dstO, srcO, size, stride, ds);
}

void MNN2BitcopyFast_RVV(uint8_t* dstO, const uint8_t* srcO, int size, int stride, int ds) {
    bitcopyFastRvv<uint16_t>(dstO, srcO, size, stride, ds);
}

void MNN1BitCopyFast_RVV(uint8_t* dstO, const uint8_t* srcO, int size, int stride, int ds) {
    bitcopyFastRvv<uint8_t>(dstO, srcO, size, stride, ds);
}
