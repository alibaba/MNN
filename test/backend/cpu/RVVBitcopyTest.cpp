// Copyright © 2026, Alibaba Group Holding Limited
//
// Direct-kernel regression test for the RVV CPURaster blit kernels:
//   MNN{4,2,1}BitcopyWithStride  and  MNN{4,2,1}BitcopyFast
//
// The checks live in MNNTestRVVBitcopyFunctions(), which test/op/RasterTest.cpp
// calls from the already-registered "op/blitc4" case. That keeps them on the
// default run_test.out path instead of hiding behind a standalone main().
//
// Everything RVV-specific stays under MNN_TEST_RVV_ENABLED: the kernels only
// exist in the MNNRVV object library, so an MNN_USE_RVV=OFF build has to keep
// compiling and linking without them.
//
// What this covers, and why each piece is here:
//   * size 0..257 -- the vector loop's tail (`size % VLMAX`) is where the
//     strided kernels differ from the scalar ones.
//   * stride / ds 0, 1, equal, mutually prime -- `stride == 0` is a broadcast
//     read, `ds == 0` is an overlapping write whose scalar order must survive.
//   * sentinel elements around both buffers, compared byte for byte -- an
//     out-of-range store shows up here and nowhere else.
//   * a plain-C reference implementation, so two kernels that agree on a wrong
//     answer still fail.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "MNNTestSuite.h"
#include "backend/cpu/compute/CommonOptFunction.h"

#if defined(MNN_BUILD_STATIC_LIBS) && defined(__riscv) && MNN_TEST_RVV_ENABLED
// The generic kernels have no shared header declaration (CommonOptFunction.h
// only carries the function-pointer fields), so they are declared here the same
// way they are defined: inside namespace MNN.
namespace MNN {
void MNN4BitcopyWithStride(uint8_t*, const uint8_t*, int, int, int);
void MNN2BitcopyWithStride(uint8_t*, const uint8_t*, int, int, int);
void MNN1BitcopyWithStride(uint8_t*, const uint8_t*, int, int, int);
void MNN4BitcopyFast(uint8_t*, const uint8_t*, int, int, int);
void MNN2BitcopyFast(uint8_t*, const uint8_t*, int, int, int);
void MNN1BitCopyFast(uint8_t*, const uint8_t*, int, int, int);
} // namespace MNN

// The RVV kernels are plain global symbols defined in
// source/backend/cpu/riscv/rvv/MNNBitcopy.cpp.
void MNN4BitcopyWithStride_RVV(uint8_t*, const uint8_t*, int, int, int);
void MNN2BitcopyWithStride_RVV(uint8_t*, const uint8_t*, int, int, int);
void MNN1BitcopyWithStride_RVV(uint8_t*, const uint8_t*, int, int, int);
void MNN4BitcopyFast_RVV(uint8_t*, const uint8_t*, int, int, int);
void MNN2BitcopyFast_RVV(uint8_t*, const uint8_t*, int, int, int);
void MNN1BitCopyFast_RVV(uint8_t*, const uint8_t*, int, int, int);
#endif

#if defined(MNN_BUILD_STATIC_LIBS) && defined(__riscv) && MNN_TEST_RVV_ENABLED
namespace {

// Generous guard band on both sides of every buffer. The generic Fast kernels
// read `*src` even when size == 0, so the source buffer must be padded or the
// scalar arm of the comparison would fault before it can be reported.
constexpr int kPad = 32;

// Element value in the source buffer: varied, never equal to the sentinel, and
// cheap to recompute on the reference side.
template <typename T>
inline T sourceValue(size_t index) {
    return static_cast<T>(index * 131u + 7u);
}

template <typename T>
inline T sentinelValue() {
    return static_cast<T>(0xA5A5A5A5u);
}

// dst[i * ds] = src[i * stride]. Written straight from the contract, with no
// reference to either implementation, so agreement between the two kernels on a
// wrong answer still fails here.
template <typename T>
void referenceCopy(T* dst, const T* src, int size, int stride, int ds) {
    for (int i = 0; i < size; ++i) {
        dst[static_cast<ptrdiff_t>(i) * ds] = src[static_cast<ptrdiff_t>(i) * stride];
    }
}

struct BitcopyKernels {
    const char* name;
    void (*generic)(uint8_t*, const uint8_t*, int, int, int);
    void (*rvv)(uint8_t*, const uint8_t*, int, int, int);
    // Element size in bytes.
    int bytes;
    // The Fast entry points are only routed to for ds == 1 with stride in
    // {0, 1}; outside that contract they are not required to match.
    bool restrictedToFastContract;
};

struct StrideCase {
    int stride;
    int ds;
};

// Coverage chosen against the contract rather than the code: 0 exercises the
// broadcast / overlapping forms, 1 is contiguous, >1 walks, and 2/3 are
// mutually prime so an off-by-one in the byte-stride conversion cannot cancel
// out.
const StrideCase kStrideCases[] = {
    {1, 1}, {1, 2}, {2, 1}, {2, 3}, {3, 1}, {1, 3}, {4, 4}, {5, 1},
    {1, 5}, {0, 1}, {0, 3}, {1, 0}, {0, 0}, {3, 3}, {7, 2}, {2, 7},
};

// Includes 0, every small size, VLMAX boundaries for e8/e16/e32 at VLEN=128 and
// 256, and a few long tails.
const int kSizes[] = {0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 127, 128, 129, 200, 257};

template <typename T>
bool checkKernel(const BitcopyKernels& kernel, int size, int stride, int ds, size_t& cases) {
    const ptrdiff_t srcSpan = (size > 0 ? static_cast<ptrdiff_t>(size - 1) * stride : 0) + 1;
    const ptrdiff_t dstSpan = (size > 0 ? static_cast<ptrdiff_t>(size - 1) * ds : 0) + 1;
    const size_t srcCount = static_cast<size_t>(srcSpan) + 2 * kPad;
    const size_t dstCount = static_cast<size_t>(dstSpan) + 2 * kPad;

    std::vector<T> src(srcCount, sentinelValue<T>());
    std::vector<T> dstGeneric(dstCount, sentinelValue<T>());
    std::vector<T> dstRvv(dstCount, sentinelValue<T>());
    std::vector<T> dstReference(dstCount, sentinelValue<T>());
    for (size_t i = 0; i < srcCount; ++i) {
        src[i] = sourceValue<T>(i);
    }

    auto* srcBase = src.data();
    referenceCopy(dstReference.data() + kPad, srcBase + kPad, size, stride, ds);
    const bool insideContract = !kernel.restrictedToFastContract || (ds == 1 && (stride == 0 || stride == 1));
    if (insideContract) {
        kernel.generic(reinterpret_cast<uint8_t*>(dstGeneric.data() + kPad),
                       reinterpret_cast<const uint8_t*>(srcBase + kPad), size, stride, ds);
    }
    kernel.rvv(reinterpret_cast<uint8_t*>(dstRvv.data() + kPad), reinterpret_cast<const uint8_t*>(srcBase + kPad), size,
               stride, ds);

    const size_t bytes = dstCount * sizeof(T);
    if (std::memcmp(dstRvv.data(), dstReference.data(), bytes) != 0) {
        MNN_ERROR("bitcopy %s: RVV result differs from the reference (size=%d stride=%d ds=%d)\n", kernel.name, size,
                  stride, ds);
        return false;
    }
    // Only call and compare the generic Fast kernels within their contract.
    // Outside it they may ignore ds and exceed the destination allocation.
    if (insideContract && std::memcmp(dstGeneric.data(), dstReference.data(), bytes) != 0) {
        MNN_ERROR("bitcopy %s: generic result differs from the reference (size=%d stride=%d ds=%d)\n", kernel.name,
                  size, stride, ds);
        return false;
    }
    ++cases;
    return true;
}

template <typename T>
bool checkKernelAllShapes(const BitcopyKernels& kernel, size_t& cases) {
    for (int size : kSizes) {
        for (const auto& strideCase : kStrideCases) {
            if (!checkKernel<T>(kernel, size, strideCase.stride, strideCase.ds, cases)) {
                return false;
            }
        }
    }
    return true;
}

} // namespace
#endif

bool MNNTestRVVBitcopyFunctions() {
#if defined(MNN_BUILD_STATIC_LIBS) && defined(__riscv)
    auto core = MNN::MNNGetCoreFunctions();
    if (core == nullptr) {
        MNN_ERROR("bitcopy test requires an initialized CPU backend\n");
        return false;
    }
#if MNN_TEST_RVV_ENABLED
    // A direct kernel test alone cannot catch a missing registration, so the
    // table contents are asserted as well: with RVV present every slot must be
    // the _RVV symbol, and with RVV absent it must be the generic one.
    const bool useRVV = core->supportRVV;
    MNN_PRINT("bitcopy RVV capability: %d, dispatch expected: %d\n", static_cast<int>(core->supportRVV),
              static_cast<int>(useRVV));
    const bool dispatchMatches = useRVV == (core->MNN4BitcopyWithStride == MNN4BitcopyWithStride_RVV) &&
                                 useRVV == (core->MNN2BitcopyWithStride == MNN2BitcopyWithStride_RVV) &&
                                 useRVV == (core->MNN1BitcopyWithStride == MNN1BitcopyWithStride_RVV) &&
                                 useRVV == (core->MNN4BitcopyFast == MNN4BitcopyFast_RVV) &&
                                 useRVV == (core->MNN2BitcopyFast == MNN2BitcopyFast_RVV) &&
                                 useRVV == (core->MNN1BitcopyFast == MNN1BitCopyFast_RVV);
    if (!dispatchMatches) {
        MNN_ERROR("bitcopy RVV dispatch mismatch: supportRVV=%d\n", static_cast<int>(core->supportRVV));
        return false;
    }
    if (!useRVV) {
        MNN_PRINT("bitcopy: dispatch passed; numerical checks skipped, runtime reports supportRVV=0\n");
        return true;
    }

    const BitcopyKernels kernels[] = {
        {"MNN4BitcopyWithStride", MNN::MNN4BitcopyWithStride, MNN4BitcopyWithStride_RVV, 4, false},
        {"MNN2BitcopyWithStride", MNN::MNN2BitcopyWithStride, MNN2BitcopyWithStride_RVV, 2, false},
        {"MNN1BitcopyWithStride", MNN::MNN1BitcopyWithStride, MNN1BitcopyWithStride_RVV, 1, false},
        {"MNN4BitcopyFast", MNN::MNN4BitcopyFast, MNN4BitcopyFast_RVV, 4, true},
        {"MNN2BitcopyFast", MNN::MNN2BitcopyFast, MNN2BitcopyFast_RVV, 2, true},
        {"MNN1BitCopyFast", MNN::MNN1BitCopyFast, MNN1BitCopyFast_RVV, 1, true},
    };
    size_t cases = 0;
    for (const auto& kernel : kernels) {
        bool ok = true;
        switch (kernel.bytes) {
            case 4:
                ok = checkKernelAllShapes<uint32_t>(kernel, cases);
                break;
            case 2:
                ok = checkKernelAllShapes<uint16_t>(kernel, cases);
                break;
            default:
                ok = checkKernelAllShapes<uint8_t>(kernel, cases);
                break;
        }
        if (!ok) {
            return false;
        }
    }
    MNN_PRINT("bitcopy RVV: %zu cases passed\n", cases);
#endif
#endif
    return true;
}
