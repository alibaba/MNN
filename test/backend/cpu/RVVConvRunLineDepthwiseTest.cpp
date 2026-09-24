// Verification for the RVV depthwise line-convolution kernel.
//
// Two things are checked, and the second is the one that matters:
//
//  1. MNNConvRunForLineDepthwise_RVV reproduces the scalar reference in
//     compute/ConvOpt.cpp over a sweep of widths, filter sizes, strides and
//     dilations, including widths either side of common LMUL=2 and LMUL=8
//     vector lengths.
//  2. CoreFunctions::MNNConvRunForLineDepthwise really points at the RVV kernel
//     at runtime. A correct kernel that is never registered changes nothing, and
//     a same-named definition would have C++ linkage while ConvOpt.h declares the
//     generic entry point extern "C" -- both copies end up in the library and
//     every call site keeps resolving to the generic one. Only the table check
//     can tell those two situations apart.
//
// This file registers no test case of its own: MNNTestRVVLineDepthwiseFunctions()
// is called from the already-registered "op/convolution/depthwise_conv" case in
// test/op/ConvolutionTest.cpp, so the checks run on the normal run_test.out path
// in both MNN_USE_RVV=ON and OFF builds.
//
// MNNConvRunForLineDepthwise_RVV only exists in the MNNRVV object library, so the
// kernel call and the table comparison live under MNN_TEST_RVV_ENABLED. An
// MNN_USE_RVV=OFF build still compiles and links; it reports the checks as skipped.
#if defined(MNN_BUILD_STATIC_LIBS) && defined(__riscv)

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>
#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif
#include "MNNTestSuite.h"
#include "backend/cpu/compute/CommonOptFunction.h"
#include "backend/cpu/compute/ConvOpt.h"

#if MNN_TEST_RVV_ENABLED
void MNNConvRunForLineDepthwise_RVV(float* dst, const float* src, const float* weight, size_t width,
                                    size_t src_w_setup, size_t fw, size_t fh, size_t dilateX_step, size_t dilateY_step,
                                    size_t height, size_t srcHStep, size_t dstHStep, const float* bias,
                                    const float* parameters);

static float nextValue(uint32_t& state) {
    state = state * 1664525u + 1013904223u;
    return static_cast<float>((state >> 8) % 2001) * 0.001f - 1.0f;
}

static bool checkShape(size_t width, size_t fw, size_t fh, size_t dilateX, size_t dilateY, size_t srcWSetup,
                       size_t height) {
    const size_t dstHStep = width * 4 + 8;
    const size_t srcHStep = (width - 1) * srcWSetup + (fh - 1) * dilateY + (fw - 1) * dilateX + 8;
    const size_t srcSize = height * srcHStep;
    const size_t dstSize = (height - 1) * dstHStep + width * 4 + 4;
    const size_t weightSize = fh * fw * 4;

    std::vector<float> src(srcSize), weight(weightSize), bias(4), params(2);
    std::vector<float> expected(dstSize, 0.0f), actual(dstSize, 0.0f);

    uint32_t state = 12345u + static_cast<uint32_t>(width * 31 + fw * 7 + fh * 3 + dilateX * 11 + dilateY * 13);
    for (size_t i = 0; i < srcSize; ++i) {
        src[i] = nextValue(state);
    }
    for (size_t i = 0; i < weightSize; ++i) {
        weight[i] = nextValue(state);
    }
    for (int i = 0; i < 4; ++i) {
        bias[i] = nextValue(state);
    }
    // Clamp both ends so the min/max pair is actually exercised.
    params[0] = -1.5f;
    params[1] = 1.5f;

    MNNConvRunForLineDepthwise(expected.data(), src.data(), weight.data(), width, srcWSetup, fw, fh, dilateX, dilateY,
                               height, srcHStep, dstHStep, bias.data(), params.data());
    MNNConvRunForLineDepthwise_RVV(actual.data(), src.data(), weight.data(), width, srcWSetup, fw, fh, dilateX,
                                   dilateY, height, srcHStep, dstHStep, bias.data(), params.data());

    // The RVV kernel accumulates with a fused multiply-add, the scalar reference
    // rounds after every multiply, so the two agree only to a relative tolerance.
    for (size_t i = 0; i < dstSize; ++i) {
        const float ref = expected[i];
        const float got = actual[i];
        const float tol = 1e-4f * (1.0f + std::fabs(ref) + std::fabs(got));
        if (!(std::fabs(ref - got) <= tol)) {
            MNN_ERROR("RVV depthwise mismatch: width=%zu fw=%zu fh=%zu dx=%zu dy=%zu sws=%zu height=%zu index=%zu "
                      "scalar=%f rvv=%f\n",
                      width, fw, fh, dilateX, dilateY, srcWSetup, height, i, ref, got);
            return false;
        }
    }
    return true;
}

#if defined(__linux__)
// Place an exact-length float buffer against either guard page. The two
// placements catch reads/writes before the first and after the last element.
class GuardedFloats {
public:
    GuardedFloats(size_t count, bool atEnd) {
        const long page = sysconf(_SC_PAGESIZE);
        if (page <= 0 || count == 0) {
            return;
        }
        const size_t pageBytes = static_cast<size_t>(page);
        const size_t valueBytes = count * sizeof(float);
        mDataBytes = ((valueBytes + pageBytes - 1) / pageBytes) * pageBytes;
        mMapBytes = mDataBytes + 2 * pageBytes;
        mMap = mmap(nullptr, mMapBytes, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (mMap == MAP_FAILED) {
            mMap = nullptr;
            return;
        }
        mDataPages = static_cast<char*>(mMap) + pageBytes;
        if (mprotect(mDataPages, mDataBytes, PROT_READ | PROT_WRITE) != 0) {
            munmap(mMap, mMapBytes);
            mMap = nullptr;
            return;
        }
        mValues = reinterpret_cast<float*>(static_cast<char*>(mDataPages) + (atEnd ? mDataBytes - valueBytes : 0));
    }

    ~GuardedFloats() {
        if (mMap) {
            munmap(mMap, mMapBytes);
        }
    }

    GuardedFloats(const GuardedFloats&) = delete;
    GuardedFloats& operator=(const GuardedFloats&) = delete;

    float* data() const { return mValues; }
    bool makeReadOnly() const { return mprotect(mDataPages, mDataBytes, PROT_READ) == 0; }

private:
    void* mMap = nullptr;
    void* mDataPages = nullptr;
    size_t mMapBytes = 0;
    size_t mDataBytes = 0;
    float* mValues = nullptr;
};

static bool checkBoundaryShape(size_t width, size_t height, size_t kernel, size_t dilation, size_t srcWSetup,
                               bool atEnd) {
    const size_t stride = srcWSetup / 4;
    const size_t inputWidth = (width ? (width - 1) * stride + (kernel - 1) * dilation + 1 : 1);
    const size_t dilateXStep = 4 * dilation;
    const size_t dilateYStep = 4 * inputWidth * dilation;
    const size_t srcHStep = 4 * inputWidth * stride;
    const size_t dstHStep = width * 4 + 4;
    const size_t srcSize =
        height == 0
            ? 1
            : (height - 1) * srcHStep +
                  (width ? (width - 1) * srcWSetup + (kernel - 1) * dilateYStep + (kernel - 1) * dilateXStep + 4 : 1);
    const size_t dstSize = height == 0 ? 1 : (height - 1) * dstHStep + (width ? width * 4 : 1);
    const float untouched = 1234.5f;
    GuardedFloats src(srcSize, atEnd), dst(dstSize, atEnd);
    if (!src.data() || !dst.data()) {
        MNN_ERROR("RVV depthwise guard-page allocation failed\n");
        return false;
    }
    std::vector<float> weight(kernel * kernel * 4), bias(4), params(2), expected(dstSize, untouched);
    uint32_t state = 12345u + static_cast<uint32_t>(width * 31 + height * 17 + kernel * 7 + dilation * 11 + srcWSetup);
    for (size_t i = 0; i < srcSize; ++i) {
        src.data()[i] = nextValue(state);
    }
    for (size_t i = 0; i < weight.size(); ++i) {
        weight[i] = nextValue(state);
    }
    for (size_t i = 0; i < bias.size(); ++i) {
        bias[i] = nextValue(state);
    }
    params[0] = -1.5f;
    params[1] = 1.5f;
    std::fill(dst.data(), dst.data() + dstSize, untouched);
    if (!src.makeReadOnly()) {
        MNN_ERROR("RVV depthwise source protection failed\n");
        return false;
    }

    MNNConvRunForLineDepthwise(expected.data(), src.data(), weight.data(), width, srcWSetup, kernel, kernel,
                               dilateXStep, dilateYStep, height, srcHStep, dstHStep, bias.data(), params.data());
    MNNConvRunForLineDepthwise_RVV(dst.data(), src.data(), weight.data(), width, srcWSetup, kernel, kernel, dilateXStep,
                                   dilateYStep, height, srcHStep, dstHStep, bias.data(), params.data());
    for (size_t i = 0; i < dstSize; ++i) {
        const float ref = expected[i];
        const float got = dst.data()[i];
        const float tol = 1e-4f * (1.0f + std::fabs(ref) + std::fabs(got));
        if (!(ref == untouched ? got == untouched : std::fabs(ref - got) <= tol)) {
            MNN_ERROR(
                "RVV depthwise guard mismatch: width=%zu height=%zu kernel=%zu dilation=%zu stride=%zu "
                "tail=%d index=%zu scalar=%f rvv=%f\n",
                width, height, kernel, dilation, stride, static_cast<int>(atEnd), i, ref, got);
            return false;
        }
    }
    return true;
}

static bool checkBoundaries() {
    const size_t widths[] = {0, 1, 7, 8, 9, 31, 32, 33};
    const size_t heights[] = {0, 1, 2};
    const size_t kernels[] = {1, 3};
    const size_t dilations[] = {1, 2};
    const size_t setups[] = {4, 8};
    for (size_t width : widths) {
        for (size_t height : heights) {
            for (size_t kernel : kernels) {
                for (size_t dilation : dilations) {
                    for (size_t setup : setups) {
                        for (int placement = 0; placement < 2; ++placement) {
                            if (!checkBoundaryShape(width, height, kernel, dilation, setup, placement != 0)) {
                                return false;
                            }
                        }
                    }
                }
            }
        }
    }
    return true;
}
#endif

static bool check() {
    // e32m2 has 8/16/32/64 lanes at VLEN=128/256/512/1024. Test each
    // boundary and its neighbors without requiring RVV intrinsics in this TU.
    const size_t widths[] = {1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129};
    const size_t heights[] = {1, 3};
    const size_t kernels[] = {1, 2, 3, 4};
    const size_t dilates[] = {1, 2};
    const size_t setups[] = {4, 8};
    for (size_t height : heights) {
        for (size_t fw : kernels) {
            for (size_t fh : kernels) {
                for (size_t dilateX : dilates) {
                    for (size_t dilateY : dilates) {
                        for (size_t setup : setups) {
                            for (size_t width : widths) {
                                if (!checkShape(width, fw, fh, dilateX, dilateY, setup, height)) {
                                    return false;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return true;
}
#endif

bool MNNTestRVVLineDepthwiseFunctions() {
    auto core = MNN::MNNGetCoreFunctions();
    if (!core) {
        MNN_ERROR("RVV depthwise test requires an initialized CPU backend\n");
        return false;
    }
#if MNN_TEST_RVV_ENABLED
    if (core->supportRVV) {
        if (core->MNNConvRunForLineDepthwise != MNNConvRunForLineDepthwise_RVV) {
            MNN_ERROR("RVV depthwise kernel is not registered on CoreFunctions\n");
            return false;
        }
        if (!check()) {
            return false;
        }
#if defined(__linux__)
        if (!checkBoundaries()) {
            return false;
        }
        MNN_PRINT("RVV depthwise guard pages: 384 boundary cases passed\n");
#endif
        MNN_PRINT("RVV depthwise line conv: scalar-oracle sweep matched, table dispatch confirmed (RVV=%d)\n",
                  static_cast<int>(core->supportRVV));
    } else {
        MNN_PRINT("RVV depthwise line conv: skipped, runtime reports supportRVV=0\n");
    }
#else
    MNN_PRINT("RVV depthwise line conv: skipped, MNN_USE_RVV=OFF (RVV=%d)\n", static_cast<int>(core->supportRVV));
#endif
    return true;
}

#else

bool MNNTestRVVLineDepthwiseFunctions() {
    return true;
}

#endif
