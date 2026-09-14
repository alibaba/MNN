// Direct-kernel regression test for the RVV matrix (Add/Sub/Prod) and
// Deconvolution depthwise kernels. See RVVMatrixDeconvTest.md.
//
// This file does not register its own test case: the assertions live in
// MNNTestRVVMatrixDeconvFunctions(), which test/op/DeconvolutionTest.cpp calls
// from the already-registered "op/Deconvolution" case. That keeps the checks on
// the default run_test.out path instead of hiding them behind a standalone
// main() guarded by a macro nobody defines.
//
// The _RVV kernels only exist in the MNNRVV object library, so the direct kernel
// calls and the function-table comparison both live under MNN_TEST_RVV_ENABLED.
// An MNN_USE_RVV=OFF build therefore still compiles and links; it just reports
// the RVV-specific checks as skipped.
#if defined(MNN_BUILD_STATIC_LIBS) && defined(__riscv)

#include <algorithm>
#include <cmath>
#include <cstring>
#include <thread>
#include <vector>
#include "MNNTestSuite.h"
#include "backend/cpu/compute/CommonOptFunction.h"
#include "backend/cpu/compute/ConvOpt.h"

using Matrix = void (*)(float*, const float*, const float*, size_t, size_t, size_t, size_t, size_t);
using Deconv = void (*)(const float*, float*, const float*, size_t, size_t, size_t, size_t, size_t);

#if MNN_TEST_RVV_ENABLED
#include <riscv_vector.h>
// The RVV kernels carry distinct names so the generic symbols stay available for
// targets without vector support; the table picks between them at runtime.
#define DECL_MATRIX(name) \
    void name##_RVV(float*, const float*, const float*, size_t, size_t, size_t, size_t, size_t);
DECL_MATRIX(MNNMatrixAdd)
DECL_MATRIX(MNNMatrixSub)
DECL_MATRIX(MNNMatrixProd)
void MNNDeconvRunForUnitDepthWise_RVV(const float*, float*, const float*, size_t, size_t, size_t, size_t, size_t);
#endif

static float sample(size_t index, unsigned seed) {
    return static_cast<float>(static_cast<int>((index * 37 + seed * 19) % 129) - 64) * 0.125f;
}

static bool same(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

#if MNN_TEST_RVV_ENABLED
static bool matrixCase(Matrix candidate, int op, size_t width, size_t height, size_t padding, int alias,
                       unsigned seed) {
    const size_t count = width * 4;
    const size_t aStride = count + padding;
    const size_t bStride = count + padding + (alias == 2 ? 0 : 4);
    const size_t cStride = alias == 1 ? aStride : (alias == 2 ? bStride : count + padding + 8);
    const size_t size = (std::max(aStride, std::max(bStride, cStride)) * (height + 1)) + 32;
    std::vector<float> a(size), b(size), c(size, -12345.0f);
    for (size_t i = 0; i < size; ++i) {
        a[i] = sample(i, seed);
        b[i] = sample(i, seed + 7);
    }
    const auto aOriginal = a;
    const auto bOriginal = b;
    const auto cOriginal = c;
    std::vector<float> expected = alias == 1 ? a : (alias == 2 ? b : c);
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < count; ++x) {
            const float av = a[8 + y * aStride + x];
            const float bv = b[8 + y * bStride + x];
            expected[8 + y * cStride + x] = op == 0 ? av + bv : (op == 1 ? av - bv : av * bv);
        }
    }
    float* dst = alias == 1 ? a.data() : (alias == 2 ? b.data() : c.data());
    candidate(dst + 8, a.data() + 8, b.data() + 8, width, cStride, aStride, bStride, height);
    const auto result = alias == 1 ? a : (alias == 2 ? b : c);
    if (!same(result, expected) || (alias != 1 && !same(a, aOriginal)) || (alias != 2 && !same(b, bOriginal))) {
        MNN_ERROR("RVV matrix mismatch op=%d w=%zu h=%zu pad=%zu alias=%d\n", op, width, height, padding, alias);
        return false;
    }
    return true;
}

static bool deconvCase(size_t fw, size_t fh, size_t dx, size_t rowMode, unsigned seed) {
    const size_t dy = rowMode == 0 ? fw * dx + 16 : (rowMode == 1 ? fw * dx : 4);
    const size_t wy = fw * 4 + (rowMode == 2 ? 12 : 0);
    const size_t srcSize = (fh + 1) * dy + (fw + 1) * dx + 32;
    const size_t weightSize = (fh + 1) * wy + fw * 4 + 32;
    std::vector<float> src(srcSize), weights(weightSize), dst(20, -34567.0f);
    for (size_t i = 0; i < src.size(); ++i)
        src[i] = sample(i, seed);
    for (size_t i = 0; i < weights.size(); ++i)
        weights[i] = sample(i, seed + 2);
    for (size_t i = 0; i < 4; ++i)
        dst[8 + i] = sample(i, seed + 3);
    const auto original = src;
    const auto weightsOriginal = weights;
    const auto dstOriginal = dst;
    auto expected = src;
    for (size_t fy = 0; fy < fh; ++fy) {
        for (size_t fx = 0; fx < fw; ++fx) {
            for (size_t c = 0; c < 4; ++c) {
                const size_t index = 8 + fy * dy + fx * dx + c;
                expected[index] = std::fma(dst[8 + c], weights[8 + fy * wy + fx * 4 + c], expected[index]);
            }
        }
    }
    MNNDeconvRunForUnitDepthWise_RVV(dst.data() + 8, src.data() + 8, weights.data() + 8, fw, fh, wy, dx, dy);
    if (!same(src, expected) || !same(weights, weightsOriginal) || !same(dst, dstOriginal)) {
        MNN_ERROR("RVV deconv mismatch fw=%zu fh=%zu dx=%zu rowMode=%zu\n", fw, fh, dx, rowMode);
        return false;
    }
    return true;
}

// MNNMatrixProd has no CoreFunctions slot, so its RVV kernel is reached through
// MNNMatrixProdCommon instead of the function table. That dispatch is the only
// thing that makes MNNMatrixProd_RVV live code, so exercise it through the same
// public entry point the callers use (Matrix::prod, CPUUnary square, ...).
static bool prodCommonCase(size_t width, size_t height, size_t padding, unsigned seed) {
    const size_t count = (width + padding) * height;
    std::vector<float> a(count), b(count), c(count), expected(count);
    for (size_t i = 0; i < count; ++i) {
        a[i] = sample(i, seed);
        b[i] = sample(i + 101, seed);
        c[i] = 0.0f;
        expected[i] = 0.0f;
    }
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            const size_t index = y * (width + padding) + x;
            expected[index] = a[index] * b[index];
        }
    }
    MNNMatrixProdCommon(c.data(), a.data(), b.data(), width, width + padding, width + padding, width + padding,
                        height);
    if (!same(c, expected)) {
        MNN_ERROR("MNNMatrixProdCommon mismatch width=%zu height=%zu pad=%zu\n", width, height, padding);
        return false;
    }
    return true;
}

struct Result {
    size_t matrix = 0;
    size_t prodCommon = 0;
    size_t deconv = 0;
    bool ok = true;
};

static Result runOnce(unsigned seed) {
    Result result;
    const size_t widths[] = {0, 1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 129};
    const size_t heights[] = {0, 1, 3, 7};
    const Matrix candidates[] = {MNNMatrixAdd_RVV, MNNMatrixSub_RVV, MNNMatrixProd_RVV};
    for (int op = 0; op < 3; ++op) {
        for (size_t w : widths)
            for (size_t h : heights)
                for (size_t pad : {size_t(0), size_t(12)}) {
                    for (int alias = 0; alias < 3; ++alias) {
                        result.ok &= matrixCase(candidates[op], op, w, h, pad, alias, seed);
                        ++result.matrix;
                    }
                }
    }
    for (size_t w : widths)
        for (size_t h : heights)
            for (size_t pad : {size_t(0), size_t(3), size_t(12)}) {
                result.ok &= prodCommonCase(w, h, pad, seed);
                ++result.prodCommon;
            }
    for (size_t fw : widths)
        for (size_t fh : heights) {
            for (size_t dx : {size_t(4), size_t(8), size_t(12), size_t(20)})
                for (size_t rowMode = 0; rowMode < 3; ++rowMode) {
                    result.ok &= deconvCase(fw, fh, dx, rowMode, seed);
                    ++result.deconv;
                }
        }
    return result;
}

static Result run(unsigned seed, int threads) {
    if (threads <= 1) {
        return runOnce(seed);
    }
    // Run the same sweep from several threads so a kernel that silently uses a
    // shared vector register file incorrectly is caught here instead of in the field.
    std::vector<Result> results(threads);
    std::vector<std::thread> workers;
    for (int t = 0; t < threads; ++t) {
        workers.emplace_back([&, t]() {
            results[t] = runOnce(seed + t);
        });
    }
    for (auto& worker : workers)
        worker.join();
    Result result;
    for (const auto& one : results) {
        result.matrix += one.matrix;
        result.prodCommon += one.prodCommon;
        result.deconv += one.deconv;
        result.ok &= one.ok;
    }
    return result;
}
#endif

bool MNNTestRVVMatrixDeconvFunctions() {
    auto core = MNN::MNNGetCoreFunctions();
    if (!core) {
        MNN_ERROR("RVV matrix/Deconv test requires an initialized CPU backend\n");
        return false;
    }
#if MNN_TEST_RVV_ENABLED
    if (core->supportRVV) {
        // A direct kernel test alone cannot catch an unregistered C++ overload,
        // so the dispatch of the shared table is checked first.
        if ((core->MNNMatrixAdd != MNNMatrixAdd_RVV) ||
            (core->MNNMatrixSub != MNNMatrixSub_RVV) ||
            (core->MNNDeconvRunForUnitDepthWise != MNNDeconvRunForUnitDepthWise_RVV)) {
            MNN_ERROR("Unexpected RVV matrix/Deconv function registration\n");
            return false;
        }
        const Result result = run(17, 4);
        if (!result.ok) {
            return false;
        }
        MNN_PRINT("RVV matrix/Deconv: matrix_cases=%zu prod_common_cases=%zu deconv_cases=%zu RVV=%d\n", result.matrix,
                  result.prodCommon, result.deconv, static_cast<int>(core->supportRVV));
    } else {
        MNN_PRINT("RVV matrix/Deconv: skipped, runtime reports supportRVV=0\n");
    }
#else
    MNN_PRINT("RVV matrix/Deconv: skipped, MNN_USE_RVV=OFF (RVV=%d)\n", static_cast<int>(core->supportRVV));
#endif
    return true;
}

#else

bool MNNTestRVVMatrixDeconvFunctions() {
    return true;
}

#endif
