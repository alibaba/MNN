// Copyright © 2026, Alibaba Group Holding Limited
#if defined(MNN_BUILD_STATIC_LIBS) && defined(__riscv)
#include "MNNTestSuite.h"
#include "backend/cpu/riscv/rvv/MNNRvvMatMulFunctions.hpp"
#include <cmath>
#include <cstring>
#include <limits>
#include <thread>
#include <vector>

static bool sameGEMV(float a, float b) {
    if (std::isnan(a) || std::isnan(b)) {
        return std::isnan(a) && std::isnan(b);
    }
    // MatMul compares numerical values, not bit patterns: contracting a tiny
    // product into an addition can change the sign of an underflowed zero.
    // NaN/Inf classification is still checked; nonzero finite results use tolerance.
    return a == b || (std::isfinite(a) && std::isfinite(b) && std::abs(a - b) <= 1e-5f * (1 + std::abs(b)));
}
#endif

bool MNNTestRVVMatMulFunctions() {
#if defined(MNN_BUILD_STATIC_LIBS) && defined(__riscv)
    auto core = MNN::MNNGetCoreFunctions();
    if (!core) {
        MNN_ERROR("GEMV test requires an initialized CPU backend\n");
        return false;
    }
    auto function = core->MNNComputeMatMulForE_1;
#if MNN_TEST_RVV_ENABLED
    auto expected = core->supportRVV ? MNNComputeMatMulForE_1_RVV : MNNComputeMatMulForE_1;
#else
    auto expected = MNNComputeMatMulForE_1;
#endif
    if (function != expected) {
        MNN_ERROR("Unexpected E=1 RVV registration\n");
        return false;
    }
    const int lengths[] = {0, 1, 3, 7, 16, 17, 33, 127};
    const int channels[] = {0, 1, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 129};
    size_t cases = 0;
    for (int k : lengths)
        for (int h : channels)
            for (bool transpose : {false, true})
                for (bool biasOn : {false, true})
                    for (int threads : {1, 4})
                        for (bool special : {false, true}) {
                            auto fail = [&](int line) {
                                MNN_ERROR("GEMV mismatch line=%d k=%d h=%d transpose=%d bias=%d threads=%d special=%d\n",
                                          line, k, h, transpose, biasOn, threads, special);
                                return false;
                            };
                            auto equal = [&](float actual, float reference) {
                                if (transpose || function == MNNComputeMatMulForE_1) {
                                    return std::memcmp(&actual, &reference, sizeof(float)) == 0;
                                }
                                return sameGEMV(actual, reference);
                            };
                            std::vector<float> a(k + 8, 12345), b(k * h + 8, 12345), bias(h + 8, 12345);
                            for (int z = 0; z < k; ++z)
                                a[z + 4] = (z % 13 - 6) * 0.125f;
                            for (int z = 0; z < k * h; ++z)
                                b[z + 4] = (z % 17 - 8) * 0.0625f;
                            for (int y = 0; y < h; ++y)
                                bias[y + 4] = (y % 7 - 3) * 0.25f;
                            if (special) {
                                const float values[] = {0.0f,
                                                        -0.0f,
                                                        std::numeric_limits<float>::denorm_min(),
                                                        std::numeric_limits<float>::infinity(),
                                                        -std::numeric_limits<float>::infinity(),
                                                        std::numeric_limits<float>::quiet_NaN()};
                                for (int z = 0; z < k; ++z)
                                    a[z + 4] = values[z % 6];
                                for (int y = 0; y < h; ++y)
                                    bias[y + 4] = values[y % 6];
                            }
                            auto aBefore = a, bBefore = b, biasBefore = bias;
                            std::vector<float> reference(h + 8, 12345), combined(h + 8, 12345);
                            MatMulParam param = {1, k, h, threads, false, transpose};
                            for (int t = 0; t < threads; ++t)
                                MNNComputeMatMulForE_1(a.data() + 4, b.data() + 4, reference.data() + 4,
                                                       biasOn ? bias.data() + 4 : nullptr, &param, t);
                            std::vector<int> owners(h, 0);
                            for (int t = threads - 1; t >= 0; --t) {
                                std::vector<float> part(h + 8, 12345);
                                function(a.data() + 4, b.data() + 4, part.data() + 4,
                                         biasOn ? bias.data() + 4 : nullptr, &param, t);
                                for (int y = 0; y < h + 8; ++y) {
                                    if (y < 4 || y >= h + 4) {
                                        if (part[y] != 12345)
                                            return fail(__LINE__);
                                    } else if (part[y] != 12345) {
                                        ++owners[y - 4];
                                        if (!equal(part[y], reference[y]))
                                            return fail(__LINE__);
                                    }
                                }
                            }
                            for (int count : owners)
                                if (count != 1)
                                    return fail(__LINE__);
                            std::vector<std::thread> workers;
                            for (int t = 0; t < threads; ++t)
                                workers.emplace_back([&, t]() {
                                    function(a.data() + 4, b.data() + 4, combined.data() + 4,
                                             biasOn ? bias.data() + 4 : nullptr, &param, t);
                                });
                            for (auto& worker : workers)
                                worker.join();
                            for (int y = 0; y < h + 8; ++y)
                                if (!equal(combined[y], reference[y]))
                                    return fail(__LINE__);
                            if (std::memcmp(a.data(), aBefore.data(), a.size() * sizeof(float)) ||
                                std::memcmp(b.data(), bBefore.data(), b.size() * sizeof(float)) ||
                                std::memcmp(bias.data(), biasBefore.data(), bias.size() * sizeof(float)))
                                return fail(__LINE__);
                            ++cases;
                        }
    MNN_PRINT("E=1 GEMV dispatch: %zu cases passed, RVV=%d\n", cases,
              static_cast<int>(function != MNNComputeMatMulForE_1));
#endif
    return true;
}
