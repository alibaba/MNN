// Direct-kernel regression test for the RVV FP32 abs-max kernel. See
// RVVAbsMaxTest.md.
//
// This file does not register its own test case: the assertions live in
// MNNTestRVVAbsMaxFunctions(), which test/op/UnaryTest.cpp calls from the
// already-registered "op/unary/abs" case. That keeps the checks on the default
// run_test.out path instead of hiding them behind a standalone main() guarded by
// a macro nobody defines.
//
// MNNAbsMaxFP32_RVV only exists in the MNNRVV object library, so both the direct
// kernel call and the function-table comparison live under MNN_TEST_RVV_ENABLED.
// An MNN_USE_RVV=OFF build therefore still compiles and links; it just reports
// the RVV-specific checks as skipped.
#if defined(MNN_BUILD_STATIC_LIBS) && defined(__riscv)

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <thread>
#include <vector>
#include "MNNTestSuite.h"
#include "backend/cpu/compute/CommonOptFunction.h"

#if MNN_TEST_RVV_ENABLED
void MNNAbsMaxFP32_RVV(const float*, float*, size_t, size_t, int);

static bool check() {
    // Sweep all small channel tails, including a maximum before the final tail.
    for (int pack : {4, 8, 16}) {
        for (size_t depth = 0; depth <= 257; ++depth) {
            for (size_t plane : {0, 1, 3, 33}) {
                for (int pattern = 0; pattern < 6; ++pattern) {
                    std::vector<float> input(std::max(size_t(1), depth * plane * pack));
                    for (size_t i = 0; i < input.size(); ++i) {
                        input[i] = float(int(i * 113 % 65521) - 32760) / 31.0f;
                        if (pattern == 1)
                            input[i] = -0.0f;
                        if (pattern == 2)
                            input[i] = std::numeric_limits<float>::quiet_NaN();
                        if (pattern == 3 && i % 11 == 0)
                            input[i] = -std::numeric_limits<float>::infinity();
                        if (pattern >= 4)
                            input[i] = 1.0f;
                    }
                    if (pattern == 4)
                        input[(input.size() - 1) / 2] = -9999.0f;
                    if (pattern == 5)
                        input.back() = -9999.0f;
                    std::vector<float> expected(plane + 2, 987.0f), actual(plane + 2, 987.0f);
                    for (size_t i = 0; i < plane; ++i) {
                        float maximum = 0.0f;
                        for (size_t z = 0; z < depth; ++z) {
                            for (int k = 0; k < pack; ++k) {
                                const float v = std::fabs(input[(z * plane + i) * pack + k]);
                                if (v > maximum)
                                    maximum = v;
                            }
                        }
                        expected[i + 1] = maximum;
                    }
                    MNNAbsMaxFP32_RVV(input.data(), actual.data() + 1, depth, plane, pack);
                    if (std::memcmp(expected.data(), actual.data(), actual.size() * sizeof(float))) {
                        MNN_ERROR("AbsMax mismatch: pack=%d depth=%zu plane=%zu pattern=%d\n", pack, depth, plane,
                                  pattern);
                        return false;
                    }
                }
            }
        }
    }
    return true;
}
#endif

bool MNNTestRVVAbsMaxFunctions() {
    auto core = MNN::MNNGetCoreFunctions();
    if (!core) {
        MNN_ERROR("RVV abs-max test requires an initialized CPU backend\n");
        return false;
    }
#if MNN_TEST_RVV_ENABLED
    if (core->supportRVV) {
        // A direct kernel test alone cannot catch an unregistered C++ overload,
        // so the dispatch of the shared table is checked first. MNNAbsMax is
        // only registered to the RVV pointer when MNN_LOW_MEMORY is enabled.
#ifdef MNN_LOW_MEMORY
        if (core->MNNAbsMax != MNNAbsMaxFP32_RVV) {
            MNN_ERROR("Unexpected RVV abs-max function registration\n");
            return false;
        }
#endif
        const size_t perWorker = 18576;
        const int workers = 4;
        std::vector<int> results(workers, 0);
        std::vector<std::thread> threads;
        for (int i = 0; i < workers; ++i) {
            threads.emplace_back([&, i]() {
                results[i] = check() ? 1 : 0;
            });
        }
        for (auto& thread : threads)
            thread.join();
        for (int i = 0; i < workers; ++i) {
            if (!results[i]) {
                return false;
            }
        }
        MNN_PRINT("RVV abs-max: %d workers, %zu exact comparisons per worker passed, RVV=%d\n", workers, perWorker,
                  static_cast<int>(core->supportRVV));
    } else {
        MNN_PRINT("RVV abs-max: skipped, runtime reports supportRVV=0\n");
    }
#else
    MNN_PRINT("RVV abs-max: skipped, MNN_USE_RVV=OFF (RVV=%d)\n", static_cast<int>(core->supportRVV));
#endif
    return true;
}

#else

bool MNNTestRVVAbsMaxFunctions() {
    return true;
}

#endif
