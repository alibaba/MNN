// Copyright 2018 Alibaba Group Holding Limited. All rights reserved.
#if defined(MNN_USE_RVV)
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <sys/types.h>
#include <vector>

#include "backend/cpu/compute/CommonOptFunction.h"

#ifdef MNN_RVV_SUM_TAIL_TEST_MAIN
void MNNSumByAxisLForMatmul_A_RVV(float* dest, int8_t* source, const float* scale, ssize_t realDstCount,
                                  SumByAxisParams sumParams);
#define RVV_SUM_KERNEL MNNSumByAxisLForMatmul_A_RVV
#else
#include "MNNTestSuite.h"
#define RVV_SUM_KERNEL MNN::MNNGetCoreFunctions()->MNNSumByAxisLForMatmul_A
#endif

namespace {
bool checkCase(int channels, int positions) {
    const int lp = 16;
    const int lu = (channels + lp - 1) / lp;
    const float guard = -98765.0f;
    const float scale = 1.0f;

    // Packed A layout for blockNum=kernelxy=1: [LU, realDstCount, LP].
    // Distinct values at each output position exercise both vacc0 and vacc1.
    // Poison padding so that a read beyond the valid input channels also fails.
    std::vector<int8_t> input(lu * positions * lp, static_cast<int8_t>(101));
    for (int c = 0; c < channels; ++c) {
        for (int w = 0; w < positions; ++w) {
            input[(c / lp) * positions * lp + w * lp + c % lp] = static_cast<int8_t>(w + 1);
        }
    }
    std::vector<float> output(positions + 2, guard);
    SumByAxisParams params = {};
    params.kernelCountUnitDouble = lu;
    params.unitColBufferSize = static_cast<ssize_t>(input.size());
    params.DST_XUNIT = 8;
    params.SRC_UNIT = lp;
    params.blockNum = 1;
    params.oneScale = 1;
    params.valid = channels % lp;
    params.kernelxy = 1;
    params.LU = lu;
    params.inputBlock = 0;

    RVV_SUM_KERNEL(output.data() + 1, input.data(), &scale, positions, params);

    bool passed = output.front() == guard && output.back() == guard;
    for (int w = 0; w < positions; ++w) {
        // Independent oracle: every valid channel at position w contains w+1.
        // These small integers are exact in FP32: no tolerance is needed.
        const float expected = static_cast<float>(channels * (w + 1));
        if (output[w + 1] != expected) {
            std::printf("FAIL C=%d E=%d position=%d: expected=%.0f actual=%.0f\n", channels, positions, w, expected,
                        output[w + 1]);
            passed = false;
        }
    }
    if (output.front() != guard || output.back() != guard) {
        std::printf("FAIL C=%d E=%d: output guard overwritten\n", channels, positions);
    }
    if (passed) {
        std::printf("PASS C=%d E=%d\n", channels, positions);
    }
    return passed;
}

bool runSumTailCases() {
    size_t vlenBytes = 0;
    __asm__ volatile("csrr %0, vlenb" : "=r"(vlenBytes));
    std::printf("RVV sum regression: VLEN=%zu bits\n", vlenBytes * 8);

    // C=16,E=1 covers the separate fast path as a control.
    // E>=2 enters the general path. C=17 forces a one-lane final channel block.
    // E=3 exercises both a pair of positions and the unpaired position branch.
    const int cases[][2] = {{16, 1}, {16, 2}, {17, 1}, {17, 2}, {17, 3}, {31, 2}, {32, 2}, {33, 2}};
    int failed = 0;
    for (const auto& test : cases) {
        if (!checkCase(test[0], test[1])) {
            ++failed;
        }
    }
    std::printf("%s: %d of %zu cases failed\n", failed ? "FAIL" : "PASS", failed, sizeof(cases) / sizeof(cases[0]));
    return failed == 0;
}
} // namespace

#ifdef MNN_RVV_SUM_TAIL_TEST_MAIN
int main() {
    return runSumTailCases() ? 0 : 1;
}
#else
class RVVSumByAxisTailTest : public MNNTestCase {
    bool run(int) override {
        // The build flag does not guarantee runtime support for the vlenb CSR.
        if (!MNN::MNNGetCoreFunctions()->supportRVV) {
            std::printf("Skip RVV sum regression: RVV is not supported by this CPU.\n");
            return true;
        }
        return runSumTailCases();
    }
};
MNNTestSuiteRegister(RVVSumByAxisTailTest, "backend/cpu/rvv/sum_by_axis_tail");
#endif

#undef RVV_SUM_KERNEL
#endif
