// Copyright 2018 Alibaba Group Holding Limited. All rights reserved.
#if defined(MNN_USE_RVV)
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>
#ifndef MNN_RVV_KERNEL_TEST_MAIN
#include "MNNTestSuite.h"
#include "backend/cpu/compute/CommonOptFunction.h"
#define RVV_TEST_KERNEL(name) MNN::MNNGetCoreFunctions()->name
#else
#define RVV_TEST_KERNEL(name) name##_RVV
#endif

void MNNRankOneUpdate_RVV(float*, const float*, const float*, size_t, size_t);
void MNNDualMatVec_RVV(const float*, const float*, const float*, float*, float*, size_t, size_t);
void MNNDecayRankOneUpdate_RVV(float*, const float*, const float*, float, size_t, size_t);
void MNNFusedGatedDelta_RVV(float*, const float*, const float*, const float*, float*, float, float, float, size_t,
                            size_t);

namespace {
bool equalValues(const std::vector<float>& a, const std::vector<float>& b, const char* label, size_t dk, size_t dv) {
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) {
            std::printf("%s dk=%zu dv=%zu index=%zu actual=%.9g reference=%.9g\n", label, dk, dv, i, a[i], b[i]);
            return false;
        }
    }
    return true;
}

bool runLinearAttentionKernels() {
    size_t cases = 0;
    for (size_t dk : {size_t(0), size_t(1), size_t(3), size_t(16), size_t(33), size_t(64), size_t(128)}) {
        for (size_t dv : {size_t(0), size_t(1), size_t(3), size_t(4), size_t(7), size_t(16), size_t(31), size_t(32),
                          size_t(33), size_t(63), size_t(64), size_t(65), size_t(127), size_t(128), size_t(129),
                          size_t(255), size_t(256), size_t(257)}) {
            // Keep guards on both ends, including zero-length calls and non-vector-aligned pointers.
            std::vector<float> state(dk * dv + 2, 1234.0f), k(dk), q(dk), v(dv), delta(dv);
            for (size_t i = 0; i < dk * dv; ++i)
                state[i + 1] = float(int(i * 17 % 37) - 18) / 64.0f;
            for (size_t i = 0; i < dk; ++i) {
                k[i] = float(int(i * 13 % 31) - 15) / 32.0f;
                q[i] = float(int(i * 7 % 29) - 14) / 32.0f;
            }
            for (size_t j = 0; j < dv; ++j) {
                v[j] = float(int(j * 11 % 23) - 11) / 16.0f;
                delta[j] = float(int(j * 5 % 19) - 9) / 16.0f;
            }
            auto expected = state;
            auto actual = state;
            for (size_t i = 0; i < dk; ++i)
                for (size_t j = 0; j < dv; ++j)
                    expected[1 + i * dv + j] = std::fma(k[i], delta[j], expected[1 + i * dv + j]);
            RVV_TEST_KERNEL(MNNRankOneUpdate)(actual.data() + 1, k.data(), delta.data(), dk, dv);
            if (!equalValues(actual, expected, "rank-one", dk, dv))
                return false;

            std::vector<float> outK(dv + 2, 1234.0f), outQ(outK), refK(outK), refQ(outK);
            for (size_t j = 0; j < dv; ++j)
                refK[j + 1] = refQ[j + 1] = 0.0f;
            for (size_t i = 0; i < dk; ++i) {
                for (size_t j = 0; j < dv; ++j) {
                    refK[j + 1] = std::fma(state[1 + i * dv + j], k[i], refK[j + 1]);
                    refQ[j + 1] = std::fma(state[1 + i * dv + j], q[i], refQ[j + 1]);
                }
            }
            RVV_TEST_KERNEL(MNNDualMatVec)(state.data() + 1, k.data(), q.data(), outK.data() + 1, outQ.data() + 1, dk,
                                           dv);
            if (!equalValues(outK, refK, "dual-k", dk, dv) || !equalValues(outQ, refQ, "dual-q", dk, dv))
                return false;
            for (float decay : {0.0f, 0.73f, 1.0f}) {
                expected = state;
                actual = state;
                for (size_t i = 0; i < dk; ++i)
                    for (size_t j = 0; j < dv; ++j)
                        expected[1 + i * dv + j] = std::fma(decay, state[1 + i * dv + j], k[i] * delta[j]);
                RVV_TEST_KERNEL(MNNDecayRankOneUpdate)(actual.data() + 1, k.data(), delta.data(), decay, dk, dv);
                if (!equalValues(actual, expected, "decay-rank-one", dk, dv))
                    return false;

                for (int normalized = 0; normalized < 2; ++normalized) {
                    for (float beta : {0.0f, 0.37f, 1.0f}) {
                        expected = state;
                        actual = state;
                        // Exercise normalized inference keys and the original unnormalized stress case.
                        auto recurrentK = k;
                        auto recurrentQ = q;
                        const float normalization =
                            normalized ? 1.0f / std::sqrt(float(std::max(size_t(1), dk))) : 1.0f;
                        for (size_t i = 0; i < dk; ++i) {
                            recurrentK[i] *= normalization;
                            recurrentQ[i] *= normalization;
                        }
                        std::vector<float> out(dv + 2, 1234.0f), refOut(out);
                        const float kq = -0.21f;
                        // Repeated decode steps exercise accumulated recurrent state error.
                        for (size_t step = 0; step < 8; ++step) {
                            for (size_t j = 0; j < dv; ++j) {
                                float sk = 0.0f, sq = 0.0f;
                                for (size_t i = 0; i < dk; ++i) {
                                    sk = std::fma(expected[1 + i * dv + j], recurrentK[i], sk);
                                    sq = std::fma(expected[1 + i * dv + j], recurrentQ[i], sq);
                                }
                                const float correction = beta * std::fma(-decay, sk, v[j]);
                                refOut[j + 1] = std::fma(decay, sq, kq * correction);
                                for (size_t i = 0; i < dk; ++i)
                                    expected[1 + i * dv + j] =
                                        std::fma(decay, expected[1 + i * dv + j], recurrentK[i] * correction);
                            }
                            RVV_TEST_KERNEL(MNNFusedGatedDelta)(actual.data() + 1, recurrentK.data(), recurrentQ.data(),
                                                                v.data(), out.data() + 1, decay, beta, kq, dk, dv);
                            if (!equalValues(out, refOut, "gated-output", dk, dv) ||
                                !equalValues(actual, expected, "gated-state", dk, dv))
                                return false;
                            ++cases;
                        }
                    }
                }
        }
    }
    }
    std::printf("RVV linear attention: %zu recurrent cases passed (plus rank-one, dual and decay checks)\n", cases);
    return true;
}
} // namespace

#ifdef MNN_RVV_KERNEL_TEST_MAIN
int main() {
    return runLinearAttentionKernels() ? 0 : 1;
}
#else
class RVVLinearAttentionTest : public MNNTestCase {
    bool run(int) override { return runLinearAttentionKernels(); }
};
MNNTestSuiteRegister(RVVLinearAttentionTest, "backend/cpu/rvv/linear_attention");
#endif
#endif

#undef RVV_TEST_KERNEL
