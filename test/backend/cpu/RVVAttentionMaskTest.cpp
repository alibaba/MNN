// Copyright 2018 Alibaba Group Holding Limited. All rights reserved.
#if defined(MNN_USE_RVV)
#include <cstdio>
#include <cstring>
#include <vector>
#ifndef MNN_RVV_KERNEL_TEST_MAIN
#include "MNNTestSuite.h"
#include "backend/cpu/compute/CommonOptFunction.h"
#define RVV_TEST_KERNEL(name) MNN::MNNGetCoreFunctions()->name
#else
#define RVV_TEST_KERNEL(name) name##_RVV
#endif

void MNNAttentionMaskQK_RVV(float*, const float*, size_t, size_t, int, int, int, int, const float*, const float*,
                            size_t, bool, bool);

namespace {
// Scalar FP32 oracle follows CPUAttention::_maskQK, including scale-before-bound-check on the final lane.
void maskReference(float* data, float scale, size_t seq, size_t processed, int pack, int kv, int offset, int pad,
                   const float* mask, size_t maskSize, bool scaleApplied, bool triangular) {
    if (triangular && scaleApplied)
        return;
    const size_t blocks = (processed + pack - 1) / pack;
    if (triangular) {
        for (size_t i = 0; i < blocks * pack * seq; ++i)
            data[i] *= scale;
        return;
    }
    if (mask == nullptr)
        return;
    const bool full = maskSize == (seq + pad) * (kv + pad);
    const int gap = full ? 0 : int(kv - seq);
    const size_t cols = full ? kv + pad : seq + pad;
    for (size_t i = 0; i < blocks; ++i) {
        for (size_t j = 0; j < seq; ++j) {
            for (int k = 0; k < pack; ++k) {
                float& value = data[(i * seq + j) * pack + k];
                if (!scaleApplied)
                    value *= scale;
                const int col = offset + int(i) * pack + k;
                if (col < gap)
                    continue;
                if (size_t(col - gap) >= cols)
                    break;
                value += mask[j * cols + col - gap];
            }
        }
    }
}

bool runMaskKernels() {
    size_t cases = 0;
    for (size_t seq : {size_t(1), size_t(3), size_t(16), size_t(33)}) {
        for (int history : {0, 1, 7, 32}) {
            const int kv = int(seq) + history;
            for (int pad : {0, 1, 3}) {
                for (int pack : {4, 8, 16}) {
                    for (int offset = 0; offset < kv; offset += 7) {
                        for (size_t processed : {size_t(1), size_t(3), size_t(7), size_t(kv - offset)}) {
                            if (processed > size_t(kv - offset))
                                continue;
                            const size_t length = (processed + pack - 1) / pack * pack * seq;
                            for (int full = 0; full <= 1; ++full) {
                                const size_t cols = full ? kv + pad : seq + pad;
                                std::vector<float> mask((seq + pad) * cols);
                                for (size_t i = 0; i < mask.size(); ++i)
                                    mask[i] = i % 5 == 0 ? -__builtin_inff() : float(int(i % 11) - 5) / 16.0f;
                                for (int hasMask = 0; hasMask <= 1; ++hasMask) {
                                    for (int scaled = 0; scaled <= 1; ++scaled) {
                                        for (int triangular = 0; triangular <= 1; ++triangular) {
                                            for (float scale : {0.125f, 0.73f}) {
                                                std::vector<float> actual(length + 2, 1234.0f);
                                                for (size_t i = 0; i < length; ++i)
                                                    actual[i + 1] = float(int(i * 13 % 31) - 15) / 16.0f;
                                                auto expected = actual;
                                                const float* maskPtr = hasMask ? mask.data() : nullptr;
                                                const float sink = 0.25f;
                                                maskReference(expected.data() + 1, scale, seq, processed, pack, kv,
                                                              offset, pad, maskPtr, mask.size(), scaled, triangular);
                                                RVV_TEST_KERNEL(MNNAttentionMaskQK)(
                                                    actual.data() + 1, &scale, seq, processed, pack, kv, offset, pad,
                                                    &sink, maskPtr, mask.size(), scaled, triangular);
                                                if (std::memcmp(actual.data(), expected.data(),
                                                                actual.size() * sizeof(float))) {
                                                    std::printf(
                                                        "mask mismatch seq=%zu kv=%d offset=%d processed=%zu pack=%d "
                                                        "pad=%d full=%d mask=%d scaled=%d triangular=%d\n",
                                                        seq, kv, offset, processed, pack, pad, full, hasMask, scaled,
                                                        triangular);
                                                    return false;
                                                }
                                                ++cases;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    std::printf("RVV attention mask: %zu scalar comparisons passed\n", cases);
    return true;
}
} // namespace

#ifdef MNN_RVV_KERNEL_TEST_MAIN
int main() {
    return runMaskKernels() ? 0 : 1;
}
#else
class RVVAttentionMaskTest : public MNNTestCase {
    bool run(int) override { return runMaskKernels(); }
};
MNNTestSuiteRegister(RVVAttentionMaskTest, "backend/cpu/rvv/attention_mask");
#endif
#endif

#undef RVV_TEST_KERNEL
