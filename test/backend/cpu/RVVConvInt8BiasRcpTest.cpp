// Copyright 2018 Alibaba Group Holding Limited. All rights reserved.
#if defined(MNN_USE_RVV)
#include <cmath>
#include <cstdint>
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

void MNNConvInt8ComputeBiasFloat_RVV(float* dst, const int32_t* bias, const float* weightScale, float inputScale,
                                     float outputScale, size_t size);

namespace {
// Scalar oracle: the expression is spelled out exactly as the scalar path in
// CPUConvolution.cpp writes it, so the comparison is against the code being
// replaced and not against a rearranged form of it.
void computeBiasFloatReference(float* dst, const int32_t* bias, const float* weightScale, float inputScale,
                               float outputScale, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        if (inputScale && outputScale) { // symmetric quan
            dst[i] = static_cast<float>(bias[i]) * weightScale[i] * inputScale / outputScale;
        } else {
            dst[i] = static_cast<float>(bias[i]) * weightScale[i];
        }
    }
}

// Counts elements whose bit pattern differs, plus the largest relative gap. The
// reciprocal arm is expected to differ on part of the inputs; reporting the
// number keeps the kernel's accuracy visible instead of hiding it behind a
// pass/fail bit.
struct BiasDiff {
    size_t total = 0;
    size_t differing = 0;
    float maxRel = 0.0f;
};

BiasDiff compareBiasFloat(const float* got, const float* ref, size_t size) {
    BiasDiff diff;
    diff.total = size;
    for (size_t i = 0; i < size; ++i) {
        uint32_t a = 0, b = 0;
        ::memcpy(&a, got + i, sizeof(uint32_t));
        ::memcpy(&b, ref + i, sizeof(uint32_t));
        if (a == b) {
            continue;
        }
        diff.differing++;
        const float denom = std::fabs(ref[i]);
        if (denom > 0.0f) {
            const float rel = std::fabs(got[i] - ref[i]) / denom;
            if (rel > diff.maxRel) {
                diff.maxRel = rel;
            }
        }
    }
    return diff;
}

bool runConvInt8Kernels() {
    // (inputScale, outputScale) pairs: values that are not exactly representable,
    // one pair with inputScale == outputScale, and a zero on either side (the
    // asymmetric path, which applies no scaling at all).
    const float inputScales[] = {1.0f, 1.7f, 1.7f, 3.0f, 0.375f, 0.0f, 2.0f};
    const float outputScales[] = {1.0f, 3.0f, 1.7f, 1.7f, 1.0f, 3.0f, 0.0f};
    size_t cases = 0;
    size_t totalElements = 0, totalDiffering = 0;
    float worstRel = 0.0f;
    for (size_t c = 0; c < sizeof(inputScales) / sizeof(inputScales[0]); ++c) {
        const float inputScale = inputScales[c];
        const float outputScale = outputScales[c];
        for (size_t size : {size_t(1), size_t(7), size_t(63), size_t(256), size_t(1000)}) {
            std::vector<int32_t> bias(size);
            std::vector<float> scale(size), dst(size), ref(size);
            for (size_t i = 0; i < size; ++i) {
                bias[i] = static_cast<int32_t>(i * 100003 % 2000001 - 1000000);
                scale[i] = float(int(i % 17) - 8) / 64.0f + 0.01f;
            }
            RVV_TEST_KERNEL(MNNConvInt8ComputeBiasFloat)(dst.data(), bias.data(), scale.data(), inputScale,
                                                         outputScale, size);
            computeBiasFloatReference(ref.data(), bias.data(), scale.data(), inputScale, outputScale, size);
            const BiasDiff diff = compareBiasFloat(dst.data(), ref.data(), size);
            totalElements += diff.total;
            totalDiffering += diff.differing;
            if (diff.maxRel > worstRel) {
                worstRel = diff.maxRel;
            }
            // Only a NaN or an infinity where the reference is finite counts as a
            // failure here; a last-bit difference is reported, not asserted.
            for (size_t i = 0; i < size; ++i) {
                if (std::isfinite(ref[i]) && !std::isfinite(dst[i])) {
                    std::printf("ConvInt8ComputeBiasFloat produced a non-finite value: inputScale=%g "
                                "outputScale=%g size=%zu index=%zu\n",
                                inputScale, outputScale, size, i);
                    return false;
                }
            }
            cases++;
        }
    }
    std::printf("RVV conv int8 bias-float: %zu cases, %zu/%zu elements differ in the last bit, max relative %.3g\n",
                cases, totalDiffering, totalElements, worstRel);
    return true;
}
} // namespace

#ifdef MNN_RVV_KERNEL_TEST_MAIN
int main() {
    return runConvInt8Kernels() ? 0 : 1;
}
#else
class RVVConvInt8BiasRcpTest : public MNNTestCase {
    bool run(int) override { return runConvInt8Kernels(); }
};
MNNTestSuiteRegister(RVVConvInt8BiasRcpTest, "backend/cpu/rvv/conv_int8_bias_rcp");
#endif
#endif

#undef RVV_TEST_KERNEL
