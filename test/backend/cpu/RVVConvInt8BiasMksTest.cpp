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
// CPUConvolution.cpp writes it. The kernel is expected to match it bit for bit
// on finite values, so the comparison is a memcmp rather than a tolerance.
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

bool runConvInt8Kernels() {
    // (inputScale, outputScale) pairs: values that are not exactly representable,
    // one pair with inputScale == outputScale, and a zero on either side (the
    // asymmetric path, which applies no scaling at all).
    const float inputScales[] = {1.0f, 1.7f, 1.7f, 3.0f, 0.375f, 0.0f, 2.0f};
    const float outputScales[] = {1.0f, 3.0f, 1.7f, 1.7f, 1.0f, 3.0f, 0.0f};
    size_t cases = 0;
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
            if (::memcmp(dst.data(), ref.data(), size * sizeof(float)) != 0) {
                for (size_t i = 0; i < size; ++i) {
                    uint32_t a = 0, b = 0;
                    ::memcpy(&a, dst.data() + i, sizeof(uint32_t));
                    ::memcpy(&b, ref.data() + i, sizeof(uint32_t));
                    if (a != b) {
                        std::printf("ConvInt8ComputeBiasFloat mismatch: inputScale=%g outputScale=%g size=%zu "
                                    "index=%zu got=%.9g (0x%08x) ref=%.9g (0x%08x)\n",
                                    inputScale, outputScale, size, i, dst[i], a, ref[i], b);
                        break;
                    }
                }
                return false;
            }
            cases++;
        }
    }
    std::printf("RVV conv int8 bias-float: %zu cases, all bit-exact against the scalar oracle\n", cases);
    return true;
}
} // namespace

#ifdef MNN_RVV_KERNEL_TEST_MAIN
int main() {
    return runConvInt8Kernels() ? 0 : 1;
}
#else
class RVVConvInt8BiasMksTest : public MNNTestCase {
    bool run(int) override { return runConvInt8Kernels(); }
};
MNNTestSuiteRegister(RVVConvInt8BiasMksTest, "backend/cpu/rvv/conv_int8_bias_mks");
#endif
#endif

#undef RVV_TEST_KERNEL
