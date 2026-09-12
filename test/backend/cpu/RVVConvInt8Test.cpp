// Copyright 2018 Alibaba Group Holding Limited. All rights reserved.
#if defined(MNN_USE_RVV)
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

void MNNConvInt8ComputeBiasFloat_RVV(float* dst, const int32_t* bias, const float* weightScale, float scaleRatio,
                                     size_t size);
void MNNConvInt8ComputeWeightKernelSum_RVV(int* kernelSum, int32_t* bias, const int8_t* weight, int kernelNum,
                                           int kernelSize, const float* scale, const float* weightBias,
                                           bool compensateSseOffset);

namespace {
// Scalar oracles follow the original scalar implementations in CPUConvolution.cpp.
void computeBiasFloatReference(float* dst, const int32_t* bias, const float* weightScale, float scaleRatio,
                               size_t size) {
    for (size_t i = 0; i < size; ++i) {
        dst[i] = static_cast<float>(bias[i]) * weightScale[i] * scaleRatio;
    }
}

void computeWeightKernelSumReference(int* kernelSum, int32_t* bias, const int8_t* weight, int kernelNum,
                                     int kernelSize, const float* scale, const float* weightBias,
                                     bool compensateSseOffset) {
    for (int i = 0; i < kernelNum; ++i) {
        int temp = 0;
        const int offset = i * kernelSize;
        for (int j = 0; j < kernelSize; ++j) {
            temp += static_cast<int>(weight[offset + j]);
        }
        kernelSum[i] = temp + kernelSize * (weightBias[i] / scale[i]);
        if (compensateSseOffset) {
            bias[i] -= 128 * temp;
        }
    }
}

bool runConvInt8Kernels() {
    size_t cases = 0;
    // ConvInt8ComputeBiasFloat: exact same op order as the scalar path.
    // ratio == 1.0f takes the branch that skips the second multiply; it must
    // stay bit-identical to multiplying by 1.0f.
    for (float ratio : {1.0f, 1.7f, 0.0f}) {
        for (size_t size : {size_t(1), size_t(7), size_t(63), size_t(256), size_t(1000)}) {
            std::vector<int32_t> bias(size);
            std::vector<float> scale(size), dst(size), ref(size);
            for (size_t i = 0; i < size; ++i) {
                bias[i] = static_cast<int32_t>(i * 100003 % 2000001 - 1000000);
                scale[i] = float(int(i % 17) - 8) / 64.0f + 0.01f;
            }
            RVV_TEST_KERNEL(MNNConvInt8ComputeBiasFloat)(dst.data(), bias.data(), scale.data(), ratio, size);
            computeBiasFloatReference(ref.data(), bias.data(), scale.data(), ratio, size);
            if (::memcmp(dst.data(), ref.data(), size * sizeof(float)) != 0) {
                std::printf("ConvInt8ComputeBiasFloat mismatch: ratio=%g size=%zu\n", ratio, size);
                return false;
            }
            cases++;
        }
    }
    // ConvInt8ComputeWeightKernelSum: integer arithmetic, exact match expected.
    // kernelSize spans the scalar fallback (below one vector) and multi-chunk
    // vector paths, plus a tail that is not a multiple of the vector length.
    for (int kernelNum : {1, 4, 9}) {
        for (int kernelSize : {1, 3, 27, 31, 32, 33, 129, 576}) {
            std::vector<int8_t> weight(static_cast<size_t>(kernelNum) * kernelSize);
            for (size_t i = 0; i < weight.size(); ++i) {
                weight[i] = static_cast<int8_t>((i * 89 + 23) % 256 - 128);
            }
            std::vector<float> scale(kernelNum), weightBias(kernelNum);
            for (int i = 0; i < kernelNum; ++i) {
                scale[i] = 0.25f + 0.5f * i / kernelNum;
                weightBias[i] = float((i % 7) - 3);
            }
            for (int sse = 0; sse <= 1; ++sse) {
                std::vector<int32_t> biasSrc(kernelNum), biasRef(kernelNum);
                std::vector<int> dst(kernelNum), ref(kernelNum);
                for (int i = 0; i < kernelNum; ++i) {
                    biasSrc[i] = biasRef[i] = 1234567 + i * 7919;
                }
                RVV_TEST_KERNEL(MNNConvInt8ComputeWeightKernelSum)(dst.data(), biasSrc.data(), weight.data(),
                                                                   kernelNum, kernelSize, scale.data(),
                                                                   weightBias.data(), sse != 0);
                computeWeightKernelSumReference(ref.data(), biasRef.data(), weight.data(), kernelNum, kernelSize,
                                                scale.data(), weightBias.data(), sse != 0);
                if (::memcmp(dst.data(), ref.data(), kernelNum * sizeof(int)) != 0 ||
                    ::memcmp(biasSrc.data(), biasRef.data(), kernelNum * sizeof(int32_t)) != 0) {
                    std::printf("ConvInt8ComputeWeightKernelSum mismatch: kernelNum=%d kernelSize=%d sse=%d\n",
                                kernelNum, kernelSize, sse);
                    return false;
                }
                cases++;
            }
        }
    }
    std::printf("RVV conv int8 kernels: %zu scalar comparisons passed\n", cases);
    return true;
}
} // namespace

#ifdef MNN_RVV_KERNEL_TEST_MAIN
int main() {
    return runConvInt8Kernels() ? 0 : 1;
}
#else
class RVVConvInt8Test : public MNNTestCase {
    bool run(int) override { return runConvInt8Kernels(); }
};
MNNTestSuiteRegister(RVVConvInt8Test, "backend/cpu/rvv/conv_int8");
#endif
#endif

#undef RVV_TEST_KERNEL
