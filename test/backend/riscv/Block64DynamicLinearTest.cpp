//
//  Block64DynamicLinearTest.cpp
//  MNN
//
//  Created by MNN on 2026/11/19.
//  Copyright © 2026, Alibaba Group Holding Limited
//

#include <cmath>
#include <MNN/expr/ExprCreator.hpp>
#include "CommonOpCreator.hpp"
#include "MNNTestSuite.h"

using namespace MNN;
using namespace MNN::Express;

class Block64DynamicLinearTest : public MNNTestCase {
    static bool testDynamicLinear(int ic, int oc, const std::vector<int>& lengths, float largeInput = 0.0f) {
        const int blockSize = 64, blockCount = ic / blockSize;
        std::vector<float> weight(ic * oc), alpha(2 * oc * blockCount), bias(oc);
        for (int o = 0; o < oc; ++o) {
            bias[o] = (o - 7) / 64.0f;
            for (int b = 0; b < blockCount; ++b) {
                const float scale = (1 + (o + b) % 4) / 1024.0f;
                const float minimum = -5.0f * scale;
                alpha[2 * (o * blockCount + b)] = minimum;
                alpha[2 * (o * blockCount + b) + 1] = scale;
                for (int u = 0; u < blockSize; ++u) {
                    const int code = (o * 7 + o / 8 + b * 11 + u * 3 + u / 8) % 16;
                    // Exact dequantized W4 values; all 16 codes occur in every block.
                    weight[o * ic + b * blockSize + u] = minimum + code * scale;
                }
            }
        }
        for (bool relu6 : {false, true}) {
            auto input = _Input({1, ic, 1, 1}, NCHW, halide_type_of<float>());
            auto output = _Convert(_HybridConv(weight, bias, alpha, _Convert(input, NC4HW4), {ic, oc},
                                               {1, 1}, PaddingMode::CAFFE, {1, 1}, {1, 1}, 1, {0, 0},
                                               false, relu6, 4, true), NCHW);
            for (size_t step = 0; step < lengths.size(); ++step) {
                const int rows = lengths[step];
                const bool zeroInput = step + 1 == lengths.size();
                MNNTEST_ASSERT(input->resize({rows, ic, 1, 1}));
                // Retain an independent copy: execution may recycle the input buffer.
                std::vector<float> source(rows * ic), expected(rows * oc);
                for (int r = 0; r < rows; ++r) {
                    for (int c = 0; c < ic; ++c) {
                        float value = ((c * 13 + r * 7 + static_cast<int>(step) * 3) % 257 - 128) / 128.0f;
                        if (r % 5 == 1) {
                            value = 1.0f;
                        } else if (r % 5 == 2) {
                            value = -1.0f;
                        } else if (r % 5 == 3 || zeroInput) {
                            value = 0.0f;
                        }
                        if (largeInput > 0.0f) {
                            // A single nonzero activation is exactly representable by dynamic quantization.
                            value = c == 0 ? (r % 2 ? -largeInput : largeInput) : 0.f;
                        }
                        source[r * ic + c] = value;
                    }
                    for (int o = 0; o < oc; ++o) {
                        double sum = bias[o];
                        for (int c = 0; c < ic; ++c) {
                            sum += double(source[r * ic + c]) * weight[o * ic + c];
                        }
                        expected[r * oc + o] = relu6 ? fminf(6.0f, fmaxf(0.0f, sum)) : float(sum);
                    }
                }
                auto inputPtr = input->writeMap<float>();
                MNNTEST_ASSERT(inputPtr != nullptr);
                ::memcpy(inputPtr, source.data(), source.size() * sizeof(float));
                auto actual = output->readMap<float>();
                auto info = output->getInfo();
                MNNTEST_ASSERT(actual != nullptr && info != nullptr);
                MNNTEST_ASSERT(info->dim == INTS({rows, oc, 1, 1}) && info->size == rows * oc);
                float maxError = 0.0f;
                for (int i = 0; i < info->size; ++i) {
                    // Account for dynamic activation quantization and reduced-precision GEMM.
                    maxError = fmaxf(maxError, fabsf(actual[i] - expected[i]));
                    // In the overflow case the integer fallback also rounds its epilogue.
                    const float tolerance = largeInput > 0.0f ? 0.01f + 0.002f * fabsf(expected[i]) :
                                            zeroInput ? 1e-6f : 0.03f + 0.02f * fabsf(expected[i]);
                    if (!std::isfinite(actual[i]) || fabsf(actual[i] - expected[i]) > tolerance) {
                        MNN_ERROR("Dynamic linear: M=%d K=%d N=%d relu6=%d index=%d ref=%f actual=%f\n",
                                  rows, ic, oc, int(relu6), i, expected[i], actual[i]);
                        return false;
                    }
                }
                MNN_PRINT("block64 M=%d K=%d N=%d relu6=%d largeInput=%g max_abs=%g\n",
                          rows, ic, oc, int(relu6), largeInput, maxError);
            }
        }
        return true;
    }

public:
    bool run(int precision) override {
        MNNTEST_ASSERT(testDynamicLinear(2048, 16, {1, 8, 9, 345, 346, 347, 9, 1}));
        MNNTEST_ASSERT(testDynamicLinear(512, 512, {128, 1, 9, 17, 129, 1}));
        MNNTEST_ASSERT(testDynamicLinear(2048, 256, {129, 1}));
        float largeInput = 32512.0f;
#ifdef MNN_USE_RVV
        // RVV stores FP32 activations even in Low mode. Exercise the IME2 narrowing fallback.
        if (MNNTestSuite::get()->pStaus.forwardType == MNN_FORWARD_CPU) {
            largeInput = 130048.0f;
        }
#endif
        MNNTEST_ASSERT(testDynamicLinear(512, 512, {64}, largeInput));
        return true;
    }
};
MNNTestSuiteRegister(Block64DynamicLinearTest, "op/lowMemory/DenseConv/dynamic_block64");
