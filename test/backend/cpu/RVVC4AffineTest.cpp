// Copyright © 2026, Alibaba Group Holding Limited
#if defined(MNN_BUILD_STATIC_LIBS)
#include "MNNTestSuite.h"
#include "backend/cpu/compute/CommonOptFunction.h"
#include <cstring>
#include <vector>
#endif

bool MNNTestC4AffineFunctions() {
#if defined(MNN_BUILD_STATIC_LIBS)
    auto core = MNN::MNNGetCoreFunctions();
    if (core == nullptr) {
        MNN_ERROR("C4 affine test requires an initialized CPU backend\n");
        return false;
    }
    if (core->pack != 4 || core->bytes != 4) {
        return true;
    }
#if defined(__riscv)
    const bool expectRVV = MNN_TEST_RVV_ENABLED && core->supportRVV;
    MNN_PRINT("C4 affine RVV capability: %d, dispatch expected: %d\n", static_cast<int>(core->supportRVV),
              static_cast<int>(expectRVV));
    // A direct kernel test alone cannot catch an unregistered C++ overload.
    if (expectRVV != (core->MNNScaleAndAddBias != MNNScaleAndAddBias) ||
        expectRVV != (core->MNNReluWithSlopeChannel != MNNReluWithSlopeChannel)) {
        MNN_ERROR("Unexpected C4 affine function registration\n");
        return false;
    }
#endif
    const size_t areas[] = {0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 257};
    const size_t depths[] = {0, 1, 2, 3, 8, 17};
    size_t cases = 0;
    for (size_t area : areas) {
        for (size_t depth : depths) {
            const size_t count = area * depth * 4;
            std::vector<float> input(count + 8, 12345.0f), alpha(depth * 4), bias(depth * 4);
            for (size_t i = 0; i < count; ++i) {
                input[i + 4] = (static_cast<int>(i % 37) - 18) * 0.125f;
            }
            for (size_t i = 0; i < depth * 4; ++i) {
                alpha[i] = (static_cast<int>(i % 9) - 4) * 0.25f;
                bias[i] = (static_cast<int>(i % 7) - 3) * 0.125f;
            }
            for (bool inplace : {false, true}) {
                for (bool scale : {false, true}) {
                    auto output = input;
                    const float* src = inplace ? output.data() + 4 : input.data() + 4;
                    if (scale) {
                        core->MNNScaleAndAddBias(output.data() + 4, src, bias.data(), alpha.data(), area, depth);
                    } else {
                        core->MNNReluWithSlopeChannel(output.data() + 4, src, alpha.data(), area, depth);
                    }
                    for (size_t i = 0; i < count + 8; ++i) {
                        float expected = input[i];
                        if (i >= 4 && i < count + 4) {
                            size_t index = i - 4;
                            size_t channel = (index / (area * 4)) * 4 + index % 4;
                            expected = scale ? expected * alpha[channel] + bias[channel]
                                             : (expected < 0 ? expected * alpha[channel] : expected);
                        }
                        if (std::memcmp(&output[i], &expected, sizeof(float)) != 0) {
                            MNN_ERROR("C4 affine mismatch: area=%zu depth=%zu inplace=%d scale=%d index=%zu\n", area,
                                      depth, inplace, scale, i);
                            return false;
                        }
                    }
                    ++cases;
                }
            }
        }
    }
    MNN_PRINT("C4 affine dispatch: %zu cases passed\n", cases);
#endif
    return true;
}
