#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/MNNForwardType.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "MNNTestSuite.h"

using namespace MNN::Express;

namespace {

enum class TestUnaryType {
    SIGMOID,
    TANH,
    SILU,
    GELU,
    LOG,
};

VARP applyUnary(VARP input, TestUnaryType type) {
    switch (type) {
        case TestUnaryType::SIGMOID:
            return _Sigmoid(input);
        case TestUnaryType::TANH:
            return _Tanh(input);
        case TestUnaryType::SILU:
            return _Silu(input);
        case TestUnaryType::GELU:
            return _Gelu(input);
        case TestUnaryType::LOG:
            return _Log(input);
    }
    return nullptr;
}

float referenceUnary(float x, TestUnaryType type) {
    switch (type) {
        case TestUnaryType::SIGMOID:
            return 1.0f / (1.0f + std::exp(-x));
        case TestUnaryType::TANH:
            return std::tanh(x);
        case TestUnaryType::SILU:
            return x / (1.0f + std::exp(-x));
        case TestUnaryType::GELU: {
            constexpr float kSqrtTwoOverPi = 0.7978845608028654f;
            const float inner = kSqrtTwoOverPi * (x + 0.044715f * x * x * x);
            return 0.5f * x * (1.0f + std::tanh(inner));
        }
        case TestUnaryType::LOG:
            return x > 0.0f ? std::log(x) : -65504.0f;
    }
    return x;
}

const char* unaryName(TestUnaryType type) {
    switch (type) {
        case TestUnaryType::SIGMOID:
            return "sigmoid";
        case TestUnaryType::TANH:
            return "tanh";
        case TestUnaryType::SILU:
            return "silu";
        case TestUnaryType::GELU:
            return "gelu";
        case TestUnaryType::LOG:
            return "log";
    }
    return "unknown";
}

} // namespace

class HexagonUnaryPWLTest : public MNNTestCase {
public:
    virtual bool run(int precision) override {
        if (MNNTestSuite::get()->pStaus.forwardType != MNN_FORWARD_HEXAGON) {
            MNN_PRINT("Skip HexagonUnaryPWLTest: Hexagon backend is not selected\n");
            return true;
        }

        constexpr int kSize = 8193;
        std::vector<float> inputData(kSize);
        for (int i = 0; i < kSize; ++i) {
            inputData[i] = -12.0f + 24.0f * static_cast<float>(i) / static_cast<float>(kSize - 1);
        }

        // Exercise both sides of every 0.25-wide segment boundary as well as
        // the saturation boundaries used by the PWL kernels.
        int cursor = 0;
        for (int edge = -32; edge <= 32; ++edge) {
            const float x = 0.25f * static_cast<float>(edge);
            inputData[cursor++] = x - 0.01f;
            inputData[cursor++] = x;
            inputData[cursor++] = x + 0.01f;
        }
        const float specialValues[] = {-100.0f, -12.0f, -8.0f, -4.0f, -0.0f, 0.0f, 4.0f, 8.0f, 12.0f, 100.0f};
        for (float value : specialValues) {
            inputData[cursor++] = value;
        }

        struct Case {
            TestUnaryType type;
            float maxAbsoluteError;
        };
        const Case cases[] = {
            {TestUnaryType::SIGMOID, 0.005f}, {TestUnaryType::TANH, 0.009f}, {TestUnaryType::SILU, 0.008f},
            {TestUnaryType::GELU, 0.009f},    {TestUnaryType::LOG, 0.02f},
        };

        for (const auto& testCase : cases) {
            auto input = _Input({kSize}, NCHW, halide_type_of<float>());
            std::copy(inputData.begin(), inputData.end(), input->writeMap<float>());
            auto output = applyUnary(input, testCase.type);
            const float* value = output->readMap<float>();
            if (value == nullptr) {
                MNN_ERROR("HexagonUnaryPWLTest: failed to read %s output\n", unaryName(testCase.type));
                return false;
            }

            float maxError = 0.0f;
            int maxIndex = 0;
            for (int i = 0; i < kSize; ++i) {
                const float expected = referenceUnary(inputData[i], testCase.type);
                const float error = std::fabs(value[i] - expected);
                if (error > maxError) {
                    maxError = error;
                    maxIndex = i;
                }
            }
            MNN_PRINT("Hexagon Unary %s max abs error: %.8f at x=%.8f\n", unaryName(testCase.type), maxError,
                      inputData[maxIndex]);
            if (maxError > testCase.maxAbsoluteError) {
                MNN_ERROR("HexagonUnaryPWLTest: %s error %.8f exceeds %.8f\n", unaryName(testCase.type), maxError,
                          testCase.maxAbsoluteError);
                const int begin = std::max(0, maxIndex - 3);
                const int end = std::min(kSize, maxIndex + 4);
                for (int i = begin; i < end; ++i) {
                    MNN_ERROR("  index=%d x=%.8f expected=%.8f actual=%.8f\n", i, inputData[i],
                              referenceUnary(inputData[i], testCase.type), value[i]);
                }
                return false;
            }
        }
        return true;
    }
};

MNNTestSuiteRegister(HexagonUnaryPWLTest, "op/hexagon/unary-pwl");
