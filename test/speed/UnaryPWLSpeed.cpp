#include <MNN/AutoTime.hpp>
#include <MNN/MNNForwardType.h>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>

#include <cmath>

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

class UnaryPWLSpeed : public MNNTestCase {
public:
    virtual bool run(int precision) override {
        if (MNNTestSuite::get()->pStaus.forwardType != MNN_FORWARD_HEXAGON) {
            MNN_PRINT("Skip UnaryPWLSpeed: Hexagon backend is not selected\n");
            return true;
        }

        constexpr int kElementCount = 262144;
        constexpr int kChainDepth = 8;
        constexpr int kWarmup = 10;
        constexpr int kIterations = 50;
        const TestUnaryType cases[] = {TestUnaryType::SIGMOID, TestUnaryType::TANH, TestUnaryType::SILU,
                                       TestUnaryType::GELU, TestUnaryType::LOG};

        for (TestUnaryType type : cases) {
            auto input = _Input({kElementCount}, NCHW, halide_type_of<float>());
            float* inputPtr = input->writeMap<float>();
            for (int i = 0; i < kElementCount; ++i) {
                const float x = -8.0f + 16.0f * static_cast<float>(i % 4096) / 4095.0f;
                inputPtr[i] = type == TestUnaryType::LOG ? std::exp(x) : x;
            }

            VARP output = input;
            for (int i = 0; i < kChainDepth; ++i) {
                output = applyUnary(output, type);
            }

            for (int i = 0; i < kWarmup; ++i) {
                input->writeMap<float>()[0] = static_cast<float>(i) * 0.001f;
                output->readMap<float>();
            }

            MNN::Timer timer;
            for (int i = 0; i < kIterations; ++i) {
                input->writeMap<float>()[0] = static_cast<float>(i) * 0.001f;
                output->readMap<float>();
            }
            const float averageMs = timer.durationInUs() / 1000.0f / static_cast<float>(kIterations);
            const float nsPerElement = averageMs * 1.0e6f / static_cast<float>(kElementCount * kChainDepth);
            MNN_PRINT("Hexagon Unary speed %-7s: %.4f ms, %.4f ns/element/op (%d elements x %d ops)\n", unaryName(type),
                      averageMs, nsPerElement, kElementCount, kChainDepth);
        }
        return true;
    }
};

MNNTestSuiteRegister(UnaryPWLSpeed, "speed/UnaryPWL");
