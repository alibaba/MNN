//
//  DynamicQuantConstantTest.cpp
//  MNNTests
//
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <algorithm>
#include <cmath>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include "MNN_generated.h"

using namespace MNN;
using namespace MNN::Express;

// Nonzero one-sided ranges must include zero before computing quantization parameters.
// Use 16 elements to avoid testing SIMD tail handling at the same time.
static bool checkDynamicQuantEndpoint(float value, bool constant) {
    auto input = _Input({1, 16}, NCHW, halide_type_of<float>());
    auto data = input->writeMap<float>();
    for (int i = 0; i < 16; ++i) {
        data[i] = constant || i % 2 == 0 ? value : 0.0f;
    }
    input->unMap();

    std::unique_ptr<OpT> op(new OpT);
    op->type = OpType_DynamicQuant;
    op->main.type = OpParameter_NONE;
    auto expr = Expr::create(std::move(op), {input}, 3);
    auto quantized = Variable::create(expr, 0);
    auto scale = Variable::create(expr, 1);
    auto zero = Variable::create(expr, 2);
    // The backend's int8 storage is biased by +128 on x86. Use its own
    // int8-to-float conversion to observe logical signed quantized values.
    auto logicalQuantized = _Int8ToFloat(quantized, _Const(1.0f));
    auto restored = (logicalQuantized - zero) * scale;

    const auto scalePtr = scale->readMap<float>();
    const auto zeroPtr = zero->readMap<float>();
    const auto quantPtr = logicalQuantized->readMap<float>();
    const auto restoredPtr = restored->readMap<float>();
    if (!scalePtr || !zeroPtr || !quantPtr || !restoredPtr) {
        MNN_ERROR("DynamicQuant endpoint: null output\n");
        return false;
    }
    const float expectedScale = 2.0f / 255.0f;
    const float expectedZero = value > 0.0f ? -128.0f : 127.0f;
    MNN_PRINT(
        "DynamicQuant value=%g constant=%d shape=[1,16] "
        "scale=%g expectedScale=%g zero=%g expectedZero=%g "
        "logicalQ0=%g restored0=%g expectedRestored0=%g\n",
        value, constant, scalePtr[0], expectedScale, zeroPtr[0], expectedZero, quantPtr[0], restoredPtr[0], value);
    bool pass = std::isfinite(scalePtr[0]) && std::fabs(scalePtr[0] - expectedScale) < 1e-8f &&
                std::isfinite(zeroPtr[0]) && zeroPtr[0] == expectedZero;
    for (int i = 0; i < 16; ++i) {
        const float expected = constant || i % 2 == 0 ? value : 0.0f;
        pass = pass && std::isfinite(restoredPtr[i]) && std::fabs(restoredPtr[i] - expected) < 1e-6f;
    }
    return pass;
}

class DynamicQuantPositiveConstantTest : public MNNTestCase {
    bool run(int precision) override { return checkDynamicQuantEndpoint(2.0f, true); }
};
class DynamicQuantNegativeConstantTest : public MNNTestCase {
    bool run(int precision) override { return checkDynamicQuantEndpoint(-2.0f, true); }
};
class DynamicQuantPositiveControlTest : public MNNTestCase {
    bool run(int precision) override { return checkDynamicQuantEndpoint(2.0f, false); }
};
class DynamicQuantNegativeControlTest : public MNNTestCase {
    bool run(int precision) override { return checkDynamicQuantEndpoint(-2.0f, false); }
};

MNNTestSuiteRegister(DynamicQuantPositiveConstantTest, "op/dynamic_quant_constant/positive");
MNNTestSuiteRegister(DynamicQuantNegativeConstantTest, "op/dynamic_quant_constant/negative");
MNNTestSuiteRegister(DynamicQuantPositiveControlTest, "op/dynamic_quant_control/positive");
MNNTestSuiteRegister(DynamicQuantNegativeControlTest, "op/dynamic_quant_control/negative");

class DynamicQuantRangeAndTailTest : public MNNTestCase {
    bool run(int precision) override {
        const int lengths[] = {1, 3, 4, 7, 8, 9, 15, 16, 17};
        const std::vector<std::vector<float>> patterns = {
            {2.0f}, {-2.0f}, {1.0f, 2.0f}, {-2.0f, -1.0f}, {-3.0f, 5.0f, 0.0f, 1.0f, -2.0f, 2.0f}};
        bool pass = true;
        int total = 0, failed = 0;
        for (int length : lengths) {
            for (int p = 0; p < patterns.size(); ++p) {
                auto input = _Input({1, length}, NCHW, halide_type_of<float>());
                auto data = input->writeMap<float>();
                std::vector<float> expected(length);
                float minimum = 0.0f, maximum = 0.0f;
                for (int i = 0; i < length; ++i) {
                    data[i] = patterns[p][i % patterns[p].size()];
                    expected[i] = data[i];
                    minimum = std::min(minimum, data[i]);
                    maximum = std::max(maximum, data[i]);
                }
                input->unMap();
                std::unique_ptr<OpT> op(new OpT);
                op->type = OpType_DynamicQuant;
                op->main.type = OpParameter_NONE;
                auto expr = Expr::create(std::move(op), {input}, 3);
                auto quantized = Variable::create(expr, 0);
                auto scale = Variable::create(expr, 1);
                auto zero = Variable::create(expr, 2);
                auto restored = (_Int8ToFloat(quantized, _Const(1.0f)) - zero) * scale;
                const auto scalePtr = scale->readMap<float>();
                const auto zeroPtr = zero->readMap<float>();
                const auto restoredPtr = restored->readMap<float>();
                const float expectedScale = (maximum - minimum) / 255.0f;
                const float expectedZero = std::round(-minimum / expectedScale) - 128.0f;
                bool ok = scalePtr && zeroPtr && restoredPtr;
                if (ok) {
                    ok = std::isfinite(scalePtr[0]) && std::fabs(scalePtr[0] - expectedScale) < 1e-8f &&
                         std::isfinite(zeroPtr[0]) && zeroPtr[0] == expectedZero;
                    for (int i = 0; i < length; ++i) {
                        ok = ok && std::isfinite(restoredPtr[i]) &&
                             std::fabs(restoredPtr[i] - expected[i]) <= 0.5001f * expectedScale + 1e-6f;
                    }
                }
                ++total;
                if (!ok) {
                    ++failed;
                    MNN_ERROR(
                        "DynamicQuant range/tail failed: length=%d pattern=%d scale=%g expected=%g "
                        "zero=%g expected=%g\n",
                        length, p, scalePtr ? scalePtr[0] : -1.0f, expectedScale, zeroPtr ? zeroPtr[0] : -1.0f,
                        expectedZero);
                }
                pass = ok && pass;
            }
        }
        MNN_PRINT("DynamicQuant range/tail cases: total=%d failed=%d passed=%d\n", total, failed, total - failed);
        return pass;
    }
};
MNNTestSuiteRegister(DynamicQuantRangeAndTailTest, "op/dynamic_quant_range_and_tail");
