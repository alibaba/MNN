//
//  DetTest.cpp
//  MNNTests
//
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include "MNN_generated.h"
#include <cmath>
#include <cstring>

using namespace MNN::Express;
class DetTest : public MNNTestCase {
public:
    virtual bool run(int precision) override {
        MNNTEST_CPU_ONLY();
        auto check = [](const std::vector<float>& data, const std::vector<int>& shape,
                        const std::vector<float>& expected) {
            auto input = _Input(shape, NCHW, halide_type_of<float>());
            ::memcpy(input->writeMap<float>(), data.data(), data.size() * sizeof(float));
            std::unique_ptr<MNN::OpT> op(new MNN::OpT);
            op->type = MNN::OpType_Det;
            auto output = Variable::create(Expr::create(std::move(op), {input}));
            auto result = output->readMap<float>();
            if (nullptr == result) {
                return false;
            }
            for (size_t i = 0; i < expected.size(); ++i) {
                if (!std::isfinite(result[i]) || std::fabs(result[i] - expected[i]) > 1e-5f) {
                    MNN_ERROR("DetTest output %d: got %f, expected %f\n", (int)i, result[i], expected[i]);
                    return false;
                }
            }
            return true;
        };
        // Odd swap, no swap with a negative determinant, and a singular matrix in one batch.
        if (!check({0, 1, 1, 0, -2, 0, 0, 3, 1, 2, 2, 4}, {3, 2, 2}, {-1, -6, 0})) {
            return false;
        }
        // Two swaps preserve the sign; a swap after the first pivot must still reverse it.
        return check({0, 1, 0, 0, 0, 1, 1, 0, 0}, {3, 3}, {1}) &&
               check({1, 0, 0, 0, 0, 1, 0, 1, 0}, {3, 3}, {-1});
    }
};
MNNTestSuiteRegister(DetTest, "op/det");
