//
//  TileTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/01/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include "TestUtils.h"
#include <MNN/expr/ExecutorScope.hpp>

using namespace MNN::Express;
class TileTest : public MNNTestCase {
public:
    virtual ~TileTest() = default;
    bool _run(int precision, bool lazy) {
        auto input = _Input({2, 2}, NCHW);
        input->setName("input_tensor");
        // set input data
        const float inpudata[] = {-1.0, -2.0, 3.0, 4.0};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 4 * sizeof(float));
        input->unMap();
        const int mul_data[]                    = {2, 2};
        auto mul                                = _Const(mul_data, {2}, NCHW, halide_type_of<int>());
        auto output                             = _Tile(input, mul);
        const std::vector<float> expectedOutput = {-1.0, -2.0, -1.0, -2.0, 3.0, 4.0, 3.0, 4.0,
                                                   -1.0, -2.0, -1.0, -2.0, 3.0, 4.0, 3.0, 4.0};
        auto gotOutput                          = output->readMap<float>();
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 16, 0.0001)) {
            MNN_ERROR("TileTest test failed!\n");
            return false;
        }
        return true;
    }
    virtual bool run(int precision) {
        ExecutorScope::Current()->lazyEval = false;
        auto res = _run(precision, false);
        if (!res) {
            FUNC_PRINT(1);
            return false;
        }
        ExecutorScope::Current()->lazyEval = true;
        ExecutorScope::Current()->setLazyComputeMode(MNN::Express::Executor::LAZY_CONTENT);
        res = _run(precision, true);
        if (!res) {
            FUNC_PRINT(1);
            return false;
        }
        ExecutorScope::Current()->setLazyComputeMode(MNN::Express::Executor::LAZY_FULL);
        res = _run(precision, true);
        return res;
    }
};
MNNTestSuiteRegister(TileTest, "op/tile");
