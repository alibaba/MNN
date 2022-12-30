//
//  RasrerTest.cpp
//  MNNTests
//
//  Created by MNN on 2021/12/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include "TestUtils.h"

using namespace MNN::Express;
class RasrerTest : public MNNTestCase {
public:
    virtual ~RasrerTest() = default;
    bool _run(int precision, bool lazy) {
        auto input = _Input({2, 2}, NCHW);
        input->setName("input_tensor");
        // set input data
        const float inpudata[] = {1, 2, 3, 4};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 4 * sizeof(float));
        // transpose
        auto output                             = _Raster({input}, {0, 4, 1, 2, 0, 4, 2, 1, 1, 2, 2}, {2, 2});
        const std::vector<float> expectedOutput = {1, 3, 2, 4};
        auto gotOutput                          = output->readMap<float>();
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 4, 0.01)) {
            MNN_ERROR("RasterTest transpose test failed!\n");
            return false;
        }
        auto output0                             = _Raster({input}, {2, 4, 2, 1, 0, 4, 2, 1, 1, 1, 2}, {2});
        const std::vector<float> expectedOutput0 = {3, 4};
        auto gotOutput0                          = output0->readMap<float>();
        if (!checkVector<float>(gotOutput0, expectedOutput0.data(), 2, 0.01)) {
            MNN_ERROR("RasterTest slice test failed!\n");
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
MNNTestSuiteRegister(RasrerTest, "op/raster");
