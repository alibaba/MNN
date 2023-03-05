//
//  SizeTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/12/13.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include "TestUtils.h"

using namespace MNN::Express;
class SizeTest : public MNNTestCase {
public:
    virtual ~SizeTest() = default;
    bool _run(int precision, bool lazy) {
        auto input = _Input({2, 2}, NCHW);
        input->setName("input_tensor");
        // set input data
        const float inpudata[] = {-1.0, -2.0, 3.0, 4.0};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 4 * sizeof(float));
        input->unMap();
        auto output                           = _Size(input);
        const std::vector<int> expectedOutput = {4};
        auto gotOutput                        = output->readMap<int>();
        if (!checkVector<int>(gotOutput, expectedOutput.data(), 1, 0)) {
            MNN_ERROR("SizeTest test failed!\n");
            return false;
        }
        auto dims = output->getInfo()->dim;
        if (dims.size() != 0) {
            MNN_ERROR("SizeTest test failed!\n");
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
MNNTestSuiteRegister(SizeTest, "op/size");
