//
//  SoftplusTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/12/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include "TestUtils.h"

using namespace MNN::Express;
class SoftplusTest : public MNNTestCase {
public:
    virtual ~SoftplusTest() = default;
    virtual bool run() {
        auto input = _Input({4,}, NCHW);
        input->setName("input_tensor");
        // set input data
        const float inpudata[] = {-1.0, -2.0, 3.0, 4.0};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 4 * sizeof(float));
        input->unMap();
        auto output = _Softplus(input);
        const std::vector<float> expectedOutput = {0.31326166, 0.12692805, 3.0485873 , 4.01815};
        auto gotOutput = output->readMap<float>();
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 4, 0.0001)) {
            MNN_ERROR("SoftplusTest test failed!\n");
            return false;
        }
        return true;
    }
};
MNNTestSuiteRegister(SoftplusTest, "op/softplus");
