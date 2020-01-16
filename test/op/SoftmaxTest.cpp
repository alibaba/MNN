//
//  SoftmaxTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/01/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include "TestUtils.h"

using namespace MNN::Express;
class SoftmaxTest : public MNNTestCase {
public:
    virtual ~SoftmaxTest() = default;
    virtual bool run() {
        auto input = _Input({2,2}, NCHW);
        input->setName("input_tensor");
        // set input data
        const float inpudata[] = {-1.0, -2.0, 3.0, 4.0};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 4 * sizeof(float));
        input->unMap();
        auto output = _Softmax(input);
        const std::vector<float> expectedOutput = {0.7310586 , 0.26894143, 0.26894143, 0.7310586};
        auto gotOutput = output->readMap<float>();
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 4, 0.0001)) {
            MNN_ERROR("SoftmaxTest test failed!\n");
            return false;
        }
        return true;
    }
};
MNNTestSuiteRegister(SoftmaxTest, "op/softmax");
