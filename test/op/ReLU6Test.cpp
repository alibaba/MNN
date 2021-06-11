//
//  ReLU6Test.cpp
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
class Relu6Test : public MNNTestCase {
public:
    virtual ~Relu6Test() = default;
    virtual bool run(int precision) {
        auto input = _Input(
            {
                4,
            },
            NCHW);
        input->setName("input_tensor");
        // set input data
        const float inpudata[] = {-1.0, 3.0, 6.0, 9.0};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 4 * sizeof(float));
        input->unMap();
        auto output                             = _Relu6(input);
        const std::vector<float> expectedOutput = {0.0, 3.0, 6.0, 6.0};
        auto gotOutput                          = output->readMap<float>();
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 4, 0.01)) {
            MNN_ERROR("Relu6Test test failed!\n");
            return false;
        }
        return true;
    }
};
MNNTestSuiteRegister(Relu6Test, "op/relu6");

class ClampTest : public MNNTestCase {
public:
    virtual ~ClampTest() = default;
    virtual bool run(int precision) {
        auto input = _Input(
            {
                4,
            },
            NCHW);
        input->setName("input_tensor");
        // set input data
        const float inpudata[] = {-1.0, 3.0, 6.0, 9.0};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 4 * sizeof(float));
        input->unMap();
        auto output                             = _Relu6(input, 1.0f, 3.0f);
        const std::vector<float> expectedOutput = {1.0, 3.0, 3.0, 3.0};
        auto gotOutput                          = output->readMap<float>();
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 4, 0.01)) {
            MNN_ERROR("ClampTest test failed!\n");
            return false;
        }
        return true;
    }
};
MNNTestSuiteRegister(ClampTest, "op/clamp");
