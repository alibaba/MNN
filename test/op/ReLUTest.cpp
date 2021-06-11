//
//  ReLUTest.cpp
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
class ReluTest : public MNNTestCase {
public:
    virtual ~ReluTest() = default;
    virtual bool run(int precision) {
        auto input = _Input(
            {
                4,
            },
            NCHW);
        input->setName("input_tensor");
        // set input data
        const float inpudata[] = {-1.0, -2.0, 3.0, 4.0};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 4 * sizeof(float));
        input->unMap();
        auto output                             = _Relu(input, 0.5);
        const std::vector<float> expectedOutput = {-0.5, -1, 3.0, 4.0};
        auto gotOutput                          = output->readMap<float>();
        for (int i = 0; i < 4; ++i) {
            auto diff = ::fabsf(gotOutput[i] - expectedOutput[i]);
            if (diff > 0.01) {
                MNN_ERROR("ReluTest test failed: %f - %f!\n", expectedOutput[i], gotOutput[i]);
                return false;
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(ReluTest, "op/relu");
