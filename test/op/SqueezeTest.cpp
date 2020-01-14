//
//  SqueezeTest.cpp
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
class SqueezeTest : public MNNTestCase {
public:
    virtual ~SqueezeTest() = default;
    virtual bool run() {
        auto input = _Input({1, 1, 1, 4}, NCHW);
        input->setName("input_tensor");
        // set input data
        const float inpudata[] = {-1.0, -2.0, 3.0, 4.0};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 4 * sizeof(float));
        input->unMap();
        auto output = _Squeeze(input);
        const std::vector<float> expectedOutput = {-1.0, -2.0, 3.0, 4.0};
        auto gotOutput = output->readMap<float>();
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 4, 0.0001)) {
            MNN_ERROR("SqueezeTest test failed!\n");
            return false;
        }
        const std::vector<int> expectedDim = {4};
        auto gotDim = output->getInfo()->dim;
        if (!checkVector<int>(gotDim.data(), expectedDim.data(), 1, 0)) {
            MNN_ERROR("SqueezeTest test failed!\n");
            return false;
        }
        return true;
    }
};
MNNTestSuiteRegister(SqueezeTest, "op/squeeze");
