//
//  ThresholdTest.cpp
//  MNNTests
//
//  Created by MNN on 2020/07/27.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include "TestUtils.h"

using namespace MNN::Express;
class ThresholdTest : public MNNTestCase {
public:
    virtual ~ThresholdTest() = default;
    virtual bool run(int precision) {
        auto input = _Input({4}, NCHW);
        input->setName("input_tensor");
        // set input data
        const float inpudata[] = {-2.0, 0.0, 2.0, 3.0};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 4 * sizeof(float));
        input->unMap();
        auto output                             = _Threshold(input, 2.0);
        const std::vector<float> expectedOutput = {0.0, 0.0, 0.0, 1.0};
        auto gotOutput                          = output->readMap<float>();
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 4, 0.01)) {
            MNN_ERROR("ThresholdTest test failed!\n");
            return false;
        }
        return true;
    }
};
MNNTestSuiteRegister(ThresholdTest, "op/threshold");
