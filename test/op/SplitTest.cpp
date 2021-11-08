//
//  SplitTest.cpp
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
class SplitTest : public MNNTestCase {
public:
    virtual ~SplitTest() = default;
    virtual bool run(int precision) {
        auto input = _Input({2, 4}, NCHW);
        input->setName("input");
        // set input data
        const float input_data[] = {1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0};
        auto inputPtr            = input->writeMap<float>();
        memcpy(inputPtr, input_data, 8 * sizeof(float));
        auto outputs                             = _Split(input, {1, 3}, 1);
        const std::vector<float> expectedOutput0 = {1.0, 3.0};
        auto gotOutput0                          = outputs[0]->readMap<float>();
        if (!checkVector<float>(gotOutput0, expectedOutput0.data(), 2, 0.0001)) {
            MNN_ERROR("SplitTest test failed!\n");
            return false;
        }
        const std::vector<int> expectedDim0 = {2, 1};
        auto gotDim0                        = outputs[0]->getInfo()->dim;
        if (!checkVector<int>(gotDim0.data(), expectedDim0.data(), 2, 0)) {
            MNN_ERROR("SplitTest test failed!\n");
            return false;
        }
        const std::vector<float> expectedOutput1 = {2.0, 5.0, 6.0, 4.0, 7.0, 8.0};
        auto gotOutput1                          = outputs[1]->readMap<float>();
        if (!checkVector<float>(gotOutput1, expectedOutput1.data(), 6, 0.0001)) {
            MNN_ERROR("SplitTest test failed!\n");
            return false;
        }
        const std::vector<int> expectedDim1 = {2, 3};
        auto gotDim1                        = outputs[1]->getInfo()->dim;
        if (!checkVector<int>(gotDim1.data(), expectedDim1.data(), 2, 0)) {
            MNN_ERROR("SplitTest test failed!\n");
            return false;
        }
        return true;
    }
};
MNNTestSuiteRegister(SplitTest, "op/split");
