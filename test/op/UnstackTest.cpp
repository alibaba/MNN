//
//  UnstackTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/12/18.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include "TestUtils.h"

using namespace MNN::Express;
class UnstackTest : public MNNTestCase {
public:
    virtual ~UnstackTest() = default;
    virtual bool run(int precision) {
        auto input = _Input({3, 1, 2, 2}, NCHW);
        input->setName("input_tensor");
        // set input data
        const float inpudata[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 12 * sizeof(float));
        input->unMap();
        auto outputs                              = _Unstack(input, -2);
        const std::vector<float> expectedOutput_0 = {1.0, 2.0, 5.0, 6.0, 9.0, 10.0};
        const std::vector<float> expectedOutput_1 = {3.0, 4.0, 7.0, 8.0, 11.0, 12.0};
        auto gotOutput_0                          = outputs[0]->readMap<float>();
        auto gotOutput_1                          = outputs[1]->readMap<float>();
        if (!checkVector<float>(gotOutput_0, expectedOutput_0.data(), 6, 0.01)) {
            MNN_ERROR("UnstackTest test failed!\n");
            return false;
        }
        if (!checkVector<float>(gotOutput_1, expectedOutput_1.data(), 6, 0.01)) {
            MNN_ERROR("UnstackTest test failed!\n");
            return false;
        }
        return true;
    }
};
MNNTestSuiteRegister(UnstackTest, "op/unstack");
