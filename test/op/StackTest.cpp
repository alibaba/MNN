//
//  StackTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/12/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include "TestUtils.h"

using namespace MNN::Express;
class StackTest : public MNNTestCase {
public:
    virtual ~StackTest() = default;
    virtual bool run(int precision) {
        auto input0 = _Input({3, 1, 2}, NCHW);
        input0->setName("input0");
        const float input0_data[] = {1.0, 2.0, 5.0, 6.0, 9.0, 10.0};
        auto input0Ptr            = input0->writeMap<float>();
        memcpy(input0Ptr, input0_data, 6 * sizeof(float));
        input0->unMap();
        auto input1 = _Input({3, 1, 2}, NCHW);
        input1->setName("input1");
        const float input1_data[] = {3.0, 4.0, 7.0, 8.0, 11.0, 12.0};
        auto input1Ptr            = input1->writeMap<float>();
        memcpy(input1Ptr, input1_data, 6 * sizeof(float));
        input1->unMap();
        auto output                             = _Stack({input0, input1}, 2);
        const std::vector<float> expectedOutput = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
        auto gotOutput                          = output->readMap<float>();
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 12, 0.01)) {
            MNN_ERROR("StackTest test failed!\n");
            return false;
        }
        return true;
    }
};

MNNTestSuiteRegister(StackTest, "op/stack");
