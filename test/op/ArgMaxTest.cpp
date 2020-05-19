//
//  ArgMaxTest.cpp
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
class ArgMaxTest : public MNNTestCase {
public:
    virtual ~ArgMaxTest() = default;
    virtual bool run() {
        auto input = _Input({4,4}, NHWC);
        input->setName("input_tensor");
        // set input data
        const float inpudata[] = {-1.0, 2.0, -3.0, 4.0,
                                  5.0, -6.0, 7.0, -8.0,
                                  -9.0, -10.0, 11.0, 12.0,
                                  13.0, 14.0, -15.0, -16.0}; 
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 16 * sizeof(float));
        input->unMap();
        auto output_0 = _ArgMax(input, 0);
        auto output_1 = _ArgMax(input, 1);
        const std::vector<int> expectedOutput_0 = {3, 3, 2, 2};
        const std::vector<int> expectedOutput_1 = {3, 2, 3, 1};
        auto gotOutput_0 = output_0->readMap<int>();
        auto gotOutput_1 = output_1->readMap<int>();
        if (!checkVector<int>(gotOutput_0, expectedOutput_0.data(), 4, 0)) {
            MNN_ERROR("ArgMaxTest test axis_0 failed!\n");
            return false;
        } 
        if (!checkVector<int>(gotOutput_1, expectedOutput_1.data(), 4, 0)) {
            MNN_ERROR("ArgMaxTest test axis_1 failed!\n");
            return false;
        } 
        return true;
    }
};
class ArgMinTest : public MNNTestCase {
public:
    virtual ~ArgMinTest() = default;
    virtual bool run() {
        auto input = _Input({4,4}, NHWC);
        input->setName("input_tensor");
        // set input data
        const float inpudata[] = {-1.0, 2.0, -3.0, 4.0,
                                  5.0, -6.0, 7.0, -8.0,
                                  -9.0, -10.0, 11.0, 12.0,
                                  13.0, 14.0, -15.0, -16.0};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 16 * sizeof(float));
        input->unMap();
        auto output_0 = _ArgMin(input, 0);
        auto output_1 = _ArgMin(input, 1);
        const std::vector<int> expectedOutput_0 = {2, 2, 3, 3};
        const std::vector<int> expectedOutput_1 = {2, 3, 1, 3};
        auto gotOutput_0 = output_0->readMap<int>();
        auto gotOutput_1 = output_1->readMap<int>();
        if (!checkVector<int>(gotOutput_0, expectedOutput_0.data(), 4, 0)) {
            MNN_ERROR("ArgMinTest test axis_0 failed!\n");
            return false;
        }
        if (!checkVector<int>(gotOutput_1, expectedOutput_1.data(), 4, 0)) {
            MNN_ERROR("ArgMinTest test axis_1 failed!\n");
            return false;
        }
        return true;
    }
};
MNNTestSuiteRegister(ArgMaxTest, "op/argmax");
MNNTestSuiteRegister(ArgMinTest, "op/argmin");
