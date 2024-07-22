//
//  PReLUTest.cpp
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
class PreluTest : public MNNTestCase {
public:
    virtual ~PreluTest() = default;
    virtual bool run(int precision) {
        auto input = _Input({1, 4, 1, 1}, NCHW);
        input->setName("input_tensor");
        // set input data
        const float inpudata[] = {-1.0, 2.0, -3.0, 4.0};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 4 * sizeof(float));
        input->unMap();
        input                                   = _Convert(input, NC4HW4);
        auto output                             = _PRelu(input, {3.0, 1.5, 1.5, 1.5});
        output                                  = _Convert(output, NCHW);
        const std::vector<float> expectedOutput = {-3.0, 2.0, -4.5, 4.0};
        auto gotOutput                          = output->readMap<float>();
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 4, 0.01)) {
            MNN_ERROR("PreluTest test failed!\n");
            return false;
        }
        return true;
    }
};

class PreluTestInt8 : public MNNTestCase {
public:
    virtual ~PreluTestInt8() = default;
    virtual bool run(int precision) {
        auto input = _Input({1, 12, 4, 2}, NCHW);
        input->setName("input_tensor");
        // set input data
        input->writeScaleMap(0.03567, 1.0);
        const float inpudata[] = {-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
                                  2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                  -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0,
                                  4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                                  -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
                                  2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                  -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0,
                                  4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                                  -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
                                  2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                  -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0,
                                  4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 4 * sizeof(float));
        input->unMap();
        input                                   = _Convert(input, NC4HW4);
        auto output                             = _PRelu(input, {3.0, 1.5, 1.5, 1.5, 3.0, 1.5, 1.5, 1.5, 3.0, 1.5, 1.5, 1.5});
        output                                  = _Convert(output, NCHW);
        const std::vector<float> expectedOutput = {-3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0,
                                                   2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                                   -4.5, -4.5, -4.5, -4.5, -4.5, -4.5, -4.5, -4.5,
                                                   4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                                                   -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0,
                                                   2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                                   -4.5, -4.5, -4.5, -4.5, -4.5, -4.5, -4.5, -4.5,
                                                   4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                                                   -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0,
                                                   2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                                   -4.5, -4.5, -4.5, -4.5, -4.5, -4.5, -4.5, -4.5,
                                                   4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0
                                                   };
        output->writeScaleMap(0.03567, 1.0);
        auto gotOutput                          = output->readMap<float>();
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 4, 0.05)) {
            MNN_ERROR("PreluTest test failed!\n");
            return false;
        }
        return true;
    }
};

MNNTestSuiteRegister(PreluTest, "op/prelu");
MNNTestSuiteRegister(PreluTestInt8, "op/preluInt8");
