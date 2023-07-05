//
//  ScaleTest.cpp
//  MNNTests
//
//  Created by MNN on 2021/10/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include "TestUtils.h"

using namespace MNN::Express;
class ScaleTest : public MNNTestCase {
public:
    virtual ~ScaleTest() = default;
    virtual bool run(int precision) {
        auto input = _Input({1, 2, 2, 1}, NCHW);
        // set input data
        const float inpudata[] = {-1.0, -2.0, 3.0, 4.0};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 4 * sizeof(float));
        input = _Convert(input, NC4HW4);
        auto output = _Scale(input, 2, {2.0, 1.0}, {3.0, 4.0});
        output = _Convert(output, NCHW);
        const std::vector<float> expectedOutput = {1, -1, 7, 8};
        auto gotOutput                        = output->readMap<float>();
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 4, 1e-5)) {
            MNN_ERROR("ScaleTest test failed!\n");
            return false;
        }
        return true;
    }
};

class ScaleInt8Test : public MNNTestCase {
public:
    virtual ~ScaleInt8Test() = default;
    virtual bool run(int precision) {
        auto input = _Input({1, 2, 2, 1}, NCHW);
        input->writeScaleMap(0.0313725, 1.f);
        // set input data
        const float inpudata[] = {-1.0, -2.0, 3.0, 4.0};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 4 * sizeof(float));
        input = _Convert(input, NC4HW4);
        auto output = _Scale(input, 2, {2.0, 1.0}, {3.0, 4.0});
        output = _Convert(output, NCHW);
        output->writeScaleMap(0.0628, 1.f);
        const std::vector<float> expectedOutput = {1, -1, 7, 8};
        auto gotOutput                        = output->readMap<float>();
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 4, 1e-1)) {
            MNN_ERROR("ScaleTestInt8 test failed!\n");
            return false;
        }
        return true;
    }
};

MNNTestSuiteRegister(ScaleTest, "op/scale");
MNNTestSuiteRegister(ScaleInt8Test, "op/scaleInt8");
