//
//  PReLUGradTest.cpp
//  MNNTests
//
//  Created by MNN on 2022/07/12.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include "TestUtils.h"
#include "../tools/train/source/grad/OpGrad.hpp"

using namespace MNN;
using namespace MNN::Express;

class PReLUGradTest : public MNNTestCase {
public:
    char name[20] = "PReLU";
    virtual ~PReLUGradTest() = default;

    virtual bool run(int precision) {
        const int len = 5;
        auto input = _Input({1, len, 1, 1}, NCHW);
        const float inpudata[] = {-1.0, -2.0, 0.0, 4.0, -5.0};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, len * sizeof(float));

        auto output = _PRelu(_Convert(input, NC4HW4), {0.25, 0.5, -0.3, 0.2, 0.1});
        auto opExpr = output->expr().first;

        auto grad = OpGrad::get(opExpr->get()->type());
        std::vector<float> outputDiff = {0.1, -0.2, -0.3, 0.4, 0.5};
        auto outputDiffVar = _Const(outputDiff.data(), {1, len, 1, 1}, NCHW);
        auto inputGrad = grad->onGrad(opExpr, {_Convert(outputDiffVar, NC4HW4)});

        const std::vector<float> expectedOutput = {0.025, -0.1, 0.09, 0.4, 0.05};
        auto gotOutput = _Convert(inputGrad[0], NCHW)->readMap<float>();

        for (int i = 0; i < len; ++i) {
            auto diff = ::fabsf(gotOutput[i] - expectedOutput[i]);
            if (diff > 0.000001) {
                MNN_ERROR("%s grad test failed, expected: %f, but got: %f!\n", name, expectedOutput[i], gotOutput[i]);
                return false;
            }
        }

        return true;
    }
};

MNNTestSuiteRegister(PReLUGradTest, "grad/prelu");
