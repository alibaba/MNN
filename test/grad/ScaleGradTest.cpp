//
//  ScaleGradTest.cpp
//  MNNTests
//
//  Created by MNN on 2022/08/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include "TestUtils.h"
#include "../tools/train/source/grad/OpGrad.hpp"

using namespace MNN;
using namespace MNN::Express;

class ScaleGradTest : public MNNTestCase {
public:
    char name[20] = "Scale";
    virtual ~ScaleGradTest() = default;

    virtual bool run(int precision) {
        const int len = 4;
        auto input = _Input({1, len, 1, 1}, NCHW);
        const float inpudata[] = {-1.0, -2.0, 0.0, 4.0};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, len * sizeof(float));

        std::vector<float> scale = {0.1, 0.2, 0.3, 0.4};
        std::vector<float> bias = {1, 2, 3, 4};
        auto output = _Scale(input, len, std::move(scale), std::move(bias));
        auto opExpr = output->expr().first;

        auto grad = OpGrad::get(opExpr->get()->type());
        float outputDiff[len] = {0.1, -0.2, -0.3, 0.4};
        auto inputGrad = grad->onGrad(opExpr, {_Const(outputDiff, {1, len, 1, 1})});

        const std::vector<float> expectedOutput = {0.01, -0.04, -0.09, 0.16};
        auto gotOutput = inputGrad[0]->readMap<float>();

        for (int i = 0; i < len; ++i) {
            auto diff = ::fabsf(gotOutput[i] - expectedOutput[i]);
            if (diff > 0.0001) {
                MNN_ERROR("%s grad test failed, expected: %f, but got: %f!\n", name, expectedOutput[i], gotOutput[i]);
                return false;
            }
        }

        return true;
    }
};

MNNTestSuiteRegister(ScaleGradTest, "grad/scale");
