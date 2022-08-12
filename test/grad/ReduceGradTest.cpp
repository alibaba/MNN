//
//  ReduceGradTest.cpp
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

class ReduceGradTest : public MNNTestCase {
public:
    char name[20] = "Reduce";
    virtual ~ReduceGradTest() = default;

    bool checkResult(VARP output, VARP outputDiff, std::vector<float> expectedOutput, const char* subname) {
        const int len = expectedOutput.size();
        auto opExpr = output->expr().first;
        auto grad = OpGrad::get(opExpr->get()->type());
        if (grad == nullptr) {
            MNN_ERROR("no grad defined for: %s %s\n", name, subname);
        }
        auto inputGrad = grad->onGrad(opExpr, {outputDiff});
        auto gotOutput = inputGrad[0]->readMap<float>();

        for (int i = 0; i < len; ++i) {
            auto diff = ::fabsf(gotOutput[i] - expectedOutput[i]);
            if (diff > 0.001) {
                MNN_ERROR("%s %s grad test failed, expected: %f, but got: %f!\n", name, subname, expectedOutput[i], gotOutput[i]);
                return false;
            }
        }
        return true;
    }

    virtual bool run(int precision) {
        const int len = 5;
        auto input = _Input({len}, NCHW);
        const float inpudata[] = {-1.0, -2.0, 0.0, 4.0, -5.0};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, len * sizeof(float));
        std::vector<float> outputDiffVec = {0.1};

        {
            auto output = _ReduceSum(input);
            auto ptr = output->readMap<float>();
            const std::vector<float> expectedOutput = {0.1, 0.1, 0.1, 0.1, 0.1};
            auto outputDiff = _Const(outputDiffVec.data(), {});
            if (!checkResult(output, outputDiff, expectedOutput, "ReduceSum")) {
                return false;
            }
        }

        {
            auto output = _ReduceMean(input);
            const std::vector<float> expectedOutput = {0.0200, 0.0200, 0.0200, 0.0200, 0.0200};
            auto outputDiff = _Const(outputDiffVec.data(), {});
            if (!checkResult(output, outputDiff, expectedOutput, "ReduceMean")) {
                return false;
            }
        }

        {
            const float inpudata[] = {-1.0, -2.0, 0.0, 4.0, 4.0};
            auto inputPtr          = input->writeMap<float>();
            memcpy(inputPtr, inpudata, len * sizeof(float));
            auto output = _ReduceMax(input);
            const std::vector<float> expectedOutput = {0.0, 0.0, 0.0, 0.05, 0.05};
            auto outputDiff = _Const(outputDiffVec.data(), {});
            if (!checkResult(output, outputDiff, expectedOutput, "ReduceMax")) {
                return false;
            }
        }

        {
            const float inpudata[] = {-2.0, -2.0, 0.0, 4.0, 4.0};
            auto inputPtr          = input->writeMap<float>();
            memcpy(inputPtr, inpudata, len * sizeof(float));
            auto output = _ReduceMin(input);
            const std::vector<float> expectedOutput = {0.05, 0.05, 0.0, 0.0, 0.0};
            auto outputDiff = _Const(outputDiffVec.data(), {});
            if (!checkResult(output, outputDiff, expectedOutput, "ReduceMin")) {
                return false;
            }
        }

        {
            const float inpudata[] = {-1.0, -2.0, 1.0, 4.0, -5.0};
            auto inputPtr          = input->writeMap<float>();
            memcpy(inputPtr, inpudata, len * sizeof(float));
            auto output = _ReduceProd(input);
            const std::vector<float> expectedOutput = {4.0000, 2.0000, -4.0000, -1.0000, 0.8000};
            auto outputDiff = _Const(outputDiffVec.data(), {});
            if (!checkResult(output, outputDiff, expectedOutput, "ReduceProd")) {
                return false;
            }
        }

        return true;
    }
};

MNNTestSuiteRegister(ReduceGradTest, "grad/reduce");
