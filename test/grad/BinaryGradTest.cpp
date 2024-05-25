//
//  BinaryGradTest.cpp
//  MNNTests
//
//  Created by MNN on 2022/07/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include "TestUtils.h"
#include "../tools/train/source/grad/OpGrad.hpp"

using namespace MNN;
using namespace MNN::Express;

class BinaryGradTest : public MNNTestCase {
public:
    char name[20] = "Binary";
    virtual ~BinaryGradTest() = default;

    bool checkResult(VARP output, std::vector<float> outputDiff, std::vector<float> expectedOutputA, std::vector<float> expectedOutputB, const char* subname) {
        const int len = outputDiff.size();
        auto opExpr = output->expr().first;
        auto grad = OpGrad::get(opExpr->get()->type());
        if (grad == nullptr) {
            MNN_ERROR("no grad defined for: %s %s\n", name, subname);
        }
        auto inputGrad = grad->onGrad(opExpr, {_Const(outputDiff.data(), {len})});
        auto gotOutputA = inputGrad[0]->readMap<float>();
        auto gotOutputB = inputGrad[1]->readMap<float>();

        const float threshold = 1e-4;

        bool res = true;
        for (int i = 0; i < len; ++i) {
            auto diff = ::fabsf(gotOutputA[i] - expectedOutputA[i]);
            if (diff > threshold) {
                MNN_ERROR("%s %s %d grad test failed for input A, expected: %f, but got: %f!\n", name, subname, i, expectedOutputA[i], gotOutputA[i]);
                res = false;
            }
            diff = ::fabsf(gotOutputB[i] - expectedOutputB[i]);
            if (diff > threshold) {
                MNN_ERROR("%s %s %d grad test failed for input B, expected: %f, but got: %f!\n", name, subname, i, expectedOutputB[i], gotOutputB[i]);
                res = false;
            }
        }
        return res;
    }

    virtual bool run(int precision) {
        const int len = 5;
        auto inputA = _Input({len}, NCHW);
        const float inputAData[] = {-1.0, 2.0, 0.0, 4.0, -5.0};
        auto inputAPtr = inputA->writeMap<float>();
        memcpy(inputAPtr, inputAData, len * sizeof(float));

        auto inputB = _Input({len}, NCHW);
        const float inputBData[] = {1.0, 2.0, 0.0, -4.0, 5.0};
        auto inputBPtr = inputB->writeMap<float>();
        memcpy(inputBPtr, inputBData, len * sizeof(float));

        std::vector<float> outputDiff = {0.1, -0.2, -0.3, 0.4, 0.5};

        {
            auto output = _Add(inputA, inputB);
            const std::vector<float> expectedOutputA = {0.1, -0.2, -0.3, 0.4, 0.5};
            const std::vector<float> expectedOutputB = {0.1, -0.2, -0.3, 0.4, 0.5};
            if (!checkResult(output, outputDiff, expectedOutputA, expectedOutputB, "Add")) {
                return false;
            }
        }

        {
            auto output = _Subtract(inputA, inputB);
            const std::vector<float> expectedOutputA = {0.1, -0.2, -0.3, 0.4, 0.5};
            const std::vector<float> expectedOutputB = {-0.1, 0.2, 0.3, -0.4, -0.5};
            if (!checkResult(output, outputDiff, expectedOutputA, expectedOutputB, "Sub")) {
                return false;
            }
        }

        {
            auto output = _Multiply(inputA, inputB);
            const std::vector<float> expectedOutputA = {0.1000, -0.4000, -0.0000, -1.6000, 2.5000};
            const std::vector<float> expectedOutputB = {-0.1000, -0.4000, -0.0000,  1.6000, -2.5000};
            if (!checkResult(output, outputDiff, expectedOutputA, expectedOutputB, "Mul")) {
                return false;
            }
        }

        {
            auto output = _Divide(inputA, inputB);
            const std::vector<float> expectedOutputA = {0.1000, -0.1000, -INFINITY, -0.1000, 0.1000};
            const std::vector<float> expectedOutputB = {0.1000,  0.1000, NAN, -0.1000, 0.1000};
            if (!checkResult(output, outputDiff, expectedOutputA, expectedOutputB, "RealDiv")) {
                return false;
            }
        }

        {
            auto output = _Pow(inputA, inputB);
            const std::vector<float> expectedOutputA = {1.0000e-01, -8.0000e-01,  0.0000e+00, -1.5625e-03,  1.5625e+03};
            const std::vector<float> expectedOutputB = {NAN, -0.5545, INFINITY, 0.0022, NAN};
            if (!checkResult(output, outputDiff, expectedOutputA, expectedOutputB, "Pow")) {
                return false;
            }
        }

        {
            auto output = _Maximum(inputA, inputB);
            const std::vector<float> expectedOutputA = {0.0000, -0.1000, -0.1500,  0.4000,  0.0000};
            const std::vector<float> expectedOutputB = {0.1000, -0.1000, -0.1500,  0.0000,  0.5000};
            if (!checkResult(output, outputDiff, expectedOutputA, expectedOutputB, "Maximum")) {
                return false;
            }
        }

        {
            auto output = _Minimum(inputA, inputB);
            const std::vector<float> expectedOutputA = {0.1000, -0.1000, -0.1500,  0.0000,  0.5000};
            const std::vector<float> expectedOutputB = {0.0000, -0.1000, -0.1500,  0.4000,  0.0000};
            if (!checkResult(output, outputDiff, expectedOutputA, expectedOutputB, "Minimum")) {
                return false;
            }
        }
        
        {
            auto output = _Atan2(inputA, inputB);
            const std::vector<float> expectedOutputA = {0.0500, -0.0500, NAN, -0.0500, 0.0500};
            const std::vector<float> expectedOutputB = {0.0500, 0.0500, NAN, -0.0500, 0.0500};
            if (!checkResult(output, outputDiff, expectedOutputA, expectedOutputB, "Atan2")) {
                return false;
            }
        }

        {
            auto output = _SquaredDifference(inputA, inputB);
            const std::vector<float> expectedOutputA = {-0.4000, -0.0000, -0.0000, 6.4000, -10.0000};
            const std::vector<float> expectedOutputB = {0.4000, 0.0000, 0.0000, -6.4000, 10.0000};
            if (!checkResult(output, outputDiff, expectedOutputA, expectedOutputB, "Atan2")) {
                return false;
            }
        }

        return true;
    }
};

MNNTestSuiteRegister(BinaryGradTest, "grad/binary");
