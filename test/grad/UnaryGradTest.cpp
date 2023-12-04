//
//  UnaryGradTest.cpp
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

class UnaryGradTest : public MNNTestCase {
public:
    char name[20] = "Unary";
    virtual ~UnaryGradTest() = default;

    bool checkResult(VARP output, std::vector<float> outputDiff, std::vector<float> expectedOutput, const char* subname) {
        const int len = outputDiff.size();
        auto opExpr = output->expr().first;
        auto grad = OpGrad::get(opExpr->get()->type());
        if (grad == nullptr) {
            MNN_ERROR("no grad defined for: %s %s\n", name, subname);
        }
        auto inputGrad = grad->onGrad(opExpr, {_Const(outputDiff.data(), {len})});
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
        std::vector<float> outputDiff = {0.1, -0.2, -0.3, 0.4, 0.5};

        {
            auto output = _Log1p(input);
            const std::vector<float> expectedOutput = {INFINITY, 0.2000, -0.3000,  0.0800, -0.1250};
            if (!checkResult(output, outputDiff, expectedOutput, "Log1p")) {
                return false;
            }
        }

        {
            auto output = _Exp(input);
            const std::vector<float> expectedOutput = {3.6787946e-02, -2.7067056e-02, -3.0000e-01,  2.1839260e+01, 3.3689735e-03};
            if (!checkResult(output, outputDiff, expectedOutput, "Exp")) {
                return false;
            }
        }

        {
            auto output = _Log(input);
            const std::vector<float> expectedOutput = {-0.1000,  0.1000, -INFINITY,  0.1000, -0.1000};
            if (!checkResult(output, outputDiff, expectedOutput, "Log")) {
                return false;
            }
        }

        {
            auto output = _Cos(input);
            const std::vector<float> expectedOutput = {0.0841, -0.1819,  0.0000,  0.3027, -0.4795};
            if (!checkResult(output, outputDiff, expectedOutput, "Cos")) {
                return false;
            }
        }

        {
            auto output = _Sin(input);
            const std::vector<float> expectedOutput = {0.0540,  0.0832,  -0.3000, -0.2615,  0.1418};
            if (!checkResult(output, outputDiff, expectedOutput, "Sin")) {
                return false;
            }
        }

        {
            auto output = _Abs(input);
            const std::vector<float> expectedOutput = {-0.1000,  0.2000, 0.0000,  0.4000, -0.5000};
            if (!checkResult(output, outputDiff, expectedOutput, "Abs")) {
                return false;
            }
        }

        {
            auto output = _Negative(input);
            const std::vector<float> expectedOutput = {-0.1, 0.2, 0.3, -0.4, -0.5};
            if (!checkResult(output, outputDiff, expectedOutput, "Negative")) {
                return false;
            }
        }

        {
            auto output = _Sqrt(input);
            const std::vector<float> expectedOutput = {NAN, NAN, 0.0f, 0.1000, NAN};
            if (!checkResult(output, outputDiff, expectedOutput, "Sqrt")) {
                return false;
            }
        }

        {
            auto output = _Square(input);
            const std::vector<float> expectedOutput = {-0.2000,  0.8000, 0.0000,  3.2000, -5.0000};
            if (!checkResult(output, outputDiff, expectedOutput, "Square")) {
                return false;
            }
        }

        {
            auto output = _Sigmoid(input);
            const std::vector<float> expectedOutput = {0.0197, -0.0210, -0.0750,  0.0071,  0.0033};
            if (!checkResult(output, outputDiff, expectedOutput, "Sigmoid")) {
                return false;
            }
        }

        {
            auto output = _Tanh(input);
            const std::vector<float> expectedOutput = {4.1997e-02, -1.4130e-02, -3.0000e-01,  5.3636e-04,  9.0774e-05};
            if (!checkResult(output, outputDiff, expectedOutput, "Tanh")) {
                return false;
            }
        }

        {
            auto output = _Rsqrt(input);
            const std::vector<float> expectedOutput = {NAN, NAN, INFINITY, -0.0250, NAN};
            if (!checkResult(output, outputDiff, expectedOutput, "Rsqrt")) {
                return false;
            }
        }

        {
            auto output = _Tan(input);
            const std::vector<float> expectedOutput = {0.3426, -1.1549, -0.3000, 0.9362, 6.2139};
            if (!checkResult(output, outputDiff, expectedOutput, "Tan")) {
                return false;
            }
        }

        {
            const float inpudata[] = {-0.1, -0.2, 0.0, 4.0, -5.0};
            memcpy(inputPtr, inpudata, len * sizeof(float));
            auto output = _Asin(input);
            const std::vector<float> expectedOutput = {0.1005, -0.2041, -0.3000, NAN, NAN};
            if (!checkResult(output, outputDiff, expectedOutput, "Asin")) {
                return false;
            }
        }

        {
            const float inpudata[] = {-0.1, -0.2, 0.0, 4.0, -5.0};
            memcpy(inputPtr, inpudata, len * sizeof(float));
            auto output = _Acos(input);
            const std::vector<float> expectedOutput = {-0.1005, 0.2041, 0.3000, NAN, NAN};
            if (!checkResult(output, outputDiff, expectedOutput, "Acos")) {
                return false;
            }
        }

        {
            memcpy(inputPtr, inpudata, len * sizeof(float));
            auto output = _Atan(input);
            const std::vector<float> expectedOutput = {0.0500, -0.0400, -0.3000,  0.0235,  0.0192};
            if (!checkResult(output, outputDiff, expectedOutput, "Atan")) {
                return false;
            }
        }

        {
            auto output = _Reciprocal(input);
            const std::vector<float> expectedOutput = {-0.1000,  0.0500,  INFINITY, -0.0250, -0.0200};
            if (!checkResult(output, outputDiff, expectedOutput, "Reciprocal")) {
                return false;
            }
        }

        {
            auto output = _Acosh(input);
            const std::vector<float> expectedOutput = {INFINITY, -0.1155, NAN,  0.1033,  0.1021};
            if (!checkResult(output, outputDiff, expectedOutput, "Acosh")) {
                return false;
            }
        }

        {
            auto output = _Sinh(input);
            const std::vector<float> expectedOutput = {0.1543, -0.7524, -0.3000, 10.9233, 37.1050};
            if (!checkResult(output, outputDiff, expectedOutput, "Sinh")) {
                return false;
            }
        }

        {
            auto output = _Cosh(input);
            const std::vector<float> expectedOutput = {-0.1175,   0.7254,  0.0000,  10.9160, -37.1016};
            if (!checkResult(output, outputDiff, expectedOutput, "Cosh")) {
                return false;
            }
        }

        {
            auto output = _Asinh(input);
            const std::vector<float> expectedOutput = {0.0707, -0.0894, -0.3000,  0.0970,  0.0981};
            if (!checkResult(output, outputDiff, expectedOutput, "Asinh")) {
                return false;
            }
        }

        {
            auto output = _Atanh(input);
            const std::vector<float> expectedOutput = {INFINITY, 0.0667, -0.3000, -0.0267, -0.0208};
            if (!checkResult(output, outputDiff, expectedOutput, "Atanh")) {
                return false;
            }
        }

        {
            auto output = _Erf(input);
            const std::vector<float> expectedOutput = {4.1511e-02, -4.1334e-03, -3.3851e-01,  5.0793e-08,  7.8354e-12};
            if (!checkResult(output, outputDiff, expectedOutput, "Erf")) {
                return false;
            }
        }

        {
            auto output = _Erfc(input);
            const std::vector<float> expectedOutput = {-4.1511e-02, 4.1334e-03, 3.3851e-01,  -5.0793e-08,  -7.8354e-12};
            if (!checkResult(output, outputDiff, expectedOutput, "Erfc")) {
                return false;
            }
        }

        {
            const float inpudata[] = {-0.1, -0.2, 0.0, 4.0, -5.0};
            memcpy(inputPtr, inpudata, len * sizeof(float));
            auto output = _Erfinv(input);
            const std::vector<float> expectedOutput = {0.0893, -0.1830, -0.2659, NAN, NAN};
            if (!checkResult(output, outputDiff, expectedOutput, "Erfinv")) {
                return false;
            }
        }

        {
            memcpy(inputPtr, inpudata, len * sizeof(float));
            auto output = _Expm1(input);
            const std::vector<float> expectedOutput = {3.6788e-02, -2.7067e-02, -3.0000e-01,  2.1839e+01,  3.3690e-03};
            if (!checkResult(output, outputDiff, expectedOutput, "Expm1")) {
                return false;
            }
        }

        {
            auto output = _Hardswish(input);
            const std::vector<float> expectedOutput = {0.0167,  0.0333, -0.1500,  0.4000,  0.0000};
            if (!checkResult(output, outputDiff, expectedOutput, "Hardswish")) {
                return false;
            }
        }

        {
            auto output = _Gelu(input);
            const std::vector<float> expectedOutput = {-8.3315e-03, 1.7046e-02, -1.5000e-01, 4.0020e-01, -3.5614e-06};
            if (!checkResult(output, outputDiff, expectedOutput, "Gelu")) {
                return false;
            }
        }

        return true;
    }
};

MNNTestSuiteRegister(UnaryGradTest, "grad/unary");
