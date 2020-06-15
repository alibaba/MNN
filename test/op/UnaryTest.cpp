//
//  UnaryTest.cpp
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
class AbsTest : public MNNTestCase {
public:
    virtual ~AbsTest() = default;
    virtual bool run() {
        auto input = _Input({4,}, NCHW);
        input->setName("input_tensor");
        // set input data
        const float inpudata[] = {-1.0, -2.0, 3.0, 4.0};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 4 * sizeof(float));
        input->unMap();
        auto output = _Abs(input);
        const std::vector<float> expectedOutput = {1.0, 2.0, 3.0, 4.0};
        auto gotOutput = output->readMap<float>();
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 4, 0.01)) {
            MNN_ERROR("AbsTest test failed!\n");
            return false;
        }
        return true;
    }
};
class NegativeTest : public MNNTestCase {
public:
    virtual ~NegativeTest() = default;
    virtual bool run() {
        auto input = _Input({4,}, NCHW);
        input->setName("input_tensor");
        // set input data
        const float inpudata[] = {-1.0, -2.0, 3.0, 4.0};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 4 * sizeof(float));
        input->unMap();
        auto output = _Negative(input);
        const std::vector<float> expectedOutput = {1.0, 2.0, -3.0, -4.0};
        auto gotOutput = output->readMap<float>();
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 4, 0.01)) {
            MNN_ERROR("NegativeTest test failed!\n");
            return false;
        }
        return true;
    }
};
class FloorTest : public MNNTestCase {
public:
    virtual ~FloorTest() = default;
    virtual bool run() {
        auto input = _Input({4,}, NCHW);
        input->setName("input_tensor");
        // set input data
        const float inpudata[] = {-1.3, -2.6, 3.2, 4.6};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 4 * sizeof(float));
        input->unMap();
        auto output = _Floor(input);
        const std::vector<float> expectedOutput = {-2.0, -3.0, 3.0, 4.0};
        auto gotOutput = output->readMap<float>();
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 4, 0.01)) {
            MNN_ERROR("FloorTest test failed!\n");
            return false;
        }
        return true;
    }
};
class CeilTest : public MNNTestCase {
public:
    virtual ~CeilTest() = default;
    virtual bool run() {
        auto input = _Input({4,}, NCHW);
        input->setName("input_tensor");
        // set input data
        const float inpudata[] = {-1.3, -2.6, 3.2, 4.6};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 4 * sizeof(float));
        input->unMap();
        auto output = _Ceil(input);
        const std::vector<float> expectedOutput = {-1.0, -2.0, 4.0, 5.0};
        auto gotOutput = output->readMap<float>();
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 4, 0.01)) {
            MNN_ERROR("CeilTest test failed!\n");
            return false;
        }
        return true;
    }
};
class SquareTest : public MNNTestCase {
public:
    virtual ~SquareTest() = default;
    virtual bool run() {
        auto input = _Input({4,}, NCHW);
        input->setName("input_tensor");
        // set input data
        const float inpudata[] = {-1.0, -2.0, 3.0, 4.0};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 4 * sizeof(float));
        input->unMap();
        auto output = _Square(input);
        const std::vector<float> expectedOutput = {1.0, 4.0, 9.0, 16.0};
        auto gotOutput = output->readMap<float>();
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 4, 0.01)) {
            MNN_ERROR("SquareTest test failed!\n");
            return false;
        }
        return true;
    }
};
class SqrtTest : public MNNTestCase {
public:
    virtual ~SqrtTest() = default;
    virtual bool run() {
        auto input = _Input({4,}, NCHW);
        input->setName("input_tensor");
        // set input data
        const float inpudata[] = {1.0, 4.0, 9.0, 16.0};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 4 * sizeof(float));
        input->unMap();
        auto output = _Sqrt(input);
        const std::vector<float> expectedOutput = {1.0, 2.0, 3.0, 4.0};
        auto gotOutput = output->readMap<float>();
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 4, 0.01)) {
            MNN_ERROR("SqrtTest test failed!\n");
            return false;
        }
        return true;
    }
};
class RsqrtTest : public MNNTestCase {
public:
    virtual ~RsqrtTest() = default;
    virtual bool run() {
        auto input = _Input({4,}, NCHW);
        input->setName("input_tensor");
        // set input data
        const float inpudata[] = {1.0, 4.0, 9.0, 16.0};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 4 * sizeof(float));
        input->unMap();
        auto output = _Rsqrt(input);
        const std::vector<float> expectedOutput = {1.0, 1.0/2.0, 1.0/3.0, 1.0/4.0};
        auto gotOutput = output->readMap<float>();
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 4, 0.01)) {
            MNN_ERROR("RsqrtTest test failed!\n");
            return false;
        }
        return true;
    }
};
class ExpTest : public MNNTestCase {
public:
    virtual ~ExpTest() = default;
    virtual bool run() {
        auto input = _Input({4,}, NCHW);
        input->setName("input_tensor");
        // set input data
        const float inpudata[] = {1.0, 2.0, 3.0, 4.0};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 4 * sizeof(float));
        input->unMap();
        auto output = _Exp(input);
        const std::vector<float> expectedOutput = {2.718, 7.389, 20.086, 54.598};
        auto gotOutput = output->readMap<float>();
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 4, 0.01)) {
            MNN_ERROR("ExpTest test failed!\n");
            return false;
        }
        return true;
    }
};
class LogTest : public MNNTestCase {
public:
    virtual ~LogTest() = default;
    virtual bool run() {
        auto input = _Input({4,}, NCHW);
        input->setName("input_tensor");
        // set input data
        const float inpudata[] = {2.718, 7.389, 20.086, 54.598};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 4 * sizeof(float));
        input->unMap();
        auto output = _Log(input);
        const std::vector<float> expectedOutput = {1.0, 2.0, 3.0, 4.0};
        auto gotOutput = output->readMap<float>();
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 4, 0.01)) {
            MNN_ERROR("LogTest test failed!\n");
            return false;
        }
        return true;
    }
};
class SinTest : public MNNTestCase {
public:
    virtual ~SinTest() = default;
    virtual bool run() {
        auto input = _Input({4,}, NCHW);
        input->setName("input_tensor");
        // set input data
        const float inpudata[] = {0.0, 3.14/2.0, 3.14, 3.14*3.0/2.0};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 4 * sizeof(float));
        input->unMap();
        auto output = _Sin(input);
        const std::vector<float> expectedOutput = {0.0, 1.0, 0.0, -1.0};
        auto gotOutput = output->readMap<float>();
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 4, 0.01)) {
            MNN_ERROR("SinTest test failed!\n");
            return false;
        }
        return true;
    }
};
class CosTest : public MNNTestCase {
public:
    virtual ~CosTest() = default;
    virtual bool run() {
        auto input = _Input({4,}, NCHW);
        input->setName("input_tensor");
        // set input data
        const float inpudata[] = {0.0, 3.14/2.0, 3.14, 3.14*3.0/2.0};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 4 * sizeof(float));
        input->unMap();
        auto output = _Cos(input);
        const std::vector<float> expectedOutput = {1.0, 0.0, -1.0, 0.0};
        auto gotOutput = output->readMap<float>();
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 4, 0.01)) {
            MNN_ERROR("CosTest test failed!\n");
            return false;
        }
        return true;
    }
};
class TanTest : public MNNTestCase {
public:
    virtual ~TanTest() = default;
    virtual bool run() {
        auto input = _Input({4,}, NCHW);
        input->setName("input_tensor");
        // set input data
        const float inpudata[] = {100.0, 200.0, 300.0, 400.0};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 4 * sizeof(float));
        input->unMap();
        auto output = _Tan(input);
        const std::vector<float> expectedOutput = {-0.59, -1.79, 45.24, 1.62};
        auto gotOutput = output->readMap<float>();
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 4, 0.01)) {
            MNN_ERROR("TanTest test failed!\n");
            return false;
        }
        return true;
    }
};
class AsinTest : public MNNTestCase {
public:
    virtual ~AsinTest() = default;
    virtual bool run() {
        auto input = _Input({4,}, NCHW);
        input->setName("input_tensor");
        // set input data
        const float inpudata[] = {-1.0, 0.0, 1.0, 0.707};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 4 * sizeof(float));
        input->unMap();
        auto output = _Asin(input);
        const std::vector<float> expectedOutput = {-3.14/2.0, 0.0, 3.14/2.0, 3.14/4.0};
        auto gotOutput = output->readMap<float>();
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 4, 0.01)) {
            MNN_ERROR("AsinTest test failed!\n");
            return false;
        }
        return true;
    }
};
class AcosTest : public MNNTestCase {
public:
    virtual ~AcosTest() = default;
    virtual bool run() {
        auto input = _Input({4,}, NCHW);
        input->setName("input_tensor");
        // set input data
        const float inpudata[] = {-1.0, 0.0, 1.0, 0.707};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 4 * sizeof(float));
        input->unMap();
        auto output = _Acos(input);
        const std::vector<float> expectedOutput = {3.14, 1.57, 0.0, 3.14/4.0};
        auto gotOutput = output->readMap<float>();
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 4, 0.01)) {
            MNN_ERROR("AcosTest test failed!\n");
            return false;
        }
        return true;
    }
};
class AtanTest : public MNNTestCase {
public:
    virtual ~AtanTest() = default;
    virtual bool run() {
        auto input = _Input({4,}, NCHW);
        input->setName("input_tensor");
        // set input data
        const float inpudata[] = {-2.0, -1.0, 0.0, 1.0};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 4 * sizeof(float));
        input->unMap();
        auto output = _Atan(input);
        const std::vector<float> expectedOutput = {-1.11, -3.14/4.0, 0.0, 3.14/4.0};
        auto gotOutput = output->readMap<float>();
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 4, 0.01)) {
            MNN_ERROR("AtanTest test failed!\n");
            return false;
        }
        return true;
    }
};
class ReciprocalTest : public MNNTestCase {
public:
    virtual ~ReciprocalTest() = default;
    virtual bool run() {
        auto input = _Input({4,}, NCHW);
        input->setName("input_tensor");
        // set input data
        const float inpudata[] = {-2.0, -4.0, 2.0, 4.0};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 4 * sizeof(float));
        input->unMap();
        auto output = _Reciprocal(input);
        const std::vector<float> expectedOutput = {-0.5, -0.25, 0.50, 0.25};
        auto gotOutput = output->readMap<float>();
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 4, 0.01)) {
            MNN_ERROR("ReciprocalTest test failed!\n");
            return false;
        }
        return true;
    }
};
class Log1PTest : public MNNTestCase {
public:
    virtual ~Log1PTest() = default;
    virtual bool run() {
        auto input = _Input({4,}, NCHW);
        input->setName("input_tensor");
        // set input data
        const float inpudata[] = {0.0, 1.0, 2.0, 3.0};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 4 * sizeof(float));
        input->unMap();
        auto output = _Log1p(input);
        const std::vector<float> expectedOutput = {0.0, 0.69, 1.10, 1.39};
        auto gotOutput = output->readMap<float>();
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 4, 0.01)) {
            MNN_ERROR("Log1PTest test failed!\n");
            return false;
        }
        return true;
    }
};
class TanhTest : public MNNTestCase {
public:
    virtual ~TanhTest() = default;
    virtual bool run() {
        auto input = _Input({4,}, NCHW);
        input->setName("input_tensor");
        // set input data
        const float inpudata[] = {-1.0, 0.0, 1.0, 2.0};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 4 * sizeof(float));
        input->unMap();
        auto output = _Tanh(input);
        const std::vector<float> expectedOutput = {-0.76, 0.0, 0.76, 0.96};
        auto gotOutput = output->readMap<float>();
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 4, 0.01)) {
            MNN_ERROR("TanhTest test failed!\n");
            return false;
        }
        return true;
    }
};
class SigmoidTest : public MNNTestCase {
public:
    virtual ~SigmoidTest() = default;
    virtual bool run() {
        auto input = _Input({4,}, NCHW);
        input->setName("input_tensor");
        // set input data
        const float inpudata[] = {-1.0, 0.0, 1.0, 2.0};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 4 * sizeof(float));
        input->unMap();
        auto output = _Sigmoid(input);
        const std::vector<float> expectedOutput = {0.27, 0.50, 0.73, 0.88};
        auto gotOutput = output->readMap<float>();
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 4, 0.01)) {
            MNN_ERROR("SigmoidTest test failed!\n");
            return false;
        }
        return true;
    }
};
class AcoshTest : public MNNTestCase {
public:
    virtual ~AcoshTest() = default;
    virtual bool run() {
        auto input = _Input({4,}, NCHW);
        input->setName("input_tensor");
        // set input data
        const float inpudata[] = {1.0, 2.0, 3.0, 4.0};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 4 * sizeof(float));
        input->unMap();
        auto output = _Acosh(input);
        const std::vector<float> expectedOutput = {0., 1.3169579, 1.76274717, 2.06343707};
        auto gotOutput = output->readMap<float>();
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 4, 0.01)) {
            MNN_ERROR("AcoshTest test failed!\n");
            return false;
        }
        return true;
    }
};
class AsinhTest : public MNNTestCase {
public:
    virtual ~AsinhTest() = default;
    virtual bool run() {
        auto input = _Input({4,}, NCHW);
        input->setName("input_tensor");
        // set input data
        const float inpudata[] = {1.0, 2.0, 3.0, 4.0};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 4 * sizeof(float));
        input->unMap();
        auto output = _Asinh(input);
        const std::vector<float> expectedOutput = {0.88137359, 1.44363548, 1.81844646, 2.09471255};
        auto gotOutput = output->readMap<float>();
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 4, 0.01)) {
            MNN_ERROR("AsinhTest test failed!\n");
            return false;
        }
        return true;
    }
};
class AtanhTest : public MNNTestCase {
public:
    virtual ~AtanhTest() = default;
    virtual bool run() {
        auto input = _Input({4,}, NCHW);
        input->setName("input_tensor");
        // set input data
        const float inpudata[] = {0., 0.1, 0.2, 0.3};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 4 * sizeof(float));
        input->unMap();
        auto output = _Atanh(input);
        const std::vector<float> expectedOutput = {0., 0.10033535, 0.20273255, 0.3095196};
        auto gotOutput = output->readMap<float>();
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 4, 0.01)) {
            MNN_ERROR("AtanhTest test failed!\n");
            return false;
        }
        return true;
    }
};
class RoundTest : public MNNTestCase {
public:
    virtual ~RoundTest() = default;
    virtual bool run() {
        auto input = _Input({4,}, NCHW);
        input->setName("input_tensor");
        // set input data
        const float inpudata[] = {-1.2, -0.6, 0.4, 1.6};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 4 * sizeof(float));
        input->unMap();
        auto output = _Round(input);
        const std::vector<float> expectedOutput = {-1., -1., 0., 2.};
        auto gotOutput = output->readMap<float>();
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 4, 0.01)) {
            MNN_ERROR("RoundTest test failed!\n");
            return false;
        }
        return true;
    }
};
class SignTest : public MNNTestCase {
public:
    virtual ~SignTest() = default;
    virtual bool run() {
        auto input = _Input({4,}, NCHW);
        input->setName("input_tensor");
        // set input data
        const float inpudata[] = {-1.2, 0., 0.4, 1.6};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 4 * sizeof(float));
        input->unMap();
        auto output = _Sign(input);
        const std::vector<float> expectedOutput = {-1., 0., 1., 1.};
        auto gotOutput = output->readMap<float>();
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 4, 0.01)) {
            MNN_ERROR("SignTest test failed!\n");
            return false;
        }
        return true;
    }
};
class CoshTest : public MNNTestCase {
public:
    virtual ~CoshTest() = default;
    virtual bool run() {
        auto input = _Input({4,}, NCHW);
        input->setName("input_tensor");
        // set input data
        const float inpudata[] = {-1.2, 0., 0.4, 1.6};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 4 * sizeof(float));
        input->unMap();
        auto output = _Cosh(input);
        const std::vector<float> expectedOutput = {1.81065557, 1., 1.08107237, 2.57746447};
        auto gotOutput = output->readMap<float>();
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 4, 0.01)) {
            MNN_ERROR("CoshTest test failed!\n");
            return false;
        }
        return true;
    }
};
class ErfTest : public MNNTestCase {
public:
    virtual ~ErfTest() = default;
    virtual bool run() {
        auto input = _Input({4,}, NCHW);
        input->setName("input_tensor");
        // set input data
        const float inpudata[] = {-1.2, 0., 0.4, 1.6};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 4 * sizeof(float));
        input->unMap();
        auto output = _Erf(input);
        const std::vector<float> expectedOutput = {-0.91031396,  0.,  0.42839235,  0.9763484};
        auto gotOutput = output->readMap<float>();
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 4, 0.01)) {
            MNN_ERROR("ErfTest test failed!\n");
            return false;
        }
        return true;
    }
};
class ErfcTest : public MNNTestCase {
public:
    virtual ~ErfcTest() = default;
    virtual bool run() {
        auto input = _Input({4,}, NCHW);
        input->setName("input_tensor");
        // set input data
        const float inpudata[] = {-1.2, 0., 0.4, 1.6};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 4 * sizeof(float));
        input->unMap();
        auto output = _Erfc(input);
        const std::vector<float> expectedOutput = {1.910314, 1., 0.57160765, 0.02365161};
        auto gotOutput = output->readMap<float>();
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 4, 0.01)) {
            MNN_ERROR("ErfcTest test failed!\n");
            return false;
        }
        return true;
    }
};
class ErfinvTest : public MNNTestCase {
public:
    virtual ~ErfinvTest() = default;
    virtual bool run() {
        auto input = _Input({4,}, NCHW);
        input->setName("input_tensor");
        // set input data
        const float inpudata[] = {0, 0.4, 0.6, 0.9};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 4 * sizeof(float));
        input->unMap();
        auto output = _Erfinv(input);
        const std::vector<float> expectedOutput = {0., 0.37080714, 0.5951161 , 1.1630871};
        auto gotOutput = output->readMap<float>();
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 4, 0.01)) {
            MNN_ERROR("ErfinvTest test failed!\n");
            return false;
        }
        return true;
    }
};
class Expm1Test : public MNNTestCase {
public:
    virtual ~Expm1Test() = default;
    virtual bool run() {
        auto input = _Input({4,}, NCHW);
        input->setName("input_tensor");
        // set input data
        const float inpudata[] = {-1.2, 0, 0.4, 1.6};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 4 * sizeof(float));
        input->unMap();
        auto output = _Expm1(input);
        const std::vector<float> expectedOutput = {-0.6988058 ,  0.,  0.49182472,  3.9530325};
        auto gotOutput = output->readMap<float>();
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 4, 0.01)) {
            MNN_ERROR("Expm1Test test failed!\n");
            return false;
        }
        return true;
    }
};
class SinhTest : public MNNTestCase {
public:
    virtual ~SinhTest() = default;
    virtual bool run() {
        auto input = _Input({4,}, NCHW);
        input->setName("input_tensor");
        // set input data
        const float inpudata[] = {-1.2, 0, 0.4, 1.6};
        auto inputPtr          = input->writeMap<float>();
        memcpy(inputPtr, inpudata, 4 * sizeof(float));
        input->unMap();
        auto output = _Sinh(input);
        const std::vector<float> expectedOutput = {-1.5094614, 0., 0.41075233, 2.375568};
        auto gotOutput = output->readMap<float>();
        if (!checkVector<float>(gotOutput, expectedOutput.data(), 4, 0.01)) {
            MNN_ERROR("SinhTest test failed!\n");
            return false;
        }
        return true;
    }
};
MNNTestSuiteRegister(AbsTest, "op/unary/abs");
MNNTestSuiteRegister(NegativeTest, "op/unary/negative");
MNNTestSuiteRegister(FloorTest, "op/unary/floor");
MNNTestSuiteRegister(CeilTest, "op/unary/ceil");
MNNTestSuiteRegister(SquareTest, "op/unary/square");
MNNTestSuiteRegister(SqrtTest, "op/unary/sqrt");
MNNTestSuiteRegister(RsqrtTest, "op/unary/rsqrt");
MNNTestSuiteRegister(ExpTest, "op/unary/exp");
MNNTestSuiteRegister(LogTest, "op/unary/log");
MNNTestSuiteRegister(SinTest, "op/unary/sin");
MNNTestSuiteRegister(CosTest, "op/unary/cos");
MNNTestSuiteRegister(TanTest, "op/unary/tan");
MNNTestSuiteRegister(AsinTest, "op/unary/asin");
MNNTestSuiteRegister(AcosTest, "op/unary/acos");
MNNTestSuiteRegister(AtanTest, "op/unary/atan");
MNNTestSuiteRegister(ReciprocalTest, "op/unary/reciprocal");
MNNTestSuiteRegister(Log1PTest, "op/unary/log1p");
MNNTestSuiteRegister(TanhTest, "op/unary/tanh");
MNNTestSuiteRegister(SigmoidTest, "op/unary/sigmoid");
MNNTestSuiteRegister(AcoshTest, "op/unary/acosh");
MNNTestSuiteRegister(AsinhTest, "op/unary/asinh");
MNNTestSuiteRegister(AtanhTest, "op/unary/atanh");
MNNTestSuiteRegister(RoundTest, "op/unary/round");
MNNTestSuiteRegister(SignTest, "op/unary/sign");
MNNTestSuiteRegister(CoshTest, "op/unary/cosh");
MNNTestSuiteRegister(ErfTest, "op/unary/erf");
MNNTestSuiteRegister(ErfcTest, "op/unary/erfc");
MNNTestSuiteRegister(ErfinvTest, "op/unary/erfinv");
MNNTestSuiteRegister(Expm1Test, "op/unary/expm1");
MNNTestSuiteRegister(SinhTest, "op/unary/sinh");
