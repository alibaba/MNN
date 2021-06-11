//
//  UnaryTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/01/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <cmath>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include "TestUtils.h"

using namespace MNN::Express;
using namespace std;

class UnaryTestCommon : public MNNTestCase {
protected:
    template<typename Tin, typename Tout>
    bool test(VARP (*opFunc)(VARP), string name, Tout threshold, const vector<Tin>& data_in,
              const vector<Tout>& data_out, const vector<int>& shape_in, const vector<int>& shape_out) {
        int size_in = 1, size_out = 1;
        for (int i = 0; i < shape_in.size(); ++i) {
            size_in *= shape_in[i];
        }
        for (int i = 0; i < shape_out.size(); ++i) {
            size_out *= shape_out[i];
        }

        auto input = _Input(shape_in, NCHW, halide_type_of<Tin>());
        input->setName("input_tensor");
        // set input data
        auto ptr_in = input->template writeMap<Tin>();
        memcpy(ptr_in, data_in.data(), size_in * sizeof(Tin));
        input->unMap();
        auto output = opFunc(input);
        auto gotOutput = output->template readMap<Tout>();

        auto shape_got = output->getInfo()->dim;
        if (shape_got.size() != shape_out.size()) {
            MNN_ERROR("%s shape compute error!\n", name.c_str());
            return false;
        }
        for (int i = 0; i < shape_got.size(); i++) {
            if (shape_got[i] != shape_out[i]) {
                MNN_ERROR("%s shape compute error!\n", name.c_str());
                return false;
            }
        }

        if (!checkVector<Tout>(gotOutput, data_out.data(), size_out, threshold)) {
            MNN_ERROR("%s test failed!\n", name.c_str());
            return false;
        }
        return true;
    }
};

class AbsTest : public UnaryTestCommon {
public:
    virtual ~AbsTest() = default;
    virtual bool run(int precision) {
        return test<float, float>(_Abs, "AbsTest", 0.01,
                    {-1.0, -2.0, 3.0, 4.0, -1.0, -2.0, 3.0, 4.0}, {1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0},
                    {8}, {8});
    }
};
class NegativeTest : public UnaryTestCommon {
public:
    virtual ~NegativeTest() = default;
    virtual bool run(int precision) {
        return test<float, float>(_Negative, "NegativeTest", 0.01,
                    {-1.0, -2.0, 3.0, 4.0, -1.0, -2.0, 3.0, 4.0}, {1.0, 2.0, -3.0, -4.0, 1.0, 2.0, -3.0, -4.0},
                    {8}, {8});
    }
};
class FloorTest : public UnaryTestCommon {
public:
    virtual ~FloorTest() = default;
    virtual bool run(int precision) {
        return test<float, float>(_Floor, "FloorTest", 0.01,
                    {-1.3, -2.6, 3.2, 4.6}, {-2.0, -3.0, 3.0, 4.0},
                    {4}, {4});
    }
};
class CeilTest : public UnaryTestCommon {
public:
    virtual ~CeilTest() = default;
    virtual bool run(int precision) {
        return test<float, float>(_Ceil, "CeilTest", 0.01,
                    {-1.3, -2.6, 3.2, 4.6}, {-1.0, -2.0, 4.0, 5.0},
                    {4}, {4});
    }
};
class SquareTest : public UnaryTestCommon {
public:
    virtual ~SquareTest() = default;
    virtual bool run(int precision) {
        return test<float, float>(_Square, "SquareTest", 0.01,
                    {-1.0, -2.0, 3.0, 4.0, -1.0, -2.0, 3.0, 4.0}, {1.0, 4.0, 9.0, 16.0, 1.0, 4.0, 9.0, 16.0},
                    {8}, {8});
    }
};
class SqrtTest : public UnaryTestCommon {
public:
    virtual ~SqrtTest() = default;
    virtual bool run(int precision) {
        return test<float, float>(_Sqrt, "SqrtTest", 0.01,
                    {1.0, 4.0, 9.0, 16.0, 1.0, 4.0, 9.0, 16.0}, {1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0},
                    {8}, {8});
    }
};
class RsqrtTest : public UnaryTestCommon {
public:
    virtual ~RsqrtTest() = default;
    virtual bool run(int precision) {
        return test<float, float>(_Rsqrt, "RsqrtTest", 0.01,
                    {1.0, 4.0, 9.0, 16.0, 1.0, 4.0, 9.0, 16.0},
                    {1.0, 1.0 / 2.0, 1.0 / 3.0, 1.0 / 4.0, 1.0, 1.0 / 2.0, 1.0 / 3.0, 1.0 / 4.0},
                    {8}, {8});
    }
};
class ExpTest : public UnaryTestCommon {
public:
    virtual ~ExpTest() = default;
    virtual bool run(int precision) {
        return test<float, float>(_Exp, "ExpTest", 0.01,
                    {1.0, 2.0, 3.0, 4.0}, {2.718, 7.389, 20.086, 54.598},
                    {4}, {4});
    }
};
class LogTest : public UnaryTestCommon {
public:
    virtual ~LogTest() = default;
    virtual bool run(int precision) {
        return test<float, float>(_Log, "LogTest", 0.01,
                    {2.718, 7.389, 20.086, 54.598}, {1.0, 2.0, 3.0, 4.0},
                    {4}, {4});
    }
};
class SinTest : public UnaryTestCommon {
public:
    virtual ~SinTest() = default;
    virtual bool run(int precision) {
        return test<float, float>(_Sin, "SinTest", 0.01,
                    {0.0, 3.14 / 2.0, 3.14, 3.14 * 3.0 / 2.0}, {0.0, 1.0, 0.0, -1.0},
                    {4}, {4});
    }
};
class CosTest : public UnaryTestCommon {
public:
    virtual ~CosTest() = default;
    virtual bool run(int precision) {
        return test<float, float>(_Cos, "CosTest", 0.01,
                    {0.0, 3.14 / 2.0, 3.14, 3.14 * 3.0 / 2.0}, {1.0, 0.0, -1.0, 0.0},
                    {4}, {4});
    }
};
class TanTest : public UnaryTestCommon {
public:
    virtual ~TanTest() = default;
    virtual bool run(int precision) {
        return test<float, float>(_Tan, "TanTest", 0.01,
                    {100.0, 200.0, 300.0, 400.0}, {-0.59, -1.79, 45.24, 1.62},
                    {4}, {4});
    }
};
class AsinTest : public UnaryTestCommon {
public:
    virtual ~AsinTest() = default;
    virtual bool run(int precision) {
        return test<float, float>(_Asin, "AsinTest", 0.01,
                    {-1.0, 0.0, 1.0, 0.707}, {-3.14 / 2.0, 0.0, 3.14 / 2.0, 3.14 / 4.0},
                    {4}, {4});
    }
};
class AcosTest : public UnaryTestCommon {
public:
    virtual ~AcosTest() = default;
    virtual bool run(int precision) {
        return test<float, float>(_Acos, "AcosTest", 0.01,
                    {-1.0, 0.0, 1.0, 0.707}, {3.14, 1.57, 0.0, 3.14 / 4.0},
                    {4}, {4});
    }
};
class AtanTest : public UnaryTestCommon {
public:
    virtual ~AtanTest() = default;
    virtual bool run(int precision) {
        return test<float, float>(_Atan, "AtanTest", 0.01,
                    {-2.0, -1.0, 0.0, 1.0}, {-1.11, -3.14 / 4.0, 0.0, 3.14 / 4.0},
                    {4}, {4});
    }
};
class ReciprocalTest : public UnaryTestCommon {
public:
    virtual ~ReciprocalTest() = default;
    virtual bool run(int precision) {
        return test<float, float>(_Reciprocal, "ReciprocalTest", 0.01,
                    {-2.0, -4.0, 2.0, 4.0, -2.0, -4.0, 2.0, 4.0, 4.0}, {-0.5, -0.25, 0.50, 0.25, -0.5, -0.25, 0.50, 0.25, 0.25},
                    {9}, {9});
    }
};
class Log1PTest : public UnaryTestCommon {
public:
    virtual ~Log1PTest() = default;
    virtual bool run(int precision) {
        return test<float, float>(_Log1p, "Log1pTest", 0.01,
                    {0.0, 1.0, 2.0, 3.0}, {0.0, 0.69, 1.10, 1.39},
                    {4}, {4});
    }
};
class TanhTest : public UnaryTestCommon {
public:
    virtual ~TanhTest() = default;
    virtual bool run(int precision) {
        return test<float, float>(_Tanh, "TanhTest", 0.01,
                    {-1.0f, 0.0f, 1.0f, 2.0f, -98.0f, 90.0f}, {-0.76f, 0.0f, 0.76f, 0.96f, -1.0f, 1.0f},
                    {6}, {6});
    }
};
class SigmoidTest : public UnaryTestCommon {
public:
    virtual ~SigmoidTest() = default;
    virtual bool run(int precision) {
        int size = 32;
        std::vector<float> data_in(size), data_out(size);
        for (int i = 0; i < size; ++i) {
            data_in[i] = 0.25 * i - 4;
            data_out[i] = 1 / (1 + expf(-data_in[i]));
        }
        return test<float, float>(_Sigmoid, "SigmoidTest", 0.01,
                    data_in, data_out, {size}, {size});
    }
};
class AcoshTest : public UnaryTestCommon {
public:
    virtual ~AcoshTest() = default;
    virtual bool run(int precision) {
        return test<float, float>(_Acosh, "AcoshTest", 0.01,
                    {1.0, 2.0, 3.0, 4.0}, {0., 1.3169579, 1.76274717, 2.06343707},
                    {4}, {4});
    }
};
class AsinhTest : public UnaryTestCommon {
public:
    virtual ~AsinhTest() = default;
    virtual bool run(int precision) {
        return test<float, float>(_Asinh, "AsinhTest", 0.01,
                    {1.0, 2.0, 3.0, 4.0}, {0.88137359, 1.44363548, 1.81844646, 2.09471255},
                    {4}, {4});
    }
};
class AtanhTest : public UnaryTestCommon {
public:
    virtual ~AtanhTest() = default;
    virtual bool run(int precision) {
        return test<float, float>(_Atanh, "AtanhTest", 0.01,
                    {0., 0.1, 0.2, 0.3}, {0., 0.10033535, 0.20273255, 0.3095196},
                    {4}, {4});
    }
};
class RoundTest : public UnaryTestCommon {
public:
    virtual ~RoundTest() = default;
    virtual bool run(int precision) {
        return test<float, float>(_Round, "RoundTest", 0.01,
                    {-1.2, -0.6, 0.4, 1.6}, {-1., -1., 0., 2.},
                    {4}, {4});
    }
};
class SignTest : public UnaryTestCommon {
public:
    virtual ~SignTest() = default;
    virtual bool run(int precision) {
        return test<float, float>(_Sign, "SignTest", 0.01,
                    {-1.2, 0., 0.4, 1.6}, {-1., 0., 1., 1.},
                    {4}, {4});
    }
};
class CoshTest : public UnaryTestCommon {
public:
    virtual ~CoshTest() = default;
    virtual bool run(int precision) {
        return test<float, float>(_Cosh, "CoshTest", 0.01,
                    {-1.2, 0., 0.4, 1.6}, {1.81065557, 1., 1.08107237, 2.57746447},
                    {4}, {4});
    }
};
class ErfTest : public UnaryTestCommon {
public:
    virtual ~ErfTest() = default;
    virtual bool run(int precision) {
        return test<float, float>(_Erf, "ErfTest", 0.01,
                    {-1.2, 0., 0.4, 1.6}, {-0.91031396, 0., 0.42839235, 0.9763484},
                    {4}, {4});
    }
};
class ErfcTest : public UnaryTestCommon {
public:
    virtual ~ErfcTest() = default;
    virtual bool run(int precision) {
        return test<float, float>(_Erfc, "ErfcTest", 0.01,
                    {-1.2, 0., 0.4, 1.6}, {1.910314, 1., 0.57160765, 0.02365161},
                    {4}, {4});
    }
};
class ErfinvTest : public UnaryTestCommon {
public:
    virtual ~ErfinvTest() = default;
    virtual bool run(int precision) {
        return test<float, float>(_Erfinv, "ErfinvTest", 0.01,
                    {0, 0.4, 0.6, 0.9}, {0., 0.37080714, 0.5951161, 1.1630871},
                    {4}, {4});
    }
};
class Expm1Test : public UnaryTestCommon {
public:
    virtual ~Expm1Test() = default;
    virtual bool run(int precision) {
        return test<float, float>(_Expm1, "Expm1Test", 0.01,
                    {-1.2, 0, 0.4, 1.6}, {-0.6988058, 0., 0.49182472, 3.9530325},
                    {4}, {4});
    }
};
class SinhTest : public UnaryTestCommon {
public:
    virtual ~SinhTest() = default;
    virtual bool run(int precision) {
        return test<float, float>(_Sinh, "SinhTest", 0.01,
                    {-1.2, 0, 0.4, 1.6}, {-1.5094614, 0., 0.41075233, 2.375568},
                    {4}, {4});
    }
};
class GeluTest : public UnaryTestCommon {
public:
    virtual ~GeluTest() = default;
    virtual bool run(int precision) {
        return test<float, float>(_Gelu, "GeluTest", 0.01,
                    {-1.1126,  1.5541, -0.9805,  1.5448,  0.1681,  0.5264, -0.6206, -0.1101, 0.3287, -0.0688},
                    {-0.1479,  1.4607, -0.1602,  1.4503,  0.0952,  0.3689, -0.1660, -0.0502, 0.2067, -0.0325},
                    {10}, {10});
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
MNNTestSuiteRegister(GeluTest, "op/unary/gelu");
