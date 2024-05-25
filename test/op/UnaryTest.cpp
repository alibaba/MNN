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
#include "MNN_generated.h"
#include "MNNTestSuite.h"
#include "TestUtils.h"
#define MNN_DEFAULT_FLATBUFFER_SIZE 32
using namespace MNN::Express;
using namespace std;
using namespace MNN;

static VARP _UnaryInt8(VARP x, UnaryOpOperation operation, std::vector<int8_t> buffer) {
    flatbuffers::FlatBufferBuilder builder(MNN_DEFAULT_FLATBUFFER_SIZE);
    auto bufferOffset = builder.CreateVector(buffer);
    UnaryOpBuilder parameter(builder);
    parameter.add_tableInt8(bufferOffset);
    parameter.add_opType(operation);
    auto paOffset = parameter.Finish();
    OpBuilder opB(builder);
    opB.add_main(paOffset.Union());
    opB.add_type(OpType_UnaryOp);
    opB.add_main_type(OpParameter_UnaryOp);
    builder.Finish(opB.Finish());
    std::shared_ptr<BufferStorage> extra(new BufferStorage);
    extra->storage = builder.ReleaseRaw(extra->allocated_size, extra->offset);
    return Variable::create(Expr::create(extra, {x}, 1));
}
VARP _SquareInt8(VARP x, float* scale, float* zero) {
    std::vector<int8_t> buffer;
    for (int i = -127; i <= 127; ++i) {
        float fx = (i - zero[0]) * scale[0];
        int qx = roundf((fx * fx) / scale[1]) + zero[1];
        if (qx > 127) {
            qx = 127;
        }
        if (qx < -127) {
            qx = -127;
        }
        buffer.push_back(qx);
    }
    return _UnaryInt8(x, UnaryOpOperation_SQUARE, buffer);
}
VARP _SqrtInt8(VARP x, float* scale, float* zero) {
    std::vector<int8_t> buffer(255, 0);
    for (int i = -127; i <= 127; ++i) {
        double fx = (i - zero[0]) * scale[0];
        if (fx >= 0) {
            int qx = roundf(sqrt(fx) / scale[1]) + zero[1];
            if (qx > 127) {
                qx = 127;
            }
            buffer[i + 127] = qx;
        }
    }
    return _UnaryInt8(x, UnaryOpOperation_SQRT, buffer);
}
VARP _RsqrtInt8(VARP x, float* scale, float* zero)
{
    std::vector<int8_t> buffer(255, 0);
    for (int i = -127; i <= 127; ++i) {
        double fx = (i - zero[0]) * scale[0];
        if (fx >= 0) {
            int qx = 127;
            if (sqrt(fx) != 0) {
                qx = roundf((1.0f / sqrt(fx)) / scale[1]) + zero[1];
            }
            if (qx > 127) {
                qx = 127;
            }
            buffer[i + 127] = qx;
        }
    }
    return _UnaryInt8(x, UnaryOpOperation_RSQRT, buffer);
}
VARP _ExpInt8(VARP x, float* scale, float* zero) {
    std::vector<int8_t> buffer;
    for (int i = -127; i <= 127; ++i) {
        double fx = (i - zero[0]) * scale[0];
        int qx = roundf(exp(fx) / scale[1]) + zero[1];
        if (qx > 127) {
            qx = 127;
        }
        if (qx < -127) {
            qx = -127;
        }
        buffer.push_back(qx);

    }
    return _UnaryInt8(x, UnaryOpOperation_EXP, buffer);
}
VARP _LogInt8(VARP x, float* scale, float* zero) {
    std::vector<int8_t> buffer(255, 0);
    for (int i = -127; i <= 127; ++i) {
        double fx = (i - zero[0]) * scale[0];
        if (fx < 0) {
            continue;
        }
        int qx = roundf(log(fx) / scale[1]) + zero[1];
        if (qx > 127) {
            qx = 127;
        }
        if (qx < -127) {
            qx = -127;
        }
        buffer[i + 127] = qx;

    }
    return _UnaryInt8(x, UnaryOpOperation_LOG, buffer);
}
VARP _SinInt8(VARP x, float* scale, float* zero) {
    std::vector<int8_t> buffer;
    for (int i = -127; i <= 127; ++i) {
        double fx = (i - zero[0]) * scale[0];
        int qx = roundf(sin(fx) / scale[1]) + zero[1];
        if (qx > 127) {
            qx = 127;
        }
        if (qx < -127) {
            qx = -127;
        }
        buffer.push_back(qx);

    }
    return _UnaryInt8(x, UnaryOpOperation_SIN, buffer);
}
VARP _CosInt8(VARP x, float* scale, float* zero) {
    std::vector<int8_t> buffer;
    for (int i = -127; i <= 127; ++i) {
        double fx = (i - zero[0]) * scale[0];
        int qx = roundf(cos(fx) / scale[1]) + zero[1];
        if (qx > 127) {
            qx = 127;
        }
        if (qx < -127) {
            qx = -127;
        }
        buffer.push_back(qx);

    }
    return _UnaryInt8(x, UnaryOpOperation_COS, buffer);
}
VARP _TanInt8(VARP x, float* scale, float* zero) {
    std::vector<int8_t> buffer;
    for (int i = -127; i <= 127; ++i) {
        double fx = (i - zero[0]) * scale[0];
        int qx = roundf(tan(fx) / scale[1]) + zero[1];
        if (qx > 127) {
            qx = 127;
        }
        if (qx < -127) {
            qx = -127;
        }
        buffer.push_back(qx);

    }
    return _UnaryInt8(x, UnaryOpOperation_TAN, buffer);
}
VARP _AsinInt8(VARP x, float* scale, float* zero) {
    std::vector<int8_t> buffer;
    for (int i = -127; i <= 127; ++i) {
        double fx = (i - zero[0]) * scale[0];
        int qx = roundf(asin(fx) / scale[1]) + zero[1];
        if (qx > 127) {
            qx = 127;
        }
        if (qx < -127) {
            qx = -127;
        }
        buffer.push_back(qx);

    }
    return _UnaryInt8(x, UnaryOpOperation_ASIN, buffer);
}
VARP _AcosInt8(VARP x, float* scale, float* zero) {
    std::vector<int8_t> buffer;
    for (int i = -127; i <= 127; ++i) {
        double fx = (i - zero[0]) * scale[0];
        int qx = roundf(acos(fx) / scale[1]) + zero[1];
        if (qx > 127) {
            qx = 127;
        }
        if (qx < -127) {
            qx = -127;
        }
        buffer.push_back(qx);

    }
    return _UnaryInt8(x, UnaryOpOperation_ACOS, buffer);
}
VARP _AcoshInt8(VARP x, float* scale, float* zero) {
    std::vector<int8_t> buffer;
    for (int i = -127; i <= 127; ++i) {
        double fx = (i - zero[0]) * scale[0];
        if (fx < 1) {
            buffer.push_back(0);
            continue;
        }
        float val = acosh(fx);
        int qx = roundf(acosh(fx) / scale[1]) + zero[1];
        if (qx > 127) {
            qx = 127;
        }
        if (qx < -127) {
            qx = -127;
        }
        buffer.push_back(qx);

    }
    return _UnaryInt8(x, UnaryOpOperation_ACOSH, buffer);
}
VARP _AsinhInt8(VARP x, float* scale, float* zero) {
    std::vector<int8_t> buffer;
    for (int i = -127; i <= 127; ++i) {
        double fx = (i - zero[0]) * scale[0];
        int qx = roundf(asinh(fx) / scale[1]) + zero[1];
        if (qx > 127) {
            qx = 127;
        }
        if (qx < -127) {
            qx = -127;
        }
        buffer.push_back(qx);

    }
    return _UnaryInt8(x, UnaryOpOperation_ASINH, buffer);
}
VARP _AtanhInt8(VARP x, float* scale, float* zero) {
    std::vector<int8_t> buffer;
    for (int i = -127; i <= 127; ++i) {
        double fx = (i - zero[0]) * scale[0];
        if (fx <= -1 || fx >= 1) {
            buffer.push_back(0);
            continue;
        }
        int qx = roundf(atanh(fx) / scale[1]) + zero[1];
        if (qx > 127) {
            qx = 127;
        }
        if (qx < -127) {
            qx = -127;
        }
        buffer.push_back(qx);

    }
    return _UnaryInt8(x, UnaryOpOperation_ATANH, buffer);
}
VARP _CoshInt8(VARP x, float* scale, float* zero) {
    std::vector<int8_t> buffer;
    for (int i = -127; i <= 127; ++i) {
        double fx = (i - zero[0]) * scale[0];
        int qx = roundf(cosh(fx) / scale[1]) + zero[1];
        if (qx > 127) {
            qx = 127;
        }
        if (qx < -127) {
            qx = -127;
        }
        buffer.push_back(qx);

    }
    return _UnaryInt8(x, UnaryOpOperation_COSH, buffer);
}
VARP _SinhInt8(VARP x, float* scale, float* zero) {
    std::vector<int8_t> buffer;
    for (int i = -127; i <= 127; ++i) {
        double fx = (i - zero[0]) * scale[0];
        int qx = roundf(sinh(fx) / scale[1]) + zero[1];
        if (qx > 127) {
            qx = 127;
        }
        if (qx < -127) {
            qx = -127;
        }
        buffer.push_back(qx);

    }
    return _UnaryInt8(x, UnaryOpOperation_SINH, buffer);
}
VARP _ErfInt8(VARP x, float* scale, float* zero) {
    std::vector<int8_t> buffer;
    for (int i = -127; i <= 127; ++i) {
        double fx = (i - zero[0]) * scale[0];
        int qx = roundf(erf(fx) / scale[1]) + zero[1];
        if (qx > 127) {
            qx = 127;
        }
        if (qx < -127) {
            qx = -127;
        }
        buffer.push_back(qx);

    }
    return _UnaryInt8(x, UnaryOpOperation_ERF, buffer);
}
VARP _ErfcInt8(VARP x, float* scale, float* zero) {
    std::vector<int8_t> buffer;
    for (int i = -127; i <= 127; ++i) {
        double fx = (i - zero[0]) * scale[0];
        int qx = roundf(erfc(fx) / scale[1]) + zero[1];
        if (qx > 127) {
            qx = 127;
        }
        if (qx < -127) {
            qx = -127;
        }
        buffer.push_back(qx);

    }
    return _UnaryInt8(x, UnaryOpOperation_ERFC, buffer);
}
VARP _ErfinvInt8(VARP x, float* scale, float* zero) {
    std::vector<int8_t> buffer;
    const int kDegree = 9;
    const std::vector<float> w_less_than_5_constants = {
            2.81022636e-08f,  3.43273939e-07f, -3.5233877e-06f,
            -4.39150654e-06f, 0.00021858087f,  -0.00125372503f,
            -0.00417768164f,  0.246640727f,    1.50140941f};
    const std::vector<float> w_greater_than_5_constants = {
            -0.000200214257f, 0.000100950558f, 0.00134934322f,
            -0.00367342844f,  0.00573950773f,  -0.0076224613f,
            0.00943887047f,   1.00167406f,     2.83297682f};
    for (int i = -127; i <= 127; ++i) {
        float fx = (i - zero[0]) * scale[0];
        auto w = -log(-fx * fx + 1);
        bool lt = (w < 5.0);
        auto coefficient = [&](int i) {
            if (lt) {
                return w_less_than_5_constants[i];
            } else {
                return w_greater_than_5_constants[i];
            }
        };
        if (lt) {
            w = w - 2.5;
        } else {
            w = sqrt(w) - 3.0;
        }
        auto p = coefficient(0);
        for (int i = 1; i < kDegree; i++) {
            p = coefficient(i) + p * w;
        }
        auto result = p * fx;
        float val = 0;
        if (fabsf(fabsf(fx) - 1) < 1e-8) {
            val = std::numeric_limits<float>::infinity();
        } else {
            val = result;
        }
        int qx = roundf(val / scale[1]) + zero[1];
        if (qx > 127) {
            qx = 127;
        }
        if (qx < -127) {
            qx = -127;
        }
        buffer.push_back(qx);

    }
    return _UnaryInt8(x, UnaryOpOperation_ERFINV, buffer);
}
VARP _AtanInt8(VARP x, float* scale, float* zero) {
    std::vector<int8_t> buffer;
    for (int i = -127; i <= 127; ++i) {
        double fx = (i - zero[0]) * scale[0];
        int qx = roundf(atan(fx) / scale[1]) + zero[1];
        if (qx > 127) {
            qx = 127;
        }
        if (qx < -127) {
            qx = -127;
        }
        buffer.push_back(qx);

    }
    return _UnaryInt8(x, UnaryOpOperation_ATAN, buffer);
}
VARP _ReciprocalInt8(VARP x, float* scale, float* zero) {
    std::vector<int8_t> buffer;
    for (int i = -127; i <= 127; ++i) {
        float fx = (i - zero[0]) * scale[0];
        if (fx == 0) {
            buffer.push_back(0);
            continue;
        }
        int qx = roundf((1.0f / fx) / scale[1]) + zero[1];
        if (qx > 127) {
            qx = 127;
        }
        if (qx < -127) {
            qx = -127;
        }
        buffer.push_back(qx);

    }
    return _UnaryInt8(x, UnaryOpOperation_RECIPROCAL, buffer);
}
VARP _Log1pInt8(VARP x, float* scale, float* zero) {
    std::vector<int8_t> buffer;
    for (int i = -127; i <= 127; ++i) {
        double fx = (i - zero[0]) * scale[0];
        if (fx + 1 <= 0) {
            buffer.push_back(-127);
            continue;
        }
        int qx = roundf(log(fx + 1) / scale[1]) + zero[1];
        if (qx > 127) {
            qx = 127;
        }
        if (qx < -127) {
            qx = -127;
        }
        buffer.push_back(qx);

    }
    return _UnaryInt8(x, UnaryOpOperation_LOG1P, buffer);
}
VARP _GeluInt8(VARP x, float* scale, float* zero) {
    auto tanhf_poly = [](float value) -> float {
        if (value > 5.0f) {
            return 1.0f;
        } else if (value <= -5.0f) {
            return -1.0f;
        } else {
            float x2 = value * value;
            float a  = value * (135135.0f + x2 * (17325.0f + x2 * (378.0f + x2)));
            float b  = 135135.0f + x2 * (62370.0f + x2 * (3150.0f + x2 * 28.0f));
            return a / b;
        }
    };
    std::vector<int8_t> buffer;
    for (int i = -127; i <= 127; ++i) {
        float fx = (i - zero[0]) * scale[0];
        float temp = 0.044715f * fx * fx * fx;
        temp = 0.79788458f * (temp + fx);
        float val = (1.0f + tanhf_poly(temp)) * fx * 0.5f;
        int qx = roundf(val / scale[1]) + zero[1];
        if (qx > 127) {
            qx = 127;
        }
        if (qx < -127) {
            qx = -127;
        }
        buffer.push_back(qx);

    }
    return _UnaryInt8(x, UnaryOpOperation_GELU, buffer);
}
VARP _TanhInt8(VARP x, float* scale, float* zero) {
    std::vector<int8_t> buffer;
    for (int i = -127; i <= 127; ++i) {
        double fx = (i - zero[0]) * scale[0];
        int qx = roundf(tanh(fx) / scale[1]) + zero[1];
        if (qx > 127) {
            qx = 127;
        }
        if (qx < -127) {
            qx = -127;
        }
        buffer.push_back(qx);
    }
    return _UnaryInt8(x, UnaryOpOperation_TANH, buffer);
}
VARP _SigmoidInt8(VARP x, float* scale, float* zero) {
    std::vector<int8_t> buffer;
    for (int i = -127; i <= 127; ++i) {
        float fx = (i - zero[0]) * scale[0];
        float val = 1.0f / (1 + exp(-fx));
        int qx = roundf(val / scale[1]) + zero[1];
        if (qx > 127) {
            qx = 127;
        }
        if (qx < -127) {
            qx = -127;
        }
        buffer.push_back(qx);

    }
    return _UnaryInt8(x, UnaryOpOperation_SIGMOID, buffer);
}
VARP _Expm1Int8(VARP x, float* scale, float* zero) {
    std::vector<int8_t> buffer;
    for (int i = -127; i <= 127; ++i) {
        double fx = (i - zero[0]) * scale[0];
        int qx = roundf((exp(fx) - 1.0f)) / scale[1] + zero[1];
        if (qx > 127) {
            qx = 127;
        }
        if (qx < -127) {
            qx = -127;
        }
        buffer.push_back(qx);

    }
    return _UnaryInt8(x, UnaryOpOperation_EXPM1, buffer);
}
class UnaryTestCommon : public MNNTestCase {
protected:
    template<typename Tin, typename Tout>
    bool test(VARP (*opFunc)(VARP), string name, Tout threshold, const vector<Tin>& data_in,
              const vector<Tout>& data_out, const vector<int>& shape_in, const vector<int>& shape_out, float* quanScales=nullptr, float* quanZeroPoints=nullptr, VARP (*opFuncInt8)(VARP, float*, float*)=nullptr) {
        int size_in = 1, size_out = 1;
        for (int i = 0; i < shape_in.size(); ++i) {
            size_in *= shape_in[i];
        }
        for (int i = 0; i < shape_out.size(); ++i) {
            size_out *= shape_out[i];
        }

        auto input = _Input(shape_in, NCHW, halide_type_of<Tin>());
        input->setName("input_tensor");
        if (quanScales) {
            input->writeScaleMap(quanScales[0], quanZeroPoints[0]);
        }
        // set input data
        auto ptr_in = input->template writeMap<Tin>();
        memcpy(ptr_in, data_in.data(), size_in * sizeof(Tin));
        input->unMap();
        VARP output;
        if (quanScales && opFuncInt8) {
            output = opFuncInt8(input, quanScales, quanZeroPoints);
        } else {
            output = opFunc(input);
        }

        if (quanScales) {
            output->writeScaleMap(quanScales[1], quanZeroPoints[1]);
        }
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

        if (!checkVectorByRelativeError<Tout>(gotOutput, data_out.data(), size_out, threshold)) {
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
        auto res = test<float, float>(MNN::Express::_Abs, "AbsTest", 0.01,
                    {-1.0, -2.0, 3.0, 4.0, -1.0, -2.0, 3.0, 4.0}, {1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0},
                    {8}, {8});
        return res && test<int32_t, int32_t>(MNN::Express::_Abs, "AbsTest", 0,
                                         {-1, -2, 3, 4, -1, -2, 3, 4}, {1, 2, 3, 4, 1, 2, 3, 4},
                                         {8}, {8});
    }
};
class NegativeTest : public UnaryTestCommon {
public:
    virtual ~NegativeTest() = default;
    virtual bool run(int precision) {
        auto res = test<float, float>(_Negative, "NegativeTest", 0.01,
                    {-1.0, -2.0, 3.0, 4.0, -1.0, -2.0, 3.0, 4.0}, {1.0, 2.0, -3.0, -4.0, 1.0, 2.0, -3.0, -4.0},
                    {8}, {8});
        return res && test<int32_t, int32_t>(MNN::Express::_Negative, "NegativeTest", 0,
                                         {-1, -2, 3, 4, -1, -2, 3, 4}, {1, 2, -3, -4, 1, 2, -3, -4},
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
        auto res = test<float, float>(_Square, "SquareTest", 0.01,
                    {-1.0, -2.0, 3.0, 4.0, -1.0, -2.0, 3.0, 4.0}, {1.0, 4.0, 9.0, 16.0, 1.0, 4.0, 9.0, 16.0},
                    {8}, {8});
        return res && test<int32_t, int32_t>(_Square, "SquareTest", 0,
                                         {-1, -2, 3, 4, -1, -2, 3, 4}, {1, 4, 9, 16, 1, 4, 9, 16},
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
        float errorScale = precision <= MNN::BackendConfig::Precision_High ? 1 : 10;
        return test<float, float>(_Sin, "SinTest", 0.01 * errorScale,
                    {0.0, 3.14 / 2.0, 3.14, 3.14 * 3.0 / 2.0}, {0.0, 1.0, 0.0, -1.0},
                    {4}, {4});
    }
};
class CosTest : public UnaryTestCommon {
public:
    virtual ~CosTest() = default;
    virtual bool run(int precision) {
        float errorScale = precision <= MNN::BackendConfig::Precision_High ? 1 : 10;
        return test<float, float>(_Cos, "CosTest", 0.01 * errorScale,
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
        auto res = test<float, float>(_Sign, "SignTest", 0.01,
                    {-1.2, 0., 0.4, 1.6}, {-1., 0., 1., 1.},
                    {4}, {4});
        return res && test<int32_t, int32_t>(_Sign, "SignTest", 0,
                                         {-1, 0, 2, 1}, {-1, 0, 1, 1},
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
                    {-4.914062, -1.1126,  1.5541, -0.9805,  1.5448,  0.1681,  0.5264, -0.6206, -0.1101, 0.3287, -0.0688},
                    {-0, -0.1479,  1.4607, -0.1602,  1.4503,  0.0952,  0.3689, -0.1660, -0.0502, 0.2067, -0.0325},
                    {10}, {10});
    }
};
static VARP _GeluStand(VARP x) {
    flatbuffers::FlatBufferBuilder builder(MNN_DEFAULT_FLATBUFFER_SIZE);
    UnaryOpBuilder parameter(builder);
    parameter.add_opType(UnaryOpOperation_GELU_STANDARD);
    auto paOffset = parameter.Finish();
    OpBuilder opB(builder);
    opB.add_main(paOffset.Union());
    opB.add_type(OpType_UnaryOp);
    opB.add_main_type(OpParameter_UnaryOp);
    builder.Finish(opB.Finish());
    std::shared_ptr<BufferStorage> extra(new BufferStorage);
    extra->storage = builder.ReleaseRaw(extra->allocated_size, extra->offset);
    return Variable::create(Expr::create(extra, {x}, 1));
}
class GeluStandTest : public UnaryTestCommon {
public:
    virtual ~GeluStandTest() = default;
    virtual bool run(int precision) {
        return test<float, float>(_GeluStand, "GeluStandardTest", 0.01,
                    {-4.914062, -1.1126,  1.5541, -0.9805,  1.5448,  0.1681,  0.5264, -0.6206, -0.1101, 0.3287, -0.0688},
                    {-0, -0.1479,  1.4607, -0.1602,  1.4503,  0.0952,  0.3689, -0.1660, -0.0502, 0.2067, -0.0325},
                    {10}, {10});
    }
};
/* Unary Int8 test*/
class AbsTestInt8 : public UnaryTestCommon {
public:
    virtual ~AbsTestInt8() = default;
    virtual bool run(int precision) {
        float scale[2] = {0.0315, 0.0315}, zeros[2] = {1.0, 1.0};
        return test<float, float>(MNN::Express::_Abs, "AbsTestInt8", 0.01,
                    {-1.0, -2.0, 3.0, 4.0, -1.0, -2.0, 3.0, 4.0}, {1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0},
                                  {8}, {8}, scale, zeros);
    }
};
class SignTestInt8 : public UnaryTestCommon {
public:
    virtual ~SignTestInt8() = default;
    virtual bool run(int precision) {
        float scale[2] = {0.0125490196, 1.0}, zeros[2] = {1.0, 0.};
        float inp[5] = {-1.2, 0., 0.4, 1.6, 0.4};
        float oup[5] = {-1., 0., 1., 1., 1.};
        std::vector<float> input(20), output(20);
        for (int i = 0; i < 4; ++i) {
            ::memcpy(input.data() + i * 5, inp, 5 * sizeof(float));
            ::memcpy(output.data() + i * 5, oup, 5 * sizeof(float));
        }
        return test<float, float>(_Sign, "SignTestInt8", 0.01,
                    input, output, {20}, {20}, scale, zeros);
    }
};
class NegativeTestInt8 : public UnaryTestCommon {
public:
    virtual ~NegativeTestInt8() = default;
    virtual bool run(int precision) {
        float scale[2] = {0.0315, 0.0315}, zeros[2] = {1.0, 2.0};
        return test<float, float>(_Negative, "NegativeTestInt8", 0.01,
                    {-1.0, -2.0, 3.0, 4.0, -1.0, -2.0, 3.0, 4.0}, {1.0, 2.0, -3.0, -4.0, 1.0, 2.0, -3.0, -4.0},
                    {8}, {8}, scale, zeros);
    }
};
class SquareTestInt8 : public UnaryTestCommon {
public:
    virtual ~SquareTestInt8() = default;
    virtual bool run(int precision) {
        float scale[2] = {0.0315, 0.131}, zeros[2] = {1.0, 2.0};
        return test<float, float>(_Square, "SquareTestInt8", 0.03,
                    {-1.0, -2.0, 3.0, 4.0, -1.0, -2.0, 3.0, 4.0}, {1.0, 4.0, 9.0, 16.0, 1.0, 4.0, 9.0, 16.0},
                    {8}, {8}, scale, zeros, _SquareInt8);
    }
};
class SqrtTestInt8 : public UnaryTestCommon {
public:
    virtual ~SqrtTestInt8() = default;
    virtual bool run(int precision) {
        float scale[2] = {0.502, 0.063}, zeros[2] = {1.0, 1.0};
        return test<float, float>(_Sqrt, "SqrtTestInt8", 0.01,
                    {1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0},
                    {8}, {8}, scale, zeros, _SqrtInt8);
    }
};
class RsqrtTestInt8 : public UnaryTestCommon {
public:
    virtual ~RsqrtTestInt8() = default;
    virtual bool run(int precision) {
        float scale[2] = {0.502, 0.0112}, zeros[2] = {1.0, 1.0};
        return test<float, float>(_Rsqrt, "RsqrtTestInt8", 0.01,
                    {1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0},
                    {1.0, 1.0 / 2.0, 1.0 / 3.0, 1.0 / 4.0, 1.0 / 5.0, 1.0 / 6.0, 1.0 / 7.0, 1.0 / 8.0},
                    {8}, {8}, scale, zeros, _RsqrtInt8);
    }
};
class ExpTestInt8 : public UnaryTestCommon {
public:
    virtual ~ExpTestInt8() = default;
    virtual bool run(int precision) {
        float scale[2] = {0.032, 0.45}, zeros[2] = {1.0, 1.0};
        return test<float, float>(_Exp, "ExpTestInt8", 0.01,
                    {1.0, 2.0, 3.0, 4.0}, {2.718, 7.389, 20.086, 54.598},
                    {4}, {4}, scale, zeros, _ExpInt8);
    }
};
class LogTestInt8 : public UnaryTestCommon {
public:
    virtual ~LogTestInt8() = default;
    virtual bool run(int precision) {
        float scale[2] = {0.46, 0.031589137243899765}, zeros[2] = {1.0, 1.0};
        return test<float, float>(_Log, "LogTestInt8", 0.01,
                    {2.718, 7.389, 20.086, 54.598}, {1.0, 2.0, 3.0, 4.0},
                    {4}, {4}, scale, zeros, _LogInt8);
    }
};
class SinTestInt8 : public UnaryTestCommon {
public:
    virtual ~SinTestInt8() = default;
    virtual bool run(int precision) {
        float scale[2] = {0.037233, 0.0097}, zeros[2] = {1.0, 1.0};
        float errorScale = precision <= MNN::BackendConfig::Precision_High ? 1 : 10;
        return test<float, float>(_Sin, "SinTestInt8", 0.01 * errorScale,
                    {0.0, 3.14 / 2.0, 3.14, 3.14 * 3.0 / 2.0}, {0.0, 1.0, 0.0, -1.0},
                    {4}, {4}, scale, zeros, _SinInt8);
    }
};
class CosTestInt8 : public UnaryTestCommon {
public:
    virtual ~CosTestInt8() = default;
    virtual bool run(int precision) {
        float scale[2] = {0.038, 0.0087}, zeros[2] = {1.0, 1.0};
        float errorScale = precision <= MNN::BackendConfig::Precision_High ? 1 : 10;
        return test<float, float>(_Cos, "CosTestInt8", 0.01 * errorScale,
                    {0.0, 3.14 / 2.0, 3.14, 3.14 * 3.0 / 2.0}, {1.0, 0.0, -1.0, 0.0},
                    {4}, {4}, scale, zeros, _CosInt8);
    }
};
class AtanTestInt8 : public UnaryTestCommon {
public:
    virtual ~AtanTestInt8() = default;
    virtual bool run(int precision) {
        float scale[2] = {0.016, 0.013}, zeros[2] = {1.0, 1.0};
        return test<float, float>(_Atan, "AtanTestInt8", 0.01,
                    {-2.0, -1.0, 0.0, 1.0}, {-1.11, -3.14 / 4.0, 0.0, 3.14 / 4.0},
                    {4}, {4}, scale, zeros, _AtanInt8);
    }
};
class ReciprocalTestInt8 : public UnaryTestCommon {
public:
    virtual ~ReciprocalTestInt8() = default;
    virtual bool run(int precision) {
        float scale[2] = {0.04743, 0.125346}, zeros[2] = {1.0, 1.0};
        return test<float, float>(_Reciprocal, "ReciprocalTestInt8", 0.01,
                    {-2.0, -4.0, 2.0, 4.0}, {-0.5, -0.25, 0.50, 0.25},
                    {4}, {4}, scale, zeros, _ReciprocalInt8);
    }
};
class Log1PTestInt8 : public UnaryTestCommon {
public:
    virtual ~Log1PTestInt8() = default;
    virtual bool run(int precision) {
        float scale[2] = {0.024, 0.011}, zeros[2] = {1.0, 1.0};
        return test<float, float>(_Log1p, "Log1pTestInt8", 0.01,
                    {0.0, 1.0, 2.0, 3.0}, {0.0, 0.69, 1.10, 1.39},
                    {4}, {4}, scale, zeros, _Log1pInt8);
    }
};
class TanhTestInt8 : public UnaryTestCommon {
public:
    virtual ~TanhTestInt8() = default;
    virtual bool run(int precision) {
        float scale[2] = {0.024, 0.008}, zeros[2] = {1.0, 1.0};
        return test<float, float>(_Tanh, "TanhTestInt8", 0.01,
                    {-1.0f, 0.0f, 1.0f, 2.0f, -98.0f, 90.0f}, {-0.76f, 0.0f, 0.76f, 0.96f, -1.0f, 1.0f},
                    {6}, {6}, scale, zeros, _TanhInt8);
    }
};
class SigmoidTestInt8 : public UnaryTestCommon {
public:
    virtual ~SigmoidTestInt8() = default;
    virtual bool run(int precision) {
        int size = 15;
        float scale[2] = {0.03162, 0.003956}, zeros[2] = {1.0, 1.0};
        std::vector<float> data_in(size), data_out(size);
        for (int i = 0; i < size; ++i) {
            data_in[i] = 0.25 * i - 4;
            data_out[i] = 1 / (1 + expf(-data_in[i]));
        }
        return test<float, float>(_Sigmoid, "SigmoidTestInt8", 0.05,
                    data_in, data_out, {size}, {size}, scale, zeros, _SigmoidInt8);
    }
};
class AcoshTestInt8 : public UnaryTestCommon {
public:
    virtual ~AcoshTestInt8() = default;
    virtual bool run(int precision) {
        float scale[2] = {0.032, 0.018}, zeros[2] = {1.0, 1.0};
        return test<float, float>(_Acosh, "AcoshTestInt8", 0.01,
                    {1.0, 2.0, 3.0, 4.0}, {0., 1.3169579, 1.76274717, 2.06343707},
                    {4}, {4}, scale, zeros, _AcoshInt8);
    }
};
class AsinhTestInt8 : public UnaryTestCommon {
public:
    virtual ~AsinhTestInt8() = default;
    virtual bool run(int precision) {
        float scale[2] = {0.0316, 0.0248}, zeros[2] = {1.0, 1.0};
        return test<float, float>(_Asinh, "AsinhTestInt8", 0.01,
                    {1.0, 2.0, 3.0, 4.0}, {0.88137359, 1.44363548, 1.81844646, 2.09471255},
                    {4}, {4}, scale, zeros, _AsinhInt8);
    }
};
class AtanhTestInt8 : public UnaryTestCommon {
public:
    virtual ~AtanhTestInt8() = default;
    virtual bool run(int precision) {
        float scale[2] = {0.00237, 0.002476}, zeros[2] = {1.0, 1.0};
        return test<float, float>(_Atanh, "AtanhTestInt8", 0.05,
                    {0., -0.3, 0.2, 0.3}, {0., -0.3095196, 0.20273255, 0.3095196},
                    {4}, {4}, scale, zeros, _AtanhInt8);
    }
};
class CoshTestInt8 : public UnaryTestCommon {
public:
    virtual ~CoshTestInt8() = default;
    virtual bool run(int precision) {
        float scale[2] = {0.0127, 0.0248}, zeros[2] = {1.0, 1.0};
        return test<float, float>(_Cosh, "CoshTestInt8", 0.01,
                    {-1.2, 0., 0.4, 1.6}, {1.81065557, 1., 1.08107237, 2.57746447},
                    {4}, {4}, scale, zeros, _CoshInt8);
    }
};
class ErfTestInt8 : public UnaryTestCommon {
public:
    virtual ~ErfTestInt8() = default;
    virtual bool run(int precision) {
        float scale[2] = {0.0127, 0.0078}, zeros[2] = {1.0, 1.0};
        return test<float, float>(_Erf, "ErfTestInt8", 0.01,
                    {-1.2, 0., 0.4, 1.6}, {-0.91031396, 0., 0.42839235, 0.9763484},
                    {4}, {4}, scale, zeros, _ErfInt8);
    }
};
class ErfcTestInt8 : public UnaryTestCommon {
public:
    virtual ~ErfcTestInt8() = default;
    virtual bool run(int precision) {
        float scale[2] = {0.0127, 0.02}, zeros[2] = {1.0, 1.0};
        return test<float, float>(_Erfc, "ErfcTestInt8", 0.01,
                    {-1.2, 0., 0.4, 1.6}, {1.910314, 1., 0.57160765, 0.02365161},
                    {4}, {4}, scale, zeros, _ErfcInt8);
    }
};
class ErfinvTestInt8 : public UnaryTestCommon {
public:
    virtual ~ErfinvTestInt8() = default;
    virtual bool run(int precision) {
        float scale[2] = {0.0128, 0.016}, zeros[2] = {1.0, 1.0};
        return test<float, float>(_Erfinv, "ErfinvTestInt8", 0.05,
                    {0, 0.4, 0.6, 0.9}, {0., 0.37080714, 0.5951161, 1.1630871},
                    {4}, {4}, scale, zeros, _ErfinvInt8);
    }
};
class Expm1TestInt8 : public UnaryTestCommon {
public:
    virtual ~Expm1TestInt8() = default;
    float scale[2] = {0.0127, 0.0145}, zeros[2] = {1.0, 1.0};
    virtual bool run(int precision) {
        return test<float, float>(_Expm1, "Expm1TestInt8", 0.01,
                    {-1.2, 0, 0.4, 1.6}, {-0.6988058, 0., 0.49182472, 3.9530325},
                    {4}, {4});
    }
};
class SinhTestInt8 : public UnaryTestCommon {
public:
    virtual ~SinhTestInt8() = default;
    virtual bool run(int precision) {
        float scale[2] = {0.0316, 0.0242}, zeros[2] = {1.0, 1.0};
        return test<float, float>(_Sinh, "SinhTestInt8", 0.01,
                    {-1.2, 0, 0.4, 1.6}, {-1.5094614, 0., 0.41075233, 2.375568},
                    {4}, {4}, scale, zeros, _SinhInt8);
    }
};
class GeluTestInt8 : public UnaryTestCommon {
public:
    virtual ~GeluTestInt8() = default;
    virtual bool run(int precision) {
        float scale[2] = {0.0123, 0.01173}, zeros[2] = {1.0, 1.0};
        return test<float, float>(_Gelu, "GeluTestInt8", 0.01,
                    {-1.1126,  1.5541, -0.9805,  1.5448,  0.1681,  0.5264, -0.6206, -0.1101, 0.3287, -0.0688},
                    {-0.1479,  1.4607, -0.1602,  1.4503,  0.0952,  0.3689, -0.1660, -0.0502, 0.2067, -0.0325},
                    {10}, {10}, scale, zeros, _GeluInt8);
    }
};
class AsinTestInt8 : public UnaryTestCommon {
public:
    virtual ~AsinTestInt8() = default;
    virtual bool run(int precision) {
        float scale[2] = {0.0079, 0.0124}, zeros[2] = {1.0, 1.0};
        return test<float, float>(_Asin, "AsinTestInt8", 0.01,
                    {-1.0, 0.0, 1.0, 0.707}, {-3.14 / 2.0, 0.0, 3.14 / 2.0, 3.14 / 4.0},
                    {4}, {4});
    }
};
class TanTestInt8 : public UnaryTestCommon {
public:
    virtual ~TanTestInt8() = default;
    virtual bool run(int precision) {
        float scale[2] = {3.162, 0.0124}, zeros[2] = {1.0, 1.0};
        return test<float, float>(_Tan, "TanTestInt8", 0.01,
                    {100.0, 200.0, 300.0, 400.0}, {-0.59, -1.79, 45.24, 1.62},
                    {4}, {4});
    }
};
class AcosTestInt8 : public UnaryTestCommon {
public:
    virtual ~AcosTestInt8() = default;
    virtual bool run(int precision) {
        float scale[2] = {0.0079, 0.0248}, zeros[2] = {1.0, 1.0};
        return test<float, float>(_Acos, "AcosTestInt8", 0.01,
                    {-1.0, 0.0, 1.0, 0.707}, {3.14, 1.57, 0.0, 3.14 / 4.0},
                    {4}, {4});
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
MNNTestSuiteRegister(GeluStandTest, "op/unary/gelustandard");
MNNTestSuiteRegister(AbsTestInt8, "op/unary/absInt8");
MNNTestSuiteRegister(SignTestInt8, "op/unary/signInt8");
MNNTestSuiteRegister(NegativeTestInt8, "op/unary/negativeInt8");
MNNTestSuiteRegister(SquareTestInt8, "op/unary/squareInt8");
MNNTestSuiteRegister(SqrtTestInt8, "op/unary/sqrtInt8");
MNNTestSuiteRegister(ExpTestInt8, "op/unary/expInt8");
MNNTestSuiteRegister(LogTestInt8, "op/unary/logInt8");
MNNTestSuiteRegister(SinTestInt8, "op/unary/sinInt8");
MNNTestSuiteRegister(CosTestInt8, "op/unary/cosInt8");
MNNTestSuiteRegister(TanTestInt8, "op/unary/tanInt8");
MNNTestSuiteRegister(AsinTestInt8, "op/unary/asinInt8");
MNNTestSuiteRegister(AcosTestInt8, "op/unary/acosInt8");
MNNTestSuiteRegister(AtanTestInt8, "op/unary/atanInt8");
MNNTestSuiteRegister(ReciprocalTestInt8, "op/unary/reciprocalInt8");
MNNTestSuiteRegister(Log1PTestInt8, "op/unary/log1pInt8");
MNNTestSuiteRegister(TanhTestInt8, "op/unary/tanhInt8");
MNNTestSuiteRegister(SigmoidTestInt8, "op/unary/sigmoidInt8");
MNNTestSuiteRegister(AcoshTestInt8, "op/unary/acoshInt8");
MNNTestSuiteRegister(AsinhTestInt8, "op/unary/asinhInt8");
MNNTestSuiteRegister(AtanhTestInt8, "op/unary/atanhInt8");
MNNTestSuiteRegister(CoshTestInt8, "op/unary/coshInt8");
MNNTestSuiteRegister(ErfTestInt8, "op/unary/erfInt8");
MNNTestSuiteRegister(ErfcTestInt8, "op/unary/erfcInt8");
MNNTestSuiteRegister(ErfinvTestInt8, "op/unary/erfinvInt8");
MNNTestSuiteRegister(Expm1TestInt8, "op/unary/expm1Int8");
MNNTestSuiteRegister(SinhTestInt8, "op/unary/sinhInt8");
MNNTestSuiteRegister(GeluTestInt8, "op/unary/geluInt8");
