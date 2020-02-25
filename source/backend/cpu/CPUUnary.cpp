//
//  CPUUnary.cpp
//  MNN
//
//  Created by MNN on 2018/08/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/CPUUnary.hpp"
#include <cmath>
#include "backend/cpu/CPUBackend.hpp"
#include "core/Macro.h"
#include "core/Concurrency.h"
#include <vector>
#include <limits>

namespace MNN {
CPUUnary::CPUUnary(Backend *b, UnaryOpOperation type) : MNN::Execution(b), mType(type) {
    // nothing to do
}

ErrorCode CPUUnary::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(1 == outputs.size());
    auto dtype = inputs[0]->getType();
    MNN_ASSERT(dtype == halide_type_of<float>() || dtype == halide_type_of<int32_t>());
    return NO_ERROR;
}

template <typename Func, typename T>
static ErrorCode _unaryOp(void* inputPtr, void* outputPtr, int elementSize, Backend* bn) {
    Func f;
    auto backend = [bn]() {
        return bn;
    };
    const T *inputData = (T*)inputPtr;
    T *outputData      = (T *)outputPtr;
    auto numberThread = ((CPUBackend*)bn)->threadNumber();
    MNN_CONCURRENCY_BEGIN(tId, numberThread) {
        for (int i=tId; i<elementSize; i+=numberThread) {
            outputData[i] = f(inputData[i]);
        }
    }
    MNN_CONCURRENCY_END();
    return NO_ERROR;
}

template <typename T>
struct UnarySquare : std::unary_function<T, T> {
    T operator()(const T &x) const {
        return x * x;
    }
};

template <typename T>
struct UnaryRsqrt : std::unary_function<T, T> {
    T operator()(const T &x) const {
        return 1.f / sqrt(x);
    }
};

template <typename T>
struct UnarySqrt : std::unary_function<T, T> {
    T operator()(const T &x) const {
        return sqrt(x);
    }
};

template <typename T>
struct UnaryNeg {
    T operator()(const T &x) const {
        return -x;
    }
};

template <typename T>
struct UnaryExp : std::unary_function<T, T> {
    T operator()(const T &x) const {
        return exp(x);
    }
};

template <typename T>
struct UnaryAbs : std::unary_function<T, T> {
    T operator()(const T &x) const {
        return abs(x);
    }
};

template <typename T>
struct UnaryCeil : std::unary_function<T, T> {
    T operator()(const T &x) const {
        return ceil(x);
    }
};
template <typename T>
struct UnaryRecipocal : std::unary_function<T, T> {
    T operator()(const T &x) const {
        return (T)1 / (x);
    }
};
template <typename T>
struct UnaryLog1p : std::unary_function<T, T> {
    T operator()(const T &x) const {
        return (T)log((T)1 + (x));
    }
};
template <typename T>
struct UnaryLog : std::unary_function<T, T> {
    T operator()(const T &x) const {
        return (T)log((T)(x));
    }
};
template <typename T>
struct UnaryCos : std::unary_function<T, T> {
    T operator()(const T &x) const {
        return (T)cosf((T)(x));
    }
};
template <typename T>
struct UnarySin : std::unary_function<T, T> {
    T operator()(const T &x) const {
        return (T)sinf((T)(x));
    }
};
template <typename T>
struct UnaryTan : std::unary_function<T, T> {
    T operator()(const T &x) const {
        return (T)tanf((T)(x));
    }
};
template <typename T>
struct UnaryATan : std::unary_function<T, T> {
    T operator()(const T &x) const {
        return (T)atanf((T)(x));
    }
};

template <typename T>
struct UnaryFloor : std::unary_function<T, T> {
    T operator()(const T &x) const {
        return (T)floor((T)(x));
    }
};

template <typename T>
struct UnarySign : std::unary_function<T, T> {
    T operator()(const T &x) const {
        if (x > 0) {
            return 1;
        }
        if (x < 0) {
            return -1;
        }
        return 0;
    }
};

template <typename T>
struct UnaryBNLL : std::unary_function<T, T> {
    T operator()(const T &x) const {
        float r = x > 0 ? (x + log(1. + exp(-x))) : log(1. + exp(x));
        return (T)r;
    }
};

template <typename T>
struct UnaryAcosh : std::unary_function<T, T> {
    T operator()(const T &x) const {
        return (T)acoshf((T)(x));
    }
};

template <typename T>
struct UnarySinh : std::unary_function<T, T> {
    T operator()(const T &x) const {
        return (T)sinhf((T)(x));
    }
};

template <typename T>
struct UnaryAsinh : std::unary_function<T, T> {
    T operator()(const T &x) const {
        return (T)asinhf((T)(x));
    }
};

template <typename T>
struct UnaryAtanh : std::unary_function<T, T> {
    T operator()(const T &x) const {
        return (T)atanhf((T)(x));
    }
};
template <typename T>
struct UnaryRound : std::unary_function<T, T> {
    T operator()(const T &x) const {
        return (T)roundf((T)(x));
    }
};

template <typename T>
struct UnaryCosh : std::unary_function<T, T> {
    T operator()(const T &x) const {
        return (T)coshf((T)(x));
    }
};

template <typename T>
T evalPoly(T x, const std::vector<float> kErfTCoefficient) {
    auto poly = 0.0f;
    for (auto c : kErfTCoefficient) {
        poly = poly * x + c;
    }
    return poly;
}

template <typename T>
T erfImpl(T x) {
    // Coefficients for by erf(f32), from Cephes. tensorflow
    static const std::vector<float> kErfTCoefficient {
            +7.853861353153693E-5, -8.010193625184903E-4, +5.188327685732524E-3,
            -2.685381193529856E-2, +1.128358514861418E-1, -3.761262582423300E-1,
            +1.128379165726710E+0,
    };
    return x * evalPoly(x * x, kErfTCoefficient);
}

template <typename T>
T erfcImpl(T x) {
    // Coefficients for erfc(f32), from Cephes. tensorflow
    const double kMaxlog = 88.72283905206835;
    // erfc(x) = exp(-x^2) P(1/x^2), 1 < x < 2
    static const std::vector<float> kErfcPCoefficient{
            +2.326819970068386E-2, -1.387039388740657E-1, +3.687424674597105E-1,
            -5.824733027278666E-1, +6.210004621745983E-1, -4.944515323274145E-1,
            +3.404879937665872E-1, -2.741127028184656E-1, +5.638259427386472E-1,
    };
    // erfc(x) = exp(-x^2) R(1/x^2), 2 <= x < kMaxlog
    static const std::vector<float> kErfcRCoefficient{
            -1.047766399936249E+1, +1.297719955372516E+1, -7.495518717768503E+0,
            +2.921019019210786E+0, -1.015265279202700E+0, +4.218463358204948E-1,
            -2.820767439740514E-1, +5.641895067754075E-1,
    };
    float absX = fabsf(x);
    float z = expf(-x * x);
    float q = 1.0 / absX;
    float y = q * q;
    float p;
    if (absX < 2.0f) {
        p = evalPoly(y, kErfcPCoefficient);
    } else {
        p = evalPoly(y, kErfcRCoefficient);
    }
    y = z * q * p;
    float yClamp;
    if (z < -kMaxlog) {
        yClamp = 0.0f;
    } else {
        yClamp = y;
    }
    if (x < 0) {
        return T(2.0f - yClamp);
    } else {
        return T(yClamp);
    }
}

template <typename T>
struct UnaryErf : std::unary_function<T, T> {
    T operator()(const T &x) const {
        if (abs(x) < T(1.)) {
            return erfImpl(x);
        } else {
            return T(1.) - erfcImpl(x);
        }
    }
};

template <typename T>
struct UnaryErfc : std::unary_function<T, T> {
    T operator()(const T &x) const {
        if (abs(x) > T(1.)) {
            return erfcImpl(x);
        } else {
            return T(1.) - erfImpl(x);
        }
    }
};

template <typename T>
struct UnaryErfinv : std::unary_function<T, T> {
    // referenced from tensorflow
    const int kDegree = 9;
    const std::vector<float> w_less_than_5_constants = {
            2.81022636e-08f,  3.43273939e-07f, -3.5233877e-06f,
            -4.39150654e-06f, 0.00021858087f,  -0.00125372503f,
            -0.00417768164f,  0.246640727f,    1.50140941f};
    const std::vector<float> w_greater_than_5_constants = {
            -0.000200214257f, 0.000100950558f, 0.00134934322f,
            -0.00367342844f,  0.00573950773f,  -0.0076224613f,
            0.00943887047f,   1.00167406f,     2.83297682f};

    T operator()(const T &x) const {
        // Compute logarithm of (1+arg) using log1p(arg) which is more precise than
        // log(1+arg) when arg is close to zero. For more details, see
        // https://en.cppreference.com/w/cpp/numeric/math/log1p
        auto w = -log1p(-x * x);
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
        auto result = p * x;
        if (fabsf(fabsf(x) - 1) < 1e-8) {
            return std::numeric_limits<float>::infinity();
        } else {
            return result;
        }
    }
};

template <typename T>
struct UnaryExpm1 : std::unary_function<T, T> {
    T operator()(const T &x) const {
        return (T)expm1((T)(x));
    }
};

template <typename T>
struct UnaryAsin : std::unary_function<T, T> {
    T operator()(const T &x) const {
        return (T)asin((T)(x));
    }
};

template <typename T>
struct UnaryAcos : std::unary_function<T, T> {
    T operator()(const T &x) const {
        return (T)acos((T)(x));
    }
};

ErrorCode CPUUnary::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input  = inputs[0];
    auto output = outputs[0];
    auto dtype  = input->getType().code;

    if (dtype == halide_type_int) {
        switch (mType) {
            case UnaryOpOperation_ABS:
                return _unaryOp<UnaryAbs<int32_t>, int32_t>(input->host<void>(), output->host<void>(), input->elementSize(), backend());
            case UnaryOpOperation_NEG:
                return _unaryOp<UnaryNeg<int32_t>, int32_t>(input->host<void>(), output->host<void>(), input->elementSize(), backend());
            case UnaryOpOperation_SQUARE:
                return _unaryOp<UnarySquare<int32_t>, int32_t>(input->host<void>(), output->host<void>(), input->elementSize(), backend());
            default:
                MNN_ERROR("Int-Unary not support %d\n", mType);
                break;
        }
        return NO_ERROR;
    }
    switch (mType) {
        case UnaryOpOperation_SQUARE:
            return _unaryOp<UnarySquare<float>, float>(input->host<void>(), output->host<void>(), input->elementSize(), backend());
        case UnaryOpOperation_RSQRT:
            return _unaryOp<UnaryRsqrt<float>, float>(input->host<void>(), output->host<void>(), input->elementSize(), backend());
        case UnaryOpOperation_NEG:
            return _unaryOp<UnaryNeg<float>, float>(input->host<void>(), output->host<void>(), input->elementSize(), backend());
        case UnaryOpOperation_EXP:
            return _unaryOp<UnaryExp<float>, float>(input->host<void>(), output->host<void>(), input->elementSize(), backend());
        case UnaryOpOperation_COS:
            return _unaryOp<UnaryCos<float>, float>(input->host<void>(), output->host<void>(), input->elementSize(), backend());
        case UnaryOpOperation_SIN:
            return _unaryOp<UnarySin<float>, float>(input->host<void>(), output->host<void>(), input->elementSize(), backend());
        case UnaryOpOperation_TAN:
            return _unaryOp<UnaryTan<float>, float>(input->host<void>(), output->host<void>(), input->elementSize(), backend());
        case UnaryOpOperation_ATAN:
            return _unaryOp<UnaryATan<float>, float>(input->host<void>(), output->host<void>(), input->elementSize(), backend());
        case UnaryOpOperation_SQRT:
            return _unaryOp<UnarySqrt<float>, float>(input->host<void>(), output->host<void>(), input->elementSize(), backend());
        case UnaryOpOperation_ABS:
            return _unaryOp<UnaryAbs<float>, float>(input->host<void>(), output->host<void>(), input->elementSize(), backend());
        case UnaryOpOperation_CEIL:
            return _unaryOp<UnaryCeil<float>, float>(input->host<void>(), output->host<void>(), input->elementSize(), backend());
        case UnaryOpOperation_RECIPROCAL:
            return _unaryOp<UnaryRecipocal<float>, float>(input->host<void>(), output->host<void>(), input->elementSize(), backend());
        case UnaryOpOperation_LOG1P:
            return _unaryOp<UnaryLog1p<float>, float>(input->host<void>(), output->host<void>(), input->elementSize(), backend());
        case UnaryOpOperation_LOG:
            return _unaryOp<UnaryLog<float>, float>(input->host<void>(), output->host<void>(), input->elementSize(), backend());
        case UnaryOpOperation_FLOOR:
            return _unaryOp<UnaryFloor<float>, float>(input->host<void>(), output->host<void>(), input->elementSize(), backend());
        case UnaryOpOperation_BNLL:
            return _unaryOp<UnaryBNLL<float>, float>(input->host<void>(), output->host<void>(), input->elementSize(), backend());
        case UnaryOpOperation_ACOSH:
            return _unaryOp<UnaryAcosh<float>, float>(input->host<void>(), output->host<void>(), input->elementSize(), backend());
        case UnaryOpOperation_SINH:
            return _unaryOp<UnarySinh<float>, float>(input->host<void>(), output->host<void>(), input->elementSize(), backend());
        case UnaryOpOperation_ASINH:
            return _unaryOp<UnaryAsinh<float>, float>(input->host<void>(), output->host<void>(), input->elementSize(), backend());
        case UnaryOpOperation_ATANH:
            return _unaryOp<UnaryAtanh<float>, float>(input->host<void>(), output->host<void>(), input->elementSize(), backend());
        case UnaryOpOperation_SIGN:
            return _unaryOp<UnarySign<float>, float>(input->host<void>(), output->host<void>(), input->elementSize(), backend());
        case UnaryOpOperation_ROUND:
            return _unaryOp<UnaryRound<float>, float>(input->host<void>(), output->host<void>(), input->elementSize(), backend());
        case UnaryOpOperation_COSH:
            return _unaryOp<UnaryCosh<float>, float>(input->host<void>(), output->host<void>(), input->elementSize(), backend());
        case UnaryOpOperation_ERF:
            return _unaryOp<UnaryErf<float>, float>(input->host<void>(), output->host<void>(), input->elementSize(), backend());
        case UnaryOpOperation_ERFC:
            return _unaryOp<UnaryErfc<float>, float>(input->host<void>(), output->host<void>(), input->elementSize(), backend());
        case UnaryOpOperation_ERFINV:
            return _unaryOp<UnaryErfinv<float>, float>(input->host<void>(), output->host<void>(), input->elementSize(), backend());
        case UnaryOpOperation_EXPM1:
            return _unaryOp<UnaryExpm1<float>, float>(input->host<void>(), output->host<void>(), input->elementSize(), backend());
        case UnaryOpOperation_ASIN:
            return _unaryOp<UnaryAsin<float>, float>(input->host<void>(), output->host<void>(), input->elementSize(), backend());
        case UnaryOpOperation_ACOS:
            return _unaryOp<UnaryAcos<float>, float>(input->host<void>(), output->host<void>(), input->elementSize(), backend());
        default:
            MNN_ASSERT(false);
            break;
    }

    return NO_ERROR;
}

class CPUUnaryCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        return new CPUUnary(backend, op->main_as_UnaryOp()->opType());
    }
};

REGISTER_CPU_OP_CREATOR(CPUUnaryCreator, OpType_UnaryOp);

} // namespace MNN
