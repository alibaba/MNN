//
//  CPUUnary.cpp
//  MNN
//
//  Created by MNN on 2018/08/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CPUUnary.hpp"
#include <cmath>
#include "CPUBackend.hpp"
#include "Macro.h"

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
static ErrorCode _unaryOp(Tensor *input, Tensor *output) {
    Func f;

    const T *inputData = input->host<T>();
    T *outputData      = (T *)output->buffer().host;

    auto elementSize = input->elementSize();

    for (int i = 0; i < elementSize; i++) {
        outputData[i] = f(inputData[i]);
    }

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
        return std::exp(x);
    }
};

template <typename T>
struct UnaryAbs : std::unary_function<T, T> {
    T operator()(const T &x) const {
        return std::abs(x);
    }
};

template <typename T>
struct UnaryCeil : std::unary_function<T, T> {
    T operator()(const T &x) const {
        return std::ceil(x);
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
        return (T)std::log((T)1 + (x));
    }
};
template <typename T>
struct UnaryLog : std::unary_function<T, T> {
    T operator()(const T &x) const {
        return (T)std::log((T)(x));
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

ErrorCode CPUUnary::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input  = inputs[0];
    auto output = outputs[0];
    auto dtype  = input->getType().code;

    if (dtype == halide_type_int) {
        switch (mType) {
            case UnaryOpOperation_ABS:
                return _unaryOp<UnaryAbs<int32_t>, int32_t>(input, output);
            case UnaryOpOperation_NEG:
                return _unaryOp<UnaryNeg<int32_t>, int32_t>(input, output);
            case UnaryOpOperation_SQUARE:
                return _unaryOp<UnarySquare<int32_t>, int32_t>(input, output);
            default:
                MNN_ERROR("Int-Unary not support %d\n", mType);
                break;
        }
        return NO_ERROR;
    }
    switch (mType) {
        case UnaryOpOperation_SQUARE:
            return _unaryOp<UnarySquare<float>, float>(input, output);
        case UnaryOpOperation_RSQRT:
            return _unaryOp<UnaryRsqrt<float>, float>(input, output);
        case UnaryOpOperation_NEG:
            return _unaryOp<UnaryNeg<float>, float>(input, output);
        case UnaryOpOperation_EXP:
            return _unaryOp<UnaryExp<float>, float>(input, output);
        case UnaryOpOperation_COS:
            return _unaryOp<UnaryCos<float>, float>(input, output);
        case UnaryOpOperation_SIN:
            return _unaryOp<UnarySin<float>, float>(input, output);
        case UnaryOpOperation_TAN:
            return _unaryOp<UnaryTan<float>, float>(input, output);
        case UnaryOpOperation_ATAN:
            return _unaryOp<UnaryATan<float>, float>(input, output);
        case UnaryOpOperation_SQRT:
            return _unaryOp<UnarySqrt<float>, float>(input, output);
        case UnaryOpOperation_ABS:
            return _unaryOp<UnaryAbs<float>, float>(input, output);
        case UnaryOpOperation_CEIL:
            return _unaryOp<UnaryCeil<float>, float>(input, output);
        case UnaryOpOperation_RECIPROCAL:
            return _unaryOp<UnaryRecipocal<float>, float>(input, output);
        case UnaryOpOperation_LOG1P:
            return _unaryOp<UnaryLog1p<float>, float>(input, output);
        case UnaryOpOperation_LOG:
            return _unaryOp<UnaryLog<float>, float>(input, output);
        case UnaryOpOperation_FLOOR:
            return _unaryOp<UnaryFloor<float>, float>(input, output);
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
