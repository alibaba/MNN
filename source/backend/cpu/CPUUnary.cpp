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
    // we only support floats now
    MNN_ASSERT(inputs[0]->buffer().type.code == halide_type_float && inputs[0]->buffer().type.bits == 32);

    return NO_ERROR;
}

template <typename Func>
static ErrorCode _unaryOp(Tensor *input, Tensor *output) {
    Func f;

    const float *inputData = input->host<float>();
    float *outputData      = (float *)output->buffer().host;

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

ErrorCode CPUUnary::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input  = inputs[0];
    auto output = outputs[0];

    switch (mType) {
        case UnaryOpOperation_SQUARE:
            return _unaryOp<UnarySquare<float>>(input, output);

        case UnaryOpOperation_RSQRT:
            return _unaryOp<UnaryRsqrt<float>>(input, output);

        case UnaryOpOperation_NEG:
            return _unaryOp<UnaryNeg<float>>(input, output);

        case UnaryOpOperation_EXP:
            return _unaryOp<UnaryExp<float>>(input, output);

        case UnaryOpOperation_SQRT:
            return _unaryOp<UnarySqrt<float>>(input, output);

        case UnaryOpOperation_ABS:
            return _unaryOp<UnaryAbs<float>>(input, output);
        case UnaryOpOperation_CEIL:
            return _unaryOp<UnaryCeil<float>>(input, output);
        case UnaryOpOperation_RECIPROCAL:
            return _unaryOp<UnaryRecipocal<float>>(input, output);
        case UnaryOpOperation_LOG1P:
            return _unaryOp<UnaryLog1p<float>>(input, output);
        case UnaryOpOperation_LOG:
            return _unaryOp<UnaryLog<float>>(input, output);
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
