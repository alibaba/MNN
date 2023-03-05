//
//  NNAPIUnary.cpp
//  MNN
//
//  Created by MNN on 2022/09/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NNAPIUnary.hpp"

namespace MNN {


NNAPIUnary::NNAPIUnary(MNN::Backend *b, const MNN::Op *op, const std::vector<Tensor *> &inputs, const std::vector<MNN::Tensor *> &outputs) : NNAPICommonExecution(b, op) {
}

ErrorCode NNAPIUnary::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(inputs.size() == 1 && outputs.size() == 1);
    std::map<UnaryOpOperation, int> unary_map {
        {UnaryOpOperation_ABS, ANEURALNETWORKS_ABS},
        {UnaryOpOperation_EXP, ANEURALNETWORKS_EXP},
        {UnaryOpOperation_SQRT, ANEURALNETWORKS_SQRT},
        {UnaryOpOperation_RSQRT, ANEURALNETWORKS_RSQRT},
        {UnaryOpOperation_LOG, ANEURALNETWORKS_LOG},
        {UnaryOpOperation_RECIPROCAL, -1},
        {UnaryOpOperation_SIN, ANEURALNETWORKS_SIN},
        {UnaryOpOperation_ASIN, -1},
        {UnaryOpOperation_SINH, -1},
        {UnaryOpOperation_ASINH, -1},
        {UnaryOpOperation_COS, -1},
        {UnaryOpOperation_ACOS, -1},
        {UnaryOpOperation_COSH, -1},
        {UnaryOpOperation_ACOSH, -1},
        {UnaryOpOperation_TAN, -1},
        {UnaryOpOperation_ATAN, -1},
        {UnaryOpOperation_TANH, ANEURALNETWORKS_TANH},
        {UnaryOpOperation_ATANH, -1},
        {UnaryOpOperation_ERF, -1},
        {UnaryOpOperation_CEIL, -1},
        {UnaryOpOperation_FLOOR, ANEURALNETWORKS_FLOOR},
        {UnaryOpOperation_ROUND, -1},
        {UnaryOpOperation_SIGN, -1},
        {UnaryOpOperation_SIGMOID, -1},
        {UnaryOpOperation_LOG1P, -1},
        {UnaryOpOperation_SQUARE, -1},
        {UnaryOpOperation_NEG, ANEURALNETWORKS_NEG},
        {UnaryOpOperation_HARDSWISH, ANEURALNETWORKS_HARD_SWISH},
        {UnaryOpOperation_GELU, -1},
        {UnaryOpOperation_GELU_STANDARD, -1},
        {UnaryOpOperation_EXPM1, -1},
        {UnaryOpOperation_ERFC, -1},
        {UnaryOpOperation_BNLL, -1},
        {UnaryOpOperation_ERFINV, -1}
    };
    auto opType = mOp->main_as_UnaryOp()->opType();
    auto iter = unary_map.find(opType);
    if (iter == unary_map.end() || iter->second < 0) {
        MNN_ERROR("NNAPI Unary not support %s\n", MNN::EnumNameUnaryOpOperation(opType));
        return NOT_SUPPORT;
    }
    return buildOperation(iter->second, getTensorIdxs(inputs), getTensorIdxs(outputs));
}

REGISTER_NNAPI_OP_CREATOR(NNAPIUnary, OpType_UnaryOp)
} // namespace MNN
