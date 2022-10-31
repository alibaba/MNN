//
//  NNAPIBinary.cpp
//  MNN
//
//  Created by MNN on 2022/09/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NNAPIBinary.hpp"

namespace MNN {


NNAPIBinary::NNAPIBinary(MNN::Backend *b, const MNN::Op *op, const std::vector<Tensor *> &inputs, const std::vector<MNN::Tensor *> &outputs) : NNAPICommonExecution(b, op) {
}

ErrorCode NNAPIBinary::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(inputs.size() == 2 && outputs.size() == 1);
    std::map<BinaryOpOperation, int> binary_map {
        {BinaryOpOperation_ADD, ANEURALNETWORKS_ADD},
        {BinaryOpOperation_SUB, ANEURALNETWORKS_SUB},
        {BinaryOpOperation_MUL, ANEURALNETWORKS_MUL},
        {BinaryOpOperation_DIV, ANEURALNETWORKS_DIV}
    };
    BinaryOpOperation binaryType;
    if (mOp->type() == OpType_BinaryOp) {
        binaryType = static_cast<BinaryOpOperation>(mOp->main_as_BinaryOp()->opType());
    } else if (mOp->type() == OpType_Eltwise) {
        auto elemType = mOp->main_as_Eltwise()->type();
        switch (elemType) {
            case EltwiseType_PROD:
                binaryType = BinaryOpOperation_MUL;
                break;
            case EltwiseType_SUM:
                binaryType = BinaryOpOperation_ADD;
                break;
            case EltwiseType_SUB:
                binaryType = BinaryOpOperation_SUB;
                break;
            case EltwiseType_MAXIMUM:
                binaryType = BinaryOpOperation_MAXIMUM;
                break;
        }
    }
    auto iter = binary_map.find(binaryType);
    if (iter == binary_map.end() || iter->second < 0) {
        MNN_ERROR("[NNAPI] Binary not support %s\n", MNN::EnumNameBinaryOpOperation(binaryType));
        return NOT_SUPPORT;
    }
    auto inputIdxs = getTensorIdxs(inputs);
    inputIdxs.push_back(buildScalar(ANEURALNETWORKS_FUSED_NONE));
    return buildOperation(iter->second, inputIdxs, getTensorIdxs(outputs));
}

REGISTER_NNAPI_OP_CREATOR(NNAPIBinary, OpType_BinaryOp)
REGISTER_NNAPI_OP_CREATOR(NNAPIBinary, OpType_Eltwise)
} // namespace MNN
