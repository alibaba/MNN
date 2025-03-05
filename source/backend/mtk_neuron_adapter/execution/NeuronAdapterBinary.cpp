//
//  NeuronAdapterBinary.cpp
//  MNN
//
//  Created by MNN on 2022/09/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NeuronAdapterBinary.hpp"

namespace MNN {


NeuronAdapterBinary::NeuronAdapterBinary(MNN::Backend *b, const MNN::Op *op, const std::vector<Tensor *> &inputs, const std::vector<MNN::Tensor *> &outputs) : NeuronAdapterCommonExecution(b, op) {
}

ErrorCode NeuronAdapterBinary::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(inputs.size() == 2 && outputs.size() == 1);
    std::map<BinaryOpOperation, int> binary_map {
        {BinaryOpOperation_ADD, NEURON_ADD},
        {BinaryOpOperation_SUB, NEURON_SUB},
        {BinaryOpOperation_MUL, NEURON_MUL},
        {BinaryOpOperation_DIV, NEURON_DIV},
        {BinaryOpOperation_POW, NEURON_POW},
        {BinaryOpOperation_REALDIV, NEURON_DIV},
        {BinaryOpOperation_MINIMUM, NEURON_MINIMUM},
        {BinaryOpOperation_MAXIMUM, NEURON_MAXIMUM},
        {BinaryOpOperation_GREATER, NEURON_GREATER},
        {BinaryOpOperation_GREATER_EQUAL, NEURON_GREATER_EQUAL},
        {BinaryOpOperation_LESS, NEURON_LESS},
        {BinaryOpOperation_FLOORDIV, -1},
        {BinaryOpOperation_SquaredDifference, -1},
        {BinaryOpOperation_LESS_EQUAL, NEURON_LESS_EQUAL},
        {BinaryOpOperation_FLOORMOD, -1},
        {BinaryOpOperation_EQUAL, NEURON_EQUAL},
        {BinaryOpOperation_MOD, -1},
        {BinaryOpOperation_ATAN2, -1},
        {BinaryOpOperation_LOGICALOR, NEURON_LOGICAL_OR},
        {BinaryOpOperation_NOTEQUAL, NEURON_NOT_EQUAL},
        {BinaryOpOperation_BITWISE_AND, -1},
        {BinaryOpOperation_BITWISE_OR, -1},
        {BinaryOpOperation_BITWISE_XOR, -1},
        {BinaryOpOperation_LOGICALXOR, -1},
        {BinaryOpOperation_LEFTSHIFT, -1},
        {BinaryOpOperation_RIGHTSHIFT, -1}
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
        MNN_ERROR("[NeuronAdapter] Binary not support %s\n", MNN::EnumNameBinaryOpOperation(binaryType));
        return NOT_SUPPORT;
    }
    if (TensorUtils::getDescribe(outputs[0])->quantAttr.get()) {
        outputs[0]->buffer().type = halide_type_of<int8_t>();
    }
    auto inputIdxs = getTensorIdxs(inputs);
    NeuronAdapterFuseCode fusecode = NEURON_FUSED_NONE;
    if (mOp->main_as_BinaryOp()->activationType()) {
        fusecode = NEURON_FUSED_RELU;
    }
    inputIdxs.push_back(buildScalar(fusecode));
    if(iter->second == -1) {
        MNN_PRINT("[NeuronAdapter] unsupported binary op type: %s\n", MNN::EnumNameBinaryOpOperation(binaryType));
        return NOT_SUPPORT;;
    }
    return buildOperation((NeuronOperationType)iter->second, inputIdxs, getTensorIdxs(outputs));
}

REGISTER_NeuronAdapter_OP_CREATOR(NeuronAdapterBinary, OpType_BinaryOp)
REGISTER_NeuronAdapter_OP_CREATOR(NeuronAdapterBinary, OpType_Eltwise)
} // namespace MNN
