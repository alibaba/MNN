//
//  QNNBinary.cpp
//  MNN
//
//  Created by MNN on b'2025/04/10'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "QNNBinary.hpp"

namespace MNN {
namespace QNN {

ErrorCode QNNBinary::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(inputs.size() == 2 && outputs.size() == 1);

    std::map<BinaryOpOperation, std::string> binaryMap {
        {BinaryOpOperation_ADD, "ElementWiseAdd"},
        {BinaryOpOperation_SUB, "ElementWiseSubtract"},
        {BinaryOpOperation_MUL, "ElementWiseMultiply"},
        {BinaryOpOperation_DIV, "ElementWiseDivide"},
        {BinaryOpOperation_POW, "ElementWisePower"},
        {BinaryOpOperation_REALDIV, "ElementWiseDivide"},
        {BinaryOpOperation_MINIMUM, "ElementWiseMinimum"},
        {BinaryOpOperation_MAXIMUM, "ElementWiseMaximum"},
        {BinaryOpOperation_GREATER, "ElementWiseGreater"},
        {BinaryOpOperation_GREATER_EQUAL, "ElementWiseGreaterEqual"},
        {BinaryOpOperation_LESS, "ElementWiseLess"},
        {BinaryOpOperation_FLOORDIV, ""},
        {BinaryOpOperation_SquaredDifference, ""},
        {BinaryOpOperation_LESS_EQUAL, "ElementWiseLessEqual"},
        {BinaryOpOperation_FLOORMOD, ""},
        {BinaryOpOperation_EQUAL, "ElementWiseEqual"},
        {BinaryOpOperation_MOD, ""},
        {BinaryOpOperation_ATAN2, ""},
        {BinaryOpOperation_LOGICALOR, "ElementWiseOr"},
        {BinaryOpOperation_NOTEQUAL, "ElementWiseNotEqual"},
        {BinaryOpOperation_BITWISE_AND, ""},
        {BinaryOpOperation_BITWISE_OR, ""},
        {BinaryOpOperation_BITWISE_XOR, ""},
        {BinaryOpOperation_LOGICALXOR, ""},
        {BinaryOpOperation_LEFTSHIFT, ""},
        {BinaryOpOperation_RIGHTSHIFT, ""}
    };

    BinaryOpOperation binaryType;
    if (mOp->type() == OpType_BinaryOp) {
        binaryType = static_cast<BinaryOpOperation>(mOp->main_as_BinaryOp()->opType());
    } else {
        auto elewiseType = mOp->main_as_Eltwise()->type();
        switch (elewiseType) {
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
            default:
                MNN_QNN_NOT_SUPPORT_SPECIAL_CASE;
        }
    }
    auto iter = binaryMap.find(binaryType);
    if (iter == binaryMap.end() || iter->second.empty()) {
        MNN_QNN_NOT_SUPPORT_SPECIAL_CASE;
    }
    mNodeType = iter->second;

    // if (TensorUtils::getDescribe(outputs[0])->quantAttr.get()) {
    //     outputs[0]->buffer().type = halide_type_of<int8_t>();
    // }

    this->addNodeCommon(inputs, outputs);

    return NO_ERROR;
}


class QNNBinaryCreator : public QnnBackend::Creator {
public:
    virtual QNNCommonExecution * onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op,
                                Backend* backend) const override {
        return new QNNBinary(backend, op);
    }
};

REGISTER_QNN_OP_CREATOR(QNNBinaryCreator, OpType_BinaryOp)
REGISTER_QNN_OP_CREATOR(QNNBinaryCreator, OpType_Eltwise)

} // end namespace QNN
} // end namespace MNN
