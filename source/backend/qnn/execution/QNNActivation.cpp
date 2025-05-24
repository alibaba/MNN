//
//  QNNActivation.cpp
//  MNN
//
//  Created by MNN on b'2025/04/10'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "QNNActivation.hpp"

namespace MNN {
namespace QNN {

ErrorCode QNNActivation::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto opType = mOp->type();
    switch (opType) {
        case OpType_ReLU:
            mNodeType = "Relu";
            break;
        case OpType_ReLU6:
            mNodeType = "ReluMinMax";
            this->createParamScalar("min_value", mOp->main_as_Relu6()->minValue());
            this->createParamScalar("max_value", mOp->main_as_Relu6()->maxValue());
            break;
        case OpType_Sigmoid:
            mNodeType = "Sigmoid";
            break;
        case OpType_ELU:
            mNodeType = "Elu";
            this->createParamScalar("alpha", mOp->main_as_ELU()->alpha());
            break;
        default:
            MNN_QNN_NOT_SUPPORT_SPECIAL_CASE;
    }

    this->addNodeCommon(inputs, outputs);

    return NO_ERROR;
}


class QNNActivationCreator : public QnnBackend::Creator {
public:
    virtual QNNCommonExecution * onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op,
                                Backend* backend) const override {
        return new QNNActivation(backend, op);
    }
};

REGISTER_QNN_OP_CREATOR(QNNActivationCreator, OpType_ReLU)
REGISTER_QNN_OP_CREATOR(QNNActivationCreator, OpType_ReLU6)
REGISTER_QNN_OP_CREATOR(QNNActivationCreator, OpType_Sigmoid)
REGISTER_QNN_OP_CREATOR(QNNActivationCreator, OpType_ELU)

} // end namespace QNN
} // end namespace MNN
