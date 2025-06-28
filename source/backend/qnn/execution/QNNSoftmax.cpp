//
//  QNNSoftmax.cpp
//  MNN
//
//  Created by MNN on b'2025/04/10'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "QNNSoftmax.hpp"

namespace MNN {
namespace QNN {

ErrorCode QNNSoftmax::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mNodeType = "Softmax";
    const auto softmaxParam = mOp->main_as_Axis();
    int axis = softmaxParam->axis();
    if (axis < 0) {
        axis = inputs[0]->dimensions() + axis;
    }
    axis = getNHWCAxis(axis, inputs[0]->dimensions(), TensorUtils::getDimType(inputs[0]));
    if (axis != inputs[0]->dimensions() - 1) {
        MNN_QNN_NOT_SUPPORT_NATIVE_CONSTRAINT;
    }

    this->createParamScalar("axis", (uint32_t) axis);

    this->addNodeCommon(inputs, outputs);

    return NO_ERROR;
}


class QNNSoftmaxCreator : public QnnBackend::Creator {
public:
    virtual QNNCommonExecution * onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op,
                                Backend* backend) const override {
        return new QNNSoftmax(backend, op);
    }
};

REGISTER_QNN_OP_CREATOR(QNNSoftmaxCreator, OpType_Softmax)

} // end namespace QNN
} // end namespace MNN

