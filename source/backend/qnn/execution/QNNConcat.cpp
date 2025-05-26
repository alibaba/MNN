//
//  QNNConcat.cpp
//  MNN
//
//  Created by MNN on b'2025/04/10'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "QNNConcat.hpp"

namespace MNN {
namespace QNN {

ErrorCode QNNConcat::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mNodeType = "Concat";
    Tensor * output = outputs[0];
    if (output->dimensions() > 5) {
        MNN_QNN_NOT_SUPPORT_NATIVE_CONSTRAINT;
    }

    int axis = mOp->main_as_Axis()->axis();
    axis = getNHWCAxis(axis, outputs[0]->dimensions(), TensorUtils::getDimType(outputs[0]));
    this->createParamScalar("axis", (uint32_t)axis);

    this->addNodeCommon(inputs, outputs);

    return NO_ERROR;
}


class QNNConcatCreator : public QnnBackend::Creator {
public:
    virtual QNNCommonExecution * onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op,
                                Backend* backend) const override {
        return new QNNConcat(backend, op);
    }
};

REGISTER_QNN_OP_CREATOR(QNNConcatCreator, OpType_Concat)

} // end namespace QNN
} // end namespace MNN

