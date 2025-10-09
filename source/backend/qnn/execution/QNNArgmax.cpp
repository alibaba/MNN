//
//  QNNArgmax.cpp
//  MNN
//
//  Created by MNN on b'2025/04/10'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "QNNArgmax.hpp"

namespace MNN {
namespace QNN {
#ifdef ENABLE_QNN_ONLINE_FINALIZE

ErrorCode QNNArgmax::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    if (mOp->type() == OpType_ArgMin) {
        mNodeType = "Argmin";
    } else {
        mNodeType = "Argmax";
    }

    if (inputs[0]->dimensions() > 4) {
        MNN_QNN_NOT_SUPPORT_NATIVE_CONSTRAINT;
    }

    int axis = mOp->main_as_ArgMax()->axis();
    axis = getNHWCAxis(axis, inputs[0]->dimensions(), TensorUtils::getDimType(inputs[0]));

    this->createParamScalar("axis", (uint32_t)axis);

    this->addNodeCommon(inputs, outputs);

    return NO_ERROR;
}


class QNNArgmaxCreator : public QnnBackend::Creator {
public:
    virtual QNNCommonExecution * onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op,
                                Backend* backend) const override {
        return new QNNArgmax(backend, op);
    }
};

REGISTER_QNN_OP_CREATOR(QNNArgmaxCreator, OpType_ArgMax)
REGISTER_QNN_OP_CREATOR(QNNArgmaxCreator, OpType_ArgMin)
#endif
} // end namespace QNN
} // end namespace MNN
