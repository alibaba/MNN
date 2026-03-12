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
#ifdef ENABLE_QNN_ONLINE_FINALIZE

ErrorCode QNNConcat::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    int axis;
    if (mOp->type() == OpType_Concat) {
        mNodeType = "Concat";
        axis = mOp->main_as_Axis()->axis();
    } else if (mOp->type() == OpType_Pack) {
        mNodeType = "Pack";
        axis = mOp->main_as_PackParam()->axis();
    } else if (mOp->type() == OpType_Unpack) {
        mNodeType = "UnPack";
        axis = mOp->main_as_Axis()->axis();
    }
    Tensor * output = outputs[0];

    if (inputs.size() == 2 && inputs[0]->elementSize() == 0) {
        this->addNodeCommonReshape("Reshape", *(mBackend->getNativeTensor(inputs[1])), *(mBackend->getNativeTensor(outputs[0])));
        return NO_ERROR;
    }

    int dim = outputs[0]->dimensions();
    if (axis < 0) {
        axis = dim + axis;
    }
    MNN_ASSERT(axis >= 0 && axis < dim);
    this->createParamScalar("axis", (uint32_t)axis);

    this->addNodeCommon(inputs, outputs);

    return NO_ERROR;
}


class QNNConcatCreator : public QnnBackend::Creator {
public:
    virtual QNNCommonExecution * onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op,
                                Backend* backend) const override {
        // MNN_PRINT("MNN_QNN: Checking Concat. Name %s. Input Num is %d.\n", op->name()->c_str(), (int) inputs.size());
        // for (int i = 0; i < inputs.size(); i++) {
        //     MNN_PRINT("Input%d elementSize is %d.\n", i, inputs[i]->elementSize());
        // }
        if (outputs[0]->dimensions() > 5) {
            MNN_ERROR("QNN Don't support dimension > 5 for concat / pack\n");
            return nullptr;
        }
        return new QNNConcat(backend, op);
    }
};

REGISTER_QNN_OP_CREATOR(QNNConcatCreator, OpType_Concat)
REGISTER_QNN_OP_CREATOR(QNNConcatCreator, OpType_Pack)
REGISTER_QNN_OP_CREATOR(QNNConcatCreator, OpType_Unpack)
#endif
} // end namespace QNN
} // end namespace MNN

