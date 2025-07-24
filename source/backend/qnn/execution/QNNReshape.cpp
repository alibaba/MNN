//
//  QNNReshape.cpp
//  MNN
//
//  Created by MNN on b'2025/04/10'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "QNNReshape.hpp"

namespace MNN {
namespace QNN {

ErrorCode QNNReshape::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mNodeType = "Reshape";

    mInputs.push_back(*(mBackend->getNativeTensor(inputs[0])));
    mOutputs.push_back(*(mBackend->getNativeTensor(outputs[0])));
    mBackend->addNodeToGraph(mOpConfigVersion, mNodeName.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    return NO_ERROR;
}


class QNNReshapeCreator : public QnnBackend::Creator {
public:
    virtual QNNCommonExecution * onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op,
                                Backend* backend) const override {
        return new QNNReshape(backend, op);
    }
};

// REGISTER_QNN_OP_CREATOR(QNNReshapeCreator, OpType_Reshape)
// REGISTER_QNN_OP_CREATOR(QNNReshapeCreator, OpType_Squeeze)
// REGISTER_QNN_OP_CREATOR(QNNReshapeCreator, OpType_Unsqueeze)
REGISTER_QNN_OP_CREATOR(QNNReshapeCreator, OpType_ConvertTensor)

} // end namespace QNN
} // end namespace MNN
