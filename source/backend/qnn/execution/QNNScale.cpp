//
//  QNNScale.cpp
//  MNN
//
//  Created by MNN on b'2025/04/10'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "QNNScale.hpp"

namespace MNN {
namespace QNN {

ErrorCode QNNScale::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    // create temp tensors
    {
        auto scaleParam = mOp->main_as_Scale();
        int channel = inputs[0]->channel();
        MNN_ASSERT(channel == scaleParam->channels() && channel == scaleParam->scaleData()->size() && channel == scaleParam->biasData()->size());

        Qnn_DataType_t dataType         = mBackend->getNativeTensor(inputs[0])->v1.dataType;

        this->createStaticFloatTensor("weight", dataType, {(uint32_t)channel}, scaleParam->scaleData()->data());
        this->createStaticFloatTensor("bias", dataType, {(uint32_t)channel}, scaleParam->biasData()->data());
        this->createStageTensor("Stage", dataType, getNHWCShape(inputs[0]));
    }

    // add nodes
    this->mulWeight(inputs[0]);
    this->addBias(outputs[0]);

    return NO_ERROR;
}

void QNNScale::mulWeight(Tensor * input) {
    mNodeType = "ElementWiseMultiply";
    std::string name = mNodeName + "_mul";
    mParams.clear();
    mInputs.clear();
    mOutputs.clear();

    mInputs.push_back(*(mBackend->getNativeTensor(input)));
    mInputs.push_back(*(mTempTensorWrappers[0]->getNativeTensor()));

    mOutputs.push_back(*(mTempTensorWrappers[2]->getNativeTensor()));

    mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
}


void QNNScale::addBias(Tensor * output) {
    mNodeType = "ElementWiseAdd";
    std::string name = mNodeName + "_add";
    mParams.clear();
    mInputs.clear();
    mOutputs.clear();

    mInputs.push_back(*(mTempTensorWrappers[2]->getNativeTensor()));
    mInputs.push_back(*(mTempTensorWrappers[1]->getNativeTensor()));

    mOutputs.push_back(*(mBackend->getNativeTensor(output)));

    mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
}


class QNNScaleCreator : public QnnBackend::Creator {
public:
    virtual QNNCommonExecution * onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op,
                                Backend* backend) const override {
        return new QNNScale(backend, op);
    }
};

REGISTER_QNN_OP_CREATOR(QNNScaleCreator, OpType_Scale)

} // end namespace QNN
} // end namespace MNN
