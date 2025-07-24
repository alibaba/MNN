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

QNNScale::QNNScale(Backend *backend, const Op *op) : QNNCommonExecution(backend, op) {
    auto scaleParam = mOp->main_as_Scale();
    uint32_t paramSize = scaleParam->scaleData()->size();

    mWeightData.resize(paramSize);
    mBiasData.resize(paramSize);
    ::memcpy(mWeightData.data(), scaleParam->scaleData()->data(), scaleParam->scaleData()->size() * sizeof(float));
    ::memcpy(mBiasData.data(), scaleParam->biasData()->data(), scaleParam->scaleData()->size() * sizeof(float));
}

ErrorCode QNNScale::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    // create temp tensors
    {
        int channel = inputs[0]->channel();
        MNN_ASSERT(channel == mWeightData.size());

        Qnn_DataType_t dataType = mBackend->getNativeTensor(inputs[0])->v1.dataType;

        this->createStaticFloatTensor("weight", dataType, {(uint32_t)channel}, mWeightData.data());
        this->createStaticFloatTensor("bias", dataType, {(uint32_t)channel}, mBiasData.data());
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

ErrorCode QNNScale::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    std::string nodeNameBase = "Scale";
    nodeNameBase += "_";
    std::string inputTag = "I_";
    std::string outputTag = "O_";

    for (int i = 0; i < inputs.size(); i++) {
        inputTag += std::to_string(mBackend->getTensorIdx(inputs[i]));
        inputTag += "_";
    }

    for (int j = 0; j < outputs.size() - 1; j++) {
        outputTag += std::to_string(mBackend->getTensorIdx(outputs[j]));
        outputTag += "_";
    }
    outputTag += std::to_string(mBackend->getTensorIdx(outputs[outputs.size() - 1]));

    mNodeName = nodeNameBase + inputTag + outputTag;

    ErrorCode result = this->onEncode(inputs, outputs);
    if (result != NO_ERROR) {
        return result;
    }

    this->clean();

    return NO_ERROR;
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
