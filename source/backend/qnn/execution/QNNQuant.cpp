//
//  QNNQuant.cpp
//  MNN
//
//  Created by MNN on b'2025/05/29'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "QNNQuant.hpp"

namespace MNN {
namespace QNN {

ErrorCode QNNQuant::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    this->createStageTensor("Cast", QNN_DATATYPE_FLOAT_32, getNHWCShape(outputs[0]));
     // Stage one  fp16 -> fp32
    {
        mNodeType = "Cast";
        std::string name = mNodeName + "_Cast";
    
        mInputs.push_back(*(mBackend->getNativeTensor(inputs[0]))); // input
        mOutputs.push_back(*(mTempTensorWrappers[0]->getNativeTensor())); // stage tensor
        mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    }

    // Stage two  fp32 -> int8
    {
        mNodeType.clear();
        mParams.clear();
        mInputs.clear();
        mOutputs.clear();
        mNodeType = "Quantize";
        std::string name = mNodeName;
    
        mInputs.push_back(*(mTempTensorWrappers[0]->getNativeTensor())); // stage tensor
        mOutputs.push_back(*(mBackend->getNativeTensor(outputs[0]))); // output
        mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    }
    return NO_ERROR;
}


class QNNQuantCreator : public QnnBackend::Creator {
public:
    virtual QNNCommonExecution * onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op,
                                Backend* backend) const override {
        return new QNNQuant(backend, op);
    }
};

ErrorCode QNNDeQuant::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
     // Stage one  int8 -> fp16
    {
        mNodeType.clear();
        mParams.clear();
        mInputs.clear();
        mOutputs.clear();
        mNodeType = "Dequantize";
        std::string name = mNodeName;
    
        mInputs.push_back(*(mBackend->getNativeTensor(inputs[0]))); // input
        mOutputs.push_back(*(mBackend->getNativeTensor(outputs[0]))); // output
        mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    }
    return NO_ERROR;
}


class QNNDeQuantCreator : public QnnBackend::Creator {
public:
    virtual QNNCommonExecution * onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op,
                                Backend* backend) const override {
        return new QNNDeQuant(backend, op);
    }
};

REGISTER_QNN_OP_CREATOR(QNNQuantCreator, OpType_FloatToInt8)
REGISTER_QNN_OP_CREATOR(QNNDeQuantCreator, OpType_Int8ToFloat)

} // end namespace QNN
} // end namespace MNN
