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
#ifdef ENABLE_QNN_ONLINE_FINALIZE

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
        mNeedQuantDequant = dataType != QNN_DATATYPE_FLOAT_16 && dataType != QNN_DATATYPE_FLOAT_32;
        if(mNeedQuantDequant){
            Qnn_DataType_t tempDataType = QNN_DATATYPE_FLOAT_32;
            if(mBackend->getUseFP16()){
                tempDataType = QNN_DATATYPE_FLOAT_16;
            }
            this->createStaticFloatTensor("weight", tempDataType, {(uint32_t)channel}, mWeightData.data());
            this->createStaticFloatTensor("bias", tempDataType, {(uint32_t)channel}, mBiasData.data());
            this->createStageTensor("Stage", tempDataType, getNHWCShape(inputs[0]));
            this->createStageTensor("Stage_dequantize_input", tempDataType, getNHWCShape(inputs[0]));
            this->createStageTensor("Stage_add_output", tempDataType, getNHWCShape(outputs[0]));
            if(mBackend->getUseFP16()){
                this->createStageTensor("Stage_cast_output", QNN_DATATYPE_FLOAT_32, getNHWCShape(outputs[0]));
            }
        }else{
            this->createStaticFloatTensor("weight", dataType, {(uint32_t)channel}, mWeightData.data());
            this->createStaticFloatTensor("bias", dataType, {(uint32_t)channel}, mBiasData.data());
            this->createStageTensor("Stage", dataType, getNHWCShape(inputs[0]));
        }
    }

    // add nodes
    this->mulWeight(inputs[0]);
    this->addBias(outputs[0]);

    return NO_ERROR;
}

void QNNScale::mulWeight(Tensor * input) {
    Qnn_DataType_t dataType = mBackend->getNativeTensor(input)->v1.dataType;
    // need dequantize to float16
    if(mNeedQuantDequant){
        mNodeType.clear();
        mParams.clear();
        mInputs.clear();
        mOutputs.clear();
        mNodeType = "Dequantize";
        std::string name = mNodeName + "_Dequantize";
    
        mInputs.push_back(*(mBackend->getNativeTensor(input))); // input
        mOutputs.push_back(*(mTempTensorWrappers[3]->getNativeTensor())); //Stage_dequantize_input
        mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    }
    {
        mNodeType.clear();
        mParams.clear();
        mInputs.clear();
        mOutputs.clear();
        mNodeType = "ElementWiseMultiply";
        std::string name = mNodeName + "_mul";
        
        if(mNeedQuantDequant){
            mInputs.push_back(*(mTempTensorWrappers[3]->getNativeTensor())); //Stage_dequantize_input
        }else{
            mInputs.push_back(*(mBackend->getNativeTensor(input)));
        }
        mInputs.push_back(*(mTempTensorWrappers[0]->getNativeTensor()));
        
        mOutputs.push_back(*(mTempTensorWrappers[2]->getNativeTensor()));
        
        mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    }
}


void QNNScale::addBias(Tensor * output) {
    Qnn_DataType_t dataType = mBackend->getNativeTensor(output)->v1.dataType;
    {
        mNodeType.clear();
        mParams.clear();
        mInputs.clear();
        mOutputs.clear();
        mNodeType = "ElementWiseAdd";
        std::string name = mNodeName + "_add";
        
        mInputs.push_back(*(mTempTensorWrappers[2]->getNativeTensor()));
        mInputs.push_back(*(mTempTensorWrappers[1]->getNativeTensor()));
        
        if(mNeedQuantDequant){
            mOutputs.push_back(*(mTempTensorWrappers[4]->getNativeTensor())); // Stage_add_output
        }else{
            mOutputs.push_back(*(mBackend->getNativeTensor(output)));
        }
        
        mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    }
    
    // need quantize output
    if(mNeedQuantDequant){
        // Stage one  fp16 -> fp32
        if(mBackend->getUseFP16()){
           mNodeType.clear();
           mParams.clear();
           mInputs.clear();
           mOutputs.clear();
           mNodeType = "Cast";
           std::string name = mNodeName + "_Cast";
       
           mInputs.push_back(*(mTempTensorWrappers[4]->getNativeTensor())); // Stage_add_output
           mOutputs.push_back(*(mTempTensorWrappers[5]->getNativeTensor())); // Stage_cast_output
           mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
       }

       // Stage two  fp32 -> int8
       {
           mNodeType.clear();
           mParams.clear();
           mInputs.clear();
           mOutputs.clear();
           mNodeType = "Quantize";
           std::string name = mNodeName + "_Quantize";
           
           if(mBackend->getUseFP16()){
               mInputs.push_back(*(mTempTensorWrappers[5]->getNativeTensor())); // Stage_cast_output
           }else{
               mInputs.push_back(*(mTempTensorWrappers[4]->getNativeTensor())); // Stage_add_output
           }
           mOutputs.push_back(*(mBackend->getNativeTensor(output))); // output
           mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
       }
    }
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
#endif
} // end namespace QNN
} // end namespace MNN
