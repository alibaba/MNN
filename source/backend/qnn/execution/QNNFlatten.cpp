//
//  QNNFlatten.cpp
//  MNN
//
//  Created by MNN on b'2025/04/10'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "QNNFlatten.hpp"

namespace MNN {
namespace QNN {
#ifdef ENABLE_QNN_ONLINE_FINALIZE
ErrorCode QNNFlatten::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    Tensor::DimensionType inputDimType = inputs[0]->getDimensionType();
    Tensor::DimensionType outputDimType = outputs[0]->getDimensionType();

    MNN_ASSERT(inputDimType == outputDimType);
    std::vector<uint32_t> inputQnnShape = getNHWCShape(inputs[0]);
    std::vector<uint32_t> outputQnnShape = getNHWCShape(outputs[0]);
    if(TensorUtils::getDescribe(inputs[0])->dimensionFormat == MNN_DATA_FORMAT_NC4HW4) {
        if(inputQnnShape[inputs[0]->dimensions() - 1] != outputQnnShape[outputs[0]->dimensions() - 1]){
            this->ReshapeTranspose(inputs, outputs);
            return NO_ERROR;
        }
    }
    
    mNodeType = "Reshape";
    // this->addNodeCommon(inputs, outputs);
    this->addNodeCommonReshape("Reshape", *(mBackend->getNativeTensor(inputs[0])), *(mBackend->getNativeTensor(outputs[0])));

    return NO_ERROR;
}

void QNNFlatten::ReshapeTranspose(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    if(inputs[0]->shape().size() == 4 && outputs[0]->shape().size() == 3) {
        std::vector<int> outputShape = outputs[0]->shape();
        Qnn_DataType_t dataType = mBackend->getNativeTensor(inputs[0])->v1.dataType;
        this->createStageTensor("reshape_temp", dataType, outputShape, inputs[0]);
        // first reshape
        {
            mNodeType = "Reshape";
            std::string name = mNodeName + "_0_reshape";
            mParams.clear();
            mInputs.clear();
            mOutputs.clear();
            mInputs.push_back(*(mBackend->getNativeTensor(inputs[0])));
            mOutputs.push_back(*(mTempTensorWrappers[0]->getNativeTensor()));
            mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
        }
        // second transpose
        {
            mNodeType = "Transpose";
            std::string name = mNodeName + "_1_transpose";
            mParams.clear();
            mInputs.clear();
            mOutputs.clear();
            std::vector<uint32_t> permData{0, 2, 1};
            this->createParamTensor("perm", QNN_DATATYPE_UINT_32, {(uint32_t)3}, (void *)permData.data(), "_1_transpose");
            mParams.push_back(*(mParamTensorWrappers.back()->getNativeParam()));
            mInputs.push_back(*(mTempTensorWrappers[0]->getNativeTensor()));
            mOutputs.push_back(*(mBackend->getNativeTensor(outputs[0])));
            mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
        }
    } else {
        std::vector<int> inputShape = inputs[0]->shape();
        Qnn_DataType_t dataType = mBackend->getNativeTensor(inputs[0])->v1.dataType;
        this->createStageTensor("reshape_temp", dataType, inputShape, inputs[0]);
        // first transpose
        {
            mNodeType = "Transpose";
            std::string name = mNodeName + "_0_transpose";
            mParams.clear();
            mInputs.clear();
            mOutputs.clear();
            std::vector<uint32_t> permData{0, 2, 1};
            this->createParamTensor("perm", QNN_DATATYPE_UINT_32, {(uint32_t)3}, (void *)permData.data(), "_0_transpose");
            mParams.push_back(*(mParamTensorWrappers.back()->getNativeParam()));
            mInputs.push_back(*(mBackend->getNativeTensor(inputs[0])));
            mOutputs.push_back(*(mTempTensorWrappers[0]->getNativeTensor()));
            mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
        }
        // second reshape
        {
            mNodeType = "Reshape";
            std::string name = mNodeName + "_1_reshape";
            mParams.clear();
            mInputs.clear();
            mOutputs.clear();
            mInputs.push_back(*(mTempTensorWrappers[0]->getNativeTensor()));
            mOutputs.push_back(*(mBackend->getNativeTensor(outputs[0])));
            mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
        }
    }
}

class QNNFlattenCreator : public QnnBackend::Creator {
public:
    virtual QNNCommonExecution * onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op,
                                Backend* backend) const override {
        return new QNNFlatten(backend, op);
    }
};

REGISTER_QNN_OP_CREATOR(QNNFlattenCreator, OpType_Squeeze)
REGISTER_QNN_OP_CREATOR(QNNFlattenCreator, OpType_Unsqueeze)
REGISTER_QNN_OP_CREATOR(QNNFlattenCreator, OpType_Reshape)
REGISTER_QNN_OP_CREATOR(QNNFlattenCreator, OpType_Flatten)
#endif
} // end namespace QNN
} // end namespace MNN
