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
    std::vector<uint32_t> inputShape = getNHWCShape(inputs[0]);
    std::vector<uint32_t> outputShape = getNHWCShape(outputs[0]);
    int inputDim = inputs[0]->shape().size();
    int outputDim = outputs[0]->shape().size();
    std::vector<uint32_t> inputReshape(inputShape);
    std::vector<uint32_t> outputReshape(outputShape);
    std::vector<uint32_t> inputPerm(inputDim, 0);
    std::vector<uint32_t> outputPerm(outputDim, 0);
    inputReshape[0] = inputShape[0];
    outputReshape[0] = outputShape[0];
    bool permuteInput = false;
    bool permuteOutput = false;
    int inputTempIndex, outputTempIndex;
    int tempNum = 0;

    if (inputDim > 2) {
        permuteInput = true;
        for (int i = 1; i < inputDim - 1; ++i) {
            inputPerm[i + 1] = i;
            inputReshape[i + 1] = inputShape[i];
        }
        inputPerm[1] = inputDim - 1;
        inputReshape[1] = inputShape[inputDim - 1];
        Qnn_DataType_t dataType = mBackend->getNativeTensor(inputs[0])->v1.dataType;
        this->createStageTensor("permute_input", dataType, inputReshape, inputs[0]);
        inputTempIndex = tempNum;
        tempNum++;
    }
    if (outputDim > 2) {
        permuteOutput = true;
        for (int i = 1; i < outputDim - 1; ++i) {
            outputPerm[i] = i + 1;
            outputReshape[i + 1] = outputShape[i];
        }
        outputPerm[outputDim - 1] = 1;
        outputReshape[1] = outputShape[outputDim - 1];
        Qnn_DataType_t dataType = mBackend->getNativeTensor(outputs[0])->v1.dataType;
        this->createStageTensor("permute_output", dataType, outputReshape, outputs[0]);
        outputTempIndex = tempNum;
        tempNum++;
    }

    // nhwc -> nchw
    if (permuteInput) {
        mNodeType = "Transpose";
        std::string name = mNodeName + "_input_transpose";
        mParams.clear();
        mInputs.clear();
        mOutputs.clear();
        this->createParamTensor("perm", QNN_DATATYPE_UINT_32, {(uint32_t)inputPerm.size()}, (void*)inputPerm.data(),
                                "_input_transpose");
        mParams.push_back(*(mParamTensorWrappers.back()->getNativeParam()));
        mInputs.push_back(*(mBackend->getNativeTensor(inputs[0])));
        mOutputs.push_back(*(mTempTensorWrappers[inputTempIndex]->getNativeTensor()));
        mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams,
                                 mInputs, mOutputs);
    }

    // reshape
    {
        mNodeType = "Reshape";
        std::string name = mNodeName;
        mParams.clear();
        mInputs.clear();
        mOutputs.clear();
        if (permuteInput) {
            mInputs.push_back(*(mTempTensorWrappers[inputTempIndex]->getNativeTensor()));
        } else {
            mInputs.push_back(*(mBackend->getNativeTensor(inputs[0])));
        }
        if (permuteOutput) {
            mOutputs.push_back(*(mTempTensorWrappers[outputTempIndex]->getNativeTensor()));
        } else {
            mOutputs.push_back(*(mBackend->getNativeTensor(outputs[0])));
        }
        mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams,
                                 mInputs, mOutputs);
    }

    // nchw -> nhwc
    {
        mNodeType = "Transpose";
        std::string name = mNodeName + "_output_transpose";
        mParams.clear();
        mInputs.clear();
        mOutputs.clear();
        this->createParamTensor("perm", QNN_DATATYPE_UINT_32, {(uint32_t)outputPerm.size()}, (void*)outputPerm.data(),
                                "_output_transpose");
        mParams.push_back(*(mParamTensorWrappers.back()->getNativeParam()));
        mInputs.push_back(*(mTempTensorWrappers[outputTempIndex]->getNativeTensor()));
        mOutputs.push_back(*(mBackend->getNativeTensor(outputs[0])));
        mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams,
                                 mInputs, mOutputs);
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
