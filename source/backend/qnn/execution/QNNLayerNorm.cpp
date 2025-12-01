//
//  QNNLayerNorm.cpp
//  MNN
//
//  Created by MNN on b'2025/04/10'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "QNNLayerNorm.hpp"

namespace MNN {
namespace QNN {
#ifdef ENABLE_QNN_ONLINE_FINALIZE

QNNLayerNorm::QNNLayerNorm(Backend *backend, const Op *op, Tensor * input) : QNNCommonExecution(backend, op) {
    auto param = mOp->main_as_LayerNorm();

    mQnnDataType = mBackend->getUseFP16() ? QNN_DATATYPE_FLOAT_16 : QNN_DATATYPE_FLOAT_32;

    mInputDim = input->dimensions();

    mDimType = TensorUtils::getDimType(input);

    mEpsilon = param->epsilon();

    mUseRMSNorm = param->useRMSNorm();

    uint32_t axesSize = param->axis()->size();
    const int * axesData = param->axis()->data();
    int rawAxis = (axesData[0] >= 0) ? axesData[0] : (mInputDim + axesData[0]);
    mRealAxis = rawAxis;

    // set gamma and beta
    {
        bool hasGammaBeta = (param->gamma() && param->beta());
        mGammaBetaSize = 0;
        if (hasGammaBeta) {
            MNN_ASSERT(param->gamma()->size() == param->beta()->size());
            mGammaBetaSize = param->gamma()->size();
        }
        hasGammaBeta = hasGammaBeta || (param->external() && param->external()->size() > 1 && param->external()->data()[1] > 0);
        if (hasGammaBeta && mGammaBetaSize == 0) {
            mGammaBetaSize = param->external()->data()[1] / sizeof(float);
        }

        if(mGammaBetaSize > 0) {
            mGammaData.resize(mGammaBetaSize, 1.0f);
            mBetaData.resize(mGammaBetaSize);
            ::memcpy(mGammaData.data(), param->gamma()->data(), mGammaBetaSize * sizeof(float));
            ::memcpy(mBetaData.data(), param->beta()->data(), mGammaBetaSize * sizeof(float));
        }
    }

}

ErrorCode QNNLayerNorm::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    std::string nodeNameBase = "LayerNorm";
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

ErrorCode QNNLayerNorm::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input = inputs[0];

    std::vector<uint32_t> realInputShape = getNHWCShape(input);

    // Create Resources.
    this->createParamScalar("epsilon", mEpsilon);                                                                          // mParamScalarWrappers[0], epsilon

    uint32_t tempPtr[1] = {(uint32_t) mInputDim - 1}; // Qnn only allows the last dim for norm.
    this->createParamTensor("axes", QNN_DATATYPE_UINT_32, {1}, (void *)tempPtr);                                         // mParamTensorWrappers[0], axes

    if (mGammaBetaSize == 0) {
        mGammaBetaSize = realInputShape[mRealAxis];
        #ifdef QNN_VERBOSE
        MNN_PRINT("LayerNorm do not have original gamma beta, %d", mGammaBetaSize);
        #endif
        mGammaData.resize(mGammaBetaSize, 1.0f);
        mBetaData.resize(mGammaBetaSize, 0.0f);
    } else {
        MNN_ASSERT(mGammaBetaSize == realInputShape[mRealAxis]);
    }
    
    Qnn_DataType_t dataType = mBackend->getNativeTensor(inputs[0])->v1.dataType;
    this->createStaticFloatTensor("gamma", dataType, {(uint32_t) mGammaBetaSize}, mGammaData.data());                                      // mTempTensorWrappers[0], gamma
    this->createStaticFloatTensor("beta", dataType, {(uint32_t) mGammaBetaSize}, mBetaData.data());                                        // mTempTensorWrappers[1], beta

    // Extra resources needed by Case Permute.
    bool needPermute = (mRealAxis == (mInputDim - 1)) ? false : true;
    if (needPermute) {
        std::vector<uint32_t> realInputShape = getNHWCShape(inputs[0]);

        std::vector<uint32_t> permData(mInputDim, 0);
        std::vector<uint32_t> tempInputOutputShape(mInputDim, 0);

        for (int i = 0; i < mRealAxis; i++) {
            permData[i] = i;
            tempInputOutputShape[i] = realInputShape[i];
        }
        permData[mRealAxis] = mInputDim - 1;
        tempInputOutputShape[mRealAxis] = realInputShape[mInputDim - 1];
        for (int j = mRealAxis + 1; j < mInputDim - 1; j++) {
            permData[j] = j;
            tempInputOutputShape[j] = realInputShape[j];
        }
        permData[mInputDim - 1] = mRealAxis;
        tempInputOutputShape[mInputDim - 1] = realInputShape[mRealAxis];

        #ifdef QNN_VERBOSE
        MNN_PRINT("QNN LayerNorm Permute data:");
        for(int i = 0; i < permData.size(); i++) {
            MNN_PRINT("%d ", permData[i]);
        }
        MNN_PRINT("\n");
        MNN_PRINT("QNN LayerNorm tempShape data:");
        for(int i = 0; i < tempInputOutputShape.size(); i++) {
            MNN_PRINT("%d ", tempInputOutputShape[i]);
        }
        MNN_PRINT("\n");
        #endif

        this->createParamTensor("perm", QNN_DATATYPE_UINT_32, {(uint32_t) mInputDim}, (void *) permData.data(), "before");           // mParamTensorWrappers[1], perm before
        this->createParamTensor("perm", QNN_DATATYPE_UINT_32, {(uint32_t) mInputDim}, (void *) permData.data(), "after");            // mParamTensorWrappers[2], perm after
        this->createStageTensor("tempInput", mQnnDataType, tempInputOutputShape);                                                       // mTempTensorWrappers[2], temp input
        this->createStageTensor("tempOutput", mQnnDataType, tempInputOutputShape);                                                      // mTempTensorWrappers[3], temp output
    }


    #ifdef QNN_VERBOSE
    MNN_PRINT("QNN LayerNorm useFp16:%d \ninput0:", mBackend->getUseFP16());
    auto shape0 = inputs[0]->shape();
    for(int i = 0; i < shape0.size(); i++) {
        MNN_PRINT("%d x ", shape0[i]);
    }
    MNN_PRINT("\noutput:");
    auto outShape = outputs[0]->shape();
    for(int i = 0; i < outShape.size(); i++) {
        MNN_PRINT("%d x ", outShape[i]);
    }
    MNN_PRINT("\n");
    MNN_PRINT("need Permute:%d, gamma:%d, reduceAxis:%d,\n", needPermute, mGammaBetaSize, mRealAxis);

    int rank = inputs.at(0)->dimensions();
    for(int i = 0; i < rank; i++) {
        MNN_PRINT("%d ", inputs.at(0)->length(i));
    }
    #endif

    // Add Nodes to Graph.
    if (needPermute) {
        return this->onEncodeNormWithPermute(inputs, outputs);
    }

    #ifdef QNN_LAYERNORM_RESHAPE_3D
    if(mInputDim == 4)
    {
        uint32_t tempPtr[1] = {(uint32_t)2}; // Qnn only allows the last dim for norm.
        this->createParamTensor("axes", QNN_DATATYPE_UINT_32, {1}, (void *)tempPtr, "redefine");      
        this->createStageTensor("InputReshapeTensor", dataType, std::vector<int>({inputs[0]->length(0), inputs[0]->length(2) * inputs[0]->length(3), inputs[0]->length(1)}));
        this->createStageTensor("OutputReshapeTensor", dataType, std::vector<int>({inputs[0]->length(0), inputs[0]->length(2) * inputs[0]->length(3), inputs[0]->length(1)}));
        // reshape input
        {
            std::string name = mNodeName + "_input_reshape";
            mParams.clear();
            mInputs.clear();
            mOutputs.clear();
            mNodeType = "Reshape";

            mInputs.push_back(*(mBackend->getNativeTensor(inputs[0]))); // input0
            mOutputs.push_back(*(mTempTensorWrappers[2]->getNativeTensor())); // temp input
            mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
        }

        {
            std::string name = mNodeName + "_norm";
            mParams.clear();
            mInputs.clear();
            mOutputs.clear();
            mNodeType = mUseRMSNorm ? "RmsNorm" : "LayerNorm";

            mInputs.push_back(*(mTempTensorWrappers[2]->getNativeTensor()));
            mInputs.push_back(*(mTempTensorWrappers[0]->getNativeTensor())); // gamma
            mInputs.push_back(*(mTempTensorWrappers[1]->getNativeTensor())); // beta

            mParams.push_back(*(mParamScalarWrappers[0]->getNativeParam())); // eps
            mParams.push_back(*(mParamTensorWrappers[1]->getNativeParam())); // axes

            mOutputs.push_back(*(mTempTensorWrappers[3]->getNativeTensor()));

            mBackend->addNodeToGraph(mOpConfigVersion, mNodeName.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
        }
        // reshape output
        {
            std::string name = mNodeName + "_output_reshape";
            mParams.clear();
            mInputs.clear();
            mOutputs.clear();
            mNodeType = "Reshape";

            mInputs.push_back(*(mTempTensorWrappers[3]->getNativeTensor())); // temp output
            mOutputs.push_back(*(mBackend->getNativeTensor(outputs[0]))); // input0
            mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
        }
        return NO_ERROR;
    }
    #endif

    mNodeType = mUseRMSNorm ? "RmsNorm" : "LayerNorm";

    mInputs.push_back(*(mBackend->getNativeTensor(inputs[0])));
    mInputs.push_back(*(mTempTensorWrappers[0]->getNativeTensor())); // gamma
    mInputs.push_back(*(mTempTensorWrappers[1]->getNativeTensor())); // beta

    mParams.push_back(*(mParamScalarWrappers[0]->getNativeParam())); // eps
    mParams.push_back(*(mParamTensorWrappers[0]->getNativeParam())); // axes

    mOutputs.push_back(*(mBackend->getNativeTensor(outputs[0])));

    mBackend->addNodeToGraph(mOpConfigVersion, mNodeName.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);

    return NO_ERROR;
}

ErrorCode QNNLayerNorm::onEncodeNormWithPermute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    // Permute before norm.
    {
        mNodeType.clear();
        mInputs.clear();
        mParams.clear();
        mOutputs.clear();

        std::string name = mNodeName + "_before";
        mNodeType = "Transpose";
        mInputs.push_back(*(mBackend->getNativeTensor(inputs[0]))); // inputs[0]
        mParams.push_back(*(mParamTensorWrappers[1]->getNativeParam())); // perm before
        mOutputs.push_back(*(mTempTensorWrappers[2]->getNativeTensor())); // temp input

        mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    }

    // Norm.
    {
        std::string name = mNodeName + "_norm";
        mNodeType.clear();
        mInputs.clear();
        mParams.clear();
        mOutputs.clear();

        mNodeType = mUseRMSNorm ? "RmsNorm" : "LayerNorm";
        mInputs.push_back(*(mTempTensorWrappers[2]->getNativeTensor())); // temp input
        mInputs.push_back(*(mTempTensorWrappers[0]->getNativeTensor())); // gamma
        mInputs.push_back(*(mTempTensorWrappers[1]->getNativeTensor())); // beta

        mParams.push_back(*(mParamScalarWrappers[0]->getNativeParam())); // eps
        mParams.push_back(*(mParamTensorWrappers[0]->getNativeParam())); // axes
    
        mOutputs.push_back(*(mTempTensorWrappers[3]->getNativeTensor())); // temp output
    
        mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    }

    // Permute after norm.
    {
        mNodeType.clear();
        mInputs.clear();
        mParams.clear();
        mOutputs.clear();

        std::string name = mNodeName + "_after";
        mNodeType = "Transpose";
        mInputs.push_back(*(mTempTensorWrappers[3]->getNativeTensor())); // temp output
        mParams.push_back(*(mParamTensorWrappers[2]->getNativeParam())); // perm after
        mOutputs.push_back(*(mBackend->getNativeTensor(outputs[0]))); // outputs[0]

        mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    }

    return NO_ERROR;
}

class QNNLayerNormCreator : public QnnBackend::Creator {
public:
    virtual QNNCommonExecution * onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op, Backend* backend) const override {
        auto inputDim = inputs[0]->dimensions();
        if (inputDim > 4) {
            return nullptr;
        }

        auto param = op->main_as_LayerNorm();

        if (param->group() > 1) {
            return nullptr;
        }

        if (param->axis()->size() != 1) {
            return nullptr;
        }

        return new QNNLayerNorm(backend, op, inputs[0]);
    }
};

REGISTER_QNN_OP_CREATOR(QNNLayerNormCreator, OpType_LayerNorm)
#endif
} // end namespace MNN
} // end namespace QNN
