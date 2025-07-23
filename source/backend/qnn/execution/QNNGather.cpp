
#include "QNNGather.hpp"

namespace MNN {
namespace QNN {

ErrorCode QNNGather::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input = inputs[0];
    auto indices = inputs[1];
    auto axisTensor = inputs[2];
    auto output = outputs[0];

    // Create public resources.
    mInputDim = input->dimensions();
    mOutputDim = output->dimensions();
    mDimType = TensorUtils::getDimType(input);
    mRawAxis = axisTensor->host<int32_t>()[0];
    mRawAxis = (mRawAxis >= 0) ? mRawAxis : (input->buffer().dimensions + mRawAxis);
    mQnnDataType = mBackend->getNativeTensor(input)->v1.dataType;
    mFlagScalarIndices = (indices->dimensions() == 0) ? true : false;

    // Goto branches.
    if (mDimType == Tensor::DimensionType::TENSORFLOW && mFlagScalarIndices) {
        return this->onEncodeNHWCScalar(inputs, outputs);
    }

    if (mDimType == Tensor::DimensionType::TENSORFLOW && !mFlagScalarIndices) {
        return this->onEncodeNHWCTensor(inputs, outputs);
    }

    if (mDimType == Tensor::DimensionType::CAFFE && mFlagScalarIndices) {
        return this->onEncodeNCHWScalar(inputs, outputs);
    }

    if (mDimType == Tensor::DimensionType::CAFFE && !mFlagScalarIndices) {
        return this->onEncodeNCHWTensor(inputs, outputs);
    }

    return NO_ERROR;
}

ErrorCode QNNGather::onEncodeNHWCScalar(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    // Create resources.
    this->createParamScalar("axis", mRawAxis);

    std::vector<int> shapeStageOutput = inputs[0]->shape();
    shapeStageOutput[mRawAxis] = 1;

    this->createStageTensor("stageOutput", mQnnDataType, shapeStageOutput);

    // Add Nodes.
    this->addNodeGather("Gather",
                        *(mBackend->getNativeTensor(inputs[0])),
                        *(mBackend->getNativeTensor(inputs[1])),
                        *(mParamScalarWrappers[0]->getNativeParam()),
                        *(mTempTensorWrappers[0]->getNativeTensor()));

    this->addNodeReshape("Reshape",
                        *(mTempTensorWrappers[0]->getNativeTensor()),
                        *(mBackend->getNativeTensor(outputs[0])));

    return NO_ERROR;
}

ErrorCode QNNGather::onEncodeNHWCTensor(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    // Create resources.
    this->createParamScalar("axis", mRawAxis);

    // Add Node.
    this->addNodeGather("Gather",
                        *(mBackend->getNativeTensor(inputs[0])),
                        *(mBackend->getNativeTensor(inputs[1])),
                        *(mParamScalarWrappers[0]->getNativeParam()),
                        *(mBackend->getNativeTensor(outputs[0])));

    return NO_ERROR;
}

ErrorCode QNNGather::onEncodeNCHWScalar(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    // Create resources.
    this->createParamScalar("axis", mRawAxis);

    std::vector<int> shapeStageInput = inputs[0]->shape();
    std::vector<int> shapeStageOutput0 = shapeStageInput; shapeStageOutput0[mRawAxis] = 1;
    std::vector<int> shapeStageOutput1 = outputs[0]->shape();
    this->createStageTensor("stageInput", mQnnDataType, shapeStageInput);
    this->createStageTensor("stageOutput0", mQnnDataType, shapeStageOutput0);
    this->createStageTensor("stageOutput1", mQnnDataType, shapeStageOutput1);

    std::vector<uint32_t> permBeforeData(mInputDim, 0);
    std::vector<uint32_t> permAfterData(mOutputDim, 0);
    for (int i = 0; i < mInputDim; i++) {
        permBeforeData[i] = getNHWCAxis(i, mInputDim, Tensor::DimensionType::CAFFE);
    }
    for (int j = 0; j < mOutputDim; j++) {
        permAfterData[j] = getNCHWAxis(j, mOutputDim, Tensor::DimensionType::TENSORFLOW);
    }
    this->createParamTensor("perm", QNN_DATATYPE_UINT_32, {(uint32_t) mInputDim}, (void *) permBeforeData.data(), "before");
    this->createParamTensor("perm", QNN_DATATYPE_UINT_32, {(uint32_t) mOutputDim}, (void *) permAfterData.data(), "after");

    this->addNodeCommonPermute("PermuteBefore",
                                *(mBackend->getNativeTensor(inputs[0])),
                                *(mParamTensorWrappers[0]->getNativeParam()),
                                *(mTempTensorWrappers[0]->getNativeTensor()));

    this->addNodeGather("Gather",
                        *(mTempTensorWrappers[0]->getNativeTensor()),
                        *(mBackend->getNativeTensor(inputs[1])),
                        *(mParamScalarWrappers[0]->getNativeParam()),
                        *(mTempTensorWrappers[1]->getNativeTensor()));

    this->addNodeReshape("Squeeze",
                         *(mTempTensorWrappers[1]->getNativeTensor()),
                         *(mTempTensorWrappers[2]->getNativeTensor()));

    this->addNodeCommonPermute("PermuteAfter",
                               *(mTempTensorWrappers[2]->getNativeTensor()),
                               *(mParamTensorWrappers[1]->getNativeParam()),
                               *(mBackend->getNativeTensor(outputs[0])));

    return NO_ERROR;
}

ErrorCode QNNGather::onEncodeNCHWTensor(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    // Create resources.
    this->createParamScalar("axis", mRawAxis);

    std::vector<int> shapeStageInput = inputs[0]->shape();
    std::vector<int> shapeStageOutput = outputs[0]->shape();
    this->createStageTensor("stageInput", mQnnDataType, shapeStageInput);
    this->createStageTensor("stageOutput", mQnnDataType, shapeStageOutput);

    std::vector<uint32_t> permBeforeData(mInputDim, 0);
    std::vector<uint32_t> permAfterData(mOutputDim, 0);
    for (int i = 0; i < mInputDim; i++) {
        permBeforeData[i] = getNHWCAxis(i, mInputDim, Tensor::DimensionType::CAFFE);
    }
    for (int j = 0; j < mOutputDim; j++) {
        permAfterData[j] = getNCHWAxis(j, mOutputDim, Tensor::DimensionType::TENSORFLOW);
    }

    this->createParamTensor("perm", QNN_DATATYPE_UINT_32, {(uint32_t) mInputDim}, (void *) permBeforeData.data(), "before");
    this->createParamTensor("perm", QNN_DATATYPE_UINT_32, {(uint32_t) mOutputDim}, (void *) permAfterData.data(), "after");

    // Add Nodes.
    this->addNodeCommonPermute("PermuteBefore",
                               *(mBackend->getNativeTensor(inputs[0])),
                               *(mParamTensorWrappers[0]->getNativeParam()),
                               *(mTempTensorWrappers[0]->getNativeTensor()));

    this->addNodeGather("Gather",
                        *(mTempTensorWrappers[0]->getNativeTensor()),
                        *(mBackend->getNativeTensor(inputs[1])),
                        *(mParamScalarWrappers[0]->getNativeParam()),
                        *(mTempTensorWrappers[1]->getNativeTensor()));

    this->addNodeCommonPermute("PermuteAfter",
                               *(mTempTensorWrappers[1]->getNativeTensor()),
                               *(mParamTensorWrappers[1]->getNativeParam()),
                               *(mBackend->getNativeTensor(outputs[0])));

    return NO_ERROR;
}

void QNNGather::addNodeGather(const std::string & nodeNamePostfix, const Qnn_Tensor_t & input0, const Qnn_Tensor_t & input1, const Qnn_Param_t & paramAxis, const Qnn_Tensor_t & output) {
    CLEAR_BEFORE_ADDING_NODE;

    std::string name = mNodeName + "_" + nodeNamePostfix;
    mNodeType = "Gather";

    mInputs.push_back(input0);
    mInputs.push_back(input1);
    mParams.push_back(paramAxis);
    mOutputs.push_back(output);

    mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    return;
}

void QNNGather::addNodeReshape(const std::string & nodeNamePostfix, const Qnn_Tensor_t & input, const Qnn_Tensor_t & output) {
    CLEAR_BEFORE_ADDING_NODE;

    std::string name = mNodeName + "_" + nodeNamePostfix;
    mNodeType = "Reshape";

    mInputs.push_back(input);
    mOutputs.push_back(output);

    mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    return;
}

class QNNGatherCreator : public QnnBackend::Creator {
public:
    virtual QNNCommonExecution * onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op,
                                Backend* backend) const override {
        if (op->main_type() == OpParameter_Axis) {
            return nullptr;
        }

        if (inputs.size() != 3) {
            return nullptr;
        }

        // Avoid Permute for indices.
        if (inputs[1]->dimensions() > 2) {
            return  nullptr;
        }

        return new QNNGather(backend, op);
    }
};

REGISTER_QNN_OP_CREATOR(QNNGatherCreator, OpType_GatherV2)

} // end namespace QNN
} // end namespace MNN
