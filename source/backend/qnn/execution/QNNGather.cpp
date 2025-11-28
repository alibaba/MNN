
#include "QNNGather.hpp"

namespace MNN {
namespace QNN {
#ifdef ENABLE_QNN_ONLINE_FINALIZE

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

#ifdef QNN_VERBOSE
    MNN_PRINT("QNN Gather inputs shape:\n");
    for(int i = 0; i < inputs.size(); i++) {
        auto shape = inputs[i]->shape();
        for(int j = 0; j < shape.size(); j++) {
            MNN_PRINT("%d ", shape[j]);
        }
        MNN_PRINT("\n");
    }
    MNN_PRINT("QNN Gather axis: %d %d\n", mRawAxis, mDimType);
    MNN_PRINT("QNN Gather outputs shape:\n");
    for(int i = 0; i < outputs.size(); i++) {
        auto shape = outputs[i]->shape();
        for(int j = 0; j < shape.size(); j++) {
            MNN_PRINT("%d ", shape[j]);
        }
        MNN_PRINT("\n");
    }
#endif

    // Goto branches.
    if (mFlagScalarIndices) {
        return this->onEncodeScalar(inputs, outputs);
    }else{
        return this->onEncodeTensor(inputs, outputs);
    }

    return NO_ERROR;
}

ErrorCode QNNGather::onEncodeScalar(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    // Create resources.
    this->createParamScalar("axis", mRawAxis);

    std::vector<int> shapeStageOutput = inputs[0]->shape();
    shapeStageOutput[mRawAxis] = 1;
    
    this->createStageTensor("stageOutput", mQnnDataType, shapeStageOutput, outputs[0]);

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

ErrorCode QNNGather::onEncodeTensor(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    // Create resources.
    if(mOp->type() == OpType_GatherElements) {
        this->createParamScalar("axis", (uint32_t)mRawAxis);
    } else {
        this->createParamScalar("axis", (int)mRawAxis);
    }
    // Add Node.
    this->addNodeGather("Gather",
                        *(mBackend->getNativeTensor(inputs[0])),
                        *(mBackend->getNativeTensor(inputs[1])),
                        *(mParamScalarWrappers[0]->getNativeParam()),
                        *(mBackend->getNativeTensor(outputs[0])));

    return NO_ERROR;
}

void QNNGather::addNodeGather(const std::string & nodeNamePostfix, const Qnn_Tensor_t & input0, const Qnn_Tensor_t & input1, const Qnn_Param_t & paramAxis, const Qnn_Tensor_t & output) {
    CLEAR_BEFORE_ADDING_NODE;

    std::string name = mNodeName + "_" + nodeNamePostfix;
    mNodeType = "Gather";

    if(mOp->type() == OpType_GatherElements) {
        mNodeType = "GatherElements";    
    }
    // MNN_PRINT("mNodeType: %s\n", mNodeType.c_str());
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
            MNN_ERROR("QNN Gather type error fallback\n");
            return nullptr;
        }

        if (inputs.size() < 2) {
            MNN_ERROR("QNN Gather inputs size:%d error fallback\n", inputs.size());
            return nullptr;
        }

        return new QNNGather(backend, op);
    }
};

REGISTER_QNN_OP_CREATOR(QNNGatherCreator, OpType_GatherV2)
REGISTER_QNN_OP_CREATOR(QNNGatherCreator, OpType_GatherElements)
#endif
} // end namespace QNN
} // end namespace MNN
