//
//  QNNSoftmax.cpp
//  MNN
//
//  Created by MNN on b'2025/04/10'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "QNNSoftmax.hpp"

namespace MNN {
namespace QNN {
#ifdef ENABLE_QNN_ONLINE_FINALIZE

ErrorCode QNNSoftmax::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    this->createParamScalar("axis", (uint32_t) inputs[0]->dimensions() - 1);

    if (mAxis != inputs[0]->dimensions() - 1) {
        return this->onEncodePermute(inputs, outputs);
    }

    mNodeType = "Softmax";
    this->addNodeCommon(inputs, outputs);

    return NO_ERROR;
}

ErrorCode QNNSoftmax::onEncodePermute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    // Create resources.
    int dim = inputs[0]->dimensions();
    std::vector<uint32_t> permData(dim, 0);
    for (int i = 0; i < dim; i++) {
        if (i == mAxis) {
            permData[i] = dim - 1;
            continue;
        }
        if (i == dim - 1) {
            permData[i] = mAxis;
            continue;
        }
        permData[i] = i;
    }

    std::vector<uint32_t> shapeStageTensor = getNHWCShape(inputs[0]);
    {
        uint32_t temp = shapeStageTensor[dim - 1];
        shapeStageTensor[dim - 1] = shapeStageTensor[mAxis];
        shapeStageTensor[mAxis] = temp;
    }

    this->createParamTensor("perm", QNN_DATATYPE_UINT_32, {(uint32_t) dim}, (void *) permData.data(), "before"); // mParamTensorWrappers[0], permBefore
    this->createParamTensor("perm", QNN_DATATYPE_UINT_32, {(uint32_t) dim}, (void *) permData.data(), "after"); // mParamTensorWrappers[1], permAfter

    Qnn_DataType_t qnnDataType = mBackend->getNativeTensor(inputs[0])->v1.dataType;
    this->createStageTensor("stageInput", qnnDataType, shapeStageTensor, inputs[0]); // mTempTensorWrappers[0], stage input
    this->createStageTensor("stageOutput", qnnDataType, shapeStageTensor, outputs[0]); // mTempTensorWrappers[1], stage output

    // Add nodes.
    {
        this->addNodeCommonPermute("PermBefore",
                                   *(mBackend->getNativeTensor(inputs[0])),
                                   *(mParamTensorWrappers[0]->getNativeParam()),
                                   *(mTempTensorWrappers[0]->getNativeTensor()));
    }

    {
        CLEAR_BEFORE_ADDING_NODE;

        std::string name = mNodeName + "_Softmax";
        mNodeType = "Softmax";
        mInputs.push_back(*(mTempTensorWrappers[0]->getNativeTensor()));
        mParams.push_back(*(mParamScalarWrappers[0]->getNativeParam()));
        mOutputs.push_back(*(mTempTensorWrappers[1]->getNativeTensor()));
    
        mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    }

    {
        this->addNodeCommonPermute("PermAfter",
                                   *(mTempTensorWrappers[1]->getNativeTensor()),
                                   *(mParamTensorWrappers[1]->getNativeParam()),
                                   *(mBackend->getNativeTensor(outputs[0])));
    }

    return NO_ERROR;
}

class QNNSoftmaxCreator : public QnnBackend::Creator {
public:
    virtual QNNCommonExecution * onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op,
                                Backend* backend) const override {
        const auto softmaxParam = op->main_as_Axis();
        int axis = softmaxParam->axis();
        if (axis < 0) {
            axis = inputs[0]->dimensions() + axis;
        }

        return new QNNSoftmax(backend, op, axis);
    }
};

REGISTER_QNN_OP_CREATOR(QNNSoftmaxCreator, OpType_Softmax)
#endif
} // end namespace QNN
} // end namespace MNN
