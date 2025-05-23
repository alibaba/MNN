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

ErrorCode QNNFlatten::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    Tensor::DimensionType inputDimType = inputs[0]->getDimensionType();
    Tensor::DimensionType outputDimType = outputs[0]->getDimensionType();

    MNN_ASSERT(inputDimType == outputDimType);

    if (inputDimType == Tensor::TENSORFLOW) {
        mNodeType = "Reshape";
        this->addNodeCommon(inputs, outputs);
    } else {
        {
            std::vector<int> inputShape = inputs[0]->shape();
            std::vector<int> outputShape = outputs[0]->shape();
            Qnn_DataType_t dataType = mBackend->getNativeTensor(inputs[0])->v1.dataType;
            this->createStageTensor("pre", dataType, inputShape);
            this->createStageTensor("post", dataType, outputShape);
        }

        this->NHWC2NCHW(inputs);
        this->Reshape(inputs, outputs);
        this->NCHW2NHWC(outputs);
    }

    return NO_ERROR;
}


void QNNFlatten::NHWC2NCHW(const std::vector<Tensor *> &inputs) {
    mNodeType = "Transpose";
    std::string name = mNodeName + "_pre";
    mParams.clear();
    mInputs.clear();
    mOutputs.clear();

    // shape(out[0])[i] = shape(in[0])[perm[i]]
    uint32_t dim = inputs[0]->shape().size();
    std::vector<uint32_t> permData(dim, 0);
    for (int i = 0; i < dim ; i++) {
        permData[i] = getNHWCAxis(i, dim, Tensor::CAFFE);
    }
    this->createParamTensor("perm", QNN_DATATYPE_UINT_32, {dim}, (void *)permData.data(), "pre");
    mParams.push_back(*(mParamTensorWrappers.back()->getNativeParam()));
    mInputs.push_back(*(mBackend->getNativeTensor(inputs[0])));
    mOutputs.push_back(*(mTempTensorWrappers[0]->getNativeTensor()));
    mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
}


void QNNFlatten::Reshape(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mNodeType = "Reshape";
    std::string name = mNodeName + "_reshape";
    mParams.clear();
    mInputs.clear();
    mOutputs.clear();

    mInputs.push_back(*(mTempTensorWrappers[0]->getNativeTensor()));
    mOutputs.push_back(*(mTempTensorWrappers[1]->getNativeTensor()));
    mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
}


void QNNFlatten::NCHW2NHWC(const std::vector<Tensor *> &outputs) {
    mNodeType = "Transpose";
    std::string name = mNodeName + "_post";
    mParams.clear();
    mInputs.clear();
    mOutputs.clear();

    // shape(out[0])[i] = shape(in[0])[perm[i]]
    uint32_t dim = outputs[0]->shape().size();
    std::vector<uint32_t> permData(dim, 0);
    for (int i = 0; i < dim ; i++) {
        permData[i] = getNCHWAxis(i, dim, Tensor::TENSORFLOW);
    }
    this->createParamTensor("perm", QNN_DATATYPE_UINT_32, {dim}, (void *)permData.data(), "post");
    mParams.push_back(*(mParamTensorWrappers.back()->getNativeParam()));
    mInputs.push_back(*(mTempTensorWrappers[1]->getNativeTensor()));
    mOutputs.push_back(*(mBackend->getNativeTensor(outputs[0])));
    mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
}


class QNNFlattenCreator : public QnnBackend::Creator {
public:
    virtual QNNCommonExecution * onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op,
                                Backend* backend) const override {
        return new QNNFlatten(backend, op);
    }
};

REGISTER_QNN_OP_CREATOR(QNNFlattenCreator, OpType_Flatten)

} // end namespace QNN
} // end namespace MNN
