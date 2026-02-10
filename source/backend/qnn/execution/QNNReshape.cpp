//
//  QNNReshape.cpp
//  MNN
//
//  Created by MNN on b'2025/04/10'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "QNNReshape.hpp"

namespace MNN {
namespace QNN {
#ifdef ENABLE_QNN_ONLINE_FINALIZE

ErrorCode QNNReshape::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto inputDimFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
    auto outputDimFormat = TensorUtils::getDescribe(outputs[0])->dimensionFormat;
    mInputs.push_back(*(mBackend->getNativeTensor(inputs[0])));
    mOutputs.push_back(*(mBackend->getNativeTensor(outputs[0])));
    
    if(inputDimFormat == MNN_DATA_FORMAT_NC4HW4 && outputDimFormat == MNN_DATA_FORMAT_NCHW){
        mNodeType = "Transpose";
        uint32_t dim = inputs[0]->shape().size();
        std::vector<uint32_t> permData(dim, 0);
        permData[0] = 0;
        permData[1] = dim - 1;
        for (int i = 2; i < dim; i++) {
            permData[i] = i - 1;
        }
        this->createParamTensor("perm", QNN_DATATYPE_UINT_32, {dim}, (void *)permData.data());
        mParams.push_back(*(mParamTensorWrappers.back()->getNativeParam()));
        mBackend->addNodeToGraph(mOpConfigVersion, mNodeName.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
        return NO_ERROR;
    }
    
    if(outputDimFormat == MNN_DATA_FORMAT_NC4HW4 && inputDimFormat == MNN_DATA_FORMAT_NCHW){
        mNodeType = "Transpose";
        uint32_t dim = inputs[0]->shape().size();
        std::vector<uint32_t> permData(dim, 0);
        permData[0] = 0;
        permData[dim - 1] = 1;
        for (int i = 1; i < dim - 1; i++) {
            permData[i] = i + 1;
        }
        this->createParamTensor("perm", QNN_DATATYPE_UINT_32, {dim}, (void *)permData.data());
        mParams.push_back(*(mParamTensorWrappers.back()->getNativeParam()));
        mBackend->addNodeToGraph(mOpConfigVersion, mNodeName.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
        return NO_ERROR;
    }
    
    mNodeType = "Reshape";
    mBackend->addNodeToGraph(mOpConfigVersion, mNodeName.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    return NO_ERROR;
}


class QNNReshapeCreator : public QnnBackend::Creator {
public:
    virtual QNNCommonExecution * onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op,
                                Backend* backend) const override {
        return new QNNReshape(backend, op);
    }
};

// REGISTER_QNN_OP_CREATOR(QNNReshapeCreator, OpType_Reshape)
// REGISTER_QNN_OP_CREATOR(QNNReshapeCreator, OpType_Squeeze)
// REGISTER_QNN_OP_CREATOR(QNNReshapeCreator, OpType_Unsqueeze)
REGISTER_QNN_OP_CREATOR(QNNReshapeCreator, OpType_ConvertTensor)
#endif
} // end namespace QNN
} // end namespace MNN
