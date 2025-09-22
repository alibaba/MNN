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

static bool canDirectReshapeOpt(const Tensor* input, const Tensor* output) {
    if(input->elementSize() == output->elementSize()) {
        // [N, C, H*W] -> [N, C, H, W]
        if(input->shape().size() == 3 && output->shape().size() == 4) {
            if(input->length(0) == output->length(0) && input->length(1) == output->length(1)) {
                return true;
            }
        }

        // [N, C, H, W] -> [N, C, H*W]
        if(input->shape().size() == 4 && output->shape().size() == 3) {
            if(input->length(0) == output->length(0) && input->length(1) == output->length(1)) {
                return true;
            }
        }

        // all dimensions same
        if(input->shape().size() == output->shape().size()) {
            bool allSame = true;
            for(int i = 0; i < input->shape().size(); i++) {
                if(input->shape()[i] != output->shape()[i]) {
                    allSame = false;
                    break;
                }
            }
            if(allSame) {
                return true;
            }
        }
    }
    
    return false;
}
static bool canTransposeReshapeOpt(const Tensor* input, const Tensor* output) {
    // [N, C, 1, 1] <-> [1, N, C]
    if(input->elementSize() == output->elementSize()) {
        if(input->shape().size() == 3 && output->shape().size() == 4) {
            if(input->length(0) == 1 && output->length(2) == 1 && output->length(3) == 1) {
                return true;
            }
        }
        if(input->shape().size() == 4 && output->shape().size() == 3) {
            if(input->length(2) == 1 && input->length(3) == 1 && output->length(0) == 1) {
                return true;
            }
        }
    }
    return false;
}
ErrorCode QNNFlatten::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    Tensor::DimensionType inputDimType = inputs[0]->getDimensionType();
    Tensor::DimensionType outputDimType = outputs[0]->getDimensionType();

    MNN_ASSERT(inputDimType == outputDimType);

    if (inputDimType == Tensor::TENSORFLOW) {
        mNodeType = "Reshape";
        // this->addNodeCommon(inputs, outputs);
        this->addNodeCommonReshape("Reshape", *(mBackend->getNativeTensor(inputs[0])), *(mBackend->getNativeTensor(outputs[0])));
    } else {
        if(canDirectReshapeOpt(inputs[0], outputs[0])) {
            this->addNodeCommonReshape("Reshape", *(mBackend->getNativeTensor(inputs[0])), *(mBackend->getNativeTensor(outputs[0])));
            return NO_ERROR;
        }

        if(canTransposeReshapeOpt(inputs[0], outputs[0])) {
            this->ReshapeTranspose(inputs, outputs);
            return NO_ERROR;
        }

        // general case
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

void QNNFlatten::ReshapeTranspose(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    if(inputs[0]->shape().size() == 4 && outputs[0]->shape().size() == 3) {
        std::vector<int> outputShape = outputs[0]->shape();
        Qnn_DataType_t dataType = mBackend->getNativeTensor(inputs[0])->v1.dataType;
        this->createStageTensor("reshape_temp", dataType, outputShape);
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
        this->createStageTensor("reshape_temp", dataType, inputShape);
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
