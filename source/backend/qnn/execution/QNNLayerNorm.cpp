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

ErrorCode QNNLayerNorm::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    int inputDim = inputs[0]->dimensions();
    std::vector<uint32_t> realInputShape = getNHWCShape(inputs[0]);
    Qnn_DataType_t dataType = mBackend->getNativeTensor(inputs[0])->v1.dataType;
    mNodeType = "LayerNorm";

    auto param = mOp->main_as_LayerNorm();

    // param epsilon
    float eps = param->epsilon();
    this->createParamScalar("epsilon", eps);
    mParams.push_back(*(mParamScalarWrappers.back()->getNativeParam()));
    
    // param axes
    uint32_t axesSize = param->axis()->size();
    const int * axesData = param->axis()->data();
    std::vector<uint32_t> realAxeses(axesSize, 0);
    for (int i = 0; i < axesSize; i++) {
        realAxeses[i] = getNHWCAxis(axesData[i], inputDim, TensorUtils::getDimType(inputs[0]));
    }
    this->createParamTensor("axes", QNN_DATATYPE_UINT_32, {axesSize}, (void *) realAxeses.data());
    mParams.push_back(*(mParamTensorWrappers.back()->getNativeParam()));

    // in[0] input
    mInputs.push_back(*(mBackend->getNativeTensor(inputs[0])));

    // in[1] gamma
    // in[2] beta
    MNN_ASSERT(param->gamma()->size() == param->beta()->size());
    const float * gammaRaw = param->gamma()->data();
    const float * betaRaw = param->beta()->data();
    std::vector<uint32_t> gammaBetaShape(axesSize, 0);
    for (int i = 0; i < axesSize; i++) {
        gammaBetaShape[i] = realInputShape[realAxeses[i]];
    }
    this->createStaticFloatTensor("gamma", dataType, gammaBetaShape, gammaRaw);
    mInputs.push_back(*(mTempTensorWrappers.back()->getNativeTensor()));
    this->createStaticFloatTensor("beta", dataType, gammaBetaShape, betaRaw);
    mInputs.push_back(*(mTempTensorWrappers.back()->getNativeTensor()));

    // outputs[0]
    mOutputs.push_back(*(mBackend->getNativeTensor(outputs[0])));

    mBackend->addNodeToGraph(mOpConfigVersion, mNodeName.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);

    return NO_ERROR;
}

class QNNLayerNormCreator : public QnnBackend::Creator {
public:
    virtual QNNCommonExecution * onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op, Backend* backend) const override {
        auto param = op->main_as_LayerNorm();

        if (param->group() > 1 || param->useRMSNorm()) {
            return nullptr;
        }

        return new QNNLayerNorm(backend, op);
    }
};

REGISTER_QNN_OP_CREATOR(QNNLayerNormCreator, OpType_LayerNorm)

} // end namespace MNN
} // end namespace QNN
