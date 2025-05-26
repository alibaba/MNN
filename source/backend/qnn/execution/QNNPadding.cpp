//
//  QNNPadding.cpp
//  MNN
//
//  Created by MNN on b'2025/04/10'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "QNNPadding.hpp"

namespace MNN {
namespace QNN {

ErrorCode QNNPadding::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(inputs.size() == 2);

    mNodeType = "Pad";

    auto input            = inputs[0];
    auto padding         = inputs[1];
    auto output           = outputs[0];
    int * paddingDataSrc = padding->host<int>();

    auto size = padding->elementSize();
    auto dimensions = input->dimensions();
    MNN_ASSERT(size == (2 * dimensions));
    Tensor::DimensionType dimType = input->getDimensionType();
    std::vector<uint32_t> padAmountData(size, 0);
    for (int i = 0; i < dimensions; ++i) {
        int axis  = getNHWCAxis(i, dimensions, dimType);
        padAmountData[2 * axis + 0] = (uint32_t) paddingDataSrc[2 * i + 0];
        padAmountData[2 * axis + 1] = (uint32_t) paddingDataSrc[2 * i + 1];
    }

    this->createParamScalar("scheme", (uint32_t) 0); // 0 means 'CONSTANT'
    mParams.push_back(*(mParamScalarWrappers.back()->getNativeParam()));

    this->createParamTensor("pad_amount", QNN_DATATYPE_UINT_32, {(uint32_t)dimensions, 2}, (void *) padAmountData.data());
    mParams.push_back(*(mParamTensorWrappers.back()->getNativeParam()));

    mInputs.push_back(*(mBackend->getNativeTensor(inputs[0])));
    mOutputs.push_back(*(mBackend->getNativeTensor(outputs[0])));
    mBackend->addNodeToGraph(mOpConfigVersion, mNodeName.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);

    return NO_ERROR;
}


class QNNPaddingCreator : public QnnBackend::Creator {
public:
    virtual QNNCommonExecution * onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op,
                                Backend* backend) const override {
        return new QNNPadding(backend, op);
    }
};

REGISTER_QNN_OP_CREATOR(QNNPaddingCreator, OpType_Padding)

} // end namespace QNN
} // end namespace MNN
