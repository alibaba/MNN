//
//  QNNStridedSlice.cpp
//  MNN
//
//  Created by MNN on b'2025/04/10'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "QNNStridedSlice.hpp"

namespace MNN {
namespace QNN {

ErrorCode QNNStridedSlice::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto param = mOp->main_as_StridedSliceParam();
    if (inputs.size()!= 4 || param->fromType() != 0 || param->newAxisMask() != 0) {
        MNN_QNN_NOT_SUPPORT_SPECIAL_CASE;
    }

    mNodeType = "StridedSlice";

    auto inputTensor = inputs[0];
    auto beginTensor = inputs[1];
    auto endTensor = inputs[2];
    auto strideTensor = inputs[3];

    bool isConstBgein = (TensorUtils::getDescribe(beginTensor)->usage == Tensor::InsideDescribe::Usage::CONSTANT);
    bool isConstEnd = (TensorUtils::getDescribe(endTensor)->usage == Tensor::InsideDescribe::Usage::CONSTANT);
    bool isConstStride = (TensorUtils::getDescribe(strideTensor)->usage == Tensor::InsideDescribe::Usage::CONSTANT);

    MNN_ASSERT(isConstBgein && isConstEnd && isConstStride);

    int inputDim = inputTensor->dimensions();
    int strideSize = beginTensor->length(0);
    Tensor::DimensionType dimType = inputTensor->getDimensionType();

    std::vector<int> rangeData(inputDim * 3, 0);

    std::vector<int> beginRaw(inputDim, 0);
    std::vector<int> endRaw = inputTensor->shape();
    std::vector<int> strideRaw(inputDim, 1);
    auto beginRawSource = beginTensor->host<int>();
    auto endRawSource = endTensor->host<int>();
    auto strideRawSource = strideTensor->host<int>();
    for (int i = 0; i < strideSize; i++) {
        beginRaw[i] = beginRawSource[i];
        endRaw[i] = endRawSource[i];
        strideRaw[i] = strideRawSource[i];
    }

    for (int axis = 0; axis < inputDim; axis++) {
        int realAxis = getNHWCAxis(axis, inputDim, dimType);
        rangeData[3 * realAxis + 0] = beginRaw[axis];
        rangeData[3 * realAxis + 1] = endRaw[axis];
        rangeData[3 * realAxis + 2] = strideRaw[axis];
    }

    uint32_t beginMaskData = computeMask(param->beginMask(), inputDim, dimType);
    uint32_t endMaskData =  computeMask(param->endMask(), inputDim, dimType);
    uint32_t shrinkAxesData =  computeMask(param->shrinkAxisMask(), inputDim, dimType);
    uint32_t newAxesMaskData = 0;

    this->createParamTensor("ranges", QNN_DATATYPE_INT_32, {(uint32_t) inputDim, 3}, (void *) rangeData.data());
    this->createParamScalar("begin_mask", beginMaskData);
    this->createParamScalar("end_mask", endMaskData);
    this->createParamScalar("shrink_axes", shrinkAxesData);
    this->createParamScalar("new_axes_mask", newAxesMaskData);

    mParams.push_back(*(mParamTensorWrappers[0]->getNativeParam()));
    for (int i = 0; i < mParamScalarWrappers.size(); i++) {
        mParams.push_back(*(mParamScalarWrappers[i]->getNativeParam()));
    }
    mInputs.push_back(*(mBackend->getNativeTensor(inputs[0])));
    mOutputs.push_back(*(mBackend->getNativeTensor(outputs[0])));

    mBackend->addNodeToGraph(mOpConfigVersion, mNodeName.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);

    return NO_ERROR;
}


uint32_t QNNStridedSlice::computeMask(uint32_t rawMask, int dim, Tensor::DimensionType dimType) {
    if (rawMask == 0) return 0;

    uint32_t result = 0;
    for (int axis = 0; axis < dim; axis++) {
        int realAxis = getNHWCAxis(axis, dim, dimType);
        result |= ((rawMask >> axis) & 1) << realAxis; // If the axis-th bit of rawMask is 1, set the realAxis-th bit of result to 1.
    }

    return result;
}

class QNNStridedSliceCreator : public QnnBackend::Creator {
public:
    virtual QNNCommonExecution * onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op,
                                Backend* backend) const override {
        return new QNNStridedSlice(backend, op);
    }
};

REGISTER_QNN_OP_CREATOR(QNNStridedSliceCreator, OpType_StridedSlice)

} // end namespace QNN
} // end namespace MNN

