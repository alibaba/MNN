//
//  QNNStridedSlice.cpp
//  MNN
//
//  Created by MNN on b'2025/04/10'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "QNNStridedSlice.hpp"

#define CLIP(input, min, max) ((input) < (min) ? (min) : ((input) > (max) ? (max) : (input)))

namespace MNN {
namespace QNN {
#ifdef ENABLE_QNN_ONLINE_FINALIZE

QNNStridedSlice::QNNStridedSlice(Backend *backend, const Op *op) : QNNCommonExecution(backend, op) {
    if(op->type() == OpType_Slice) {
        mIsSlice = true;
    }
}

ErrorCode QNNStridedSlice::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto inputTensor = inputs[0];
    mInputDim = inputTensor->dimensions();
    mDimType = inputTensor->getDimensionType();
    auto inputShape = inputTensor->shape();
    if (TensorUtils::getDescribe(inputs[0])->dimensionFormat == MNN_DATA_FORMAT_NC4HW4) {
        // Turn to nhwc
        for (int index = 2; index < mInputDim; ++index) {
            inputShape[index - 1] = inputTensor->length(index);
        }
        if (mInputDim >= 2) {
            inputShape[mInputDim-1] = inputTensor->length(1);
        }
    }

    if(mIsSlice) {
        auto param = mOp->main_as_Slice();
        auto axis = param->axis();
        if (axis < 0) {
            axis = inputTensor->dimensions() + axis;
        }
        int64_t slice_num = 0;
        if (param->slicePoints() != nullptr) {
            if (param->slicePoints()->size() < outputs.size()) {
                slice_num = static_cast<int64_t>(outputs.size());
            } else if (param->slicePoints()->size() == 1) {
                slice_num = static_cast<int64_t>(param->slicePoints()->Get(0));
            } else {
                slice_num = static_cast<int64_t>(param->slicePoints()->size());
            }
        } else {
            slice_num = static_cast<int64_t>(outputs.size());
        }
        auto shape = inputShape;
        #ifdef QNN_VERBOSE
        MNN_PRINT("slice:%d %d %d %d, axis:%d, slice_num:%d output_num:%d, dim:%d\n", shape[0], shape[1], shape[2], shape[3], axis, slice_num, outputs.size(), mInputDim);
        #endif
        int realAxis = axis;
        if (TensorUtils::getDescribe(inputs[0])->dimensionFormat == MNN_DATA_FORMAT_NC4HW4) {
            if (axis > 1) {
                realAxis = axis - 1;
            } else if (axis == 1) {
                realAxis = mInputDim - 1;
            }
        }

        // Slices may be UNEQUAL (e.g. size_splits = [512,512,128]); rely on each output's real
        // length along the slice axis and accumulate the offset instead of assuming even division.
        int sliceOffset = 0;
        for(int index = 0; index < slice_num; index++) {
            int cur_size = outputs[index]->length(axis);
            std::vector<int> rangeData(mInputDim * 3, 0);
            for (int i = 0; i < mInputDim; i++) {
                rangeData[3 * i + 0] = 0;
                rangeData[3 * i + 1] = shape[i];
                rangeData[3 * i + 2] = 1;
            }
            rangeData[3 * realAxis + 0] = sliceOffset;
            rangeData[3 * realAxis + 1] = sliceOffset + cur_size;
            sliceOffset += cur_size;
            this->createParamTensor("ranges", QNN_DATATYPE_INT_32, {(uint32_t) mInputDim, 3}, (void *) rangeData.data(), std::to_string(index));

            // Add Node.
            mNodeType = "StridedSlice";
            mParams.clear();
            mInputs.clear();
            mOutputs.clear();
            std::string name =  mNodeName + "_part" + std::to_string(index);
            mParams.push_back(*(mParamTensorWrappers[index]->getNativeParam()));
            mInputs.push_back(*(mBackend->getNativeTensor(inputs[0])));
            mOutputs.push_back(*(mBackend->getNativeTensor(outputs[index])));

            mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
        }
        return NO_ERROR;
    }
    bool isNC4HW4 = (TensorUtils::getDescribe(inputs[0])->dimensionFormat == MNN_DATA_FORMAT_NC4HW4);

    auto param = mOp->main_as_StridedSliceParam();
    mNodeType = "StridedSlice";

    // Deal with ranges (in MNN's NCHW logical order).
    std::vector<int> beginRaw(mInputDim, 0);
    std::vector<int> endRaw = inputTensor->shape();
    std::vector<int> strideRaw(mInputDim, 1);
    if (param->fromType() == 0) {
        this->computeRangesType0(inputs, beginRaw, endRaw, strideRaw);
    } else {
        this->computeRangesType1(inputs, beginRaw, endRaw, strideRaw);
    }

    // NC4HW4 layout: QNN sees NHWC, so remap ranges from NCHW → NHWC
    if (isNC4HW4 && mInputDim >= 2) {
        // NCHW order: [N, C, H, W, ...]  →  NHWC order: [N, H, W, ..., C]
        // Save channel (axis=1) values
        int beginC = beginRaw[1], endC = endRaw[1], strideC = strideRaw[1];
        // Shift spatial dims left: axis 2..N-1 → axis 1..N-2
        for (int i = 2; i < mInputDim; i++) {
            beginRaw[i - 1]  = beginRaw[i];
            endRaw[i - 1]    = endRaw[i];
            strideRaw[i - 1] = strideRaw[i];
        }
        // Channel becomes last
        beginRaw[mInputDim - 1]  = beginC;
        endRaw[mInputDim - 1]    = endC;
        strideRaw[mInputDim - 1] = strideC;

        // Also remap endRaw to use NHWC shape from inputShape (already remapped above)
        for (int i = 0; i < mInputDim; i++) {
            if (endRaw[i] > inputShape[i]) {
                endRaw[i] = inputShape[i];
            }
        }
    }

    std::vector<int> rangeData(mInputDim * 3, 0);
    for (int axis = 0; axis < mInputDim; axis++) {
        rangeData[3 * axis + 0] = beginRaw[axis];
        rangeData[3 * axis + 1] = endRaw[axis];
        rangeData[3 * axis + 2] = strideRaw[axis];
    }
    this->createParamTensor("ranges", QNN_DATATYPE_INT_32, {(uint32_t) mInputDim, 3}, (void *) rangeData.data());

    // Deal with masks.
    uint32_t beginMaskData = computeMask(param->beginMask(), mInputDim, mDimType);
    uint32_t endMaskData =  computeMask(param->endMask(), mInputDim, mDimType);
    uint32_t shrinkAxesData =  computeMask(param->shrinkAxisMask(), mInputDim, mDimType);
    uint32_t newAxesMaskData = computeMask(param->newAxisMask(), mInputDim, mDimType);

    this->createParamScalar("begin_mask", beginMaskData);
    this->createParamScalar("end_mask", endMaskData);
    this->createParamScalar("shrink_axes", shrinkAxesData);
    this->createParamScalar("new_axes_mask", newAxesMaskData);

    // Add Node.
    mParams.push_back(*(mParamTensorWrappers[0]->getNativeParam()));
    for (int i = 0; i < mParamScalarWrappers.size(); i++) {
        mParams.push_back(*(mParamScalarWrappers[i]->getNativeParam()));
    }
    mInputs.push_back(*(mBackend->getNativeTensor(inputs[0])));
    mOutputs.push_back(*(mBackend->getNativeTensor(outputs[0])));

    mBackend->addNodeToGraph(mOpConfigVersion, mNodeName.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);

    return NO_ERROR;
}

void QNNStridedSlice::computeRangesType0(const std::vector<Tensor *> &inputs, std::vector<int> & beginRaw, std::vector<int> & endRaw, std::vector<int> & strideRaw) {
    auto inputTensor = inputs[0];
    auto beginTensor = inputs[1];
    auto endTensor = inputs[2];
    auto strideTensor = inputs[3];
    auto beginRawSource = beginTensor->host<int>();
    auto endRawSource = endTensor->host<int>();
    auto strideRawSource = strideTensor->host<int>();

    int sliceDim = beginTensor->length(0);
    MNN_ASSERT(sliceDim == endTensor->length(0) && sliceDim == strideTensor->length(0));

    for (int i = 0; i < sliceDim; i++) {
        beginRaw[i] = CLIP(beginRawSource[i], 0, inputs[0]->length(i) - 1);
        endRaw[i] = CLIP(endRawSource[i], 1, inputs[0]->length(i));
        strideRaw[i] = strideRawSource[i];
    }
    return;
}

void QNNStridedSlice::computeRangesType1(const std::vector<Tensor *> &inputs, std::vector<int> & beginRaw, std::vector<int> & endRaw, std::vector<int> & strideRaw) {
    auto inputTensor = inputs[0];
    auto beginTensor = inputs[1];
    auto endTensor = inputs[2];
    auto beginRawSource = beginTensor->host<int>();
    auto endRawSource = endTensor->host<int>();

    // fromType=1: inputs = [data, begin, end, axes, (stride)]
    // stride is optional — when absent (4 inputs), default all strides to 1
    bool hasStride = (inputs.size() >= 5);
    int* strideRawSource = nullptr;
    if (hasStride) {
        strideRawSource = inputs[4]->host<int>();
    }

    auto axisTensor = inputs[3];
    int sliceDim = beginTensor->length(0);

    for (int i = 0; i < sliceDim; i++) {
        int tempAxis = axisTensor->host<int>()[i];
        tempAxis = tempAxis >= 0 ? tempAxis : (tempAxis + mInputDim);
        beginRaw[tempAxis] = CLIP(beginRawSource[i], 0, inputs[0]->length(tempAxis) - 1);
        endRaw[tempAxis] = CLIP(endRawSource[i], 1, inputs[0]->length(tempAxis));
        strideRaw[tempAxis] = hasStride ? strideRawSource[i] : 1;
    }
    return;
}


uint32_t QNNStridedSlice::computeMask(uint32_t rawMask, int dim, Tensor::DimensionType mDimType) {
    if (rawMask == 0) return 0;

    uint32_t result = 0;
    for (int axis = 0; axis < dim; axis++) {
        int realAxis = axis;
        result |= ((rawMask >> axis) & 1) << realAxis; // If the axis-th bit of rawMask is 1, set the realAxis-th bit of result to 1.
    }

    return result;
}

class QNNStridedSliceCreator : public QnnBackend::Creator {
public:
    virtual QNNCommonExecution * onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op,
                                Backend* backend) const override {
        if(op->type() == OpType_Slice) {
            return new QNNStridedSlice(backend, op);
        }
        auto param = op->main_as_StridedSliceParam();
        if (nullptr == param) {
            MNN_PRINT("MNN_QNN StridedSlice: param is null\n");
            return nullptr;
        }

        MNN_PRINT("MNN_QNN StridedSlice: fromType=%d, inputs=%zu, beginMask=%d, endMask=%d, shrinkMask=%d, newAxisMask=%d, ellipsisMask=%d\n",
                   param->fromType(), inputs.size(), param->beginMask(), param->endMask(),
                   param->shrinkAxisMask(), param->newAxisMask(), param->ellipsisMask());

        // <begin>, <end> and <stride> should be static.
        for (int i = 1; i < inputs.size(); i++) {
            if (TensorUtils::getDescribe(inputs[i])->usage != Tensor::InsideDescribe::Usage::CONSTANT) {
                MNN_PRINT("MNN_QNN StridedSlice: input[%d] is NOT constant (usage=%d), skip\n", i, (int)TensorUtils::getDescribe(inputs[i])->usage);
                return nullptr;
            }
        }

        if (param->fromType() == 1) {
            if (param->shrinkAxisMask() != 0 || param->newAxisMask() != 0 || param->ellipsisMask() != 0) {
                MNN_PRINT("MNN_QNN StridedSlice: fromType1 unsupported masks\n");
                return nullptr;
            }
            // fromType=1: inputs = [data, begin, end, axes, (stride)]
            // stride is optional — default to 1 when absent (4 inputs)
            if (inputs.size() != 4 && inputs.size() != 5) {
                MNN_PRINT("MNN_QNN StridedSlice: fromType1 inputs.size=%zu, need 4 or 5\n", inputs.size());
                return nullptr;
            }
            return new QNNStridedSlice(backend, op);
        }

        if (param->fromType() == 0) {
            if (inputs.size() == 4 && param->newAxisMask() == 0 && param->ellipsisMask() == 0) {
                return new QNNStridedSlice(backend, op);
            } else {
                MNN_PRINT("MNN_QNN StridedSlice: fromType0 rejected: inputs=%zu, newAxisMask=%d, ellipsisMask=%d\n",
                           inputs.size(), param->newAxisMask(), param->ellipsisMask());
                return nullptr;
            }
        }

        // Shouldn't reach here.
        return nullptr;
    }
};

REGISTER_QNN_OP_CREATOR(QNNStridedSliceCreator, OpType_StridedSlice)
REGISTER_QNN_OP_CREATOR(QNNStridedSliceCreator, OpType_Slice)
#endif
} // end namespace QNN
} // end namespace MNN

