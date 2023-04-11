//
//  NNAPICommonExecution.cpp
//  MNN
//
//  Created by MNN on 2022/09/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NNAPICommonExecution.hpp"
namespace MNN {

NNAPICommonExecution::NNAPICommonExecution(Backend *backend, const Op *op) : Execution(backend), mOp(op) {
    mNNAPIBackend = (NNAPIBackend*)backend;
    mNCHW = mNNAPIBackend->NCHW();
}

ErrorCode NNAPICommonExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    printf("NNAPICommonExecution::onResize\n");
    return NO_ERROR;
}

ErrorCode NNAPICommonExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    return NO_ERROR;
}

std::vector<uint32_t> NNAPICommonExecution::getTensorIdxs(const std::vector<Tensor*>& tensors) {
    std::vector<uint32_t> idxs(tensors.size());
    for (int i = 0; i < tensors.size(); i++) {
        idxs[i] = mNNAPIBackend->getTensorIdx(tensors[i], true);
    }
    return idxs;
}

uint32_t NNAPICommonExecution::buildConstant(const void* data, size_t size, OperandCode dtype, std::vector<uint32_t> dims, const float* scales, int zero) {
    return mNNAPIBackend->buildOperand(data, size, dtype, dims, scales, zero);
}

uint32_t NNAPICommonExecution::buildVector(const std::vector<int32_t>& vec) {
    return buildConstant(vec.data(), vec.size() * sizeof(int), ANEURALNETWORKS_TENSOR_INT32, {static_cast<uint32_t>(vec.size())});
}

uint32_t NNAPICommonExecution::buildVector(const std::vector<float>& vec) {
    return buildConstant(vec.data(), vec.size() * sizeof(float), ANEURALNETWORKS_TENSOR_FLOAT32, {static_cast<uint32_t>(vec.size())});
}

uint32_t NNAPICommonExecution::buildTensor(OperandCode dtype, std::vector<int> dims) {
    std::vector<uint32_t> udims(dims.begin(), dims.end());
    if (!mNCHW) {
        // NCHW -> NHWC
        udims[0] = dims[0];
        udims[1] = dims[2];
        udims[2] = dims[3];
        udims[3] = dims[1];
    }
    return mNNAPIBackend->buildOperand(nullptr, 0, dtype, udims);
}

int NNAPICommonExecution::formatAxis(int axis, const Tensor* t) {
    if (t->dimensions() < 4) {
        return axis;
    }
    if (!mNCHW && TensorUtils::getDescribe(t)->dimensionFormat != MNN_DATA_FORMAT_NHWC) {
        // NCHW -> NHWC
        const int axisChange[4] = {0, 3, 1, 2};
        if (axis > 3) {
            axis = axis;
        } else {
            axis = axisChange[axis];
        }
    }
    return axis;
}
ErrorCode NNAPICommonExecution::buildOperation(int op, const std::vector<uint32_t> &inputs, const std::vector<uint32_t> &outputs) {
    auto name = mOp->name() ? mOp->name()->c_str() : EnumNameOpType(mOp->type());
    return mNNAPIBackend->buildOperation(op, inputs, outputs, name);
}
}; // namespace MNN