//
//  CoreMLArgMax.cpp
//  MNN
//
//  Created by MNN on 2021/03/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CoreMLArgMax.hpp"

namespace MNN {

CoreMLArgMax::CoreMLArgMax(MNN::Backend* b, const MNN::Op* op, const std::vector<Tensor*>& inputs,
                           const std::vector<MNN::Tensor*>& outputs)
    : CoreMLCommonExecution(b, op) {
    auto param = op->main_as_ArgMax();
    // These Caffe extensions were not supported by the original Core ML backend.
    mValid = param->softmaxThreshold() == 0 && param->topK() <= 1 && param->outMaxVal() == 0;
    const int rank = inputs[0]->dimensions();
    int axis = param->axis();
    if (axis < 0) {
        axis += rank;
    }
    const bool packed = TensorUtils::getDescribe(inputs[0])->dimensionFormat == MNN_DATA_FORMAT_NC4HW4;
    mValid = mValid && axis >= 0 && axis < rank && !(packed && param->axis() == 0);
    if (mValid && outputs[0]->getType() == halide_type_of<float>()) {
        mValid = inputs[0]->length(axis) <= 2049;
    }
}

ErrorCode CoreMLArgMax::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    MNN_ASSERT(inputs.size() == 1 && outputs.size() == 1);
    auto argmaxParam = mOp->main_as_ArgMax();
    bool keep = outputs[0]->dimensions() == inputs[0]->dimensions();
    auto value = operation(mOp->type() == OpType_ArgMax ? "reduce_argmax" : "reduce_argmin")
                     .input("x", inputs[0])
                     .scalar("axis", argmaxParam->axis())
                     .scalar("keep_dims", keep)
                     .output(outputs[0]->shape(), "index", CORE_ML__SPECIFICATION__MILSPEC__DATA_TYPE__INT32);
    operation("cast")
        .input("x", value)
        .scalar("dtype", std::string(outputs[0]->getType() == halide_type_of<int>() ? "int32" : "fp16"))
        .output(outputs);
    return NO_ERROR;
}

CoreMLTopK::CoreMLTopK(MNN::Backend* b, const MNN::Op* op, const std::vector<Tensor*>& inputs,
                       const std::vector<MNN::Tensor*>& outputs)
    : CoreMLCommonExecution(b, op) {}

ErrorCode CoreMLTopK::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    MNN_ASSERT((inputs.size() == 2 || inputs.size() == 3) && outputs.size() == 2);
    auto inputTensor = inputs[0];
    auto kTensor = inputs[1];
    auto param = mOp->main_as_TopKV2();
    bool ascending = !param->largest();
    if (!kTensor->host<int32_t>() || kTensor->getType() != halide_type_of<int>() ||
        TensorUtils::getRawSize(kTensor) != 1) {
        return NOT_SUPPORT;
    }
    int axis = inputTensor->dimensions() - 1;
    if (inputs.size() == 3) {
        if (inputs[2]->getType() != halide_type_of<int>() || !inputs[2]->host<int>() ||
            TensorUtils::getRawSize(inputs[2]) != 1) {
            return NOT_SUPPORT;
        }
        axis = inputs[2]->host<int>()[0];
    }
    if (axis < 0) {
        axis += inputTensor->dimensions();
    }
    int k = kTensor->host<int32_t>()[0];
    if (axis < 0 || axis >= inputTensor->dimensions() || k <= 0 || k > inputTensor->length(axis)) {
        return NOT_SUPPORT;
    }
    operation("topk")
        .input("x", inputTensor)
        .scalar("k", k)
        .scalar("axis", axis)
        .scalar("ascending", ascending)
        .output(outputs);
    return NO_ERROR;
}

REGISTER_COREML_OP_CREATOR(CoreMLArgMax, OpType_ArgMax);
REGISTER_COREML_OP_CREATOR(CoreMLArgMax, OpType_ArgMin);
REGISTER_COREML_OP_CREATOR(CoreMLTopK, OpType_TopKV2);
} // namespace MNN
