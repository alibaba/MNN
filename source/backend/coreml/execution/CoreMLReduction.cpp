//
//  CoreMLReduction.cpp
//  MNN
//
//  Created by MNN on 2021/03/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CoreMLReduction.hpp"

namespace MNN {

CoreMLReduction::CoreMLReduction(MNN::Backend* b, const MNN::Op* op, const std::vector<Tensor*>& inputs,
                                 const std::vector<MNN::Tensor*>& outputs)
    : CoreMLCommonExecution(b, op) {
    auto param = op->main_as_ReductionParam();
    mValid = param && !inputs.empty() && outputs.size() == 1;
    if (!mValid) {
        return;
    }
    switch (param->operation()) {
        case ReductionType_MEAN:
        case ReductionType_SUM:
        case ReductionType_MINIMUM:
        case ReductionType_MAXIMUM:
        case ReductionType_PROD:
            break;
        case ReductionType_ANY:
        case ReductionType_ALL:
            mValid = inputs[0]->getType() == halide_type_of<int>();
            break;
        default:
            mValid = false;
            break;
    }
}

ErrorCode CoreMLReduction::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    MNN_ASSERT(!inputs.empty() && outputs.size() == 1);
    auto reduceParam = mOp->main_as_ReductionParam();
    auto opType = reduceParam->operation();
    auto dim = reduceParam->dim();
    bool keep_dims = reduceParam->keepDims();
    std::vector<int> axes;
    if (dim) {
        axes.assign(dim->begin(), dim->end());
    } else if (inputs.size() > 1) {
        auto count = TensorUtils::getRawSize(inputs[1]);
        if (inputs[1]->getType() != halide_type_of<int>() || (count > 0 && !inputs[1]->host<int>())) {
            return NOT_SUPPORT;
        }
        if (count > 0) {
            axes.assign(inputs[1]->host<int>(), inputs[1]->host<int>() + count);
        }
    } else {
        for (int axis = 0; axis < inputs[0]->dimensions(); ++axis) {
            axes.push_back(axis);
        }
    }
    if (axes.empty()) {
        operation("identity").input("x", inputs[0]).output(outputs);
        return NO_ERROR;
    }
    for (auto& axis : axes) {
        if (axis < 0) {
            axis += inputs[0]->dimensions();
        }
        if (axis < 0 || axis >= inputs[0]->dimensions()) {
            return NOT_SUPPORT;
        }
    }
    auto value = mCoreMLBackend->getTensorName(inputs[0]);
    if (opType == ReductionType_ANY || opType == ReductionType_ALL) {
        // MNN ANY tests positive values; ALL tests for nonzero values.
        auto predicate = operation(opType == ReductionType_ANY ? "greater" : "not_equal")
                             .input("x", value)
                             .scalar("y", 0, "zero")
                             .output(inputs[0]->shape(), "predicate", CORE_ML__SPECIFICATION__MILSPEC__DATA_TYPE__BOOL);
        value = operation("cast")
                    .input("x", predicate)
                    .scalar("dtype", "int32", "predicate_dtype")
                    .output(inputs[0]->shape(), "predicate_int", CORE_ML__SPECIFICATION__MILSPEC__DATA_TYPE__INT32);
    }
    const char* type = nullptr;
    switch (opType) {
        case ReductionType_MEAN:
            type = "reduce_mean";
            break;
        case ReductionType_SUM:
            type = "reduce_sum";
            break;
        case ReductionType_MINIMUM:
            type = "reduce_min";
            break;
        case ReductionType_MAXIMUM:
            type = "reduce_max";
            break;
        case ReductionType_PROD:
            type = "reduce_prod";
            break;
        case ReductionType_ANY:
            type = "reduce_max";
            break;
        case ReductionType_ALL:
            type = "reduce_min";
            break;
        default:
            return NOT_SUPPORT;
    }
    operation(type).input("x", value).array("axes", axes).scalar("keep_dims", keep_dims).output(outputs);
    return NO_ERROR;
}

REGISTER_COREML_OP_CREATOR(CoreMLReduction, OpType_Reduction);
} // namespace MNN
