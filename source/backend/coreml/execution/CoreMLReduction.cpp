//
//  CoreMLReduction.cpp
//  MNN
//
//  Created by MNN on 2021/03/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CoreMLReduction.hpp"

namespace MNN {

CoreMLReduction::CoreMLReduction(MNN::Backend *b, const MNN::Op *op, const std::vector<Tensor *> &inputs, const std::vector<MNN::Tensor *> &outputs) : CoreMLCommonExecution(b, op) {
    initLayer();
}

ErrorCode CoreMLReduction::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(inputs.size() == 1 && outputs.size() == 1);
    mLayer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_REDUCE;
    mLayer_->reduce = mCoreMLBackend->create<CoreML__Specification__ReduceLayerParams>();
    core_ml__specification__reduce_layer_params__init(mLayer_->reduce);
    auto reduceParam = mOp->main_as_ReductionParam();
    auto reduceAxis = reduceParam->dim()->Get(0);
    auto opType = reduceParam->operation();
    // C = axis [-3], H = axis [-2], W = axis [-1]
    switch (reduceAxis - inputs[0]->dimensions()) {
        case -3:
            mLayer_->reduce->axis = CORE_ML__SPECIFICATION__REDUCE_LAYER_PARAMS__REDUCE_AXIS__C;
            break;
        case -2:
            mLayer_->reduce->axis = CORE_ML__SPECIFICATION__REDUCE_LAYER_PARAMS__REDUCE_AXIS__H;
            break;
        case -1:
            mLayer_->reduce->axis = CORE_ML__SPECIFICATION__REDUCE_LAYER_PARAMS__REDUCE_AXIS__W;
            break;
        default:
            MNN_ERROR("NPU Reduction not support reduceAxis=%d\n", reduceAxis);
            break;
    }
    switch (opType) {
        case ReductionType_MEAN:
            mLayer_->reduce->mode = CORE_ML__SPECIFICATION__REDUCE_LAYER_PARAMS__REDUCE_OPERATION__AVG;
            break;
        case ReductionType_SUM:
            mLayer_->reduce->mode = CORE_ML__SPECIFICATION__REDUCE_LAYER_PARAMS__REDUCE_OPERATION__SUM;
            break;
        case ReductionType_MINIMUM:
            mLayer_->reduce->mode = CORE_ML__SPECIFICATION__REDUCE_LAYER_PARAMS__REDUCE_OPERATION__MIN;
            break;
        case ReductionType_MAXIMUM:
            mLayer_->reduce->mode = CORE_ML__SPECIFICATION__REDUCE_LAYER_PARAMS__REDUCE_OPERATION__MAX;
            break;
        case ReductionType_PROD:
            mLayer_->reduce->mode = CORE_ML__SPECIFICATION__REDUCE_LAYER_PARAMS__REDUCE_OPERATION__PROD;
            break;
            /*
        // Don't support Op
        case ReductionType_ANY:
        case ReductionType_ALL:
             */
        default:
            MNN_ERROR("NPU Reduction not support %s\n", MNN::EnumNameReductionType(opType));
            break;
    }
    setLayerInputsAndOutputs(mLayer_, {mCoreMLBackend->getTensorName(inputs[0])}, {mCoreMLBackend->getTensorName(outputs[0])});
    mCoreMLBackend->addLayer(mLayer_);
    return NO_ERROR;
}

REGISTER_COREML_OP_CREATOR(CoreMLReduction, OpType_Reduction)
} // namespace MNN
