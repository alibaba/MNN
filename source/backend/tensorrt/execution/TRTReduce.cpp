//
//  TRTReduce.cpp
//  MNN
//
//  Created by MNN on 2019/09/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TRTReduce.hpp"
#include <core/TensorUtils.hpp>
#include "TRTBackend.hpp"

using namespace std;

namespace MNN {

TRTReduce::TRTReduce(Backend *b, const Op *op, const std::vector<Tensor *> &inputs,
                     const std::vector<Tensor *> &outputs)
    : MNN::TRTCommonExecution(b, op) {
    inputDim = inputs[0]->dimensions();
}

std::vector<ITensor *> TRTReduce::onEncode(const std::vector<ITensor *> &xOp) {
#ifdef TRT_LOG
    printf("TRTReduce in\n");
#endif

    ReduceOperation operation = ReduceOperation::kSUM;
    switch (mOp->main_as_ReductionParam()->operation()) {
        case ReductionType_MEAN:
            operation = ReduceOperation::kAVG;
            break;
        case ReductionType_SUM:
            operation = ReduceOperation::kSUM;
            break;
        case ReductionType_MINIMUM:
            operation = ReduceOperation::kMIN;
            break;
        case ReductionType_MAXIMUM:
            operation = ReduceOperation::kMAX;
            break;
        case ReductionType_PROD:
            operation = ReduceOperation::kPROD;
            break;
        default:
            MNN_ASSERT(false);
            break;
    }
    uint32_t mAxis = mOp->main_as_ReductionParam()->dim()->data()[0];
    if (mAxis < 0) {
        mAxis += inputDim;
    }
    MNN_ASSERT(mAxis >= 0 && mAxis < inputDim);

    bool keepdims = mOp->main_as_ReductionParam()->keepDims();

    // printf("reduce type:%d axis:%d keepdim:%d\n", mOp->main_as_ReductionParam()->operation(), mAxis, keepdims);

    auto Reduce_layer = mTrtBackend->getNetwork()->addReduce(*(xOp[0]), operation, 1U << mAxis, keepdims);
    auto output       = Reduce_layer->getOutput(0);
    return {output};
}

TRTCreatorRegister<TypedCreator<TRTReduce>> __Reduce_op(OpType_Reduction);

} // namespace MNN
