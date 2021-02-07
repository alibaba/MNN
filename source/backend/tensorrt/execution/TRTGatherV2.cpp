//
//  TRTGatherV2.cpp
//  MNN
//
//  Created by MNN on 2019/09/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TRTGatherV2.hpp"
#include <core/TensorUtils.hpp>
#include "TRTBackend.hpp"

using namespace std;

namespace MNN {

TRTGatherV2::TRTGatherV2(Backend *b, const Op *op, const std::vector<Tensor *> &inputs,
                       const std::vector<Tensor *> &outputs)
    : MNN::TRTCommonExecution(b, op) {
    auto params  = inputs[0];
    mAxis = 0;
    if (inputs.size() == 3) {
        const Tensor *axisTensor = inputs[2];
        mAxis                     = axisTensor->host<int32_t>()[0];
    }
    if (mOp->main_type() == OpParameter_Axis) {
        mAxis = mOp->main_as_Axis()->axis();
    }
    MNN_ASSERT(mAxis > -params->buffer().dimensions && mAxis < params->buffer().dimensions);

    if (mAxis < 0) {
        mAxis = params->buffer().dimensions + mAxis;
    }

}

std::vector<ITensor *> TRTGatherV2::onEncode(const std::vector<ITensor *> &xOp) {
#ifdef TRT_LOG
    printf("TRTGatherV2 in\n");
#endif

    auto gather_layer = mTrtBackend->getNetwork()->addGather(*(xOp[0]), *(xOp[1]), mAxis);
    auto output        = gather_layer->getOutput(0);
    return {output};

}

// TRTCreatorRegister<TypedCreator<TRTGatherV2>> __gatherV2_op(OpType_GatherV2);

} // namespace MNN
