//
//  TRTSelect.cpp
//  MNN
//
//  Created by MNN on 2019/09/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef MNN_USE_TRT7

#include "TRTSelect.hpp"
#include <core/TensorUtils.hpp>
#include "TRTBackend.hpp"

using namespace std;

namespace MNN {

TRTSelect::TRTSelect(Backend *b, const Op *op, const std::vector<Tensor *> &inputs,
                     const std::vector<Tensor *> &outputs)
    : MNN::TRTCommonExecution(b, op) {
}

std::vector<ITensor *> TRTSelect::onEncode(const std::vector<ITensor *> &xOp) {
    MNN_ASSERT(xOp.size() == 3);

    const int maxDims =
        std::max({xOp[0]->getDimensions().nbDims, xOp[1]->getDimensions().nbDims, xOp[2]->getDimensions().nbDims});
    MNN_ASSERT(xOp[0]->getDimensions().nbDims == maxDims);
    MNN_ASSERT(xOp[1]->getDimensions().nbDims == maxDims);
    MNN_ASSERT(xOp[2]->getDimensions().nbDims == maxDims);

    auto select_layer = mTrtBackend->getNetwork()->addSelect(*(xOp[0]), *(xOp[1]), *(xOp[2]));
    return {select_layer->getOutput(0)};
}

TRTCreatorRegister<TypedCreator<TRTSelect>> __select_op(OpType_Select);
} // namespace MNN
#endif
