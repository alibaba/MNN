//
//  TRTElewise.cpp
//  MNN
//
//  Created by MNN on 2019/09/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TRTElewise.hpp"
#include <core/TensorUtils.hpp>
#include "TRTBackend.hpp"

using namespace std;

namespace MNN {

TRTElewise::TRTElewise(Backend *b, const Op *op, const std::vector<Tensor *> &inputs,
                       const std::vector<Tensor *> &outputs)
    : MNN::TRTCommonExecution(b, op) {
}

std::vector<ITensor *> TRTElewise::onEncode(const std::vector<ITensor *> &xOp) {
#ifdef TRT_LOG
    printf("TRTElewise in\n");
#endif

    ElementWiseOperation nvOp = ElementWiseOperation::kSUM;
    switch (mOp->main_as_Eltwise()->type()) {
        case EltwiseType_SUM:
            nvOp = ElementWiseOperation::kSUM;
            break;
        case EltwiseType_PROD:
            nvOp = ElementWiseOperation::kPROD;
            break;
        case EltwiseType_MAX:
            nvOp = ElementWiseOperation::kMAX;
            break;
        default:
            break;
    }

    auto elewise_layer = mTrtBackend->getNetwork()->addElementWise(*(xOp[0]), *(xOp[1]), nvOp);
    auto output        = elewise_layer->getOutput(0);
    for (int i = 2; i < xOp.size(); ++i) {
        auto newLayer = mTrtBackend->getNetwork()->addElementWise(*(xOp[2]), *output, nvOp);
        output        = newLayer->getOutput(0);
    }
    return {output};
}

TRTCreatorRegister<TypedCreator<TRTElewise>> __elewise_op(OpType_Eltwise);

} // namespace MNN
