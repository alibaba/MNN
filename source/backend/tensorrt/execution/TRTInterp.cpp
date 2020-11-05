//
//  TRTInterp.cpp
//  MNN
//
//  Created by MNN on 2019/09/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

/*
#include "TRTInterp.hpp"
#include <core/TensorUtils.hpp>
#include "TRTBackend.hpp"

using namespace std;

namespace MNN {

TRTInterp::TRTInterp(Backend *b, const Op *op, const std::vector<Tensor *> &inputs,
                               const std::vector<Tensor *> &outputs)
    : MNN::TRTCommonExecution(b,op) {
        auto shape = outputs[0]->shape();
        dims.nbDims = shape.size();
        ::memcpy(dims.d, shape.data(), dims.nbDims * sizeof(int32_t));
    }

std::vector<ITensor*> TRTInterp::onEncode(const std::vector<ITensor*>& xOp) {
  #ifdef TRT_LOG
    printf("TRTInterp in\n");
  #endif

    bool ifAlignCorners  = mOp->main_as_Interp()->alignCorners();
    bool halfPixelCenters = mOp->main_as_Interp()->halfPixelCenters();
    int resizeType = mOp->main_as_Interp()->resizeType();

    nvinfer1::ResizeMode mode = ResizeMode::kNEAREST;
    if(resizeType == 1) {
        mode = ResizeMode::kNEAREST;
    } else if(resizeType == 2) {
        mode = ResizeMode::kLINEAR;
    } else {
        MNN_PRINT("cast trt mode:%d not support\n", resizeType);
        MNN_ASSERT(false);
    }

    //printf("Interp Type:%d in ->: [%d, %d, %d, %d] size:\n", resizeType, mInputs[0]->batch(), mInputs[0]->channel(),
mInputs[0]->height(), mInputs[0]->width());
    // printf("Interp out ->: [%d, %d, %d, %d] size:\n", outputs[0]->batch(), outputs[0]->channel(),
outputs[0]->height(), outputs[0]->width());


    auto interp_layer = mTrtBackend->getNetwork()->addResize(*(xOp[0]));
    interp_layer->setAlignCorners(ifAlignCorners);
    interp_layer->setResizeMode(mode);
    interp_layer->setOutputDimensions(dims);

    return {interp_layer->getOutput(0)};
}

//TRTCreatorRegister<TypedCreator<TRTInterp>> __interp_op(OpType_Interp);

} // namespace MNN
*/