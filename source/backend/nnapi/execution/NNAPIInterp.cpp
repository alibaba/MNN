//
//  NNAPIInterp.cpp
//  MNN
//
//  Created by MNN on 2022/09/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NNAPIInterp.hpp"

namespace MNN {


NNAPIInterp::NNAPIInterp(MNN::Backend *b, const MNN::Op *op, const std::vector<Tensor *> &inputs, const std::vector<MNN::Tensor *> &outputs) : NNAPICommonExecution(b, op) {
}

ErrorCode NNAPIInterp::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto output = outputs[0];
    auto interpParam = mOp->main_as_Interp();
    // nearstneighbor/bilinear: [input, ow/sw, oh/sh, NCHW/NHWC, aligncorners, halfpixelcenter]
    auto inputIdxs = getTensorIdxs(inputs);
    // shape or scale
#if 1
    inputIdxs.push_back(buildScalar(output->width()));
    inputIdxs.push_back(buildScalar(output->height()));
#else
    inputIdxs.push_back(buildScalar(interpParam->widthScale()));
    inputIdxs.push_back(buildScalar(interpParam->heightScale()));
#endif
    inputIdxs.push_back(buildScalar(mNCHW));
    inputIdxs.push_back(buildScalar(interpParam->alignCorners()));
    // inputIdxs.push_back(buildScalar(interpParam->halfPixelCenters()));
    inputIdxs.push_back(buildScalar(!interpParam->halfPixelCenters()));
    int activateType = -1;
    if (interpParam->resizeType() == 1) {
        activateType = ANEURALNETWORKS_RESIZE_NEAREST_NEIGHBOR;        
    } else if (interpParam->resizeType() == 2) {
        activateType = ANEURALNETWORKS_RESIZE_BILINEAR;
    } else {
        MNN_ERROR("[NNAPI] Interp Don't support [Cubic, NearestneighborRound] mode.");
        return NOT_SUPPORT;
    }
    return buildOperation(activateType, inputIdxs, getTensorIdxs(outputs));
}

REGISTER_NNAPI_OP_CREATOR(NNAPIInterp, OpType_Interp)
} // namespace MNN
