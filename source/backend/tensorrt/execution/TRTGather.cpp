//
//  TRTScale.cpp
//  MNN
//
//  Created by MNN on 2019/09/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TRTGather.hpp"
#include <core/TensorUtils.hpp>
#include "TRTBackend.hpp"
#include "schema/current/MNNPlugin_generated.h"

using namespace std;

namespace MNN {
TRTGather::TRTGather(Backend *b, const Op *op, const std::vector<Tensor *> &inputs,
                     const std::vector<Tensor *> &outputs)
    : MNN::TRTCommonExecution(b, op) {
    // Do nothing
}

std::vector<ITensor *> TRTGather::onEncode(const std::vector<ITensor *> &xOp) {
#ifdef TRT_LOG
    MNN_PRINT("\n\n TRTGather in\n\n");
#endif
    auto plu = createPluginWithOutput(mOutputs);
    auto params  = mInputs[0];
    int axis = 0;
    bool input3 = false;
    if (mInputs.size() == 3) {
        //hangxing TODO : only support axis == 0 now
        input3 = true;
    }
    if (mOp->main_type() == OpParameter_Axis) {
        axis = mOp->main_as_Axis()->axis();
    }
    MNN_ASSERT(axis > -params->buffer().dimensions && axis < params->buffer().dimensions);

    if (axis < 0) {
        axis = params->buffer().dimensions + axis;
    }
    
    auto indices = mInputs[1];
    int N      = indices->elementSize();
    int inside = 1;
    int outside = 1;
    for (int i=0; i<axis; ++i) {
        outside *= params->length(i);
    }
    for (int i=axis+1; i<params->dimensions(); ++i) {
        inside *= params->length(i);
    }
    int limit = params->length(axis);

    const int insideStride = inside;
    const int outputOutsideStride = inside * N;
    const int inputOutsideStride = inside * mInputs[0]->length(axis);

    plu->main.type  = MNNTRTPlugin::Parameter_GatherInfo;
    plu->main.value = new MNNTRTPlugin::GatherInfoT;
    auto gather     = plu->main.AsGatherInfo();

    gather->limit  = limit;
    gather->insideStride  = inside;
    gather->N  = N;
    gather->outputOutsideStride  = outputOutsideStride;
    gather->inputOutsideStride  = inputOutsideStride;
    gather->outside  = outside;
    gather->input3  = input3;

    auto gatherPlugin = (nvinfer1::IPluginExt *)MNNTRTCreatePlugion(mOp, plu.get());
    nvinfer1::IPluginLayer *plugin =
        mTrtBackend->getNetwork()->addPluginExt(&xOp[0], mInputs.size(), *((nvinfer1::IPluginExt *)gatherPlugin));
    if (plugin == nullptr) {
        MNN_PRINT("Gather plugin == nullptr !!!\n");
    }
    mTrtBackend->pushReleaseLayer(gatherPlugin);
    return {plugin->getOutput(0)};
}

TRTCreatorRegister<TypedCreator<TRTGather>> __gather_op(OpType_Gather);
TRTCreatorRegister<TypedCreator<TRTGather>> __gatherv2_op(OpType_GatherV2);

} // namespace MNN

