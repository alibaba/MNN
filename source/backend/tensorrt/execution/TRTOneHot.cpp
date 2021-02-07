//
//  TRTOneHot.cpp
//  MNN
//
//  Created by MNN on 2019/09/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TRTOneHot.hpp"
#include <core/TensorUtils.hpp>
#include "TRTBackend.hpp"
#include "schema/current/MNNPlugin_generated.h"

using namespace std;

namespace MNN {

TRTOneHot::TRTOneHot(Backend *b, const Op *op, const std::vector<Tensor *> &inputs,
                       const std::vector<Tensor *> &outputs)
    : MNN::TRTCommonExecution(b, op) {
    mAxis = op->main_as_OneHotParam()->axis();
}

std::vector<ITensor *> TRTOneHot::onEncode(const std::vector<ITensor *> &xOp) {
#ifdef TRT_LOG
    printf("TRTOneHot in\n");
#endif

    auto plu = createPluginWithOutput(mOutputs);

    auto indices        = mInputs[0];
    auto onValueTensor  = mInputs[2];
    auto offValueTensor = mInputs[3];

    int axis = mAxis;
    if (axis < 0) {
        axis += mOutputs[0]->dimensions();
    }
    int outerSize = 1;
    for (int i = 0; i < axis; ++i) {
        outerSize *= indices->length(i);
    }

    const int innerSize   = indices->elementSize() / outerSize;

    auto dataType    = onValueTensor->getType();

    MNN_ASSERT(offValueTensor->getType() == dataType);
    MNN_ASSERT(offValueTensor->getType() != halide_type_of<int>());

    plu->main.type  = MNNTRTPlugin::Parameter_OneHotInfo;
    plu->main.value = new MNNTRTPlugin::OneHotInfoT;
    auto onehotp     = plu->main.AsOneHotInfo();

    onehotp->outerSize   = outerSize;
    onehotp->innerSize   = innerSize;

    auto interpPlugin = (nvinfer1::IPluginExt *)MNNTRTCreatePlugion(mOp, plu.get());
    nvinfer1::IPluginLayer *plugin = mTrtBackend->getNetwork()->addPluginExt(&xOp[0], mInputs.size(), *((nvinfer1::IPluginExt *)interpPlugin));
    if (plugin == nullptr) {
        printf("Interp plugin == nullptr !!!\n");
    }
    mTrtBackend->pushReleaseLayer(interpPlugin);
    return {plugin->getOutput(0)};

}

TRTCreatorRegister<TypedCreator<TRTOneHot>> __ont_hot_op(OpType_OneHot);

} // namespace MNN
