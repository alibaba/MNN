//
//  TRTBinary.cpp
//  MNN
//
//  Created by MNN on 2019/09/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TRTBinary.hpp"
#include <core/TensorUtils.hpp>
#include "schema/current/MNNPlugin_generated.h"
using namespace std;
namespace MNN {

TRTBinary::TRTBinary(Backend *b, const Op *op, const std::vector<Tensor *> &inputs,
                     const std::vector<Tensor *> &outputs)
    : MNN::TRTCommonExecution(b, op) {
    mActivationType = op->main_as_BinaryOp()->activationType();
}

std::vector<ITensor *> TRTBinary::onEncode(const std::vector<ITensor *> &xOp) {
#ifdef TRT_LOG
    MNN_PRINT("\n\nTRTBinary in\n\n");
#endif
    auto plu                            = createPluginWithOutput(mOutputs);
    plu->main.type                      = MNNTRTPlugin::Parameter_BroadCastInfo;
    plu->main.value                     = new MNNTRTPlugin::BroadCastInfoT;
    plu->main.AsBroadCastInfo()->input0 = mInputs[0]->elementSize() == 1;
    plu->main.AsBroadCastInfo()->input1 = mInputs[1]->elementSize() == 1;
    auto binaryPlugin                   = (nvinfer1::IPluginExt *)MNNTRTCreatePlugion(mOp, plu.get());
    nvinfer1::IPluginLayer *plugin =
        mTrtBackend->getNetwork()->addPluginExt(&xOp[0], 2, *((nvinfer1::IPluginExt *)binaryPlugin));
    if (plugin == nullptr) {
        MNN_PRINT("binary plugin == nullptr !!!\n");
    }
    mTrtBackend->pushReleaseLayer(binaryPlugin);

    auto output = plugin->getOutput(0);
    if (mActivationType == 1) {
        mActivationLayer = mTrtBackend->getNetwork()->addActivation(*output, ActivationType::kRELU);
        return {mActivationLayer->getOutput(0)};
    }
    return {output};
}

TRTCreatorRegister<TypedCreator<TRTBinary>> __binary_op(OpType_BinaryOp);
} // namespace MNN
