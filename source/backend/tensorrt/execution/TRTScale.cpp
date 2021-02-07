//
//  TRTScale.cpp
//  MNN
//
//  Created by MNN on 2019/09/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TRTScale.hpp"
#include <core/TensorUtils.hpp>
#include "TRTBackend.hpp"
#include "schema/current/MNNPlugin_generated.h"

using namespace std;

namespace MNN {

TRTScale::TRTScale(Backend *b, const Op *op, const std::vector<Tensor *> &inputs,
                                 const std::vector<Tensor *> &outputs)
    : MNN::TRTCommonExecution(b, op) {
}

std::vector<ITensor *> TRTScale::onEncode(const std::vector<ITensor *> &xOp) {
#ifdef TRT_LOG
    MNN_PRINT("TRTScale in\n");
#endif
    auto plu         = createPluginWithOutput(mOutputs);
    auto preluPlugin = (nvinfer1::IPluginExt *)MNNTRTCreatePlugion(mOp, plu.get());
    nvinfer1::IPluginLayer *plugin =
        mTrtBackend->getNetwork()->addPluginExt(&xOp[0], 1, *((nvinfer1::IPluginExt *)preluPlugin));
    if (plugin == nullptr) {
        MNN_PRINT("plugin == nullptr !!!");
    }
    mTrtBackend->pushReleaseLayer(preluPlugin);
    return {plugin->getOutput(0)};
}

TRTCreatorRegister<TypedCreator<TRTScale>> __scale_op(OpType_Scale);
TRTCreatorRegister<TypedCreator<TRTScale>> __prelu_op(OpType_PReLU);

} // namespace MNN

