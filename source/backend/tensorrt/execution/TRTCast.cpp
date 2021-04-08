//
//  TRTCast.cpp
//  MNN
//
//  Created by MNN on 2019/09/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TRTCast.hpp"
#include <core/TensorUtils.hpp>
#include "TRTBackend.hpp"
#include "schema/current/MNNPlugin_generated.h"

using namespace std;

namespace MNN {

TRTCast::TRTCast(Backend *b, const Op *op, const std::vector<Tensor *> &inputs,
                     const std::vector<Tensor *> &outputs)
    : MNN::TRTCommonExecution(b, op) {
}

std::vector<ITensor *> TRTCast::onEncode(const std::vector<ITensor *> &xOp) {
    auto plu = createPluginWithOutput(mOutputs);
    auto castPara = mOp->main_as_CastParam();
    DataType srcT = castPara->srcT();
    DataType dstT = castPara->dstT();

    plu->main.type  = MNNTRTPlugin::Parameter_OneHotInfo;
    plu->main.value = new MNNTRTPlugin::OneHotInfoT;
    auto onehotp     = plu->main.AsOneHotInfo();

    onehotp->outerSize = mInputs[0]->elementSize();

    if((srcT == DataType_DT_INT32 || srcT == DataType_DT_INT64) && dstT == DataType_DT_FLOAT){
        auto interpPlugin = (nvinfer1::IPluginExt *)MNNTRTCreatePlugion(mOp, plu.get());
        nvinfer1::IPluginLayer *plugin = mTrtBackend->getNetwork()->addPluginExt(&xOp[0], 1, *((nvinfer1::IPluginExt *)interpPlugin));
        if (plugin == nullptr) {
            printf("Interp plugin == nullptr !!!\n");
        }
        mTrtBackend->pushReleaseLayer(interpPlugin);
        return {plugin->getOutput(0)};
    }else{
        MNN_ASSERT(false);
        return {};
    }

}

TRTCreatorRegister<TypedCreator<TRTCast>> __cast_op(OpType_Cast);
} // namespace MNN
