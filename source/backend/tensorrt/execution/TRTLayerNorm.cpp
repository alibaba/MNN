//
//  TRTLayerNorm.cpp
//  MNN
//
//  Created by MNN on 2021/02/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TRTLayerNorm.hpp"
#include <core/TensorUtils.hpp>
#include "TRTBackend.hpp"
#include "schema/current/MNNPlugin_generated.h"

using namespace std;

namespace MNN {

TRTLayerNorm::TRTLayerNorm(Backend *b, const Op *op, const std::vector<Tensor *> &inputs,
                       const std::vector<Tensor *> &outputs)
    : MNN::TRTCommonExecution(b, op) {
}

std::vector<ITensor *> TRTLayerNorm::onEncode(const std::vector<ITensor *> &xOp) {
#ifdef TRT_LOG
    printf("TRTLayerNorm in\n");
#endif

    auto plu = createPluginWithOutput(mOutputs);

    const auto* layer_norm_param = mOp->main_as_LayerNorm();
    int axis_size = layer_norm_param->axis()->size();
    std::vector<int> axis_;
    axis_.resize(axis_size);
    for (int i = 0; i < axis_size; ++i) {
        axis_[i] = layer_norm_param->axis()->Get(i);
    }

    int outter_size_ = 1;
    int inner_size_ = 1;
    int rank = mInputs[0]->dimensions();
    std::vector<int> axis(axis_.size());
    for (int i = 0; i < axis_.size(); ++i) {
        if (axis_[i] < 0) {
            axis[i] += rank;
        }
    }
    std::sort(axis.begin(), axis.end());
    for (int i = 0; i < rank - axis.size(); ++i) {
        outter_size_ *= mInputs[0]->length(i);
    }
    for (int i = rank - axis.size(); i < rank; ++i) {
        inner_size_ *= mInputs[0]->length(i);
    }

    plu->main.type  = MNNTRTPlugin::Parameter_OneHotInfo;
    plu->main.value = new MNNTRTPlugin::OneHotInfoT;
    auto onehotp     = plu->main.AsOneHotInfo();

    onehotp->outerSize   = outter_size_;
    onehotp->innerSize   = inner_size_;

    auto interpPlugin = (nvinfer1::IPluginExt *)MNNTRTCreatePlugion(mOp, plu.get());
    nvinfer1::IPluginLayer *plugin = mTrtBackend->getNetwork()->addPluginExt(&xOp[0], mInputs.size(), *((nvinfer1::IPluginExt *)interpPlugin));
    if (plugin == nullptr) {
        printf("Interp plugin == nullptr !!!\n");
    }
    mTrtBackend->pushReleaseLayer(interpPlugin);
    return {plugin->getOutput(0)};

}

TRTCreatorRegister<TypedCreator<TRTLayerNorm>> __layer_norm_op(OpType_LayerNorm);

} // namespace MNN
