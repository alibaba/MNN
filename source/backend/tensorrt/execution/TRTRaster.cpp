//
//  TRTScale.cpp
//  MNN
//
//  Created by MNN on 2019/09/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TRTRaster.hpp"
#include <core/TensorUtils.hpp>
#include "TRTBackend.hpp"
#include "schema/current/MNNPlugin_generated.h"
#include "core/OpCommonUtils.hpp"

using namespace std;

namespace MNN {


TRTRaster::TRTRaster(Backend *b, const Op *op, const std::vector<Tensor *> &inputs,
                     const std::vector<Tensor *> &outputs)
    : MNN::TRTCommonExecution(b, op) {
    // Do nothing
}

std::vector<ITensor *> TRTRaster::onEncode(const std::vector<ITensor *> &xOp) {
#ifdef TRT_LOG
    MNN_PRINT("TRTRaster in\n");
#endif
    OpCommonUtils::rasterInputReset(mInputs, mOutputs[0]);
    std::vector<ITensor *> inputTensors;
    std::map<const Tensor *, int> tensorMap;
    auto des = TensorUtils::getDescribe(mOutputs[0]);
    for (auto &reg : des->regions) {
        if (tensorMap.find(reg.origin) == tensorMap.end()) {
            tensorMap.insert(std::make_pair(reg.origin, tensorMap.size()));
        }
    }
    inputTensors.resize(tensorMap.size());
    for (auto &iter : tensorMap) {
        inputTensors[iter.second] = mTrtBackend->getTensorOps(iter.first);
    }
    auto plu        = createPluginWithOutput(mOutputs);
    plu->main.type  = MNNTRTPlugin::Parameter_RasterInfo;
    plu->main.value = new MNNTRTPlugin::RasterInfoT;
    auto raster     = plu->main.AsRasterInfo();
    raster->regions.resize(des->regions.size());
    for (int i = 0; i < des->regions.size(); ++i) {
        raster->regions[i].reset(new MNNTRTPlugin::RegionT);
        auto &dst = raster->regions[i];
        auto &src = des->regions[i];
        dst->src.reset(new MNNTRTPlugin::ViewT);
        dst->dst.reset(new MNNTRTPlugin::ViewT);
        dst->size        = {src.size[0], src.size[1], src.size[2]};
        dst->index       = tensorMap[src.origin];
        dst->src->offset = src.src.offset;
        dst->src->stride = {src.src.stride[0], src.src.stride[1], src.src.stride[2]};
        dst->dst->offset = src.dst.offset;
        dst->dst->stride = {src.dst.stride[0], src.dst.stride[1], src.dst.stride[2]};
    }
    raster->extra = MNNTRTPlugin::ExtraType_Normal;
    if (!TensorUtils::regionIsFull(mOutputs[0])) {
        raster->extra = MNNTRTPlugin::ExtraType_Fill;
    }
    auto preluPlugin               = (nvinfer1::IPluginExt *)MNNTRTCreatePlugion(mOp, plu.get());
    nvinfer1::IPluginLayer *plugin = mTrtBackend->getNetwork()->addPluginExt(&inputTensors[0], inputTensors.size(),
                                                                             *((nvinfer1::IPluginExt *)preluPlugin));
    if (plugin == nullptr) {
        MNN_PRINT("plugin == nullptr !!!");
    }
    // delete preluPlugin;
#ifdef TRT_LOG
    MNN_PRINT("TRTRaster out\n");
#endif
    mTrtBackend->pushReleaseLayer(preluPlugin);

    return {plugin->getOutput(0)};
}

TRTCreatorRegister<TypedCreator<TRTRaster>> __raster_op(OpType_Raster);

} // namespace MNN

