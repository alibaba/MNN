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

static std::shared_ptr<MNNTRTPlugin::PluginT> createPluginWithOutput(const std::vector<Tensor *> &outputs) {
    std::shared_ptr<MNNTRTPlugin::PluginT> plu(new MNNTRTPlugin::PluginT);
    plu->outputs.resize(outputs.size());
    for (int i = 0; i < outputs.size(); ++i) {
        auto shape = outputs[0]->shape();
        plu->outputs[i].reset(new MNNTRTPlugin::ShapeT);
        plu->outputs[i]->dim   = shape;
        plu->outputs[i]->bytes = outputs[i]->getType().bytes();
        plu->outputs[i]->type  = outputs[i]->getType().code;
    }
    return plu;
}

TRTBinary::TRTBinary(Backend *b, const Op *op, const std::vector<Tensor *> &inputs,
                     const std::vector<Tensor *> &outputs)
    : MNN::TRTCommonExecution(b, op) {
    // Do nothing
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
    return {plugin->getOutput(0)};
}

TRTNormalPlugin::TRTNormalPlugin(Backend *b, const Op *op, const std::vector<Tensor *> &inputs,
                                 const std::vector<Tensor *> &outputs)
    : MNN::TRTCommonExecution(b, op) {
}

std::vector<ITensor *> TRTNormalPlugin::onEncode(const std::vector<ITensor *> &xOp) {
#ifdef TRT_LOG
    MNN_PRINT("TRTNormalPlugin in\n");
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

TRTRaster::TRTRaster(Backend *b, const Op *op, const std::vector<Tensor *> &inputs,
                     const std::vector<Tensor *> &outputs)
    : MNN::TRTCommonExecution(b, op) {
    // Do nothing
}

std::vector<ITensor *> TRTRaster::onEncode(const std::vector<ITensor *> &xOp) {
#ifdef TRT_LOG
    MNN_PRINT("TRTRaster in\n");
#endif
    std::vector<ITensor *> inputTensors;
    std::map<const Tensor *, int> tensorMap;
    auto des = TensorUtils::getDescribe(mInputs[0]);
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
    if (!TensorUtils::regionIsFull(mInputs[0])) {
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

TRTScatterNd::TRTScatterNd(Backend *b, const Op *op, const std::vector<Tensor *> &inputs,
                           const std::vector<Tensor *> &outputs)
    : MNN::TRTCommonExecution(b, op) {
    // Do nothing
}

std::vector<ITensor *> TRTScatterNd::onEncode(const std::vector<ITensor *> &xOp) {
#ifdef TRT_LOG
    MNN_PRINT("\n\nTRTScatterNd in\n\n");
#endif
    auto plu        = createPluginWithOutput(mOutputs);
    plu->main.type  = MNNTRTPlugin::Parameter_ScatterNdInfo;
    plu->main.value = new MNNTRTPlugin::ScatterNdInfoT;
    auto scatter    = plu->main.AsScatterNdInfo();

    MNN_ASSERT(mInputs.size() == 3);
    auto indices               = mInputs[0];
    auto updates               = mInputs[1];
    auto shape                 = mInputs[2];
    auto output                = mOutputs[0];
    const int indicesDimension = indices->dimensions();
    scatter->indicesLastDim    = indices->length(indicesDimension - 1);
    scatter->indexes           = indices->elementSize() / scatter->indicesLastDim;

    scatter->accNumber = 1;
    for (int i = indicesDimension - 1; i < updates->dimensions(); ++i) {
        scatter->accNumber *= updates->length(i);
    }

    const int outputElementSize = output->elementSize();
    scatter->outElementSize     = outputElementSize;
    int remainSize              = outputElementSize;
    std::vector<int> temp(scatter->indicesLastDim, 0);
    for (int i = 0; i < scatter->indicesLastDim; ++i) {
        temp[i]    = remainSize / output->length(i);
        remainSize = temp[i];
    }
    scatter->dimsToCount.assign(temp.begin(), temp.end());

    auto scatterNdPlugin = (nvinfer1::IPluginExt *)MNNTRTCreatePlugion(mOp, plu.get());
    nvinfer1::IPluginLayer *plugin =
        mTrtBackend->getNetwork()->addPluginExt(&xOp[0], 2, *((nvinfer1::IPluginExt *)scatterNdPlugin));
    if (plugin == nullptr) {
        MNN_PRINT("scatterNd plugin == nullptr !!!\n");
    }
    mTrtBackend->pushReleaseLayer(scatterNdPlugin);
    return {plugin->getOutput(0)};
}

static float resizeScale(int inputSize, int outputSize, bool isAlign) {
    int corner = 0;
    if (isAlign) {
        corner = 1;
    }
    return (float)(inputSize - corner) / (float)(outputSize - corner);
}

TRTInterp::TRTInterp(Backend *b, const Op *op, const std::vector<Tensor *> &inputs,
                     const std::vector<Tensor *> &outputs)
    : MNN::TRTCommonExecution(b, op) {
    // Do nothing
}

std::vector<ITensor *> TRTInterp::onEncode(const std::vector<ITensor *> &xOp) {
#ifdef TRT_LOG
    MNN_PRINT("\n\nTRTInterp in\n\n");
#endif
    auto plu = createPluginWithOutput(mOutputs);

    int inputChannel = mInputs[0]->channel();
    int inputBatch   = mInputs[0]->batch();

    int inputHeight  = mInputs[0]->height();
    int inputWidth   = mInputs[0]->width();
    int outputHeight = mOutputs[0]->height();
    int outputWidth  = mOutputs[0]->width();

    bool alignCorners = mOp->main_as_Interp()->alignCorners();
    // TODO, not used now
    bool halfPixelCenters = mOp->main_as_Interp()->halfPixelCenters();
    int resizeType        = mOp->main_as_Interp()->resizeType();
    if(resizeType != 1 && resizeType != 2) {
        MNN_PRINT("Interp Type not support!\n");
    }
    plu->main.type  = MNNTRTPlugin::Parameter_InterpInfo;
    plu->main.value = new MNNTRTPlugin::InterpInfoT;
    auto interp     = plu->main.AsInterpInfo();

    interp->inputChannel  = inputChannel;
    interp->heightScale   = resizeScale(inputHeight, outputHeight, alignCorners);
    interp->widthScale    = resizeScale(inputWidth, outputWidth, alignCorners);
    interp->channelBlocks = inputChannel * inputBatch;
    interp->outputWidth   = outputWidth;
    interp->outputH_N     = outputHeight * inputBatch;
    interp->inputHeight   = inputHeight;
    interp->inputWidth    = inputWidth;
    interp->outputHeight  = outputHeight;
    // MNN_PRINT("hs:%f, ws:%f, c:%d, h:%d, w:%d\n", interp->heightScale, interp->widthScale, interp->channelBlocks,
    // interp->outputHeight, interp->outputWidth);

    auto interpPlugin = (nvinfer1::IPluginExt *)MNNTRTCreatePlugion(mOp, plu.get());
    nvinfer1::IPluginLayer *plugin =
        mTrtBackend->getNetwork()->addPluginExt(&xOp[0], 1, *((nvinfer1::IPluginExt *)interpPlugin));
    if (plugin == nullptr) {
        MNN_PRINT("Interp plugin == nullptr !!!\n");
    }
    mTrtBackend->pushReleaseLayer(interpPlugin);
    return {plugin->getOutput(0)};
}


TRTGather::TRTGather(Backend *b, const Op *op, const std::vector<Tensor *> &inputs,
                     const std::vector<Tensor *> &outputs)
    : MNN::TRTCommonExecution(b, op) {
    // Do nothing
}

std::vector<ITensor *> TRTGather::onEncode(const std::vector<ITensor *> &xOp) {
#ifdef TRT_LOG
    MNN_PRINT("\n\nTRTGather in\n\n");
#endif
    auto plu = createPluginWithOutput(mOutputs);

    auto params  = mInputs[0];
    int axis = 0;
    if (mInputs.size() == 3) {
        cudaMemcpy(&axis, (void *)mInputs[2]->deviceId(), sizeof(int), cudaMemcpyDeviceToHost);
    }
    if (mOp->main_type() == OpParameter_Axis) {
        axis = mOp->main_as_Axis()->axis();
    }
    MNN_ASSERT(axis > -params->buffer().dimensions && axis < params->buffer().dimensions);

    if (axis < 0) {
        axis = params->buffer().dimensions + axis;
    }

    auto indices = mInputs[1];
    int outNum      = indices->elementSize();
    int inside = 1;
    int outside = 1;
    for (int i=0; i<axis; ++i) {
        outside *= params->length(i);
    }
    for (int i=axis+1; i<params->dimensions(); ++i) {
        inside *= params->length(i);
    }
    int inpNum = params->length(axis);

    plu->main.type  = MNNTRTPlugin::Parameter_GatherInfo;
    plu->main.value = new MNNTRTPlugin::GatherInfoT;
    auto gather     = plu->main.AsGatherInfo();

    gather->outside = outside;
    gather->inpNum  = inpNum;
    gather->inside  = inside;
    gather->outNum  = outNum;

    auto gatherPlugin = (nvinfer1::IPluginExt *)MNNTRTCreatePlugion(mOp, plu.get());
    nvinfer1::IPluginLayer *plugin =
        mTrtBackend->getNetwork()->addPluginExt(&xOp[0], mInputs.size(), *((nvinfer1::IPluginExt *)gatherPlugin));
    if (plugin == nullptr) {
        MNN_PRINT("Gather plugin == nullptr !!!\n");
    }
    mTrtBackend->pushReleaseLayer(gatherPlugin);
    return {plugin->getOutput(0)};
}

TRTCreatorRegister<TypedCreator<TRTNormalPlugin>> __prelu_op(OpType_PReLU);
TRTCreatorRegister<TypedCreator<TRTBinary>> __binary_op(OpType_BinaryOp);
TRTCreatorRegister<TypedCreator<TRTRaster>> __raster_op(OpType_Raster);
TRTCreatorRegister<TypedCreator<TRTNormalPlugin>> __scale_op(OpType_Scale);
TRTCreatorRegister<TypedCreator<TRTScatterNd>> __scatterNd_op(OpType_ScatterNd);
TRTCreatorRegister<TypedCreator<TRTInterp>> __interp_op(OpType_Interp);
//TRTCreatorRegister<TypedCreator<TRTGather>> __gather_op(OpType_Gather);
//TRTCreatorRegister<TypedCreator<TRTGather>> __gatherv2_op(OpType_GatherV2);

} // namespace MNN
