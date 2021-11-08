//  CommonPlugin.cpp
//  MNN
//
//  Created by MNN on b'2020/08/13'.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#include "CommonPlugin.hpp"
#include <MNN/MNNDefine.h>
#include "BinaryPlugin.hpp"
#include "InterpPlugin.hpp"
#include "PreluPlugin.hpp"
#include "RasterPlugin.hpp"
#include "ScalePlugin.hpp"
#include "ScatterNdPlugin.hpp"
#include "UnaryPlugin.hpp"
#include "GatherPlugin.hpp"
#include "DetectionPostProcessPlugin.hpp"
#include "OneHotPlugin.hpp"
#include "CastPlugin.hpp"
#include "LayerNormPlugin.hpp"

namespace MNN {

static CommonPlugin::Enqueue* create(const Op* op, const MNNTRTPlugin::Plugin* plugin) {
    if (op->type() == OpType_Raster) {
        return new RasterPlugin(plugin);
    }
    if (op->type() == OpType_PReLU || op->type() == OpType_ReLU) {
        return new PreluPlugin(op, plugin);
    }
    if (op->type() == OpType_BinaryOp) {
        return new BinaryPlugin(op, plugin);
    }
    if (op->type() == OpType_Scale) {
        return new ScalePlugin(op, plugin);
    }
    if (op->type() == OpType_ScatterNd) {
        return new ScatterNdPlugin(op, plugin);
    }
    if (op->type() == OpType_Interp) {
        return new InterpPlugin(op, plugin);
    }
    if (op->type() == OpType_UnaryOp) {
        return new UnaryPlugin(op, plugin);
    }
    if (op->type() == OpType_Gather || op->type() == OpType_GatherV2) {
        return new GatherPlugin(op, plugin);
    }
    if (op->type() == OpType_DetectionPostProcess) {
        return new DetectionPostProcessPlugin(op, plugin);
    }
    if (op->type() == OpType_OneHot) {
        return new OneHotPlugin(op, plugin);
    }
    if (op->type() == OpType_Cast) {
        return new CastPlugin(op, plugin);
    }
    if (op->type() == OpType_LayerNorm) {
        return new LayerNormPlugin(op, plugin);
    }
    MNN_PRINT("not find plugin type : %d !!! \n");
    return nullptr;
}
CommonPlugin::CommonPlugin(const void* buffer, size_t size) {
    auto int64Buffer = (int64_t*)buffer;
    auto cBuffer     = (const char*)buffer;
    auto opSize      = int64Buffer[0];
    auto pluginSize  = int64Buffer[1];
    mDataType = (nvinfer1::DataType)int64Buffer[2];
    cBuffer += 3 * sizeof(int64_t);
    mOpBuffer.resize(opSize);
    mPluginBuffer.resize(pluginSize);
    ::memcpy(mOpBuffer.data(), cBuffer, opSize);
    ::memcpy(mPluginBuffer.data(), cBuffer + opSize, pluginSize);
}

int CommonPlugin::initialize() {
    auto plugin = flatbuffers::GetMutableRoot<MNNTRTPlugin::Plugin>(mPluginBuffer.data());
    auto op     = flatbuffers::GetMutableRoot<Op>(mOpBuffer.data());
    mExe.reset(create(op, plugin));
    MNN_ASSERT(nullptr != mExe);
    return 0;
}
void CommonPlugin::terminate() {
    mExe.reset();
}

CommonPlugin::CommonPlugin(const Op* op, const MNNTRTPlugin::PluginT* plugin) {
    {
        std::unique_ptr<OpT> opT(op->UnPack());
        flatbuffers::FlatBufferBuilder builder;
        auto last = Op::Pack(builder, opT.get());
        builder.Finish(last);
        mOpBuffer.resize(builder.GetSize());
        ::memcpy(mOpBuffer.data(), builder.GetBufferPointer(), mOpBuffer.size());
    }
    {
        flatbuffers::FlatBufferBuilder builder;
        auto last = MNNTRTPlugin::Plugin::Pack(builder, plugin);
        builder.Finish(last);
        mPluginBuffer.resize(builder.GetSize());
        ::memcpy(mPluginBuffer.data(), builder.GetBufferPointer(), mPluginBuffer.size());
    }
}
int CommonPlugin::getNbOutputs() const {
    auto plugin = flatbuffers::GetRoot<MNNTRTPlugin::Plugin>(mPluginBuffer.data());
    MNN_ASSERT(nullptr != plugin->outputs());
    return plugin->outputs()->size();
}

size_t CommonPlugin::getSerializationSize() {
    return mOpBuffer.size() + mPluginBuffer.size() + 3 * sizeof(int64_t);
}
void CommonPlugin::serialize(void* buffer) {
    auto int64Buffer = (int64_t*)buffer;
    auto cBuffer     = (char*)buffer;
    int64Buffer[0]   = mOpBuffer.size();
    int64Buffer[1]   = mPluginBuffer.size();
    int64Buffer[2]   = (int64_t)mDataType;
    cBuffer += 3 * sizeof(int64_t);
    ::memcpy(cBuffer, mOpBuffer.data(), mOpBuffer.size());
    ::memcpy(cBuffer + mOpBuffer.size(), mPluginBuffer.data(), mPluginBuffer.size());
}
nvinfer1::Dims CommonPlugin::getOutputDimensions(int index, const nvinfer1::Dims* inputs, int nbInputDims) {
    auto plugin = flatbuffers::GetRoot<MNNTRTPlugin::Plugin>(mPluginBuffer.data());
    MNN_ASSERT(nullptr != plugin->outputs());
    MNN_ASSERT(index < plugin->outputs()->size());
    auto shape = plugin->outputs()->GetAs<MNNTRTPlugin::Shape>(index);
    nvinfer1::Dims res;
    res.nbDims = shape->dim()->size();
    for (int i = 0; i < shape->dim()->size(); ++i) {
        res.d[i] = shape->dim()->data()[i];
    }
    return res;
}

} // namespace MNN
