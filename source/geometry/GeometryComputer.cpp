//
//  GeometryComputer.cpp
//  MNN
//
//  Created by MNN on 2020/04/01.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <mutex>
#include "geometry/GeometryComputer.hpp"
#include "core/Backend.hpp"
#include "core/OpCommonUtils.hpp"
#include "shape/SizeComputer.hpp"
#include "core/TensorUtils.hpp"

namespace MNN {
GeometryComputer::Context::~Context() {
    for (auto& iter : mConstTensors) {
        for (auto& t : iter.second) {
            auto des = TensorUtils::getDescribe(t.get());
            des->backend->onReleaseBuffer(t.get(), Backend::STATIC);
        }
    }
}

GeometryComputer::Context::Context(std::shared_ptr<Backend> allocBackend, bool permitVirtual) {
    mPermitVirtual = permitVirtual;
    mBackend       = allocBackend;
    flatbuffers::FlatBufferBuilder builder;
    OpBuilder opBuilder(builder);
    opBuilder.add_type(OpType_Raster);
    auto lastOffset = opBuilder.Finish();
    builder.Finish(lastOffset);
    mRasterOp.resize(builder.GetSize());
    ::memcpy(mRasterOp.data(), builder.GetBufferPointer(), builder.GetSize());
}

void GeometryComputer::Context::clear() {
    mRasterCache.clear();
    pOutputs.clear();
}
const std::vector<std::shared_ptr<Tensor>>& GeometryComputer::Context::searchConst(const Op* op) const {
    auto iter = mConstTensors.find(op);
    if (iter == mConstTensors.end()) {
        return mEmpty;
    }
    return iter->second;
}
std::shared_ptr<Tensor> GeometryComputer::Context::allocConst(const Op* key, const std::vector<int>& shape,
                                                              halide_type_t type, Tensor::DimensionType dimType) {
    auto iter = mConstTensors.find(key);
    if (iter == mConstTensors.end()) {
        mConstTensors.insert(std::make_pair(key, std::vector<std::shared_ptr<Tensor>>{}));
        iter = mConstTensors.find(key);
    }
    std::shared_ptr<Tensor> tensor(Tensor::createDevice(shape, type, dimType));
    TensorUtils::getDescribe(tensor.get())->usage = Tensor::InsideDescribe::CONSTANT;
    auto res                                      = mBackend->onAcquireBuffer(tensor.get(), Backend::STATIC);
    if (!res) {
        return nullptr;
    }
    TensorUtils::getDescribe(tensor.get())->backend = mBackend.get();
    iter->second.emplace_back(tensor);
    return tensor;
}

Tensor* GeometryComputer::Context::getRasterCacheCreateRecurrse(Tensor* src, CommandBuffer& cmd) {
    auto srcDes = TensorUtils::getDescribe(src);
    if (srcDes->memoryType != Tensor::InsideDescribe::MEMORY_VIRTUAL) {
        return src;
    }
    for (auto& input : srcDes->regions) {
        MNN_ASSERT(input.origin != src);
        auto inputDes = TensorUtils::getDescribe(input.origin);
        while (inputDes->memoryType == Tensor::InsideDescribe::MEMORY_VIRTUAL) {
            if (1 != inputDes->regions.size()) {
                break;
            }
            bool merge = TensorUtils::fuseRegion(inputDes->regions[0], input);
            if (!merge) {
                break;
            }
            inputDes = TensorUtils::getDescribe(input.origin);
        }
        input.origin = getRasterCacheCreateRecurrse(input.origin, cmd);
        if (input.offset != nullptr) {
            input.offset = getRasterCacheCreateRecurrse(input.offset, cmd);
        }
        MNN_ASSERT(TensorUtils::getDescribe(input.origin)->memoryType != Tensor::InsideDescribe::MEMORY_VIRTUAL);
    }
    return getRasterCacheCreate(src, cmd);
}
std::shared_ptr<Tensor> GeometryComputer::Context::getCachedTensor(Tensor* t) {
    auto findIter = mRasterCache.find(t);
    if (findIter != mRasterCache.end()) {
        return findIter->second;
    }
    auto tDes = TensorUtils::getDescribe(t);
    for (auto& iter : mRasterCache) {
        Tensor* s = iter.first;
        bool shapeEqual = s->dimensions() == t->dimensions();
        shapeEqual &= s->getType() == t->getType();
        shapeEqual &= TensorUtils::getDescribe(s)->dimensionFormat  == TensorUtils::getDescribe(t)->dimensionFormat;
        for (int i = 0; i < t->dimensions() && shapeEqual; i++) {
            shapeEqual &= s->length(i) == t->length(i);
        }
        if (!shapeEqual) {
            continue;
        }
        auto sDes = TensorUtils::getDescribe(s);
        if (tDes->regions.size() == sDes->regions.size()) {
            bool equal = true;
            for (int i = 0; i < sDes->regions.size(); i++) {
                auto sReg = sDes->regions[i];
                auto tReg = tDes->regions[i];
                equal &= !::memcmp(&sReg, &tReg, sizeof(sReg));
            }
            if (equal) {
                return iter.second;
            }
        }
    }
    return nullptr;
}
Tensor* GeometryComputer::Context::getRasterCacheCreate(Tensor* src, CommandBuffer& cmdBuffer) {
    auto srcDes = TensorUtils::getDescribe(src);
    if (srcDes->memoryType != Tensor::InsideDescribe::MEMORY_VIRTUAL) {
        return src;
    }
    auto cached = getCachedTensor(src);
    if (cached) {
        return cached.get();
    }
    Command cmd;
    cmd.op = flatbuffers::GetRoot<Op>(mRasterOp.data());
    auto iter = pOutputs.find(src);
    if (iter != pOutputs.end()) {
        auto output = src;
        auto oldDes = TensorUtils::getDescribe(output);
        MNN_ASSERT(oldDes->memoryType == Tensor::InsideDescribe::MEMORY_VIRTUAL);
        std::shared_ptr<Tensor> newTensor(new Tensor);
        TensorUtils::copyShape(output, newTensor.get(), true);
        newTensor->buffer().type = output->getType();
        auto newDes = TensorUtils::getDescribe(newTensor.get());
        newDes->regions = std::move(oldDes->regions);
        newDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
        oldDes->memoryType = Tensor::InsideDescribe::MEMORY_BACKEND;
        cmd.inputs = {newTensor.get()};
        cmd.outputs = {src};
        cmdBuffer.command.emplace_back(std::move(cmd));
        cmdBuffer.extras.emplace_back(newTensor);
        pOutputs.erase(iter);
        return src;
    }
    std::shared_ptr<Tensor> newTensor(new Tensor);
    TensorUtils::copyShape(src, newTensor.get(), true);
    newTensor->buffer().type = src->getType();
    TensorUtils::adjustTensorForCompability(newTensor.get());
    cmd.inputs = {src};
    cmd.outputs = {newTensor.get()};
    cmdBuffer.command.emplace_back(std::move(cmd));
    cmdBuffer.extras.emplace_back(newTensor);
    mRasterCache.insert(std::make_pair(src, newTensor));
    return newTensor.get();
}
bool GeometryComputer::compute(const Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs, GeometryComputer::Context& context,
                               CommandBuffer& cmdBuffer) const {
    std::map<std::shared_ptr<Tensor>, Tensor*> rasterMap;
    auto status = this->onCompute(op, inputs, outputs, context, cmdBuffer);
    for (int i = 0; i < outputs.size(); ++i) {
        auto oldDes = TensorUtils::getDescribe(outputs[i]);
        if (oldDes->memoryType != Tensor::InsideDescribe::MEMORY_VIRTUAL) {
            continue;
        }
        if (!context.supportVirtual()) {
            context.pOutputs.insert(outputs[i]);
        } else {
            if (oldDes->usage == Tensor::InsideDescribe::OUTPUT) {
                context.pOutputs.insert(outputs[i]);
            }
        }
    }
    return status;
}

bool DefaultGeometryComputer::onCompute(const Op* op, const std::vector<Tensor*>& originInputs,
                                        const std::vector<Tensor*>& outputs, GeometryComputer::Context& context,
                                        CommandBuffer& res) const {
    auto inputs = originInputs;
    // Last Command
    Command cmd;
    cmd.op      = op;
    cmd.inputs  = std::move(inputs);
    cmd.outputs = std::move(outputs);
    res.command.emplace_back(std::move(cmd));
    return true;
}

class GeometryComputerManager {
public:
    GeometryComputer* search(int type) {
        auto iter = mTable.find(type);
        if (iter != mTable.end()) {
            // FUNC_PRINT(type);
            return iter->second.get();
        }
        return &mDefault;
    }
    static void init() {
        gInstance = new GeometryComputerManager;
    }
    static GeometryComputerManager* get() {
        return gInstance;
    }
    void insert(std::shared_ptr<GeometryComputer> c, int type) {
        mTable.insert(std::make_pair(type, c));
    }

private:
    std::map<int, std::shared_ptr<GeometryComputer>> mTable;
    static GeometryComputerManager* gInstance;
    DefaultGeometryComputer mDefault;
};

GeometryComputerManager* GeometryComputerManager::gInstance;
void GeometryComputer::registerGeometryComputer(std::shared_ptr<GeometryComputer> comp, std::vector<int> type) {
    auto ins = GeometryComputerManager::get();
    for (auto t : type) {
        ins->insert(comp, t);
    }
}
void GeometryComputer::init() {
    if (nullptr == GeometryComputerManager::get()) {
        GeometryComputerManager::init();
        registerGeometryOps();
    }
}

const GeometryComputer* GeometryComputer::search(int type) {
    return GeometryComputerManager::get()->search(type);
}

Command GeometryComputer::makeRaster(Tensor* input, Tensor* output) {
    flatbuffers::FlatBufferBuilder builder;
    OpBuilder opBuilder(builder);
    opBuilder.add_type(OpType_Raster);
    auto lastOffset = opBuilder.Finish();
    builder.Finish(lastOffset);
    Command cmd;
    cmd.buffer.resize(builder.GetSize());
    ::memcpy(cmd.buffer.data(), builder.GetBufferPointer(), cmd.buffer.size());
    cmd.inputs  = {input};
    cmd.outputs = {output};
    cmd.op      = flatbuffers::GetMutableRoot<Op>(cmd.buffer.data());
    return cmd;
}

} // namespace MNN
