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

GeometryComputer::Context::Context(std::shared_ptr<Backend> allocBackend, bool permitVirtual, MNNForwardType type) {
    mPermitVirtual = permitVirtual;
    mBackend       = allocBackend;
    flatbuffers::FlatBufferBuilder builder;
    OpBuilder opBuilder(builder);
    opBuilder.add_type(OpType_Raster);
    auto lastOffset = opBuilder.Finish();
    builder.Finish(lastOffset);
    mRasterOp.resize(builder.GetSize());
    ::memcpy(mRasterOp.data(), builder.GetBufferPointer(), builder.GetSize());
    mForwardType = type;
}

void GeometryComputer::Context::clear() {
    pOutputs.clear();
    for (auto& t : mTempConstTensors) {
        auto des = TensorUtils::getDescribe(t.get());
        des->backend->onReleaseBuffer(t.get(), Backend::STATIC);
    }
    mTempConstTensors.clear();
}
const std::vector<std::shared_ptr<Tensor>>& GeometryComputer::Context::searchConst(const Op* op) {
    auto iter = mConstTensors.find(op);
    if (iter == mConstTensors.end()) {
        mConstTensors.insert(std::make_pair(op, std::vector<std::shared_ptr<Tensor>>{}));
        return mEmpty;
    }
    return iter->second;
}
std::shared_ptr<Tensor> GeometryComputer::Context::allocConst(const Op* key, const std::vector<int>& shape,
                                                              halide_type_t type, Tensor::DimensionType dimType) {
    std::shared_ptr<Tensor> tensor(Tensor::createDevice(shape, type, dimType));
    TensorUtils::getDescribe(tensor.get())->usage = Tensor::InsideDescribe::CONSTANT;
    auto res                                      = mBackend->onAcquireBuffer(tensor.get(), Backend::STATIC);
    if (!res) {
        return nullptr;
    }
    TensorUtils::getDescribe(tensor.get())->backend = mBackend.get();
    auto iter = mConstTensors.find(key);
    if (iter != mConstTensors.end()) {
        iter->second.emplace_back(tensor);
    } else {
        mTempConstTensors.emplace_back(tensor);
    }
    return tensor;
}

bool GeometryComputer::Context::allocTensor(Tensor* tensor) {
    auto res = mBackend->onAcquireBuffer(tensor, Backend::STATIC);
    if (!res) {
        return false;
    }
    TensorUtils::getDescribe(tensor)->usage = Tensor::InsideDescribe::CONSTANT;
    TensorUtils::getDescribe(tensor)->backend = mBackend.get();
    return true;
}

void GeometryComputer::Context::getRasterCacheCreateRecurrse(Tensor* src, CommandBuffer& cmd) {
    auto srcDes = TensorUtils::getDescribe(src);
    if (srcDes->memoryType != Tensor::InsideDescribe::MEMORY_VIRTUAL) {
        return;
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
        getRasterCacheCreateRecurrse(input.origin, cmd);
    }
    getRasterCacheCreate(src, cmd);
}
void GeometryComputer::Context::getRasterCacheCreate(Tensor* src, CommandBuffer& cmdBuffer) {
    auto srcDes = TensorUtils::getDescribe(src);
    if (srcDes->memoryType != Tensor::InsideDescribe::MEMORY_VIRTUAL) {
        return;
    }
    Command cmd;
    cmd.op = flatbuffers::GetRoot<Op>(mRasterOp.data());
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
            context.pOutputs.emplace_back(outputs[i]);
        } else {
            if (oldDes->usage == Tensor::InsideDescribe::OUTPUT) {
                context.pOutputs.emplace_back(outputs[i]);
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
    GeometryComputer* search(int type, Runtime::CompilerType compType) {
        if (Runtime::Compiler_Origin == compType) {
            return &mDefault;
        }
        if (Runtime::Compiler_Loop == compType) {
            auto iter = mLoopTable.find(type);
            if (iter != mLoopTable.end()) {
                return iter->second.get();
            }
        }
        // Geometry
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
    void insert(std::shared_ptr<GeometryComputer> c, int type, Runtime::CompilerType compType) {
        if (Runtime::Compiler_Geometry == compType) {
            mTable.insert(std::make_pair(type, c));
        } else if (Runtime::Compiler_Loop == compType) {
            mLoopTable.insert(std::make_pair(type, c));
        }
    }
private:
    std::map<int, std::shared_ptr<GeometryComputer>> mTable;
    std::map<int, std::shared_ptr<GeometryComputer>> mLoopTable;
    static GeometryComputerManager* gInstance;
    DefaultGeometryComputer mDefault;
};

GeometryComputerManager* GeometryComputerManager::gInstance;
void GeometryComputer::registerGeometryComputer(std::shared_ptr<GeometryComputer> comp, std::vector<int> type, Runtime::CompilerType compType) {
    auto ins = GeometryComputerManager::get();
    for (auto t : type) {
        ins->insert(comp, t, compType);
    }
}
void GeometryComputer::init() {
    if (nullptr == GeometryComputerManager::get()) {
        GeometryComputerManager::init();
        registerGeometryOps();
    }
}

const GeometryComputer* GeometryComputer::search(int type, Runtime::CompilerType compType) {
    return GeometryComputerManager::get()->search(type, compType);
}
} // namespace MNN
