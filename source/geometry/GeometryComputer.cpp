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
    // Do nothing
}
GeometryComputer::Context::Context(std::shared_ptr<Backend> allocBackend, bool permitVirtual, MNNForwardType type) {
    mBackend       = allocBackend;
    flatbuffers::FlatBufferBuilder builder(32);
    OpBuilder opBuilder(builder);
    opBuilder.add_type(OpType_Raster);
    auto lastOffset = opBuilder.Finish();
    builder.Finish(lastOffset);
    mRasterOp.reset(new BufferStorage);
    mRasterOp->storage = builder.ReleaseRaw(mRasterOp->allocated_size, mRasterOp->offset);
    mForwardType = type;
}
void GeometryComputer::Context::pushCache(const CommandBuffer& buffer) {
    for (auto cmd : buffer.command) {
        if (cmd->op->type() == OpType_Raster) {
            mRasterCmdCache.emplace_back(cmd);
        }
    }
}
void GeometryComputer::Context::clear() {
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
    srcDes->memoryType = Tensor::InsideDescribe::MEMORY_BACKEND;
    if (mRasterCmdCache.empty()) {
        SharedPtr<Command> cmdP(new Command);
        auto& cmd = *cmdP;
        cmd.op = flatbuffers::GetRoot<Op>(mRasterOp->buffer());
        cmd.buffer = mRasterOp;
        cmd.inputs = {src};
        cmd.outputs = {src};
        cmdBuffer.command.emplace_back(std::move(cmdP));
        return;
    }
    auto iter = mRasterCmdCache.begin() + ((int)mRasterCmdCache.size() - 1);
    auto cmdP = *iter;
    mRasterCmdCache.erase(iter);
    cmdP->inputs[0] = src;
    cmdP->outputs[0] = src;
    cmdBuffer.command.emplace_back(std::move(cmdP));
}

bool DefaultGeometryComputer::onRecompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                         Context& context, CommandBuffer& cmd) const {
    if (1 != cmd.command.size()) {
        return false;
    }
    return true;
}

bool DefaultGeometryComputer::onCompute(const Op* op, const std::vector<Tensor*>& originInputs,
                                        const std::vector<Tensor*>& outputs, GeometryComputer::Context& context,
                                        CommandBuffer& res) const {
    auto inputs = originInputs;
    // Last Command
    SharedPtr<Command> cmdP(new Command);
    auto& cmd = *cmdP;
    cmd.op      = op;
    cmd.inputs  = std::move(inputs);
    cmd.outputs = std::move(outputs);
    res.command.emplace_back(std::move(cmdP));
    return true;
}

class GeometryComputerManager {
public:
    GeometryComputer* search(int type, Runtime::CompilerType compType) {
        if (Runtime::Compiler_Origin == compType) {
            return &mDefault;
        }
        if (Runtime::Compiler_Loop == compType) {
            auto iter = mLoopTable[type].get();
            if (iter != nullptr) {
                return iter;
            }
        }
        // Geometry
        auto iter = mTable[type].get();
        if (iter != nullptr) {
            // FUNC_PRINT(type);
            return iter;
        }
        return &mDefault;
    }
    static void init() {
        gInstance = new GeometryComputerManager;
        gInstance->mTable.resize(OpType_MAX + 1);
        gInstance->mLoopTable.resize(OpType_MAX + 1);
    }
    static GeometryComputerManager* get() {
        return gInstance;
    }
    void insert(std::shared_ptr<GeometryComputer> c, int type, Runtime::CompilerType compType) {
        if (Runtime::Compiler_Geometry == compType) {
            mTable[type] = c;
        } else if (Runtime::Compiler_Loop == compType) {
            mLoopTable[type] = c;
        }
    }
private:
    std::vector<std::shared_ptr<GeometryComputer>> mTable;
    std::vector<std::shared_ptr<GeometryComputer>> mLoopTable;
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
