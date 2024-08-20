//
//  Pipeline.cpp
//  MNN
//
//  Created by MNN on 2019/01/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <string.h>
#include "core/Pipeline.hpp"
#include "core/Backend.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "core/WrapExecution.hpp"
#include "geometry/GeometryComputerUtils.hpp"
#include "shape/SizeComputer.hpp"
#include "core/OpCommonUtils.hpp"

// TODO: Find better way for debug
//#define MNN_OP_SEPERATE
//#define MNN_PIPELINE_DEBUG
namespace MNN {

// FIXME: Move in Backend
static bool _supportQuant(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, MNNForwardType type) {
    auto otype = op->type();
    switch (otype) {
        case OpType_Convolution:
        case OpType_ConvolutionDepthwise:
            if (op->main_as_Convolution2D() && op->main_as_Convolution2D()->weight() != nullptr) {
                return false;
            } else {
                return true;
            }
        case OpType_Deconvolution:
            if (op->main_as_Convolution2D() && op->main_as_Convolution2D()->weight() != nullptr) {
                return false;
            } else {
                return true;
            }
        case OpType_ConvInt8:
        case OpType_DepthwiseConvInt8:
            return true;
            // case OpType_Eltwise:
        case OpType_Raster:
        {
            for (auto input : inputs) {
                if (TensorUtils::getDescribe(input)->quantAttr.get() != TensorUtils::getDescribe(outputs[0])->quantAttr.get()) {
                    return false;
                }
            }
            return true;
        }
        case OpType_ReLU:
            if (TensorUtils::getDescribe(inputs[0])->quantAttr.get() != TensorUtils::getDescribe(outputs[0])->quantAttr.get()) {
                return false;
            }
            // now just relu without slope support quant
            if ((op->main_as_Relu() == nullptr) || op->main_as_Relu()->slope() == 0.f) {
                return true;
            } else {
                return false;
            }
        case OpType_Pooling:
            if (op->main_as_Pool() && op->main_as_Pool()->type() == PoolType_MAXPOOL ) {
                return true;
            } else if (op->main_as_Pool() && op->main_as_Pool()->type() == PoolType_AVEPOOL) {
                return true;
            } else {
                return false;
            }
        case OpType_BinaryOp:
            return true;
        case OpType_Softmax:
            return true;
        case OpType_Scale:
            return true;
        case OpType_Interp:
            return true;
        case OpType_LayerNorm:
            return true;
        case OpType_UnaryOp:
            if (op->main_as_UnaryOp()->tableInt8() || op->main_as_UnaryOp()->opType() == UnaryOpOperation_NEG || op->main_as_UnaryOp()->opType() == UnaryOpOperation_ABS || op->main_as_UnaryOp()->opType() == UnaryOpOperation_SIGN) {
                return true;
            } else {
                return false;
            }
        case OpType_PReLU:
            return true;
        default:
            break;
    }
    return false;
}

OperatorInfo::OperatorInfo() {
    mContent = new Info;
    MNN_ASSERT(nullptr != mContent);
}
OperatorInfo::~OperatorInfo() {
    delete mContent;
}

const std::string& OperatorInfo::name() const {
    return mContent->name;
}

const std::string& OperatorInfo::type() const {
    return mContent->type;
}

float OperatorInfo::flops() const {
    return mContent->flops;
}
static Backend::StorageType _getTensorStorageType(const Tensor* tensor, bool outputStatic) {
    auto des   = TensorUtils::getDescribe(tensor);
    auto usage = des->usage;
    if (TensorUsage::OUTPUT == usage && outputStatic) {
        return Backend::STATIC;
    }
    if (TensorUsage::CONSTANT == usage || TensorUsage::INPUT == usage || TensorUsage::TRAINABLE == usage) {
        return Backend::DYNAMIC_SEPERATE;
    }
    return Backend::DYNAMIC;
}

static bool _needRelease(const Tensor* tensor, bool inputOutside) {
    auto des   = TensorUtils::getDescribe(tensor);
    auto desO   = TensorUtils::getDescribeOrigin(tensor);
    auto usage = des->usage;
    if (0 != des->useCount) {
        return false;
    }
    if (des->memoryType == Tensor::InsideDescribe::MEMORY_HOST || des->memoryType == Tensor::InsideDescribe::MEMORY_OUTSIDE) {
        return false;
    }
    if (nullptr == desO->getBackend()) {
        return false;
    }
    if (inputOutside) {
        return usage == Tensor::InsideDescribe::NORMAL;
    }
    if (tensor->buffer().type.code == halide_type_handle) {
        return false;
    }
    if (TensorUsage::CONSTANT == usage || TensorUsage::TRAINABLE == usage || TensorUsage::OUTPUT == usage) {
        return false;
    }
    return true;
}
static void _releaseTensor(Tensor* origin, bool mAllocInput, int group) {
    auto des = TensorUtils::getDescribe(origin);
    if (des->usage != Tensor::InsideDescribe::CONSTANT) {
        des->useCount -= 1;
    }
    if (des->group != group) {
        return;
    }
    auto needRelease = _needRelease(origin, !mAllocInput);
    if (needRelease) {
        TensorUtils::getDescribeOrigin(origin)->mem = nullptr;
    }
}

static bool _allocTensor(Tensor* t, Backend* curBackend, bool outputStatic, int group) {
    auto memoryType = _getTensorStorageType(t, outputStatic);
    auto bn         = TensorUtils::getDescribeOrigin(t)->getBackend();
    auto des = TensorUtils::getDescribe(t);
    if (des->group != group) {
        return true;
    }
    if (nullptr == TensorUtils::getDescribeOrigin(t)->mem.get()) {
        TensorUtils::setLinearLayout(t);
        auto res     = curBackend->onAcquireBuffer(t, memoryType);
        return res;
    }
    return true;
}

void Pipeline::UnitInfo::setUp(const Command& command, int index, const Op* originOp, int totalIndex) {
    if (nullptr != command.op->name()) {
        mContent->name = command.op->name()->str();
    } else {
        if (nullptr != originOp && nullptr != originOp->name()) {
            char buffer[20];
            sprintf(buffer, "%d", index);
            mContent->name = originOp->name()->str() + "_raster_" + buffer;
        } else {
            char buffer[20];
            sprintf(buffer, "_raster_%d", totalIndex);
            mContent->name = buffer;
        }
    }
#ifdef MNN_OP_SEPERATE
    if (command.op->type() == OpType_UnaryOp) {
        mContent->type = EnumNameUnaryOpOperation(command.op->main_as_UnaryOp()->opType());
    } else if (command.op->type() == OpType_BinaryOp) {
        mContent->type = EnumNameBinaryOpOperation((BinaryOpOperation)(command.op->main_as_BinaryOp()->opType()));
    } else if (command.op->type() == OpType_Reduction) {
        mContent->type = EnumNameReductionType(command.op->main_as_ReductionParam()->operation());
    } else {
        mContent->type = EnumNameOpType(command.op->type());
    }
#else
    mContent->type = EnumNameOpType(command.op->type());
#endif
#ifndef MNN_BUILD_MINI
    mContent->flops = SizeComputer::computeFlops(command.op, command.inputs, command.outputs);
#endif
}

Pipeline::Pipeline(const std::string& externalFile, Schedule::PipelineInfo&& info, bool allocInput, bool outputStatic, const TuningAttr& tune, const Runtime* rt, const Runtime* cpuRt, int geometryMask)
#ifndef MNN_BUILD_MINI
    : mContext(geometryMask, info.first.cache.second, info.first.cache.first->type(), info.first.info.user ? info.first.info.user->precision :  BackendConfig::Precision_Normal), mUseGeometry(rt->onGetCompilerType()) {
#else
{
#endif
    mExternalFile = externalFile;
    rt->onCheckInfo(info.first.info);
    mRuntime = rt;
    mCpuRuntime = cpuRt;
    mTuneAttr = tune;
    mAllocInput    = allocInput;
    mOutputStatic  = outputStatic;
    mInfo          = std::move(info);
    mIsQuantModel = false;
    for (auto& iter : mInfo.second) {
        for (auto t : iter.outputs) {
            if (TensorUtils::getDescribe(t)->quantAttr.get() != nullptr) {
                mIsQuantModel = true;
                break;
            }
        }
        for (auto t : iter.inputs) {
            if (TensorUtils::getDescribe(t)->quantAttr.get() != nullptr) {
                mIsQuantModel = true;
                break;
            }
        }
        if (mIsQuantModel) {
            break;
        }
    }

}
ErrorCode Pipeline::encode(bool supportDebug, bool permitCodegen) {
    auto& mBackend = mInfo.first.cache.first;
    auto& mBackupBackend = mInfo.first.cache.second;
    // Static Model just copy info to command buffer
    if (!mInfo.first.needComputeGeometry) {
        for (int i=0; i<mInfo.second.size(); ++i) {
            auto& info = mInfo.second[i];
            std::shared_ptr<Command> cmd(new Command);
            cmd->op      = info.op;
            if (cmd->op->type() == OpType_Raster) {
                // Compability for Origin Static Model
                cmd->outputs  = info.outputs;
                if (TensorUtils::getDescribe(info.outputs[0])->regions.empty() && info.inputs.size() > 0 && TensorUtils::getDescribe(info.inputs[0])->regions.size() > 0) {
                    TensorUtils::getDescribe(info.outputs[0])->regions = std::move(TensorUtils::getDescribe(info.inputs[0])->regions);
                    TensorUtils::setRasterInputs(cmd.get());
                } else {
                    cmd->inputs  = info.inputs;
                }
            } else {
                cmd->inputs  = info.inputs;
                cmd->outputs = info.outputs;
            }
            info.executeBuffer.command = {cmd};
        }
    } else {
#ifndef MNN_BUILD_MINI
        mContext.clear();
        FileLoader l(mExternalFile.c_str());
        /** Size Compute and compute Const Begin */
        auto res = GeometryComputerUtils::shapeComputeAndGeometryTransform(&l, mInfo.second, mContext, mInfo.first.cache.second, mUseGeometry, false, permitCodegen);
        if (res != NO_ERROR) {
            return res;
        }
#endif
    }
    // Propagate Scale and insert new command
    if (mIsQuantModel && (mBackend->type() == MNN_FORWARD_CPU || mBackend->type() == MNN_FORWARD_CPU_EXTENSION || mBackend->type() == MNN_FORWARD_CUDA || mBackend->type() == MNN_FORWARD_NN || mBackend->type() == MNN_FORWARD_OPENCL)) {
        // get propagate map
        using PropagateMap = std::map<const MNN::Tensor*, std::set<const MNN::Tensor*>>;
        PropagateMap forwardMap, backwardMap;
        auto insertPropagateMap = [](PropagateMap& propagateMap, const Tensor* s, const Tensor* t) {
            if (propagateMap.find(s) == propagateMap.end()) {
                propagateMap[s] = std::set<const Tensor*>({t});
            } else {
                propagateMap[s].insert(t);
            }
        };
        std::set<OpType> propagateOpTypes = { OpType_Raster, OpType_ReLU, OpType_ReLU6, OpType_Pooling,
                                              OpType_Interp, OpType_CropAndResize, OpType_ROIPooling, OpType_Gather,
                                              OpType_GatherV2, OpType_GatherV2, OpType_ScatterNd};
        for (auto& info : mInfo.second) {
            auto& buffer = info.executeBuffer;
            for (const auto& cmdP : buffer.command) {
                auto& cmd = *cmdP;
                const auto type = cmd.op->type();
                const auto output = cmd.outputs[0];
                if (propagateOpTypes.find(type) != propagateOpTypes.end()) {
                    for (auto t : cmd.inputs) {
                        insertPropagateMap(forwardMap, t, output);
                        insertPropagateMap(backwardMap, output, t);
                    }
                }
            }
        }
        auto getStart = [&forwardMap, &backwardMap](bool forward) {
            auto& propagateMap = forward ? forwardMap : backwardMap;
            auto& antiMap = forward ? backwardMap : forwardMap;
            // delete N->1 Map of Op
            for (const auto& iter : antiMap) {
                if (iter.second.size() > 1) {
                    for (auto t : iter.second) {
                        auto res = propagateMap.find(t);
                        if (res != propagateMap.end()) {
                            propagateMap.erase(res);
                        }
                    }
                }
            }
            std::set<const Tensor*> root, leaf, start;
            for (const auto& iter : propagateMap) {
                root.insert(iter.first);
                for (auto t : iter.second) {
                    leaf.insert(t);
                }
            }
            std::set_difference(root.begin(), root.end(), leaf.begin(), leaf.end(), std::inserter(start, start.begin()));
            return start;
        };
        auto forwardStart = getStart(true);
        auto backwardStart = getStart(false);
        // propagate scale
        auto propagateScale = [](PropagateMap& propagateMap, std::set<const Tensor*>& start) {
            std::function<bool(const Tensor*)> scalePropagate = [&propagateMap, &scalePropagate](const Tensor* t) {
                if (TensorUtils::getDescribe(t)->quantAttr.get() == nullptr) {
                    return false;
                }
                if (propagateMap.find(t) == propagateMap.end()) {
                    return false;
                }
                bool change = false;
                for (auto x : propagateMap[t]) {
                    if (TensorUtils::getDescribe(x)->quantAttr != TensorUtils::getDescribe(t)->quantAttr) {
                        TensorUtils::getDescribe(x)->quantAttr = TensorUtils::getDescribe(t)->quantAttr;
                        change = true;
                    }
                    change |= scalePropagate(x);
                }
                return change;
            };
            bool change = false;
            for (auto t : start) {
                change |= scalePropagate(t);
            }
            return change;
        };
        for (int i = 0; i < 3 && (propagateScale(forwardMap, forwardStart) || propagateScale(backwardMap, backwardStart)); i++);
        
        // Insert cast
        std::map<const Tensor*, Tensor*> cachedCastTensor;
        for (auto& info : mInfo.second) {
            auto bufferCommand = std::move(info.executeBuffer.command);
            bool hasConvert = false;
            for (auto cmdP : bufferCommand) {
                auto& cmd = *cmdP;
                auto& outputs = cmd.outputs;
                auto& inputs = cmd.inputs;
                auto opType = cmd.op->type();
                // Check if need use quant op
                DataType runType = DataType_DT_FLOAT;
                bool useQuant = false;
                if (outputs.size() == 1) {
                    // Quant: output and all input has quantAttr and op support
                    if (TensorUtils::getDescribe(outputs[0])->quantAttr != nullptr) {
                        useQuant = _supportQuant(cmd.op, inputs, outputs, mBackend->type());
                    }
                    if (useQuant) {
                        for (auto t : inputs) {
                            if (TensorUtils::getDescribe(t)->quantAttr == nullptr) {
                                useQuant = false;
                                break;
                            }
                        }
                    }
                }
                if (useQuant) {
                    runType = DataType_DT_INT8;
                }
                
                for (auto o : outputs) {
                    auto quan = TensorUtils::getDescribe(o)->quantAttr;
                    if (nullptr != quan) {
                        TensorUtils::getDescribe(o)->type = runType;
                    }
                }
                auto makeCommand = [&cachedCastTensor, &info](CommandBuffer& cmdBuffer, Tensor* input, DataType runType) {
                    if (cachedCastTensor.find(input) != cachedCastTensor.end()) {
                        return cachedCastTensor[input];
                    }
                    std::shared_ptr<Tensor> wrapTensor(new Tensor);
                    TensorUtils::copyShape(input, wrapTensor.get(), true);
                    TensorUtils::setLinearLayout(wrapTensor.get());
                    auto des = TensorUtils::getDescribe(wrapTensor.get());
                    auto originDes = TensorUtils::getDescribe(input);
                    if (originDes->quantAttr != nullptr) {
                        des->quantAttr.reset(new QuantAttr);
                        *des->quantAttr = *originDes->quantAttr;
                        des->type = runType;
                    }
                    cmdBuffer.extras.emplace_back(wrapTensor);
                    std::shared_ptr<Command> command(new Command);
                    command->inputs = {input};
                    command->outputs = {wrapTensor.get()};
                    info.cacheBuffer.hasWrap = true;
                    flatbuffers::FlatBufferBuilder builder;
                    OpBuilder opB(builder);
                    if (runType == DataType_DT_INT8) {
                        opB.add_type(OpType_FloatToInt8);
                    } else {
                        opB.add_type(OpType_Int8ToFloat);
                    }
                    builder.Finish(opB.Finish());
                    command->buffer.reset(new BufferStorage);
                    command->buffer->storage = builder.ReleaseRaw(command->buffer->allocated_size, command->buffer->offset);
                    command->op = flatbuffers::GetRoot<Op>(command->buffer->buffer());
                    info.executeBuffer.command.emplace_back(std::move(command));
                    return wrapTensor.get();
                };
                // judge is it need CastWrap
                if (OpType_Raster == opType) {
                    for (int v=0; v<cmd.inputs.size(); ++v) {
                        auto input = cmd.inputs[v];
                        bool needCast = CPUBackend::getDataType(input) != runType;
                        if (needCast) {
                            cmd.inputs[v] = makeCommand(info.executeBuffer, input, runType);
                        }
                    }
                } else {
                    for (int i = 0; i < cmd.inputs.size(); i++) {
                        if (OpCommonUtils::opNeedContent(cmd.op, i) && inputs[i]->getType() != halide_type_of<int>()) {
                            bool needCast = CPUBackend::getDataType(inputs[i]) != runType;
                            if (needCast) {
                                cmd.inputs[i] = makeCommand(info.executeBuffer, inputs[i], runType);
                            }
                        }
                    }
                }
                info.executeBuffer.command.emplace_back(cmdP);
            }
        }
    }
    /** Prepare DebugInfo*/
    if (supportDebug) {
        mFlops = 0.0f;
        int totalIndex = 0;
        for (auto& info : mInfo.second) {
            auto& buffer = info.executeBuffer;
            int index = 0;
            for (auto& cmdP : buffer.command) {
                auto& cmd = *cmdP;
                cmd.info.reset(new UnitInfo);
                static_cast<UnitInfo*>(cmd.info.get())->setUp(cmd, index++, info.op, totalIndex++);
                mFlops += cmd.info->flops();
            }
        }
    }
    return NO_ERROR;
}

void Pipeline::_pushTuningTask(std::vector<Schedule::OpCacheInfo>&& initInfos) {
    // Dup Tensors for initInfos;
    std::map<Tensor*, std::shared_ptr<Tensor>> holdTensors;
    auto& mBackend = mInfo.first.cache.first;
    auto& mBackupBackend = mInfo.first.cache.second;

    for (auto& info : initInfos) {
        auto& buffer = info.executeBuffer;
        if (info.type == Schedule::CONSTANT) {
            continue;
        }
        for (int v=0; v<buffer.command.size(); ++v) {
            auto iterP = buffer.command[v];
            auto& iter = *iterP;
            buffer.command[v].reset(new Command);
            iterP = buffer.command[v];
            iterP->inputs = iter.inputs;
            iterP->outputs = iter.outputs;
            iterP->op = iter.op;
            iterP->buffer = iter.buffer;
#ifndef MNN_BUILD_MINI
            if (iter.op->type() == OpType_Raster) {
                iterP->buffer = mContext.mRasterOp;
            }
#endif
            auto copyTensor = [&](std::vector<Tensor*>& tensors) {
                for (int v=0; v<tensors.size(); ++v) {
                    auto t = tensors[v];
                    auto findIter = holdTensors.find(t);
                    if (findIter != holdTensors.end()) {
                        tensors[v] = findIter->second.get();
                        continue;
                    }
                    std::shared_ptr<Tensor> newTensor(new Tensor);
                    newTensor->buffer().type = t->getType();
                    TensorUtils::copyShape(t, newTensor.get(), true);
                    TensorUtils::getDescribe(newTensor.get())->regions = TensorUtils::getDescribe(t)->regions;
                    tensors[v] = newTensor.get();
                    holdTensors.insert(std::make_pair(t, newTensor));
                    holdTensors.insert(std::make_pair(newTensor.get(), newTensor));
                }
            };
            copyTensor(iterP->inputs);
            copyTensor(iterP->outputs);
        }
    }
    // Make async task for tuning
    const_cast<Runtime*>(mRuntime)->mCancelled = false;
    auto future = std::async(std::launch::async, [&, this](std::vector<Schedule::OpCacheInfo>&& infos, std::map<Tensor*, std::shared_ptr<Tensor>>&& tensors, std::shared_ptr<Backend> backend, const std::atomic_bool& cancelled) -> int {
        FileLoader loader(mExternalFile.c_str());

        backend->onClearBuffer();
        backend->onResizeBegin();
        std::vector<std::shared_ptr<BufferStorage>> tmpStorage;
        for (auto& info : infos) {
            if (info.type == Schedule::CONSTANT) {
                continue;
            }
            auto& buffer = info.executeBuffer;
            for (auto& iterP : buffer.command) {
                if(cancelled) {
                    return -1;
                }
                auto& iter = *iterP;
                // FIXME: Remove onMaskOpReady in future
                const_cast<Runtime*>(mRuntime)->onMaskOpReady(iter.inputs, iter.outputs, iter.op);
                std::shared_ptr<BufferStorage> tmp;
                // If create op failed, we can also mask the op is ready for runtime
                auto exePtr = OpCommonUtils::createExecutionWithExternal(backend.get(), iter.inputs, iter.outputs, iter.op, &loader, tmp);
                std::shared_ptr<Execution> exe(exePtr);
                if (nullptr == exe) {
                    continue;
                }
                if (nullptr != tmp) {
                    tmpStorage.emplace_back(tmp);
                }
                std::vector<Tensor*> forRelease;
                std::shared_ptr<void> _defer(nullptr, [&forRelease](void*) {
                    for (auto t : forRelease) {
                        TensorUtils::getDescribeOrigin(t)->mem = nullptr;
                    }
                });
                // Alloc inputs and outputs
                for (auto t : iter.inputs) {
                    auto des = TensorUtils::getDescribe(t);
                    bool allocRes = backend->onAcquireBuffer(t, Backend::DYNAMIC);
                    if (!allocRes) {
                        return -1;
                    }
                    forRelease.emplace_back(t);
                }
                for (auto t : iter.outputs) {
                    bool allocRes = backend->onAcquireBuffer(t, Backend::DYNAMIC);
                    if (!allocRes) {
                        return -1;
                    }
                    forRelease.emplace_back(t);
                }
                auto code = exe->onResize(iter.inputs, iter.outputs);
                if (NO_ERROR != code) {
                    return -1;
                }
            }
        }
        backend->onResizeEnd();
        return 0;
    }, std::move(initInfos), std::move(holdTensors), mBackend, std::ref(const_cast<Runtime*>(mRuntime)->mCancelled));
    const_cast<Runtime*>(mRuntime)->setAsyncWork(std::move(future));
}

static ErrorCode _createExecutions(Schedule::PipelineInfo& mInfo, const std::string& externalFile, std::vector<std::shared_ptr<BufferStorage>>& extraStorage) {
    FileLoader loader(externalFile.c_str());
    auto& mBackend = mInfo.first.cache.first;
    auto& mBackupBackend = mInfo.first.cache.second;
    for (auto& info : mInfo.second) {
        if (!info.computeCache.needComputeShape) {
            continue;
        }
        if (info.type == Schedule::CONSTANT) {
            continue;
        }
        auto& buffer = info.executeBuffer;
        // MNN_PRINT("before resize, mInfo.second size:%lu, command size:%lu,op type:%s, op name:%s\n", mInfo.second.size(), buffer.command.size(), EnumNameOpType(info.op->type()), info.op->name()->c_str());
        for (auto& iterP : buffer.command) {
            auto& iter = *iterP;
            // Create exe
            // Find Cache
            bool cached    = false;
            if (nullptr == iter.execution) {
                /** Cache origin execution for fast resize*/
                auto exeIter = info.executionCache.find(iter.op);
                if (exeIter != info.executionCache.end()) {
                    iter.execution = exeIter->second;
                    cached         = true;
                }
            }
            std::shared_ptr<BufferStorage> tmpStorage;
            if (nullptr == iter.execution) {
                iter.execution.reset(OpCommonUtils::createExecutionWithExternal(mBackend.get(), iter.inputs, iter.outputs, iter.op, &loader, tmpStorage));
            }
            if (nullptr == iter.execution) {
                // Try Backup
                iter.execution.reset(OpCommonUtils::createExecutionWithExternal(mBackupBackend.get(), iter.inputs, iter.outputs, iter.op, &loader, tmpStorage));
                if (nullptr == iter.execution) {
                    if (mInfo.first.reportError) {
                        MNN_ERROR("Create execution error : %d\n", iter.op->type());
                    }
                    return NOT_SUPPORT;
                }
            }
            if (nullptr != tmpStorage.get()) {
                extraStorage.emplace_back(tmpStorage);
            }
            // invalid means memory alloc failed
            if (!iter.execution->valid()) {
                iter.execution = nullptr;
                iter.execution = nullptr;
                return OUT_OF_MEMORY;
            }
            if ((!cached) && iter.buffer == nullptr && (iter.op->type() != OpType_Raster) && (iter.op->type() != OpType_BinaryOp)) {
                info.executionCache.insert(std::make_pair(iter.op, iter.execution));
            }
        }
    }
    return NO_ERROR;
}
static void _SetTensorBackend(Schedule::PipelineInfo& mInfo, bool ownInputs) {
    // Clear Valid Tensor's Backend
    for (int infoIndex=0; infoIndex < mInfo.second.size(); ++infoIndex) {
        auto& info = mInfo.second[infoIndex];
        if (info.type == Schedule::CONSTANT) {
            continue;
        }
        auto& buffer = info.executeBuffer;
        // MNN_PRINT("before resize, mInfo.second size:%lu, command size:%lu,op type:%s, op name:%s\n", mInfo.second.size(), buffer.command.size(), EnumNameOpType(info.op->type()), info.op->name()->c_str());
        for (int iterIndex=0; iterIndex<buffer.command.size(); ++iterIndex) {
            auto& iterP = buffer.command[iterIndex];
            auto& iter = *iterP;
            if (iter.op->type() == OpType_Copy) {
                continue;
            }
            auto curBackend = iter.execution->backend();
            if (ownInputs) {
                for (auto t : iter.inputs) {
                    auto des = TensorUtils::getDescribeOrigin(t);
                    if (nullptr == des->mem.get()) {
                        des->setBackend(nullptr);
                    }
                }
            }
            for (auto t : iter.outputs) {
                auto des = TensorUtils::getDescribeOrigin(t);
                if (nullptr == des->mem.get()) {
                    des->setBackend(nullptr);
                }
            }
        }
    }

    // Set Tensor's Backend
    for (auto& info : mInfo.second) {
        if (info.type == Schedule::CONSTANT) {
            continue;
        }
        auto& buffer = info.executeBuffer;
        // MNN_PRINT("before resize, mInfo.second size:%lu, command size:%lu,op type:%s, op name:%s\n", mInfo.second.size(), buffer.command.size(), EnumNameOpType(info.op->type()), info.op->name()->c_str());
        for (auto& iterP : buffer.command) {
            auto& iter = *iterP;
            if (iter.op->type() == OpType_Copy) {
                continue;
            }
            auto curBackend = iter.execution->backend();
            if (ownInputs) {
                for (auto t : iter.inputs) {
                    auto des = TensorUtils::getDescribeOrigin(t);
                    if (nullptr == des->mem.get() && nullptr == des->getBackend()) {
                        des->setBackend(curBackend);
                    }
                }
            }
            for (auto t : iter.outputs) {
                auto des = TensorUtils::getDescribeOrigin(t);
                if (nullptr == des->mem.get() && nullptr == des->getBackend()) {
                    des->setBackend(curBackend);
                }
            }
        }
    }
}
static void _makeCopyOp(std::shared_ptr<BufferStorage>& copyOp) {
    if (copyOp.get() == nullptr) {
        flatbuffers::FlatBufferBuilder builder(32);
        OpBuilder builder_(builder);
        builder_.add_type(OpType_Copy);
        builder.Finish(builder_.Finish());
        copyOp.reset(new BufferStorage);
        copyOp->storage = builder.ReleaseRaw(copyOp->allocated_size, copyOp->offset);
    }
}
static ErrorCode _InsertCopy(Schedule::PipelineInfo& mInfo, std::map<Tensor*, std::shared_ptr<Tensor>>& mCacheConstTensors, Pipeline::WrapTensorCache& shapeFixConstCache, bool ownInput, bool permitCodegen) {
    std::shared_ptr<BufferStorage> copyOp;
    for (auto iterP = shapeFixConstCache.begin(); iterP != shapeFixConstCache.end();) {
        auto& iter = *iterP;
        if (iter.second.first.lock() == nullptr) {
            // Has released, remove cache
            iterP = shapeFixConstCache.erase(iterP);
            continue;
        }
        auto des = iter.first.first;
        bool needReset = true;
        if (des->usage == Tensor::InsideDescribe::CONSTANT && ((des->stageMask & Tensor::InsideDescribe::CONTENT_NOT_CHANGE) != 0)) {
            // If the tensor is not compute in shape-geometry stage, needn't recopy it
            needReset = false;
        }
        if (needReset) {
            TensorUtils::getDescribeOrigin(iter.second.second.get())->setBackend(nullptr);
            TensorUtils::getDescribeOrigin(iter.second.second.get())->mem = nullptr;
        }
        iterP++;
    }
    for (auto& info : mInfo.second) {
        if (info.type == Schedule::CONSTANT) {
            continue;
        }
        auto& buffer = info.executeBuffer;
        if (buffer.command.empty()) {
            continue;
        }
        auto commands = std::move(buffer.command);
        for (auto& iterP : commands) {
            auto& iter = *iterP;
            if (iter.op->type() == OpType_Copy) {
                continue;
            }
            // Check If need wrap
            auto curBackend = iter.execution->backend();
#ifdef MNN_PIPELINE_DEBUG
            if (nullptr != iter.op->name()) {
                MNN_PRINT("%s Run on %d\n", iter.op->name()->c_str(), curBackend->type());
            }
#endif
            iter.workInputs = iter.inputs;
            for (int v=0; v<iter.inputs.size(); ++v) {
                auto t = iter.inputs[v];
                auto des = TensorUtils::getDescribe(t);
                if (WrapExecution::needWrap(t, curBackend)) {
                    do {
                        Tensor* newTensor = nullptr;
                        if (!des->isMutable) {
                            newTensor = WrapExecution::copyConstCache(t, curBackend, mCacheConstTensors, permitCodegen);
                            if (nullptr != newTensor) {
                                iter.workInputs[v] = newTensor;
                                break;
                            }
                        }
                        if (!ownInput) {
                            if (TensorUtils::getDescribe(t)->usage == Tensor::InsideDescribe::INPUT) {
                                auto inputCacheIter = mInfo.first.inputTensorCopyCache.find(t);
                                if (inputCacheIter != mInfo.first.inputTensorCopyCache.end()) {
                                    auto& tensorCache = inputCacheIter->second;
                                    if (nullptr == std::get<0>(tensorCache) || WrapExecution::needWrap(std::get<0>(tensorCache), curBackend)) {
                                        std::shared_ptr<Tensor> wrapTensor = WrapExecution::makeCopyTensor(t, curBackend);
                                        TensorUtils::getDescribe(wrapTensor.get())->usage = Tensor::InsideDescribe::CONSTANT;
                                        std::get<0>(tensorCache) = wrapTensor.get();
                                        std::get<1>(tensorCache) = wrapTensor;
                                        std::get<2>(tensorCache) = true;
                                        std::get<3>(tensorCache) = true;
                                    }
                                    iter.workInputs[v] = std::get<0>(tensorCache);
                                    if (std::get<2>(tensorCache)) {
                                        auto allocRes = curBackend->onAcquireBuffer(std::get<1>(tensorCache).get(), Backend::STATIC);
                                        if (!allocRes) {
                                            return OUT_OF_MEMORY;
                                        }
                                        std::get<2>(tensorCache) = false;
                                    }
                                    break;
                                }
                            }
                        }
                        {
                            auto titer = shapeFixConstCache.find(std::make_pair(des, curBackend));
                            if (titer != shapeFixConstCache.end()) {
                                newTensor = titer->second.second.get();
                            } else {
                                std::shared_ptr<MNN::Tensor> tensor(new Tensor);
                                shapeFixConstCache.insert(std::make_pair(std::make_pair(des, curBackend), std::make_pair(std::weak_ptr<Tensor::InsideDescribe::NativeInsideDescribe>(TensorUtils::getDescribeOrigin(t)->mContent), tensor)));
                                newTensor = tensor.get();
                            }
                            iter.workInputs[v] = newTensor;
                        }
                        auto newMemory = TensorUtils::getDescribeOrigin(newTensor);
                        if (newMemory->getBackend() != nullptr) {
                            // The memory has been init, skip it
                            break;
                        }
                        TensorUtils::copyShape(t, newTensor, true, true);
                        if (des->usage == Tensor::InsideDescribe::CONSTANT) {
                            TensorUtils::getDescribe(newTensor)->usage = des->usage;
                            auto tempRes = WrapExecution::allocAndCopy(curBackend, t, newTensor);
                            if (!tempRes) {
                                return OUT_OF_MEMORY;
                            }
                            break;
                        }
                        newMemory->setBackend(curBackend);
                        auto copyWrap = WrapExecution::makeCopyExecution(curBackend, mInfo.first.cache.second.get());
                        _makeCopyOp(copyOp);
                        std::shared_ptr<Command> cmdP(new Command);
                        auto& cmd = *cmdP;
                        cmd.buffer = copyOp;
                        cmd.workInputs  = {t};
                        cmd.workOutputs = {newTensor};
                        cmd.op      = flatbuffers::GetRoot<Op>(cmd.buffer->buffer());
                        cmd.execution.reset(copyWrap);
                        buffer.command.emplace_back(cmdP);
                    } while(false);
                }
            }
            buffer.command.emplace_back(iterP);
            iter.workOutputs = iter.outputs;
            for (int v=0; v<iter.workOutputs.size(); ++v) {
                auto t = iter.workOutputs[v];
                if (WrapExecution::needWrap(t, curBackend)) {
                    auto copyWrap = WrapExecution::makeCopyExecution(curBackend, mInfo.first.cache.second.get());
                    std::shared_ptr<Tensor> newTensor(new Tensor);
                    TensorUtils::copyShape(t, newTensor.get(), true, true);
                    iterP->workOutputs[v] = newTensor.get();
                    _makeCopyOp(copyOp);
                    std::shared_ptr<Command> cmdP(new Command);
                    auto& cmd = *cmdP;
                    cmd.buffer = copyOp;
                    cmd.workInputs  = {newTensor.get()};
                    cmd.workOutputs = {t};
                    cmd.op      = flatbuffers::GetRoot<Op>(cmd.buffer->buffer());
                    buffer.extras.emplace_back(newTensor);
                    cmd.execution.reset(copyWrap);
                    buffer.command.emplace_back(cmdP);
                    for(int i = 0; i < iter.inputs.size(); ++i){
                        if(t == iter.inputs[i]){
                            iterP->workOutputs[v] = iter.workInputs[i];
                            cmd.workInputs = {iter.workInputs[i]};
                        }
                    }
                }
            }
        }
    }
    return NO_ERROR;
}

void Pipeline::_recycleDynamicMemory(Command* command) {
    for (auto& t : command->workOutputs) {
        auto memoryType = _getTensorStorageType(t, mOutputStatic);
        if (Backend::DYNAMIC == memoryType) {
            TensorUtils::getDescribeOrigin(t)->mem = nullptr;
        }
    }
    for (auto& t : command->workInputs) {
        auto memoryType = _getTensorStorageType(t, mOutputStatic);
        if (Backend::DYNAMIC == memoryType) {
            TensorUtils::getDescribeOrigin(t)->mem = nullptr;
        }
    }
}
void Pipeline::openResizeCheck() {
    for (auto& info : mInfo.second) {
        info.computeCache.open();
    }
}

ErrorCode Pipeline::fixResizeCache() {
    for (auto& info : mInfo.second) {
        if (info.type == Schedule::CONSTANT && (!info.computeCache.needExecuteConst)) {
            info.executeBuffer.command.clear();
            info.executeBuffer.extras.clear();
            info.cacheBuffer.command.clear();
            info.cacheBuffer.extras.clear();
        }
    }
    auto res = mInfo.first.cache.first->onSelectDynamicAllocator(1, 2);
    res = res && mInfo.first.cache.second->onSelectDynamicAllocator(1, 2);
    if (!res) {
        MNN_PRINT("%d backend don't support resize fix optimize\n", mInfo.first.cache.first->type());
        return NOT_SUPPORT;
    }
    size_t totalNumber = 0;
    size_t fixNumber = 0;
    // Mask begin
    for (auto& info : mInfo.second) {
        auto& buffer = info.executeBuffer;
        if (info.type != Schedule::CONSTANT) {
            totalNumber += buffer.command.size();
        }
        if ((!info.computeCache.canCache()) && info.computeCache.needComputeShape) {
            // If the session has been resized and the op is checked will change shape, set as shape mutable
            info.computeCache.close(false);
            continue;
        }
        info.computeCache.close(true);
        if (info.type == Schedule::CONSTANT) {
            continue;
        }
        // TODO: OCL and Vulkan don't support input vary
        bool notSupportInputVarying = !OpCommonUtils::supportDynamicInputMemory(mInfo.first.cache.first->type());
        for (int cmdIndex=0; cmdIndex<buffer.command.size(); ++cmdIndex) {
            auto& cmd = *buffer.command[cmdIndex];
            cmd.group = 1;
            if (notSupportInputVarying) {
                for (auto t : cmd.workInputs) {
                    if (TensorUtils::getDescribe(t)->group < 0 || TensorUtils::getDescribe(t)->usage != Tensor::InsideDescribe::NORMAL) {
                        cmd.group = 0;
                        break;
                    }
                }
            }
            if (1 == cmd.group) {
                fixNumber++;
            }
            for (auto t : cmd.workInputs) {
                if (TensorUtils::getDescribe(t)->group == 0) {
                    TensorUtils::getDescribe(t)->group = 1;
                }
            }
            for (auto t : cmd.workOutputs) {
                TensorUtils::getDescribe(t)->group = 1;
            }
        }
    }
    // Mask End
    _allocForTensor(1, true);

    mInfo.first.cache.first->onSelectDynamicAllocator(0, 2);
    res && mInfo.first.cache.second->onSelectDynamicAllocator(0, 2);
    MNN_PRINT("Fix: %d - Total: %d, rate = %f\n", fixNumber, totalNumber, (float)fixNumber / (float)totalNumber);
    return NO_ERROR;
}
ErrorCode Pipeline::_allocForTensor(int index, bool allocInput) {
#ifdef MNN_PIPELINE_DEBUG
    int resizeNumber = 0;
#endif
    // Compute RefCount Begin
    for (auto& info : mInfo.second) {
        if (info.type == Schedule::CONSTANT) {
            continue;
        }
        auto& buffer = info.executeBuffer;
        // MNN_PRINT("before resize, mInfo.second size:%lu, command size:%lu,op type:%s, op name:%s\n", mInfo.second.size(), buffer.command.size(), EnumNameOpType(info.op->type()), info.op->name()->c_str());
        for (auto& iterP : buffer.command) {
            auto& iter = *iterP;
            for (auto t : iter.workInputs) {
                auto des = TensorUtils::getDescribe(t);
                if (des->usage != Tensor::InsideDescribe::CONSTANT) {
                    des->useCount = 0;
                }
            }
        }
    }
    for (auto& info : mInfo.second) {
        if (info.type == Schedule::CONSTANT) {
            continue;
        }
        auto& buffer = info.executeBuffer;
        for (auto& iterP : buffer.command) {
            auto& iter = *iterP;
            for (auto t : iter.workInputs) {
                auto des = TensorUtils::getDescribe(t);
                if (des->usage != Tensor::InsideDescribe::CONSTANT) {
                    des->useCount += 1;
                }
            }
        }
    }
    // Compute RefCount End
    auto& mBackend = mInfo.first.cache.first;
    auto& mBackupBackend = mInfo.first.cache.second;
    mBackend->onResizeBegin();
    mBackupBackend->onResizeBegin();
    for (auto& info : mInfo.second) {
        if (info.type == Schedule::CONSTANT) {
            continue;
        }
        auto& buffer = info.executeBuffer;
        for (int cmdIndex=0; cmdIndex < buffer.command.size(); ++cmdIndex) {
            auto& iterP = buffer.command[cmdIndex];
            auto& iter = *iterP;
#ifdef MNN_PIPELINE_DEBUG
            auto memory = const_cast<Runtime*>(mRuntime)->onGetMemoryInMB();
            if (nullptr != info.op->name()) {
                MNN_PRINT("%f, before Resize: %s - %d\n", memory, info.op->name()->c_str(), cmdIndex);
            }
#endif
            // Alloc for Tensors
            auto curBackend = iter.execution->backend();
            if (allocInput) {
                for (auto t : iter.workInputs) {
                    auto allocRes = _allocTensor(t, curBackend, mOutputStatic, index);
                    if (!allocRes) {
                        return OUT_OF_MEMORY;
                    }
                }
            }
            {
                for (auto t : iter.workOutputs) {
                    auto res = _allocTensor(t, curBackend, mOutputStatic, index);
                    if (!res) {
                        return OUT_OF_MEMORY;
                    }
                }
            }
#ifdef MNN_PIPELINE_DEBUG
            if (iter.info != nullptr) {
                MNN_PRINT("before Resize 2, calling: %s - %d \n", iter.info->name().c_str(), cmdIndex);
            }
#endif
            if (iter.group == index) {
#ifdef MNN_PIPELINE_DEBUG
                resizeNumber++;
#endif
                auto code = iter.execution->onResize(iter.workInputs, iter.workOutputs);
                if (NO_ERROR != code) {
#ifdef MNN_PIPELINE_DEBUG
                    MNN_ERROR("Pipeline Resize error: %d\n", code);
#endif
                    if (iter.info.get()) {
                        MNN_ERROR("Resize error for type = %s, name = %s \n", iter.info->type().c_str(), iter.info->name().c_str());
                    }
                    return code;
                }
            }
            // Free mid tensor
            for (auto t : iter.workInputs) {
                _releaseTensor(t, allocInput, index);
            }
        }
    }
    // Recycle All Dynamic Tensor
    for (auto& info : mInfo.second) {
        auto& buffer = info.executeBuffer;
        for (auto& c : buffer.command) {
            if (c->group != index) {
                continue;
            }
            _recycleDynamicMemory(c.get());
        }
    }
    auto code = mBackend->onResizeEnd();
    if (code != NO_ERROR) {
        return code;
    }
#ifdef MNN_PIPELINE_DEBUG
    MNN_PRINT("Resize %d op for index: %d\n", resizeNumber, index);
#endif
    code = mBackupBackend->onResizeEnd();
    return code;
}
ErrorCode Pipeline::allocMemory(bool firstMalloc, bool forbidReplace) {
    // MNN_PRINT("allocMemory mtype:%d, cpubackendType:%d, cpuBackend runtime:%p\n", mBackend->type(), mBackupBackend->type(), mBackupBackend->getRuntime());
    if (!firstMalloc) {
        // For session setNeedMalloc, if session's output is set as some input, It may cause error
        // Dup des to avoid it
        for (auto& info : mInfo.second) {
            auto& buffer = info.executeBuffer;
            for (const auto& infoP : buffer.command) {
                auto& info = *infoP;
                for (auto t : info.workOutputs) {
                    if (!TensorUtils::getDescribe(t)->isMutable) {
                        continue;
                    }
                    auto des = TensorUtils::getDescribe(t);
                    auto usage = des->usage;
                    if (TensorUtils::getDescribeOrigin(t)->mContent.use_count() > 1 && usage != Tensor::InsideDescribe::CONSTANT) {
                        TensorUtils::getDescribeOrigin(t)->mem = nullptr;
                        auto res = TensorUtils::getDescribeOrigin(t)->getBackend()->onAcquireBuffer(t, Backend::STATIC);
                        if (!res) {
                            return OUT_OF_MEMORY;
                        }
                    }
                }
            }
        }
        if (OpCommonUtils::supportDynamicInputMemory(mInfo.first.cache.first->type()) && (!mInfo.first.inputBackendChange)) {
            return NO_ERROR;
        }
    }

    /* Create Execution Begin */
    auto& mBackend = mInfo.first.cache.first;
    auto& mBackupBackend = mInfo.first.cache.second;
    mBackend->onClearBuffer();
    mBackupBackend->onClearBuffer();
    // Check If we need a lone time for init
    if (mBackend->type() != MNN_FORWARD_CPU && mBackend->type() != MNN_FORWARD_CPU_EXTENSION && mTuneAttr.autoSetOpType) {
        Runtime::OpInfo dstInfo;
        int currentInitCount = 0;
        std::vector<Schedule::OpCacheInfo> initInfos;
        for (auto& info : mInfo.second) {
            if (info.type == Schedule::CONSTANT) {
                continue;
            }
            auto& buffer = info.executeBuffer;
            for (auto& iterP : buffer.command) {
                auto& iter = *iterP;
                dstInfo.initCostLong = false;
                mRuntime->onMeasure(iter.inputs, iter.outputs, iter.op, dstInfo);
                if (dstInfo.initCostLong) {
                    initInfos.emplace_back(info);
                    currentInitCount++;
                    break;
                }
            }
            if (currentInitCount >= mTuneAttr.maxTuningNumber) {
                break;
            }
        }
        if (currentInitCount > 0) {
            MNN_PRINT("Turn back to cpu\n");
            // Reset execution
            for (auto& info : mInfo.second) {
                info.executionCache.clear();
                for (auto& iterP : info.executeBuffer.command) {
                    iterP->execution = nullptr;
                    iterP->execution = nullptr;
                    _recycleDynamicMemory(iterP.get());
                }
            }
            if (!mRuntime->hasAsyncWork()) {
                _pushTuningTask(std::move(initInfos));
            }
            mBackend.reset(mCpuRuntime->onCreate(nullptr));
        }
    }
    {
        auto code = _createExecutions(mInfo, mExternalFile, mExternalStorage);
        if (NO_ERROR != code) {
            return code;
        }
    }
    /* Create Execution End */

    _SetTensorBackend(mInfo, mAllocInput);
    // Insert Wrap If needed
    {
        auto insertCode = _InsertCopy(mInfo, mCacheConstTensors, mWrapTensors, mAllocInput, forbidReplace);
        if (NO_ERROR != insertCode) {
            return insertCode;
        }
    }
    /* Insert Wrap End*/

    return _allocForTensor(0, mAllocInput);
}

void Pipeline::_copyInputs() {
    for (auto& iter : mInfo.first.inputTensorCopyCache) {
        auto& tensorCache = iter.second;
        if (std::get<0>(tensorCache) == nullptr) {
            continue;
        }
        if (!std::get<3>(tensorCache)) {
            continue;
        }
        auto curBackend = TensorUtils::getDescribeOrigin(std::get<0>(tensorCache))->getBackend();
        if (curBackend->type() == MNN_FORWARD_CPU) {
            TensorUtils::getDescribeOrigin(iter.first)->getBackend()->onCopyBuffer(iter.first, std::get<0>(tensorCache));
        } else {
            curBackend->onCopyBuffer(iter.first, std::get<0>(tensorCache));
        }
        std::get<3>(tensorCache) = false;
    }
}
ErrorCode Pipeline::execute() {
    _copyInputs();
    auto& mBackend = mInfo.first.cache.first;
    auto& mBackupBackend = mInfo.first.cache.second;
    mBackend->onExecuteBegin();
    for (auto& info : mInfo.second) {
        if (info.type == Schedule::CONSTANT) {
            continue;
        }
        auto& buffer = info.executeBuffer;
        for (int cmdIndex=0; cmdIndex<buffer.command.size(); ++cmdIndex) {
            auto& cmd = *buffer.command[cmdIndex];
#ifdef MNN_PIPELINE_DEBUG
            if (info.op->name() != nullptr) {
                std::string groupOfInput = "input group: [";
                for (int v=0; v<cmd.workInputs.size(); ++v) {
                    groupOfInput = groupOfInput + " " + std::to_string(TensorUtils::getDescribe(cmd.workInputs[v])->group) + " ";
                }
                groupOfInput += "]";
                std::string deviceOfInput = "input: [";
                for (int v=0; v<cmd.workInputs.size(); ++v) {
                    deviceOfInput = deviceOfInput + " " + std::to_string(cmd.workInputs[v]->deviceId()) + " ";
                }
                deviceOfInput += "]";
                std::string deviceOfOutput = "output: [";
                for (int v=0; v<cmd.workOutputs.size(); ++v) {
                    deviceOfOutput = deviceOfOutput + " " + std::to_string(cmd.workOutputs[v]->deviceId()) + " ";
                }
                deviceOfOutput += "]";
                MNN_PRINT("Group: %d, %s - %d, type=%s, inputs: %s, devices: %s - %s\n", info.group, info.op->name()->c_str(), cmdIndex, EnumNameOpType(cmd.op->type()), groupOfInput.c_str(), deviceOfInput.c_str(), deviceOfOutput.c_str());
            }
#endif
            auto code = cmd.execution->onExecute(cmd.workInputs, cmd.workOutputs);
            if (NO_ERROR != code) {
                mBackend->onExecuteEnd();
                return code;
            }
        }
    }
    mBackend->onExecuteEnd();
    return NO_ERROR;
}

ErrorCode Pipeline::executeCallBack(const TensorCallBackWithInfo& before, const TensorCallBackWithInfo& after) {
    _copyInputs();
    auto& mBackend = mInfo.first.cache.first;
    auto& mBackupBackend = mInfo.first.cache.second;
    mBackend->onExecuteBegin();
    for (auto& info : mInfo.second) {
        if (info.type == Schedule::CONSTANT) {
            continue;
        }
        auto& buffer = info.executeBuffer;
        for (int cmdIndex=0; cmdIndex < buffer.command.size(); ++cmdIndex) {
            auto cmdP = buffer.command[cmdIndex];
            auto& cmd = *cmdP;
            if (nullptr == cmd.info.get()) {
                auto code = cmd.execution->onExecute(cmd.workInputs, cmd.workOutputs);
                if (NO_ERROR != code) {
                    mBackend->onExecuteEnd();
                    return code;
                }
                continue;
            }
            auto run = before(cmd.workInputs, cmd.info.get());
            if (run) {
                auto code = cmd.execution->onExecute(cmd.workInputs, cmd.workOutputs);
                if (NO_ERROR != code) {
                    mBackend->onExecuteEnd();
                    return code;
                }
            }
            auto stop = !(after(cmd.workOutputs, cmd.info.get()));
            if (stop) {
                mBackend->onExecuteEnd();
                return CALL_BACK_STOP;
            }
        }
    }
    mBackend->onExecuteEnd();
    return NO_ERROR;
}

Pipeline::~Pipeline() {
    auto& bn = mInfo.first.cache.first;
    auto& backupbn = mInfo.first.cache.second;
    bn->onClearBuffer();
    backupbn->onClearBuffer();
    mInfo.second.clear();
    mCacheConstTensors.clear();
    mWrapTensors.clear();
}

} // namespace MNN
