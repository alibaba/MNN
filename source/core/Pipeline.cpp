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
            return true;
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
    auto usage = des->usage;
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
static void _releaseTensor(Tensor* origin, bool mAllocInput) {
    if (TensorUtils::getDescribe(origin)->usage != Tensor::InsideDescribe::CONSTANT) {
        TensorUtils::getDescribe(origin)->useCount -= 1;
    }
    if (0 == TensorUtils::getDescribe(origin)->useCount &&
        TensorUtils::getDescribe(origin)->memoryType == Tensor::InsideDescribe::MEMORY_BACKEND) {
        auto needRelease = _needRelease(origin, !mAllocInput);
        auto bn          = TensorUtils::getDescribe(origin)->getBackend();
        if (nullptr != bn && needRelease) {
            // For zeroshape may not has bn
            bn->onReleaseBuffer(origin, Backend::DYNAMIC);
        }
    }
}

static bool _allocTensor(Tensor* t, Backend* curBackend, bool outputStatic) {
    auto memoryType = _getTensorStorageType(t, outputStatic);
    auto bn         = TensorUtils::getDescribe(t)->getBackend();
    auto des = TensorUtils::getDescribe(t);
    if (nullptr == des->mem.get()) {
        MNN_ASSERT(des->memoryType != Tensor::InsideDescribe::MEMORY_VIRTUAL);
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

Pipeline::Pipeline(Schedule::PipelineInfo&& info, bool allocInput, bool outputStatic, const TuningAttr& tune, const Runtime* rt, const Runtime* cpuRt)
#ifndef MNN_BUILD_MINI
    : mContext(info.first.cache.second, info.first.cache.first->type(), info.first.info.user ? info.first.info.user->precision :  BackendConfig::Precision_Normal), mUseGeometry(rt->onGetCompilerType()) {
#else
{
#endif
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
            SharedPtr<Command> cmd = new Command;
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
        /** Size Compute and compute Const Begin */
        auto res = GeometryComputerUtils::shapeComputeAndGeometryTransform(mInfo.second, mContext, mInfo.first.cache.second, mUseGeometry, false, permitCodegen);
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
                    SharedPtr<Command> command(new Command);
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
#ifndef MNN_BUILD_MINI
    else {
        for (auto& info : mInfo.second) {
            auto& buffer = info.executeBuffer;
            for (auto& cmdP : buffer.command) {
                mFlops += SizeComputer::computeFlops(cmdP->op, cmdP->inputs, cmdP->outputs);
            }
        }
    }
#endif

    return NO_ERROR;
}

void Pipeline::_pushTuningTask(std::vector<Schedule::OpCacheInfo>&& initInfos) {
    // Dup Tensors for initInfos;
    std::map<Tensor*, std::shared_ptr<Tensor>> holdTensors;
    auto& mBackend = mInfo.first.cache.first;
    auto& mBackupBackend = mInfo.first.cache.second;

    for (auto& info : initInfos) {
        auto& buffer = info.executeBuffer;
        for (int v=0; v<buffer.command.size(); ++v) {
            auto iterP = buffer.command[v];
            auto& iter = *iterP;
            buffer.command[v] = new Command;
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
    auto future = std::async(std::launch::async, [&, this](std::vector<Schedule::OpCacheInfo>&& infos, std::map<Tensor*, std::shared_ptr<Tensor>>&& tensors, std::shared_ptr<Backend> backend, const std::atomic_bool& cancelled) {

        backend->onClearBuffer();
        backend->onResizeBegin();
        for (auto& info : infos) {
            auto& buffer = info.executeBuffer;
            for (auto& iterP : buffer.command) {
                if(cancelled) {
                    return -1;
                }
                auto& iter = *iterP;
                // FIXME: Remove onMaskOpReady in future
                const_cast<Runtime*>(mRuntime)->onMaskOpReady(iter.inputs, iter.outputs, iter.op);

                // If create op failed, we can also mask the op is ready for runtime
                std::shared_ptr<Execution> exe(backend->onCreate(iter.inputs, iter.outputs, iter.op));
                if (nullptr == exe) {
                    continue;
                }
                std::vector<Tensor*> forRelease;
                std::shared_ptr<void> _defer(nullptr, [&forRelease](void*) {
                    for (auto t : forRelease) {
                        TensorUtils::getDescribe(t)->mem.reset(nullptr);
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

static ErrorCode _createExecutions(Schedule::PipelineInfo& mInfo) {
    auto& mBackend = mInfo.first.cache.first;
    auto& mBackupBackend = mInfo.first.cache.second;
    for (auto& info : mInfo.second) {
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
            if (nullptr == iter.execution) {
                iter.execution.reset(mBackend->onCreate(iter.inputs, iter.outputs, iter.op));
            }
            if (nullptr == iter.execution) {
                // Try Backup
                iter.execution.reset(mBackupBackend->onCreate(iter.inputs, iter.outputs, iter.op));
                if (nullptr == iter.execution) {
                    if (mInfo.first.reportError) {
                        MNN_ERROR("Create execution error : %d\n", iter.op->type());
                    }
                    return NOT_SUPPORT;
                }
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
                    auto des = TensorUtils::getDescribe(t);
                    if (nullptr == des->mem.get()) {
                        des->setBackend(nullptr);
                    }
                }
            }
            for (auto t : iter.outputs) {
                auto des = TensorUtils::getDescribe(t);
                if (nullptr == des->mem.get()) {
                    des->setBackend(nullptr);
                }
            }
        }
    }

    // Set Tensor's Backend
    for (auto& info : mInfo.second) {
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
                    auto des = TensorUtils::getDescribe(t);
                    if (nullptr == des->mem.get() && nullptr == des->getBackend()) {
                        des->setBackend(curBackend);
                    }
                }
            }
            for (auto t : iter.outputs) {
                auto des = TensorUtils::getDescribe(t);
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
static ErrorCode _InsertCopy(Schedule::PipelineInfo& mInfo, std::map<Tensor*, std::shared_ptr<Tensor>>& mCacheConstTensors, std::map<Tensor*, std::shared_ptr<Tensor>>& shapeFixConstCache, bool ownInput, bool permitCodegen) {
    std::map<std::pair<Tensor*, Backend*>, std::shared_ptr<Tensor>> wrapCache;
    std::shared_ptr<BufferStorage> copyOp;
    shapeFixConstCache.clear();
    for (auto& info : mInfo.second) {
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
                        } else if (des->usage == Tensor::InsideDescribe::CONSTANT) {
                            newTensor = WrapExecution::copyConstCache(t, curBackend, shapeFixConstCache, permitCodegen);
                        }
                        if (nullptr != newTensor) {
                            iter.workInputs[v] = newTensor;
                            break;
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
                        auto copyWrap = WrapExecution::makeCopyExecution(curBackend, mInfo.first.cache.second.get(), t, wrapCache, true);
                        iter.workInputs[v] = copyWrap.second.get();
                        if (nullptr != copyWrap.first) {
                            _makeCopyOp(copyOp);
                            SharedPtr<Command> cmdP = new Command;
                            auto& cmd = *cmdP;
                            cmd.buffer = copyOp;
                            cmd.workInputs  = {t};
                            cmd.workOutputs = {copyWrap.second.get()};
                            cmd.op      = flatbuffers::GetRoot<Op>(cmd.buffer->buffer());
                            buffer.extras.emplace_back(copyWrap.second);
                            cmd.execution.reset(copyWrap.first);
                            buffer.command.emplace_back(cmdP);
                        }
                    } while(false);
                }
            }
            buffer.command.emplace_back(iterP);
            iter.workOutputs = iter.outputs;
            for (int v=0; v<iter.workOutputs.size(); ++v) {
                auto t = iter.workOutputs[v];
                if (WrapExecution::needWrap(t, curBackend)) {
                    auto copyWrap = WrapExecution::makeCopyExecution(curBackend, mInfo.first.cache.second.get(), t, wrapCache, false);
                    iterP->workOutputs[v] = copyWrap.second.get();
                    _makeCopyOp(copyOp);
                    SharedPtr<Command> cmdP = new Command;
                    auto& cmd = *cmdP;
                    cmd.buffer = copyOp;
                    cmd.workInputs  = {copyWrap.second.get()};
                    cmd.workOutputs = {t};
                    cmd.op      = flatbuffers::GetRoot<Op>(cmd.buffer->buffer());
                    buffer.extras.emplace_back(copyWrap.second);
                    cmd.execution.reset(copyWrap.first);
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
            TensorUtils::getDescribe(t)->mem.reset(nullptr);
        }
    }
    for (auto& t : command->workInputs) {
        auto memoryType = _getTensorStorageType(t, mOutputStatic);
        if (Backend::DYNAMIC == memoryType) {
            TensorUtils::getDescribe(t)->mem.reset(nullptr);
        }
    }
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
                    if (TensorUtils::getDescribeOrigin(t)->mContent->count() > 1) {
                        TensorUtils::getDescribeOrigin(t)->mContent = new Tensor::InsideDescribe::NativeInsideDescribe;
                        auto dstDes = TensorUtils::getDescribe(t);
                        t->buffer().dim = dstDes->dims;
                        ::memcpy(t->buffer().dim, des->dims, MNN_MAX_TENSOR_DIM * sizeof(halide_dimension_t));
                        dstDes->dimensionFormat = des->dimensionFormat;
                        dstDes->usage = usage;
                        dstDes->regions = des->regions;
                        dstDes->quantAttr = des->quantAttr;
                        dstDes->tensorArrayAttr = des->tensorArrayAttr;
                    }
                }
            }
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
        auto code = _createExecutions(mInfo);
        if (NO_ERROR != code) {
            return code;
        }
    }
    /* Create Execution End */

    _SetTensorBackend(mInfo, mAllocInput);
    // Insert Wrap If needed
    {
        auto insertCode = _InsertCopy(mInfo, mCacheConstTensors, mShapeFixConstCache, mAllocInput, forbidReplace);
        if (NO_ERROR != insertCode) {
            return insertCode;
        }
    }
    /* Insert Wrap End*/

    // Compute RefCount Begin
    for (auto& info : mInfo.second) {
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

    // Alloc tensor
    mBackend->onResizeBegin();
    mBackupBackend->onResizeBegin();
    for (auto& info : mInfo.second) {
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

            // MNN_PRINT("before Resize: optype:%s, name:%s, input0:%p, output0:%p, mAllocInput:%d\n", EnumNameOpType(iter.op->type()), iter.info->name().c_str(), iter.inputs[0], iter.outputs[0], mAllocInput);
            // Alloc for Tensors
            auto curBackend = iter.execution->backend();
            if (mAllocInput) {
                for (auto t : iter.workInputs) {
                    auto allocRes = _allocTensor(t, curBackend, mOutputStatic);
                    if (!allocRes) {
                        return OUT_OF_MEMORY;
                    }
                }
            }
            {
                for (auto t : iter.workOutputs) {
                    auto res = _allocTensor(t, curBackend, mOutputStatic);
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
            // Free mid tensor
            for (auto t : iter.workInputs) {
                _releaseTensor(t, mAllocInput);
            }
        }
    }
    // Recycle All Dynamic Tensor
    for (auto& info : mInfo.second) {
        auto& buffer = info.executeBuffer;
        for (auto& c : buffer.command) {
            _recycleDynamicMemory(c.get());
        }
    }
    auto code = mBackend->onResizeEnd();
    if (code != NO_ERROR) {
        return code;
    }
    code = mBackupBackend->onResizeEnd();
    return code;
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
        auto curBackend = TensorUtils::getDescribe(std::get<0>(tensorCache))->getBackend();
        if (curBackend->type() == MNN_FORWARD_CPU) {
            TensorUtils::getDescribe(iter.first)->getBackend()->onCopyBuffer(iter.first, std::get<0>(tensorCache));
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
        auto& buffer = info.executeBuffer;
//#define LOG_VERPOSE
#ifdef LOG_VERPOSE
        FUNC_PRINT_ALL(info.op->name()->c_str(), s);
#endif
        for (auto& cmdP : buffer.command) {
            auto& cmd = *cmdP;
            auto code = cmd.execution->onExecute(cmd.workInputs, cmd.workOutputs);
// #define LOG_VERPOSE
#ifdef LOG_VERPOSE
            auto dumpT = [](Tensor* t) {
                auto size = TensorUtils::getRawSize(t);
                size = size > 10 ? 10 : size;
                if (t->getType() == halide_type_of<float>()) {
                    for (int i=0; i<size; ++i) {
                        MNN_PRINT("%f, ", t->host<float>()[i]);
                    }
                } else {
                    for (int i=0; i<size; ++i) {
                        MNN_PRINT("%d, ", t->host<int>()[i]);
                    }
                }
                MNN_PRINT("\n");
            };
            if (/* cmd.op->name() && cmd.op->name()->str() == "/embed/embed_/Gather_output_0"*/
                cmd.op->type() == OpType_Convolution) {
                MNN_PRINT("%s Input begin:\n", EnumNameOpType(cmd.op->type()));
                for (auto t : cmd.workInputs) {
                    dumpT(t);
                }
                MNN_PRINT("%s Output begin:\n", EnumNameOpType(cmd.op->type()));
                for (auto t : cmd.workOutputs) {
                    dumpT(t);
                }
            }
#endif
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
            auto run   = before(cmd.inputs, cmd.info.get());
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
    mShapeFixConstCache.clear();
}

} // namespace MNN
