//
//  Pipeline.cpp
//  MNN
//
//  Created by MNN on 2019/01/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "core/Pipeline.hpp"
#include <string.h>
#include "core/Backend.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "core/WrapExecution.hpp"
#include "geometry/GeometryComputerUtils.hpp"
#include "shape/SizeComputer.hpp"

// TODO: Find better way for debug
//#define MNN_OP_SEPERATE

namespace MNN {

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
static Backend::StorageType _getTensorStorageType(const Tensor* tensor, bool allocInput, bool outputStatic) {
    auto des   = TensorUtils::getDescribe(tensor);
    auto usage = des->usage;
    if (TensorUsage::OUTPUT == usage && outputStatic) {
        return Backend::STATIC;
    }
    if (TensorUsage::CONSTANT == usage || TensorUsage::INPUT == usage || TensorUsage::TRAINABLE == usage) {
        return Backend::DYNAMIC_SEPERATE;
    }
    if (tensor->buffer().type.code == halide_type_handle) {
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
    TensorUtils::getDescribe(origin)->useCount -= 1;
    if (0 == TensorUtils::getDescribe(origin)->useCount &&
        TensorUtils::getDescribe(origin)->memoryType == Tensor::InsideDescribe::MEMORY_BACKEND) {
        auto needRelease = _needRelease(origin, !mAllocInput);
        auto bn          = TensorUtils::getDescribe(origin)->backend;
        if (nullptr != bn && needRelease) {
            // For zeroshape may not has bn
            bn->onReleaseBuffer(origin, Backend::DYNAMIC);
        }
    }
}

static bool _allocTensor(Tensor* t, Backend* curBackend, bool allocInput, bool outputStatic) {
    auto memoryType = _getTensorStorageType(t, allocInput, outputStatic);
    auto bn         = TensorUtils::getDescribe(t)->backend;
    auto des = TensorUtils::getDescribe(t);
    MNN_ASSERT(des->memoryType != Tensor::InsideDescribe::MEMORY_VIRTUAL);
    if (nullptr == des->mem.get()) {
        TensorUtils::setLinearLayout(t);
        des->backend = curBackend;
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

Pipeline::Pipeline(std::vector<Schedule::PipelineInfo>&& infos, std::shared_ptr<Backend> backend,
                   std::shared_ptr<Backend> cpuBackend, std::shared_ptr<Backend> constBackend, bool allocInput, bool outputStatic, Runtime::CompilerType compilerType, CacheExecutionMap& cacheMap) : mOriginExecution(cacheMap)
#ifndef MNN_BUILD_MINI
    , mContext(cpuBackend, true, backend->type()), mUseGeometry(compilerType) {
#else
{
#endif
    MNN_ASSERT(nullptr != backend);
    MNN_ASSERT(nullptr != cpuBackend);
    mBackupBackend = cpuBackend;
    mBackend       = backend;
    mConstBackend  = constBackend;
    mAllocInput    = allocInput;
    mOutputStatic  = outputStatic;
    mInfo          = std::move(infos);
    mIsQuantModel = false;
    for (auto& iter : mInfo) {
        for (auto t : iter.outputs) {
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
ErrorCode Pipeline::encode(bool isStatic, bool supportDebug) {
    // Static Model just copy info to command buffer
    if (isStatic) {
        for (int i=0; i<mInfo.size(); ++i) {
            auto& info = mInfo[i];
            SharedPtr<Command> cmd = new Command;
            cmd->outputs = info.outputs;
            cmd->inputs  = info.inputs;
            cmd->op      = info.op;
            info.executeBuffer.command = {cmd};
        }
    } else {
#ifndef MNN_BUILD_MINI
        mContext.clear();
        /** Size Compute and compute Const Begin */
        auto res = GeometryComputerUtils::shapeComputeAndGeometryTransform(mInfo, mContext, mBackupBackend, mUseGeometry);
        if (res != NO_ERROR) {
            return res;
        }
#endif
    }
    // Propagate Scale
    if (mIsQuantModel) {
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
        std::set<OpType> propagateOpTypes = { OpType_Raster, OpType_ReLU, OpType_ReLU6,
                                              OpType_Interp, OpType_CropAndResize, OpType_ROIPooling, OpType_Gather,
                                              OpType_GatherV2, OpType_GatherV2, OpType_ScatterNd };
        for (auto& info : mInfo) {
            auto& buffer = info.executeBuffer;
            for (const auto& cmdP : buffer.command) {
                auto& cmd = *cmdP;
                const auto type = cmd.op->type();
                const auto output = cmd.outputs[0];
                if (propagateOpTypes.find(type) != propagateOpTypes.end()) {
                    if (type == OpType_Raster) {
                        const auto des = MNN::TensorUtils::getDescribe(cmd.inputs[0]);
                        for (auto& r : des->regions) {
                            insertPropagateMap(forwardMap, r.origin, output);
                            insertPropagateMap(backwardMap, output, r.origin);
                        }
                    } else {
                        for (auto t : cmd.inputs) {
                            insertPropagateMap(forwardMap, t, output);
                            insertPropagateMap(backwardMap, output, t);
                        }
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
    }
    /** Prepare DebugInfo*/
    if (supportDebug) {
        mFlops = 0.0f;
        int totalIndex = 0;
        for (auto& info : mInfo) {
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
        
ErrorCode Pipeline::allocMemory(bool firstMalloc) {
    if (!firstMalloc) {
        // For session setNeedMalloc, if session's output is set as some input, It may cause error
        // Dup des to avoid it
        for (auto& info : mInfo) {
            auto& buffer = info.executeBuffer;
            for (const auto& infoP : buffer.command) {
                auto& info = *infoP;
                for (auto t : info.outputs) {
                    if (!TensorUtils::getDescribe(t)->isMutable) {
                        continue;
                    }
                    auto usage = TensorUtils::getDescribe(t)->usage;
                    if (TensorUtils::getDescribeOrigin(t)->mContent->count() > 1) {
                        auto des = TensorUtils::getDescribe(t);
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
    // Compute RefCount
    for (auto& info : mInfo) {
        auto& buffer = info.executeBuffer;
        for (auto& iterP : buffer.command) {
            auto& iter = *iterP;
            bool isRaster = iter.op->type() == OpType_Raster;
            for (auto t : iter.inputs) {
                auto des = TensorUtils::getDescribe(t);
                if (isRaster) {
                    for (auto& r : des->regions) {
                        TensorUtils::getDescribe(r.origin)->useCount = 0;
                    }
                } else {
                    des->useCount = 0;
                }
            }
        }
    }
    for (auto& info : mInfo) {
        auto& buffer = info.executeBuffer;
        for (auto& iterP : buffer.command) {
            auto& iter = *iterP;
            bool isRaster = iter.op->type() == OpType_Raster;
            for (auto t : iter.inputs) {
                auto des = TensorUtils::getDescribe(t);
                if (isRaster) {
                    for (auto& r : des->regions) {
                        TensorUtils::getDescribe(r.origin)->useCount += 1;
                    }
                } else {
                    des->useCount += 1;
                }
            }
        }
    }
    mBackend->onClearBuffer();
    mBackupBackend->onClearBuffer();
    // Create Execution and Alloc
    mBackend->onResizeBegin();
    for (auto& info : mInfo) {
        auto& buffer = info.executeBuffer;
        for (auto& iterP : buffer.command) {
            auto& iter = *iterP;
            // MNN_PRINT("%d - %s\n", i, EnumNameOpType(iter.op->type()));
            // MNN_PRINT("%s\n", iter.name.c_str());
            if (nullptr == iter.executionOrigin) {
                bool cached    = false;
                /** Cache origin execution for fast resize*/
                auto exeIter = mOriginExecution.find(iter.op);
                if (exeIter != mOriginExecution.end()) {
                    iter.executionOrigin = exeIter->second.first;
                    cached         = true;
                    for (auto t : iter.outputs) {
                        TensorUtils::getDescribe(t)->type = exeIter->second.second;
                    }
                }
                // Create exe
                if (nullptr == iter.executionOrigin) {
                    iter.executionOrigin.reset(mBackend->onCreate(iter.inputs, iter.outputs, iter.op));
                    if (nullptr == iter.executionOrigin) {
                        iter.executionOrigin.reset(mBackupBackend->onCreate(iter.inputs, iter.outputs, iter.op));
                        if (nullptr == iter.executionOrigin) {
                            MNN_ERROR("Create exection error : %d\n", iter.op->type());
                            return NOT_SUPPORT;
                        }
                    }
                }
                // invalid means memory alloc failed
                if (!iter.executionOrigin->valid()) {
                    iter.executionOrigin = nullptr;
                    iter.execution = nullptr;
                    return OUT_OF_MEMORY;
                }
                // FIXME: The cached execution may cause wrap error. Fix it in future
                if ((!cached) && iter.buffer == nullptr && (iter.op->type() != OpType_Raster) && (iter.op->type() != OpType_BinaryOp)) {
                    if (iter.outputs.size() > 0) {
                        auto type = TensorUtils::getDescribe(iter.outputs[0])->type;
                        mOriginExecution.insert(std::make_pair(iter.op, std::make_pair(iter.executionOrigin, type)));
                    }
                }
            }
            auto curBackend = iter.executionOrigin->backend();
            // Alloc for Tensors
            bool wrap          = false;
            if (mAllocInput) {
                for (auto t : iter.inputs) {
                    auto des = TensorUtils::getDescribe(t);
                    if (iter.op->type() == OpType_Raster) {
                        // Raster's inputs
                        for (auto& r : des->regions) {
                            auto allocRes = _allocTensor(r.origin, curBackend, mAllocInput, mOutputStatic);
                            if (!allocRes) {
                                return OUT_OF_MEMORY;
                            }
                        }
                    } else {
                        auto allocRes = _allocTensor(t, curBackend, mAllocInput, mOutputStatic);
                        if (!allocRes) {
                            return OUT_OF_MEMORY;
                        }
                    }
                }
            }
            // Check If need wrap
            bool isRaster = iter.op->type() == OpType_Raster;
            for (int v=0; v<iter.inputs.size(); ++v) {
                auto t = iter.inputs[v];
                auto des = TensorUtils::getDescribe(t);
                if (isRaster) {
                    // Raster's inputs
                    for (auto& r : des->regions) {
                        MNNForwardType type = MNN_FORWARD_CPU;
                        auto origin     = r.origin;
                        if (WrapExecution::needWrap(origin, curBackend)) {
                            auto newTensor = WrapExecution::copyConstCache(origin, curBackend, mCacheConstTensors);
                            if (nullptr != newTensor) {
                                r.origin = newTensor;
                            } else {
                                wrap = true;
                            }
                        }
                    }
                } else {
                    if (WrapExecution::needWrap(t, curBackend)) {
                        auto newTensor = WrapExecution::copyConstCache(t, curBackend, mCacheConstTensors);
                        if (nullptr != newTensor) {
                            iter.inputs[v] = newTensor;
                        } else {
                            wrap = true;
                        }
                    }
                }
            }
            {
                for (auto t : iter.outputs) {
                    auto res = _allocTensor(t, curBackend, mAllocInput, mOutputStatic);
                    if (!res) {
                        return OUT_OF_MEMORY;
                    }
                }
            }
            // Wrap If needed
            if (wrap) {
                iter.execution.reset(new WrapExecution(mBackupBackend.get(), iter.executionOrigin));
            } else {
                iter.execution = iter.executionOrigin;
            }
            auto code = iter.execution->onResize(iter.inputs, iter.outputs);
            if (NO_ERROR != code && (!iter.info.get())) {
                MNN_ERROR("Resize error for type = %s, name = %s \n", iter.info->type().c_str(), iter.info->name().c_str());
                return code;
            }
            // Free mid tensor
            for (auto t : iter.inputs) {
                auto des = TensorUtils::getDescribe(t);
                if (iter.op->type() == OpType_Raster) {
                    // Raster's inputs
                    for (auto& r : des->regions) {
                        _releaseTensor(r.origin, mAllocInput);
                    }
                } else {
                    _releaseTensor(t, mAllocInput);
                }
            }
        }
    }
    // Recycle All Dynamic Tensor
    for (auto& info : mInfo) {
        auto& buffer = info.executeBuffer;
        for (auto& c : buffer.command) {
            for (auto& t : c->outputs) {
                auto memoryType = _getTensorStorageType(t, mAllocInput, mOutputStatic);
                if (Backend::DYNAMIC == memoryType) {
                    TensorUtils::getDescribe(t)->mem.reset(nullptr);
                }
            }
        }
    }
    mBackend->onResizeEnd();
    return NO_ERROR;
}

ErrorCode Pipeline::execute() {
    mBackend->onExecuteBegin();
    for (auto& info : mInfo) {
        auto& buffer = info.executeBuffer;
        for (auto& cmdP : buffer.command) {
            auto& cmd = *cmdP;
            auto code = cmd.execution->onExecute(cmd.inputs, cmd.outputs);
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
    mBackend->onExecuteBegin();
    for (auto& info : mInfo) {
        auto& buffer = info.executeBuffer;
        for (auto& cmdP : buffer.command) {
            auto& cmd = *cmdP;
            if (nullptr == cmd.info.get()) {
                auto code = cmd.execution->onExecute(cmd.inputs, cmd.outputs);
                if (NO_ERROR != code) {
                    mBackend->onExecuteEnd();
                    return code;
                }
                continue;
            }
            auto run   = before(cmd.inputs, cmd.info.get());
            if (run) {
                auto code = cmd.execution->onExecute(cmd.inputs, cmd.outputs);
                if (NO_ERROR != code) {
                    mBackend->onExecuteEnd();
                    return code;
                }
            }
            auto stop = !(after(cmd.outputs, cmd.info.get()));
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
    mInfo.clear();
    mCacheConstTensors.clear();
}

} // namespace MNN
