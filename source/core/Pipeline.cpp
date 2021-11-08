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
static Backend::StorageType _getTensorStorageType(const Tensor* tensor) {
    auto des   = TensorUtils::getDescribe(tensor);
    auto usage = des->usage;
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

static bool _allocTensor(Tensor* t, Backend* curBackend) {
    auto memoryType = _getTensorStorageType(t);
    auto bn         = TensorUtils::getDescribe(t)->backend;
    auto des = TensorUtils::getDescribe(t);
    if (nullptr == bn) {
        TensorUtils::setLinearLayout(t);
        des->backend = curBackend;
        auto res     = curBackend->onAcquireBuffer(t, memoryType);
        return res;
    }
    return true;
}

void Pipeline::UnitInfo::setUp(const Command& command, int index) {
    if (nullptr != command.op->name()) {
        mContent->name = command.op->name()->str();
    } else {
        char buffer[20];
        sprintf(buffer, "%d", index);
        mContent->name = std::string(EnumNameOpType(command.op->type())) + buffer;
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
                   std::shared_ptr<Backend> cpuBackend, std::shared_ptr<Backend> constBackend, bool allocInput, Runtime::CompilerType compilerType)
#ifndef MNN_BUILD_MINI
    : mContext(cpuBackend, true, backend->type()), mUseGeometry(compilerType) {
#else
{
#endif
    MNN_ASSERT(nullptr != backend);
    MNN_ASSERT(nullptr != cpuBackend);
    mBackupBackend = cpuBackend;
    mBackend       = backend;
    mConstBackend  = constBackend;
    mAllocInput    = allocInput;
    mInfo          = std::move(infos);
#ifndef MNN_BUILD_MINI
    GeometryComputerUtils::buildConstantTensors(mInfo, mBackupBackend, !mAllocInput, mMidConstTensors);
#endif
}
void Pipeline::cloneExecution(const std::map<const Op*, std::shared_ptr<Execution>>& cache) {
    Execution* dst;
    for (auto& iter : cache) {
        dst = nullptr;
        auto backend = mBackend.get();
        if (iter.second->backend()->type() != mBackend->type() &&
            iter.second->backend()->type() == mBackupBackend->type()) {
            backend = mBackupBackend.get();
        }
        bool res = iter.second->onClone(backend, iter.first, &dst);
        if (!res) {
            continue;
        }
        MNN_ASSERT(nullptr != dst);
        mOriginExecution.insert(std::make_pair(iter.first, std::shared_ptr<Execution>(dst)));
    }
}

ErrorCode Pipeline::encode(bool isStatic, bool supportDebug) {
    // Static Model just copy info to command buffer
    if (isStatic) {
        for (auto& info : mInfo) {
            Command cmd;
            cmd.outputs = info.outputs;
            cmd.inputs  = info.inputs;
            cmd.op      = info.op;
            mBuffer.command.push_back(cmd);
            // mBuffer.command.emplace_back(GeometryComputerUtils::makeCommand(info.op->UnPack(), info.inputs,
            // info.outputs));
        }
    } else {
#ifndef MNN_BUILD_MINI
        mContext.clear();
        mBuffer.command.clear();
        mBuffer.extras.clear();
        /** Size Compute and compute Const Begin */
        if (mInit) {
            for (auto t : mMidConstTensors) {
                if (t->elementSize() > 0) {
                    mBackupBackend->onReleaseBuffer(t, Backend::STATIC);
                }
                TensorUtils::getDescribe(t)->backend = nullptr;
            }
        }
        mInit = true;
        auto res = GeometryComputerUtils::shapeComputeAndGeometryTransform(mInfo, mBuffer, mContext, mBackupBackend, mUseGeometry);
        if (res != NO_ERROR) {
            return res;
        }
#endif
    }
    bool isQuantModel = false;
    // Set Op
    for (auto& iter : mBuffer.command) {
        if (!iter.buffer.empty()) {
            iter.op = flatbuffers::GetRoot<Op>((void*)iter.buffer.data());
        }
        for (auto t : iter.outputs) {
            if (TensorUtils::getDescribe(t)->quantAttr.get() != nullptr) {
                isQuantModel = true;
            }
        }
    }
    // Propagate Scale
    if (isQuantModel) {
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
        for (const auto& cmd : mBuffer.command) {
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
    mExecutions.resize(mBuffer.command.size());
    for (int i = 0; i < mBuffer.command.size(); ++i) {
        mExecutions[i] = nullptr;
    }
    /** Prepare DebugInfo*/
    if (supportDebug) {
        mFlops = 0.0f;
        mDebugInfos.clear();
        mDebugInfos.resize(mBuffer.command.size());
        for (int i = 0; i < mBuffer.command.size(); ++i) {
            mDebugInfos[i].setUp(mBuffer.command[i], i);
            mFlops += mDebugInfos[i].flops();
        }
    }
    return NO_ERROR;
}

ErrorCode Pipeline::allocMemory() {
    // Compute RefCount
    for (auto& iter : mBuffer.command) {
        for (auto t : iter.inputs) {
            auto des = TensorUtils::getDescribe(t);
            if (des->memoryType == Tensor::InsideDescribe::MEMORY_VIRTUAL) {
                for (auto& r : des->regions) {
                    TensorUtils::getDescribe(r.origin)->useCount = 0;
                }
            } else {
                des->useCount = 0;
            }
        }
    }
    for (auto& iter : mBuffer.command) {
        for (auto t : iter.inputs) {
            auto des = TensorUtils::getDescribe(t);
            if (des->memoryType == Tensor::InsideDescribe::MEMORY_VIRTUAL) {
                for (auto& r : des->regions) {
                    TensorUtils::getDescribe(r.origin)->useCount += 1;
                }
            } else {
                des->useCount += 1;
            }
        }
    }
    mBackend->onClearBuffer();
    mBackupBackend->onClearBuffer();
    for (auto& c : mBuffer.command) {
        for (auto& t : c.outputs) {
            if (TensorUtils::getDescribe(t)->usage != Tensor::InsideDescribe::CONSTANT && TensorUtils::getDescribe(t)->usage != Tensor::InsideDescribe::TRAINABLE) {
                // Don't realloc for Const / Trainable tensor
                TensorUtils::getDescribe(t)->backend = nullptr;
            }
        }
    }
    // Create Execution and Alloc
    mBackend->onResizeBegin();
    for (int i = 0; i < mBuffer.command.size(); ++i) {
        auto& iter = mBuffer.command[i];
        // MNN_PRINT("%d - %s\n", i, EnumNameOpType(iter.op->type()));
        // MNN_PRINT("%s\n", iter.name.c_str());
        if (nullptr == mExecutions[i]) {
            bool cached    = false;
            /** Cache origin execution for fast resize*/
            auto exeIter = mOriginExecution.find(iter.op);
            if (exeIter != mOriginExecution.end()) {
                mExecutions[i] = exeIter->second;
                cached         = true;
            }
            // Create exe
            if (nullptr == mExecutions[i]) {
                mExecutions[i].reset(mBackend->onCreate(iter.inputs, iter.outputs, iter.op));
                if (nullptr == mExecutions[i]) {
                    mExecutions[i].reset(mBackupBackend->onCreate(iter.inputs, iter.outputs, iter.op));
                    if (nullptr == mExecutions[i]) {
                        MNN_ERROR("Create exection error : %d\n", iter.op->type());
                        return NOT_SUPPORT;
                    }
                }
            }
            // invalid means memory alloc failed
            if (!mExecutions[i]->valid()) {
                mExecutions[i] = nullptr;
                return OUT_OF_MEMORY;
            }
            // FIXME: The cached execution may cause wrap error. Fix it in future
            if ((!cached) && iter.buffer.empty() && (iter.op->type() != OpType_Raster) && (iter.op->type() != OpType_BinaryOp)) {
                mOriginExecution.insert(std::make_pair(iter.op, mExecutions[i]));
            }
        }
        auto curBackend = mExecutions[i]->backend();
        // Alloc for Tensors
        bool wrap          = false;
        auto allocFunction = [&](const std::vector<Tensor*>& tensors) {
            for (auto t : tensors) {
                auto des = TensorUtils::getDescribe(t);
                if (des->memoryType == Tensor::InsideDescribe::MEMORY_VIRTUAL) {
                    // Raster's inputs
                    for (auto& r : des->regions) {
                        auto allocRes = _allocTensor(r.origin, curBackend);
                        if (!allocRes) {
                            return OUT_OF_MEMORY;
                        }
                    }
                } else {
                    auto allocRes = _allocTensor(t, curBackend);
                    if (!allocRes) {
                        return OUT_OF_MEMORY;
                    }
                }
            }
            return NO_ERROR;
        };
        if (mAllocInput) {
            auto code = allocFunction(iter.inputs);
            if (NO_ERROR != code) {
                return code;
            }
        }
        for (auto t : iter.inputs) {
            auto des = TensorUtils::getDescribe(t);
            if (des->memoryType == Tensor::InsideDescribe::MEMORY_VIRTUAL) {
                // Raster's inputs
                for (auto& r : des->regions) {
                    MNNForwardType type = MNN_FORWARD_CPU;
                    auto origin     = r.origin;
                    wrap = wrap || (WrapExecution::needWrap(origin, curBackend));
                }
            } else {
                wrap = wrap || (WrapExecution::needWrap(t, curBackend));
            }
        }
        {
            auto code = allocFunction(iter.outputs);
            if (NO_ERROR != code) {
                return code;
            }
        }
        // Wrap If needed
        if (wrap) {
            mExecutions[i].reset(new WrapExecution(mBackupBackend.get(), mExecutions[i]));
        }
        auto code = mExecutions[i]->onResize(iter.inputs, iter.outputs);
        if (NO_ERROR != code && (!mDebugInfos.empty())) {
            MNN_ERROR("Resize error for type = %s, name = %s \n", mDebugInfos[i].type().c_str(), mDebugInfos[i].name().c_str());
            return code;
        }
        // Free mid tensor
        for (auto t : iter.inputs) {
            auto des = TensorUtils::getDescribe(t);
            if (des->memoryType == Tensor::InsideDescribe::MEMORY_VIRTUAL) {
                // Raster's inputs
                for (auto& r : des->regions) {
                    _releaseTensor(r.origin, mAllocInput);
                }
            } else {
                _releaseTensor(t, mAllocInput);
            }
        }
    }
    mBackend->onResizeEnd();
    return NO_ERROR;
}

ErrorCode Pipeline::execute() {
    mBackend->onExecuteBegin();
    for (int i = 0; i < mBuffer.command.size(); ++i) {
        auto& cmd = mBuffer.command[i];
        auto code = mExecutions[i]->onExecute(cmd.inputs, cmd.outputs);
        if (NO_ERROR != code) {
            mBackend->onExecuteEnd();
            return code;
        }
    }
    mBackend->onExecuteEnd();
    return NO_ERROR;
}

ErrorCode Pipeline::executeCallBack(const TensorCallBackWithInfo& before, const TensorCallBackWithInfo& after) {
    if (mDebugInfos.empty()) {
        // don't support debug
        return execute();
    }
    mBackend->onExecuteBegin();
    for (int i = 0; i < mBuffer.command.size(); ++i) {
        auto& cmd  = mBuffer.command[i];
        auto& info = mDebugInfos[i];
        auto run   = before(cmd.inputs, &info);
        if (run) {
            auto code = mExecutions[i]->onExecute(cmd.inputs, cmd.outputs);
            if (NO_ERROR != code) {
                mBackend->onExecuteEnd();
                return code;
            }
        }
        auto stop = !(after(cmd.outputs, &info));
        if (stop) {
            mBackend->onExecuteEnd();
            return CALL_BACK_STOP;
        }
    }
    mBackend->onExecuteEnd();
    return NO_ERROR;
}

Pipeline::~Pipeline() {
    mExecutions.clear();
    if (mInit) {
        for (auto t : mMidConstTensors) {
            if (t->elementSize() > 0) {
                mBackupBackend->onReleaseBuffer(t, Backend::STATIC);
            }
            TensorUtils::getDescribe(t)->backend = nullptr;
        }
    }
    mOriginExecution.clear();
}

} // namespace MNN
