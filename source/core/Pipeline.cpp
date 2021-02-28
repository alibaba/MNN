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
//#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
//#define MNN_DEBUG_TENSOR_SIZE
//#define MNN_DEBUG_PREPARE

#define MNN_FAST_RESIZE
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
    mContent->type = EnumNameOpType(command.op->type());
#ifndef MNN_BUILD_MINI
    mContent->flops = SizeComputer::computeFlops(command.op, command.inputs, command.outputs);
#endif
}

Pipeline::Pipeline(std::vector<Schedule::PipelineInfo>&& infos, std::shared_ptr<Backend> backend,
                   std::shared_ptr<Backend> cpuBackend, bool allocInput, bool geometry)
#ifndef MNN_BUILD_MINI
    : mContext(cpuBackend, true), mUseGeometry(geometry) {
#else
{
#endif
    MNN_ASSERT(nullptr != backend);
    MNN_ASSERT(nullptr != cpuBackend);
    mBackupBackend = cpuBackend;
    mBackend       = backend;
    mAllocInput    = allocInput;
    mInfo          = std::move(infos);
    GeometryComputerUtils::buildConstantTensors(mInfo, mBackupBackend, !mAllocInput, mConstTensors, mMidConstTensors);
}
void Pipeline::cloneExecution(const std::map<const Op*, std::shared_ptr<Execution>>& cache) {
    Execution* dst;
    for (auto& iter : cache) {
        dst = nullptr;
        bool res = iter.second->onClone(mBackend.get(), iter.first, &dst);
        if (!res) {
            continue;
        }
        MNN_ASSERT(nullptr != dst);
        mOriginExecution.insert(std::make_pair(iter.first, std::shared_ptr<Execution>(dst)));
    }
}

ErrorCode Pipeline::encode(bool isStatic) {
    // Static Model just copy info to command buffer
    if (isStatic) {
        for (auto& info : mInfo) {
            flatbuffers::FlatBufferBuilder builder;
            auto lastOffset = Op::Pack(builder, info.op->UnPack());
            builder.Finish(lastOffset);
            Command cmd;
            cmd.buffer.resize(builder.GetSize());
            ::memcpy(cmd.buffer.data(), builder.GetBufferPointer(), cmd.buffer.size());
            cmd.outputs = info.outputs;
            cmd.inputs  = info.inputs;
            cmd.op      = flatbuffers::GetMutableRoot<Op>(cmd.buffer.data());
            mBuffer.command.push_back(cmd);
            // mBuffer.command.emplace_back(GeometryComputerUtils::makeCommand(info.op->UnPack(), info.inputs,
            // info.outputs));
        }
        return NO_ERROR;
    } else {
#ifndef MNN_BUILD_MINI
        mContext.clear();
        mBuffer.command.clear();
        mBuffer.extras.clear();
        /** Size Compute and compute Const Begin */
        for (auto t : mConstTensors) {
            TensorUtils::getDescribe(t)->backend = mBackupBackend.get();
            TensorUtils::getDescribe(t)->usage   = Tensor::InsideDescribe::CONSTANT;
        }
        if (mInit) {
            for (auto t : mMidConstTensors) {
                if (t->elementSize() > 0) {
                    mBackupBackend->onReleaseBuffer(t, Backend::STATIC);
                }
                TensorUtils::getDescribe(t)->backend = nullptr;
            }
        }
        mInit = true;
        return GeometryComputerUtils::shapeComputeAndGeometryTransform(mInfo, mBuffer, mContext, mBackupBackend, mUseGeometry);
#endif
    }
    return NO_ERROR;
}

ErrorCode Pipeline::allocMemory(bool supportDebug) {
    mExecutions.clear();
    mDebugInfos.clear();
    mBackend->onClearBuffer();
    mBackupBackend->onClearBuffer();

    /** Prepare Execution And Alloc*/
    // Compute refCount
    for (auto& iter : mBuffer.command) {
        if (!iter.buffer.empty()) {
            iter.op = flatbuffers::GetMutableRoot<Op>((void*)iter.buffer.data());
        }
        for (auto t : iter.inputs) {
            auto des = TensorUtils::getDescribe(t);
            if (des->memoryType == Tensor::InsideDescribe::MEMORY_VIRTUAL) {
                for (auto& r : des->regions) {
                    TensorUtils::getDescribe(r.origin)->useCount += 1;
                    if (nullptr != r.offset) {
                        TensorUtils::getDescribe(r.offset)->useCount += 1;
                    }
                }
            } else {
                des->useCount += 1;
            }
        }
    }
    // Create Execution and Alloc
    mBackend->onResizeBegin();
    mExecutions.resize(mBuffer.command.size());
    for (int i = 0; i < mBuffer.command.size(); ++i) {
        auto& iter = mBuffer.command[i];
        // MNN_PRINT("%d - %s\n", i, EnumNameOpType(iter.op->type()));
        mExecutions[i] = nullptr;
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
            return OUT_OF_MEMORY;
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
                        if (nullptr != r.offset) {
                            allocRes = _allocTensor(r.origin, curBackend);
                            if (!allocRes) {
                                return OUT_OF_MEMORY;
                            }
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
                    auto bn         = TensorUtils::getDescribe(origin)->backend;
                    if (nullptr != bn) {
                        type = bn->type();
                    }
                    if (type != curBackend->type()) {
                        wrap = true;
                    }
                }
            } else {
                auto bn         = TensorUtils::getDescribe(t)->backend;
                MNNForwardType type = MNN_FORWARD_CPU;
                if (nullptr != bn) {
                    type = bn->type();
                }
                if (type != curBackend->type()) {
                    wrap = true;
                }
            }
        }

        {
            auto code = allocFunction(iter.outputs);
            if (NO_ERROR != code) {
                return code;
            }
        }

        // Wrap If needed
        if (wrap && (!cached)) {
            mExecutions[i].reset(new WrapExecution(mBackupBackend.get(), mExecutions[i]));
        }
        if ((!cached) && iter.buffer.empty() && (iter.op->type() != OpType_Raster)) {
            mOriginExecution.insert(std::make_pair(iter.op, mExecutions[i]));
        }
        auto code = mExecutions[i]->onResize(iter.inputs, iter.outputs);
        if (NO_ERROR != code) {
            return code;
        }
        // Free mid tensor
        for (auto t : iter.inputs) {
            auto des = TensorUtils::getDescribe(t);
            if (des->memoryType == Tensor::InsideDescribe::MEMORY_VIRTUAL) {
                // Raster's inputs
                for (auto& r : des->regions) {
                    _releaseTensor(r.origin, mAllocInput);
                    if (nullptr != r.offset) {
                        _releaseTensor(r.offset, mAllocInput);
                    }
                }
            } else {
                _releaseTensor(t, mAllocInput);
            }
        }
    }
    mBackend->onResizeEnd();

    /** Prepare DebugInfo*/
    if (supportDebug) {
        mDebugInfos.resize(mBuffer.command.size());
        for (int i = 0; i < mBuffer.command.size(); ++i) {
            mDebugInfos[i].setUp(mBuffer.command[i], i);
        }
    }
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
    for (auto t : mConstTensors) {
        mBackupBackend->onReleaseBuffer(t, Backend::STATIC);
    }
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
