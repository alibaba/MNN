//
//  Pipeline.cpp
//  MNN
//
//  Created by MNN on 2019/01/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "core/Pipeline.hpp"
#include "core/Backend.hpp"
#include "core/Macro.h"
#include "core/SizeComputer.hpp"
#include "core/TensorUtils.hpp"
#include "core/WrapExecution.hpp"
//#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
//#define MNN_DEBUG_TENSOR_SIZE
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
    auto des = TensorUtils::getDescribe(tensor);
    auto usage = des->usage;
    if (TensorUsage::CONST == usage || TensorUsage::INPUT == usage || TensorUsage::TRAINABLE == usage) {
        return Backend::DYNAMIC_SEPERATE;
    }
    if (des->handleType != Tensor::HANDLE_NONE) {
        return Backend::DYNAMIC_SEPERATE;
    }
    return Backend::DYNAMIC;
}

static Backend::StorageType _getTensorReleaseStorageType(const Tensor* tensor) {
    auto des = TensorUtils::getDescribe(tensor);
    auto usage = des->usage;
    if (des->handleType != Tensor::HANDLE_NONE) {
        return Backend::DYNAMIC_SEPERATE;
    }
    if (TensorUsage::CONST == usage || TensorUsage::TRAINABLE == usage) {
        return Backend::DYNAMIC_SEPERATE;
    }
    return Backend::DYNAMIC;
}

bool Pipeline::Unit::_allocTensors(Backend* bn, const std::vector<Tensor*>& tensors) {
    for (auto t : tensors) {
        auto des = TensorUtils::getDescribe(t);
        if (nullptr != des->backend) {
            continue;
        }
        des->backend = bn;
        TensorUtils::setLinearLayout(t);
        auto success = bn->onAcquireBuffer(t, _getTensorStorageType(t));
        if (!success) {
            return false;
        }
    }
    return true;
}

Pipeline::Unit::Unit(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    MNN_ASSERT(nullptr != op);
    mOriginOp = op;
    mType     = op->type();
    mInputs   = inputs;
    mOutputs  = outputs;
    if (nullptr != op->name()) {
        mContent->name = op->name()->str();
    }
    auto typeStr = EnumNameOpType(mType);
    if (nullptr != typeStr) {
        mContent->type = typeStr;
    }
}



bool Pipeline::Unit::_createExecution(Backend* bn, Backend* cpuBn) {
    mExecution.reset(bn->onCreate(mInputs, mOutputs, mOriginOp));
    if (nullptr == mExecution) {
        mExecution.reset(cpuBn->onCreate(mInputs, mOutputs, mOriginOp));
    }
    if (nullptr == mExecution) {
        return false;
    }
    bool needWrap = false;

    auto executionBackend = mExecution->backend();
    for (int i = 0; i < mInputs.size(); ++i) {
        auto t   = mInputs[i];
        auto des = TensorUtils::getDescribe(t);
        if (des->backend != executionBackend && SizeComputer::opNeedContent(mOriginOp->type(), i)) {
            needWrap = true;
        }
    }
    if (needWrap) {
        // FUNC_PRINT_ALL(mOriginOp->name()->c_str(), s);
        auto tempExecution = mExecution;
        mExecution.reset(new WrapExecution(cpuBn, tempExecution));
    }
    return mExecution->valid();
}

ErrorCode Pipeline::Unit::execute() {
    if (nullptr == mExecution) {
        return NO_EXECUTION;
    }
    if (mConst) {
        return NO_ERROR;
    }
    auto code = mExecution->onExecute(mInputs, mOutputs);
    if (NO_ERROR != code) {
        MNN_ERROR("Execute Error for %s, code=%d\n", mContent->name.c_str(), code);
    }
    return code;
}
ErrorCode Pipeline::Unit::executeCallBack(const TensorCallBackWithInfo& before, const TensorCallBackWithInfo& after) {
    if (nullptr == mExecution) {
        return NO_EXECUTION;
    }
    if (mConst) {
        return NO_ERROR;
    }
    auto run = before(mInputs, this);
    if (run) {
        auto code = mExecution->onExecute(mInputs, mOutputs);
        if (NO_ERROR != code) {
            MNN_ERROR("Execute Error for %s, code=%d\n", mContent->name.c_str(), code);
            return code;
        }
    }
    auto runOthers = after(mOutputs, this);
    if (!runOthers) {
        return CALL_BACK_STOP;
    }
    return NO_ERROR;
}

ErrorCode Pipeline::Unit::prepare(Backend* bn, Backend* cpuBn) {
    for (auto t : mInputs) {
        bool valid = true;
        for (int i = 0; i < t->dimensions(); ++i) {
            if (t->length(i) <= 0) {
                valid = false;
                break;
            }
        }
        if (!valid) {
            MNN_ERROR("The %s's input is not ready\n", mContent->name.c_str());
            return COMPUTE_SIZE_ERROR;
        }
    }
    {
        auto success = _allocTensors(bn, mInputs);
        if (!success) {
            return OUT_OF_MEMORY;
        }
    }
    bool ready = SizeComputer::computeOutputSize(mOriginOp, mInputs, mOutputs);
    for (auto o : mOutputs) {
        if (o->size() <= 0) {
            ready = false;
        }
        if (o->dimensions() < 4 && TensorUtils::getDescribe(o)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4) {
            for (auto index = o->dimensions(); index < 4; ++index) {
                o->setLength(index, 1);
            }
        }
    }
    mContent->flops = SizeComputer::computeFlops(mOriginOp, mInputs, mOutputs);

#ifdef MNN_DEBUG_TENSOR_SIZE
    MNN_PRINT("\n===> compute shape: %s, [%d]\n", mOriginOp->name()->c_str(), mOriginOp->type());
    if (mInputs.size()) {
        MNN_PRINT("Inputs:\n");
        for (auto o : mInputs) {
            if (o->dimensions() == 0) {
                MNN_PRINT("\t*Scalar*");
            }
            for (int i = 0; i < o->dimensions(); ++i) {
                MNN_PRINT("%d, ", o->length(i));
            }
            MNN_PRINT("\n");
        }
    }
    MNN_PRINT("Outputs:\n");
    for (auto o : mOutputs) {
        if (o->dimensions() == 0) {
            MNN_PRINT("\t*Scalar*");
        }
        for (int i = 0; i < o->dimensions(); ++i) {
            MNN_PRINT("%d, ", o->length(i));
        }
        MNN_PRINT("\n");
    }
#endif
    if (!ready) {
        return COMPUTE_SIZE_ERROR;
    }

    // Check const
    mConst = true;
    for (int i = 0; i < mInputs.size(); ++i) {
        if (SizeComputer::opNeedContent(mOriginOp->type(), i) && (TensorUtils::getDescribe(mInputs[i])->usage != TensorUsage::CONST)) {
            mConst = false;
            break;
        }
    }
    if (mType == OpType_TrainableParam) {
        for (auto t : mOutputs) {
            TensorUtils::getDescribe(t)->usage = TensorUsage::TRAINABLE;
        }
        mConst = false;
    }

    if (mConst) {
        for (auto t : mOutputs) {
            TensorUtils::getDescribe(t)->usage = TensorUsage::CONST;
        }
        bn = cpuBn;
    }

    // Create or Resize execution
    if (nullptr == mExecution) {
        auto sucess = _createExecution(bn, cpuBn);
        if (!sucess || mExecution == nullptr) {
            return NOT_SUPPORT;
        }
    }
    bn = mExecution->backend();
    {
        auto success = _allocTensors(bn, mOutputs);
        if (!success) {
            return OUT_OF_MEMORY;
        }
    }
    auto code = mExecution->onResize(mInputs, mOutputs);
    if (TENSOR_NOT_SUPPORT == code || TENSOR_NEED_DIVIDE == code) {
        // TODO
        mExecution.reset();
        for (auto t : mOutputs) {
            auto des = TensorUtils::getDescribe(t);
            des->backend->onReleaseBuffer(t, _getTensorReleaseStorageType(t));
            des->backend = nullptr;
        }
        auto sucess = _createExecution(cpuBn, cpuBn);
        MNN_ASSERT(NO_ERROR == sucess);
        auto success = _allocTensors(mExecution->backend(), mOutputs);
        if (!success) {
            return OUT_OF_MEMORY;
        }
        code = mExecution->onResize(mInputs, mOutputs);
    }
    if (NO_ERROR != code) {
        mExecution.reset();
        return code;
    }
    if (mConst) {
        code = mExecution->onExecute(mInputs, mOutputs);
    }

    for (auto t : mInputs) {
        auto des = TensorUtils::getDescribe(t);
        des->useCount -= 1;
        if (0 == des->useCount) {
            des->backend->onReleaseBuffer(t, _getTensorReleaseStorageType(t));
        }
    }
    return code;
}

Pipeline::Pipeline(const std::vector<Schedule::PipelineInfo>& infos, Backend* backend, Backend* cpuBackend) {
    SizeComputerSuite::init();
    MNN_ASSERT(nullptr != backend);
    MNN_ASSERT(nullptr != cpuBackend);
    mBackupBackend = cpuBackend;
    mBackend       = backend;

    for (auto& info : infos) {
        std::shared_ptr<Unit> unit(new Unit(info.op, info.inputs, info.outputs));
        mUnits.emplace_back(unit);
    }
}

ErrorCode Pipeline::prepare() {
    mBackend->onResizeBegin();
    for (auto& u : mUnits) {
        auto code = u->prepare(mBackend, mBackupBackend);
        if (NO_ERROR != code) {
            if (nullptr != u->mOriginOp->name()) {
                MNN_ERROR("Resize error for %s, code=%d\n", u->mOriginOp->name()->c_str(), code);
            }
            return code;
        }
    }
    mBackend->onResizeEnd();
    return NO_ERROR;
}

ErrorCode Pipeline::execute() {
    mBackend->onExecuteBegin();
    for (int i=0; i<mUnits.size(); ++i) {
        auto& u = mUnits[i];
        auto code = u->execute();
        if (code != NO_ERROR) {
            mBackend->onExecuteEnd();
            return code;
        }
    }
    mBackend->onExecuteEnd();
    return NO_ERROR;
}

ErrorCode Pipeline::executeCallBack(const TensorCallBackWithInfo& before, const TensorCallBackWithInfo& after) {
    mBackend->onExecuteBegin();
    std::shared_ptr<char> __defer(nullptr, [this](void*) { mBackend->onExecuteEnd(); });
    for (auto& u : mUnits) {
        auto code = u->executeCallBack(before, after);
        if (code != NO_ERROR) {
            return code;
        }
    }
    return NO_ERROR;
}

ErrorCode Pipeline::releaseCache() {
    for (auto& u : mUnits) {
        if (nullptr != u->mExecution) {
            auto code = u->mExecution->onReleaseCache();
            if (NO_ERROR != code) {
                MNN_ERROR("Error for release cache for %s\n", u->name().c_str());
                return code;
            }
        }
    }
    return NO_ERROR;
}

} // namespace MNN
