//
//  WrapExecution.cpp
//  MNN
//
//  Created by MNN on 2018/09/03.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <cmath>
#include "core/WrapExecution.hpp"
#include "core/TensorUtils.hpp"
#include "core/OpCommonUtils.hpp"
#include "core/Concurrency.h"
#include "backend/cpu/CPUCast.hpp"
#include "backend/cpu/compute/CommonOptFunction.h"

// #define LOG_VERBOSE

namespace MNN {
bool WrapExecution::needWrap(const Tensor* input, Backend* curBackend) {
    auto curType = curBackend ? curBackend->type() : MNN_FORWARD_CPU;
    if (curType == MNN_FORWARD_NN) {
        return false;
    }
    auto des = TensorUtils::getDescribeOrigin(input);
    auto bn = des->getBackend();
    MNNForwardType type = MNN_FORWARD_CPU;
    int pack = 4;
    int bytes = 4;
    if (nullptr != bn) {
        type = bn->type();
        if (type == MNN_FORWARD_CPU_EXTENSION) {
            auto core = static_cast<CPUBackend*>(bn)->functions();
            pack = core->pack;
            bytes = core->bytes;
        }
    }
    if (type == curType) {
        return false;;
    }
    bool srcCpu = (type == MNN_FORWARD_CPU_EXTENSION || type == MNN_FORWARD_CPU);
    bool dstCpu = ((curType == MNN_FORWARD_CPU_EXTENSION) || (curType == MNN_FORWARD_CPU));
    if (srcCpu && dstCpu) {
        int curBytes = 4, curPack = 4;
        if (curBackend) {
            auto dstCore = static_cast<CPUBackend*>(curBackend)->functions();
            curBytes = dstCore->bytes;
            curPack = dstCore->pack;
        }
        if (curBytes == bytes) {
            if (curPack == pack || des->mContent->dimensionFormat != MNN_DATA_FORMAT_NC4HW4) {
                return false;
            }
        }
    }
    return true;
}

class WrapCopyExecution : public Execution {
public:
    WrapCopyExecution(Backend* bn, Backend* backup) : Execution(bn) {
        mBackupBackend = backup;
    }
    virtual ~ WrapCopyExecution() {
        // Do nothing
    }
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override {
        auto inputBn = TensorUtils::getDescribeOrigin(inputs[0])->getBackend();
        auto outputBn = TensorUtils::getDescribeOrigin(outputs[0])->getBackend();
        auto inputForwardtype = MNN_FORWARD_CPU;
        auto outputForwardtype = MNN_FORWARD_CPU;
        if (nullptr != inputBn) {
            inputForwardtype = inputBn->type();
        }
        if (nullptr != outputBn) {
            outputForwardtype = outputBn->type();
        }
        mMidCPUTensor = nullptr;
        if (inputForwardtype != MNN_FORWARD_CPU && outputForwardtype != MNN_FORWARD_CPU) {
            // Use Mid Buffer
            mMidCPUTensor = WrapExecution::makeCopyTensor(inputs[0], mBackupBackend);
            auto res = mBackupBackend->onAcquireBuffer(mMidCPUTensor.get(), Backend::DYNAMIC);
            if (!res) {
                return OUT_OF_MEMORY;
            }
            mBackupBackend->onReleaseBuffer(mMidCPUTensor.get(), Backend::DYNAMIC);
        }
        return NO_ERROR;
    }
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override {
        auto inputBn = TensorUtils::getDescribeOrigin(inputs[0])->getBackend();
        auto outputBn = TensorUtils::getDescribeOrigin(outputs[0])->getBackend();
        auto outputForwardtype = MNN_FORWARD_CPU;
        if (nullptr != mMidCPUTensor.get()) {
            inputBn->onCopyBuffer(inputs[0], mMidCPUTensor.get());
            outputBn->onCopyBuffer(mMidCPUTensor.get(), outputs[0]);
            return NO_ERROR;
        }
        if (nullptr != outputBn) {
            outputForwardtype = outputBn->type();
        }
        if (outputForwardtype == MNN_FORWARD_CPU) {
            MNN_ASSERT(nullptr != inputBn);
            inputBn->onCopyBuffer(inputs[0], outputs[0]);
        } else {
            outputBn->onCopyBuffer(inputs[0], outputs[0]);
        }
        return NO_ERROR;
    }
private:
    std::shared_ptr<Tensor> mMidCPUTensor;
    Backend* mBackupBackend;
};
std::shared_ptr<Tensor> WrapExecution::makeCopyTensor(Tensor* t, Backend* targetBackend) {
    std::shared_ptr<Tensor> wrapTensor(new Tensor);
    TensorUtils::copyShape(t, wrapTensor.get(), true, true);
    TensorUtils::getDescribeOrigin(wrapTensor.get())->setBackend(targetBackend);
    return wrapTensor;
}

Execution* WrapExecution::makeCopyExecution(Backend* backend, Backend* backupBackend) {
    return new WrapCopyExecution(backend, backupBackend);
}

bool WrapExecution::allocAndCopy(Backend* curBackend, const Tensor* input, Tensor* output) {
    auto tempRes = curBackend->onAcquireBuffer(output, Backend::STATIC);
    if (!tempRes) {
        return false;
    }
    TensorUtils::getDescribeOrigin(output)->setBackend(curBackend);
    if (curBackend->type() == MNN_FORWARD_CPU) {
        input->copyToHostTensor(output);
    } else {
        output->copyFromHostTensor(input);
    }
    return true;
}

Tensor* WrapExecution::copyConstCache(Tensor* t, Backend* curBackend, std::map<Tensor*, std::shared_ptr<Tensor>>& cache, bool forbidReplace) {
    auto des = TensorUtils::getDescribe(t);
    if (curBackend->type() != MNN_FORWARD_CPU) {
        auto constCacheiter = cache.find(t);
        if (constCacheiter != cache.end()) {
            // The tensor has been copy by op before, just use it
            return constCacheiter->second.get();
        } else {
            // search or create const for new backend
            std::shared_ptr<Tensor> wrapTensor = makeCopyTensor(t, curBackend);
            auto outDes = TensorUtils::getDescribe(wrapTensor.get());
            outDes->usage = des->usage;
            auto tempRes = allocAndCopy(curBackend, t, wrapTensor.get());
            if (!tempRes) {
                return nullptr;
            }
            bool canReplace = !des->isMutable;
            if (des->stageMask & Tensor::InsideDescribe::GEOMETRY_STAGE) {
                canReplace = false;
            }
            if (des->stageMask & Tensor::InsideDescribe::CONVERTED_STAGE) {
                canReplace = false;
            }
            if (des->memoryType == Tensor::InsideDescribe::MEMORY_HOST){
                canReplace = false;
            }
            if (forbidReplace) {
                canReplace = false;
            }
            if (canReplace) {
                outDes->stageMask |= Tensor::InsideDescribe::CONVERTED_STAGE;
                copyReplaceTensor(wrapTensor.get(), t);
                return t;
            } else {
                cache.insert(std::make_pair(t, wrapTensor));
            }
            return wrapTensor.get();
        }
    }
    return nullptr;
}
void WrapExecution::copyReplaceTensor(const Tensor* wrapTensor, Tensor* t) {
    TensorUtils::getDescribeOrigin(t)->mContent = TensorUtils::getDescribeOrigin(wrapTensor)->mContent;
    TensorUtils::getDescribeOrigin(t)->mem = TensorUtils::getDescribeOrigin(wrapTensor)->mem;
    TensorUtils::getDescribeOrigin(t)->setBackend( TensorUtils::getDescribeOrigin(wrapTensor)->getBackend());
    t->buffer().host = wrapTensor->buffer().host;
    t->buffer().device = wrapTensor->buffer().device;
    t->buffer().dim = TensorUtils::getDescribe(wrapTensor)->dims;
}


} // namespace MNN
