//
//  InsideExpr.cpp
//  MNN
//
//  Created by MNN on 2019/06/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "InsideExpr.hpp"
#include "Backend.hpp"
#include "SizeComputer.hpp"
#include "Tensor.hpp"
#include "TensorUtils.hpp"
#include "Utils.hpp"
#include "BasicOptimizer_generated.h"

namespace MNN {
namespace Express {
struct Command {
    std::vector<Tensor*> mInputs;
    std::vector<Tensor*> mOutputs;
    std::shared_ptr<Execution> mExecution;
    ErrorCode resize() {
        return mExecution->onResize(mInputs, mOutputs);
    }
    ErrorCode run() {
        return mExecution->onExecute(mInputs, mOutputs);
    }
};
static Tensor::DimensionType getDimType(const Tensor* origin) {
    auto dimformat = TensorUtils::getDescribe(origin)->dimensionFormat;
    switch (dimformat) {
        case MNN_DATA_FORMAT_NHWC:
            return Tensor::TENSORFLOW;
        case MNN_DATA_FORMAT_NCHW:
            return Tensor::CAFFE;
        case MNN_DATA_FORMAT_NC4HW4:
            return Tensor::CAFFE_C4;
        default:
            break;
    }
    return Tensor::CAFFE;
}
class MergeExpr : public Solution {
public:
    MergeExpr(const Optimizer::Merge* merge, int inputSize, int outputSize) : Solution(inputSize, outputSize) {
        MNN_ASSERT(nullptr != merge);
        MNN_ASSERT(nullptr != merge->tensors());
        MNN_ASSERT(nullptr != merge->backend());
        MNN_ASSERT(nullptr != merge->oplists());
        MNN_ASSERT(nullptr != merge->outputIndexes());
        MNN_ASSERT(nullptr != merge->inputIndexes());

        //Create tensors
        mTensors.resize(merge->tensors()->size());
        for (int i=0; i<mTensors.size(); ++i) {
            mTensors[i].reset(new Tensor);
            mRefCount[mTensors[i].get()] = 0;
            auto src = merge->tensors()->GetAs<Blob>(i);
            auto dst = mTensors[i].get();
            dst->setType(src->dataType());
            TensorUtils::getDescribe(dst)->dimensionFormat = src->dataFormat();
            if (nullptr == src->dims() || nullptr == src->dims()->data()) {
                dst->buffer().dimensions = 0;
            } else {
                auto dimSize = src->dims()->size();
                for (int d=0; d<dimSize; ++d) {
                    dst->setLength(d, src->dims()->data()[d]);
                }
            }
            TensorUtils::setLinearLayout(dst);
        }
        mOutputs.resize(merge->outputIndexes()->size());
        for (int i=0; i<merge->outputIndexes()->size(); ++i) {
            mOutputs[i].first = mTensors[merge->outputIndexes()->data()[i]].get();
        }
        mInputs.resize(merge->inputIndexes()->size());
        for (int i=0; i<merge->inputIndexes()->size(); ++i) {
            mInputs[i].first = mTensors[merge->inputIndexes()->data()[i]].get();
            //For cpu backend, use outputside input ptr, set refcount +1 and ensure not free it
            if (MNN_FORWARD_CPU == (MNNForwardType)merge->backend()->type()) {
                mRefCount[mInputs[i].first] = 1;
            }
        }
        //Create Backend
        auto backendInfo = merge->backend();
        auto creator = MNNGetExtraBackendCreator((MNNForwardType)backendInfo->type());
        if (nullptr == creator) {
            mValid = false;
            MNN_ERROR("Get Backend Creator Error\n");
            return;
        }
        Backend::Info info;
        info.type = (MNNForwardType)backendInfo->type();
        info.numThread = backendInfo->numberThread();
        info.mode = Backend::Info::INDIRECT;
        BackendConfig backendConfig;
        backendConfig.memory = (BackendConfig::MemoryMode)backendInfo->memroy();
        backendConfig.power = (BackendConfig::PowerMode)backendInfo->power();
        backendConfig.precision = (BackendConfig::PrecisionMode)backendInfo->precision();
        info.user = &backendConfig;
        creator->onValid(info);
        mDirect = info.mode == Backend::Info::DIRECT;
        mBackend.reset(creator->onCreate(info));
        if (nullptr == mBackend) {
            MNN_ERROR("Create Backend Error\n");
            mValid = false;
            return;
        }

        //Create Execution
        mExecutions.resize(merge->oplists()->size());
        for (int i=0; i<mExecutions.size(); ++i) {
            auto op = merge->oplists()->GetAs<Op>(i);
            if (nullptr != op->inputIndexes()) {
                mExecutions[i].mInputs.resize(op->inputIndexes()->size());
                for (int j=0; j<mExecutions[i].mInputs.size(); ++j) {
                    auto inputIndex = op->inputIndexes()->data()[j];
                    mRefCount[mTensors[inputIndex].get()] += 1;
                    mExecutions[i].mInputs[j] = mTensors[inputIndex].get();
                }
            }
            if (nullptr != op->outputIndexes()) {
                mExecutions[i].mOutputs.resize(op->outputIndexes()->size());
                for (int j=0; j<mExecutions[i].mOutputs.size(); ++j) {
                    mExecutions[i].mOutputs[j] = mTensors[op->outputIndexes()->data()[j]].get();
                }
            }
            mExecutions[i].mExecution.reset(mBackend->onCreate(mExecutions[i].mInputs, mExecutions[i].mOutputs, op));
            if(nullptr == mExecutions[i].mExecution) {
                mValid = false;
                MNN_ERROR("Create Execution Error\n");
                return;
            }
        }
    }
    
    ~ MergeExpr () {
        mExecutions.clear();
        mBackend.reset();
    }
    virtual ErrorCode onComputeInfo(const std::vector<const Variable::Info*>& inputs,
                                    const std::vector<Variable::Info*>& outputs) override {
        MNN_ASSERT(outputs.size() == mOutputs.size());
        MNN_ASSERT(inputs.size() == mInputs.size());
        for (int i=0; i<inputs.size(); ++i) {
            auto src = inputs[i];
            auto check = mInputs[i];
            if (src->dim.size() != check.first->dimensions()) {
                MNN_ERROR("Input size not match for merge executor\n");
                return COMPUTE_SIZE_ERROR;
            }
            for (int d=0; d<src->dim.size(); ++d) {
                if (src->dim[d] != check.first->length(d)) {
                    MNN_ERROR("Input size not match for merge executor\n");
                    return COMPUTE_SIZE_ERROR;
                }
            }
        }
        for (int i=0; i<outputs.size(); ++i) {
            Utils::copyTensorToInfo(outputs[i], mOutputs[i].first);
        }
        return NO_ERROR;
    }
    virtual ErrorCode onAlloc(const std::vector<const Variable::Info*>& inputs,
                              const std::vector<Variable::Info*>& outputs) override {
        mBackend->onClearBuffer();
        mBackend->onResizeBegin();
        if (MNN_FORWARD_CPU == mBackend->type()) {
            for (int i=0; i<mInputs.size(); ++i) {
                mInputs[i].first->buffer().host = (uint8_t*)inputs[i]->ptr;
            }
        } else {
            for (int i=0; i<mInputs.size(); ++i) {
                mInputs[i].second.reset(new Tensor);
                Utils::copyInfoToTensor(mInputs[i].second.get(), inputs[i]);
                mInputs[i].second->buffer().host = (uint8_t*)inputs[i]->ptr;
                mBackend->onAcquireBuffer(mInputs[i].first, Backend::DYNAMIC);
                TensorUtils::getDescribe(mInputs[i].first)->backend = mBackend.get();
            }
        }

        auto refCount = mRefCount;
        std::set<Tensor*> allocated;
        std::shared_ptr<void> _defer(nullptr, [this](void*) {
            mBackend->onResizeEnd();
        });
        for (int i=0; i<mExecutions.size(); ++i) {
            auto& cmd = mExecutions[i];
            for (auto tensor : cmd.mOutputs) {
                bool success = mBackend->onAcquireBuffer(tensor, Backend::DYNAMIC);
                if (!success) {
                    return OUT_OF_MEMORY;
                }
                TensorUtils::getDescribe(tensor)->backend = mBackend.get();
            }
            auto code = cmd.resize();
            if (NO_ERROR != code) {
                return code;
            }
            for (auto tensor : cmd.mInputs) {
                refCount[tensor]-=1;
                if (refCount[tensor] == 0) {
                    mBackend->onReleaseBuffer(tensor, Backend::DYNAMIC);
                }
            }
        }
        if (MNN_FORWARD_CPU == mBackend->type()) {
            for (int i=0; i<mOutputs.size(); ++i) {
                outputs[i]->ptr = mOutputs[i].first->host<float>();
            }
        } else {
            for (int i=0; i<mOutputs.size(); ++i) {
                mOutputs[i].second.reset(new Tensor(mOutputs[i].first, getDimType(mOutputs[i].first)));
                MNN_ASSERT(TensorUtils::getDescribe(mOutputs[i].first)->backend != nullptr);
                outputs[i]->ptr = mOutputs[i].second->host<float>();
            }
        }
        return NO_ERROR;
    }
    virtual ErrorCode onComputeContent() override {
        if (MNN_FORWARD_CPU != mBackend->type()) {
            for (auto& input : mInputs) {
                input.first->copyFromHostTensor(input.second.get());
            }
        }
        mBackend->onExecuteBegin();
        if (mDirect) {
            for (auto& cmd : mExecutions) {
                auto code = cmd.run();
                if (NO_ERROR != code) {
                    mBackend->onExecuteEnd();
                    return code;
                }
            }
        }
        mBackend->onExecuteEnd();
        if (MNN_FORWARD_CPU != mBackend->type()) {
            for (auto& tensor : mOutputs) {
                tensor.first->copyToHostTensor(tensor.second.get());
            }
        }
        return NO_ERROR;
    }
    
    // Map output's content to host
    virtual void* onMapContent(int index)  override {
        if (nullptr != mOutputs[index].second) {
            return mOutputs[index].second->host<float>();
        }
        return mOutputs[index].first->host<float>();
    }
    virtual void onUnMapContent(int index) override {
        return;
    }

    bool valid() const {return mValid;}
private:
    std::shared_ptr<Backend> mBackend;
    std::vector<Command> mExecutions;
    std::vector<std::shared_ptr<Tensor>> mTensors;
    std::vector<std::pair<Tensor*, std::shared_ptr<Tensor>>> mInputs;
    std::vector<std::pair<Tensor*, std::shared_ptr<Tensor>>> mOutputs;
    std::map<Tensor*, int> mRefCount;
    bool mValid = true;
    bool mDirect = true;
};
class InsideExpr : public Solution {
public:
    InsideExpr(std::shared_ptr<Backend> bn, const Op* op, int inputSize, int outputSize);
    ~InsideExpr();

    virtual ErrorCode onComputeInfo(const std::vector<const Variable::Info*>& inputs,
                                    const std::vector<Variable::Info*>& outputs) override;
    virtual ErrorCode onAlloc(const std::vector<const Variable::Info*>& inputs,
                              const std::vector<Variable::Info*>& outputs) override;
    virtual ErrorCode onComputeContent() override;
    virtual Solution::Requirement onGetRequirement() const override;

    // Map output's content to host
    virtual void* onMapContent(int index) override;
    virtual void onUnMapContent(int index) override;

private:
    void _makeInfo();

    std::vector<std::shared_ptr<Tensor>> mOutputs;
    std::vector<std::shared_ptr<Tensor>> mInputs;
    std::shared_ptr<Command> mCommand;
    std::shared_ptr<Backend> mBackend;
    const Op* mOp;
};

InsideExpr::InsideExpr(std::shared_ptr<Backend> bn, const Op* op, int inputSize, int outputSize)
    : Solution(inputSize, outputSize) {
    MNN_ASSERT(nullptr != bn);
    mOp = op;
    mOutputs.resize(mOutputSize);
    for (auto& v : mOutputs) {
        v.reset(new Tensor);
    }
    mBackend = bn;
}
InsideExpr::~InsideExpr() {
    for (auto& v : mOutputs) {
        if (v->host<void>() != nullptr) {
            mBackend->onReleaseBuffer(v.get(), Backend::STATIC);
        }
    }
}

void InsideExpr::_makeInfo() {
    if (nullptr == mCommand) {
        mCommand.reset(new Command);
        mCommand->mOutputs.resize(mOutputs.size());
        for (int i = 0; i < mOutputs.size(); ++i) {
            mCommand->mOutputs[i] = mOutputs[i].get();
        }
        mCommand->mInputs.resize(mInputSize);
        mInputs.resize(mInputSize);
        for (int i = 0; i < mInputSize; ++i) {
            mInputs[i].reset(new Tensor);
            mCommand->mInputs[i] = mInputs[i].get();
        }
    }
}
Solution::Requirement InsideExpr::onGetRequirement() const {
    Solution::Requirement req;
    auto op = mOp;
    req.contentNeedContent.resize(mInputSize);
    req.shapeNeedContent.resize(mInputSize);
    for (int i = 0; i < mInputSize; ++i) {
        req.contentNeedContent[i] = SizeComputer::opNeedContent(op->type(), i);
        req.shapeNeedContent[i]   = false;
    }
    auto needIndexId = SizeComputer::needInputContent(mOp);
    for (auto index : needIndexId) {
        if (index < req.shapeNeedContent.size()) {
            req.shapeNeedContent[index] = true;
        }
    }
    return req;
}

ErrorCode InsideExpr::onComputeInfo(const std::vector<const Variable::Info*>& inputs,
                                    const std::vector<Variable::Info*>& outputs) {
    _makeInfo();
    auto op = mOp;
    // TODO Support Every Op for user defined shape
    if (op->type() == OpType_Input) {
        auto inputParm              = op->main_as_Input();
        auto output                 = mCommand->mOutputs[0];
        if (nullptr != inputParm->dims()) {
            output->buffer().dimensions = inputParm->dims()->size();
            for (int i = 0; i < output->dimensions(); ++i) {
                auto dim = inputParm->dims()->data()[i];
                if (-1 == dim && 0 == i) {
                    dim = 1;
                }
                if (0 > dim) {
                    MNN_ERROR("The Input %s is not ready: order=%d, pos=%d, dim=%d\n", mOp->name()->c_str(),
                              inputParm->dformat(), i, dim);
                    return COMPUTE_SIZE_ERROR;
                }
                output->setLength(i, dim);
            }
        } else {
            output->buffer().dimensions = 0;
        }
        output->setType(inputParm->dtype());
        TensorUtils::getDescribe(output)->dimensionFormat = inputParm->dformat();
        auto shape                                        = outputs[0];
        Utils::copyTensorToInfo(shape, output);
        return NO_ERROR;
    }

    MNN_ASSERT(inputs.size() == mInputs.size());
    for (int i = 0; i < mInputs.size(); ++i) {
        auto tensor = mInputs[i];
        Utils::copyInfoToTensor(tensor.get(), inputs[i]);
    }
    bool res = SizeComputer::computeOutputSize(op, mCommand->mInputs, mCommand->mOutputs);
    if (!res) {
        // Compute Error
        FUNC_PRINT(op->type());
        return COMPUTE_SIZE_ERROR;
    }
    for (int i = 0; i < mOutputs.size(); ++i) {
        auto tensor = mCommand->mOutputs[i];
        auto shape  = outputs[i];
        Utils::copyTensorToInfo(shape, tensor);
    }
    return NO_ERROR;
}
void* InsideExpr::onMapContent(int index) {
    return mOutputs[index]->host<float>();
}
void InsideExpr::onUnMapContent(int index) {
    // Do nothing
}
ErrorCode InsideExpr::onAlloc(const std::vector<const Variable::Info*>& inputs,
                              const std::vector<Variable::Info*>& outputs) {
    for (auto& output : mOutputs) {
        if (output->host<float>() != nullptr) {
            mBackend->onReleaseBuffer(output.get(), Backend::STATIC);
            output->buffer().host = nullptr;
        }
        TensorUtils::setLinearLayout(output.get());
        auto res = mBackend->onAcquireBuffer(output.get(), Backend::STATIC);
        if (!res) {
            return OUT_OF_MEMORY;
        }
    }
    for (int i = 0; i < outputs.size(); ++i) {
        outputs[i]->ptr = mOutputs[i]->host<float>();
    }
    auto op = mOp;
    if (op->type() == OpType_Input) {
        return NO_ERROR;
    }
    for (int i = 0; i < mInputs.size(); ++i) {
        mInputs[i]->buffer().host = (uint8_t*)inputs[i]->ptr;
    }
    if (nullptr == mCommand->mExecution) {
        mCommand->mExecution.reset(mBackend->onCreate(mCommand->mInputs, mCommand->mOutputs, op));
        if (nullptr == mCommand->mExecution) {
            return NOT_SUPPORT;
        }
    }
    mCommand->mExecution->onResize(mCommand->mInputs, mCommand->mOutputs);
    return NO_ERROR;
}
ErrorCode InsideExpr::onComputeContent() {
    auto op = mOp;
    if (op->type() == OpType_Input) {
        return NO_ERROR;
    }
    auto code = mCommand->mExecution->onExecute(mCommand->mInputs, mCommand->mOutputs);
    return code;
}
DefaultSolutionCreator::DefaultSolutionCreator() {
    auto factory = MNNGetExtraBackendCreator(MNN_FORWARD_CPU);
    Backend::Info info;
    info.numThread = 1;
    info.type      = MNN_FORWARD_CPU;
    mBackend.reset(factory->onCreate(info));
}
Solution* DefaultSolutionCreator::onCreate(const Op* op, int inputSize, int outputSize) {
    if (OpType_PLUGIN != op->type()) {
        return new InsideExpr(mBackend, op, inputSize, outputSize);
    }
    if (nullptr != op->main_as_Plugin()) {
        if (op->main_as_Plugin()->type()->str() == "Merge") {
            auto blob = op->main_as_Plugin()->buffer()->GetAs<Blob>(0);
            auto merge = flatbuffers::GetRoot<MNN::Optimizer::Merge>(blob->uint8s()->data());
            return new MergeExpr(merge, inputSize, outputSize);
        }
    }
    return nullptr;
}
}; // namespace Express
}; // namespace MNN
