//
//  InsideExpr.cpp
//  MNN
//
//  Created by MNN on 2019/06/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "InsideExpr.hpp"
#include "Session.hpp"
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
        SizeComputerSuite::init();
        MNN_ASSERT(nullptr != merge);
        MNN_ASSERT(nullptr != merge->backend());
        MNN_ASSERT(nullptr != merge->oplists());
        MNN_ASSERT(nullptr != merge->outputIndexes());

        //Create tensors
        Schedule::ScheduleInfo schedule;
        std::vector<Schedule::PipelineInfo> pipelineInfos;
        schedule.allTensors.resize(merge->tensorNumber());
        for (int i=0; i<merge->tensorNumber(); ++i) {
            schedule.allTensors[i].second.reset(new Tensor);
        }
        pipelineInfos.resize(merge->oplists()->size());
        for (int i = 0; i < merge->oplists()->size(); ++i) {
            auto& pipelineInfo = pipelineInfos[i];
            auto op = merge->oplists()->GetAs<Op>(i);
            if (nullptr != op->inputIndexes()) {
                auto data = op->inputIndexes()->data();
                pipelineInfo.inputs.resize(op->inputIndexes()->size());
                for (int j = 0; j < op->inputIndexes()->size(); ++j) {
                    auto index = data[j];
                    schedule.allTensors[index].first += 1;
                    pipelineInfo.inputs[j] = schedule.allTensors[index].second.get();
                }
            }
            if (nullptr != op->outputIndexes()) {
                auto data = op->outputIndexes()->data();
                pipelineInfo.outputs.resize(op->outputIndexes()->size());
                for (int j = 0; j < op->outputIndexes()->size(); ++j) {
                    auto index = data[j];
                    pipelineInfo.outputs[j] = schedule.allTensors[index].second.get();
                }
            }
            pipelineInfo.op = op;
        }
        mOutputs.resize(merge->outputIndexes()->size());
        for (int i=0; i<merge->outputIndexes()->size(); ++i) {
            schedule.allTensors[merge->outputIndexes()->data()[i]].first += 1;
            mOutputs[i].first = schedule.allTensors[merge->outputIndexes()->data()[i]].second.get();
        }
        if (nullptr != merge->inputIndexes()) {
            mInputs.resize(merge->inputIndexes()->size());
            for (int i=0; i<merge->inputIndexes()->size(); ++i) {
                mInputs[i].first = schedule.allTensors[merge->inputIndexes()->data()[i]].second.get();
                mInputs[i].second.reset(new Tensor);
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
        schedule.pipelineInfo.emplace_back(std::make_pair(info, pipelineInfos));
        mSession.reset(new Session(schedule));
    }
    
    ~ MergeExpr () {
        //Do nothing
    }
    virtual Requirement onGetRequirement() const override {
        auto size = mInputSize;
        Solution::Requirement req;
        req.contentNeedContent.resize(size);
        req.shapeNeedContent.resize(size);
        req.supportError.resize(size);
        for (int i = 0; i < size; ++i) {
            req.contentNeedContent[i] = true;
            req.shapeNeedContent[i]   = false;
            req.supportError[i] = false;
        }
        return req;
    }
    virtual ErrorCode onComputeInfo(const std::vector<const Variable::Info*>& inputs,
                                    const std::vector<Variable::Info*>& outputs) override {
        MNN_ASSERT(outputs.size() == mOutputs.size());
        MNN_ASSERT(inputs.size() == mInputs.size());
        bool needResize = mSession->getNeedResize();
        if (!needResize) {
            for (int i=0; i<inputs.size(); ++i) {
                auto src = inputs[i];
                auto check = mInputs[i].first;
                if (src->dim.size() != check->dimensions()) {
                    needResize = true;
                    break;
                }
                for (int d=0; d<src->dim.size(); ++d) {
                    if (src->dim[d] != check->length(d)) {
                        needResize = true;
                        break;
                    }
                }
                if (needResize) {
                    break;
                }
            }
        }
        if (needResize) {
            for (int i=0; i<inputs.size(); ++i) {
                auto src = inputs[i];
                auto dst = mInputs[i].first;
                Utils::copyInfoToTensor(dst, src);
            }
            mSession->setNeedResize();
            auto code = mSession->resize();
            if (NO_ERROR != code) {
                return code;
            }
        }
        for (int i=0; i<outputs.size(); ++i) {
            mOutputs[i].second.reset(new Tensor(mOutputs[i].first, getDimType(mOutputs[i].first)));
            Utils::copyTensorToInfo(outputs[i], mOutputs[i].second.get());
        }
        return NO_ERROR;
    }
    virtual ErrorCode onAlloc(const std::vector<const Variable::Info*>& inputs,
                              const std::vector<Variable::Info*>& outputs) override {
        for (int i=0; i<inputs.size(); ++i) {
            auto src = inputs[i];
            TensorUtils::copyShape(mInputs[i].first, mInputs[i].second.get(), true);
            mInputs[i].second->buffer().host = (uint8_t*)src->ptr;
        }
        return NO_ERROR;
    }
    virtual ErrorCode onComputeContent(const std::vector<const Variable::Info*>& inputs,
    const std::vector<Variable::Info*>& outputs) override {
        for (auto& input : mInputs) {
            input.first->copyFromHostTensor(input.second.get());
        }
        auto code = mSession->run();
        if (NO_ERROR != code) {
            return code;
        }
        for (auto& tensor : mOutputs) {
            tensor.first->copyToHostTensor(tensor.second.get());
        }
        return NO_ERROR;
    }
    
    // Map output's content to host
    virtual void* onMapContent(int index)  override {
        return mOutputs[index].second->host<float>();
    }
    virtual void onUnMapContent(int index) override {
        return;
    }

    bool valid() const {return mValid;}
private:
    std::shared_ptr<Session> mSession;
    std::vector<std::pair<Tensor*, std::shared_ptr<Tensor>>> mInputs;
    std::vector<std::pair<Tensor*, std::shared_ptr<Tensor>>> mOutputs;
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
    virtual ErrorCode onComputeContent(const std::vector<const Variable::Info*>& inputs,
    const std::vector<Variable::Info*>& outputs) override;
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
    SizeComputerSuite::init();
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
    req.supportError.resize(mInputSize);
    for (int i = 0; i < mInputSize; ++i) {
        req.contentNeedContent[i] = SizeComputer::opNeedContent(op->type(), i);
        req.shapeNeedContent[i]   = false;
        if (op->type() != OpType_Concat) {
            req.supportError[i] = false;
        } else {
            req.supportError[i] = true;
        }
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
#ifdef MNN_DEBUG_INPUT
                    MNN_ERROR("The Input %s is not ready: order=%d, pos=%d, dim=%d\n", mOp->name()->c_str(),
                              inputParm->dformat(), i, dim);
#endif
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
#ifdef MNN_EXPRESS_ERROR_REPORT
        FUNC_PRINT(op->type());
#endif
        return COMPUTE_SIZE_ERROR;
    }
    for (int i = 0; i < mOutputs.size(); ++i) {
        auto tensor = mCommand->mOutputs[i];
        for (int j = 0; j < tensor->dimensions(); ++j) {
            if (tensor->length(j) <= 0) {
                auto name = op->name()->str();
#ifdef MNN_EXPRESS_ERROR_REPORT
                MNN_ERROR("Error to compute shape for %s\n", op->name()->c_str());
#endif
                return COMPUTE_SIZE_ERROR;
            }
        }
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
    if (nullptr == mCommand->mExecution) {
        mCommand->mExecution.reset(mBackend->onCreate(mCommand->mInputs, mCommand->mOutputs, op));
        if (nullptr == mCommand->mExecution) {
            return NOT_SUPPORT;
        }
    }
    for (int i = 0; i < mInputs.size(); ++i) {
        if (nullptr != inputs[i]) {
            mInputs[i]->buffer().host = (uint8_t*)inputs[i]->ptr;
        }
    }
    mCommand->mExecution->onResize(mCommand->mInputs, mCommand->mOutputs);
    return NO_ERROR;
}
ErrorCode InsideExpr::onComputeContent(const std::vector<const Variable::Info*>& inputs,
const std::vector<Variable::Info*>& outputs) {
    auto op = mOp;
    if (op->type() == OpType_Input) {
        return INPUT_DATA_ERROR;
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
    if (OpType_Extra != op->type()) {
        return new InsideExpr(mBackend, op, inputSize, outputSize);
    }
    if (nullptr != op->main_as_Extra()) {
        if (op->main_as_Extra()->type()->str() == "Session") {
            auto blob = op->main_as_Extra()->info();
            auto merge = flatbuffers::GetRoot<MNN::Optimizer::Merge>(blob->data());
            return new MergeExpr(merge, inputSize, outputSize);
        }
    }
    return nullptr;
}
}; // namespace Express
}; // namespace MNN
