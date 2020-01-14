//
//  Executor.cpp
//  MNN
//
//  Created by MNN on 2019/07/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Executor.hpp>
#include "core/Session.hpp"
#include "core/TensorUtils.hpp"
#include "Utils.hpp"
#include "core/Backend.hpp"
#include <MNN/Tensor.hpp>
#include "BasicOptimizer_generated.h"
namespace MNN {
namespace Express {
void Executor::setGlobalExecutorConfig(MNNForwardType type, const BackendConfig& config, int numberThread) {
    std::lock_guard<std::mutex> _l(mMutex);
    auto creator = MNNGetExtraBackendCreator(type);
    if (nullptr == creator) {
        MNN_ERROR("Error to find creator of %d\n", type);
        return;
    }
    mSolutions.clear();
    Backend::Info info;
    info.type = type;
    info.numThread = numberThread;
    std::shared_ptr<Backend> bn(creator->onCreate(info));
    mBackend = bn;
}
void Executor::gc(GCFlag flag) {
    std::lock_guard<std::mutex> _l(mMutex);
    mSolutions.clear();
    mBackend->onClearBuffer();
}

std::shared_ptr<Executor> Executor::getGlobalExecutor() {
    static std::once_flag of;
    static std::shared_ptr<Executor> gExecutor;
    std::call_once(of, [&]() {
        auto creator = MNNGetExtraBackendCreator(MNN_FORWARD_CPU);
        SizeComputerSuite::init();
        Backend::Info info;
        info.type = MNN_FORWARD_CPU;
        info.numThread = 1;
        std::shared_ptr<Backend> bn(creator->onCreate(info));
        gExecutor.reset(new Executor(bn));
    });
    return gExecutor;
}

class Solution {
public:
    Solution(){}
    virtual ~ Solution(){}
    virtual ErrorCode computeInfo(Expr* expr) = 0;
    virtual ErrorCode compute(Expr* expr) = 0;
};
class UnitSolution : public Solution {
public:
    UnitSolution(Expr* expr, std::shared_ptr<Backend> bn) {
        mOutputs.resize(expr->outputSize());
        mContent.resize(expr->outputSize());
        for (int i=0; i<mOutputs.size(); ++i) {
            mContent[i].reset(new Tensor);
            mOutputs[i] = mContent[i].get();
            mOutputs[i]->buffer().host = nullptr;
        }
        mInputs.resize(expr->inputs().size());
        mInputContent.resize(expr->inputs().size());
        for (int i=0; i<mInputs.size(); ++i) {
            mInputContent[i].reset(new Tensor);
            mInputs[i] = mInputContent[i].get();
            mInputs[i]->buffer().host = nullptr;
        }
        mBackend = bn;
        mExpr = expr;
    }
    ~ UnitSolution() {
        for (auto t : mOutputs) {
            if (nullptr != t->host<void>()) {
                mBackend->onReleaseBuffer(t, Backend::STATIC);
            }
        }
        mExpr->setInfoDirty();
    }
    virtual ErrorCode computeInfo(Expr* expr) override {
        auto op = expr->get();
        for (int i = 0; i < expr->inputs().size(); ++i) {
            auto inputExpr = expr->inputs()[i]->expr();
            Utils::copyInfoToTensor(mInputContent[i].get(), inputExpr.first->outputInfo(inputExpr.second));
        }
        bool res = SizeComputer::computeOutputSize(op, mInputs, mOutputs);
        if (!res) {
            // Compute Error
    #ifdef MNN_EXPRESS_ERROR_REPORT
            FUNC_PRINT(op->type());
    #endif
            return COMPUTE_SIZE_ERROR;
        }
        for (int i = 0; i < mOutputs.size(); ++i) {
            auto tensor = mOutputs[i];
            for (int j = 0; j < tensor->dimensions(); ++j) {
                if (tensor->length(j) <= 0) {
    #ifdef MNN_EXPRESS_ERROR_REPORT
                    if (nullptr != op->name()) {
                        auto name = op->name()->str();
                        MNN_ERROR("Error to compute shape for %s\n", op->name()->c_str());
                    }
    #endif
                    return COMPUTE_SIZE_ERROR;
                }
            }
            auto shape  = expr->outputInfo(i);
            Utils::copyTensorToInfo(shape, tensor);
        }
        mNeedResize = true;
        return NO_ERROR;
    }
    ErrorCode prepare(Expr* expr) {
        for (int i = 0; i < expr->inputs().size(); ++i) {
            auto inputExpr = expr->inputs()[i]->expr();
            mInputContent[i]->buffer().host = (uint8_t*)inputExpr.first->outputInfo(inputExpr.second)->ptr;
        }
        if (nullptr == mExecution) {
            mExecution.reset(mBackend->onCreate(mInputs, mOutputs, expr->get()));
        }
        for (auto& output : mOutputs) {
            if (output->host<float>() != nullptr) {
                mBackend->onReleaseBuffer(output, Backend::STATIC);
                output->buffer().host = nullptr;
            }
            TensorUtils::setLinearLayout(output);
            auto res = mBackend->onAcquireBuffer(output, Backend::STATIC);
            if (!res) {
                return OUT_OF_MEMORY;
            }
        }
        for (int i = 0; i < mOutputs.size(); ++i) {
            expr->outputInfo(i)->ptr = mOutputs[i]->host<void>();
        }
        return mExecution->onResize(mInputs, mOutputs);
    }
    virtual ErrorCode compute(Expr* expr) override {
        if (mNeedResize) {
            auto code = prepare(expr);
            if (NO_ERROR != code) {
                return code;
            }
            mNeedResize = false;
        }
        mBackend->onExecuteBegin();
        auto code = mExecution->onExecute(mInputs, mOutputs);
        mBackend->onExecuteEnd();
        return code;
    }
private:
    std::shared_ptr<Execution> mExecution;
    std::vector<Tensor*> mInputs;
    std::vector<Tensor*> mOutputs;
    std::vector<std::shared_ptr<Tensor>> mContent;
    std::vector<std::shared_ptr<Tensor>> mInputContent;
    std::shared_ptr<Backend> mBackend;
    bool mNeedResize = false;
    Expr* mExpr;
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
class MergeExpr : public Solution{
public:
    MergeExpr(const Optimizer::Merge* merge, int inputSize, int outputSize) {
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
    virtual ErrorCode computeInfo(Expr* expr) override {
        MNN_ASSERT(expr->outputSize() == mOutputs.size());
        MNN_ASSERT(expr->inputs().size() == mInputs.size());
        bool needResize = mSession->getNeedResize();
        auto& inputs = expr->inputs();
        if (!needResize) {
            for (int i=0; i<inputs.size(); ++i) {
                auto src = inputs[i]->getInfo();
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
                auto src = inputs[i]->getInfo();
                auto dst = mInputs[i].first;
                Utils::copyInfoToTensor(dst, src);
            }
            mSession->setNeedResize();
            auto code = mSession->resize();
            if (NO_ERROR != code) {
                return code;
            }
        }
        for (int i=0; i<mOutputs.size(); ++i) {
            mOutputs[i].second.reset(new Tensor(mOutputs[i].first, getDimType(mOutputs[i].first)));
            Utils::copyTensorToInfo(expr->outputInfo(i), mOutputs[i].second.get());
        }
        mResized = false;
        return NO_ERROR;
    }
    ErrorCode prepare(Expr* expr) {
        auto inputs = expr->inputs();
        for (int i=0; i<inputs.size(); ++i) {
            auto src = inputs[i]->getInfo();
            TensorUtils::copyShape(mInputs[i].first, mInputs[i].second.get(), true);
            mInputs[i].second->buffer().host = (uint8_t*)src->ptr;
        }
        for (int i=0; i<expr->outputSize(); ++i) {
            expr->outputInfo(i)->ptr = mOutputs[i].second->host<void>();
        }
        return NO_ERROR;
    }
    virtual ErrorCode compute(Expr* expr) override {
        if (!mResized) {
            auto code = prepare(expr);
            if (NO_ERROR != code) {
                return code;
            }
            mResized = true;
        }
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
    bool valid() const {return mValid;}
private:
    std::shared_ptr<Session> mSession;
    std::vector<std::pair<Tensor*, std::shared_ptr<Tensor>>> mInputs;
    std::vector<std::pair<Tensor*, std::shared_ptr<Tensor>>> mOutputs;
    bool mValid = true;
    bool mDirect = true;
    bool mResized = false;
};

Executor::Executor(std::shared_ptr<Backend> bn) {
    mBackend = bn;
}
Executor:: ~Executor() {
    for (auto iter : mSolutions) {
        iter.first->setInfoDirty();
    }
}

Executor::Requirement Executor::onGetRequirement(Expr* expr) const {
    Executor::Requirement req;
    auto op = expr->get();
    auto inputSize = expr->inputs().size();
    req.contentNeedContent.resize(inputSize);
    req.shapeNeedContent.resize(inputSize);
    req.supportError.resize(inputSize);
    if (op->type() == OpType_Extra) {
        for (int i = 0; i < inputSize; ++i) {
            req.contentNeedContent[i] = true;
            req.shapeNeedContent[i]   = false;
            req.supportError[i] = false;
        }
        return req;
    }
    for (int i = 0; i < inputSize; ++i) {
        req.contentNeedContent[i] = SizeComputer::opNeedContent(op->type(), i);
        req.shapeNeedContent[i]   = false;
        if (op->type() != OpType_Concat) {
            req.supportError[i] = false;
        } else {
            req.supportError[i] = true;
        }
    }
    auto needIndexId = SizeComputer::needInputContent(op);
    for (auto index : needIndexId) {
        if (index < req.shapeNeedContent.size()) {
            req.shapeNeedContent[index] = true;
        }
    }
    return req;
}

ErrorCode Executor::onComputeInfo(Expr* expr) {
    if (expr->get()->type() == OpType_Extra) {
        auto param = expr->get()->main_as_Extra();
        if (nullptr == param || "MNN" != param->engine()->str()) {
            FUNC_PRINT(1);
            return NOT_SUPPORT;
        }
    }
    std::lock_guard<std::mutex> _l(mMutex);
    auto iter = mSolutions.find(expr);
    std::shared_ptr<Solution> solution;
    if (iter == mSolutions.end()) {
        if (expr->get()->type() != OpType_Extra) {
            solution.reset(new UnitSolution(expr, mBackend));
        } else {
            auto param = expr->get()->main_as_Extra();
            auto blob = param->info();
            auto merge = flatbuffers::GetRoot<MNN::Optimizer::Merge>(blob->data());
            solution.reset(new MergeExpr(merge, expr->inputs().size(), expr->outputSize()));
        }
        mSolutions[expr] = solution;
    } else {
        solution = iter->second;
    }
    return solution->computeInfo(expr);
}
ErrorCode Executor::onComputeContent(Expr* expr) {
    std::lock_guard<std::mutex> _l(mMutex);
    //MNN_PRINT("Compute for %s \n", EnumNameOpType(expr->get()->type()));
    auto code = mSolutions[expr]->compute(expr);
    return code;
}
void Executor::recycle(Expr* expr) {
    std::lock_guard<std::mutex> _l(mMutex);
    mSolutions.erase(expr);
    return;
}

} // namespace Express
} // namespace MNN
