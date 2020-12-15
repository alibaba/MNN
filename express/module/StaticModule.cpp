//
//  StaticModule.cpp
//  MNN
//
//  Created by MNN on b'2020/09/10'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "StaticModule.hpp"
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/AutoTime.hpp>
#include "core/TensorUtils.hpp"
#include "core/Session.hpp"
#include <MNN/expr/Executor.hpp>
#include <MNN/AutoTime.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include "core/MNNMemoryUtils.h"
#include "Utils.hpp"

namespace MNN {
namespace Express {
StaticModule::StaticModule(const void* buffer, size_t length, const std::vector<std::string>& inputs, const std::vector<std::string>& outputs, bool shapeFix) : mInputs(inputs), mOutputs(outputs) {
    mShapeFix = shapeFix;
    mOutputNumbers = (int)outputs.size();
    /** Compute:
     std::vector<int, int> mOutputFromTensor;
     std::vector<int, int> mOutputFromInput;
     */
    for (int i=0; i<outputs.size(); ++i) {
        auto& t = outputs[i];
        bool fromInput = false;
        for (int j=0; j<inputs.size(); ++j) {
            if (inputs[j] == t) {
                fromInput = true;
                mOutputFromInput.emplace_back(std::make_pair(i, j));
                break;
            }
        }
        if (fromInput) {
            continue;
        }
        mOutputFromTensor.emplace_back(i);
    }
    if (mOutputFromTensor.empty()) {
        return;
    }

    mNet.reset(Interpreter::createFromBuffer(buffer, length));
#ifdef MNN_EXPR_ENABLE_PROFILER
    mNet->setSessionMode(Interpreter::Session_Debug);
#else
    mNet->setSessionMode(Interpreter::Session_Release);
#endif
    if (mShapeFix) {
        mNet->setSessionMode(Interpreter::Session_Input_Inside);
    } else {
        mNet->setSessionMode(Interpreter::Session_Input_User);
    }

    auto rt = Express::ExecutorScope::Current()->getRuntime();
    // TODO: Add Config
    ScheduleConfig config;
    config.numThread = 1;
    config.type = rt.first.begin()->first;
    config.saveTensors = outputs;
    mSession = mNet->createSession(config, rt);
    mInputTensors.resize(inputs.size());
    for (int i=0; i<inputs.size(); ++i) {
        mInputTensors[i] = mNet->getSessionInput(mSession, inputs[i].c_str());
    }
    mOutputTensors.resize(mOutputFromTensor.size());
    for (int i=0; i<mOutputFromTensor.size(); ++i) {
        mOutputTensors[i] = mNet->getSessionOutput(mSession, outputs[mOutputFromTensor[i]].c_str());
    }
}
StaticModule:: ~ StaticModule() {
    // Do nothing
} 
std::vector<Express::VARP> StaticModule::onForward(const std::vector<Express::VARP>& inputs) {
    AUTOTIME;
    std::vector<Express::VARP> outputs(mOutputNumbers);
    for (auto& iter : mOutputFromInput) {
        outputs[iter.first] = inputs[iter.second];
    }
    if (mOutputFromTensor.empty()) {
        return outputs;
    }
    MNN_ASSERT(inputs.size() == mInputTensors.size());
    for (int i=0; i<inputs.size(); ++i) {
        auto info = inputs[i]->getInfo();
        mInputTensors[i]->buffer().type = info->type;
        auto des = TensorUtils::getDescribe(mInputTensors[i]);
        if (info->order == Express::NCHW) {
            des->dimensionFormat = MNN_DATA_FORMAT_NCHW;
        }
        if (info->order == Express::NHWC) {
            des->dimensionFormat = MNN_DATA_FORMAT_NHWC;
        }
        if (info->order == Express::NC4HW4) {
            des->dimensionFormat = MNN_DATA_FORMAT_NC4HW4;
        }
        mNet->resizeTensor(mInputTensors[i], info->dim);
    }
    if (!mShapeFix) {
        for (int i=0; i<inputs.size(); ++i) {
            mInputTensors[i]->buffer().host = (uint8_t*)inputs[i]->readMap<void>();
        }
        // FIXME: Use Interpreter's API
        mSession->setNeedResize();
    }
    mNet->resizeSession(mSession);
    if (mShapeFix) {
        for (int i=0; i<inputs.size(); ++i) {
            auto srcPtr = inputs[i]->readMap<void>();
            // For Shape only usage input, don't alloc memory
            if (nullptr != mInputTensors[i]->host<void>() && nullptr != srcPtr) {
                ::memcpy(mInputTensors[i]->host<void>(), srcPtr, mInputTensors[i]->size());
            } else if (mInputTensors[i]->deviceId() != 0) {
                // Other backend
                // TODO: Non-copy methed
                auto exprInfo = inputs[i]->expr();
                auto inside = exprInfo.first->inside();
                mInputTensors[i]->copyFromHostTensor(inside->mOutputTensors[exprInfo.second]);
            }
        }
    }
#ifdef MNN_EXPR_ENABLE_PROFILER
    auto globalExecutor = ExecutorScope::Current();
    Timer cost;
    TensorCallBackWithInfo beforeCallBack = [&cost] (const std::vector<Tensor*>&, const OperatorInfo* info) {
        cost.reset();
        return true;
    };
    TensorCallBackWithInfo afterCallBack = [&cost, globalExecutor] (const std::vector<Tensor*>&, const OperatorInfo* info) {
        auto costTimes = (float)cost.durationInUs() / 1000.0f;
        globalExecutor->addOpCostTime(info->type(), costTimes);
        globalExecutor->addOpFlops(info->type(), info->flops());
        return true;
    };
    mNet->runSessionWithCallBackInfo(mSession, beforeCallBack, afterCallBack);
#else
    mNet->runSession(mSession);
#endif
    for (int i=0; i<mOutputTensors.size(); ++i) {
        Express::Variable::Info info;
        auto currentTensor = mOutputTensors[i];
        info.dim = currentTensor->shape();
        info.type = currentTensor->getType();
        auto format = TensorUtils::getDescribe(mOutputTensors[i])->dimensionFormat;
        info.order = Express::NHWC;
        if (format == MNN_DATA_FORMAT_NCHW) {
            info.order = Express::NCHW;
        } else if (format == MNN_DATA_FORMAT_NC4HW4) {
            info.order = Express::NC4HW4;
        }
        if (currentTensor->buffer().device != 0) {
            std::shared_ptr<Tensor> tmpTensor(new Tensor(currentTensor, Tensor::CAFFE, false));
            tmpTensor->buffer().host = (uint8_t*)MNNMemoryAllocAlign(currentTensor->size(), MNN_MEMORY_ALIGN_DEFAULT);
            currentTensor->copyToHostTensor(tmpTensor.get());
            outputs[mOutputFromTensor[i]] = Express::Variable::create(Express::Expr::create(std::move(info), tmpTensor->host<void>(), Express::VARP::CONSTANT, Expr::MemoryType::MOVE), 0);
        } else {
            outputs[mOutputFromTensor[i]] = Express::Variable::create(Express::Expr::create(std::move(info), mOutputTensors[i]->host<void>(), Express::VARP::CONSTANT, Expr::MemoryType::REF), 0);
        }
    }
    return outputs;
}

Module* StaticModule::clone(CloneContext* ctx) const {
    StaticModule* module(new StaticModule);
    module->mInputs = mInputs;
    module->mOutputs = mOutputs;

    module->mShapeFix = mShapeFix;
    module->mOutputNumbers = mOutputNumbers;
    module->mOutputFromInput = mOutputFromInput;
    module->mOutputFromTensor = mOutputFromTensor;
    if (mOutputFromTensor.empty()) {
        return this->cloneBaseTo(ctx, module);
    }

    module->mNet = mNet;

    auto rt = Express::ExecutorScope::Current()->getRuntime();
    ScheduleConfig config;
    config.numThread = 1;
    config.type = rt.first.begin()->first;
    config.saveTensors = mOutputs;
    module->mSession = module->mNet->createSession(config, rt);

    module->mInputTensors.resize(mInputs.size());
    module->mOutputTensors.resize(mOutputFromTensor.size());
    for (int i=0; i<mInputs.size(); ++i) {
        module->mInputTensors[i] =
            module->mNet->getSessionInput(module->mSession, mInputs[i].c_str());
    }
    for (int i=0; i<mOutputFromTensor.size(); ++i) {
        module->mOutputTensors[i] = module->mNet->getSessionOutput(
            module->mSession, mOutputs[mOutputFromTensor[i]].c_str());
    }
    return this->cloneBaseTo(ctx, module);
}

}
}
