//
//  Session.cpp
//  MNN
//
//  Created by MNN on 2018/07/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "Session.hpp"
#include <string.h>
#include <map>
#include <set>
#include "AutoStorage.h"
#include "AutoTime.hpp"
#include "BackendFactory.hpp"
#include "CPUBackend.hpp"
#include "CommonOptFunction.h"
#include "MNN_generated.h"
#include "TensorUtils.hpp"
#include "WrapExecution.hpp"

using namespace std;

namespace MNN {

Backend* Session::_getDefaultBackend() {
    auto defaultType = MNN_FORWARD_CPU;
    if (mBackends.find(defaultType) == mBackends.end()) {
        Backend::Info info;
        info.type      = defaultType;
        info.numThread = 1;
        mBackends[info.type].reset(BackendFactory::create(info));
    }
    auto cpuBackend = mBackends.find(defaultType)->second.get();
    return cpuBackend;
}

Session::Session(const Schedule::ScheduleInfo& info) {
    if (info.pipelineInfo.empty()) {
        mValid = false;
        return;
    }

    mTensors = info.allTensors;
    for (auto& iter : info.pipelineInfo) {
        if (mBackends.find(iter.first.type) == mBackends.end()) {
            auto newBn = BackendFactory::create(iter.first);
            if (nullptr == newBn) {
                mValid = false;
                return;
            }
            if (newBn->type() != MNN_FORWARD_CPU) {
                newBn->onLoadLibrary(info.library);
            }
            mBackends[iter.first.type].reset(newBn);
        }
        auto backend    = mBackends.find(iter.first.type)->second.get();
        auto cpuBackend = _getDefaultBackend();
        std::unique_ptr<Pipeline> newPipeline(new Pipeline(iter.second, backend, cpuBackend));
        mPipelines.emplace_back(std::move(newPipeline));
    }
    mInputs  = info.inputTensors;
    mOutputs = info.outputTensor;
    for (auto& iter : mInputs) {
        TensorUtils::getDescribe(iter.second)->isInput = true;
    }
}

Session::~Session() {
    for (auto& t : mTensors) {
        TensorUtils::clearHandleData(t.second.get());
    }
}

ErrorCode Session::run() const {
    if (mNeedResize) {
        MNN_ERROR("Can't run session because not resized");
        return COMPUTE_SIZE_ERROR;
    }
    for (auto& iter : mPipelines) {
        auto error = iter->execute();
        if (NO_ERROR != error) {
            return error;
        }
    }
    return NO_ERROR;
}

ErrorCode Session::runWithCallBack(const TensorCallBackWithInfo& before, const TensorCallBackWithInfo& end,
                                   bool sync) const {
    if (mNeedResize) {
        MNN_ERROR("Can't run session because not resized");
        return COMPUTE_SIZE_ERROR;
    }
    for (auto& iter : mPipelines) {
        auto error = iter->executeCallBack(before, end);
        if (NO_ERROR != error) {
            return error;
        }
    }
    if (sync) {
        for (auto& bn : mBackends) {
            bn.second->onWaitFinish();
        }
    }
    return NO_ERROR;
}

void Session::_clearCache() {
    for (auto& t : mTensors) {
        auto describe = TensorUtils::getDescribe(t.second.get());
        TensorUtils::clearHandleData(t.second.get());
        describe->useCount = t.first;
        describe->backend  = nullptr;
    }
}

ErrorCode Session::resize() {
    _clearCache();
    for (auto& b : mBackends) {
        b.second->onClearBuffer();
    }

    for (auto& iter : mPipelines) {
        auto error = iter->prepare();
        if (NO_ERROR != error) {
            return error;
        }
    }
    mNeedResize = false;
    for (auto& b : mBackends) {
        b.second->onAllocateBuffer();
    }

    return NO_ERROR;
}

const Backend* Session::getBackEnd(const Tensor* tensor) const {
    return TensorUtils::getDescribe(tensor)->backend;
}

Tensor* Session::getInput(const char* name) const {
    MNN_ASSERT(!mInputs.empty());
    if (nullptr == name) {
        return mInputs.begin()->second;
    }
    auto iter = mInputs.find(name);
    if (iter == mInputs.end()) {
        MNN_PRINT("Error: can't find input: %s\n", name);
        return nullptr;
    }
    return iter->second;
}

Tensor* Session::getOutput(const char* name) const {
    MNN_ASSERT(!mOutputs.empty());
    if (nullptr == name) {
        return mOutputs.begin()->second;
    }

    auto iter = mOutputs.find(name);
    if (iter == mOutputs.end()) {
        MNN_PRINT("Error: can't find output: %s\n", name);
        return nullptr;
    }
    return iter->second;
}

const std::map<std::string, Tensor*>& Session::getInputAll() const {
    return mInputs;
}

const std::map<std::string, Tensor*>& Session::getOutputAll() const {
    return mOutputs;
}

ErrorCode Session::releaseCache() {
    for (auto& p : mPipelines) {
        auto code = p->releaseCache();
        if (NO_ERROR != code) {
            return code;
        }
    }
    return NO_ERROR;
}
ErrorCode Session::updateToModel(Net* net) const {
    int opSize = net->oplists()->size();
    for (int i = 0; i < opSize; ++i) {
        auto op = net->oplists()->GetAs<Op>(i);
        if (op->type() != OpType_Const) {
            continue;
        }
        if (!op->outputIndexes() || op->outputIndexes()->size() != 1) {
            continue;
        }
        auto index = op->outputIndexes()->data()[0];
        auto blob  = op->main_as_Blob();
        if (blob->dataType() != DataType_DT_FLOAT) {
            continue;
        }
        ::memcpy((void*)blob->float32s()->data(), mTensors[index].second->host<float>(),
                 mTensors[index].second->size());
    }

    return NO_ERROR;
}

} // namespace MNN
