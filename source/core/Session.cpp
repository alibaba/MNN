//
//  Session.cpp
//  MNN
//
//  Created by MNN on 2018/07/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "core/Session.hpp"
#include <string.h>
#include <map>
#include <set>
#include "core/AutoStorage.h"
#include <MNN/AutoTime.hpp>
#include "core/BackendFactory.hpp"
#include "MNN_generated.h"
#include "core/TensorUtils.hpp"
#include "core/WrapExecution.hpp"

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
            mBackends[iter.first.type].reset(newBn);
        }
        auto backend    = mBackends.find(iter.first.type)->second.get();
        auto cpuBackend = _getDefaultBackend();
        std::shared_ptr<Pipeline> newPipeline(new Pipeline(iter.second, backend, cpuBackend));
        mPipelines.emplace_back(std::move(newPipeline));
    }
    mInputs  = info.inputTensors;
    mOutputs = info.outputTensor;
}

Session::~Session() {
    for (auto& t : mTensors) {
        TensorUtils::clearHandleData(t.second.get());
    }
    mPipelines.clear();
    mBackends.clear();
    mTensors.clear();
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
        if (net->usage() == Usage_INFERENCE && op->type() != OpType_Const) {
            continue;
        }
        if (net->usage() == Usage_TRAIN && op->type() != OpType_TrainableParam) {
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
        std::shared_ptr<Tensor> tensor = mTensors[index].second;
        if (tensor->host<void>() == nullptr && tensor->deviceId() != 0) {
            tensor.reset(Tensor::createHostTensorFromDevice(tensor.get(), true));
            if (tensor.get() == nullptr) {
                MNN_ERROR("failed to copy trained param from device to host\n");
                return INVALID_VALUE;
            }
        }
        ::memcpy((void*)blob->float32s()->data(), tensor->host<float>(), tensor->size());
    }

    return NO_ERROR;
}

} // namespace MNN
