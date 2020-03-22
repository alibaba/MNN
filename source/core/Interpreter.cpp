//
//  Interpreter.cpp
//  MNN
//
//  Created by MNN on 2018/07/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <math.h>
#include <stdio.h>
#include <algorithm>
#include <vector>
#include "MNN_generated.h"
#include "core/AutoStorage.h"
#include <MNN/Interpreter.hpp>
#include "core/Session.hpp"
#include "core/FileLoader.hpp"
namespace MNN {

struct Content {
    AutoStorage<uint8_t> buffer;
    const Net* net = nullptr;
    std::vector<std::unique_ptr<Session>> sessions;
    std::map<const Tensor*, const Session*> tensorMap;
};

Interpreter* Interpreter::createFromFile(const char* file) {
    if (nullptr == file) {
        MNN_PRINT("NULL file for create interpreter\n");
        return nullptr;
    }
    std::unique_ptr<FileLoader> loader(new FileLoader(file));
    if (!loader->valid()) {
        MNN_PRINT("Create interpreter failed, open %s error\n", file);
        return nullptr;
    }
    bool result = loader->read();
    if (!result) {
        MNN_PRINT("Read file error\n");
        return nullptr;
    }
    if (loader->size() == 0) {
        MNN_PRINT("Create interpreter failed, %s is empty\n", file);
        return nullptr;
    }
    auto net     = new Content;
    bool success = loader->merge(net->buffer);
    if (!success) {
        return nullptr;
    }
    loader.reset();
    return createFromBufferInternal(net);
}
Interpreter* Interpreter::createFromBuffer(const void* buffer, size_t size) {
    if (nullptr == buffer || 0 == size) {
        MNN_PRINT("Buffer is null for create interpreter\n");
        return nullptr;
    }
    auto net = new Content;
    net->buffer.reset((int)size);
    if (nullptr == net->buffer.get()) {
        MNN_ERROR("Memory not enought!\n");
        return nullptr;
    }
    ::memcpy(net->buffer.get(), buffer, size);

    return createFromBufferInternal(net);
}

Interpreter* Interpreter::createFromBufferInternal(Content* net) {
    if (nullptr == net) {
        MNN_PRINT("Buffer is null for create interpreter\n");
        return nullptr;
    }
    flatbuffers::Verifier verify((const uint8_t*)(net->buffer.get()), net->buffer.size());
    if (false == VerifyNetBuffer(verify)) {
        MNN_PRINT("Invalidate buffer to create interpreter\n");
        delete net;
        return nullptr;
    }
    net->net = GetNet(net->buffer.get());
    if (nullptr == net->net->oplists()) {
        MNN_ERROR("Model has no oplist\n");
        delete net;
        return nullptr;
    }
    int opSize = net->net->oplists()->size();
    for (int i=0; i<opSize; ++i) {
        auto op = net->net->oplists()->GetAs<Op>(i);
        if (nullptr == op || nullptr == op->outputIndexes()) {
            MNN_ERROR("Invalid Model, the %d op is empty\n", i);
            delete net;
            return nullptr;
        }
    }
    return new Interpreter(net);
}

Interpreter::Interpreter(Content* net) {
    MNN_ASSERT(nullptr != net);
    mNet      = net;
}

Interpreter::~Interpreter() {
    delete mNet;
}

Session* Interpreter::createMultiPathSession(const std::vector<ScheduleConfig>& configs) {
    if (nullptr == mNet->buffer.get()) {
        MNN_ERROR("The model buffer has been released. Can't create session\n");
        return nullptr;
    }
    auto info       = Schedule::schedule(mNet->net, configs);
    auto newSession = std::unique_ptr<Session>(new Session(info));
    if (!newSession->valid()) {
        MNN_PRINT("Invalide Session!!\n");
        return nullptr;
    }
    auto result = newSession.get();
    if (info.validForResize) {
        result->resize();
    }
    mNet->sessions.emplace_back(std::move(newSession));
    return result;
}

Session* Interpreter::createSession(const ScheduleConfig& config) {
    return createMultiPathSession({config});
}

bool Interpreter::releaseSession(Session* session) {
    for (auto iter = mNet->sessions.begin(); iter != mNet->sessions.end(); iter++) {
        // TODO Delete tensormap
        for (auto tIter = mNet->tensorMap.begin(); tIter != mNet->tensorMap.end();) {
            if (tIter->second == session) {
                tIter = mNet->tensorMap.erase(tIter);
                continue;
            }
            tIter++;
        }

        if ((*iter).get() == session) {
            mNet->sessions.erase(iter);
            return true;
        }
    }
    return false;
}

ErrorCode Interpreter::runSession(Session* session) const {
    return session->run();
}

Tensor* Interpreter::getSessionInput(const Session* session, const char* name) {
    MNN_ASSERT(nullptr != session);
    if (session == nullptr) {
        return nullptr;
    }
    auto tensor = session->getInput(name);
    mNet->tensorMap.insert(std::make_pair(tensor, session));
    return tensor;
}

Tensor* Interpreter::getSessionOutput(const Session* session, const char* name) {
    MNN_ASSERT(nullptr != session);
    auto tensor = session->getOutput(name);
    mNet->tensorMap.insert(std::make_pair(tensor, session));
    return tensor;
}

const std::map<std::string, Tensor*>& Interpreter::getSessionInputAll(const Session* session) const {
    auto& tensors = session->getInputAll();
    for (auto& iter : tensors) {
        mNet->tensorMap.insert(std::make_pair(iter.second, session));
    }
    return tensors;
}

const std::map<std::string, Tensor*>& Interpreter::getSessionOutputAll(const Session* session) const {
    auto& tensors = session->getOutputAll();
    for (auto& iter : tensors) {
        mNet->tensorMap.insert(std::make_pair(iter.second, session));
    }
    return tensors;
}

void Interpreter::resizeSession(Session* session) {
    if (mNet->buffer.get() == nullptr) {
        MNN_ERROR("The model buffer has been released. Can't resize session\n");
        return;
    }
    if (session->getNeedResize()) {
        session->resize();
    }
}

ErrorCode Interpreter::runSessionWithCallBack(const Session* session, const TensorCallBack& before,
                                              const TensorCallBack& after, bool sync) const {
    auto beforeWrap = [&before](const std::vector<Tensor*>& tensors, const OperatorInfo* info) {
        return before(tensors, info->name());
    };
    auto afterWrap = [&after](const std::vector<Tensor*>& tensors, const OperatorInfo* info) {
        return after(tensors, info->name());
    };
    return runSessionWithCallBackInfo(session, beforeWrap, afterWrap, sync);
}

ErrorCode Interpreter::runSessionWithCallBackInfo(const Session* session, const TensorCallBackWithInfo& before,
                                                  const TensorCallBackWithInfo& callBack, bool sync) const {
    return session->runWithCallBack(before, callBack, sync);
}

const Backend* Interpreter::getBackend(const Session* session, const Tensor* tensor) const {
    return session->getBackEnd(tensor);
}

void Interpreter::releaseModel() {
    mNet->buffer.release();
    for (auto& iter : mNet->sessions) {
        iter->releaseCache();
    }
}

void Interpreter::resizeTensor(Tensor* tensor, int batch, int channel, int height, int width) {
    if (tensor->getDimensionType() == Tensor::TENSORFLOW) {
        resizeTensor(tensor, {batch, height, width, channel});
    } else {
        resizeTensor(tensor, {batch, channel, height, width});
    }
}

void Interpreter::resizeTensor(Tensor* tensor, const std::vector<int>& dims) {
    MNN_ASSERT(nullptr != tensor);
    bool dirty = false;
    if (tensor->buffer().dimensions != dims.size()) {
        dirty = true;
    } else {
        for (int i = 0; i < dims.size(); ++i) {
            if (tensor->buffer().dim[i].extent != dims[i]) {
                dirty = true;
                break;
            }
        }
    }

    if (!dirty) {
        return;
    }

    tensor->buffer().dimensions = (int)dims.size();
    for (int i = 0; i < dims.size(); ++i) {
        tensor->buffer().dim[i].extent = dims[i];
    }

    auto relatedSessionIter = mNet->tensorMap.find(tensor);
    MNN_ASSERT(relatedSessionIter != mNet->tensorMap.end());
    ((MNN::Session*)relatedSessionIter->second)->setNeedResize();
}

const char* Interpreter::bizCode() const {
    const flatbuffers::String* code = mNet->net->bizCode();
    return code->c_str();
}

std::pair<const void*, size_t> Interpreter::getModelBuffer() const {
    return std::make_pair(mNet->buffer.get(), mNet->buffer.size());
}
ErrorCode Interpreter::updateSessionToModel(Session* session) {
    if (mNet->buffer.get() == nullptr) {
        MNN_ERROR("Can't updateSessionToModel because you called releaseModel before\n");
        return INPUT_DATA_ERROR;
    }
    return session->updateToModel((Net*)mNet->net);
}

} // namespace MNN
