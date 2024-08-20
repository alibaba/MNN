//
//  Session.cpp
//  MNN
//
//  Created by MNN on 2018/07/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "core/Session.hpp"
#include "core/WrapExecution.hpp"
#include <string.h>
#include <MNN/AutoTime.hpp>
#include <map>
#include <set>
#include "MNN_generated.h"
#include "core/AutoStorage.h"
#include "core/RuntimeFactory.hpp"
#include "core/TensorUtils.hpp"
#include "utils/InitNet.hpp"

using namespace std;

namespace MNN {
static void _createPipelineBackend(Schedule::PipelineInfo& iter, RuntimeInfo& runtime) {
    if (iter.first.cache.first != nullptr) {
        return;
    }
    auto rt    = runtime.first.find(iter.first.info.type)->second.get();
    auto cpuRuntime = runtime.second;
    bool specialUsage = false;
    if (iter.first.info.user != nullptr) {
        specialUsage = iter.first.info.user->flags > 0;
    }
    iter.first.cache.first.reset(rt->onCreate(iter.first.info.user));
    std::shared_ptr<Backend> second;
    if (iter.first.cache.first->type() == MNN_FORWARD_CPU && (!specialUsage)) {
        iter.first.cache.second = iter.first.cache.first;
    } else {
        // Const Backend shouldn't be used as default backend
        // The session may be schedule multi-thread but const backend is the same
        // We need create a new backend to do size compute / not support op compute
        BackendConfig defaultConfig;
        defaultConfig.flags = 4;
        iter.first.cache.second.reset(cpuRuntime->onCreate(&defaultConfig));
    }
}
void Session::ModeGroup::setMode(Interpreter::SessionMode mode) {
    if (mode == Interpreter::Session_Input_Inside || mode == Interpreter::Session_Input_User) {
        inputMode = mode;
    } else if (mode == Interpreter::Session_Output_User || mode == Interpreter::Session_Output_Inside) {
        outputMode = mode;
    } else if (mode == Interpreter::Session_Backend_Auto || mode == Interpreter::Session_Backend_Fix) {
        backendMode = mode;
    } else if (mode == Interpreter::Session_Debug || mode == Interpreter::Session_Release) {
        callBackMode = mode;
    } else if (mode == Interpreter::Session_Resize_Direct || mode == Interpreter::Session_Resize_Defer) {
        resizeMode = mode;
    } else if(mode == Interpreter::Session_Memory_Collect || mode == Interpreter::Session_Memory_Cache) {
        memoryUsageMode = mode;
    } else if(mode == Interpreter::Session_Codegen_Disable || mode == Interpreter::Session_Codegen_Enable) {
        codegenMode = mode;
    }
}
void Session::ModeGroup::setHint(Interpreter::HintMode mode, int hint) {
    switch (mode) {
        case Interpreter::MAX_TUNING_NUMBER:
            maxTuningNumber = hint;
            break;
        case Interpreter::MEM_ALLOCATOR_TYPE:
            runtimeHint.memoryAllocatorType = hint;
            break;
        case Interpreter::WINOGRAD_MEMORY_LEVEL:
            runtimeHint.winogradMemoryUsed = hint;
            break;
        case Interpreter::CPU_LITTLECORE_DECREASE_RATE:
            runtimeHint.cpuDecreaseRate = hint;
            break;
        case Interpreter::GEOMETRY_COMPUTE_MASK:
            geometryMask = hint;
            break;
        case Interpreter::STRICT_CHECK_MODEL:
            checkNetBuffer = hint > 0;
            break;
        case Interpreter::DYNAMIC_QUANT_OPTIONS:
            runtimeHint.dynamicQuantOption = hint;
            break;
        case Interpreter::KVCACHE_QUANT_OPTIONS:
            runtimeHint.kvcacheQuantOption = hint;
            break;
        default:
            break;
    }
}
Session::Session(Schedule::ScheduleInfo&& info, const ModeGroup& mode, RuntimeInfo&& runtime) {
    mMode = mode;
    mRuntime = std::move(runtime);
    if (info.pipelineInfo.empty()) {
        mValid = false;
        return;
    }
    mInfo = std::move(info);
    for (auto& iter : mInfo.pipelineInfo) {
        _createPipelineBackend(iter, mRuntime);
        Pipeline::TuningAttr attr;
        attr.maxTuningNumber = mode.maxTuningNumber;
        attr.autoSetOpType = mode.backendMode == Interpreter::Session_Backend_Auto;
        auto rt    = mRuntime.first.find(iter.first.info.type)->second.get();
        auto cpuRuntime = mRuntime.second;
        std::shared_ptr<Pipeline> newPipeline(new Pipeline( mInfo.externalWeightPath, std::move(iter), mode.inputMode == Interpreter::Session_Input_Inside, mode.outputMode == Interpreter::Session_Output_User, attr, rt, cpuRuntime.get(), mMode.geometryMask));
        mPipelines.emplace_back(std::move(newPipeline));
    }
    mCallBackMode = mode.callBackMode;
    mMemoryUsageMode = mode.memoryUsageMode;
    mCodegenMode = mode.codegenMode;
}

Session::~Session() {
    for (auto& iter : mRuntime.first) {
        iter.second->mCancelled = true;
    }
    waitAsyncResize();
    mInfo.allTensors.clear();
    mPipelines.clear();
    mRuntime.first.clear();
    mRuntime.second = nullptr;
}
Schedule::PipelineInfo& Session::getPipelineInfo(int index) const {
    MNN_ASSERT(index >= 0);
    MNN_ASSERT(index < mPipelines.size());
    return mPipelines[index]->getPipelineInfo();
}

bool Session::loadCache(const void* buffer, size_t size) {
    for (auto iter : mRuntime.first) {
        auto res = iter.second->onSetCache(buffer, size);
        if (res) {
            return true;
        }
    }
    return false;
}
void Session::waitAsyncResize() {
    for (auto& iter : mRuntime.first) {
        iter.second->waitAsyncWork();
    }
}

bool Session::hasAsyncWork() {
    for (auto& iter : mRuntime.first) {
        auto res = iter.second->hasAsyncWork();
        if (res) {
            return true;
        }
    }
    return false;
}


std::pair<const void*, size_t> Session::getCache() {
    // Set cancelled for quickly ending
    for (auto& iter : mRuntime.first) {
        iter.second->mCancelled = true;
    }
    waitAsyncResize();
    
    for (auto iter : mRuntime.first) {
        auto res = iter.second->onGetCache();
        if (res.first != nullptr) {
            return res;
        }
    }
    return std::make_pair(nullptr, 0);
}

ErrorCode Session::run() const {
    if (mNeedResize) {
        MNN_ERROR("Can't run session because not resized\n");
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
        MNN_ERROR("Can't run session because not resized\n");
        return COMPUTE_SIZE_ERROR;
    }
    for (auto& iter : mPipelines) {
        auto error = iter->executeCallBack(before, end);
        if (NO_ERROR != error) {
            return error;
        }
    }
    return NO_ERROR;
}

void Session::_clearCache() {
    for (auto& t : mInfo.allTensors) {
        auto describe = TensorUtils::getDescribe(t.get());
        if (describe->usage == Tensor::InsideDescribe::TRAINABLE || describe->usage == Tensor::InsideDescribe::CONSTANT) {
            continue;
        }
        describe->regions.clear();
    }
}

ErrorCode Session::resize() {
#ifdef LOG_VERBOSE
    for (auto& iter : mInfo.inputTensors) {
        auto& inputTensor = iter.second;
        MNN_PRINT("before resize, input name:%s, ptr:%p, hostPtr:%p,  shape:", iter.first.c_str(), inputTensor, inputTensor->host<void>());
        inputTensor->printShape();
        MNN_PRINT("\n");
    }
#endif
    bool permitCodegen = mCodegenMode == Interpreter::Session_Codegen_Enable;

    bool firstMalloc = false;
    if (mNeedResize) {
        bool debug = mCallBackMode == Interpreter::Session_Debug;
        for (auto& iter : mPipelines) {
            auto error = iter->encode(debug, permitCodegen);
            if (NO_ERROR != error) {
                return error;
            }
        }
        mNeedResize = false;
        mNeedMalloc = true;
        firstMalloc = true;
    }
    if (mNeedMalloc) {
        // Set needResize = true for easy for judge in runSession when error
        mNeedResize = true;
        // Turn Pipeline to Command Buffer and Malloc resource
        // TODO: Separate Schedule and Malloc
        bool forbidReplace = permitCodegen;
        if (mInfo.constReplaceBackend != nullptr) {
            forbidReplace = true;
        }
        for (auto& iter : mPipelines) {
            auto error = iter->allocMemory(firstMalloc, forbidReplace);
            if (NO_ERROR != error) {
                return error;
            }
        }
        if(mMemoryUsageMode == Interpreter::Session_Memory_Collect) {
            #ifdef LOG_VERBOSE
            float memory = 0.0f;
            #endif
            for (auto& iter : mRuntime.first) {
                iter.second->onGabageCollect(0);
                #ifdef LOG_VERBOSE
                memory += iter.second->onGetMemoryInMB();
                #endif
            }
            #ifdef LOG_VERBOSE
            FUNC_PRINT_ALL(memory, f);
            #endif
        }
        mNeedMalloc = false;
        mNeedResize = false;
    }

#ifdef LOG_VERBOSE
    MNN_PRINT("session after resize\n");
    for (auto& iter : mInfo.outputTensor) {
        auto& outputTensor = iter.second;
        MNN_PRINT("output name:%s, ptr:%p,shape:", iter.first.c_str(), outputTensor);
        outputTensor->printShape();
        MNN_PRINT("\n");
    }
#endif
    return NO_ERROR;
}
void Session::openResizeCheck() {
    for (auto& iter : mPipelines) {
        iter->openResizeCheck();
    }
}

ErrorCode Session::fixResizeCache() {
    for (auto& iter : mPipelines) {
        auto code = iter->fixResizeCache();
        if (NO_ERROR != code) {
            return code;
        }
    }
    return NO_ERROR;
}

bool Session::getInfo(Interpreter::SessionInfoCode code, void* ptr) const {
    switch (code) {
        case Interpreter::MEMORY: {
            auto dst     = (float*)ptr;
            float summer = mRuntime.second->onGetMemoryInMB();
            for (auto& r : mRuntime.first) {
                if (r.second.get() != mRuntime.second.get()) {
                    summer += r.second->onGetMemoryInMB();
                }
            }
            *dst = summer;
            return true;
        } break;
        case Interpreter::BACKENDS: {
            int pos = 0;
            auto res = (int32_t*)ptr;
            for (auto& r : mPipelines) {
                auto type = r->getMainForwardType();
                res[pos++] = type;
            }
            return true;
        } break;
        case Interpreter::FLOPS: {
            float flo = 0.0f;
            for (auto& iter : mPipelines) {
                flo += iter->flops();
            }
            auto dst     = (float*)ptr;
            *dst = flo;
            return true;
        } break;
        case Interpreter::RESIZE_STATUS: {
            auto dst = (int*)ptr;
            if (mNeedResize) {
                *dst = 2;
            } else if (mNeedMalloc) {
                *dst = 1;
            } else {
                *dst = 0;
            }
            return true;
        } break;
        case Interpreter::THREAD_NUMBER: {
            auto dst = (int*)ptr;
            if (mPipelines.empty()) {
                break;
            }
            *dst = mPipelines[0]->getPipelineInfo().first.info.numThread;
            return true;
        }
        // TODO: Support other debug info
        default:
            break;
    }
    return false;
}

const Backend* Session::getBackEnd(const Tensor* tensor) const {
    return TensorUtils::getDescribeOrigin(tensor)->getBackend();
}

Tensor* Session::getInput(const char* name) const {
    //MNN_ASSERT(!mInputs.empty());
    if (nullptr == name) {
        return mInfo.inputTensors.begin()->second;
    }
    auto iter = mInfo.inputTensors.find(name);
    if (iter == mInfo.inputTensors.end()) {
        MNN_PRINT("Error: can't find input: %s\n", name);
        return nullptr;
    }
    return iter->second;
}
Tensor* Session::getTensor(int index) const {
    return mInfo.allTensors[index].get();
}

Tensor* Session::getOutput(const char* name) const {
    MNN_ASSERT(!mInfo.outputTensor.empty());
    if (nullptr == name) {
        return mInfo.outputTensor.begin()->second;
    }

    auto iter = mInfo.outputTensor.find(name);
    if (iter == mInfo.outputTensor.end()) {
        MNN_PRINT("Error: can't find output: %s\n", name);
        return nullptr;
    }
    return iter->second;
}

const std::map<std::string, Tensor*>& Session::getInputAll() const {
    return mInfo.inputTensors;
}

const std::map<std::string, Tensor*>& Session::getOutputAll() const {
    return mInfo.outputTensor;
}

ErrorCode Session::updateToModel(Net* net) const {
    if (mNeedResize) {
        return NOT_SUPPORT;
    }
    int opSize = net->oplists()->size();
    for (int i = 0; i < opSize; ++i) {
        auto op = net->oplists()->GetAs<Op>(i);
        if (op->type() != OpType_Const && op->type() != OpType_TrainableParam) {
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
        std::shared_ptr<Tensor> tensor = mInfo.allTensors[index];
        if (WrapExecution::needWrap(tensor.get(), nullptr)) {
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

static void initTensors(std::vector<std::shared_ptr<Tensor>>& tensors, const std::vector<std::shared_ptr<Tensor>>& tensorSrc) {
    for (int i=0; i<tensors.size(); ++i) {
        // Init all tensor except for const
        if (tensors[i].get() == nullptr) {
            tensors[i].reset(new Tensor);
            TensorUtils::getDescribe(tensors[i].get())->index = i;
        }
    }
    for (int i = 0; i < tensors.size(); ++i) {
        auto srcDes = TensorUtils::getDescribe(tensorSrc[i].get());
        if (srcDes->quantAttr != nullptr) {
            TensorUtils::getDescribe(tensors[i].get())->quantAttr.reset(new QuantAttr);
            *TensorUtils::getDescribe(tensors[i].get())->quantAttr = *srcDes->quantAttr;
        }
        if (TensorUtils::getDescribe(tensors[i].get())->usage != Tensor::InsideDescribe::CONSTANT) {
            TensorUtils::copyShape(tensorSrc[i].get(), tensors[i].get(), true);
        }
    }
}

Session* Session::clone(RuntimeInfo&& runtime, std::shared_ptr<Schedule::ScheduleInfo> sharedConst) {
    // TODO: Currently only valid for Module api's onClone
    Schedule::ScheduleInfo scheduleInfo;
    scheduleInfo.defaultBackend = mInfo.defaultBackend;
    scheduleInfo.pipelineInfo.resize(1);
    scheduleInfo.externalWeightPath = mInfo.externalWeightPath;
    Session::ModeGroup modes;
    scheduleInfo.defaultBackend = sharedConst->defaultBackend;
    scheduleInfo.constReplaceBackend = sharedConst->constReplaceBackend;
    scheduleInfo.allTensors = sharedConst->allTensors;
    initTensors(scheduleInfo.allTensors, mInfo.allTensors);
    MNN_ASSERT(1 == mPipelines.size());
    auto& srcPipelineInfo = mPipelines[0]->getPipelineInfo();
    auto& opCaches = srcPipelineInfo.second;
    auto& pipelineInfo = scheduleInfo.pipelineInfo[0];
    pipelineInfo.first.info = srcPipelineInfo.first.info;
    pipelineInfo.first.config = srcPipelineInfo.first.config;
    pipelineInfo.first.info.user = &pipelineInfo.first.config;
    auto& oplists = pipelineInfo.second;
    oplists.resize(opCaches.size());
    _createPipelineBackend(pipelineInfo, runtime);
    auto first = pipelineInfo.first.cache.first;
    auto second = pipelineInfo.first.cache.second;
    for (int i=0; i<opCaches.size(); ++i) {
        auto& srcOpInfo = opCaches[i];
        auto& opInfo = oplists[i];
        opInfo.op = opCaches[i].op;
        opInfo.type = srcOpInfo.type;
        opInfo.computeCache.copyImmutable(srcOpInfo.computeCache);
        auto op = opInfo.op;
        if (nullptr != op->outputIndexes()) {
            auto data = op->outputIndexes()->data();
            for (int j = 0; j < op->outputIndexes()->size(); ++j) {
                opInfo.outputs.push_back(scheduleInfo.allTensors[data[j]].get());
            }
        }
        if (nullptr != op->inputIndexes()) {
            auto data = op->inputIndexes()->data();
            for (int j = 0; j < op->inputIndexes()->size(); ++j) {
                opInfo.inputs.push_back(scheduleInfo.allTensors[data[j]].get());
            }
        }
        for (int j=0; j<opInfo.inputs.size(); ++j) {
            if (TensorUtils::getDescribe(opInfo.inputs[j])->usage != Tensor::InsideDescribe::CONSTANT) {
                TensorUtils::getDescribe(opInfo.inputs[j])->usage = TensorUtils::getDescribe(srcOpInfo.inputs[j])->usage;
            }
        }
        for (int j=0; j<opInfo.outputs.size(); ++j) {
            TensorUtils::getDescribe(opInfo.outputs[j])->usage = TensorUtils::getDescribe(srcOpInfo.outputs[j])->usage;
        }
        // Clone cache
        for (auto& iter : srcOpInfo.executionCache) {
            Execution* copyExecution = nullptr;
            bool valid = false;
            if (first->type() == iter.second->backend()->type()) {
                valid = iter.second->onClone(first.get(), iter.first, &copyExecution);
            } else {
                valid = iter.second->onClone(second.get(), iter.first, &copyExecution);
            }
            if (valid) {
                std::shared_ptr<Execution> copyExeWrap(copyExecution);
                opInfo.executionCache.insert(std::make_pair(iter.first, copyExeWrap));
            }
        }
    }
    auto dst = new Session(std::move(scheduleInfo), mMode, std::move(runtime));
    return dst;
}


} // namespace MNN
