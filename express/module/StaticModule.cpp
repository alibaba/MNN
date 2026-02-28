//
//  StaticModule.cpp
//  MNN
//
//  Created by MNN on 2020/09/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "StaticModule.hpp"
#include <MNN/AutoTime.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "Utils.hpp"
#include "core/WrapExecution.hpp"
#include "core/MNNMemoryUtils.h"
#include "ModuleInside.hpp"
#include "RuntimeAttr.hpp"
#include "core/TensorUtils.hpp"
#include "core/FileLoader.hpp"
#include "core/OpCommonUtils.hpp"

namespace MNN {
namespace Express {

static const StaticModule* getStaticModule(const Module* m) {
    if (m->type() == "StaticModule") {
        return static_cast<const StaticModule*>(m);
    }
    if (m->getChildren().empty()) {
        return nullptr;
    }
    return getStaticModule(m->getChildren()[0].get());
}

static std::vector<std::shared_ptr<BufferStorage>> preRearrangeWeights( // NOLINT
                                                                       Schedule::ScheduleInfo& scheduleInfo, Backend* firstbackend, Backend* backupBackend, const Module* base = nullptr) {
    std::map<const std::string, std::shared_ptr<Execution>> base_executions;
    if (base != nullptr) {
        // has base module
        auto static_module = getStaticModule(base);
        if (static_module) {
            auto session = static_module->getSession();
            std::vector<Schedule::OpCacheInfo> op_caches = session->getPipelineInfo(0).second;
            for (auto& op_cache : op_caches) {
                const auto& exe_cache = op_cache.executionCache;
                for (const auto& exe_item : exe_cache) {
                    if (exe_item.first->name()) {
                        base_executions.insert(std::make_pair(exe_item.first->name()->str(), exe_item.second));
                    }
                }
            }
        }
    }
    FileLoader loader(scheduleInfo.externalWeightPath.c_str());
    auto&& pipelineInfo = scheduleInfo.pipelineInfo[0].second;
    std::vector<std::shared_ptr<BufferStorage>> splitOps(pipelineInfo.size());
    for (int i = 0; i < pipelineInfo.size(); ++i) {
        auto& info = pipelineInfo[i];
        auto op    = pipelineInfo[i].op;
        std::unique_ptr<OpT> op_table(op->UnPack());
        std::shared_ptr<Execution> exe;
        Backend* backend = firstbackend;
        if (info.type == Schedule::CONSTANT) {
            backend = backupBackend;
        }
        switch (op->type()) {
            case MNN::OpType_DepthwiseConvInt8:
            case MNN::OpType_ConvInt8:
            case MNN::OpType_ConvolutionDepthwise:
            case MNN::OpType_Convolution: {
                if (!base_executions.empty() && op->name()) {
                    auto iter = base_executions.find(op->name()->str());
                    if (iter != base_executions.end()) {
                        auto base_exe = iter->second.get();
                        Execution* copyExecution = nullptr;
                        base_exe->onClone(backend, op, &copyExecution);
                        if (copyExecution == nullptr) {
                            base_exe->onClone(backupBackend, op, &copyExecution);
                        }
                        if (copyExecution != nullptr && copyExecution->onClone(nullptr, op, nullptr)) {
                            exe.reset(copyExecution);
                        }
                    }
                }
                if (exe == nullptr) {
                    DataType type = DataType_DT_FLOAT;
                    auto conv2d = op->main_as_Convolution2D();
                    // Create Default Inputs and Outputs
                    auto tempInput = info.inputs[0];
                    auto tempOutput = info.outputs[0];
                    auto common = conv2d->common();
                    if (scheduleInfo.pipelineInfo[0].first.needComputeGeometry) {
                        // Set default shape to create execution
                        int ow = 2, oh = 2;
                        int iw = (common->kernelX() - 1) * common->dilateX() + common->strideX() * (ow - 1) + 1;
                        int ih = (common->kernelY() - 1) * common->dilateY() + common->strideY() * (oh - 1) + 1;
                        TensorUtils::getDescribe(tempInput)->dimensionFormat = MNN_DATA_FORMAT_NC4HW4;;
                        tempInput->setLength(0, 1);
                        tempInput->setLength(1, conv2d->common()->inputCount());
                        tempInput->setLength(2, ih);
                        tempInput->setLength(3, iw);
                        TensorUtils::getDescribe(tempOutput)->dimensionFormat = MNN_DATA_FORMAT_NC4HW4;;
                        tempOutput->setLength(0, 1);
                        tempOutput->setLength(1, conv2d->common()->outputCount());
                        tempOutput->setLength(2, oh);
                        tempOutput->setLength(3, ow);
                    }
                    std::shared_ptr<BufferStorage> tmpstorage;
                    exe.reset(OpCommonUtils::createExecutionWithExternal(backend, info.inputs, info.outputs, op, &loader, tmpstorage));
                    if (exe.get() == nullptr) {
                        exe.reset(OpCommonUtils::createExecutionWithExternal(backupBackend, info.inputs, info.outputs, op, &loader, tmpstorage));
                    }
                    if (nullptr == exe) {
                        break;
                    }
                    // The exe can't clone
                    if (!exe->onClone(nullptr, op, nullptr)) {
                        exe = nullptr;
                        break;
                    }
                }
                if (OpParameter_Convolution2D == op_table->main.type) {
                    op_table->main.AsConvolution2D()->bias.clear();
                    op_table->main.AsConvolution2D()->weight.clear();
                    if (nullptr != op_table->main.AsConvolution2D()->symmetricQuan) {
                        op_table->main.AsConvolution2D()->symmetricQuan->bias.clear();
                        op_table->main.AsConvolution2D()->symmetricQuan->weight.clear();
                    }
                    if (nullptr != op_table->main.AsConvolution2D()->quanParameter) {
                        op_table->main.AsConvolution2D()->quanParameter->alpha.clear();
                        op_table->main.AsConvolution2D()->quanParameter->buffer.clear();
                    }
                }
                break;
            }
            case MNN::OpType_Attention:
            case MNN::OpType_LinearAttention:
            {
                exe.reset(backend->onCreate({}, {}, op));
                if (exe.get() == nullptr) {
                    exe.reset(backupBackend->onCreate({}, {}, op));
                }
                if (nullptr == exe) {
                    break;
                }
                // The exe can't clone
                if (!exe->onClone(nullptr, op, nullptr)) {
                    exe = nullptr;
                    break;
                }
                break;
            }
            case MNN::OpType_LayerNorm: {
                if (!base_executions.empty() && op->name()) {
                    auto iter = base_executions.find(op->name()->str());
                    if (iter != base_executions.end()) {
                        auto base_exe = iter->second.get();
                        Execution* copyExecution = nullptr;
                        base_exe->onClone(backend, op, &copyExecution);
                        if (copyExecution == nullptr) {
                            base_exe->onClone(backupBackend, op, &copyExecution);
                        }
                        if (copyExecution != nullptr && copyExecution->onClone(nullptr, op, nullptr)) {
                            exe.reset(copyExecution);
                        }
                    }
                }
                if (exe == nullptr) {
                    std::shared_ptr<BufferStorage> tmpstorage;
                    exe.reset(OpCommonUtils::createExecutionWithExternal(backend, info.inputs, info.outputs, op, &loader, tmpstorage));
                    if (exe.get() == nullptr) {
                        exe.reset(OpCommonUtils::createExecutionWithExternal(backupBackend, info.inputs, info.outputs, op, &loader, tmpstorage));
                    }
                    if (nullptr == exe) {
                        break;
                    }
                }
                // The exe can't clone
                if (!exe->onClone(nullptr, op, nullptr)) {
                    exe = nullptr;
                    break;
                }
                break;
            }
            default: {
                break;
            }
        }
        flatbuffers::FlatBufferBuilder opBuilder;
        opBuilder.Finish(Op::Pack(opBuilder, op_table.get()));
        std::shared_ptr<BufferStorage> buf(new BufferStorage);
        buf->storage = opBuilder.ReleaseRaw(buf->allocated_size, buf->offset);
        info.op = flatbuffers::GetRoot<Op>(buf->buffer());
        if (nullptr != exe) {
            // Clone Execution to reset op info
            Execution* dstExe;
            exe->onClone(exe->backend(), info.op, &dstExe);
            std::shared_ptr<Execution> dstExeP(dstExe);
            info.executionCache.insert(std::make_pair(info.op, dstExeP));
        }
        splitOps[i] = buf;
    }
    return splitOps;
}

static bool _reshapeTensor(Tensor* tensor, const Tensor* dims) {
    bool dirty = false;
    if (tensor->buffer().dimensions != dims->dimensions()) {
        dirty = true;
    } else {
        for (int i = 0; i < dims->dimensions(); ++i) {
            if (tensor->buffer().dim[i].extent != dims->length(i)) {
                dirty = true;
                break;
            }
        }
    }
    return dirty;
}
static bool _resizeTensor(Tensor* tensor, const Tensor* dims, Session* session, Schedule::TENSORCACHE* cacheTensor) {
    MNN_ASSERT(nullptr != tensor);
    bool dirty = _reshapeTensor(tensor, dims);

    if (!dirty) {
        return false;
    }

    tensor->buffer().dimensions = (int)dims->dimensions();
    for (int i = 0; i < dims->dimensions(); ++i) {
        tensor->buffer().dim[i].extent = dims->length(i);
        tensor->buffer().dim[i].stride = dims->stride(i);
    }
    if (nullptr != cacheTensor) {
        auto t = std::get<1>(*cacheTensor).get();
        if (nullptr != t) {
            t->buffer().dimensions = (int)dims->dimensions();
            for (int i = 0; i < dims->dimensions(); ++i) {
                t->buffer().dim[i].extent = dims->length(i);
                t->buffer().dim[i].stride = dims->stride(i);
            }
            std::get<2>(*cacheTensor) = true;
        }
    }
    return true;
}
void StaticModule::resetInputOutputs() {
    mPrevInputTensor.resize(mResource->mInputs.size());
    mInputTensors.resize(mResource->mInputs.size());
    auto& pipelineInfo = mSession->getPipelineInfo(0);
    for (int i = 0; i < mResource->mInputs.size(); ++i) {
        mInputTensors[i] = mSession->getTensor(mResource->mInputs[i]);
        auto des = TensorUtils::getDescribe(mInputTensors[i]);
        if (des->usage != Tensor::InsideDescribe::CONSTANT && des->usage != Tensor::InsideDescribe::TRAINABLE) {
            des->usage = Tensor::InsideDescribe::INPUT;
        }
        pipelineInfo.first.inputTensorCopyCache.insert(std::make_pair(mInputTensors[i], std::make_tuple(nullptr, nullptr, true, true)));
        mPrevInputTensor[i].first = nullptr;
        mPrevInputTensor[i].second = MNN_FORWARD_CPU;
    }
    mOutputTensors.resize(mResource->mOutputFromTensor.size());
    for (int i = 0; i < mResource->mOutputFromTensor.size(); ++i) {
        mOutputTensors[i] = mSession->getTensor(mResource->mOutputs[mResource->mOutputFromTensor[i]]);
        auto des = TensorUtils::getDescribe(mOutputTensors[i]);
        if (des->usage == Tensor::InsideDescribe::NORMAL) {
            des->usage = Tensor::InsideDescribe::OUTPUT;
        }
    }
    // Mask Geometry Compute Mid Tensor release able indexes
    auto& infos = pipelineInfo;
    for (auto& info : infos.second) {
        info.releaseAbleInputs.clear();
        if (info.type != Schedule::Type::CONSTANT) {
            continue;
        }
        for (auto t : info.inputs) {
            auto des = TensorUtils::getDescribe(t);
            if (des->usage == Tensor::InsideDescribe::CONSTANT && des->isMutable) {
                des->useCount = 0;
            }
        }
    }
    for (auto& info : infos.second) {
        for (auto t : info.inputs) {
            auto des = TensorUtils::getDescribe(t);
            if (des->usage == Tensor::InsideDescribe::CONSTANT && des->isMutable) {
                des->useCount++;
            }
        }
    }
    for (int i = 0; i < mResource->mOutputFromTensor.size(); ++i) {
        mOutputTensors[i] = mSession->getTensor(mResource->mOutputs[mResource->mOutputFromTensor[i]]);
        auto des = TensorUtils::getDescribe(mOutputTensors[i]);
        if (des->usage == Tensor::InsideDescribe::CONSTANT && des->isMutable) {
            des->useCount ++;
        }
    }
    for (auto& info : infos.second) {
        if (info.type != Schedule::Type::CONSTANT) {
            continue;
        }
        for (int v=0; v<info.inputs.size(); ++v) {
            auto des = TensorUtils::getDescribe(info.inputs[v]);
            if (des->usage == Tensor::InsideDescribe::CONSTANT && des->isMutable) {
                des->useCount--;
                if (des->useCount == 0) {
                    info.releaseAbleInputs.emplace_back(v);
                }
            }
        }
    }
}

StaticModule::StaticModule(std::vector<int> inputs,
                           std::vector<int> outputs,
                           std::vector<std::shared_ptr<BufferStorage>>&& buffer,
                           Schedule::ScheduleInfo&& scheduleInfo,
                           std::shared_ptr<Schedule::ScheduleInfo> sharedConst,
                           Session::ModeGroup&& mode,
                           std::shared_ptr<Executor::RuntimeManager> rtm,
                           const Module::Config& config
                           ) {
    setType("StaticModule");
    mResource.reset(new Resource);
    mRuntimeManager = rtm;
    MNN_ASSERT(nullptr != rtm);
    auto rt = rtm->getInside()->mRuntime;
    mResource->mSharedConst = sharedConst;
    mResource->mModes = std::move(mode);
    mResource->mBnInfo.user = &mResource->mBnConfig;
    mResource->mModes.inputMode = config.shapeMutable ? Interpreter::Session_Input_User : Interpreter::Session_Input_Inside;
    mResource->mModes.outputMode = Interpreter::Session_Output_User;
    std::shared_ptr<BufferStorage> net_storage;
    std::map<const Op*, std::pair<std::shared_ptr<Execution>, DataType>> exeCache;
    MNN_ASSERT(1 == scheduleInfo.pipelineInfo.size());
    auto& bnCache = scheduleInfo.pipelineInfo[0].first;
    // Create Backend for prearrange
    Session::createPipelineBackend(scheduleInfo.pipelineInfo[0], rt);
    if (nullptr == bnCache.cache.first || nullptr == bnCache.cache.second) {
        MNN_ERROR("[MNN:Express] Create Backend Error\n");
        return;
    }
    bnCache.cache.first->pNPUModelDirPath = rtm->getInside()->mContent->mNpuDir;
    bnCache.cache.second->pNPUModelDirPath = rtm->getInside()->mContent->mNpuDir;
    if (config.rearrange) {
        mResource->mBuffer = preRearrangeWeights(scheduleInfo, bnCache.cache.first.get(), bnCache.cache.second.get(), config.base);
    } else {
        mResource->mBuffer = std::move(buffer);
    }
    mResource->mOutputNumbers = (int)outputs.size();
    /** Compute:
     std::vector<int, int> mOutputFromTensor;
     std::vector<int, int> mOutputFromInput;
     */
    for (int i = 0; i < outputs.size(); ++i) {
        auto& t        = outputs[i];
        bool fromInput = false;
        for (int j = 0; j < inputs.size(); ++j) {
            if (inputs[j] == t) {
                fromInput = true;
                mResource->mOutputFromInput.emplace_back(std::make_pair(i, j));
                break;
            }
        }
        if (fromInput) {
            continue;
        }
        mResource->mOutputFromTensor.emplace_back(i);
    }
    if (mResource->mOutputFromTensor.empty()) {
        return;
    }
    mResource->mUseContentInputs = scheduleInfo.needInputContentForShape;
    if (mResource->mUseContentInputs) {
        mResource->mModes.inputMode = Interpreter::Session_Input_User;
    }
    mResource->mInputs = std::move(inputs);
    mResource->mInputNeedCPU.resize(mResource->mInputs.size());
    for (int i=0; i<mResource->mInputs.size(); ++i) {
        mResource->mInputNeedCPU[i] = false;
    }
    if (mResource->mUseContentInputs) {
        for (int i=0; i<mResource->mInputs.size(); ++i) {
            auto subT = scheduleInfo.allTensors[mResource->mInputs[i]].get();
            if (TensorUtils::getDescribe(subT)->usage == Tensor::InsideDescribe::CONSTANT) {
                mResource->mInputNeedCPU[i] = true;
            }
        }
    }
    mResource->mOutputs = std::move(outputs);

    bool canResize = scheduleInfo.validForResize && mResource->mModes.inputMode == Interpreter::Session_Input_Inside;
    mSession.reset(new Session(std::move(scheduleInfo), mResource->mModes, std::move(rt)));
    resetInputOutputs();
    if (canResize && (!config.rearrange)) {
        mSession->resize();
    }
}
StaticModule::~StaticModule() {
    mSession         = nullptr;
}
void StaticModule::onClearCache() {
    if (nullptr != mSession) {
        for (int i=0; i<mPrevInputTensor.size(); ++i) {
            mPrevInputTensor[i].first = nullptr;
        }
        for (auto& iter : mSession->getPipelineInfo(0).first.inputTensorCopyCache) {
            std::get<3>(iter.second) = true;
        }
    }
}
ErrorCode StaticModule::_resize(const std::vector<Express::VARP>& inputs) {
    ErrorCode code = NO_ERROR;
    auto& pipelineInfo = mSession->getPipelineInfo(0);
    auto rtmInside = mRuntimeManager->getInside();
    int curStatus = 0;
    if (mResource->mModes.inputMode == Interpreter::Session_Input_User) {
        pipelineInfo.first.inputBackendChange = false;
        bool needResize = mResource->mUseContentInputs;
        for (int i = 0; i < inputs.size(); ++i) {
            if (nullptr == mInputTensors[i]) {
                continue;
            }
            auto inputTensor = Utils::getTensor(inputs[i]);
            Schedule::TENSORCACHE* cacheTensor = nullptr;
            if (mPrevInputTensor[i].first != inputTensor) {
                auto newBackend = TensorUtils::getDescribeOrigin(inputTensor)->getBackend();
                auto newType = MNN_FORWARD_CPU;
                if (nullptr != newBackend) {
                    newType = newBackend->type();
                }
                if (mPrevInputTensor[i].second != newType) {
                    pipelineInfo.first.inputBackendChange = true;
                }
                auto cacheIter = pipelineInfo.first.inputTensorCopyCache.find(mInputTensors[i]);
                cacheTensor = &cacheIter->second;
                MNN_ASSERT(cacheIter != pipelineInfo.first.inputTensorCopyCache.end());
                std::get<3>(cacheIter->second) = true;
                mPrevInputTensor[i] = std::make_pair(inputTensor, newType);
                if (std::get<1>(*cacheTensor) != nullptr) {
                    if (!WrapExecution::needWrap(inputTensor,   TensorUtils::getDescribeOrigin(std::get<0>(*cacheTensor))->getBackend())) {
                        // No need copy now, reset it
                        cacheIter->second = std::make_tuple(nullptr, nullptr, true, true);
                    }
                }
            }
            auto srcDes = TensorUtils::getDescribe(inputTensor);
            auto des = TensorUtils::getDescribe(mInputTensors[i]);
            bool needCopy = false;
            if (nullptr != srcDes->quantAttr.get()) {
                if (nullptr == des->quantAttr.get()) {
                    needCopy = true;
                }
            }
            if (mResource->mInputNeedCPU[i]) {
                if (0 != inputTensor->buffer().device) {
                    needCopy = true;
                }
            }
            if (srcDes->tensorArrayAttr.get() != nullptr) {
                // For tensorArray, don't need content
                needCopy = false;
                mSession->setNeedResize();
            }
            bool needMalloc;
            if (needCopy) {
                auto srcPtr = (uint8_t*)inputs[i]->readMap<uint8_t>();
                needMalloc = mInputTensors[i]->buffer().host != srcPtr;
                mInputTensors[i]->buffer().host = srcPtr;
                mInputTensors[i]->buffer().device = 0;
                TensorUtils::getDescribeOrigin(mInputTensors[i])->setBackend(pipelineInfo.first.cache.second.get());
                if (nullptr == srcDes->quantAttr.get()) {
                    // For device need copy, cache device tensor
                    auto cacheIter = pipelineInfo.first.inputTensorCopyCache.find(mInputTensors[i]);
                    MNN_ASSERT(cacheIter != pipelineInfo.first.inputTensorCopyCache.end());
                    std::get<0>(cacheIter->second) = inputTensor;
                    std::get<1>(cacheIter->second) = nullptr;
                    std::get<2>(cacheIter->second) = false;
                    std::get<3>(cacheIter->second) = false;
                }
            } else {
                needMalloc = TensorUtils::refTensorContent(mInputTensors[i], inputTensor);
            }
            des->applyQuant = srcDes->applyQuant;
            des->dimensionFormat = srcDes->dimensionFormat;
            des->tensorArrayAttr = srcDes->tensorArrayAttr;
            mInputTensors[i]->buffer().type = inputTensor->buffer().type;
            if (_resizeTensor(mInputTensors[i], inputTensor, mSession.get(), cacheTensor)) {
                needResize = true;
            }
            if (needMalloc) {
                mSession->setNeedMalloc();
            }
        }
        if (needResize) {
            mSession->setNeedResize();
        }
        if (!needResize) {
            // Check if output is used by other vars. If used, must realloc output to avoid the content dirty for output vars
            // If resized, the output's memory will be all released in Session::resize, don't need clear here
            for (auto& output : mOutputTensors) {
                auto desOrigin = TensorUtils::getDescribeOrigin(output);
                if ((!desOrigin->mContent->isMutable) || nullptr == desOrigin->mem.get()) {
                    continue;
                }
                auto bn = desOrigin->getBackend();
                if (nullptr == bn) {
                    continue;
                }
                if (desOrigin->mContent.use_count() > 1 && desOrigin->mContent->usage != Tensor::InsideDescribe::CONSTANT) {
                    desOrigin->mem = nullptr;
                    auto res = bn->onAcquireBuffer(output, Backend::STATIC);
                    if (!res) {
                        return OUT_OF_MEMORY;
                    }
                    mSession->setNeedMalloc();
                }
            }
        }
        mSession->getInfo(Interpreter::RESIZE_STATUS, &curStatus);
        code = mSession->resize();
    } else {
        // Resize
        for (int i = 0; i < inputs.size(); ++i) {
            if (nullptr == mInputTensors[i]) {
                continue;
            }
            auto inputTensor = Utils::getTensor(inputs[i]);
            auto srcDes = TensorUtils::getDescribe(inputTensor);
            auto des = TensorUtils::getDescribe(mInputTensors[i]);
            des->dimensionFormat = srcDes->dimensionFormat;
            mInputTensors[i]->buffer().type = inputTensor->buffer().type;
            if (_resizeTensor(mInputTensors[i], inputTensor, mSession.get(), nullptr)) {
                mSession->setNeedResize();
            }
        }
        mSession->getInfo(Interpreter::RESIZE_STATUS, &curStatus);
        code = mSession->resize();
        // Copy
        for (int i = 0; i < inputs.size(); ++i) {
            if (nullptr == mInputTensors[i]) {
                continue;
            }
            auto exprInfo    = inputs[i]->expr();
            auto inputTensor = Utils::getTensor(inputs[i]);
            mInputTensors[i]->copyFromHostTensor(inputTensor);
        }
    }
    rtmInside->mResizeStatus = ALIMAX(rtmInside->mResizeStatus, curStatus);
    return code;
}

ErrorCode StaticModule::_execute() {
    ErrorCode code;
    if (mResource->mModes.callBackMode == Interpreter::Session_Debug) {
        auto globalExecutor = ExecutorScope::Current();
        auto debug = globalExecutor->getDebugTools();
        if (debug->after != nullptr && debug->before != nullptr) {
            code = mSession->runWithCallBack(debug->before, debug->after);
        } else {
            code = mSession->run();
        }
    } else {
        code = mSession->run();
    }
    return code;
}

std::vector<Express::VARP> StaticModule::onForward(const std::vector<Express::VARP>& inputs) {

    AUTOTIME;
    std::vector<Express::VARP> outputs;
    bool runResize = (!mShapeInferSeperate) || inputs.size() > 0;
    bool runCompute = (!mShapeInferSeperate) || inputs.size() == 0;
    if (runResize) {
        outputs.resize(mResource->mOutputNumbers);
        for (auto& iter : mResource->mOutputFromInput) {
            outputs[iter.first] = inputs[iter.second];
        }
    }
    if (mResource->mOutputFromTensor.empty()) {
        return outputs;
    }
    Variable::compute(inputs);
#ifdef MNN_DUMP_MEMORY
    auto rt = Executor::getRuntime();
    auto mem = rt.second->onGetMemoryInMB();
    for (auto iter : rt.first) {
        if (iter.second.get() != rt.second.get()) {
            mem += iter.second->onGetMemoryInMB();
        }
    }
    FUNC_PRINT_ALL(mem, f);
#endif

    ErrorCode code = NO_ERROR;
    if (runResize) {
        code = _resize(inputs);
    }
    if (NO_ERROR == code && runCompute) {
        code = _execute();
    }
    if (NO_ERROR != code) {
        FUNC_PRINT(code);
        return {};
    }
    if (!runResize) {
        for (auto& var : mOutputVars) {
            // Check if needed recopy
            auto inside = var->expr().first->inside();
            if (nullptr != inside->mHostTensor) {
                inside->mOutputTensors[0]->copyToHostTensor(inside->mHostTensor);
            }
        }
        return {};
    }
    auto& pipelineInfo = mSession->getPipelineInfo(0);
    for (int i = 0; i < mOutputTensors.size(); ++i) {
        auto tensor = Tensor::clone(mOutputTensors[i]);
        outputs[mResource->mOutputFromTensor[i]] = Express::Variable::create(Express::Expr::create(tensor, true));
        auto backend = TensorUtils::getDescribeOrigin(tensor)->getBackend();
        if (backend == pipelineInfo.first.cache.first.get()) {
            outputs[mResource->mOutputFromTensor[i]]->expr().first->inside()->mHoldBackend = pipelineInfo.first.cache.first;
        } else if (backend == pipelineInfo.first.cache.second.get()) {
            outputs[mResource->mOutputFromTensor[i]]->expr().first->inside()->mHoldBackend = pipelineInfo.first.cache.second;
        } else if (backend == mResource->mSharedConst->defaultBackend.get()) {
            outputs[mResource->mOutputFromTensor[i]]->expr().first->inside()->mHoldBackend = mResource->mSharedConst->defaultBackend;
        } else if (backend == mResource->mSharedConst->constReplaceBackend.get()) {
            outputs[mResource->mOutputFromTensor[i]]->expr().first->inside()->mHoldBackend = mResource->mSharedConst->constReplaceBackend;
        }
    }
    if (mShapeInferSeperate && runResize) {
        mOutputVars = outputs;
    }
#ifdef MNN_INTERNAL_ENABLED
    auto glo = ExecutorScope::Current();
    float flops = 0.0f;
    mSession->getInfo(Interpreter::FLOPS, &flops);
    glo->getDebugTools()->flops += flops;
#endif

    return outputs;
}

Module* StaticModule::clone(CloneContext* ctx) const {
    StaticModule* module(new StaticModule);
    module->mResource = mResource;
    module->mRuntimeManager = ctx->pRuntimeManager;
    if (mResource->mOutputFromTensor.empty()) {
        return this->cloneBaseTo(ctx, module);
    }
    auto rt = ctx->pRuntimeManager->getInside()->mRuntime;
    module->mSession.reset(mSession->clone(std::move(rt), mResource->mSharedConst));
    module->resetInputOutputs();
    return this->cloneBaseTo(ctx, module);
}
int StaticModule::onOptimize(Interpreter::SessionMode stage) {
    int res = 0;
    switch (stage) {
        case MNN::Interpreter::Session_Resize_Check:
            mSession->openResizeCheck();
            break;
        case MNN::Interpreter::Session_Resize_Fix:
            mSession->fixResizeCache();
            break;
        case MNN::Interpreter::Module_Forward_Separate:
            if (mResource->mUseContentInputs || mResource->mModes.inputMode != Interpreter::Session_Input_User || mResource->mOutputFromTensor.empty()) {
                res = NOT_SUPPORT;
                break;
            }
            mShapeInferSeperate = true;
            break;
        case MNN::Interpreter::Module_Forward_Combine:
            mOutputVars.clear();
            mShapeInferSeperate = false;
            break;
        default:
            break;
    }
    return res;
}

} // namespace Express
} // namespace MNN
