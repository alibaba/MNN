//
//  StaticModule.cpp
//  MNN
//
//  Created by MNN on b'2020/09/10'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "StaticModule.hpp"
#include <MNN/AutoTime.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "Utils.hpp"
#include "core/WrapExecution.hpp"
#include "core/MNNMemoryUtils.h"
#include "RuntimeAttr.hpp"
#include "core/TensorUtils.hpp"
#include "core/FileLoader.hpp"
#include "core/OpCommonUtils.hpp"

namespace MNN {
namespace Express {

static std::vector<std::shared_ptr<BufferStorage>> preRearrangeWeights( // NOLINT
    Schedule::ScheduleInfo& scheduleInfo, Backend* backend, Backend* backupBackend) {
    FileLoader loader(scheduleInfo.externalWeightPath.c_str());
    auto&& pipelineInfo = scheduleInfo.pipelineInfo[0].second;
    std::vector<std::shared_ptr<BufferStorage>> splitOps(pipelineInfo.size());
    for (int i = 0; i < pipelineInfo.size(); ++i) {
        auto& info = pipelineInfo[i];
        auto op       = pipelineInfo[i].op;
        std::unique_ptr<OpT> op_table(op->UnPack());
        std::shared_ptr<Execution> exe;
        switch (op->type()) {
            case MNN::OpType_DepthwiseConvInt8:
            case MNN::OpType_ConvInt8:
            case MNN::OpType_ConvolutionDepthwise:
            case MNN::OpType_Convolution: {
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
                    if (op->main_as_Convolution2D()->quanParameter()) {
                        type = DataType_DT_INT8;
                        int inputIdx = op->inputIndexes()->Get(0);
                        auto& inputQuantAttr = TensorUtils::getDescribe(tempInput)->quantAttr;
                        if (nullptr != inputQuantAttr.get()) {
                            TensorUtils::getDescribe(tempInput)->type = DataType_DT_INT8;
                        }
                        auto& outputQuantAttr = TensorUtils::getDescribe(tempOutput)->quantAttr;
                        if (nullptr != outputQuantAttr.get()) {
                            TensorUtils::getDescribe(tempOutput)->type = DataType_DT_INT8;
                        }
                    }
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
            case MNN::OpType_Attention: {
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
static void _resizeTensor(Tensor* tensor, const Tensor* dims, Session* session, Schedule::TENSORCACHE* cacheTensor) {
    MNN_ASSERT(nullptr != tensor);
    bool dirty = _reshapeTensor(tensor, dims);

    if (!dirty) {
        return;
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
    session->setNeedResize();
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
        mPrevInputTensor[i].second = nullptr;
    }
    mOutputTensors.resize(mResource->mOutputFromTensor.size());
    for (int i = 0; i < mResource->mOutputFromTensor.size(); ++i) {
        mOutputTensors[i] = mSession->getTensor(mResource->mOutputs[mResource->mOutputFromTensor[i]]);
        auto des = TensorUtils::getDescribe(mOutputTensors[i]);
        if (des->usage == Tensor::InsideDescribe::NORMAL) {
            des->usage = Tensor::InsideDescribe::OUTPUT;
        }
    }
}

StaticModule::StaticModule(std::vector<int> inputs,
                           std::vector<int> outputs,
                           std::vector<std::shared_ptr<BufferStorage>>&& buffer,
                           Schedule::ScheduleInfo&& scheduleInfo,
                           std::shared_ptr<Schedule::ScheduleInfo> sharedConst,
                           Session::ModeGroup&& mode,
                           RuntimeInfo&& rt,
                           const Module::Config& config
                           ) {
    setType("StaticModule");
    mResource.reset(new Resource);
    mResource->mSharedConst = sharedConst;
    mResource->mModes = std::move(mode);
    mResource->mBnInfo.user = &mResource->mBnConfig;
    mResource->mModes.inputMode = config.shapeMutable ? Interpreter::Session_Input_User : Interpreter::Session_Input_Inside;
    mResource->mModes.outputMode = Interpreter::Session_Output_User;
    std::shared_ptr<BufferStorage> net_storage;
    std::map<const Op*, std::pair<std::shared_ptr<Execution>, DataType>> exeCache;
    MNN_ASSERT(1 == scheduleInfo.pipelineInfo.size());
    auto& bnCache = scheduleInfo.pipelineInfo[0].first;
    bnCache.cache.first.reset(rt.first[bnCache.info.type]->onCreate(bnCache.info.user));
    if (bnCache.cache.first->type() == MNN_FORWARD_CPU) {
        bnCache.cache.second = bnCache.cache.first;
    } else {
        // Use Multi-thread if user has set numberthread > 1
        BackendConfig defaultConfig;
        defaultConfig.flags = 4;
        auto cpurt = rt.first.find(MNN_FORWARD_CPU);
        if (cpurt != rt.first.end()) {
            bnCache.cache.second.reset(cpurt->second->onCreate(&defaultConfig));
        } else {
            bnCache.cache.second.reset(rt.second->onCreate(&defaultConfig));
        }
    }
    if (config.rearrange) {
        mResource->mBuffer = preRearrangeWeights(scheduleInfo, bnCache.cache.first.get(), bnCache.cache.second.get());
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

    bool needResize = scheduleInfo.validForResize && mResource->mModes.inputMode == Interpreter::Session_Input_Inside;
    mSession.reset(new Session(std::move(scheduleInfo), mResource->mModes, std::move(rt)));
    resetInputOutputs();
    if (needResize) {
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
            mPrevInputTensor[i].second = nullptr;
        }
        for (auto& iter : mSession->getPipelineInfo(0).first.inputTensorCopyCache) {
            std::get<3>(iter.second) = true;
        }
    }
}

std::vector<Express::VARP> StaticModule::onForward(const std::vector<Express::VARP>& inputs) {

    AUTOTIME;
    std::vector<Express::VARP> outputs(mResource->mOutputNumbers);
    for (auto& iter : mResource->mOutputFromInput) {
        outputs[iter.first] = inputs[iter.second];
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

    MNN_ASSERT(inputs.size() == mInputTensors.size());
    auto& pipelineInfo = mSession->getPipelineInfo(0);
    if (mResource->mModes.inputMode == Interpreter::Session_Input_User) {
        pipelineInfo.first.inputBackendChange = false;
        for (int i = 0; i < inputs.size(); ++i) {
            if (nullptr == mInputTensors[i]) {
                continue;
            }
            auto inputTensor = Utils::getTensor(inputs[i]);
            Schedule::TENSORCACHE* cacheTensor = nullptr;
            if (mPrevInputTensor[i].first != inputTensor) {
                auto newBackend = TensorUtils::getDescribeOrigin(inputTensor)->getBackend();
                if (mPrevInputTensor[i].second != newBackend) {
                    pipelineInfo.first.inputBackendChange = true;
                }
                auto cacheIter = pipelineInfo.first.inputTensorCopyCache.find(mInputTensors[i]);
                cacheTensor = &cacheIter->second;
                MNN_ASSERT(cacheIter != pipelineInfo.first.inputTensorCopyCache.end());
                std::get<3>(cacheIter->second) = true;
                mPrevInputTensor[i] = std::make_pair(inputTensor, newBackend);
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
            des->type = srcDes->type;
            des->dimensionFormat = srcDes->dimensionFormat;
            des->tensorArrayAttr = srcDes->tensorArrayAttr;
            mInputTensors[i]->buffer().type = inputTensor->buffer().type;
            _resizeTensor(mInputTensors[i], inputTensor, mSession.get(), cacheTensor);
            if (needMalloc) {
                mSession->setNeedMalloc();
            }
        }
        if (mResource->mUseContentInputs) {
            mSession->setNeedResize();
        }
        auto code = mSession->resize();
        if (NO_ERROR != code) {
            FUNC_PRINT(code);
            return {};
        }
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
            _resizeTensor(mInputTensors[i], inputTensor, mSession.get(), nullptr);
        }
        mSession->resize();
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


#ifdef LOG_VERBOSE
    for (auto& inputTensor : mInputTensors) {
        MNN_PRINT("static module, before run, input ptr:%p, hostPtr:%p,  shape:", inputTensor, inputTensor->host<void>());
        inputTensor->printShape();
        MNN_PRINT("\n");
        auto shape = inputTensor->shape();
    }
    MNN_PRINT("staticmodule before run\n");
#endif


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
    if (NO_ERROR != code) {
        return {};
    }
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
    if (mResource->mOutputFromTensor.empty()) {
        return this->cloneBaseTo(ctx, module);
    }
    // TODO: If RuntimeManager is not the same as Runtime, may copy error
    auto rt = Executor::getRuntime();
    module->mSession.reset(mSession->clone(std::move(rt), mResource->mSharedConst));
    module->resetInputOutputs();
    return this->cloneBaseTo(ctx, module);
}
int StaticModule::onOptimize(Interpreter::SessionMode stage) {
    switch (stage) {
        case MNN::Interpreter::Session_Resize_Check:
            mSession->openResizeCheck();
            break;
        case MNN::Interpreter::Session_Resize_Fix:
            mSession->fixResizeCache();
            break;
        default:
            break;
    }
    return 0;
}

} // namespace Express
} // namespace MNN
