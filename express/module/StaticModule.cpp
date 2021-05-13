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
#include "core/MNNMemoryUtils.h"
#include "core/Session.hpp"
#include "core/TensorUtils.hpp"

namespace MNN {
namespace Express {

static std::shared_ptr<BufferStorage> preRearrangeWeights( // NOLINT
    const MNN::Net* net, std::map<const Op*, std::shared_ptr<Execution>>& cache, Backend* backend) {
    std::unique_ptr<MNN::NetT> net_table(net->UnPack());
    std::map<int, std::shared_ptr<Execution>> exeCache;
    bool isQuantModel = !net_table->extraTensorDescribe.empty();
    std::vector<TensorQuantInfoT*> quantInfos;
    std::vector<std::unique_ptr<Tensor>> inputTensors;
    if (isQuantModel) {
        quantInfos.resize(net_table->tensorName.size(), nullptr);
        for (auto& tensorDes : net_table->extraTensorDescribe) {
            quantInfos[tensorDes->index] = tensorDes->quantInfo.get();
        }
    }
    for (int i = 0; i < net->oplists()->size(); ++i) {
        auto op       = net->oplists()->Get(i);
        auto op_table = net_table->oplists[i].get();
        if (op->inputIndexes() == nullptr || op->inputIndexes()->size() != 1) {
            continue;
        }
        switch (op->type()) {
            case MNN::OpType_DepthwiseConvInt8:
            case MNN::OpType_ConvInt8:
            case MNN::OpType_ConvolutionDepthwise:
            case MNN::OpType_Convolution: {
                std::shared_ptr<Execution> exe;
                if (isQuantModel) {
                    int inputIdx = op->inputIndexes()->Get(0);
                    auto inputTensor = Tensor::create({1}, halide_type_of<float>());
                    inputTensors.emplace_back(inputTensor);
                    auto& inputQuantAttr = TensorUtils::getDescribe(inputTensor)->quantAttr;
                    if (quantInfos[inputIdx]) {
                        inputQuantAttr.reset(new QuantAttr);
                        inputQuantAttr->scale = quantInfos[inputIdx]->scale;
                        inputQuantAttr->min = quantInfos[inputIdx]->min;
                        inputQuantAttr->max = quantInfos[inputIdx]->max;
                        inputQuantAttr->zero = quantInfos[inputIdx]->zero;
                    } else {
                        inputQuantAttr.reset();
                    }
                    int outputIdx = op->inputIndexes()->Get(0);
                    auto outputTensor = Tensor::create({1}, halide_type_of<float>());
                    inputTensors.emplace_back(outputTensor);
                    auto& outputQuantAttr = TensorUtils::getDescribe(outputTensor)->quantAttr;
                    if (quantInfos[outputIdx]) {
                        outputQuantAttr.reset(new QuantAttr);
                        outputQuantAttr->scale = quantInfos[outputIdx]->scale;
                        outputQuantAttr->min = quantInfos[outputIdx]->min;
                        outputQuantAttr->max = quantInfos[outputIdx]->max;
                        outputQuantAttr->zero = quantInfos[outputIdx]->zero;
                    } else {
                        outputQuantAttr.reset();
                    }
                    if (inputQuantAttr && outputQuantAttr && op->main_as_Convolution2D()->quanParameter()) {
                        exe.reset(backend->onCreate({inputTensor}, {outputTensor}, op));
                    }
                } else {
                    exe.reset(backend->onCreate({}, {}, op));
                }
                if (nullptr == exe) {
                    break;
                }
                if (!exe->onClone(nullptr, op, nullptr)) {
                    break;
                }
                exeCache.insert(std::make_pair(i, exe));
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
            default: {
                break;
            }
        }
    }
    flatbuffers::FlatBufferBuilder builder(1024);
    builder.Finish(MNN::Net::Pack(builder, net_table.get()));
    // Swap the raw buffer ownership.
    std::shared_ptr<BufferStorage> net_storage(new BufferStorage);
    net_storage->storage.reset(builder.ReleaseRaw(net_storage->allocated_size, // NOLINT
                                                  net_storage->offset));
    net = GetNet(net_storage->buffer());
    for (auto& iter : exeCache) {
        auto op = net->oplists()->Get(iter.first);
        cache.insert(std::make_pair(op, iter.second));
    }
    return net_storage;
}

StaticModule::StaticModule(const void* buffer, size_t length, const std::vector<std::string>& inputs,
                           const std::vector<std::string>& outputs, const Module::Config& moduleconfig, bool copyOutput) {
    setType("StaticModule");
    mResource.reset(new Resource);
    mResource->mInputs = inputs;
    mResource->mOutputs = outputs;
    mResource->mCopyOutput = copyOutput;
    std::shared_ptr<BufferStorage> net_storage;
    std::map<const Op*, std::shared_ptr<Execution>> exeCache;
    if (moduleconfig.rearrange) {
        auto rt = Express::ExecutorScope::Current()->getRuntime();
        MNN_CHECK(rt.first.size() == 1, "The number of formal backends should be 1.");
        mResourceBackend.reset(rt.first.begin()->second->onCreate());
        net_storage = preRearrangeWeights(GetNet(buffer), exeCache, mResourceBackend.get());
        buffer      = net_storage->buffer();
        length      = net_storage->size();
    } else {
        net_storage.reset(new BufferStorage);
        net_storage->storage.reset((uint8_t*)malloc(length));
        if (nullptr == net_storage->storage.get()) {
            MNN_ERROR("Allock Error in StaticModule's net\n");
            return;
        }
        net_storage->allocated_size = length;
        net_storage->offset         = 0;
        ::memcpy(net_storage->storage.get(), buffer, length);
        buffer = net_storage->storage.get();
    }
    mResource->mNetStorage    = std::move(net_storage);
    mResource->mShapeFix      = !moduleconfig.shapeMutable;
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
    
    RuntimeInfo rt;
    if (moduleconfig.backend == nullptr) {
        rt = Express::ExecutorScope::Current()->getRuntime();
    } else {
        ScheduleConfig sche_config;
        sche_config.type = moduleconfig.backend->type;
        sche_config.backendConfig = moduleconfig.backend->config;
        rt = Interpreter::createRuntime(std::vector<ScheduleConfig>({sche_config}));
    }
    // TODO: Add Config
    mResource->mConfig.numThread   = 1;
    mResource->mConfig.type        = rt.first.begin()->first;
    mResource->mConfig.path.mode   = ScheduleConfig::Path::Mode::Tensor;
    mResource->mConfig.path.outputs = outputs;
    mResource->mConfig.saveTensors = outputs;
    mResource->mConfig.path.inputs = inputs;
    auto scheduleInfo  = Schedule::schedule(GetNet(buffer), {mResource->mConfig});
#ifdef MNN_EXPR_ENABLE_PROFILER
    Interpreter::SessionMode callBackMode = Interpreter::Session_Debug;
#else
    Interpreter::SessionMode callBackMode = Interpreter::Session_Release;
#endif
    auto isUsedContent = [&scheduleInfo](const Tensor* t) {
        const auto& infos = scheduleInfo.pipelineInfo[0].second;
        for (auto info : infos) {
            auto needInputs = SizeComputer::needInputContent(info.op, info.inputs.size());
            for (auto inputIdx : needInputs) {
                if (inputIdx < info.inputs.size() && info.inputs[inputIdx] == t) {
                    return true;
                }
            }
        }
        return false;
    };
    std::set<Tensor*> useContentInputs;
    for (const auto& iter : scheduleInfo.inputTensors) {
        if (isUsedContent(iter.second)) {
            useContentInputs.insert(iter.second);
        }
    }
    Interpreter::SessionMode inputMode =
    mResource->mShapeFix ? Interpreter::Session_Input_Inside : Interpreter::Session_Input_User;
    mSession.reset(new Session(std::move(scheduleInfo), callBackMode, inputMode, std::move(rt)));
    mSession->cloneExecution(exeCache, 0);
    if (scheduleInfo.validForResize && inputMode == Interpreter::Session_Input_Inside) {
        mSession->resize(false);
    }
    mInputTensors.resize(inputs.size());
    for (int i = 0; i < inputs.size(); ++i) {
        mInputTensors[i] = mSession->getInput(inputs[i].c_str());
        if (useContentInputs.find(mInputTensors[i]) != useContentInputs.end()) {
            mResource->mUseContentInputs.insert(i);
        }
    }
    mOutputTensors.resize(mResource->mOutputFromTensor.size());
    for (int i = 0; i < mResource->mOutputFromTensor.size(); ++i) {
        mOutputTensors[i] = mSession->getOutput(outputs[mResource->mOutputFromTensor[i]].c_str());
    }
}
StaticModule::~StaticModule() {
    mSession         = nullptr;
    mResourceBackend = nullptr;
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

    MNN_ASSERT(inputs.size() == mInputTensors.size());
    for (int i = 0; i < inputs.size(); ++i) {
        if (nullptr == mInputTensors[i]) {
            continue;
        }
        auto exprInfo    = inputs[i]->expr();
        auto inside      = exprInfo.first->inside();
        auto inputTensor = inside->mOutputTensors[exprInfo.second];
        if (nullptr != inside->mCache) {
            inputTensor = Executor::getOutput(inside->mCache.get(), inside->mCacheOffset);
        }
        auto srcDes = TensorUtils::getDescribe(inputTensor);
        auto des = TensorUtils::getDescribe(mInputTensors[i]);
        des->dimensionFormat = srcDes->dimensionFormat;
        des->tensorArrayAttr = srcDes->tensorArrayAttr;
        mInputTensors[i]->buffer().type = inputTensor->buffer().type;
        resizeTensor(mInputTensors[i], inputTensor->shape());
    }
    if (!mResource->mShapeFix) {
        for (int i = 0; i < inputs.size(); ++i) {
            if (nullptr == mInputTensors[i]) {
                continue;
            }
            auto srcPtr = (uint8_t*)inputs[i]->readMap<void>();
            if (srcPtr != mInputTensors[i]->buffer().host) {
                mInputTensors[i]->buffer().host = srcPtr;
                mSession->setNeedMalloc();
                if (mResource->mUseContentInputs.find(i) != mResource->mUseContentInputs.end()) {
                    mSession->setNeedResize();
                }
            }
        }
    }
    mSession->resize();
    if (mResource->mShapeFix) {
        for (int i = 0; i < inputs.size(); ++i) {
            if (nullptr == mInputTensors[i]) {
                continue;
            }
            auto exprInfo    = inputs[i]->expr();
            auto inside      = exprInfo.first->inside();
            auto inputTensor = inside->mOutputTensors[exprInfo.second];
            if (nullptr != inside->mCache) {
                inputTensor = Executor::getOutput(inside->mCache.get(), inside->mCacheOffset);
            }
            auto backend = TensorUtils::getDescribe(mInputTensors[i])->backend;
            if (nullptr != backend) {
                // For zero shape, backend is null
                backend->onCopyBuffer(inputTensor, mInputTensors[i]);
            }
        }
    }
#ifdef MNN_EXPR_ENABLE_PROFILER
    auto globalExecutor = ExecutorScope::Current();
    Timer cost;
    TensorCallBackWithInfo beforeCallBack = [&cost](const std::vector<Tensor*>&, const OperatorInfo* info) {
        cost.reset();
        return true;
    };
    TensorCallBackWithInfo afterCallBack = [&cost, globalExecutor](const std::vector<Tensor*>&,
                                                                   const OperatorInfo* info) {
        auto costTimes = (float)cost.durationInUs() / 1000.0f;
        globalExecutor->addOpCostTime(info->type(), costTimes);
        globalExecutor->addOpFlops(info->type(), info->flops());
        return true;
    };
    mSession->runWithCallBack(beforeCallBack, afterCallBack);
#else
    mSession->run();
#endif
    for (int i = 0; i < mOutputTensors.size(); ++i) {
        auto currentTensor = mOutputTensors[i];
        auto& quantAttr = TensorUtils::getDescribe(currentTensor)->quantAttr;
        bool isQuant = (quantAttr && TensorUtils::DataTypeToHalideType(quantAttr->type) == currentTensor->getType());
        // copy the data when reused as input tensor with data;
        if (currentTensor->elementSize() > 0 && (mResource->mReusedTensors.find(mResource->mOutputFromTensor[i]) != mResource->mReusedTensors.end() || mResource->mCopyOutput || isQuant)) {
            auto tmpTensor = new Tensor(currentTensor, currentTensor->getDimensionType(), false);
            tmpTensor->buffer().host = (uint8_t*)MNNMemoryAllocAlign(tmpTensor->size(), MNN_MEMORY_ALIGN_DEFAULT);
            auto des                 = TensorUtils::getDescribe(mOutputTensors[i]);
            if (nullptr != des->backend) {
                currentTensor->copyToHostTensor(tmpTensor);
            } else {
                MNNCPUCopyBuffer(currentTensor, tmpTensor);
            }
            TensorUtils::getDescribe(tmpTensor)->dimensionFormat = des->dimensionFormat;
            TensorUtils::getDescribe(tmpTensor)->tensorArrayAttr = des->tensorArrayAttr;
            outputs[mResource->mOutputFromTensor[i]] =
                Express::Variable::create(Express::Expr::create(tmpTensor, true), 0);
        } else {
            outputs[mResource->mOutputFromTensor[i]] = Express::Variable::create(Express::Expr::create(mOutputTensors[i]));
        }
    }
    return outputs;
}

void StaticModule::setReusedTensors(std::set<int> reused) {
    mResource->mReusedTensors = std::move(reused);
}

Module* StaticModule::clone(CloneContext* ctx) const {
    StaticModule* module(new StaticModule);
    module->mResource = mResource;
    if (mResource->mOutputFromTensor.empty()) {
        return this->cloneBaseTo(ctx, module);
    }
    auto rt             = Express::ExecutorScope::Current()->getRuntime();
    auto scheduleInfo  = Schedule::schedule(GetNet(mResource->mNetStorage->buffer()), {mResource->mConfig});
#ifdef MNN_EXPR_ENABLE_PROFILER
    Interpreter::SessionMode callBackMode = Interpreter::Session_Debug;
#else
    Interpreter::SessionMode callBackMode = Interpreter::Session_Release;
#endif
    Interpreter::SessionMode inputMode =
        mResource->mShapeFix ? Interpreter::Session_Input_Inside : Interpreter::Session_Input_User;
    module->mSession.reset(new Session(std::move(scheduleInfo), callBackMode, inputMode, std::move(rt)));
    module->mSession->cloneExecution(mSession->getExecution(0), 0);
    if (scheduleInfo.validForResize && inputMode == Interpreter::Session_Input_Inside) {
        module->mSession->resize(false);
    }
    module->mResourceBackend = mResourceBackend;
    module->mInputTensors.resize(mResource->mInputs.size());
    module->mOutputTensors.resize(mResource->mOutputFromTensor.size());
    for (int i = 0; i < mResource->mInputs.size(); ++i) {
        module->mInputTensors[i] = module->mSession->getInput(mResource->mInputs[i].c_str());
    }
    for (int i = 0; i < mResource->mOutputFromTensor.size(); ++i) {
        module->mOutputTensors[i] = module->mSession->getOutput(mResource->mOutputs[mResource->mOutputFromTensor[i]].c_str());
    }
    return this->cloneBaseTo(ctx, module);
}

void StaticModule::resizeTensor(Tensor* tensor, const std::vector<int>& dims) {
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

    MNN_ASSERT(nullptr != mSession);
    mSession->setNeedResize();
}

} // namespace Express
} // namespace MNN
