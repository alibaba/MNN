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
#include "RuntimeAttr.hpp"
#include "core/TensorUtils.hpp"

namespace MNN {
namespace Express {

static std::shared_ptr<BufferStorage> preRearrangeWeights( // NOLINT
    const MNN::Net* net, std::map<const Op*, std::pair<std::shared_ptr<Execution>, DataType>>& cache, Backend* backend, Backend* backupBackend) {
    std::unique_ptr<MNN::NetT> net_table(net->UnPack());
    std::map<int, std::pair<std::shared_ptr<Execution>, DataType>> exeCache;
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
                DataType type = DataType_DT_FLOAT;
                if (isQuantModel) {
                    type = DataType_DT_INT8;
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
                        // Input Set float to create CastWrapExecution
                        // FIXME: Use better way
                        TensorUtils::getDescribe(inputTensor)->type = DataType_DT_FLOAT;
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
                        // Output Set int8 to create Int8 Execution
                        // FIXME: Use better way
                        TensorUtils::getDescribe(outputTensor)->type = DataType_DT_INT8;
                    } else {
                        outputQuantAttr.reset();
                    }
                    if (inputQuantAttr && outputQuantAttr && op->main_as_Convolution2D()->quanParameter()) {
                        exe.reset(backend->onCreate({inputTensor}, {outputTensor}, op));
                        if (exe.get() == nullptr) {
                            exe.reset(backupBackend->onCreate({inputTensor}, {outputTensor}, op));
                        }
                    }
                } else {
                    exe.reset(backend->onCreate({}, {}, op));
                    if (exe.get() == nullptr) {
                        exe.reset(backupBackend->onCreate({}, {}, op));
                    }
                }
                if (nullptr == exe) {
                    break;
                }
                if (!exe->onClone(nullptr, op, nullptr)) {
                    break;
                }
                exeCache.insert(std::make_pair(i, std::make_pair(exe, type)));
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
    net_storage->storage = builder.ReleaseRaw(net_storage->allocated_size, // NOLINT
                                                  net_storage->offset);
    net = GetNet(net_storage->buffer());
    for (auto& iter : exeCache) {
        auto op = net->oplists()->Get(iter.first);
        cache.insert(std::make_pair(op, iter.second));
    }
    return net_storage;
}

static void _resizeTensor(Tensor* tensor, const Tensor* dims, Session* session) {
    MNN_ASSERT(nullptr != tensor);
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

    if (!dirty) {
        return;
    }

    tensor->buffer().dimensions = (int)dims->dimensions();
    for (int i = 0; i < dims->dimensions(); ++i) {
        tensor->buffer().dim[i].extent = dims->length(i);
        tensor->buffer().dim[i].stride = dims->stride(i);
    }
    session->setNeedResize();
}

StaticModule::StaticModule(const void* buffer, size_t length, const std::vector<std::string>& inputs,
                           const std::vector<std::string>& outputs, std::shared_ptr<MNN::Express::Executor::RuntimeManager> rtMgr, const Module::Config& moduleconfig, bool copyOutput, std::shared_ptr<Schedule::ScheduleInfo> sharedConst) {
    setType("StaticModule");
    mResource.reset(new Resource);
    mResource->mInputs = inputs;
    mResource->mOutputs = outputs;
    mResource->mSharedConst = sharedConst;
    mResource->mModes.inputMode = moduleconfig.shapeMutable ? Interpreter::Session_Input_User : Interpreter::Session_Input_Inside;
    mResource->mModes.outputMode = Interpreter::Session_Output_User;
    std::shared_ptr<BufferStorage> net_storage;
    std::map<const Op*, std::pair<std::shared_ptr<Execution>, DataType>> exeCache;
    RuntimeInfo rt;;
    if(nullptr == rtMgr && moduleconfig.backend != nullptr) {
        ScheduleConfig sche_config;
        sche_config.type = moduleconfig.backend->type;
        sche_config.backendConfig = moduleconfig.backend->config;
        rtMgr.reset(Executor::RuntimeManager::createRuntimeManager(sche_config));
    }
    const BackendConfig* userConfig = nullptr;
    if (nullptr == rtMgr) {
        rt = Executor::getRuntime();
    } else {
        mResource->mModes = rtMgr->getInside()->modes;
        rt = rtMgr->getInside()->mRuntime;
        userConfig = &rtMgr->getInside()->mConfig;
    }
    if (moduleconfig.rearrange) {
        mResourceBackend.reset(rt.first.begin()->second->onCreate(userConfig));
        if (mResourceBackend->type() == MNN_FORWARD_CPU) {
            mBackupResourceBackend = mResourceBackend;
        } else {
            BackendConfig defaultConfig;
            defaultConfig.flags = 4;
            mBackupResourceBackend.reset(rt.second->onCreate(&defaultConfig));
        }
        net_storage = preRearrangeWeights(GetNet(buffer), exeCache, mResourceBackend.get(), mBackupResourceBackend.get());
        buffer      = net_storage->buffer();
        length      = net_storage->size();
    } else {
        net_storage.reset(new BufferStorage);
        net_storage->storage = new uint8_t[length];
        if (nullptr == net_storage->storage) {
            MNN_ERROR("Allock Error in StaticModule's net\n");
            return;
        }
        net_storage->allocated_size = length;
        net_storage->offset         = 0;
        ::memcpy(net_storage->storage, buffer, length);
        buffer = net_storage->storage;
    }
    mResource->mNetStorage    = std::move(net_storage);
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
    // TODO: Add Config
    mResource->mConfig.numThread   = 1;
    mResource->mConfig.type        = rt.first.begin()->first;
    mResource->mConfig.path.mode   = ScheduleConfig::Path::Mode::Tensor;
    mResource->mConfig.path.outputs = outputs;
    mResource->mConfig.saveTensors = outputs;
    mResource->mConfig.path.inputs = inputs;
    mResource->mConfig.backendConfig = (BackendConfig*)userConfig;
    Schedule::ScheduleInfo scheduleInfo;
    // Copy Const
    if (nullptr != mResource->mSharedConst) {
        scheduleInfo.defaultBackend = mResource->mSharedConst->defaultBackend;
        scheduleInfo.allTensors = mResource->mSharedConst->allTensors;
    }
    // Schedule
    auto res = Schedule::schedule(scheduleInfo, GetNet(buffer), {mResource->mConfig}, rt);
    if (!res) {
        return;
    }

    mResource->mUseContentInputs = scheduleInfo.needInputContentForShape;
    if (mResource->mUseContentInputs) {
        mResource->mModes.inputMode = Interpreter::Session_Input_User;
    }
    mSession.reset(new Session(std::move(scheduleInfo), mResource->mModes, std::move(rt)));
    mSession->cloneExecution(exeCache);
    if (scheduleInfo.validForResize && mResource->mModes.inputMode == Interpreter::Session_Input_Inside) {
        mSession->resize(false);
    }
    mInputTensors.resize(inputs.size());
    for (int i = 0; i < inputs.size(); ++i) {
        mInputTensors[i] = mSession->getInput(inputs[i].c_str());
#ifdef LOG_VERBOSE
        MNN_PRINT("init Staticmodule %d th input ptr:%p,  hostPtr:%p, name:%s\n", i, mInputTensors[i], mInputTensors[i]->host<void>(), inputs[i].c_str());
#endif
    }
    mOutputTensors.resize(mResource->mOutputFromTensor.size());
    for (int i = 0; i < mResource->mOutputFromTensor.size(); ++i) {
        mOutputTensors[i] = mSession->getOutput(outputs[mResource->mOutputFromTensor[i]].c_str());
    }
}
StaticModule::~StaticModule() {
    mSession         = nullptr;
    mResourceBackend = nullptr;
    mBackupResourceBackend = nullptr;
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
    if (mResource->mModes.inputMode == Interpreter::Session_Input_User) {
        for (int i = 0; i < inputs.size(); ++i) {
            if (nullptr == mInputTensors[i]) {
                continue;
            }
            auto inputTensor = Utils::getTensor(inputs[i]);
            auto srcDes = TensorUtils::getDescribe(inputTensor);
            auto des = TensorUtils::getDescribe(mInputTensors[i]);
            des->quantAttr = srcDes->quantAttr;
            des->type = srcDes->type;
            des->dimensionFormat = srcDes->dimensionFormat;
            des->tensorArrayAttr = srcDes->tensorArrayAttr;
            des->backend = srcDes->backend;
            mInputTensors[i]->buffer().type = inputTensor->buffer().type;
            _resizeTensor(mInputTensors[i], inputTensor, mSession.get());
            if (mInputTensors[i]->buffer().host != inputTensor->buffer().host || mInputTensors[i]->buffer().device != inputTensor->buffer().device) {
                mSession->setNeedMalloc();
            }
            mInputTensors[i]->buffer().host = inputTensor->buffer().host;
            mInputTensors[i]->buffer().device = inputTensor->buffer().device;

            if (mResource->mUseContentInputs) {

                if (nullptr == mInputTensors[i]->buffer().host && 0 != mInputTensors[i]->buffer().device ) {

                    auto exprInfo    = inputs[i]->expr();
                    auto inside      = exprInfo.first->inside();
                    auto srcPtr = inputs[i]->readMap<void>();
                    mInputTensors[i]->buffer().host = inside->mHostTensor->buffer().host;
                }
            }

        }
        if (mResource->mUseContentInputs) {
            mSession->setNeedResize();
        }
        mSession->resize();
    } else {
        // Resize
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
            mInputTensors[i]->buffer().type = inputTensor->buffer().type;
            _resizeTensor(mInputTensors[i], inputTensor, mSession.get());
        }
        mSession->resize();
        // Copy
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
    auto rt             = Express::ExecutorScope::Current()->getRuntime();
    Schedule::ScheduleInfo scheduleInfo;
    if (nullptr != mResource->mSharedConst) {
        scheduleInfo.defaultBackend = mResource->mSharedConst->defaultBackend;
        scheduleInfo.allTensors = mResource->mSharedConst->allTensors;
    }
    auto res = Schedule::schedule(scheduleInfo, GetNet(mResource->mNetStorage->buffer()), {mResource->mConfig}, rt);
    if (!res) {
        return nullptr;
    }
    module->mSession.reset(new Session(std::move(scheduleInfo), mResource->mModes, std::move(rt)));
    module->mSession->cloneExecution(mSession->getExecution());
    if (scheduleInfo.validForResize && mResource->mModes.inputMode == Interpreter::Session_Input_Inside) {
        module->mSession->resize(false);
    }
    module->mResourceBackend = mResourceBackend;
    module->mBackupResourceBackend = mBackupResourceBackend;
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

} // namespace Express
} // namespace MNN
