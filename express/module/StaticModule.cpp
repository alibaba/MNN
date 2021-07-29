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
#include <queue>

namespace MNN {
namespace Express {

static std::shared_ptr<BufferStorage> preRearrangeWeights( // NOLINT
    const MNN::Net* net, std::map<const Op*, std::shared_ptr<Execution>>& cache, Backend* backend, Backend* backupBackend) {
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

class UseContentCFG {
public:
    UseContentCFG(const Schedule::ScheduleInfo& scheduleInfo) {
        static const std::set<OpType> OutputNotDependInputContent = { OpType_Shape, OpType_Rank, OpType_Size, OpType_PriorBox };
        // build a cfg of use content info: edge is Op, node is Tensor
        for (const auto& info : scheduleInfo.pipelineInfo[0].second) {
            if (OutputNotDependInputContent.find(info.op->type()) != OutputNotDependInputContent.end()) {
                continue;
            }
            opToInput[info.op] = {};
            for (auto input : info.inputs) {
                opToInput[info.op].push_back(input);
            }
            for (auto output : info.outputs) {
                outputToOp[output] = info.op;
            }
            auto needInputs = SizeComputer::needInputContent(info.op, info.inputs.size());
            for (auto inputIdx : needInputs) {
                if (info.inputs.size() > inputIdx) {
                    tensorQueue.emplace(info.inputs[inputIdx]);
                }
            }
        }
    }
    bool hasUseContentTensor(const std::vector<Tensor*>& inputs) {
        // bfs find all tensor content used by shape compute in graph
        std::set<const Tensor*> visited;
        while (!tensorQueue.empty()) {
            auto t = tensorQueue.front();
            tensorQueue.pop();
            if (std::find(inputs.begin(), inputs.end(), t) != inputs.end()) {
                return true;
            }
            if (visited.find(t) == visited.end()) {
                visited.insert(t);
                auto op = outputToOp.find(t);
                if (op != outputToOp.end()) {
                    auto inputs = opToInput.find(op->second);
                    if (inputs != opToInput.end()) {
                        for (const auto input : inputs->second) {
                            tensorQueue.push(input);
                        }
                    }
                }
            }
        }
        return false;
    }
private:
    // output_tensor -> op
    std::map<const Tensor*, const Op*> outputToOp;
    // op -> input_tensors
    std::map<const Op*, std::vector<const Tensor*>> opToInput;
    // root: init use content tensors
    std::queue<const Tensor*> tensorQueue;
};

StaticModule::StaticModule(const void* buffer, size_t length, const std::vector<std::string>& inputs,
                           const std::vector<std::string>& outputs, const Module::Config& moduleconfig, bool copyOutput, std::shared_ptr<Schedule::ScheduleInfo> sharedConst) {
    setType("StaticModule");
    mResource.reset(new Resource);
    mResource->mInputs = inputs;
    mResource->mOutputs = outputs;
    mResource->mCopyOutput = copyOutput;
    mResource->mSharedConst = sharedConst;
    std::shared_ptr<BufferStorage> net_storage;
    std::map<const Op*, std::shared_ptr<Execution>> exeCache;
    if (moduleconfig.rearrange) {
        auto rt = Express::ExecutorScope::Current()->getRuntime();
        MNN_CHECK(rt.first.size() == 1, "The number of formal backends should be 1.");
        mResourceBackend.reset(rt.first.begin()->second->onCreate());
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
    Schedule::ScheduleInfo scheduleInfo;
    // Copy Const
    if (nullptr != mResource->mSharedConst) {
        scheduleInfo.defaultBackend = mResource->mSharedConst->defaultBackend;
        scheduleInfo.allTensors = mResource->mSharedConst->allTensors;
    }
    // Schedule
    auto res = Schedule::schedule(scheduleInfo, GetNet(buffer), {mResource->mConfig}, rt, true);
    if (!res) {
        return;
    }
#ifdef MNN_EXPR_ENABLE_PROFILER
    Interpreter::SessionMode callBackMode = Interpreter::Session_Debug;
#else
    Interpreter::SessionMode callBackMode = Interpreter::Session_Release;
#endif
    UseContentCFG cfg(scheduleInfo);
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
    }
    mResource->mUseContentInputs = cfg.hasUseContentTensor(mInputTensors);
    mOutputTensors.resize(mResource->mOutputFromTensor.size());
    for (int i = 0; i < mResource->mOutputFromTensor.size(); ++i) {
        mOutputTensors[i] = mSession->getOutput(outputs[mResource->mOutputFromTensor[i]].c_str());
    }
    mOutputTensorsWrap.resize(mOutputTensors.size());
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
            }
        }
        if (mResource->mUseContentInputs) {
            mSession->setNeedResize();
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
    ErrorCode code;
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
    code = mSession->runWithCallBack(beforeCallBack, afterCallBack);
#else
    code = mSession->run();
#endif
    if (NO_ERROR != code) {
        return {};
    }
    for (int i = 0; i < mOutputTensors.size(); ++i) {
        auto currentTensor = mOutputTensors[i];
        auto& quantAttr = TensorUtils::getDescribe(currentTensor)->quantAttr;
        bool isQuant = (quantAttr && TensorUtils::DataTypeToHalideType(quantAttr->type) == currentTensor->getType());
        // copy the data when reused as input tensor with data;
        if (currentTensor->elementSize() > 0 && (mResource->mReusedTensors.find(mResource->mOutputFromTensor[i]) != mResource->mReusedTensors.end() || mResource->mCopyOutput || isQuant)) {
            std::shared_ptr<Tensor> tmpTensor(new Tensor(currentTensor, currentTensor->getDimensionType(), true));
            auto des                 = TensorUtils::getDescribe(mOutputTensors[i]);
            if (nullptr != des->backend) {
                currentTensor->copyToHostTensor(tmpTensor.get());
            } else {
                MNNCPUCopyBuffer(currentTensor, tmpTensor.get());
            }
            TensorUtils::getDescribe(tmpTensor.get())->dimensionFormat = des->dimensionFormat;
            TensorUtils::getDescribe(tmpTensor.get())->tensorArrayAttr = des->tensorArrayAttr;
            outputs[mResource->mOutputFromTensor[i]] =
                Express::Variable::create(Express::Expr::create(tmpTensor.get()), 0);
            mOutputTensorsWrap[i] = tmpTensor;
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
    Schedule::ScheduleInfo scheduleInfo;
    if (nullptr != mResource->mSharedConst) {
        scheduleInfo.defaultBackend = mResource->mSharedConst->defaultBackend;
        scheduleInfo.allTensors = mResource->mSharedConst->allTensors;
    }
    auto res = Schedule::schedule(scheduleInfo, GetNet(mResource->mNetStorage->buffer()), {mResource->mConfig}, rt, true);
    if (!res) {
        return nullptr;
    }
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
