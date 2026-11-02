//
//  CoreMLBackend.cpp
//  MNN
//
//  Created by MNN on 2021/03/24.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CoreMLBackend.hpp"
#include "backend/arm82/Arm82OptFunc.hpp"
#include <algorithm>
#include <cstdlib>
#include <chrono>
#include <cstring>
#include <climits>
#include <CommonCrypto/CommonDigest.h>
#include "core/Command.hpp"

extern bool isAvailable();
namespace MNN {
void registerCoreMLOps();
namespace {
std::map<OpType, CoreMLBackend::Creator*>& creators() {
    static std::map<OpType, CoreMLBackend::Creator*> result;
    return result;
}

class CoreMLMemObj : public Backend::MemObj {
public:
    CoreMLMemObj(std::shared_ptr<EagerBufferAllocator> allocator, MemChunk chunk)
        : mAllocator(std::move(allocator)), mChunk(chunk) {}
    ~CoreMLMemObj() override { mAllocator->free(mChunk); }
    MemChunk chunk() override { return mChunk; }

private:
    std::shared_ptr<EagerBufferAllocator> mAllocator;
    MemChunk mChunk;
};

// Op builders keep local contexts; CV execution groups bind the merged graph.
// Dynamic-weight programs retain independent bindings and state.
class CoreMLProgramExecution : public Execution {
public:
    CoreMLProgramExecution(CoreMLBackend* backend, std::unique_ptr<Execution> execution,
                           std::unique_ptr<CoreMLProgramContext> context, const CoreMLBackend::Creator* creator,
                           const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const Op* op)
        : Execution(backend),
          mExecution(std::move(execution)),
          mContext(std::move(context)),
          mCreator(creator),
          mExecutionType(op->type()),
          mExecutionName(op->name() ? op->name()->str() : "") {
        for (auto tensor : inputs) {
            mInputs.emplace_back(new Tensor(tensor, TensorUtils::getDimType(tensor), false));
            mConstantInputs.push_back(TensorUtils::getDescribe(tensor)->usage == Tensor::InsideDescribe::CONSTANT);
        }
        for (auto tensor : outputs) {
            mOutputs.emplace_back(new Tensor(tensor, TensorUtils::getDimType(tensor), false));
        }
    }
    bool onClone(Backend* bn, const Op* op, Execution** dst) override {
        if (!dst) {
            return true;
        }
        if (!bn || bn->type() != MNN_FORWARD_NN) {
            return false;
        }
        if (op->type() == OpType_Attention && op->main_as_AttentionParam()->kv_shared_layer_index() >= 0) {
            return false;
        }
        auto target = static_cast<CoreMLBackend*>(bn);
        std::unique_ptr<CoreMLProgramContext> context(new CoreMLProgramContext);
        context->weights = mContext->weights;
        if (bn->getMetaPtr() == backend()->getMetaPtr()) {
            context->layerState = mContext->layerState;
        }
        std::vector<Tensor*> inputs, outputs;
        for (auto& tensor : mInputs) {
            inputs.push_back(tensor.get());
        }
        for (auto& tensor : mOutputs) {
            outputs.push_back(tensor.get());
        }
        target->setProgramContext(context.get());
        std::unique_ptr<Execution> execution(mCreator->onCreate(inputs, outputs, op, target));
        target->setProgramContext(nullptr);
        if (!execution || !execution->valid() || !context->valid) {
            return false;
        }
        context->builder.markStaticOperations();
        auto cloned =
            new CoreMLProgramExecution(target, std::move(execution), std::move(context), mCreator, inputs, outputs, op);
        cloned->mConstantInputs = mConstantInputs;
        *dst = cloned;
        return true;
    }
    ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        auto bn = static_cast<CoreMLBackend*>(backend());
        mContext->layoutViews.clear();
        mContext->group.reset();
        mContext->lastInGroup = false;
        mCallbackGroup.reset();
        mContext->builder.resetOperations();
        mContext->flatIO = false;
        mContext->names.clear();
        mContext->outputs.clear();
        mContext->constants.clear();
        mContext->bindings.clear();
        mContext->boundNames.clear();
        for (size_t i = 0; i < inputs.size(); ++i) {
            mContext->names[inputs[i]] = "input_" + std::to_string(i);
            if (i < mConstantInputs.size() && mConstantInputs[i]) {
                mContext->constants.insert(inputs[i]);
            }
        }
        for (size_t i = 0; i < outputs.size(); ++i) {
            mContext->names[outputs[i]] = "output_" + std::to_string(i);
            mContext->outputs.insert(outputs[i]);
        }
        bn->setProgramContext(mContext.get());
        std::vector<Tensor*> nativeInputs = inputs, nativeOutputs = outputs;
        bool spatial = mExecutionType == OpType_Convolution || mExecutionType == OpType_ConvolutionDepthwise ||
                       mExecutionType == OpType_Deconvolution || mExecutionType == OpType_DeconvolutionDepthwise ||
                       mExecutionType == OpType_Pooling || mExecutionType == OpType_Interp;
        auto view = [&](Tensor* tensor, const std::string& name) {
            std::shared_ptr<Tensor> result(new Tensor(tensor, Tensor::CAFFE, false));
            auto pointer = result.get();
            mContext->layoutViews.push_back(result);
            mContext->names[pointer] = name;
            mContext->outputs.insert(pointer);
            return pointer;
        };
        auto transpose = [&](Tensor* src, Tensor* dst, bool toNCHW, const std::string& tag) {
            int rank = src->dimensions();
            std::vector<int> perm(rank);
            perm[0] = 0;
            if (toNCHW) {
                perm[1] = rank - 1;
                for (int i = 2; i < rank; ++i) {
                    perm[i] = i - 1;
                }
            } else {
                for (int i = 1; i < rank - 1; ++i) {
                    perm[i] = i + 1;
                }
                perm.back() = 1;
            }
            std::unique_ptr<MilOperationBuilder> op(new MilOperationBuilder("transpose"));
            op->setName(tag);
            op->addInput("x", bn->getTensorName(src));
            op->addInput("perm", bn->buildArray(perm, tag + "_perm"));
            auto shape = dst->shape();
            op->addOutput(bn->getTensorName(dst), bn->typeConvert(dst->getType()),
                          std::vector<int64_t>(shape.begin(), shape.end()));
            bn->addOperation(std::move(op));
        };
        if (spatial && !inputs.empty() && inputs[0]->dimensions() > 2 &&
            TensorUtils::getDescribe(inputs[0])->dimensionFormat == MNN_DATA_FORMAT_NHWC) {
            nativeInputs[0] = view(inputs[0], "layout_input");
            transpose(inputs[0], nativeInputs[0], true, "to_nchw");
        }
        if (spatial && outputs.size() == 1 && outputs[0]->dimensions() > 2 &&
            TensorUtils::getDescribe(outputs[0])->dimensionFormat == MNN_DATA_FORMAT_NHWC) {
            nativeOutputs[0] = view(outputs[0], "layout_output");
        }
        auto code = mExecution->onResize(nativeInputs, nativeOutputs);
        if (code == NO_ERROR && nativeOutputs != outputs) {
            transpose(nativeOutputs[0], outputs[0], false, "to_nhwc");
        }
        // Constructor-time weights need to be rebound after resetting the model.
        for (const auto& weight : mContext->weights->tensors) {
            bn->addInput(weight.second.get(), weight.first);
        }
        if (code == NO_ERROR && mContext->builder.numOperations() != 0) {
            if (static_cast<const CoreMLRuntime*>(bn->getRuntime())->lowMemory() ||
                mExecutionType == OpType_Attention) {
                code = bn->buildProgram(*mContext, outputs);
            } else {
                bn->recordProgram(this, mContext.get(), outputs);
            }
        }
        bn->setProgramContext(nullptr);
        if (code != NO_ERROR) {
            MNN_ERROR("[CoreML] Resize failed for %s (%s): %d\n", EnumNameOpType(mExecutionType),
                      mExecutionName.c_str(), code);
        }
        mCopyOnly = mContext->builder.numOperations() == 0;
        return code;
    }
    ErrorCode onExecuteWithCallBack(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if (!mContext->group) {
            return onExecute(inputs, outputs);
        }
        auto group = mContext->group;
        bool last = mContext->lastInGroup;
        if (!mCallbackGroup) {
            auto bn = static_cast<CoreMLBackend*>(backend());
            auto code = bn->buildCallbackProgram(this, *mContext, inputs, outputs);
            if (code != NO_ERROR) {
                mContext->group = group;
                mContext->lastInGroup = last;
                return code;
            }
            mCallbackGroup = mContext->group;
        }
        mContext->group = mCallbackGroup;
        mContext->lastInGroup = true;
        auto code = onExecute(inputs, outputs);
        mContext->group = group;
        mContext->lastInGroup = last;
        return code;
    }
    ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if (mContext->group) {
            if (!mContext->lastInGroup) {
                return NO_ERROR;
            }
            auto& group = *mContext->group;
            if (!group.session->invokModel(group.inputs, group.staging)) {
                return NO_EXECUTION;
            }
            for (size_t i = 0; i < group.outputs.size(); ++i) {
                auto tensor = group.outputs[i].first;
                ::memcpy(tensor->host<void>(), group.staging[i].first->host<void>(),
                         TensorUtils::getRawSize(tensor) *
                             static_cast<CoreMLBackend*>(backend())->getBytes(tensor->getType()));
            }
            if (std::getenv("MNN_COREML_TRACE")) {
                MNN_PRINT("[CoreML] executed fused graph\n");
            }
            return NO_ERROR;
        }
        std::lock_guard<std::mutex> stateLock(mContext->layerState->mutex);
        auto runtime = static_cast<const CoreMLRuntime*>(backend()->getRuntime());
        std::unique_lock<std::mutex> weightLock;
        if (!mContext->weights->streamed.empty()) {
            weightLock = runtime->lockWeights();
            if (!runtime->prepareWeights(*mContext)) {
                return OUT_OF_MEMORY;
            }
        }
        auto code = mExecution->onExecute(inputs, outputs);
        if (code != NO_ERROR) {
            MNN_ERROR("[CoreML] Prepare failed for %s (%s): %d\n", EnumNameOpType(mExecutionType),
                      mExecutionName.c_str(), code);
            return code;
        }
        if (mCopyOnly) {
            if (inputs.empty() || outputs.size() != 1 ||
                TensorUtils::getRawSize(inputs[0]) != TensorUtils::getRawSize(outputs[0])) {
                MNN_ERROR("[CoreML] Invalid copy for %s (%s)\n", EnumNameOpType(mExecutionType),
                          mExecutionName.c_str());
                return NOT_SUPPORT;
            }
            auto bn = static_cast<CoreMLBackend*>(backend());
            ::memmove(outputs[0]->host<void>(), inputs[0]->host<void>(),
                      TensorUtils::getRawSize(outputs[0]) * bn->getBytes(outputs[0]->getType()));
            return NO_ERROR;
        }
        if (mExecutionType == OpType_Attention && mContext->session != mContext->layerState->session) {
            auto session = mContext->program->createSession(mContext->layerState->session.get());
            if (!session) {
                return NO_EXECUTION;
            }
            mContext->session = session;
            mContext->layerState->session = session;
        }
        std::vector<std::pair<const Tensor*, std::string>> out;
        for (auto tensor : outputs) {
            out.emplace_back(tensor, mContext->names.at(tensor));
        }
        bool success = mContext->session && mContext->session->invokModel(mContext->bindings, out, mContext->flatIO);
        if (success && std::getenv("MNN_COREML_TRACE")) {
            MNN_PRINT("[CoreML] executed %s (%s)\n", EnumNameOpType(mExecutionType), mExecutionName.c_str());
        }
        return success ? NO_ERROR : NO_EXECUTION;
    }

private:
    std::unique_ptr<Execution> mExecution;
    std::unique_ptr<CoreMLProgramContext> mContext;
    bool mCopyOnly = false;
    const CoreMLBackend::Creator* mCreator;
    OpType mExecutionType;
    std::string mExecutionName;
    std::vector<std::shared_ptr<Tensor>> mInputs, mOutputs;
    std::vector<bool> mConstantInputs;
    std::shared_ptr<CoreMLProgramGroup> mCallbackGroup;
};
} // namespace

ErrorCode CoreMLBackend::onResizeEnd() {
    for (auto& entry : mPending) {
        mContext = entry.second.context;
        auto code = buildProgram(*mContext, entry.second.outputs);
        mContext = nullptr;
        if (code != NO_ERROR) {
            return code;
        }
    }
    return NO_ERROR;
}

ErrorCode CoreMLBackend::onResizeEnd(const std::vector<Command*>& commands) {
    if (mPending.empty()) {
        return NO_ERROR;
    }
    if (std::getenv("MNN_COREML_DISABLE_FUSION")) {
        return onResizeEnd();
    }
    return buildProgramGroups(commands, mPending);
}

ErrorCode CoreMLBackend::buildCallbackProgram(Execution* execution, CoreMLProgramContext& context,
                                              const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    Command command{};
    command.workInputs = inputs;
    command.workOutputs = outputs;
    command.execution = std::shared_ptr<Execution>(execution, [](Execution*) {});
    std::map<Execution*, PendingProgram> pending;
    pending[execution] = {&context, outputs};
    return buildProgramGroups({&command}, pending, true);
}

ErrorCode CoreMLBackend::buildProgramGroups(const std::vector<Command*>& commands,
                                            const std::map<Execution*, PendingProgram>& programs,
                                            bool forceFileCompilation) {
    std::map<const Tensor*, std::vector<size_t>> consumers;
    for (size_t i = 0; i < commands.size(); ++i) {
        for (auto tensor : commands[i]->workInputs) {
            consumers[tensor].push_back(i);
        }
    }
    for (size_t begin = 0; begin < commands.size();) {
        auto found = programs.find(commands[begin]->execution.get());
        if (found == programs.end()) {
            ++begin;
            continue;
        }
        if (found->second.context->flatIO) {
            // Keep the compiler workaround for arbitrary rank-five Raster regions.
            mContext = found->second.context;
            auto code = buildProgram(*mContext, found->second.outputs);
            mContext = nullptr;
            if (code != NO_ERROR) {
                return code;
            }
            ++begin;
            continue;
        }
        size_t end = begin;
        const char* groupLimit = std::getenv("MNN_COREML_GROUP_SIZE");
        size_t limit = groupLimit ? std::max(1, atoi(groupLimit)) : commands.size();
        for (; end < commands.size() && end - begin < limit; ++end) {
            auto entry = programs.find(commands[end]->execution.get());
            if (entry == programs.end() || entry->second.context->flatIO) {
                break;
            }
        }
        if (std::getenv("MNN_COREML_VERBOSE")) {
            MNN_PRINT("[CoreML] building group %zu..%zu: %s\n", begin, end,
                      commands[begin]->op && commands[begin]->op->name() ? commands[begin]->op->name()->c_str()
                                                                         : "unnamed");
            fflush(stdout);
        }
        MlModelBuilder builder(9);
        std::shared_ptr<CoreMLProgramGroup> group(new CoreMLProgramGroup);
        std::map<const Tensor*, std::string> values;
        // Count the serialized blob, including metadata and alignment, before allocating it.
        constexpr size_t memoryWeightLimit = 10 * 1024 * 1024;
        size_t weightBytes = 64;
        std::set<const Tensor*> seen;
        for (size_t i = begin; i < end && weightBytes <= memoryWeightLimit; ++i) {
            auto& pending = programs.at(commands[i]->execution.get());
            auto& context = *pending.context;
            for (const auto& binding : context.bindings) {
                const Tensor* tensor = binding.first;
                if (!context.builder.usesInput(binding.second) || !seen.insert(tensor).second ||
                    !(context.weights->tensors.count(binding.second) || context.constants.count(tensor))) {
                    continue;
                }
                size_t elements = TensorUtils::getRawSize(tensor);
                size_t elementBytes = getBytes(tensor->getType());
                if (elementBytes == 0) {
                    return NO_EXECUTION;
                }
                if (elements > memoryWeightLimit / elementBytes) {
                    weightBytes = memoryWeightLimit + 1;
                    break;
                }
                size_t entryBytes = (64 + elements * elementBytes + 63) / 64 * 64;
                if (entryBytes > memoryWeightLimit - weightBytes) {
                    weightBytes = memoryWeightLimit + 1;
                    break;
                }
                weightBytes += entryBytes;
            }
            for (auto tensor : pending.outputs) {
                seen.insert(tensor);
            }
        }
        std::string path;
        std::shared_ptr<void> cleanup;
        std::shared_ptr<std::vector<uint8_t>> weightData;
        std::unique_ptr<MilWeightWriter> writer;
        if (weightBytes <= memoryWeightLimit) {
            weightData = std::make_shared<std::vector<uint8_t>>();
            writer.reset(new MilWeightWriter(weightData, weightBytes));
        } else {
            cleanup = createCoreMLTemporaryDirectory(path);
            if (!cleanup) {
                return NO_EXECUTION;
            }
            path += "/weight.bin";
            writer.reset(new MilWeightWriter(path));
        }
        bool hasWeights = false;
        auto addInfo = [&](const Tensor* tensor, const std::string& value, bool output) {
            std::unique_ptr<MilTensorInfo> info(new MilTensorInfo);
            info->name = value;
            info->data_type = typeConvert(tensor->getType());
            auto shape = tensor->shape();
            info->shape.assign(shape.begin(), shape.end());
            info->type = output ? MilTensorInfo::Type::OUTPUT : MilTensorInfo::Type::INPUT;
            if (output) {
                builder.addOutput(std::move(info));
            } else {
                builder.addInput(std::move(info));
            }
        };
        for (size_t i = begin; i < end; ++i) {
            auto& pending = programs.at(commands[i]->execution.get());
            auto& context = *pending.context;
            const auto prefix = "node_" + std::to_string(i - begin) + "_";
            std::map<std::string, std::string> names;
            for (const auto& binding : context.bindings) {
                if (!context.builder.usesInput(binding.second)) {
                    continue;
                }
                const Tensor* tensor = binding.first;
                auto known = values.find(tensor);
                if (known != values.end()) {
                    names[binding.second] = known->second;
                    continue;
                }
                const auto value = prefix + binding.second;
                names[binding.second] = value;
                values[tensor] = value;
                bool constant = context.weights->tensors.count(binding.second) || context.constants.count(tensor);
                if (constant) {
                    auto shape = tensor->shape();
                    size_t elements = TensorUtils::getRawSize(tensor);
                    const void* data = tensor->host<void>();
                    std::vector<int16_t> converted;
                    auto owner = TensorUtils::getDescribeOrigin(tensor)->getBackend();
                    if (tensor->getType() == halide_type_of<float>() && (!owner || owner->type() != MNN_FORWARD_NN)) {
                        converted.resize(elements);
                        MNNQuantizeFP16(tensor->host<float>(), converted.data(), elements);
                        data = converted.data();
                    }
                    if (!data) {
                        return NO_EXECUTION;
                    }
                    auto dtype = typeConvert(tensor->getType());
                    auto offset = writer->write(data, elements * getBytes(tensor->getType()), dtype);
                    std::unique_ptr<MilOperationBuilder> op(new MilOperationBuilder("const"));
                    op->setName(value);
                    op->addOutput(value, dtype, std::vector<int64_t>(shape.begin(), shape.end()));
                    op->addAttribute("val", MilValueBuilder::createMilValueBuilder("@model_path/weights/weight.bin",
                                                                                   dtype, shape, offset));
                    builder.addOperation(std::move(op));
                    hasWeights = true;
                } else {
                    addInfo(tensor, value, false);
                    group->inputs.emplace_back(tensor, value);
                }
            }
            for (auto tensor : pending.outputs) {
                auto local = context.names.at(tensor);
                auto value = prefix + local;
                names[local] = value;
                values[tensor] = value;
            }
            builder.appendOperations(context.builder, names, prefix);
            for (auto tensor : pending.outputs) {
                bool external = TensorUtils::getDescribe(tensor)->usage == Tensor::InsideDescribe::OUTPUT;
                auto uses = consumers.find(tensor);
                if (uses == consumers.end()) {
                    external = true;
                } else {
                    for (auto use : uses->second) {
                        external = external || use < begin || use >= end;
                    }
                }
                if (!external) {
                    continue;
                }
                const auto& value = values.at(tensor);
                addInfo(tensor, value, true);
                group->outputs.emplace_back(tensor, value);
                auto dtype = tensor->getType().code == halide_type_float ? halide_type_t(halide_type_float, 16)
                                                                         : tensor->getType();
                std::shared_ptr<Tensor> staging(Tensor::create(tensor->shape(), dtype, nullptr, Tensor::CAFFE));
                if (!staging || !staging->host<void>()) {
                    return OUT_OF_MEMORY;
                }
                group->staging.emplace_back(staging.get(), value);
                group->storage.push_back(std::move(staging));
            }
        }
        if (!writer->valid()) {
            return NO_EXECUTION;
        }
        size_t actualWeightBytes = writer->tell();
        writer.reset();
        if (weightData && weightData->size() != weightBytes) {
            MNN_ERROR("[CoreML] Weight blob size changed during graph construction\n");
            return NO_EXECUTION;
        }
        if (std::getenv("MNN_COREML_VERBOSE") && hasWeights) {
            MNN_PRINT("[CoreML] weight storage: %s (%zu bytes)\n", weightData ? "memory" : "file", actualWeightBytes);
        }
        auto program = mRuntime->getProgram(builder.build(), hasWeights ? path : "", hasWeights ? weightData : nullptr,
                                            forceFileCompilation);
        if (!program) {
            return NO_EXECUTION;
        }
        group->session = program->createSession();
        if (!group->session) {
            return NO_EXECUTION;
        }
        for (size_t i = begin; i < end; ++i) {
            auto context = programs.at(commands[i]->execution.get()).context;
            context->group = group;
            context->lastInGroup = i + 1 == end;
        }
        if (std::getenv("MNN_COREML_VERBOSE")) {
            MNN_PRINT("[CoreML] fused %zu commands: %zu inputs, %zu outputs, static_weights=%d\n", end - begin,
                      group->inputs.size(), group->outputs.size(), hasWeights);
        }
        begin = end;
    }
    return NO_ERROR;
}

bool CoreMLBackend::addCreator(OpType type, Creator* creator) {
    return creators().emplace(type, creator).second;
}
CoreMLBackend::CoreMLBackend(const CoreMLRuntime* runtime) : Backend(MNN_FORWARD_NN), mRuntime(runtime) {
    mBuffers.reset(new EagerBufferAllocator(BufferAllocator::Allocator::createDefault()));
}
Execution* CoreMLBackend::onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                   const Op* op) {
    auto unsupported = [&](const char* reason) -> Execution* {
        if (std::getenv("MNN_COREML_TRACE")) {
            MNN_PRINT("[CoreML] unsupported %s (%s)\n", EnumNameOpType(op->type()), reason);
        }
        return nullptr;
    };
    // Core ML's FP32 score path uses CPU while retaining full-capacity MLState
    // and ANE transfer buffers. The native CPU attention uses only the active KV.
    if (op->type() == OpType_Attention && mRuntime->lowMemory()) {
        return unsupported("low-memory policy");
    }
    if (op->type() == OpType_LayerNorm && (inputs.size() != 1 || outputs.size() != 1)) {
        return unsupported("dynamic LayerNorm parameters");
    }
    // MIL supports int32 indices and comparison results as well as floating-point tensors.
    for (auto tensor : outputs) {
        if (tensor->getType() != halide_type_of<float>() && tensor->getType() != halide_type_of<int>()) {
            return unsupported("tensor type");
        }
    }
    for (auto tensor : inputs) {
        if (tensor->getType() != halide_type_of<float>() && tensor->getType() != halide_type_of<int>()) {
            return unsupported("tensor type");
        }
    }
    auto iter = creators().find(op->type());
    if (iter == creators().end()) {
        return unsupported("no creator");
    }
    std::unique_ptr<CoreMLProgramContext> context(new CoreMLProgramContext);
    mContext = context.get();
    std::unique_ptr<Execution> execution(iter->second->onCreate(inputs, outputs, op, this));
    mContext = nullptr;
    if (!execution || !execution->valid() || !context->valid) {
        return unsupported("parameters");
    }
    context->builder.markStaticOperations();
    return new CoreMLProgramExecution(this, std::move(execution), std::move(context), iter->second, inputs, outputs,
                                      op);
}
Backend::MemObj* CoreMLBackend::onAcquire(const Tensor* tensor, StorageType storageType) {
    size_t size = TensorUtils::getRawSize(tensor) * getBytes(tensor->getType());
    auto chunk = mBuffers->alloc(std::max<size_t>(size, 1), storageType != DYNAMIC);
    if (chunk.invalid()) {
        return nullptr;
    }
    const_cast<Tensor*>(tensor)->buffer().host = chunk.ptr();
    // Express must copy device FP16 storage before exposing a float host view.
    const_cast<Tensor*>(tensor)->buffer().device = 1;
    return new CoreMLMemObj(mBuffers, chunk);
}
bool CoreMLBackend::onClearBuffer() {
    mBuffers->release(false);
    return true;
}
int CoreMLBackend::getBytes(const halide_type_t& type) const {
    return type.code == halide_type_float ? 2 : type.bytes();
}
CoreML__Specification__MILSpec__DataType CoreMLBackend::typeConvert(const halide_type_t& type) const {
    if (type.code == halide_type_float) {
        return CORE_ML__SPECIFICATION__MILSPEC__DATA_TYPE__FLOAT16;
    }
    if (type == halide_type_of<int>()) {
        return CORE_ML__SPECIFICATION__MILSPEC__DATA_TYPE__INT32;
    }
    if (type == halide_type_of<int16_t>()) {
        return CORE_ML__SPECIFICATION__MILSPEC__DATA_TYPE__INT16;
    }
    if (type == halide_type_of<int8_t>()) {
        return CORE_ML__SPECIFICATION__MILSPEC__DATA_TYPE__INT8;
    }
    if (type == halide_type_of<uint8_t>()) {
        return CORE_ML__SPECIFICATION__MILSPEC__DATA_TYPE__UINT8;
    }
    return CORE_ML__SPECIFICATION__MILSPEC__DATA_TYPE__UNUSED_TYPE;
}
void CoreMLBackend::onCopyBuffer(const Tensor* src, const Tensor* dst) const {
    auto srcBackend = TensorUtils::getDescribeOrigin(src)->getBackend();
    auto dstBackend = TensorUtils::getDescribeOrigin(dst)->getBackend();
    bool fromCoreML = srcBackend && srcBackend->type() == MNN_FORWARD_NN;
    bool toCoreML = dstBackend && dstBackend->type() == MNN_FORWARD_NN;
    if (!fromCoreML && !toCoreML) {
        if (std::getenv("MNN_COREML_TRACE")) {
            MNN_PRINT("[CoreML] host copy %d -> %d\n", srcBackend ? srcBackend->type() : -1,
                      dstBackend ? dstBackend->type() : -1);
        }
        if (srcBackend && srcBackend->type() == MNN_FORWARD_CPU_EXTENSION) {
            srcBackend->onCopyBuffer(src, dst);
        } else if (dstBackend && dstBackend->type() == MNN_FORWARD_CPU_EXTENSION) {
            dstBackend->onCopyBuffer(src, dst);
        } else {
            MNNCPUCopyBuffer(src, dst);
        }
        return;
    }
    if (fromCoreML && toCoreML) {
        ::memmove(dst->host<void>(), src->host<void>(), TensorUtils::getRawSize(src) * getBytes(src->getType()));
        return;
    }
    if (src->getType().code != halide_type_float) {
        auto tensor = toCoreML ? dst : src;
        auto format = TensorUtils::getDimType(tensor) == Tensor::TENSORFLOW ? Tensor::TENSORFLOW : Tensor::CAFFE;
        std::shared_ptr<Tensor> dense(Tensor::create(tensor->shape(), tensor->getType(), nullptr, format));
        size_t bytes = TensorUtils::getRawSize(tensor) * tensor->getType().bytes();
        if (toCoreML) {
            MNNCPUCopyBuffer(src, dense.get());
            ::memcpy(dst->host<void>(), dense->host<void>(), bytes);
        } else {
            ::memcpy(dense->host<void>(), src->host<void>(), bytes);
            MNNCPUCopyBuffer(dense.get(), dst);
        }
        return;
    }
    // Device storage is dense logical NCHW, including tensors marked NC4HW4.
    if (toCoreML) {
        std::shared_ptr<Tensor> dense(Tensor::create<float>(
            dst->shape(), nullptr,
            TensorUtils::getDimType(dst) == Tensor::TENSORFLOW ? Tensor::TENSORFLOW : Tensor::CAFFE));
        if (srcBackend && srcBackend->type() == MNN_FORWARD_CPU_EXTENSION) {
            srcBackend->onCopyBuffer(src, dense.get());
        } else {
            MNNCPUCopyBuffer(src, dense.get());
        }
        MNNQuantizeFP16(dense->host<float>(), dst->host<int16_t>(), TensorUtils::getRawSize(dst));
    } else if (fromCoreML) {
        std::shared_ptr<Tensor> dense(Tensor::create<float>(
            src->shape(), nullptr,
            TensorUtils::getDimType(src) == Tensor::TENSORFLOW ? Tensor::TENSORFLOW : Tensor::CAFFE));
        MNNDequantizeFP16(src->host<int16_t>(), dense->host<float>(), TensorUtils::getRawSize(src));
        if (dstBackend && dstBackend->type() == MNN_FORWARD_CPU_EXTENSION) {
            dstBackend->onCopyBuffer(dense.get(), dst);
        } else {
            MNNCPUCopyBuffer(dense.get(), dst);
        }
    }
}

std::string CoreMLBackend::addInput(const Tensor* tensor, const std::string& name) {
    if (mContext->weights->streamed.count(name)) {
        // Borrowed addresses belong to this execution, even when clones share
        // the immutable source across different runtimes/workspaces.
        auto& view = mContext->weightViews[name];
        if (!view) {
            view.reset(new Tensor(tensor, Tensor::CAFFE, false));
        }
        tensor = view.get();
    }
    if (mContext->boundNames.insert(name).second) {
        std::unique_ptr<MilTensorInfo> info(new MilTensorInfo);
        info->name = name;
        info->data_type = typeConvert(tensor->getType());
        auto shape = tensorShape(tensor);
        info->shape.assign(shape.begin(), shape.end());
        info->type = MilTensorInfo::Type::INPUT;
        mContext->builder.addInput(std::move(info));
        mContext->bindings.emplace_back(tensor, name);
    }
    return name;
}
std::string CoreMLBackend::getTensorName(const Tensor* tensor) {
    auto iter = mContext->names.find(tensor);
    if (iter == mContext->names.end()) {
        iter = mContext->names.emplace(tensor, "tensor_" + std::to_string(mContext->names.size())).first;
    }
    const auto& name = iter->second;
    if (mContext->outputs.count(tensor)) {
        return name;
    }
    // Shape/axis tensors must remain compile-time constants. Floating-point
    // weights remain runtime inputs and never participate in the program key.
    if (mContext->constants.count(tensor) && tensor->getType() == halide_type_of<int>()) {
        if (tensor->dimensions() == 0) {
            return buildScalar(tensor->host<int>()[0], name);
        }
        return buildArray(std::vector<int>(tensor->host<int>(), tensor->host<int>() + tensor->elementSize()), name);
    }
    if (tensor->dimensions() == 0 && tensor->getType().code == halide_type_float && mContext->constants.count(tensor)) {
        float value;
        auto owner = TensorUtils::getDescribeOrigin(tensor)->getBackend();
        if (owner && owner->type() == MNN_FORWARD_NN) {
            MNNDequantizeFP16(tensor->host<int16_t>(), &value, 1);
        } else {
            value = tensor->host<float>()[0];
        }
        return buildScalar(value, name);
    }
    return addInput(tensor, name);
}
void CoreMLBackend::replaceTensorWith(const Tensor* src, const Tensor* replace) {
    mContext->names[src] = getTensorName(replace);
}
std::string CoreMLBackend::addState(const std::string& name, const std::vector<int>& shape,
                                    CoreML__Specification__MILSpec__DataType dtype) {
    std::unique_ptr<MilTensorInfo> info(new MilTensorInfo);
    info->name = name;
    info->data_type = dtype;
    info->shape.assign(shape.begin(), shape.end());
    info->type = MilTensorInfo::Type::STATE;
    mContext->builder.addInput(std::move(info));
    return name;
}
void CoreMLBackend::addOperation(std::unique_ptr<MilOperationBuilder> builder) {
    mContext->builder.addOperation(std::move(builder));
}
std::string CoreMLBackend::buildBlobConst(const void* data, size_t size, CoreML__Specification__MILSpec__DataType dtype,
                                          const std::vector<int>& shape, std::string name) {
    if (name.empty()) {
        name = "weight_" + std::to_string(mContext->weights->tensors.size());
    }
    auto iter = mContext->weights->tensors.find(name);
    if (iter == mContext->weights->tensors.end()) {
        size_t count = 1;
        for (auto dim : shape) {
            if (dim <= 0) {
                mContext->valid = false;
                return "";
            }
            count *= dim;
        }
        bool fp32 = dtype == CORE_ML__SPECIFICATION__MILSPEC__DATA_TYPE__FLOAT32;
        bool fp16 = dtype == CORE_ML__SPECIFICATION__MILSPEC__DATA_TYPE__FLOAT16;
        if (!data || (!fp32 && !fp16) || size != count * (fp32 ? 4 : 2)) {
            mContext->valid = false;
            return "";
        }
        halide_type_t type = dtype == CORE_ML__SPECIFICATION__MILSPEC__DATA_TYPE__FLOAT32 ||
                                     dtype == CORE_ML__SPECIFICATION__MILSPEC__DATA_TYPE__FLOAT16
                                 ? halide_type_t(halide_type_float, 16)
                                 : halide_type_of<int>();
        std::shared_ptr<Tensor> tensor(Tensor::create(shape, type, nullptr, Tensor::CAFFE));
        if (!tensor || !tensor->host<void>()) {
            return "";
        }
        if (dtype == CORE_ML__SPECIFICATION__MILSPEC__DATA_TYPE__FLOAT32) {
            MNNQuantizeFP16(static_cast<const float*>(data), tensor->host<int16_t>(), size / sizeof(float));
        } else {
            ::memcpy(tensor->host<void>(), data, size);
        }
        iter = mContext->weights->tensors.emplace(name, std::move(tensor)).first;
    }
    return addInput(iter->second.get(), name);
}
ErrorCode CoreMLBackend::buildProgram(CoreMLProgramContext& context, const std::vector<Tensor*>& outputs) {
    for (auto tensor : outputs) {
        std::unique_ptr<MilTensorInfo> info(new MilTensorInfo);
        info->name = getTensorName(tensor);
        info->data_type = typeConvert(tensor->getType());
        auto shape = tensorShape(tensor);
        info->shape.assign(shape.begin(), shape.end());
        info->type = MilTensorInfo::Type::OUTPUT;
        context.builder.addOutput(std::move(info));
    }
    if (!context.valid) {
        return NOT_SUPPORT;
    }
    // Per-command callback execution can create hundreds of models; file compilation avoids retaining large assets.
    auto program = mRuntime->getProgram(context.builder.build(), "", nullptr, true);
    if (!program) {
        return NO_EXECUTION;
    }
    if (context.program != program) {
        std::lock_guard<std::mutex> stateLock(context.layerState->mutex);
        auto session = program->createSession(context.layerState->session.get());
        if (!session) {
            return NO_EXECUTION;
        }
        context.program = std::move(program);
        context.session = std::move(session);
        context.layerState->session = context.session;
    }
    return NO_ERROR;
}
CoreMLRuntime::CoreMLRuntime(const Backend::Info& info) {
    mPrecision = info.user ? info.user->precision : BackendConfig::Precision_Normal;
    mMemory = info.user ? info.user->memory : BackendConfig::Memory_Normal;
}
CoreMLRuntime::~CoreMLRuntime() {
    if (std::getenv("MNN_COREML_VERBOSE")) {
        MNN_PRINT("[CoreML] cache: loads=%zu hits=%zu source_compilations=%zu\n", mProgramLoads, mCacheHits,
                  mCompilations);
        MNN_PRINT(
            "[CoreML] weights: loads=%zu hits=%zu retained_fp16_bytes=%zu quantized_bytes=%zu scratch_bytes=%zu\n",
            mWeightLoads, mWeightHits, mWeightBytes, mQuantizedBytes, mWeightScratch.size() * sizeof(int16_t));
    }
}
std::shared_ptr<CoreMLWeights> CoreMLRuntime::findWeights(const std::string& key) const {
    std::lock_guard<std::mutex> lock(mWeightMutex);
    auto found = mWeights.find(key);
    if (found == mWeights.end()) {
        return {};
    }
    auto result = found->second.lock();
    if (!result) {
        mWeights.erase(found);
        return nullptr;
    }
    ++mWeightHits;
    return result;
}
void CoreMLRuntime::cacheWeights(const std::string& key, const std::shared_ptr<CoreMLWeights>& weights) const {
    std::lock_guard<std::mutex> lock(mWeightMutex);
    mWeights[key] = weights;
    for (const auto& weight : weights->tensors) {
        if (weights->streamed.count(weight.first)) {
            mQuantizedBytes += weights->streamed.at(weight.first)->storageBytes;
        } else {
            mWeightBytes += TensorUtils::getRawSize(weight.second.get()) * weight.second->getType().bytes();
        }
    }
    ++mWeightLoads;
}
bool CoreMLBackend::restoreWeights(const std::string& key) {
    if (key.empty()) {
        return false;
    }
    auto weights = mRuntime->findWeights(key);
    if (!weights) {
        return false;
    }
    mContext->weights = std::move(weights);
    return true;
}
void CoreMLBackend::cacheWeights(const std::string& key) {
    if (!key.empty()) {
        mRuntime->cacheWeights(key, mContext->weights);
    }
}
Tensor* CoreMLBackend::allocateWeight(const std::string& name, const std::vector<int>& shape,
                                      std::shared_ptr<CoreMLWeightSource> source) {
    auto& tensor = mContext->weights->tensors[name];
    if (!tensor) {
        tensor.reset(source ? Tensor::createDevice(shape, halide_type_t(halide_type_float, 16), Tensor::CAFFE)
                            : Tensor::create(shape, halide_type_t(halide_type_float, 16), nullptr, Tensor::CAFFE));
    }
    if (!tensor || (!source && !tensor->host<void>())) {
        mContext->valid = false;
        return nullptr;
    }
    if (source) {
        mContext->weights->streamed[name] = std::move(source);
    }
    addInput(tensor.get(), name);
    return tensor.get();
}
bool CoreMLRuntime::prepareWeights(const CoreMLProgramContext& context) const {
    // Hold lockWeights() until synchronous prediction has consumed this buffer.
    // Each projection may bind different-shaped views onto the same storage.
    size_t count = 0;
    for (const auto& source : context.weights->streamed) {
        count += source.second->elements;
    }
    if (count > INT_MAX) {
        return false;
    }
    if (mWeightScratch.size() < count) {
        mWeightScratch.reset(static_cast<int>(count));
    }
    if (!mWeightScratch.get()) {
        return false;
    }
    auto ptr = mWeightScratch.get();
    for (const auto& entry : context.weights->streamed) {
        auto tensor = context.weightViews.at(entry.first);
        tensor->buffer().host = reinterpret_cast<uint8_t*>(ptr);
        entry.second->decode(ptr);
        ptr += entry.second->elements;
    }
    return true;
}
Backend* CoreMLRuntime::onCreate(const BackendConfig*, Backend*) const {
    return new CoreMLBackend(this);
}
CoreMLRuntime::CompilerType CoreMLRuntime::onGetCompilerType() const {
    return lowMemory() ? Compiler_Geometry : Compiler_Origin;
}
void CoreMLRuntime::onGabageCollect(int level) {
    if (level > 1) {
        std::lock_guard<std::mutex> lock(mCacheMutex);
        for (auto it = mPrograms.begin(); it != mPrograms.end();) {
            if (it->second.program.use_count() == 1) {
                it = mPrograms.erase(it);
            } else {
                ++it;
            }
        }
        std::lock_guard<std::mutex> weightLock(mWeightMutex);
        for (auto it = mWeights.begin(); it != mWeights.end();) {
            if (it->second.expired()) {
                it = mWeights.erase(it);
            } else {
                ++it;
            }
        }
        if (mWeights.empty()) {
            auto executionLock = lockWeights();
            mWeightScratch.release();
        }
    }
}
std::shared_ptr<CoreMLExecutorWrapper> CoreMLRuntime::getProgram(CoreML__Specification__Model* model,
                                                                 const std::string& weightsPath,
                                                                 const std::shared_ptr<std::vector<uint8_t>>& weights,
                                                                 bool forceFileCompilation) const {
    std::string key(core_ml__specification__model__get_packed_size(model), '\0');
    core_ml__specification__model__pack(model, reinterpret_cast<uint8_t*>(&key[0]));
    key.push_back(forceFileCompilation ? 1 : 0);
    if (weights) {
        CC_SHA256_CTX hash;
        CC_SHA256_Init(&hash);
        CC_SHA256_Update(&hash, weights->data(), static_cast<CC_LONG>(weights->size()));
        unsigned char digest[CC_SHA256_DIGEST_LENGTH];
        CC_SHA256_Final(digest, &hash);
        key.append(reinterpret_cast<const char*>(digest), sizeof(digest));
    } else if (!weightsPath.empty()) {
        FILE* file = fopen(weightsPath.c_str(), "rb");
        if (!file) {
            return nullptr;
        }
        CC_SHA256_CTX hash;
        CC_SHA256_Init(&hash);
        char buffer[65536];
        size_t count;
        while ((count = fread(buffer, 1, sizeof(buffer), file))) {
            CC_SHA256_Update(&hash, buffer, count);
        }
        bool failed = ferror(file);
        fclose(file);
        if (failed) {
            return nullptr;
        }
        unsigned char digest[CC_SHA256_DIGEST_LENGTH];
        CC_SHA256_Final(digest, &hash);
        key.append(reinterpret_cast<const char*>(digest), sizeof(digest));
    }
    std::lock_guard<std::mutex> lock(mCacheMutex);
    auto iter = mPrograms.find(key);
    if (iter != mPrograms.end()) {
        ++mCacheHits;
        iter->second.lastUse = ++mClock;
        return iter->second.program;
    }
    auto started = std::chrono::steady_clock::now();
    std::shared_ptr<CoreMLExecutorWrapper> program(new CoreMLExecutorWrapper(mPrecision));
    if (!program->compileModel(model, weightsPath, weights, forceFileCompilation)) {
        return nullptr;
    }
    ++mProgramLoads;
    if (program->compiledFromSource()) {
        ++mCompilations;
    }
    if (std::getenv("MNN_COREML_VERBOSE")) {
        MNN_PRINT(
            "[CoreML] loaded program #%zu: %zu bytes, %zu inputs, %zu states, source_compilations=%zu "
            "prepare_ms=%.3f\n",
            mProgramLoads, key.size(), model->description->n_input, model->description->n_state, mCompilations,
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count());
    }
    // Bound retained cache entries; executions retain their active program.
    if (mPrograms.size() >= 64) {
        auto oldest = std::min_element(
            mPrograms.begin(), mPrograms.end(),
            [](const std::pair<const std::string, CacheEntry>& a, const std::pair<const std::string, CacheEntry>& b) {
                return a.second.lastUse < b.second.lastUse;
            });
        mPrograms.erase(oldest);
    }
    mPrograms.emplace(std::move(key), CacheEntry{program, ++mClock});
    return program;
}
size_t CoreMLRuntime::compilationCount() const {
    std::lock_guard<std::mutex> lock(mCacheMutex);
    return mCompilations;
}
size_t CoreMLRuntime::cacheHitCount() const {
    std::lock_guard<std::mutex> lock(mCacheMutex);
    return mCacheHits;
}
struct CoreMLBackendCreator : RuntimeCreator {
    Runtime* onCreate(const Backend::Info& info) const override { return new CoreMLRuntime(info); }
    bool onValid(Backend::Info&) const override { return true; }
};
void registerCoreMLRuntimeCreator() {
    if (isAvailable()) {
        registerCoreMLOps();
        MNNInsertExtraRuntimeCreator(MNN_FORWARD_NN, new CoreMLBackendCreator, false);
    }
}
} // namespace MNN
