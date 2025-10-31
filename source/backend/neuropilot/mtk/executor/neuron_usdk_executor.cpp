#include <MNN/MNNDefine.h>
#include "executor/neuron_usdk_executor.h"

#include "backend/api/neuron/NeuronAdapter.h"
#include "backend/api/neuron/NeuronAdapterShim.h"
#include "backend/api/neuron/Types.h"
#include "common/cpp11_compat.h"
#include "common/file_mem_mapper.h"
#include "common/logging.h"
#include "common/scope_profiling.h"
#include "executor/allocator.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <iostream>
#include <string>

#define RESTORE_DLA_EXTENSION_OPERATION_TYPE 0x0000
#define RESTORE_DLA_EXTENSION_NAME           "com.mediatek.compiled_network"

namespace mtk {

void NeuronUsdkExecutor::createNeuronMemory(IOBuffer& ioBuf) {
    if (!ioBuf.isAllocated() || ioBuf.neuronMemory != nullptr) {
        return; // Do nothing
    }

    if (ioBuf.ahwbHandle != nullptr) {
        NeuronMemory_createFromAHardwareBuffer(ioBuf.ahwbHandle, &ioBuf.neuronMemory);
    } else if (ioBuf.fd != -1) {
        NeuronMemory_createFromFd(
            ioBuf.sizeBytes, PROT_READ | PROT_WRITE, ioBuf.fd, 0, &ioBuf.neuronMemory);
    } else {
        LOG(ERROR) << "Failed to create neuron memory from invalid/uninitialized buffer.";
    }
    mCreatedNeuronMems.push_back(ioBuf.neuronMemory);
}

std::unique_ptr<Allocator> NeuronUsdkExecutor::createMemoryAllocator() {
    return cpp11_compat::make_unique<AhwBufferAllocator>();
}

void NeuronUsdkExecutor::initialize() {
    initAllocator();
    if (isSharedWeightsUsed()) {
        reserveInputBuffer(mInputSize);
        loadSharedWeights(mInputSize);
    }
    Executor::initialize();
    createUsdkNeuronMemory();
}

bool NeuronUsdkExecutor::loadDla(const void* buffer, const size_t size, UsdkRuntime* runtime) {
    // NOTE: DLA init via USDK requires knowing the model IO count.
    // So user needs to call setNumInputs() and setNumOutputs() before calling initialize().
    CHECK(getNumInputs() > 0 && getNumOutputs() > 0)
        << "NeuronUsdkExecutor requires manual setting of model IO count before initialization.";

    auto& model = runtime->model;
    auto& compilation = runtime->compilation;
    auto& execution = runtime->execution;

    int err = NeuronModel_create(&model);

    std::vector<uint32_t> inputNode;
    std::vector<uint32_t> outputNode;

    constexpr int32_t dummyType = NEURON_TENSOR_QUANT16_SYMM;
    constexpr float dummyScale = 0.0f;
    constexpr int32_t dummyZeroPoint = 0;
    constexpr uint32_t dummyDimCount = 4;
    constexpr uint32_t dummyDim[4] = {0, 0, 0, 0};

    for (int i = 0; i < getNumInputs(); i++) {
        NeuronOperandType tensorInputType;
        tensorInputType.type = dummyType;
        tensorInputType.scale = dummyScale;
        tensorInputType.zeroPoint = dummyZeroPoint;
        tensorInputType.dimensionCount = dummyDimCount;
        tensorInputType.dimensions = dummyDim; // shape of each input

        err |= NeuronModel_addOperand(model, &tensorInputType);
        inputNode.emplace_back(i);
    }

    auto getDlaExtOperandType = [] {
        NeuronRuntimeVersion version;
        Neuron_getVersion(&version);
        LOG(DEBUG) << "Neuron version: " << int(version.major) << "." << int(version.minor) << "."
                   << int(version.patch);
        if (version.major >= 8)
            return 0x0200;
        else
            return 0x0100;
    };

    int32_t operandType = 0;
    const uint16_t network_operand_restore_data = getDlaExtOperandType();
    const char* extensionRestroeCompiledNetwork = RESTORE_DLA_EXTENSION_NAME;
    err |= NeuronModel_getExtensionOperandType(
        model, extensionRestroeCompiledNetwork, network_operand_restore_data, &operandType);

    NeuronOperandType extenOperandType;
    extenOperandType.type = operandType;
    extenOperandType.scale = dummyScale;
    extenOperandType.zeroPoint = dummyZeroPoint;
    extenOperandType.dimensionCount = 0;

    err |= NeuronModel_addOperand(model, &extenOperandType);
    inputNode.emplace_back(inputNode.size());

    for (int i = 0; i < getNumOutputs(); i++) {
        NeuronOperandType tensorOutputType;
        tensorOutputType.type = dummyType;
        tensorOutputType.scale = dummyScale;
        tensorOutputType.zeroPoint = dummyZeroPoint;
        tensorOutputType.dimensionCount = dummyDimCount;
        tensorOutputType.dimensions = dummyDim;

        err |= NeuronModel_addOperand(model, &tensorOutputType);
        outputNode.emplace_back(i + inputNode.size());
    }

    if (err != NEURON_NO_ERROR) {
        LOG(ERROR) << "addOperand fail";
        return false;
    }
    err |= NeuronModel_setOperandValue(model, inputNode.back(), buffer, size);

    int32_t operationType = 0;
    const uint16_t network_operation_type_restore = RESTORE_DLA_EXTENSION_OPERATION_TYPE;
    err |= NeuronModel_getExtensionOperationType(
        model, extensionRestroeCompiledNetwork, network_operation_type_restore, &operationType);

    if (err != NEURON_NO_ERROR) {
        LOG(ERROR) << "get ExtensionOperationType fail";
        return false;
    }

    // Add extension operation
    err |= NeuronModel_addOperation(model, static_cast<NeuronOperationType>(operationType),
                                    inputNode.size(), inputNode.data(), outputNode.size(),
                                    outputNode.data());

    if (err != NEURON_NO_ERROR) {
        LOG(ERROR) << "get addOperation fail";
        return false;
    }

    // Identify input and output
    err |= NeuronModel_identifyInputsAndOutputs(
        model, inputNode.size() - 1, inputNode.data(), outputNode.size(), outputNode.data());

    err |= NeuronModel_finish(model);

    if (err != NEURON_NO_ERROR) {
        LOG(ERROR) << "get model_finish fail";
        return false;
    }

    if (NeuronCompilation_createWithOptions(model, &compilation, kOptions.c_str())
        != NEURON_NO_ERROR) {
        LOG(ERROR) << "NeuronCompilation_create fail";
        return false;
    };

    NeuronCompilation_setPriority(compilation, NEURON_PRIORITY_HIGH);
    NeuronCompilation_setPreference(compilation, NEURON_PREFER_SUSTAINED_SPEED);

    if (!kOptions.empty()) {
        NeuronCompilation_setOptimizationString(compilation, kOptions.c_str());
    }

    // NOTE: NeuronCompilation_finish is currently not thread-safe.
    // However, this is where the actual DLA reading happens.
    // So init without weight sharing will not get much benefit from mult-threading.
    {
        std::lock_guard<std::mutex> lock(mMutex);
        if (NeuronCompilation_finish(compilation) != NEURON_NO_ERROR) {
            LOG(ERROR) << "NeuronCompilation_finish fail";
            return false;
        };
    }

    if (NeuronExecution_create(compilation, &execution) != NEURON_NO_ERROR) {
        LOG(ERROR) << "NeuronExecution_create fail";
        return false;
    };
    if (NeuronExecution_setBoostHint(execution, 100) != NEURON_NO_ERROR) {
        LOG(ERROR) << "NeuronExecution_setBoostHint fail";
        return false;
    };

    return true;
}

void NeuronUsdkExecutor::runInferenceImpl() {
    DLOG_FUNC_LATENCY(ms)
    NeuronExecution_compute(getUsdkRuntime()->execution);
}

void NeuronUsdkExecutor::createUsdkNeuronMemory() {
    for (auto& ioBuf : mInputs) {
        createNeuronMemory(ioBuf);
    }
    for (auto& ioBuf : mOutputs) {
        createNeuronMemory(ioBuf);
    }
}

void NeuronUsdkExecutor::updateModelIO() {
    Executor::updateModelIO();
    createUsdkNeuronMemory(); // To ensure all IO buffers have their own NeuronMemory
}

void* NeuronUsdkExecutor::createRuntime(FileSource modelFile) {
    DLOG_FUNC_LATENCY(ms)

    if (!modelFile.valid()) {
        LOG(FATAL) << "Cannot load file: " << modelFile;
    }

    const auto fileData = modelFile.get();
    const auto dlaBuffer = fileData.first;
    const auto dlaSize = fileData.second;

    // Create runtime
    auto runtime = cpp11_compat::make_unique<UsdkRuntime>();

    if (loadDla(dlaBuffer, dlaSize, runtime.get()) == false) {
        LOG(ERROR) << "load dla fail";
        return nullptr;
    };

    return runtime.release();
}

void NeuronUsdkExecutor::releaseRuntime(void* runtime) {
    DLOG_FUNC_LATENCY(ms)
    // Release neuron adapter
    auto usdkRuntime = reinterpret_cast<UsdkRuntime*>(runtime);
    auto model = usdkRuntime->model;
    auto compilation = usdkRuntime->compilation;
    auto execution = usdkRuntime->execution;
    if (execution != nullptr) {
        NeuronExecution_free(execution);
    }
    if (compilation != nullptr) {
        NeuronCompilation_free(compilation);
    }
    if (model != nullptr) {
        NeuronModel_free(model);
    }
    delete usdkRuntime;
}

void NeuronUsdkExecutor::release() {
    for (auto neuronMemory : mCreatedNeuronMems) {
        NeuronMemory_free(neuronMemory);
    }
    mCreatedNeuronMems.clear();
    Executor::release();
}

void NeuronUsdkExecutor::resetIORegistration() {
    mRegisteredInputs.clear();
    mRegisteredOutputs.clear();
}

void NeuronUsdkExecutor::registerRuntimeInputsImpl() {
    DLOG_FUNC_LATENCY(ms)
    mRegisteredInputs.resize(getNumInputs());

#define NEURON_USDK_SET_INPUT(inputIdx, ioBuf, size) \
    NeuronExecution_setInputFromMemory(              \
        getUsdkRuntime()->execution, inputIdx, NULL, ioBuf.neuronMemory, 0, size);

    for (int i = 0; i < this->getNumInputs(); i++) {
        const auto sizeAllocated = this->getInputBufferSizeBytes(i);
        const auto sizeRequired = this->getModelInputSizeBytes(i);
        if (sizeAllocated < sizeRequired) {
            LOG(ERROR) << "Insufficient buffer allocated for Input[" << i << "]: Allocated "
                       << sizeAllocated << " but need " << sizeRequired;
        }
        auto& curInput = this->getInput(i);
        if (mRegisteredInputs[i] == curInput.neuronMemory) {
            continue; // Already registered, so we can skip this input
        }
        // import_forever requires the full allocated size during the first call to set input/output
        NEURON_USDK_SET_INPUT(i, curInput, sizeAllocated)
        mRegisteredInputs[i] = curInput.neuronMemory; // Mark registered
    }
#undef NEURON_USDK_SET_INPUT
}

void NeuronUsdkExecutor::registerRuntimeOutputsImpl() {
    DLOG_FUNC_LATENCY(ms)
    mRegisteredOutputs.resize(getNumOutputs());

#define NEURON_USDK_SET_OUTPUT(outputIdx, ioBuf, size) \
    NeuronExecution_setOutputFromMemory(               \
        getUsdkRuntime()->execution, outputIdx, NULL, ioBuf.neuronMemory, 0, size);

    for (int i = 0; i < this->getNumOutputs(); i++) {
        const auto sizeAllocated = this->getOutputBufferSizeBytes(i);
        const auto sizeRequired = this->getModelOutputSizeBytes(i);
        if (sizeAllocated < sizeRequired) {
            LOG(ERROR) << "Insufficient buffer allocated for Output[" << i << "]: Allocated "
                       << sizeAllocated << " but need " << sizeRequired;
        }
        auto& curOutput = this->getOutput(i);
        if (mRegisteredOutputs[i] == curOutput.neuronMemory) {
            continue; // Already registered, so we can skip this output
        }
        // import_forever requires the full allocated size during the first call to set input/output
        NEURON_USDK_SET_OUTPUT(i, this->getOutput(i), sizeAllocated)
        mRegisteredOutputs[i] = curOutput.neuronMemory; // Mark registered
    }
#undef NEURON_USDK_SET_OUTPUT
}

void NeuronUsdkExecutor::setRuntimeOffsetedInput(const size_t index, const size_t offset) {
    const auto& ioBuf = this->getInput(index);
    NeuronExecution_setInputFromMemory(
        getUsdkRuntime()->execution, index, NULL, ioBuf.neuronMemory, offset, ioBuf.usedSizeBytes);
}

void NeuronUsdkExecutor::setRuntimeOffsetedOutput(const size_t index, const size_t offset) {
    const auto& ioBuf = this->getOutput(index);
    NeuronExecution_setOutputFromMemory(
        getUsdkRuntime()->execution, index, NULL, ioBuf.neuronMemory, offset, ioBuf.usedSizeBytes);
}

size_t NeuronUsdkExecutor::getRuntimeNumInputs() const {
    size_t numInputs = 0;
    auto compilation = getUsdkRuntime()->compilation;
    size_t size;
    while (true) {
        if (NeuronCompilation_getInputPaddedSize(compilation, numInputs, &size) == NEURON_BAD_DATA)
            break;
        numInputs++;
    }
    return numInputs;
}

size_t NeuronUsdkExecutor::getRuntimeNumOutputs() const {
    size_t numOutputs = 0;
    auto compilation = getUsdkRuntime()->compilation;
    size_t size;
    while (true) {
        if (NeuronCompilation_getOutputPaddedSize(compilation, numOutputs, &size)
            == NEURON_BAD_DATA)
            break;
        numOutputs++;
    }
    return numOutputs;
}

size_t NeuronUsdkExecutor::getRuntimeInputSizeBytes(const size_t index) const {
    // NOTE: Assume user model is always with suppress-io
    size_t inputSizeBytes;

    auto compilation = getUsdkRuntime()->compilation;
    NeuronCompilation_getInputPaddedSize(compilation, index, &inputSizeBytes);

    // clang-format off
    uint32_t dims[4];
    NeuronCompilation_getInputPaddedDimensions(compilation, index, dims);
    LOG(DEBUG) << this->getModelName() << ":\n Input[" << index << "] Size (padded): "
               << inputSizeBytes << "\n Input[" << index << "] Dims (padded): "
               << dims[0] << "x"
               << dims[1] << "x"
               << dims[2] << "x"
               << dims[3];
    // clang-format on
    // FIXME: Remove below workaround once "shared weights 0 size bug" is resolved in Neuron Adapter
    if (inputSizeBytes == 0)
        return dims[3];
    return inputSizeBytes;
}

size_t NeuronUsdkExecutor::getRuntimeOutputSizeBytes(const size_t index) const {
    // NOTE: Assume user model is always with suppress-io
    size_t inputSizeBytes;

    auto compilation = getUsdkRuntime()->compilation;
    NeuronCompilation_getOutputPaddedSize(compilation, index, &inputSizeBytes);

    // clang-format off
    uint32_t dims[4];
    NeuronCompilation_getOutputPaddedDimensions(compilation, index, dims);
    LOG(DEBUG) << this->getModelName() << ":\n Input[" << index << "] Size (padded): "
               << inputSizeBytes << "\n Input[" << index << "] Dims (padded): "
               << dims[0] << "x"
               << dims[1] << "x"
               << dims[2] << "x"
               << dims[3];
    // clang-format on
    return inputSizeBytes;
}

void NeuronUsdkExecutor::getRuntimeInputShape(const size_t index, uint32_t* shape) const {
    uint32_t tmpShape[kDimensionSize];
    NeuronCompilation_getInputPaddedDimensions(getUsdkRuntime()->compilation, index, tmpShape);
    std::memcpy(shape, tmpShape, sizeof(tmpShape));
}

void NeuronUsdkExecutor::getRuntimeOutputShape(const size_t index, uint32_t* shape) const {
    uint32_t tmpShape[kDimensionSize];
    NeuronCompilation_getOutputPaddedDimensions(getUsdkRuntime()->compilation, index, tmpShape);
    std::memcpy(shape, tmpShape, sizeof(tmpShape));
}
std::mutex NeuronUsdkExecutor::mMutex;

} // namespace mtk
