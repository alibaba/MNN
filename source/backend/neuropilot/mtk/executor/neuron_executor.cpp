#include "executor/neuron_executor.h"

#include "backend/neuron_api.h"
#include "common/cpp11_compat.h"
#include "common/file_mem_mapper.h"
#include "common/logging.h"
#include "common/scope_profiling.h"
#include "executor/allocator.h"

namespace mtk {

using namespace backend::neuron_api;

std::unique_ptr<Allocator> NeuronExecutor::createMemoryAllocator() {
    return cpp11_compat::make_unique<DmaBufferAllocator>();
}

void NeuronExecutor::runInferenceImpl() {
    DLOG_FUNC_LATENCY(ms)

    if (!mInferenceEnqueued) {
        CHECK_NEURON_ERROR(fnNeuronRuntime_inferenceEnqueue(this->getRuntime()));
    }
    CHECK_NEURON_ERROR(fnNeuronRuntime_inferenceTrigger(this->getRuntime()));
    mInferenceEnqueued = false;
}

void NeuronExecutor::runInferencePrologue() {
    Executor::runInferencePrologue();
    CHECK_NEURON_ERROR(fnNeuronRuntime_inferenceEnqueue(this->getRuntime()));
    mInferenceEnqueued = true;
}

void* NeuronExecutor::createRuntime(FileSource modelFile) {
    DLOG_FUNC_LATENCY(ms)
    if (!modelFile.valid()) {
        LOG(FATAL) << "Cannot load file: " << modelFile;
    }

    const auto fileData = modelFile.get();
    const auto dlaBuffer = fileData.first;
    const auto dlaSize = fileData.second;

    // NOTE: Starting from NP7, EnvOptions will be ignored.
    EnvOptions envOptions = {.deviceKind = kEnvOptHardware, .MDLACoreOption = MDLACoreMode::Auto};

    // Create runtime
    void* runtime;
    CHECK_NEURON_ERROR(fnNeuronRuntime_create_with_options(
        "--apusys-config \"{ \\\"high_addr\\\": true, \\\"import_forever\\\": true }\"",
        &envOptions, &runtime));

    // Load model
    CHECK_NEURON_ERROR(fnNeuronRuntime_loadNetworkFromBuffer(runtime, dlaBuffer, dlaSize));

    // Set QoS option
    mQosOptions.preference = NEURONRUNTIME_PREFER_PERFORMANCE;
    mQosOptions.boostValue = 100;
    mQosOptions.priority = NEURONRUNTIME_PRIORITY_HIGH;
    mQosOptions.powerPolicy = NEURONRUNTIME_POWER_POLICY_SUSTAINABLE;
    CHECK_NEURON_ERROR(fnNeuronRuntime_setQoSOption(runtime, &mQosOptions));

    return runtime;
}

void NeuronExecutor::releaseRuntime(void* runtime) {
    DLOG_FUNC_LATENCY(ms)
    // Release current runtime
    fnNeuronRuntime_release(runtime);
}

void NeuronExecutor::resetIORegistration() {
    mRegisteredInputFds.clear();
    mRegisteredOutputFds.clear();
}

void NeuronExecutor::registerRuntimeInputsImpl() {
    DLOG_FUNC_LATENCY(ms)
    mRegisteredInputFds.resize(getNumInputs(), NON_ION_FD);

#define NEURON_RUNTIME_SET_INPUT(inputIdx, ioBuf, size) \
    CHECK_NEURON_ERROR(fnNeuronRuntime_setInput(        \
        this->getRuntime(), inputIdx, reinterpret_cast<void*>(ioBuf.buffer), size, {ioBuf.fd}));

    for (int i = 0; i < this->getNumInputs(); i++) {
        const auto sizeAllocated = this->getInputBufferSizeBytes(i);
        const auto sizeRequired = this->getModelInputSizeBytes(i);
        if (sizeAllocated < sizeRequired) {
            LOG(ERROR) << "Insufficient buffer allocated for Input[" << i << "]: Allocated "
                       << sizeAllocated << " but need " << sizeRequired;
        }
        auto& curInput = this->getInput(i);
        if (mRegisteredInputFds[i] == curInput.fd && curInput.fd != NON_ION_FD) {
            continue; // Already registered with ION buffer, so we can skip this input
        }
        // import_forever requires the full allocated size during the first call to set input/output
        NEURON_RUNTIME_SET_INPUT(i, curInput, sizeAllocated)
        mRegisteredInputFds[i] = curInput.fd; // Mark registered
    }
#undef NEURON_RUNTIME_SET_INPUT
}

void NeuronExecutor::registerRuntimeOutputsImpl() {
    DLOG_FUNC_LATENCY(ms)
    mRegisteredOutputFds.resize(getNumOutputs(), NON_ION_FD);

#define NEURON_RUNTIME_SET_OUTPUT(outputIdx, ioBuf, size) \
    CHECK_NEURON_ERROR(fnNeuronRuntime_setOutput(         \
        this->getRuntime(), outputIdx, reinterpret_cast<void*>(ioBuf.buffer), size, {ioBuf.fd}));

    for (int i = 0; i < this->getNumOutputs(); i++) {
        const auto sizeAllocated = this->getOutputBufferSizeBytes(i);
        const auto sizeRequired = this->getModelOutputSizeBytes(i);
        if (sizeAllocated < sizeRequired) {
            LOG(ERROR) << "Insufficient buffer allocated for Output[" << i << "]: Allocated "
                       << sizeAllocated << " but need " << sizeRequired;
        }
        auto& curOutput = this->getOutput(i);
        if (mRegisteredOutputFds[i] == curOutput.fd && curOutput.fd != NON_ION_FD) {
            continue; // Already registered with ION buffer, so we can skip this output
        }
        // import_forever requires the full allocated size during the first call to set input/output
        NEURON_RUNTIME_SET_OUTPUT(i, curOutput, sizeAllocated)
        mRegisteredOutputFds[i] = curOutput.fd; // Mark registered
    }
#undef NEURON_RUNTIME_SET_OUTPUT
}

void NeuronExecutor::setRuntimeOffsetedInput(const size_t index, const size_t offset) {
    const auto& ioBuf = this->getInput(index);
    CHECK_NEURON_ERROR(fnNeuronRuntime_setOffsetedInput(
        this->getRuntime(), index, reinterpret_cast<void*>(ioBuf.buffer),
        this->getModelInputSizeBytes(index), {ioBuf.fd}, offset));
}

void NeuronExecutor::setRuntimeOffsetedOutput(const size_t index, const size_t offset) {
    const auto& ioBuf = this->getOutput(index);
    CHECK_NEURON_ERROR(fnNeuronRuntime_setOffsetedOutput(
        this->getRuntime(), index, reinterpret_cast<void*>(ioBuf.buffer),
        this->getModelOutputSizeBytes(index), {ioBuf.fd}, offset));
}

size_t NeuronExecutor::getRuntimeNumInputs() const {
    size_t numInputs;
    CHECK_NEURON_ERROR(fnNeuronRuntime_getInputNumber(this->getRuntime(), &numInputs));
    return numInputs;
}

size_t NeuronExecutor::getRuntimeNumOutputs() const {
    size_t numOutputs;
    CHECK_NEURON_ERROR(fnNeuronRuntime_getOutputNumber(this->getRuntime(), &numOutputs));
    return numOutputs;
}

size_t NeuronExecutor::getRuntimeInputSizeBytes(const size_t index) const {
    // NOTE: Assume user model is always with suppress-io
    size_t inputSizeBytes;
    CHECK_NP_ERROR(fnNeuronRuntime_getInputPaddedSize(this->getRuntime(), index, &inputSizeBytes));

    // clang-format off
    RuntimeAPIDimensions dims;
    fnNeuronRuntime_getInputPaddedDimensions(this->getRuntime(), index, &dims);
    LOG(DEBUG) << this->getModelName() << ":\n Input[" << index << "] Size (padded): "
               << inputSizeBytes << "\n Input[" << index << "] Dims (padded): "
               << dims.dimensions[0] << "x"
               << dims.dimensions[1] << "x"
               << dims.dimensions[2] << "x"
               << dims.dimensions[3];
    // clang-format on
    return inputSizeBytes;
}

size_t NeuronExecutor::getRuntimeOutputSizeBytes(const size_t index) const {
    // NOTE: Assume user model is always with suppress-io
    size_t outputSizeBytes;
    CHECK_NP_ERROR(
        fnNeuronRuntime_getOutputPaddedSize(this->getRuntime(), index, &outputSizeBytes));

    // clang-format off
    RuntimeAPIDimensions dims;
    fnNeuronRuntime_getOutputPaddedDimensions(this->getRuntime(), index, &dims);
    LOG(DEBUG) << this->getModelName() << ":\n Output[" << index << "] Size (padded): "
               << outputSizeBytes << "\n Output[" << index << "] Dims (padded): "
               << dims.dimensions[0] << "x"
               << dims.dimensions[1] << "x"
               << dims.dimensions[2] << "x"
               << dims.dimensions[3];
    // clang-format on
    return outputSizeBytes;
}

void NeuronExecutor::getRuntimeInputShape(const size_t index, uint32_t* shape) const {
    RuntimeAPIDimensions tmpShape;
    fnNeuronRuntime_getInputPaddedDimensions(this->getRuntime(), index, &tmpShape);
    std::memcpy(shape, tmpShape.dimensions, sizeof(RuntimeAPIDimensions::dimensions));
}

void NeuronExecutor::getRuntimeOutputShape(const size_t index, uint32_t* shape) const {
    RuntimeAPIDimensions tmpShape;
    fnNeuronRuntime_getOutputPaddedDimensions(this->getRuntime(), index, &tmpShape);
    std::memcpy(shape, tmpShape.dimensions, sizeof(RuntimeAPIDimensions::dimensions));
}

} // namespace mtk
