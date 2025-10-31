#include <MNN/MNNDefine.h>
#include "executor/executor.h"

#include "backend/api/neuron/Types.h"
#include "backend/backend.h"
#include "common/cpp11_compat.h"
#include "common/logging.h"
#include "common/scope_profiling.h"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include <dirent.h>

namespace fs = mtk::cpp11_compat::fs;

namespace mtk {

#ifdef DIRECT_IO_SHARED_WEIGHTS
static constexpr bool kSharedWeightsUseDirectIO = true;
#else
static constexpr bool kSharedWeightsUseDirectIO = false;
#endif

void Executor::initialize() {
    if (isInitialized()) {
        return;
    }
    DLOG_FUNC_LATENCY(ms)
    initRuntimes();           // Need to know the number of IO here for usdk
    initModelIOInfo();        // Get number of IO, and their sizes in bytes
    assignBufferSizesToMax(); // Set each IO buffer to the possible max size before allocation
    preInitBufferProcess();   // Optional preprocessing after model IO info is obtained.
    initAllocator();          // Initialize allocator if not yet done so for IO buffer allocation
    initBuffer();             // Allocate buffer based on the model IO info
    postInitBufferProcess();  // Optional postprocessing after model IO buffers have been allocated.
    mIsInitialized = true;
}

void Executor::release() {
    DLOG_FUNC_LATENCY(ms)
    releaseRuntimes();
    releaseBuffer();
    mIsInitialized = false;
}

Executor::~Executor() {
    // Executor::release() must be called before reaching base Executor dtor because release() needs
    // to make virtual function calls.
}

bool Executor::isSharedWeightsUsed() const {
    return !kSharedWeights.empty();
}

size_t Executor::numSharedWeightsUsed() const {
    return kSharedWeights.size();
}

void Executor::loadSharedWeights(const size_t firstSharedWeightsInputIdx) {
    if (!isSharedWeightsUsed()) {
        return; // FMS shared weights not used
    }

    DLOG_FUNC_LATENCY(ms)

    const auto numSharedWeights = numSharedWeightsUsed();

    for (size_t swIdx = 0; swIdx < numSharedWeights; swIdx++) {
        auto& swBuffer = getInput(firstSharedWeightsInputIdx + swIdx); // Shared weights IO buffer

        if (kSharedWeights.isPreloaded(swIdx)) {
            LOG(DEBUG) << "Executor: Using preloaded shared weights buffer for index " << swIdx;
            swBuffer = kSharedWeights.buffers[swIdx];
            continue;
        }

        const auto& swFile = kSharedWeights.files[swIdx];
        const auto swSize = swFile.getSize();
        swBuffer.sizeBytes = swSize;
        swBuffer.usedSizeBytes = swSize;

        auto& allocator = getAllocator();
        if (!allocator.allocateMemory(swBuffer)) {
            LOG(ERROR) << "Failed to allocate memory for shared weights on input["
                       << firstSharedWeightsInputIdx << "] " << "with size=" << swSize;
        }

        if (kSharedWeightsUseDirectIO) {
            swFile.directRead(swBuffer.buffer, swSize);
        } else {
            std::memcpy(swBuffer.buffer, swFile.getData(), swSize);
            kSharedWeights.files[swIdx].hint_release();
        }
    }
}

void Executor::updateModelIO() {
    resetIORegistration();  // Force re-register all IOs
    preInitBufferProcess(); // Optional preprocessing after model IO info is obtained.

    // Update actual used IO sizes by the model
    const size_t numInputs = getRuntimeNumInputs();
    const size_t numOutputs = getRuntimeNumOutputs();
    if (numInputs != getNumInputs()) {
        LOG(WARN) << "updateModelIO: Existing num inputs (" << getNumInputs()
                  << ") != new num inputs (" << numInputs << ").";
    }
    if (numOutputs != getNumOutputs()) {
        LOG(WARN) << "updateModelIO: Existing num outputs (" << getNumOutputs()
                  << ") != new num outputs (" << numOutputs << ").";
    }
    mInputs.resize(numInputs);
    for (size_t inputIdx = 0; inputIdx < numInputs; inputIdx++) {
        auto& sizeAllocated = mInputs[inputIdx].sizeBytes;
        auto& sizeRequired = mInputs[inputIdx].usedSizeBytes;
        const auto before = sizeRequired;

        sizeRequired = getRuntimeInputSizeBytes(inputIdx); // Update
        if (sizeAllocated < sizeRequired) {
            LOG(WARN) << "updateModelIO: Insufficient buffer size for input[" << inputIdx << "]. "
                      << "Requires " << sizeRequired << " but only allocated " << sizeAllocated;
        }
        if (before != sizeRequired) {
            LOG(DEBUG) << "Update Input[" << inputIdx << "] size: " << before << " -> "
                       << sizeRequired;
        }
    }
    mOutputs.resize(numOutputs);
    for (size_t outputIdx = 0; outputIdx < numOutputs; outputIdx++) {
        auto& sizeAllocated = mOutputs[outputIdx].sizeBytes;
        auto& sizeRequired = mOutputs[outputIdx].usedSizeBytes;
        const auto before = sizeRequired;

        sizeRequired = getRuntimeOutputSizeBytes(outputIdx); // Update
        if (sizeAllocated < sizeRequired) {
            LOG(WARN) << "updateModelIO: Insufficient buffer size for output[" << outputIdx << "]. "
                      << "Requires " << sizeRequired << " but only allocated " << sizeAllocated;
        }
        if (before != sizeRequired) {
            LOG(DEBUG) << "Update Output[" << outputIdx << "] size: " << before << " -> "
                       << sizeRequired;
        }
    }
}

bool Executor::isInitialized() const {
    return mIsInitialized;
}

void Executor::registerRuntimeIO() {
    registerRuntimeInputs();
    registerRuntimeOutputs();
    DLOG_FUNC_EXIT
}

void Executor::registerRuntimeInputs() {
    // Ensure all input buffers are allocated/initialized
    auto isNotAllocated = [](const IOBuffer& ioBuf) { return !ioBuf.isAllocated(); };
    const auto it = std::find_if(mInputs.begin(), mInputs.end(), isNotAllocated);
    if (it != mInputs.end()) {
        const size_t notAllocatedIdx = std::distance(mInputs.begin(), it);
        LOG(FATAL) << "[registerRuntimeInputs] Attempting to register an uninitialized input "
                   << "buffer (index=" << notAllocatedIdx << ")";
    }
    CHECK_GT(getNumInputs(), 0) << "[registerRuntimeInputs] No model input allocated. "
                                   "Please check if the model has been loaded properly.";
    registerRuntimeInputsImpl();
    mIsInputRegistered = true;
}

void Executor::registerRuntimeOutputs() {
    // Ensure all output buffers are allocated/initialized
    auto isNotAllocated = [](const IOBuffer& ioBuf) { return !ioBuf.isAllocated(); };
    const auto it = std::find_if(mOutputs.begin(), mOutputs.end(), isNotAllocated);
    if (it != mOutputs.end()) {
        const size_t notAllocatedIdx = std::distance(mOutputs.begin(), it);
        LOG(FATAL) << "[registerRuntimeOutputs] Attempting to register an uninitialized "
                   << "output buffer (index=" << notAllocatedIdx << ")";
    }
    CHECK_GT(getNumOutputs(), 0) << "[registerRuntimeOutputs] No model output allocated. "
                                    "Please check if the model has been loaded properly.";
    registerRuntimeOutputsImpl();
    mIsOutputRegistered = true;
}

template <typename T>
void Executor::runInference(const std::vector<T>& input) {
    runInference(input.data(), input.size() * sizeof(T));
    DLOG_FUNC_EXIT
}

void Executor::runInference(const void* input, size_t inputSize) {
    requiresInit();
    setModelInput(input, inputSize);
    registerRuntimeInputs();
    runInferencePrologue();
    runInference();
    runInferenceEpilogue();
    DLOG_FUNC_EXIT
}

void Executor::runInference() {
    requiresInit();
    if (!mIsInputRegistered) {
        // At least one input may require registering to backend runtime
        registerRuntimeInputs();
    }
    if (!mIsOutputRegistered) {
        // At least one output may require registering to backend runtime
        registerRuntimeOutputs();
    }
    runInferenceImpl();
}

void Executor::setNumInputs(const size_t numInputs) {
    const auto oldNumInputs = getNumInputs();
    if (oldNumInputs > numInputs) {
        LOG(WARN) << "Reducing the number of inputs from " << oldNumInputs << " to " << numInputs;
    }
    mInputs.resize(numInputs);
}

void Executor::setNumOutputs(const size_t numOutputs) {
    const auto oldNumOutputs = getNumOutputs();
    if (oldNumOutputs > numOutputs) {
        LOG(WARN) << "Reducing the number of Outputs from " << oldNumOutputs << " to "
                  << numOutputs;
    }
    mOutputs.resize(numOutputs);
}

// Either assign buffer pointer or copy buffer content. If the index is out of input buffer vector,
// the buffer will be set aside to the preallocate buffer container.
void Executor::setModelInput(const IOBuffer& buffer, const size_t index) {
    // If current buffer (mInputs[index]) is not allocated then copy the buffer POINTER,
    // otherwise copy the content from buffer to current buffer.

    if (index >= mInputs.size()) {
        // initModelIOInfo() has not yet been called
        mInputs.resize(index + 1);
    }
    auto& curBuffer = mInputs[index];
    if (!curBuffer.isAllocated()) {
        // Not yet allocated, so share the buffer
        curBuffer = buffer;
        mIsInputRegistered = false;
    } else {
        // Already allocated, do memcpy
        setModelInput(buffer.buffer, buffer.sizeBytes, index);
    }
    DLOG_FUNC_EXIT
}

// Copy buffer content
void Executor::setModelInput(const void* buffer, const size_t sizeBytes, const size_t index) {
    auto& input = getInput(index);
    if (input.sizeBytes < sizeBytes) {
        LOG(ERROR) << "[setModelInput] Insufficient buffer size (" << input.sizeBytes
                   << ") to hold the required target data size (" << sizeBytes << ")";
        return;
    }
    auto& curInputSizeBytes = input.usedSizeBytes;
    if (input.buffer == buffer && curInputSizeBytes == sizeBytes) {
        return; // The same buffer so skip it
    }
    std::memcpy(input.buffer, buffer, sizeBytes);
    if (curInputSizeBytes != sizeBytes) {
        LOG(DEBUG) << "[setModelInput]: Update model input[" << index << "] size bytes from "
                   << curInputSizeBytes << " to " << sizeBytes;
        curInputSizeBytes = sizeBytes;
    }
    mIsInputRegistered = false;
    DLOG_FUNC_EXIT
}

void Executor::reserveInputBuffer(const size_t index) {
    mReservedInputBuffers.insert(index);
}

void Executor::reserveOutputBuffer(const size_t index) {
    mReservedOutputBuffers.insert(index);
}

size_t Executor::getNumInputs() const {
    return mInputs.size();
}

size_t Executor::getNumOutputs() const {
    return mOutputs.size();
}

//=================//
// Get model input //
//=================//

const IOBuffer& Executor::getInput(const size_t index) const {
    CHECK_LT(index, getNumInputs()) << "getInput(): Index out of range.";
    return mInputs[index];
}

IOBuffer& Executor::getInput(const size_t index) {
    CHECK_LT(index, getNumInputs()) << "getInput(): Index out of range.";
    return mInputs[index];
}

void* Executor::getInputBuffer(const size_t index) {
    return getInput(index).buffer;
}

const void* Executor::getInputBuffer(const size_t index) const {
    return getInput(index).buffer;
}

size_t Executor::getInputBufferSizeBytes(const size_t index) const {
    return getInput(index).sizeBytes; // Actual allocated buffer size
}

size_t Executor::getModelInputSizeBytes(const size_t index) const {
    return getInput(index).usedSizeBytes; // Actual size used by the model
}

//==================//
// Get model output //
//==================//

const IOBuffer& Executor::getOutput(const size_t index) const {
    CHECK_LT(index, getNumOutputs()) << "getOutput(): Index out of range.";
    return mOutputs[index];
}

IOBuffer& Executor::getOutput(const size_t index) {
    CHECK_LT(index, getNumOutputs()) << "getOutput(): Index out of range.";
    return mOutputs[index];
}

void* Executor::getOutputBuffer(const size_t index) {
    return getOutput(index).buffer;
}

const void* Executor::getOutputBuffer(const size_t index) const {
    return getOutput(index).buffer;
}

size_t Executor::getOutputBufferSizeBytes(const size_t index) const {
    return getOutput(index).sizeBytes; // Actual allocated buffer size
}

size_t Executor::getModelOutputSizeBytes(const size_t index) const {
    return getOutput(index).usedSizeBytes; // Actual size used by the model
}

//==================//
// Model IO linkage //
//==================//

void Executor::linkModelIO(const size_t inputIndex, const size_t outputIndex) {
    mModelInToOutIndexLinks.emplace(inputIndex, outputIndex);
    DLOG_FUNC_EXIT
}

void Executor::setModelIOLink(const std::unordered_map<size_t, size_t>& modelIOLinks) {
    mModelInToOutIndexLinks = modelIOLinks;
    DLOG_FUNC_EXIT
}

bool Executor::inputHasLinkToOutput(const size_t inputIndex) const {
    // return (inputIndex in mModelInToOutIndexLinks)
    return mModelInToOutIndexLinks.find(inputIndex) != mModelInToOutIndexLinks.end();
}

size_t Executor::getLinkedOutputIndex(const size_t inputIndex) {
    DLOG_FUNC_EXIT
    if (inputHasLinkToOutput(inputIndex)) {
        return mModelInToOutIndexLinks[inputIndex];
    }
    return kInvalidIndex;
}

void Executor::initModelIOInfo() {
    // Inputs
    const size_t numInputs = getRuntimeNumInputs();
    setNumInputs(numInputs);
    LOG(DEBUG) << "numInputs = " << numInputs;
    for (size_t inputIdx = 0; inputIdx < numInputs; inputIdx++) {
        size_t inputSize = getRuntimeInputSizeBytes(inputIdx);
        auto& input = mInputs[inputIdx];
        input.sizeBytes = inputSize; // Initialize buffer size to input size
        input.usedSizeBytes = inputSize;
    }
    if (numInputs == 0) {
        LOG(FATAL) << "[Executor] Failed to get model input info.";
    }

    // Outputs
    const size_t numOutputs = getRuntimeNumOutputs();
    setNumOutputs(numOutputs);
    LOG(DEBUG) << "numOutputs = " << numOutputs;
    for (size_t outputIdx = 0; outputIdx < numOutputs; outputIdx++) {
        size_t outputSize = getRuntimeOutputSizeBytes(outputIdx);
        auto& output = mOutputs[outputIdx];
        output.sizeBytes = outputSize; // Initialize buffer size to output size
        output.usedSizeBytes = outputSize;
    }
    if (numOutputs == 0) {
        LOG(FATAL) << "[Executor] Failed to get model output info.";
    }
    DLOG_FUNC_EXIT
}

//=================================//
// Buffer Initialization & Release //
//=================================//

void Executor::initAllocator() {
    if (!mAllocator) {
        mAllocator = createMemoryAllocator();
    }
}

Allocator& Executor::getAllocator() {
    return *mAllocator;
}

void Executor::initBuffer() {
    DLOG_FUNC_LATENCY(ms)
    resetIORegistration();

    const size_t numInputs = getNumInputs();
    const size_t numOutputs = getNumOutputs();
    if (numInputs == 0 || numOutputs == 0) {
        LOG(FATAL) << "Attempt to init buffer before model IO info is retrieved.";
    }

    auto& allocator = getAllocator();

    auto isInputBufferReserved = [&](const size_t index) {
        return mReservedInputBuffers.find(index) != mReservedInputBuffers.end();
    };

    auto isOutputBufferReserved = [&](const size_t index) {
        return mReservedOutputBuffers.find(index) != mReservedOutputBuffers.end();
    };

    for (size_t outputIdx = 0; outputIdx < numOutputs; outputIdx++) {
        auto& output = getOutput(outputIdx);
        if (output.isAllocated()) {
            LOG(DEBUG) << "Init Buffer: Reusing preallocated output buffer " << outputIdx;
            continue;
        }
        if (isOutputBufferReserved(outputIdx)) {
            LOG(DEBUG) << "Init Buffer: Skip allocation for reserved output buffer " << outputIdx;
            continue;
        }
        if (!allocator.allocateMemory(output)) {
            LOG(ERROR) << "Failed to allocate memory for output[" << outputIdx << "]";
        }
        LOG(DEBUG) << "Init Buffer: allocating output[" << outputIdx << "]";
    }

    for (size_t inputIdx = 0; inputIdx < numInputs; inputIdx++) {
        auto& input = getInput(inputIdx);
        if (input.isAllocated()) {
            LOG(DEBUG) << "Init Buffer: Reusing preallocated input buffer " << inputIdx;
            continue;
        }
        if (isInputBufferReserved(inputIdx)) {
            LOG(DEBUG) << "Init Buffer: Skip allocation for reserved input buffer " << inputIdx;
            continue;
        }
        if (!inputHasLinkToOutput(inputIdx)) {
            if (!allocator.allocateMemory(input)) {
                LOG(ERROR) << "Failed to allocate memory for input[" << inputIdx << "]";
            }
            LOG(DEBUG) << "Init Buffer: allocating input[" << inputIdx
                       << "] with size=" << input.sizeBytes;
            continue;
        }

        const size_t inputSizeBytes = input.sizeBytes;
        const size_t linkedOutputIdx = getLinkedOutputIndex(inputIdx);
        const auto& linkedOutput = getOutput(linkedOutputIdx);
        const size_t linkedOutputSizeBytes = linkedOutput.sizeBytes;
        if (inputSizeBytes != linkedOutputSizeBytes) {
            LOG(FATAL) << "Init Buffer: Mismatch size between linked input/output! Input["
                       << inputIdx << "].size=" << inputSizeBytes << ", Output[" << linkedOutputIdx
                       << "].size=" << linkedOutputSizeBytes;
        }
        // Reuse the same buffer from output since they are linked
        input = linkedOutput;
        LOG(DEBUG) << "Init Buffer: input[" << inputIdx << "] reuse output[" << linkedOutputIdx
                   << "]";
    }
}

void Executor::releaseBuffer() {
    getAllocator().releaseAll();
    DLOG_FUNC_EXIT
}

//===================//
// Init verification //
//===================//

void Executor::requiresInit() const {
    if (!mIsInitialized) {
        LOG(FATAL) << "Executor is not initialized. Please call initialize().";
    }
}


// Explicit instantiation of runInference for some integral types
template void Executor::runInference<int>(const std::vector<int>&);
template void Executor::runInference<int16_t>(const std::vector<int16_t>&);
template void Executor::runInference<float>(const std::vector<float>&);
template void Executor::runInference<__fp16>(const std::vector<__fp16>&);

} // namespace mtk
