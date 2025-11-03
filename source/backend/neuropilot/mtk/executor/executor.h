#pragma once

#include "common/file_source.h"
#include "executor/allocator.h"
#include "executor/macros.h"
#include "executor/multi_runtime_handler.h"
#include "executor/shared_weights.h"

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace mtk {

class Executor : protected MultiRuntimeHandler {
public:
    explicit Executor(const std::vector<FileSource>& modelFiles, const SharedWeights& sharedWeights,
                      const size_t initBatchSize = 1)
        : MultiRuntimeHandler(modelFiles),
          kSharedWeights(sharedWeights),
          mModelBatchSize(initBatchSize) {}

    explicit Executor(const FileSource& modelFile, const SharedWeights& sharedWeights,
                      const size_t initBatchSize = 1)
        : MultiRuntimeHandler(modelFile),
          kSharedWeights(sharedWeights),
          mModelBatchSize(initBatchSize) {}

    virtual ~Executor();

    virtual void initialize();

    bool isInitialized() const;

    void requiresInit() const;

    // NOTE: Subclasses that override release/createRuntime/releaseRuntime must call release() in
    // its dtor.
    virtual void release();

    void runInference();

    // Convenient function to set input 0 before running
    template <typename T>
    void runInference(const std::vector<T>& input);
    virtual void runInference(const void* input, size_t inputSize);

    // Inference Prologue and Epilogue
    virtual void runInferencePrologue() {}
    virtual void runInferenceEpilogue() {}

    void registerRuntimeIO();

    size_t getBatchSize() const;

    void setNumInputs(const size_t numInputs);
    void setNumOutputs(const size_t numOutputs);

    // Add index argument to (overloaded) setModelInput functions
    // NOTE: registerRuntimeInputs() needs to be called afterwards!
    void setModelInput(const IOBuffer& buffer, const size_t index = 0);
    void setModelInput(const void* buffer, const size_t sizeBytes, const size_t index = 0);

    // Prevent allocation on the reserved input/output buffers.
    // Used when the buffer allocation is handled elsewhere.
    void reserveInputBuffer(const size_t index = 0);
    void reserveOutputBuffer(const size_t index = 0);

    // Get model IO
    IOBuffer& getInput(const size_t index = 0);
    IOBuffer& getOutput(const size_t index = 0);

    const IOBuffer& getInput(const size_t index = 0) const;
    const IOBuffer& getOutput(const size_t index = 0) const;

    void* getInputBuffer(const size_t index = 0);
    void* getOutputBuffer(const size_t index = 0);

    const void* getInputBuffer(const size_t index = 0) const;
    const void* getOutputBuffer(const size_t index = 0) const;

    // Get the actual input/output size used by the model
    size_t getModelInputSizeBytes(const size_t index = 0) const;
    size_t getModelOutputSizeBytes(const size_t index = 0) const;

    // To ensure IO count and sizes are correct, during init or model swapping
    virtual void updateModelIO();

protected:
    size_t getNumInputs() const;
    size_t getNumOutputs() const;

    // Get the allocated input/output buffer size
    size_t getInputBufferSizeBytes(const size_t index) const;
    size_t getOutputBufferSizeBytes(const size_t index) const;

protected:
    // Initialize memory allocator if not yet done so.
    void initAllocator();

    // Get memory allocator, and init allocator if not yet done so.
    Allocator& getAllocator();

    // Optional preprocessing after model IO info is obtained, and before initBuffer is called.
    virtual void preInitBufferProcess() { };

    // Optional postprocessing after model IO buffers have been allocated.
    virtual void postInitBufferProcess() {}

    // Optional step to set IO buffer sizes to their possible max before initBuffer is called.
    virtual void assignBufferSizesToMax() {}

    // Model IO linkage to share the same buffer among a pair of linked input and output
    void linkModelIO(const size_t inputIndex, const size_t outputIndex);
    void setModelIOLink(const std::unordered_map<size_t, size_t>& modelIOLinks);

    // Check if the output will feed back to the input, if so they will reuse the same buffer.
    bool inputHasLinkToOutput(const size_t inputIndex) const;
    size_t getLinkedOutputIndex(const size_t inputIndex);

    // Set buffers to runtime IO by calling registerRuntime*Impl
    void registerRuntimeInputs();
    void registerRuntimeOutputs();

    // Backend executor subclasses need to implement these
    virtual void runInferenceImpl() = 0;
    virtual void resetIORegistration() = 0;
    virtual void registerRuntimeInputsImpl() = 0;
    virtual void registerRuntimeOutputsImpl() = 0;
    virtual void setRuntimeOffsetedInput(const size_t index, const size_t offset) = 0;
    virtual void setRuntimeOffsetedOutput(const size_t index, const size_t offset) = 0;

    virtual size_t getRuntimeNumInputs() const = 0;
    virtual size_t getRuntimeNumOutputs() const = 0;
    virtual size_t getRuntimeInputSizeBytes(const size_t index) const = 0;
    virtual size_t getRuntimeOutputSizeBytes(const size_t index) const = 0;
    virtual void getRuntimeInputShape(uint64_t index, uint32_t* shape) const = 0;
    virtual void getRuntimeOutputShape(uint64_t index, uint32_t* shape) const = 0;

private:
    void initModelIOInfo();

    void initBuffer();

    void releaseBuffer();

    virtual std::unique_ptr<Allocator> createMemoryAllocator() = 0;

    virtual bool canRuntimesCoexist() const override { return isSharedWeightsUsed(); }

protected:
    // Shared weights
    bool isSharedWeightsUsed() const;
    size_t numSharedWeightsUsed() const;
    void loadSharedWeights(const size_t firstSharedWeightsInputIdx);

protected:
    static constexpr size_t kInvalidIndex = static_cast<size_t>(-1);

    // The buffer size will be initialized with the model IO sizes.
    std::vector<IOBuffer> mInputs;
    std::vector<IOBuffer> mOutputs;

    bool mIsInitialized = false;
    bool mIsInputRegistered = false; // Will be set to false when setModelInput is called
    bool mIsOutputRegistered = false;

    // Model batch size
    size_t mModelBatchSize = 1;

private:
    // Shared weights
    const SharedWeights kSharedWeights;

private:
    // Input Output Links
    std::unordered_map<size_t, size_t> mModelInToOutIndexLinks;
    std::unordered_set<size_t> mReservedInputBuffers;
    std::unordered_set<size_t> mReservedOutputBuffers;

    // Allocator
    std::unique_ptr<Allocator> mAllocator;
};

} // namespace mtk
