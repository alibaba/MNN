
#pragma once

#include "backend/api/neuron/Types.h"
#include "common/file_source.h"
#include "executor/executor.h"

namespace mtk {

// Executor class that allows user to access one input and one output
class NeuronExecutor : public Executor {
public:
    using MemoryAllocator = AhwBufferAllocator;

public:
    explicit NeuronExecutor(const FileSource& modelFile, const SharedWeights& sharedWeights = {})
        : Executor(modelFile, sharedWeights) {}

    explicit NeuronExecutor(const std::vector<FileSource>& modelFiles,
                            const SharedWeights& sharedWeights = {})
        : Executor(modelFiles, sharedWeights) {}

    virtual ~NeuronExecutor() override { release(); }

    virtual void runInferenceImpl() override;

    virtual void runInferencePrologue() override;

protected:
    virtual void* createRuntime(FileSource modelFile) override;
    virtual void releaseRuntime(void* runtime) override;

protected:
    virtual void resetIORegistration() override;

    virtual void registerRuntimeInputsImpl() override;
    virtual void registerRuntimeOutputsImpl() override;

    virtual void setRuntimeOffsetedInput(const size_t index, const size_t offset) override;
    virtual void setRuntimeOffsetedOutput(const size_t index, const size_t offset) override;

    virtual size_t getRuntimeNumInputs() const override;
    virtual size_t getRuntimeNumOutputs() const override;

    virtual size_t getRuntimeInputSizeBytes(const size_t index) const override;
    virtual size_t getRuntimeOutputSizeBytes(const size_t index) const override;

    virtual void getRuntimeInputShape(const uint64_t index, uint32_t* shape) const override;
    virtual void getRuntimeOutputShape(const uint64_t index, uint32_t* shape) const override;

private:
    virtual std::unique_ptr<Allocator> createMemoryAllocator() override;

private:
    QoSOptions mQosOptions = {};

    std::vector<int> mRegisteredInputFds;
    std::vector<int> mRegisteredOutputFds;

    bool mInferenceEnqueued = false;
};

} // namespace mtk
