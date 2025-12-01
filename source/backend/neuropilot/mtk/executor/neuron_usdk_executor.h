#pragma once

#include "common/file_source.h"
#include "executor/executor.h"

#include <deque>
#include <memory>
#include <mutex>

struct NeuronModel;
struct NeuronCompilation;
struct NeuronExecution;

namespace mtk {

class NeuronUsdkExecutor : public Executor {
private:
    struct UsdkRuntime {
        NeuronModel* model = nullptr;
        NeuronCompilation* compilation = nullptr;
        NeuronExecution* execution = nullptr;
    };

public:
    using MemoryAllocator = AhwBufferAllocator;

public:
    explicit NeuronUsdkExecutor(int inputSize, const FileSource& modelFile,
                                const SharedWeights& sharedWeights = {})
        : Executor(modelFile, sharedWeights) {
            mInputSize = inputSize;
        }

    explicit NeuronUsdkExecutor(const std::vector<FileSource>& modelFiles,
                                const SharedWeights& sharedWeights = {})
        : Executor(modelFiles, sharedWeights) {}

    virtual ~NeuronUsdkExecutor() override { release(); }

    virtual void initialize() override;

    virtual void release() override;

    virtual void runInferenceImpl() override;

    virtual void updateModelIO() override;

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

    void createNeuronMemory(IOBuffer& ioBuf);

    bool loadDla(const void* buffer, const size_t size, UsdkRuntime* runtime);

    UsdkRuntime* getUsdkRuntime() { return reinterpret_cast<UsdkRuntime*>(this->getRuntime()); }

    UsdkRuntime* getUsdkRuntime() const {
        return reinterpret_cast<UsdkRuntime*>(this->getRuntime());
    }

    void createUsdkNeuronMemory();

private:
    // A single mutex shared across all instances of NeuronUsdkExecutor for multi-threaded init
    static std::mutex mMutex;
    int mInputSize = 0;

    std::deque<NeuronMemory*> mCreatedNeuronMems;

    std::vector<NeuronMemory*> mRegisteredInputs;
    std::vector<NeuronMemory*> mRegisteredOutputs;

    const std::string kOptions =
        "--apusys-config \"{ \\\"high_addr\\\": true, \\\"import_forever\\\": true }\" "
        "--adapter-disable-map-fd";
};

} // namespace mtk
