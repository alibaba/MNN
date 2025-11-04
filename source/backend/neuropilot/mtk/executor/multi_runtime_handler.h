#pragma once

#include "common/file_source.h"

#include <string>
#include <vector>

namespace mtk {

class MultiRuntimeHandler {
public:
    explicit MultiRuntimeHandler(const std::vector<FileSource>& modelFiles,
                                 const size_t defaultRuntimeIdx = 0)
        : mModelFiles(modelFiles), mDefaultRuntimeIdx(defaultRuntimeIdx) {}

    explicit MultiRuntimeHandler(const FileSource& modelFile, const size_t defaultRuntimeIdx = 0)
        : mModelFiles({modelFile}), mDefaultRuntimeIdx(defaultRuntimeIdx) {}

    virtual ~MultiRuntimeHandler() {}

protected:
    // Initialize all runtimes if they can coexist, otherwise initialize the default runtime
    void initRuntimes();

    // Release all active runtimes
    void releaseRuntimes();

    // Get the current runtime
    void* getRuntime() const;

    // Set the current runtime
    void setRuntime(void* runtime);

    // Set the default active runtime for use by initRuntimes()
    void setDefaultRuntimeIndex(const size_t index);

    // Get the current runtime index
    size_t getRuntimeIndex() const;

    // Select the runtime of given index
    void selectRuntime(const size_t index);

    // Get total number of runtimes
    size_t getNumRuntimes() const;

    // Get the model path/name of the current runtime
    std::string getModelName() const;

    // Add new runtime post initialization
    size_t addRuntime(FileSource modelFile);

private:
    // Create and returns a runtime given a model path. To be implemented by subclass.
    virtual void* createRuntime(FileSource modelFile) = 0;

    // Release a runtime. To be implemented by subclass.
    virtual void releaseRuntime(void* runtime) = 0;

    // Determine whether multiple runtimes are allowed to be active concurrently.
    virtual bool canRuntimesCoexist() const { return false; }

private:
    std::vector<FileSource> mModelFiles;
    std::vector<void*> mRuntimes;
    size_t mCurrentRuntimeIdx = 0;
    size_t mDefaultRuntimeIdx = 0;
};

} // namespace mtk