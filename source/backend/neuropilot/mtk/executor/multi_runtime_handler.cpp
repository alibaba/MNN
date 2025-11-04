#include "multi_runtime_handler.h"

#include "executor/macros.h"

#include <string>
#include <vector>

namespace mtk {

void MultiRuntimeHandler::initRuntimes() {
    const size_t numRuntimes = mModelFiles.size();
    mRuntimes.resize(numRuntimes);
    if (!canRuntimesCoexist()) {
        selectRuntime(mDefaultRuntimeIdx);
        void* runtime = createRuntime(mModelFiles[mDefaultRuntimeIdx]);
        setRuntime(runtime);
        LOG(DEBUG) << "initRuntimes(): Loaded single exclusive model (Total=" << numRuntimes << ")";
        return;
    }

    for (size_t i = 0; i < numRuntimes; i++) {
        selectRuntime(i);
        void* runtime = createRuntime(mModelFiles[i]);
        setRuntime(runtime);
    }
    selectRuntime(mDefaultRuntimeIdx); // Select the default runtime
    LOG(DEBUG) << "initRuntimes(): Loaded multiple models (Total=" << numRuntimes << ")";
}

void MultiRuntimeHandler::releaseRuntimes() {
    if (!canRuntimesCoexist()) {
        // Select the current runtime
        auto runtime = getRuntime();
        if (runtime == nullptr) {
            return;
        }
        releaseRuntime(runtime);
        setRuntime(nullptr);
        LOG(DEBUG) << "releaseRuntimes(): Released single runtime";
        return;
    }

    const size_t numRuntimes = getNumRuntimes();
    for (size_t i = 0; i < numRuntimes; i++) {
        selectRuntime(i);
        auto runtime = getRuntime();
        if (runtime == nullptr) {
            continue;
        }
        releaseRuntime(runtime);
        setRuntime(nullptr);
    }
    LOG(DEBUG) << "releaseRuntimes(): Released multiple runtimes (Total=" << getNumRuntimes()
               << ")";
}

void* MultiRuntimeHandler::getRuntime() const {
    return mRuntimes[mCurrentRuntimeIdx];
}

void MultiRuntimeHandler::setRuntime(void* runtime) {
    mRuntimes[mCurrentRuntimeIdx] = runtime;
}

void MultiRuntimeHandler::setDefaultRuntimeIndex(const size_t index) {
    mCurrentRuntimeIdx = index;
    mDefaultRuntimeIdx = index;
}

size_t MultiRuntimeHandler::getRuntimeIndex() const {
    return mCurrentRuntimeIdx;
}

void MultiRuntimeHandler::selectRuntime(const size_t index) {
    CHECK_LT(index, getNumRuntimes()) << "selectRuntime(): Index out of range.";

    if (mCurrentRuntimeIdx == index) {
        return; // Do nothing
    } else if (canRuntimesCoexist()) {
        mCurrentRuntimeIdx = index;
        LOG(DEBUG) << "Selected runtime[" << index << "]: " << mModelFiles[index].getName();
        return;
    }

    // Release current runtime if already loaded
    if (getRuntime() != nullptr) {
        releaseRuntime(getRuntime());
        setRuntime(nullptr);
    }

    // Load new runtime
    mCurrentRuntimeIdx = index;
    void* runtime = createRuntime(mModelFiles[index]);
    setRuntime(runtime);
    LOG(DEBUG) << "Selected exclusive runtime[" << index << "]: " << mModelFiles[index].getName();
}

size_t MultiRuntimeHandler::getNumRuntimes() const {
    return mRuntimes.size();
}

std::string MultiRuntimeHandler::getModelName() const {
    return mModelFiles[mCurrentRuntimeIdx].getName();
}

size_t MultiRuntimeHandler::addRuntime(FileSource modelSource) {
    mModelFiles.push_back(std::move(modelSource));
    mRuntimes.push_back(nullptr);
    const size_t newRuntimeIdx = mRuntimes.size() - 1;
    if (canRuntimesCoexist()) {
        // Create runtime immediately
        const auto oldRuntimeIdx = getRuntimeIndex();
        selectRuntime(newRuntimeIdx);
        void* runtime = createRuntime(mModelFiles[mDefaultRuntimeIdx]);
        setRuntime(runtime);
        // Switch back to original runtime
        selectRuntime(oldRuntimeIdx);
    }
    return newRuntimeIdx;
}

} // namespace mtk
