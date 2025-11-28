#include "common/cpp11_compat.h"
#include "common/file_source.h"
#include "common/thread_pool.h"
#include "executor/allocator.h"
#include "executor/neuron_usdk_executor.h"

#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <vector>

namespace mtk {

using ExecutorBackend = NeuronUsdkExecutor;

#ifdef DIRECT_IO_SHARED_WEIGHTS
static constexpr bool kSharedWeightsUseDirectIO = true;
#else
static constexpr bool kSharedWeightsUseDirectIO = false;
#endif

using MemoryAllocator = ExecutorBackend::MemoryAllocator;

//===------------------------===//
// SharedWeights (Per DLA)
//===------------------------===//

SharedWeights::operator bool() const {
    return !empty();
}

bool SharedWeights::empty() const {
    return size() == 0;
}

size_t SharedWeights::size() const {
    const auto numSwFiles = files.size();
    const auto numSwBuffers = buffers.size();
    DCHECK(numSwBuffers == 0 || numSwBuffers == numSwFiles);
    return numSwFiles;
}

bool SharedWeights::isPreloaded(const size_t swIndex) const {
    return (swIndex < buffers.size() && buffers[swIndex].isAllocated());
}

//===------------------------===//
// SharedWeightsHandle (Global)
//===------------------------===//

SharedWeightsHandle::SharedWeightsHandle(const std::vector<FileSource>& sharedWeightsFiles,
                                         const size_t numDlaChunks)
    : kSharedWeightsFiles(sharedWeightsFiles), kNumDlaChunks(numDlaChunks) {
    mPreloadedMap.resize(sharedWeightsFiles.size()); // All false
}

SharedWeightsHandle::~SharedWeightsHandle() {
    wait(); // Ensure the thread pool is clear
    if (mAllocator)
        mAllocator->releaseAll();
}

void SharedWeightsHandle::setPreloadSubset(const std::unordered_set<size_t>& subsetIndexes,
                                           const bool repeatAllChunks) {
    if (!repeatAllChunks) {
        mPreloadIndexes = subsetIndexes;
        return;
    }

    const auto numSwFiles = kSharedWeightsFiles.size();
    DCHECK_EQ(numSwFiles % kNumDlaChunks, 0);
    const auto numSwPerChunk = numSwFiles / kNumDlaChunks;
    const auto maxSwSubsetIdx = *std::max_element(subsetIndexes.begin(), subsetIndexes.end());
    CHECK_LT(maxSwSubsetIdx, numSwPerChunk)
        << "Invalid shared weights subset index to repeat for all chunks.";

    mPreloadIndexes.clear();
    for (const auto chunkSwIdx : subsetIndexes) {
        for (size_t chunk = 0; chunk < kNumDlaChunks; chunk++) {
            mPreloadIndexes.insert(chunk * numSwPerChunk + chunkSwIdx);
        }
    }
}

void SharedWeightsHandle::preload(const bool async) {
    if (kSharedWeightsFiles.empty()) {
        return; // No shared weights to use
    }

    wait(); // Ensure no thread is still preloading in the background

    LOG(DEBUG) << "Preloading shared weights" << (async ? " with async" : "");

    if (!mAllocator)
        mAllocator = std::make_shared<MemoryAllocator>();

    const auto numSwFiles = kSharedWeightsFiles.size();
    mSharedWeightsBuffers.resize(numSwFiles);

    // Empty indexes indicates all shared weights are to be preloaded
    if (mPreloadIndexes.empty()) {
        for (size_t idx = 0; idx < numSwFiles; idx++)
            mPreloadIndexes.insert(idx);
    }

    auto allocForSharedWeights = [this](const size_t swIndex) {
        auto& swBuffer = mSharedWeightsBuffers[swIndex];
        DCHECK(!swBuffer);
        const auto swSize = kSharedWeightsFiles[swIndex].getSize();
        swBuffer = std::make_shared<SmartIOBuffer>(swSize, mAllocator);
    };

    // Load single shared weights, and allocate buffer for it if needed.
    auto loadSharedWeights = [this](const size_t swIndex) {
        const auto& swFile = kSharedWeightsFiles[swIndex];
        const auto swSize = swFile.getSize();
        auto& swBuffer = mSharedWeightsBuffers[swIndex];
        if (!swBuffer) {
            DCHECK(!isPreloaded(swIndex));
            swBuffer = std::make_shared<SmartIOBuffer>(swSize, mAllocator);
        }
        if (isPreloaded(swIndex)) {
            return;
        }

        if (kSharedWeightsUseDirectIO) {
            swFile.directRead(swBuffer->addr(), swSize);
        } else {
            std::memcpy(swBuffer->addr(), swFile.getData(), swSize);
            swFile.hint_release();
        }
        mPreloadedMap[swIndex] = true;
    };

    auto loadAllSequentially = [=] {
        for (const auto swIndex : mPreloadIndexes) {
            loadSharedWeights(swIndex);
        }
    };

    auto loadAllConcurrently = [=] {
        for (const auto swIndex : mPreloadIndexes) {
            mThreadPool.push(loadSharedWeights, swIndex);
        }
    };

    // If async is true, wait for buffer allocations then perform memcpy in the background.
    // Otherwise, perform all buffer allocations and memcpy before return.
    if (async) {
        // Allocate shared weights buffers
        for (const auto swIndex : mPreloadIndexes) {
            mThreadPool.push(allocForSharedWeights, swIndex);
        }
        mThreadPool.joinAll(); // Wait for buffer allocations only
        // Load shared weights sequentially in the background.
        // Sequential loading is faster in async mode because DRAM load is lower.
        mThreadPool.push(loadAllSequentially);
    } else {
        // Load all shared weights concurrently before return, i.e. need to join before return.
        // Faster in non-async mode, but causes higher DRAM load, making async model loading slower.
        loadAllConcurrently();
        mThreadPool.joinAll();
    }
}

void SharedWeightsHandle::preloadUnion(
    const std::vector<std::unique_ptr<SharedWeightsHandle>>& preloadHandles,
    const std::vector<std::unique_ptr<SharedWeightsHandle>>& refHandles) {
    // Convert vector of unique_ptrs to vector of pointers
    auto getRawPointers = [](const std::vector<std::unique_ptr<SharedWeightsHandle>>& uniquePtrs) -> std::vector<SharedWeightsHandle*> {
        std::vector<SharedWeightsHandle*> rawPtrs;
        rawPtrs.reserve(uniquePtrs.size());
        for (auto& uPtr : uniquePtrs)
            rawPtrs.push_back(uPtr.get());
        return rawPtrs;
    };
    preloadUnion(getRawPointers(preloadHandles), getRawPointers(refHandles));
}

void SharedWeightsHandle::preloadUnion(const std::vector<SharedWeightsHandle*>& preloadHandles,
                                       const std::vector<SharedWeightsHandle*>& refHandles) {
    // Maps the shared weights file to be preloaded to its handle that will preload it and its index
    // in the handle.
    std::unordered_map<FileSource, std::pair<SharedWeightsHandle*, size_t>> preloadFilesMap;

    auto isFilePreloaded = [&](const FileSource& file) {
        return preloadFilesMap.find(file) != preloadFilesMap.end();
    };

    // Populate all existing preloaded shared weights from reference handles
    for (const auto refHandle : refHandles) {
        CHECK(refHandle != nullptr);
        const auto& swFiles = refHandle->kSharedWeightsFiles;
        for (size_t swIdx = 0; swIdx < swFiles.size(); swIdx++) {
            if (refHandle->isPreloaded(swIdx)) {
                preloadFilesMap.emplace(swFiles[swIdx], std::make_pair(refHandle, swIdx));
            }
        }
    }

    // Track the preload indexes skippable by the each swHandle
    std::vector<std::vector<size_t>> allSkippedPreloadIdxs;
    allSkippedPreloadIdxs.resize(preloadHandles.size());

    // Search for common file across shared weights handles to be preloaded
    for (size_t swHandleIdx = 0; swHandleIdx < preloadHandles.size(); swHandleIdx++) {
        auto swHandle = preloadHandles[swHandleIdx];
        CHECK(swHandle != nullptr);
        const auto& swFiles = swHandle->kSharedWeightsFiles;
        const auto numSwFiles = swFiles.size();
        auto& preloadIdxs = swHandle->mPreloadIndexes;

        // Empty indexes indicates all shared weights are to be preloaded
        if (preloadIdxs.empty()) {
            for (size_t idx = 0; idx < numSwFiles; idx++)
                preloadIdxs.insert(idx);
        }

        auto& skippedPreloadIdxs = allSkippedPreloadIdxs[swHandleIdx];
        skippedPreloadIdxs.reserve(preloadIdxs.size());

        for (const auto preloadIdx : preloadIdxs) {
            DCHECK_LT(preloadIdx, numSwFiles);
            const auto& swFile = swFiles[preloadIdx];
            if (isFilePreloaded(swFile)) {
                // The current swFile will already be loaded by other swHandle, so we can skip.
                LOG(DEBUG) << swFile << " will be preloaded by swHandle "
                           << preloadFilesMap[swFile].first;
                skippedPreloadIdxs.push_back(preloadIdx);
                continue;
            }
            preloadFilesMap.emplace(swFile, std::make_pair(swHandle, preloadIdx));
        }

        // Temporarily remove skippable preload indexes so that preload() won't load them.
        for (const auto idx : skippedPreloadIdxs) {
            preloadIdxs.erase(idx);
        }
    }

    auto getPreloadedSwBuffer = [&](const FileSource& swFile) {
        DCHECK(isFilePreloaded(swFile));
        const auto& preloadInfo = preloadFilesMap[swFile];
        const auto& preloadHandle = preloadInfo.first;
        const auto& swIdx = preloadInfo.second;
        const auto& preloadedSwBuffer = preloadHandle->mSharedWeightsBuffers[swIdx];
        DCHECK(preloadedSwBuffer);
        return preloadedSwBuffer;
    };

    for (size_t swHandleIdx = 0; swHandleIdx < preloadHandles.size(); swHandleIdx++) {
        auto swHandle = preloadHandles[swHandleIdx];
        CHECK(swHandle != nullptr);

        // Skip preload if everything that requires preloading can be reused from other handles
        auto& preloadIdxs = swHandle->mPreloadIndexes;
        if (preloadIdxs.empty()) {
            // NOTE: preload() will interpret an empty mPreloadIndexes as requiring to preload all.
            const auto numSwFiles = swHandle->kSharedWeightsFiles.size();
            swHandle->mSharedWeightsBuffers.resize(numSwFiles);
        } else {
            swHandle->preload(/*async*/ false); // Async preload is disabled
        }

        // Add back the skipped preload indexes after the non-reusable ones have been preloaded
        auto& skippedPreloadIdxs = allSkippedPreloadIdxs[swHandleIdx];
        for (const auto swIdx : skippedPreloadIdxs) {
            DCHECK(preloadIdxs.find(swIdx) == preloadIdxs.end());
            preloadIdxs.insert(swIdx);
        }

        // Reuse common preloaded shared weights
        const auto& swFiles = swHandle->kSharedWeightsFiles;
        for (const auto swIdx : preloadIdxs) {
            const auto& swFile = swFiles[swIdx];
            DCHECK(isFilePreloaded(swFile));

            // The handle that preloaded the swFile
            auto preloadHandle = preloadFilesMap[swFile].first;

            if (preloadHandle == swHandle) {
                continue; // No need to reuse if preloaded by itself
            }
            LOG(DEBUG) << "Handle " << swHandle << " is reusing shared weights " << swIdx
                       << " from " << preloadHandle;

            auto& curSwBuffer = swHandle->mSharedWeightsBuffers[swIdx];
            auto preloadedSwBuffer = getPreloadedSwBuffer(swFile);

            // Should not overwrite existing shared weights buffer with a different preloaded buffer
            DCHECK(!curSwBuffer || curSwBuffer == preloadedSwBuffer);

            // Reuse preloaded shared weights buffer
            curSwBuffer = preloadedSwBuffer;
            swHandle->mPreloadedMap[swIdx] = true;
        }
    }
}

bool SharedWeightsHandle::isPreloaded(const size_t swIndex) const {
    if (swIndex >= mSharedWeightsBuffers.size()) {
        return false;
    }
    return mPreloadedMap[swIndex];
}

void SharedWeightsHandle::unload(const size_t swIndex) {
    if (swIndex < mSharedWeightsBuffers.size()) {
        mSharedWeightsBuffers[swIndex].reset();
        mPreloadedMap[swIndex] = false;
    }
}

void SharedWeightsHandle::wait() const {
    mThreadPool.joinAll();
}

SharedWeights SharedWeightsHandle::getSharedWeights(const size_t dlaChunkIndex) const {
    CHECK_LT(dlaChunkIndex, kNumDlaChunks);

    const auto numSwFiles = kSharedWeightsFiles.size();
    CHECK_EQ(numSwFiles % kNumDlaChunks, 0)
        << "The number of shared weights files used per DLA must be same for all DLA files.";

    const bool preloaded = !mSharedWeightsBuffers.empty();

    SharedWeights swChunk;
    const auto numSwPerDla = numSwFiles / kNumDlaChunks;
    const size_t startSwIdx = dlaChunkIndex * numSwPerDla;

    auto& swChunkFiles = swChunk.files; // Always assigned regardless preloaded or not
    auto& swChunkBuffers = swChunk.buffers;

    swChunkFiles.reserve(numSwPerDla);

    if (preloaded) {
        swChunkBuffers.reserve(numSwPerDla);
    }

    for (size_t i = startSwIdx; i < startSwIdx + numSwPerDla; i++) {
        swChunkFiles.push_back(kSharedWeightsFiles[i]);
        if (preloaded) {
            auto& swBuffer = mSharedWeightsBuffers[i];
            swChunkBuffers.push_back(swBuffer ? swBuffer->getIOBuffer() : IOBuffer());
        }
    }
    return swChunk;
}

} // namespace mtk
