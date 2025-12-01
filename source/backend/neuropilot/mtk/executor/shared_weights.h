#pragma once

#include "common/file_source.h"
#include "common/thread_pool.h"
#include "executor/allocator.h"

#include <unordered_set>
#include <vector>

namespace mtk {

// Shared weights used by a single model chunk.
struct SharedWeights {
    std::vector<FileSource> files;
    std::vector<IOBuffer> buffers;

    // Helper functions
    explicit operator bool() const;

    bool empty() const;

    size_t size() const;

    bool isPreloaded(const size_t swIndex) const;
};

// A global shared weights handle that can exist outside of LLM Runtime
class SharedWeightsHandle {
public:
    explicit SharedWeightsHandle(const std::vector<FileSource>& sharedWeightsFiles,
                                 const size_t numDlaChunks = 1);

    ~SharedWeightsHandle();

    void setPreloadSubset(const std::unordered_set<size_t>& subsetIndexes,
                          const bool repeatAllChunks = false);

    void preload(const bool async = false);

    static void preloadUnion(const std::vector<SharedWeightsHandle*>& preloadHandles,
                             const std::vector<SharedWeightsHandle*>& refHandles = {});

    static void
    preloadUnion(const std::vector<std::unique_ptr<SharedWeightsHandle>>& preloadHandles,
                 const std::vector<std::unique_ptr<SharedWeightsHandle>>& refHandles = {});

    bool isPreloaded(const size_t swIndex) const;

    void unload(const size_t swIndex);

    void wait() const;

    SharedWeights getSharedWeights(const size_t dlaChunkIndex) const;

private:
    const size_t kNumDlaChunks;
    std::shared_ptr<Allocator> mAllocator;
    std::vector<std::shared_ptr<SmartIOBuffer>> mSharedWeightsBuffers;
    std::vector<FileSource> kSharedWeightsFiles;
    std::unordered_set<size_t> mPreloadIndexes;
    std::vector<bool> mPreloadedMap;
    mutable BasicThreadPool mThreadPool;
};

} // namespace mtk