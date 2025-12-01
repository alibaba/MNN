#pragma once

#include "allocator.h"

#include <algorithm>
#include <vector>

// clang-format off

namespace mtk {

template <class Scenario>
class IoBufferInflator {
public:
    IoBufferInflator(const std::vector<Scenario>& scenarios, const size_t curBatchSize,
                     const size_t curTokenSize, const size_t curCacheSize)
        : kScenarios(scenarios),
          kNormalizer(curBatchSize * curTokenSize * curCacheSize),
          kCurBatchSize(curBatchSize),
          kCurTokenSize(curTokenSize),
          kCurCacheSize(curCacheSize) {}

    // Find the scenario that maximizes the size
    void findMaxSizeScenario() {
        auto key = [&](const auto& left, const auto& right) {
            auto getRelevantSize = [&](const auto& scenario) {
                size_t prod = 1;
                if (mUseBatchSize) prod *= scenario.batchSize;
                if (mUseTokenSize) prod *= scenario.tokenSize;
                if (mUseCacheSize) prod *= scenario.cacheSize;
                return prod;
            };
            return getRelevantSize(left) < getRelevantSize(right);
        };
        const auto& maxSizeScenario = *std::max_element(kScenarios.begin(), kScenarios.end(), key);

        // Use the current value if not dependent on it,
        // so that it will be canceled later by dividing with kNormalizer
        mMultiplier = 1;
        mMultiplier *= mUseBatchSize ? maxSizeScenario.batchSize : kCurBatchSize;
        mMultiplier *= mUseTokenSize ? maxSizeScenario.tokenSize : kCurTokenSize;
        mMultiplier *= mUseCacheSize ? maxSizeScenario.cacheSize : kCurCacheSize;
    }

    void inflate(IOBuffer& ioBuf) {
        if (mMultiplier == kNormalizer) {
            return;
        }
        const auto oldSize = ioBuf.sizeBytes;
        const auto newSize = ioBuf.usedSizeBytes * mMultiplier / kNormalizer;
        ioBuf.sizeBytes = newSize;
        LOG(DEBUG) << "Reassigned required allocation size: " << oldSize << " -> " << newSize;
    }

    IoBufferInflator& useBatchSize() {
        mUseBatchSize = true;
        return *this;
    }

    IoBufferInflator& useTokenSize() {
        mUseTokenSize = true;
        return *this;
    }

    IoBufferInflator& useCacheSize() {
        mUseCacheSize = true;
        return *this;
    }

    void resetUses() {
        mUseBatchSize = false;
        mUseTokenSize = false;
        mUseCacheSize = false;
    }

private:
    const std::vector<Scenario>& kScenarios;

    const size_t kCurBatchSize;
    const size_t kCurTokenSize;
    const size_t kCurCacheSize;

    const size_t kNormalizer;

    size_t mMultiplier = 1;

    bool mUseBatchSize = false;
    bool mUseTokenSize = false;
    bool mUseCacheSize = false;
};

} // namespace mtk