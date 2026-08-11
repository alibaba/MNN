#include "HexagonKVCacheManager.hpp"
#include "HexagonBackend.hpp"
#include "HexagonRuntime.hpp"
#include <algorithm>
#include <string.h>

namespace MNN {

static inline int roundUpDiv(int x, int y) {
    return (x + y - 1) / y;
}

HexagonKVCacheManager::HexagonKVCacheManager(Backend * backend, KVCacheConfig & kvConfig)
    : KVCacheManager(backend, kvConfig) {
    mBytes = 2; // __fp16
    mMeta = nullptr;
}

HexagonKVCacheManager::~HexagonKVCacheManager() {
    onClear();
}

void HexagonKVCacheManager::onResize(int kv_num_head, int head_dim) {
    mKvNumHead = kv_num_head;
    mHeadDim = head_dim;
    k_icP = (head_dim + 31) / 32;
    v_ocP = (head_dim + 31) / 32;
    k_ocP = mPageSize / 32;
    v_icP = mPageSize / 32;
}

size_t HexagonKVCacheManager::pageKeyBytes() const {
    return (size_t)mKvNumHead * k_icP * k_ocP * 1024 * sizeof(int16_t);
}

size_t HexagonKVCacheManager::pageValueBytes() const {
    return (size_t)mKvNumHead * v_icP * v_ocP * 1024 * sizeof(int16_t);
}

bool HexagonKVCacheManager::resizePageCount(int pageCount) {
    if (pageCount < 0) {
        pageCount = 0;
    }
    const int oldPageCount = (int)mPastKeyPages.size();
    while ((int)mPastKeyPages.size() > pageCount) {
        mBackend->onReleaseBuffer(mPastKeyPages.back().get(), Backend::STATIC);
        mBackend->onReleaseBuffer(mPastValuePages.back().get(), Backend::STATIC);
        mPastKeyPages.pop_back();
        mPastValuePages.pop_back();
    }
    auto hexagonBackend = static_cast<HexagonBackend*>(mBackend);
    const size_t kBytes = pageKeyBytes();
    const size_t vBytes = pageValueBytes();
    while ((int)mPastKeyPages.size() < pageCount) {
        std::shared_ptr<Tensor> key(Tensor::createDevice<int8_t>({(int)kBytes}));
        std::shared_ptr<Tensor> value(Tensor::createDevice<int8_t>({(int)vBytes}));
        bool keyAcquired = mBackend->onAcquireBuffer(key.get(), Backend::STATIC);
        bool valueAcquired = keyAcquired && mBackend->onAcquireBuffer(value.get(), Backend::STATIC);
        if (!keyAcquired || !valueAcquired) {
            if (keyAcquired) {
                mBackend->onReleaseBuffer(key.get(), Backend::STATIC);
            }
            mValid = false;
            return false;
        }
        ::memset(HexagonBackend::getPtr(key.get()), 0, kBytes);
        ::memset(HexagonBackend::getPtr(value.get()), 0, vBytes);
        hexagonBackend->markHostInput(key.get());
        hexagonBackend->markHostInput(value.get());
        mPastKeyPages.emplace_back(std::move(key));
        mPastValuePages.emplace_back(std::move(value));
    }

    if (!mPastKeyPages.empty()) {
        mPastKey = mPastKeyPages[0];
        mPastValue = mPastValuePages[0];
    } else {
        mPastKey.reset();
        mPastValue.reset();
    }
    mMaxLength = pageCount * mPageSize;
    k_ocP = mPageSize / 32;
    v_icP = mPageSize / 32;
    if (pageCount != oldPageCount) {
        ++mPageGeneration;
    }
    mValid = true;
    return true;
}

void HexagonKVCacheManager::onClear() {
    resizePageCount(0);
    mPastLength = 0;
    mMaxLength = 0;
}

void HexagonKVCacheManager::onAlloc(KVMeta* meta, int seq_len) {
    mMeta = meta;
    mPastLength = 0;
    const int required = seq_len > 0 ? seq_len : 1;
    if (resizePageCount(std::max(1, roundUpDiv(required, mPageSize)))) {
        mPastLength = seq_len;
    }
}

void HexagonKVCacheManager::onRealloc(KVMeta* meta) {
    mMeta = meta;
    int start = 0;
    int add = 0;
    if (meta) {
        start = (int)meta->previous - (int)meta->remove;
        add = (int)meta->add;
    }
    if (start < 0) {
        start = 0;
    }
    int required = start + add;
    if (required <= 0) {
        required = 1;
    }
    if (resizePageCount(std::max(1, roundUpDiv(required, mPageSize)))) {
        mPastLength = start + add;
    }
}

}
