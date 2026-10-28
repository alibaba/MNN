#ifndef HexagonKVCacheManager_hpp
#define HexagonKVCacheManager_hpp

#include "core/KVCacheManager.hpp"
#include <stdint.h>
#include <vector>

namespace MNN {
class HexagonKVCacheManager : public KVCacheManager {
public:
    HexagonKVCacheManager(Backend * backend, KVCacheConfig & kvConfig);
    virtual ~HexagonKVCacheManager();
    virtual void onResize(int kv_num_head, int head_dim) override;
    virtual void onClear() override;
    virtual void onAlloc(KVMeta* meta, int seq_len) override;
    virtual void onRealloc(KVMeta* meta) override;

    int getK_icP() const { return k_icP; }
    int getK_ocP() const { return k_ocP; }
    int getV_icP() const { return v_icP; }
    int getV_ocP() const { return v_ocP; }
    int pageSize() const { return mPageSize; }
    // Logical (in-use) page count. Physical pages are sticky high-water: shrinking only lowers
    // this count and never releases buffers, so page addresses and page-table contents stay
    // stable across prefill/decode switches (see resizePageCount).
    int pageCount() const { return mLogicalPages; }
    uint64_t pageGeneration() const { return mPageGeneration; }
    const std::vector<std::shared_ptr<Tensor>>& keyPages() const { return mPastKeyPages; }
    const std::vector<std::shared_ptr<Tensor>>& valuePages() const { return mPastValuePages; }
    bool valid() const { return mValid; }

private:
    bool resizePageCount(int pageCount);
    size_t pageKeyBytes() const;
    size_t pageValueBytes() const;
    int k_icP, k_ocP, v_icP, v_ocP;
    int mPageSize = 256;
    int mLogicalPages = 0;
    uint64_t mPageGeneration = 0;
    bool mValid = true;
    std::vector<std::shared_ptr<Tensor>> mPastKeyPages;
    std::vector<std::shared_ptr<Tensor>> mPastValuePages;
};
}
#endif
