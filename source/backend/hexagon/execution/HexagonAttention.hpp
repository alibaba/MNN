#ifndef HexagonAttention_hpp
#define HexagonAttention_hpp

#include "core/OpCommonUtils.hpp"
#include "HexagonExecution.hpp"
#include "HexagonKVCacheManager.hpp"

namespace MNN {

class HexagonAttention : public HexagonExecution {
public:
    HexagonAttention(Backend *backend, const Op *op);
    virtual ~HexagonAttention();
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;
    static Execution* create(Backend* backend, const Op* op) {
        return new HexagonAttention(backend, op);
    }
private:
    ErrorCode onBuildCmd(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                         std::vector<HexagonCommand>& dst) override;
    ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    ErrorCode ensurePageTableCapacity(int pageCount);
    void updatePageTable();

    float mAttnScale = 0.0f;
    std::shared_ptr<Tensor> mWorkspace;
    std::shared_ptr<Tensor> mPageTable;
    bool mUseGeneratedCausalMask = false;
    std::shared_ptr<HexagonKVCacheManager> mKVCacheManager;
    KVMeta* mMeta = nullptr;
    int mMaxKVLen = 0;
    int mPageTableCapacity = 0;
    uint64_t mSyncedPageGeneration = (uint64_t)-1;
};

} // namespace MNN

#endif /* HexagonAttention_hpp */
