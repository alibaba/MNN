#ifndef HexagonRoPE_hpp
#define HexagonRoPE_hpp

#include "HexagonExecution.hpp"
#include "HexagonLayerNorm.hpp"

namespace MNN {

class HexagonRoPE : public HexagonExecution {
public:
    HexagonRoPE(Backend* bn, const Op* op);
    virtual ~HexagonRoPE() = default;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;

    static HexagonRoPE* create(Backend* backend, const Op* op);

private:
    ErrorCode onBuildCmd(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                         std::vector<HexagonCommand>& dst) override;

    HexagonRoPE(Backend* bn);

    int mRopeCutHeadDim = 0;
    int mNumHead = 0;
    int mKvNumHead = 0;
    int mHeadDim = 0;

    std::shared_ptr<HexagonLayerNorm::Resource> mQNorm;
    std::shared_ptr<HexagonLayerNorm::Resource> mKNorm;
    bool mFuseLayerNorm = false;
};

} // namespace MNN

#endif
