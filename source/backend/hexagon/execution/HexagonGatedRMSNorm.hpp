//
//  HexagonGatedRMSNorm.hpp
//  MNN
//
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef HexagonGatedRMSNorm_hpp
#define HexagonGatedRMSNorm_hpp

#include "HexagonLayerNorm.hpp"

namespace MNN {

// out = RMSNorm(x) * silu(z) in one DSP pass. supported() is the single envelope: the geometry gate
// (GeometryGatedRMSNorm.cpp) asks the same question, because an op kept whole that this creator then
// refuses has no execution left to pick it up -- no backend implements GatedRMSNorm on CPU.
class HexagonGatedRMSNorm : public HexagonExecution {
public:
    HexagonGatedRMSNorm(std::shared_ptr<HexagonLayerNorm::Resource> res, Backend* backend);
    virtual ~HexagonGatedRMSNorm() = default;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;

    static bool supported(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs);
    static HexagonGatedRMSNorm* create(Backend* backend, const Op* op, const std::vector<Tensor*>& inputs,
                                       const std::vector<Tensor*>& outputs);

private:
    ErrorCode onBuildCmd(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                         std::vector<HexagonCommand>& dst) override;

    std::shared_ptr<HexagonLayerNorm::Resource> mResource;
};

} // namespace MNN

#endif /* HexagonGatedRMSNorm_hpp */
