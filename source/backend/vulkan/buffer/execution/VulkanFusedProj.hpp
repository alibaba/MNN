//
//  VulkanFusedProj.hpp
//  MNN
//
//  Vulkan (buffer variant) composite execution for the fused projection op
//  (OpType_FusedLinear). Mirrors the OpenCL FusedProjBufExecution: the member
//  conv1x1 / binary RMSNorm / MUL_SILU ops are driven as child encoders inside
//  one execution, which is byte-for-byte the work the geometry decomposition
//  would have emitted. Keeping the op whole is what lets a later change
//  collapse those dispatches.
//

#ifndef VulkanFusedProj_hpp
#define VulkanFusedProj_hpp

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#include "VulkanBasicExecution.hpp"
#include "core/AutoStorage.h"

namespace MNN {

class VulkanFusedProj : public VulkanBasicExecution {
public:
    VulkanFusedProj(const Op* op, Backend* backend);
    ~VulkanFusedProj() override = default;

    bool valid() const {
        return mValid;
    }
    ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                       const VulkanCommandPool::Buffer* cmdBuffer) override;

private:
    bool _createConvs(Backend* backend);
    bool _createRest(Backend* backend, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs);

    const FusedLinearParam* mParam = nullptr;
    bool mValid    = true;
    bool mIsGateUp = false;
    bool mHasLn    = false;
    int mNumConvs   = 0;
    int mNumProjOut = 0;

    // Serialized member sub-ops; the child encoders reference into these
    // buffers, so they must outlive the children.
    std::vector<std::shared_ptr<BufferStorage>> mConvOps;
    std::shared_ptr<BufferStorage> mMulSiluOp;
    std::shared_ptr<BufferStorage> mLayerNormOp;

    std::vector<std::shared_ptr<VulkanBasicExecution>> mConvs;
    std::shared_ptr<VulkanBasicExecution> mMulSilu;
    std::shared_ptr<VulkanBasicExecution> mLn;

    // Intermediates, re-acquired from the dynamic pool on every onEncode.
    std::shared_ptr<Tensor> mNormalized;
    std::shared_ptr<Tensor> mGate;
    std::shared_ptr<Tensor> mUp;
};

} // namespace MNN

#endif /* MNN_SUPPORT_TRANSFORMER_FUSE */
#endif /* VulkanFusedProj_hpp */
