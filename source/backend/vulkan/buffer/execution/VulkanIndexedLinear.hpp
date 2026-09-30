//
//  VulkanIndexedLinear.hpp
//  MNN
//
//  Created by MNN on 2026/09/29.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef VulkanIndexedLinear_hpp
#define VulkanIndexedLinear_hpp

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE
#include "VulkanBasicExecution.hpp"

namespace MNN {

class VulkanIndexedLinear : public VulkanBasicExecution {
public:
    VulkanIndexedLinear(const Op* op, Backend* backend);
    bool valid() const { return mValid; }
    ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                       const VulkanCommandPool::Buffer* command) override;
    bool onClone(Backend* backend, const Op* op, VulkanBasicExecution** dst) override;

private:
    struct Resource {
        int experts = 0, inputChannels = 0, outputChannels = 0;
        int weightBits = 0, groups = 1, groupSize = 1, weightStride = 0;
        std::shared_ptr<VulkanBuffer> weights, alpha, bias;
    };
    struct Parameters {
        int channels, outputs, experts, topk;
        int inputMode, groups, groupSize, weightStride;
        int rows, maxTiles;
    };
    VulkanIndexedLinear(const Op* op, Backend* backend, std::shared_ptr<Resource> resource);
    bool initResource(const Op* op);
    bool initPipelines();
    std::shared_ptr<Resource> mResource;
    int mInputMode = 0;
    bool mFuseSwiGLU = false;
    bool mValid = false;
    std::shared_ptr<VulkanBuffer> mRouting;
    const VulkanPipeline* mGemv = nullptr;
    const VulkanPipeline* mGemm[2] = {nullptr, nullptr};
    const VulkanPipeline* mRoute[4] = {nullptr, nullptr, nullptr, nullptr};
    const VulkanPipeline* mLargePrefix = nullptr;
    std::shared_ptr<VulkanLayout::DescriptorSet> mGemvSet, mGemmSet[2], mLargePrefixSet;
    std::shared_ptr<VulkanLayout::DescriptorSet> mRouteSet[4];
};

void registerVulkanIndexedLinearCreator();

} // namespace MNN
#endif // MNN_SUPPORT_TRANSFORMER_FUSE
#endif
