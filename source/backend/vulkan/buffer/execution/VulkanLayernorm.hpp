//
//  VulkanLayernorm.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef VulkanLayernorm_hpp
#define VulkanLayernorm_hpp

#include <stdio.h>
#include <string>
#include <vector>
#include "VulkanBasicExecution.hpp"

namespace MNN {

class VulkanLayernorm : public VulkanBasicExecution {
public:
    VulkanLayernorm(const Op* op, Backend* bn, Tensor* tensor);
    virtual ~VulkanLayernorm();
    virtual ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                               const VulkanCommandPool::Buffer* cmdBuffer) override;
    virtual bool onClone(Backend* bn, const Op* op, VulkanBasicExecution** dst) override;

private:
    VulkanLayernorm(Backend* bn, const VulkanLayernorm* src);
    std::shared_ptr<VulkanBuffer> mParam;
    std::shared_ptr<Tensor> mGamma;
    std::shared_ptr<Tensor> mBias;
    const VulkanPipeline* mPipeline = nullptr;
    std::shared_ptr<VulkanLayout::DescriptorSet> mDesSet;
    const VulkanPipeline* mOptPipeline = nullptr;
    std::shared_ptr<VulkanLayout::DescriptorSet> mOptDesSet;
    const VulkanPipeline* mOptSubgroupPipeline = nullptr;
    std::shared_ptr<VulkanLayout::DescriptorSet> mOptSubgroupDesSet;
    const VulkanPipeline* mC4Pipeline = nullptr;
    std::shared_ptr<VulkanLayout::DescriptorSet> mC4DesSet;
    const VulkanPipeline* mC4SubgroupPipeline = nullptr;
    std::shared_ptr<VulkanLayout::DescriptorSet> mC4SubgroupDesSet;
    std::string mKey;
    std::string mOptKey;
    std::string mOptSubgroupKey;
    std::string mC4Key;
    std::string mC4SubgroupKey;
    std::vector<VkDescriptorType> mDesTypes;
    std::vector<VkDescriptorType> mC4DesTypes;
    float mEps;
    bool mHasScale = false;
    bool mUseRMSNorm = false;
    int mGroup = 0;
    int mAxisSize = 0;
    bool mFP16{false};
    uint32_t mSubgroupSize = 0;
};

} // namespace MNN

#endif /* VulkanLayernorm_hpp */
