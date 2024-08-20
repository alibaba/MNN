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
#include "VulkanBasicExecution.hpp"

namespace MNN {

class VulkanLayernorm : public VulkanBasicExecution {
public:
    VulkanLayernorm(const Op* op, Backend* bn);
    virtual ~VulkanLayernorm();
    virtual ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                       const VulkanCommandPool::Buffer* cmdBuffer) override;

private:
    std::shared_ptr<VulkanBuffer> mParam;
    std::shared_ptr<Tensor> mGamma;
    std::shared_ptr<Tensor> mBias;
    const VulkanPipeline* mPipeline;
    std::shared_ptr<VulkanLayout::DescriptorSet> mDesSet;
    float mEps;
    bool mHasScale = false;
    int mGroup = 0;
    int mAxisSize = 0;
};

} // namespace MNN

#endif /* VulkanLayernorm_hpp */
