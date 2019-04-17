//
//  VulkanLRN.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef VulkanLRN_hpp
#define VulkanLRN_hpp
#include "VulkanReshape.hpp"

namespace MNN {
class VulkanLRN : public VulkanReshape {
public:
    VulkanLRN(const Op* op, Backend* bn);
    virtual ~VulkanLRN();
    ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                       const VulkanCommandPool::Buffer* cmdBuffer) override;

private:
    Tensor mTempTensor;
    float mAlpha;
    float mBeta;
    int mLocalSize;

    std::shared_ptr<VulkanBuffer> mParamBuffer;
    const VulkanPipeline* mVulkanLRNPipeline;
    std::shared_ptr<VulkanPipeline::DescriptorSet> mDescriptorSet;
};
} // namespace MNN
#endif
