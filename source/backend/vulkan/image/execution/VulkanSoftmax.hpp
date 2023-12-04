//
//  VulkanSoftmax.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef VulkanSoftmax_hpp
#define VulkanSoftmax_hpp

#include <stdio.h>
#include "VulkanBasicExecution.hpp"
#include "VulkanImageConverter.hpp"

namespace MNN {
class VulkanSoftmax : public VulkanBasicExecution {
public:
    VulkanSoftmax(const Op* op, Backend* bn);
    virtual ~VulkanSoftmax();
    ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                       const VulkanCommandPool::Buffer* cmdBuffer) override;

private:
    std::shared_ptr<VulkanBuffer> mConstBuffer;
    const VulkanPipeline* mSoftmaxPipeline;
    std::shared_ptr<VulkanLayout::DescriptorSet> mDescriptorSet;
    int mAxis;
    struct ConvertInfo {
        const VulkanPipeline* pipeline;
        std::shared_ptr<VulkanImageConverter> convert;
        std::shared_ptr<VulkanBuffer> buffer;
    };
    ConvertInfo mSource;
    ConvertInfo mOutput;
};

} // namespace MNN

#endif /* VulkanSoftmax_hpp */
