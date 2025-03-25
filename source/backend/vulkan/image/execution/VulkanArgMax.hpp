//
//  VulkanArgMax.cpp
//  MNN
//
//  Created by MNN on 2024/08/20.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef VulkanArgMax_hpp
#define VulkanArgMax_hpp

#include "VulkanBasicExecution.hpp"
#include "VulkanImageConverter.hpp"

namespace MNN {
class VulkanArgMax : public VulkanBasicExecution {

public:
    VulkanArgMax(const Op* op, Backend* bn);
    virtual ~VulkanArgMax();
    ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                       const VulkanCommandPool::Buffer* cmdBuffer) override;

private:
    const VulkanPipeline* mArgMaxPipeline;
    std::shared_ptr<VulkanLayout::DescriptorSet> mDescriptorSet;
    std::shared_ptr<VulkanBuffer> mGpuArgMaxParam;
    struct ConvertInfo {
        const VulkanPipeline* pipeline;
        std::shared_ptr<VulkanImageConverter> convert;
        std::shared_ptr<VulkanBuffer> buffer;
    };
    ConvertInfo mSource;
    ConvertInfo mOutput;
    int mAxis;
};

} // namespace MNN

#endif /* VulkanArgMax_hpp */
