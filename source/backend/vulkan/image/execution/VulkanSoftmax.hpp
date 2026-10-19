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
    VulkanSoftmax(const Op* op, Backend* bn, const uint32_t axisIndex, bool useBuffer);
    virtual ~VulkanSoftmax();
    ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                       const VulkanCommandPool::Buffer* cmdBuffer) override;

private:
    ErrorCode onEncodeImage(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                            const VulkanCommandPool::Buffer* cmdBuffer);
    ErrorCode onEncodeBuffer(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                             const VulkanCommandPool::Buffer* cmdBuffer);

    struct ConvertInfo {
        std::shared_ptr<VulkanImageConverter> convert;
        std::shared_ptr<VulkanBuffer> buffer;
    };

    std::shared_ptr<VulkanBuffer> mSoftmaxConstBuffer;
    const VulkanPipeline* mSoftmaxPipeline;
    std::shared_ptr<VulkanLayout::DescriptorSet> mDescriptorSet;
    ConvertInfo mSource;
    ConvertInfo mOutput;
    uint32_t mAxisIndex;
    int mAxis;
    bool mUseBuffer;
};

} // namespace MNN

#endif /* VulkanSoftmax_hpp */
