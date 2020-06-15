//
//  VulkanTensorConvert.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef VulkanTensorConvert_hpp
#define VulkanTensorConvert_hpp

#include <stdio.h>
#include "core/Macro.h"
#include "core/Macro.h"

#include "backend/vulkan/execution/VulkanBasicExecution.hpp"
#include "backend/vulkan/execution/VulkanImageConverter.hpp"

namespace MNN {
class VulkanTensorConvertVulkanBasicExecution : public VulkanBasicExecution {
public:
    VulkanTensorConvertVulkanBasicExecution(const Op *op, Backend *bn);
    virtual ~VulkanTensorConvertVulkanBasicExecution();
    ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                       const VulkanCommandPool::Buffer *cmdBuffer) override;

private:
    std::shared_ptr<VulkanImageConverter> mTensorConverter;
};

} // namespace MNN

#endif /* VulkanTensorConvert_hpp */
