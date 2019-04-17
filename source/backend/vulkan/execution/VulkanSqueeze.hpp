//
//  VulkanSqueeze.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef VulkanSqueeze_hpp
#define VulkanSqueeze_hpp

#include <stdio.h>
#include "VulkanBasicExecution.hpp"
namespace MNN {

class VulkanSqueeze : public VulkanBasicExecution {
public:
    VulkanSqueeze(Backend* bn);
    virtual ~VulkanSqueeze();
    ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                       const VulkanCommandPool::Buffer* cmdBuffer) override;
};

} // namespace MNN

#endif /* VulkanSqueeze_hpp */
