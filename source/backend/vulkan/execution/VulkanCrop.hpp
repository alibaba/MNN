//
//  VulkanCrop.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef VulkanCrop_hpp
#define VulkanCrop_hpp

#include <stdio.h>
#include "VulkanBasicExecution.hpp"

namespace MNN {

class VulkanCrop : public VulkanBasicExecution {
public:
    VulkanCrop(const Op* op, Backend* bn);
    virtual ~VulkanCrop();
    ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                       const VulkanCommandPool::Buffer* cmdBuffer) override;

private:
    int mAxis = 2;
    std::vector<int> mCropOffset;
};

} // namespace MNN

#endif /* VulkanCrop_hpp */
