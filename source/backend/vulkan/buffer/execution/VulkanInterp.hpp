//
//  VulkanInterp.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef VulkanInterp_hpp
#define VulkanInterp_hpp

#include <stdio.h>

#include "VulkanResize.hpp"

namespace MNN {

class VulkanInterp : public VulkanResize {
public:
    VulkanInterp(const Op *op, Backend *bn);
    virtual ~VulkanInterp();

    ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                       const VulkanCommandPool::Buffer *cmdBuffer) override;

private:
    float mCordTransform[4];
};

} // namespace MNN

#endif /* VulkanInterp_hpp */
