//
//  VulkanLoop.cpp
//  MNN
//
//  Created by MNN on 2024/10/18.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef VulkanLoop_hpp
#define VulkanLoop_hpp

#include "VulkanBasicExecution.hpp"
#include "VulkanImageConverter.hpp"

namespace MNN {

class VulkanLoop {
public:
    static VulkanBasicExecution* create(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const Op* op, Backend* bn);
};

} // namespace MNN

#endif /* VulkanLoop_hpp */
