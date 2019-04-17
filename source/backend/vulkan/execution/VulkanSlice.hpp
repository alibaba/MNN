//
//  VulkanSlice.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef VulkanSlice_hpp
#define VulkanSlice_hpp

#include <stdio.h>

#include "VulkanBasicExecution.hpp"
#include "VulkanImageConverter.hpp"

namespace MNN {

class VulkanSlice : public VulkanBasicExecution {
public:
    VulkanSlice(const Op* op, Backend* backend);
    virtual ~VulkanSlice();
    ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                       const VulkanCommandPool::Buffer* cmdBuffer) override;

private:
    int mAxis;
    // NCHW tensor
    Tensor mTempTensor;
    std::shared_ptr<VulkanImageConverter> mTensorConverter4Input;
    std::vector<std::shared_ptr<VulkanImageConverter>> mTensorConverters4Ouput;
};

} // namespace MNN

#endif /* VulkanSlice_hpp */
