//
//  VulkanReshape.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef VulkanReshape_hpp
#define VulkanReshape_hpp
#include "VulkanBasicExecution.hpp"
#include "VulkanImageConverter.hpp"

namespace MNN {
class VulkanReshape : public VulkanBasicExecution {
public:
    VulkanReshape(const Op* op, Backend* bn);
    VulkanReshape(Backend* bn);
    virtual ~VulkanReshape();
    ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                       const VulkanCommandPool::Buffer* cmdBuffer) override;

    ErrorCode setLayout(const Tensor* input, const Tensor* output);

private:
    MNN_DATA_FORMAT mDimType;

public:
    Tensor mStorage;
    Tensor mWrapTensorForInput;
    Tensor mWrapTensorForOutput;

    std::shared_ptr<VulkanImageConverter> mTensorConvert0;
    std::shared_ptr<VulkanImageConverter> mTensorConvert1;
};
} // namespace MNN
#endif
