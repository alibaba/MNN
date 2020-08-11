//
//  VulkanPadding.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef VulkanPadding_hpp
#define VulkanPadding_hpp
#include "backend/vulkan/execution/VulkanBasicExecution.hpp"
#include "backend/vulkan/execution/VulkanImageConverter.hpp"

namespace MNN {
class VulkanPadding : public VulkanBasicExecution {
public:
    VulkanPadding(PadValueMode mode, int32_t* paddings, Backend* bn);
    virtual ~VulkanPadding();
    ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                       const VulkanCommandPool::Buffer* cmdBuffer) override;

    ErrorCode setLayout(const Tensor* input, const Tensor* output);

private:
    MNN_DATA_FORMAT mDimType;

public:
    Tensor mStorage;
    Tensor mWrapTensorForInput;
    Tensor mWrapTensorForOutput;

    PadValueMode mMode;
    int32_t mPaddings[8];

    std::shared_ptr<VulkanImageConverter> mTensorConvert0;
    std::shared_ptr<VulkanImageConverter> mTensorConvert1;
};
} // namespace MNN
#endif
