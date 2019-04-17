//
//  VulkanConcat.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef VulkanConcat_hpp
#define VulkanConcat_hpp
#include "VulkanBasicExecution.hpp"
#include "VulkanImageConverter.hpp"

namespace MNN {

class VulkanConcatImageImpl {
public:
    VulkanConcatImageImpl(int axis, VulkanBackend* vkBackend);
    ~VulkanConcatImageImpl() {
    }

    ErrorCode encodeImageImpl(const std::vector<Tensor*>& inputs, Tensor* output,
                              const VulkanCommandPool::Buffer* cmdBuffer);

private:
    std::vector<std::shared_ptr<VulkanBuffer>> mConstBuffers;
    std::vector<std::shared_ptr<VulkanPipeline::DescriptorSet>> mSets;
    int mAxis = 0;
    VulkanBackend* mVkbackend;
    const VulkanSampler* mSampler;
};

class VulkanConcatBufferImpl {
public:
    VulkanConcatBufferImpl(int axis, VulkanBackend* vkBackend);
    ~VulkanConcatBufferImpl() {
    }
    ErrorCode encodeBufferImpl(const std::vector<Tensor*>& inputs, Tensor* output,
                               const VulkanCommandPool::Buffer* cmdBuffer);

private:
    int mAxis = 0;
    VulkanBackend* mVkbackend;

    std::shared_ptr<Tensor> mTempOutputTensor;
    std::vector<std::shared_ptr<Tensor>> mTempInputTensors;
    std::vector<std::shared_ptr<VulkanBuffer>> mConstBuffers;
    std::vector<std::shared_ptr<VulkanPipeline::DescriptorSet>> mSets;
    std::shared_ptr<VulkanImageConverter> mTensorConvert4Output;
    std::vector<std::shared_ptr<VulkanImageConverter>> mTensorConvert4Inputs;
};

class VulkanConcat : public VulkanBasicExecution {
public:
    VulkanConcat(const Op* op, Backend* bn);
    virtual ~VulkanConcat() {
    }
    virtual ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                               const VulkanCommandPool::Buffer* cmdBuffer) override;

private:
    int mAxis;
    VulkanBackend* mVkbackend;
    std::shared_ptr<VulkanConcatImageImpl> mImageConcat;
    std::shared_ptr<VulkanConcatBufferImpl> mBufferConcat;
};

} // namespace MNN

#endif /* VulkanConcat_hpp */
