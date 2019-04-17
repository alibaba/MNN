//
//  VulkanImageConverter.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef VulkanImageConverter_hpp
#define VulkanImageConverter_hpp
#include "Tensor.hpp"
#include "Tensor_generated.h"
#include "VulkanBuffer.hpp"
#include "VulkanCommandPool.hpp"
#include "VulkanImage.hpp"
#include "VulkanPipeline.hpp"
namespace MNN {
class VulkanBackend;
class VulkanTensorConvert {
public:
    VulkanTensorConvert() = delete;
    VulkanTensorConvert(const VulkanBackend* bn);
    virtual ~VulkanTensorConvert();

    bool encodeTensorConvert(VkBuffer source, VkBuffer dest, const Tensor* shape, MNN_DATA_FORMAT sourceFormat,
                             MNN_DATA_FORMAT destFormat, VkDeviceSize srcOffset, VkDeviceSize destOffset,
                             VkDeviceSize srcSize, VkDeviceSize dstSize, const VulkanCommandPool::Buffer* cmdBuffer);
    struct Uniforms {
        int width;
        int height;
        int channel;
        int batch;
    };

private:
    std::shared_ptr<VulkanPipeline::DescriptorSet> mTensorConvertDescriptorSet;
    std::shared_ptr<VulkanBuffer> mTensorConvertUniformBuffer;
    const VulkanPipeline* mTensorConvertPipeline;
    const VulkanBackend* mVulkanBackend;
};
class VulkanImageConverter : public NonCopyable {
public:
    VulkanImageConverter(const VulkanBackend* bn);
    virtual ~VulkanImageConverter() {
    }

    void encodeTensorToBuffer(const Tensor* srcTensor, VkBuffer destBuffer, const int bufferSize,
                              VkDeviceSize bufferOffset, MNN_DATA_FORMAT destBufferFormat,
                              const VulkanCommandPool::Buffer* cmdBuffer);
    void encodeBufferToTensor(VkBuffer srcBuffer, const Tensor* destTensor, const int bufferSize,
                              VkDeviceSize bufferOffset, MNN_DATA_FORMAT srcBufferFormat,
                              const VulkanCommandPool::Buffer* cmdBuffer);

private:
    void _encodeImageBufferConvert(const Tensor* tensor, VkBuffer destBuffer, const int bufferSize,
                                   VkDeviceSize bufferOffset, const VulkanCommandPool::Buffer* cmdBuffer,
                                   VkImageLayout layout);
    enum TYPE {
        IMAGE_TO_BUFFER,
        BUFFER_TO_BUFFER,
        BUFFER_TO_IMAGE,
    };

    void _setUpPipeline(MNN_DATA_FORMAT source, MNN_DATA_FORMAT dest, TYPE type, halide_type_t dataType);
    const VulkanBackend* mBackend;
    std::shared_ptr<VulkanPipeline::DescriptorSet> mSet;
    std::shared_ptr<VulkanBuffer> mConst;
    const VulkanPipeline* mPipeline = nullptr;
    const VulkanSampler* mSampler   = nullptr;
    MNN_DATA_FORMAT mCurrentSource;
    MNN_DATA_FORMAT mCurrentDest;

    VulkanTensorConvert mBufferConverter;
    TYPE mConvertImage;
};
} // namespace MNN
#endif /* VulkanImageConverter_hpp */
