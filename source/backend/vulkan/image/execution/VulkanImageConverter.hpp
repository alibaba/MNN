//
//  VulkanImageConverter.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef VulkanImageConverter_hpp
#define VulkanImageConverter_hpp
#include <MNN/Tensor.hpp>
#include "Tensor_generated.h"
#include "backend/vulkan/component/VulkanBuffer.hpp"
#include "backend/vulkan/component/VulkanCommandPool.hpp"
#include "backend/vulkan/component/VulkanImage.hpp"
#include "backend/vulkan/component/VulkanPipeline.hpp"
namespace MNN {
class VulkanBackend;
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

    static MNN_DATA_FORMAT getTensorLinearFormat(const Tensor* tensor);
private:
    void _encodeImageBufferConvert(const Tensor* tensor, VkBuffer destBuffer, const int bufferSize,
                                   VkDeviceSize bufferOffset, const VulkanCommandPool::Buffer* cmdBuffer,
                                   VkImageLayout layout, MNN_DATA_FORMAT bufferFormat);
    enum TYPE {
        IMAGE_TO_BUFFER,
        BUFFER_TO_IMAGE,
    };

    void _setUpPipeline(MNN_DATA_FORMAT source, MNN_DATA_FORMAT dest, TYPE type, halide_type_t dataType);
    const VulkanBackend* mBackend;
    std::vector<std::shared_ptr<VulkanLayout::DescriptorSet>> mSet;
    std::vector<std::shared_ptr<VulkanBuffer>> mOffset;
    std::shared_ptr<VulkanBuffer> mConst;
    const VulkanPipeline* mPipeline = nullptr;
    const VulkanSampler* mSampler   = nullptr;
    MNN_DATA_FORMAT mCurrentSource;
    MNN_DATA_FORMAT mCurrentDest;

    TYPE mConvertImage;
};
} // namespace MNN
#endif /* VulkanImageConverter_hpp */
