//
//  VulkanImageConverter.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/vulkan/execution/VulkanImageConverter.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "backend/vulkan/backend/VulkanBackend.hpp"
namespace MNN {
VulkanImageConverter::VulkanImageConverter(const VulkanBackend* bn) {
    mBackend = bn;
    mSampler = bn->getCommonSampler();
    mConst.reset(
        new VulkanBuffer(bn->getMemoryPool(), false, 4 * sizeof(int), nullptr, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT));
}

void VulkanImageConverter::_setUpPipeline(MNN_DATA_FORMAT sourceFormat, MNN_DATA_FORMAT destFormat, TYPE type,
                                          halide_type_t datatype) {
    if (nullptr != mPipeline && sourceFormat == mCurrentSource && destFormat == mCurrentDest && mConvertImage == type) {
        return;
    }
    mCurrentDest   = destFormat;
    mCurrentSource = sourceFormat;
    mConvertImage  = type;

    if (type != BUFFER_TO_BUFFER) {
        std::vector<VkDescriptorType> types{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
        if (type == BUFFER_TO_IMAGE) {
            types[0] = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        }

        if (mCurrentDest == MNN_DATA_FORMAT_NCHW) {
            mPipeline = mBackend->getPipeline("glsl_imageTonchw_comp",
                                              /*glsl_imageTonchw_comp, glsl_imageTonchw_comp_len,*/ types);
        } else if (mCurrentDest == MNN_DATA_FORMAT_NHWC) {
            mPipeline = mBackend->getPipeline("glsl_imageTonhwc_comp",
                                              /*glsl_imageTonhwc_comp, glsl_imageTonhwc_comp_len,*/ types);
        } else if (mCurrentSource == MNN_DATA_FORMAT_NHWC) {
            mPipeline = mBackend->getPipeline("glsl_nhwcToimage_comp",
                                              /*glsl_nhwcToimage_comp, glsl_nhwcToimage_comp_len,*/ types);
        } else if (mCurrentSource == MNN_DATA_FORMAT_NCHW) {
            mPipeline = mBackend->getPipeline("glsl_nchwToimage_comp",
                                              /*glsl_nchwToimage_comp, glsl_nchwToimage_comp_len,*/ types);
        } else if (mCurrentSource == mCurrentDest) {
            if (type == BUFFER_TO_IMAGE) {
                mPipeline = mBackend->getPipeline("glsl_nc4hw4toimage_comp",
                                                  /*glsl_nc4hw4toimage_comp, glsl_nc4hw4toimage_comp_len,*/ types);
            } else {
                mPipeline = mBackend->getPipeline("glsl_imageTonc4hw4_comp",
                                                  /*glsl_imageTonc4hw4_comp, glsl_imageTonc4hw4_comp_len,*/ types);
            }
        }
    }
    MNN_ASSERT(nullptr != mPipeline);
    mSet.reset(mPipeline->createSet());
}

void VulkanImageConverter::encodeBufferToTensor(VkBuffer srcBuffer, const Tensor* destTensor, const int bufferSize,
                                                VkDeviceSize bufferOffset, MNN_DATA_FORMAT srcBufferFormat,
                                                const VulkanCommandPool::Buffer* cmdBuffer) {
    auto destFormat   = TensorUtils::getDescribe(destTensor)->dimensionFormat;
    auto sourceFormat = srcBufferFormat;
    auto vkTensor     = mBackend->findTensor(destTensor->deviceId());
    if (vkTensor->buffer() != nullptr) {
        if (sourceFormat == destFormat) {
            VkBufferCopy region;
            ::memset(&region, 0, sizeof(VkBufferCopy));
            region.size      = destTensor->elementSize() * 4;
            region.srcOffset = 0;
            region.dstOffset = 0;
            vkCmdCopyBuffer(cmdBuffer->get(), srcBuffer, (VkBuffer)destTensor->deviceId(), 1, &region);
            return;
        }
        MNN_ASSERT(false);
        return;
    }
    auto tensor = destTensor;
    _setUpPipeline(sourceFormat, destFormat, BUFFER_TO_IMAGE, tensor->buffer().type);
    _encodeImageBufferConvert(tensor, srcBuffer, bufferSize, bufferOffset, cmdBuffer, VK_IMAGE_LAYOUT_GENERAL);
}
void VulkanImageConverter::_encodeImageBufferConvert(const Tensor* tensor, VkBuffer destBuffer, const int bufferSize,
                                                     VkDeviceSize bufferOffset,
                                                     const VulkanCommandPool::Buffer* cmdBuffer, VkImageLayout layout) {
    int w     = std::max(tensor->width(), 1);
    int h     = std::max(tensor->height(), 1);
    int cDiv4 = UP_DIV(tensor->channel(), 4);
    int b     = tensor->batch();
    auto dims = (int*)mConst->map();
    dims[0]   = w;
    dims[1]   = h;
    dims[2]   = tensor->channel();
    dims[3]   = b;
    mConst->unmap();

    auto backend = (VulkanBackend*)mBackend;
    auto vkTensor = backend->findTensor(tensor->deviceId());
    cmdBuffer->barrierImageIfNeeded(vkTensor->image(), layout);

    mSet->writeImage((VkImageView)tensor->deviceId(), mSampler->get(), layout, 0);
    mSet->writeBuffer(destBuffer, 1, bufferSize, bufferOffset);
    mSet->writeBuffer(mConst->buffer(), 2, mConst->size());

    mPipeline->bind(cmdBuffer->get(), mSet->get());
    vkCmdDispatch(cmdBuffer->get(), UP_DIV(w, 16), UP_DIV(h, 16), cDiv4 * b);
}

void VulkanImageConverter::encodeTensorToBuffer(const Tensor* srcTensor, VkBuffer destBuffer, const int bufferSize,
                                                VkDeviceSize bufferOffset, MNN_DATA_FORMAT destBufferFormat,
                                                const VulkanCommandPool::Buffer* cmdBuffer) {
    auto sourceFormat = TensorUtils::getDescribe(srcTensor)->dimensionFormat;
    auto destFormat   = destBufferFormat;
    if (sourceFormat == MNN_DATA_FORMAT_NC4HW4 && 1 >= srcTensor->width() && 1 >= srcTensor->height() &&
        srcTensor->channel() % 4 == 0) {
        destFormat = MNN_DATA_FORMAT_NC4HW4;
    }

    auto vkTensor = mBackend->findTensor(srcTensor->deviceId());
    if (vkTensor->buffer() != nullptr) {
        if (sourceFormat == destFormat) {
            VkBufferCopy region;
            ::memset(&region, 0, sizeof(VkBufferCopy));
            region.size      = srcTensor->elementSize() * 4;
            region.srcOffset = 0;
            region.dstOffset = 0;
            vkCmdCopyBuffer(cmdBuffer->get(), (VkBuffer)srcTensor->deviceId(), destBuffer, 1, &region);
            return;
        }
        MNN_ASSERT(false);
        return;
    }
    auto tensor = srcTensor;
    _setUpPipeline(sourceFormat, destFormat, IMAGE_TO_BUFFER, tensor->buffer().type);
    _encodeImageBufferConvert(tensor, destBuffer, bufferSize, bufferOffset, cmdBuffer,
                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

} // namespace MNN
