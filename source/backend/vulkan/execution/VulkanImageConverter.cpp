//
//  VulkanImageConverter.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanImageConverter.hpp"
#include "Macro.h"
#include "TensorUtils.hpp"
#include "VulkanBackend.hpp"
namespace MNN {

VulkanTensorConvert::VulkanTensorConvert(const VulkanBackend* bn) : mVulkanBackend(bn) {
    mTensorConvertUniformBuffer.reset(
        new VulkanBuffer(bn->getMemoryPool(), false, sizeof(Uniforms), nullptr, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT));
}

VulkanTensorConvert::~VulkanTensorConvert() {
}

bool VulkanTensorConvert::encodeTensorConvert(VkBuffer source, VkBuffer dest, const Tensor* srcShape,
                                              MNN_DATA_FORMAT sourceDimensionFormat,
                                              MNN_DATA_FORMAT destDimensionFormat, VkDeviceSize srcOffset,
                                              VkDeviceSize destOffset, VkDeviceSize srcSize, VkDeviceSize dstSize,
                                              const VulkanCommandPool::Buffer* cmdBuffer) {
    auto tensorConvertParam     = reinterpret_cast<Uniforms*>(mTensorConvertUniformBuffer->map());
    tensorConvertParam->width   = std::max(1, srcShape->width());
    tensorConvertParam->height  = std::max(1, srcShape->height());
    tensorConvertParam->channel = srcShape->channel();
    tensorConvertParam->batch   = srcShape->batch();

    // const uint8_t* codeBuffer = nullptr;
    // size_t codeBufferLength = 0;
    std::string key = "";
    if (MNN_DATA_FORMAT_NHWC == sourceDimensionFormat) {
        if (MNN_DATA_FORMAT_NC4HW4 == destDimensionFormat) {
            // codeBuffer = glsl_nhwcTonc4hw4_comp;
            // codeBufferLength = glsl_nhwcTonc4hw4_comp_len;
            key = "glsl_nhwcTonc4hw4_comp";
        }
    } else if (MNN_DATA_FORMAT_NC4HW4 == sourceDimensionFormat) {
        if (MNN_DATA_FORMAT_NHWC == destDimensionFormat) {
            // codeBuffer = glsl_nc4hw4Tonhwc_comp;
            // codeBufferLength = glsl_nc4hw4Tonhwc_comp_len;
            key = "glsl_nc4hw4Tonhwc_comp";
        } else if (MNN_DATA_FORMAT_NCHW == destDimensionFormat) {
            // codeBuffer = glsl_nc4hw4Tonchw_comp;
            // codeBufferLength = glsl_nc4hw4Tonchw_comp_len;
            key = "glsl_nc4hw4Tonchw_comp";
        }
    } else if (MNN_DATA_FORMAT_NCHW == sourceDimensionFormat) {
        if (MNN_DATA_FORMAT_NC4HW4 == destDimensionFormat) {
            // codeBuffer = glsl_nchwTonc4hw4_comp;
            // codeBufferLength = glsl_nchwTonc4hw4_comp_len;
            key = "glsl_nchwTonc4hw4_comp";
        }
    }

    // MNN_ASSERT(nullptr != codeBuffer);
    std::vector<VkDescriptorType> desTypes{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                           VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};

    mTensorConvertPipeline = mVulkanBackend->getPipeline(key, /*codeBuffer, codeBufferLength,*/ desTypes);
    mTensorConvertDescriptorSet.reset(mTensorConvertPipeline->createSet());
    mTensorConvertDescriptorSet->writeBuffer(source, 0, srcSize, srcOffset);
    mTensorConvertDescriptorSet->writeBuffer(dest, 1, dstSize, destOffset);
    mTensorConvertDescriptorSet->writeBuffer(mTensorConvertUniformBuffer->buffer(), 2,
                                             mTensorConvertUniformBuffer->size());

    mTensorConvertPipeline->bind(cmdBuffer->get(), mTensorConvertDescriptorSet->get());
    cmdBuffer->barrierSource(source, srcOffset, srcSize);
    vkCmdDispatch(cmdBuffer->get(), UP_DIV(tensorConvertParam->width, 16), UP_DIV(tensorConvertParam->height, 16),
                  UP_DIV(tensorConvertParam->channel, 4) * tensorConvertParam->batch);

    mTensorConvertUniformBuffer->unmap();
    return true;
}

VulkanImageConverter::VulkanImageConverter(const VulkanBackend* bn) : mBufferConverter(bn) {
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
        mBufferConverter.encodeTensorConvert(srcBuffer, vkTensor->buffer()->buffer(), destTensor, sourceFormat,
                                             destFormat, bufferOffset, 0, bufferSize, vkTensor->buffer()->size(),
                                             cmdBuffer);
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
        mBufferConverter.encodeTensorConvert(vkTensor->buffer()->buffer(), destBuffer, srcTensor, sourceFormat,
                                             destFormat, 0, bufferOffset, vkTensor->buffer()->size(), bufferSize,
                                             cmdBuffer);
        return;
    }
    auto tensor = srcTensor;
    _setUpPipeline(sourceFormat, destFormat, IMAGE_TO_BUFFER, tensor->buffer().type);
    _encodeImageBufferConvert(tensor, destBuffer, bufferSize, bufferOffset, cmdBuffer,
                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

} // namespace MNN
