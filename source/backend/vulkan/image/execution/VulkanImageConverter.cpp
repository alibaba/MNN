//
//  VulkanImageConverter.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanImageConverter.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "VulkanBackend.hpp"
namespace MNN {
VulkanImageConverter::VulkanImageConverter(const VulkanBackend* bn) {
    mBackend = bn;
    mSampler = bn->getCommonSampler();
    mConst.reset(
        new VulkanBuffer(bn->getMemoryPool(), false, 8 * sizeof(int), nullptr, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT));
}

void VulkanImageConverter::_setUpPipeline(MNN_DATA_FORMAT sourceFormat, MNN_DATA_FORMAT destFormat, TYPE type,
                                          halide_type_t datatype) {
    if (nullptr != mPipeline && sourceFormat == mCurrentSource && destFormat == mCurrentDest && mConvertImage == type) {
        return;
    }
    mCurrentDest   = destFormat;
    mCurrentSource = sourceFormat;
    mConvertImage  = type;

    std::vector<VkDescriptorType> types{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
    std::string name;
    if (type == BUFFER_TO_IMAGE) {
        types[0] = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        if (sourceFormat == MNN_DATA_FORMAT_NC4HW4) {
            name = "glsl_nc4hw4toimage_comp";
        } else {
            name = "glsl_nchwToimage_comp";
        }
    } else {
        if (destFormat == MNN_DATA_FORMAT_NC4HW4) {
            name = "glsl_imageTonc4hw4_comp";
        } else {
            name = "glsl_imageTonchw_comp";
        }
    }
//    FUNC_PRINT_ALL(name.c_str(), s);
    mPipeline = mBackend->getPipeline(name, types);
    MNN_ASSERT(nullptr != mPipeline);
}

void VulkanImageConverter::encodeBufferToTensor(VkBuffer srcBuffer, const Tensor* destTensor, const int bufferSize,
                                                VkDeviceSize bufferOffset, MNN_DATA_FORMAT srcBufferFormat,
                                                const VulkanCommandPool::Buffer* cmdBuffer) {
    auto destFormat   = TensorUtils::getDescribe(destTensor)->dimensionFormat;
    auto sourceFormat = srcBufferFormat;
    cmdBuffer->barrierSource(srcBuffer, 0, bufferSize);
    auto tensor = destTensor;
    _setUpPipeline(sourceFormat, destFormat, BUFFER_TO_IMAGE, tensor->buffer().type);
    auto vkTensor = (VulkanTensor*)(destTensor->deviceId());
    for (int i=0; i<vkTensor->imageSize(); ++i) {
        vkTensor->image(i)->barrierWrite(cmdBuffer->get());
    }
    _encodeImageBufferConvert(tensor, srcBuffer, bufferSize, bufferOffset, cmdBuffer, VK_IMAGE_LAYOUT_GENERAL, srcBufferFormat);
}
void VulkanImageConverter::_encodeImageBufferConvert(const Tensor* tensor, VkBuffer destBuffer, const int bufferSize, VkDeviceSize bufferOffset, const VulkanCommandPool::Buffer* cmdBuffer, VkImageLayout layout, MNN_DATA_FORMAT bufferFormat) {
    auto dims = (int*)mConst->map();// W, H, C, N
    auto nhwc = VulkanTensor::tensorShapeFormat(tensor);
    dims[0] = nhwc[2];
    dims[1] = nhwc[1];
    dims[2] = nhwc[3];
    dims[3] = nhwc[0];
    // Set stride for W, H, C, N
    if (bufferFormat == MNN_DATA_FORMAT_NHWC) {
        dims[4] = nhwc[3];
        dims[5] = nhwc[3] * nhwc[2];
        dims[6] = 1;
        dims[7] = nhwc[3] * nhwc[2] * nhwc[1];
    } else {
        dims[4] = 1;
        dims[5] = nhwc[2];
        dims[6] = nhwc[2] * nhwc[1];
        dims[7] = nhwc[3] * nhwc[2] * nhwc[1] ;
    }
    mConst->unmap();
    auto vkTensor = reinterpret_cast<VulkanTensor*>(tensor->deviceId());
    auto& mBlocks = vkTensor->blocks();
    auto& limits = mBackend->proty().limits;
    int wUnit = limits.maxImageDimension2D;
    int hUnit = limits.maxImageDimension2D;

    struct OffsetBuffer {
        int offset[4]; // Offset w, h, c, n
        int size[4];//w, h, c, w*h*c
    };
    mSet.resize(vkTensor->imageSize());
    mOffset.resize(vkTensor->imageSize());
    for (int y=0; y<mBlocks[1]; ++y) {
        auto ySta = y * hUnit;
        for (int x=0; x<mBlocks[0]; ++x) {
            auto xSta = x * wUnit;
            OffsetBuffer offset;
            offset.offset[0] = xSta;
            offset.offset[1] = ySta;
            auto index = y*mBlocks[0] + x;
            auto image = vkTensor->image(index);
            offset.size[0] = image->width();
            offset.size[1] = image->height();
            offset.size[2] = 0;
            offset.size[3] = image->width() * image->height();
            mOffset[index].reset(new VulkanBuffer(mBackend->getMemoryPool(), false, sizeof(offset), &offset, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT));
            mSet[index].reset(mPipeline->createSet());
            mSet[index]->writeImage(image->view(), mSampler->get(), layout, 0);
            mSet[index]->writeBuffer(destBuffer, 1, bufferSize, bufferOffset);
            mSet[index]->writeBuffer(mConst->buffer(), 2, mConst->size());
            mSet[index]->writeBuffer(mOffset[index]->buffer(), 3, mOffset[index]->size());
            mPipeline->bind(cmdBuffer->get(), mSet[index]->get());
            vkCmdDispatch(cmdBuffer->get(), UP_DIV(offset.size[3], 256), 1, 1);
        }
    }

}

void VulkanImageConverter::encodeTensorToBuffer(const Tensor* srcTensor, VkBuffer destBuffer, const int bufferSize,
                                                VkDeviceSize bufferOffset, MNN_DATA_FORMAT destBufferFormat,
                                                const VulkanCommandPool::Buffer* cmdBuffer) {
    auto sourceFormat = TensorUtils::getDescribe(srcTensor)->dimensionFormat;
    auto destFormat   = destBufferFormat;

    auto vkTensor = (VulkanTensor*)(srcTensor->deviceId());
    auto tensor = srcTensor;
    _setUpPipeline(sourceFormat, destFormat, IMAGE_TO_BUFFER, tensor->buffer().type);
    for (int i=0; i<vkTensor->imageSize(); ++i) {
        vkTensor->image(i)->barrierRead(cmdBuffer->get());
    }
    _encodeImageBufferConvert(tensor, destBuffer, bufferSize, bufferOffset, cmdBuffer,
                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, destBufferFormat);
}

MNN_DATA_FORMAT VulkanImageConverter::getTensorLinearFormat(const Tensor* tensor) {
    auto format = TensorUtils::getDescribe(tensor)->dimensionFormat;
    if (MNN_DATA_FORMAT_NC4HW4 == format) {
        return MNN_DATA_FORMAT_NCHW;
    }
    return format;
}

} // namespace MNN
