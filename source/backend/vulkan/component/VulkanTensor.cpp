//
//  VulkanTensor.cpp
//  MNN
//
//  Created by MNN on 2020/03/03.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanTensor.hpp"
#include "core/TensorUtils.hpp"
namespace MNN {
int VulkanTensor::getAlignSize(const Tensor* tensor) {
    auto format      = TensorUtils::getDescribe(tensor)->dimensionFormat;
    auto elementSize = tensor->elementSize();
    // [TODO] Find a better way
    if (format == MNN_DATA_FORMAT_NCHW) {
        if (tensor->dimensions() >= 2) {
            MNN_ASSERT(tensor->channel() > 0);
            return elementSize / tensor->channel() * ALIGN_UP4(tensor->channel());
        }
    } else if (format == MNN_DATA_FORMAT_NHWC) {
        if (tensor->dimensions() >= 4) {
            MNN_ASSERT(tensor->channel() > 0);
            return elementSize / tensor->channel() * ALIGN_UP4(tensor->channel());
        }
    }
    return ALIGN_UP4(elementSize);
}

VulkanTensor::VulkanTensor(const Tensor* shape, const VulkanMemoryPool& pool, bool forceBuffer, bool seperate) {
    auto format = TensorUtils::getDescribe(shape)->dimensionFormat;
    if (MNN_DATA_FORMAT_NC4HW4 == format && !forceBuffer) {
        mImage = std::make_shared<VulkanImage>(pool, seperate,
                                               std::vector<int>{
                                                   std::max(shape->width(), 1),
                                                   std::max(shape->height(), 1),
                                                   UP_DIV(shape->channel(), 4) * shape->batch(),
                                               },
                                               shape->getType());
    } else {
        // Compute Shader don't support uint8 / int8 / float16 / uint64, all use int32/float32
        mBuffer = std::make_shared<VulkanBuffer>(pool, seperate, getAlignSize(shape) * sizeof(float));
    }
}
void VulkanTensor::release() {
    if (nullptr != mBuffer.get()) {
        mBuffer->release();
    }
    if (nullptr != mImage.get()) {
        mImage->release();
    }
}
uint64_t VulkanTensor::deviceId() {
    if (mImage.get()) {
        return reinterpret_cast<uint64_t>(mImage->view());
    } else {
        return reinterpret_cast<uint64_t>(mBuffer->buffer());
    }
}
}
