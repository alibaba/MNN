//
//  VulkanMatrixMultier4x4.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanMatrixMultier4x4.hpp"
#include "core/Macro.h"
namespace MNN {
struct constUniform {
    ivec4 outputSize;
};

VulkanMatrixMultier4x4::~VulkanMatrixMultier4x4() {
}
std::shared_ptr<VulkanImage> VulkanMatrixMultier4x4::createKernel(VulkanBackend* backend, const float* B, int l, int h, int c) {

    auto kernel  = std::make_shared<VulkanImage>(backend->getMemoryPool(), false,
                                            std::vector<int>{ALIGN_UP4(l), UP_DIV(h, 4) * c});
    if (nullptr == B) {
        return kernel;
    }

    // Compute mKernel
    auto tempBuffer = std::make_shared<VulkanBuffer>(backend->getMemoryPool(), false,
                                                     ALIGN_UP4(l) * ALIGN_UP4(h) * c * sizeof(float));
    {
        auto dest = tempBuffer->map();
        ::memcpy(dest, B, ALIGN_UP4(l) * ALIGN_UP4(h) * c * sizeof(float));
        tempBuffer->unmap();
    }
    backend->copyBufferToImage(tempBuffer.get(), kernel.get(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    return kernel;
}

VulkanMatrixMultier4x4::VulkanMatrixMultier4x4(VulkanBackend* backend, const float* B, int l, int h, int c,  std::shared_ptr<VulkanImage> kernel) {
    mBackend     = backend;
    mWidth       = l;
    mHeight      = h;
    mDepth       = c;
    mConstBuffer = std::make_shared<VulkanBuffer>(backend->getMemoryPool(), false, sizeof(constUniform), nullptr,
                                                  VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

    {
        std::vector<VkDescriptorType> types{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
        bool supportFp16 = backend->getMemoryPool().permitFp16();
        if ((backend->gpuType() == VulkanRuntime::ADRENO || backend->gpuType() == VulkanRuntime::MALI) && supportFp16) {
            mPipeline = mBackend->getPipeline("glsl_gemm16x16_FP16_comp", types);
        } else {
            mPipeline = mBackend->getPipeline("glsl_gemm16x16_comp", types);
        }
    }
    mDescriptorSet.reset(mPipeline->createSet());
    mSampler = mBackend->getCommonSampler();
    if (nullptr == kernel) {
        kernel = createKernel(backend, B, l, h, c);
    }
    mKernel = kernel;
}
void VulkanMatrixMultier4x4::prepare(const VulkanCommandPool::Buffer* commandBuffer, int e, std::shared_ptr<VulkanImage> dst, std::shared_ptr<VulkanImage> src) {
    int sw  = ALIGN_UP4(mWidth);
    int sh  = UP_DIV(e, 4);
    int ow  = sh;
    int oh  = ALIGN_UP4(mHeight);
    mSource = src;
    mDest = dst;
    if (nullptr == dst) {
        mDest   = std::make_shared<VulkanImage>(mBackend->getDynamicMemoryPool(), false, std::vector<int>{oh, ow * mDepth});
    }
    if (nullptr == src) {
        mSource = std::make_shared<VulkanImage>(mBackend->getDynamicMemoryPool(), false, std::vector<int>{sw, sh * mDepth});
    }
    if (VK_IMAGE_LAYOUT_UNDEFINED == mSource->currentLayout()) {
        mSource->barrierRead(commandBuffer->get());
    }
    if (VK_IMAGE_LAYOUT_UNDEFINED == mDest->currentLayout()) {
        mDest->barrierRead(commandBuffer->get());
    }
    MNN_ASSERT(nullptr != mSource && nullptr != mDest);
    if (nullptr == src) {
        mSource->release();
    }
    if (nullptr == dst) {
        mDest->release();
    }
    mDescriptorSet->writeImage(mDest->view(), mSampler->get(), VK_IMAGE_LAYOUT_GENERAL, 0);
    mDescriptorSet->writeImage(mSource->view(), mSampler->get(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1);
    mDescriptorSet->writeImage(mKernel->view(), mSampler->get(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 2);

    {
        auto uniform           = (constUniform*)mConstBuffer->map();
        uniform->outputSize[0] = ow;
        uniform->outputSize[1] = oh / 4;
        uniform->outputSize[3]   = sw / 4;
        mConstBuffer->unmap();
    }
    mDescriptorSet->writeBuffer(mConstBuffer->buffer(), 3, mConstBuffer->size());
    mOutputWidth  = ow;
    mOutputHeight = oh;
}

void VulkanMatrixMultier4x4::compute(const VulkanCommandPool::Buffer* commandBuffer) const {
    mPipeline->bind(commandBuffer->get(), mDescriptorSet->get());
    mDest->barrierWrite(commandBuffer->get());
    mSource->barrierRead(commandBuffer->get());
    mKernel->barrierRead(commandBuffer->get());
    vkCmdDispatch(commandBuffer->get(), UP_DIV(mOutputWidth, 8), UP_DIV(mOutputHeight / 4, 8), mDepth);
}

} // namespace MNN
