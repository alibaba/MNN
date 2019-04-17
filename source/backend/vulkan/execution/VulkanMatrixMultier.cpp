//
//  VulkanMatrixMultier.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanMatrixMultier.hpp"
#include "Macro.h"
namespace MNN {
struct constUniform {
    ivec4 outputSize;
    int multiLength;
};

VulkanMatrixMultier::~VulkanMatrixMultier() {
}

VulkanMatrixMultier::VulkanMatrixMultier(VulkanBackend* backend, const float* B, int w, int h, int c) {
    mBackend     = backend;
    mWidth       = w;
    mHeight      = h;
    mDepth       = c;
    mConstBuffer = std::make_shared<VulkanBuffer>(backend->getMemoryPool(), false, sizeof(constUniform), nullptr,
                                                  VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

    {
        std::vector<VkDescriptorType> types{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
        if (backend->gpuType() == VulkanBackend::ADRENO || backend->gpuType() == VulkanBackend::MALI) {
            mPipeline = mBackend->getPipeline("glsl_gemm16x16Half_comp",
                                              /*glsl_gemm16x16Half_comp, glsl_gemm16x16Half_comp_len,*/ types);
        } else {
            mPipeline =
                mBackend->getPipeline("glsl_gemm16x16_comp", /*glsl_gemm16x16_comp, glsl_gemm16x16_comp_len,*/ types);
        }
    }
    mDescriptorSet.reset(mPipeline->createSet());

    mKernel  = std::make_shared<VulkanImage>(backend->getMemoryPool(), false,
                                            std::vector<int>{ALIGN_UP4(w), UP_DIV(h, 4) * c});
    mSampler = mBackend->getCommonSampler();

    // Compute mKernel
    auto tempBuffer = std::make_shared<VulkanBuffer>(backend->getMemoryPool(), false,
                                                     ALIGN_UP4(w) * ALIGN_UP4(h) * c * sizeof(float));
    {
        auto dest = tempBuffer->map();
        ::memcpy(dest, B, ALIGN_UP4(w) * ALIGN_UP4(h) * c * sizeof(float));
        tempBuffer->unmap();
    }
    mBackend->copyBufferToImage(tempBuffer.get(), mKernel.get());
}
void VulkanMatrixMultier::prepare(int srcHeight) {
    int sw  = ALIGN_UP4(mWidth);
    int sh  = UP_DIV(srcHeight, 4);
    mSource = std::make_shared<VulkanImage>(mBackend->getDynamicMemoryPool(), false, std::vector<int>{sw, sh * mDepth});
    int ow  = sh;
    int oh  = ALIGN_UP4(mHeight);
    mDest   = std::make_shared<VulkanImage>(mBackend->getDynamicMemoryPool(), false, std::vector<int>{ow * mDepth, oh});

    MNN_ASSERT(nullptr != mSource && nullptr != mDest);

    mSource->release();
    mDest->release();

    mDescriptorSet->writeImage(mDest->view(), mSampler->get(), VK_IMAGE_LAYOUT_GENERAL, 0);
    mDescriptorSet->writeImage(mSource->view(), mSampler->get(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1);
    mDescriptorSet->writeImage(mKernel->view(), mSampler->get(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 2);

    {
        auto uniform           = (constUniform*)mConstBuffer->map();
        uniform->outputSize[0] = ow;
        uniform->outputSize[1] = oh / 4;
        uniform->multiLength   = sw / 4;
        mConstBuffer->unmap();
    }
    mDescriptorSet->writeBuffer(mConstBuffer->buffer(), 3, mConstBuffer->size());
    mOutputWidth  = ow;
    mOutputHeight = oh;
}

void VulkanMatrixMultier::compute(const VulkanCommandPool::Buffer* commandBuffer) const {
    mPipeline->bind(commandBuffer->get(), mDescriptorSet->get());
    commandBuffer->barrierImage(mSource->get(), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    vkCmdDispatch(commandBuffer->get(), UP_DIV(mOutputWidth, 8), UP_DIV(mOutputHeight / 4, 8), mDepth);
}
} // namespace MNN
