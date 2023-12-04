//
//  VulkanConvolution.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef VulkanConvolution_hpp
#define VulkanConvolution_hpp

#include "VulkanBasicExecution.hpp"
#include "core/ConvolutionCommon.hpp"
namespace MNN {
class VulkanConvolutionCommon : public VulkanBasicExecution {
public:
    VulkanConvolutionCommon(const Op* op, Backend* bn);
    virtual ~VulkanConvolutionCommon();

    virtual ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                               const VulkanCommandPool::Buffer* cmdBuffer) override;

    struct ConvolutionParameter {
        ivec2 pad;
        ivec2 kernelSize;
        ivec2 stride;
        ivec2 dilate;
        ivec4 inputSize;
        ivec4 outputSize;
        ivec4 offset;
    };

    static void writeParameter(ConvolutionParameter* dest, const Convolution2DCommon* common, const Tensor* input,
                               const Tensor* output);
    static std::string getPostTreatMacro(const Convolution2DCommon* common);
    class BufferToImageCopy {
    public:
        BufferToImageCopy(const VulkanBackend* bn) {
            mBackend = bn;
            std::vector<VkDescriptorType> types{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                                VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
            mPipeline = mBackend->getPipeline("glsl_buffer2Image2D_comp", types);
            mSets.reset(mPipeline->createSet());
            mConstBuffer = std::make_shared<VulkanBuffer>(bn->getMemoryPool(), false, 2 * sizeof(int),
                                                              nullptr, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        }
        void encode(const VulkanImage* image, VkBuffer buffer, size_t bufferSize, const VulkanCommandPool::Buffer* cmdBuffer) {
            int localX = 16;
            int localY = 16;
            int localZ = 1;
            int* dim = (int*)mConstBuffer->map();
            dim[0] = image->width();
            dim[1] = image->height();
            mConstBuffer->unmap();
            image->barrierWrite(cmdBuffer->get());
            mSets->writeImage(image->view(), mBackend->getCommonSampler()->get(), VK_IMAGE_LAYOUT_GENERAL, 0);
            mSets->writeBuffer(buffer, 1, bufferSize);
            mSets->writeBuffer(mConstBuffer->buffer(), 2, mConstBuffer->size());
            mPipeline->bind(cmdBuffer->get(), mSets->get());
            cmdBuffer->barrierSource(buffer, 0, bufferSize);
            vkCmdDispatch(cmdBuffer->get(), UP_DIV(image->width(), localX), UP_DIV(image->height(), localY),
                          UP_DIV(image->depth(), localZ));
        }
    private:
        const VulkanBackend* mBackend;
        const VulkanPipeline* mPipeline;
        std::shared_ptr<VulkanLayout::DescriptorSet> mSets;
        std::shared_ptr<VulkanBuffer> mConstBuffer;
    };
    static int gImage2ColLocal;
protected:
    virtual ErrorCode onEncodeConvolution(const Convolution2DCommon* common, const std::vector<Tensor*>& inputs,
                                          const std::vector<Tensor*>& outputs,
                                          const VulkanCommandPool::Buffer* cmdBuffer,
                                          const VulkanBuffer* constConvBuffer) = 0;

private:
    const Convolution2DCommon* mCommon;
    std::shared_ptr<VulkanBuffer> mConvCons;
};

class VulkanConvolutionDepthwise : public VulkanConvolutionCommon {
public:
    VulkanConvolutionDepthwise(const float* weightData, size_t weightSize, const Op* op, Backend* bn);
    virtual ~VulkanConvolutionDepthwise();
    virtual ErrorCode onEncodeConvolution(const Convolution2DCommon* common, const std::vector<Tensor*>& inputs,
                                          const std::vector<Tensor*>& outputs,
                                          const VulkanCommandPool::Buffer* cmdBuffer,
                                          const VulkanBuffer* constConvBuffer) override;

private:
    bool _init(const float* weightData, size_t weightSize, const Op* op, Backend* bn);
    std::shared_ptr<VulkanImage> mKernel;

    const VulkanPipeline* mConvPipeline;

    std::shared_ptr<VulkanLayout::DescriptorSet> mConvSet;
    const VulkanSampler* mSampler;
    std::shared_ptr<VulkanImage> mBias;
    std::vector<std::shared_ptr<VulkanLayout::DescriptorSet>> mExtraSets;
    std::vector<std::shared_ptr<VulkanBuffer>> mExtraBuffers;

    int mLocalX = 0;
    int mLocalY = 0;
};
} // namespace MNN

#endif /* VulkanConvolution_hpp */
