//
//  VulkanMatMul.hpp
//  MNN
//
//  Created by MNN on 2020/03/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef VulkanMatMul_hpp
#define VulkanMatMul_hpp

#include "VulkanMatrixMultier4x4.hpp"
namespace MNN {

class VulkanMatMul : public VulkanBasicExecution {
public:
    class Reorder {
    public:
        struct nchwBuffer {
            ivec4 size;
            ivec4 stride;
        };
        Reorder(const VulkanBackend* bn, bool reorder, bool revert = false);
        ~ Reorder() {
            // Do nothing
        }
        int computeMiddleBufferSize(int b, int h, int w, int c) const;
        void encode(VkBuffer source, size_t sourceSize, VkBuffer middleBuffer, size_t middelBufferSize, const VulkanImage* dest, const VulkanCommandPool::Buffer* cmdBuffer, const nchwBuffer& buffer);
        void revert(VkBuffer dest, size_t destSize, VkBuffer middleBuffer, size_t middelBufferSize, const VulkanImage* source, const VulkanCommandPool::Buffer* cmdBuffer, const nchwBuffer& buffer);
    private:
        const VulkanPipeline* mFirst;
        const VulkanPipeline* mSecond;
        std::shared_ptr<VulkanPipeline::DescriptorSet> mBufferBufferSet;
        std::shared_ptr<VulkanPipeline::DescriptorSet> mImageBufferSet;
        const VulkanBackend* mBackend;
        std::shared_ptr<VulkanBuffer> mUnitBuffer;
    };
    VulkanMatMul(bool transposeA, bool transposeB, Backend* vkBn);
    ~ VulkanMatMul() {
        // Do nothing
    }
    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                               const VulkanCommandPool::Buffer *cmdBuffer) override;

private:
    std::vector<std::shared_ptr<VulkanBuffer>> mTempBuffer;
    std::shared_ptr<VulkanMatrixMultier4x4> mCore;
    bool mTransposeA;
    bool mTransposeB;
    std::vector<const VulkanPipeline*> mPipelines;
    std::vector<std::shared_ptr<VulkanPipeline::DescriptorSet>> mSets;
    std::shared_ptr<Reorder> mInputReorder;
    std::shared_ptr<Reorder> mWeightReorder;
    std::shared_ptr<Reorder> mOutputReorder;
    std::shared_ptr<VulkanImage> mKernelImage;
    std::shared_ptr<VulkanImage> mInputImage;
    std::shared_ptr<VulkanImage> mOutputImage;
};
}
#endif
