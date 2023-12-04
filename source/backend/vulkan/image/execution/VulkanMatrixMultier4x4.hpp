//
//  VulkanMatrixMultier4x4.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef VulkanMatrixMultier4x4_hpp
#define VulkanMatrixMultier4x4_hpp

#include <stdio.h>
#include "VulkanBasicExecution.hpp"
namespace MNN {
class VulkanMatrixMultier4x4 : public NonCopyable {
public:
    virtual ~VulkanMatrixMultier4x4();
    static std::shared_ptr<VulkanImage> createKernel(VulkanBackend* backend, const float* B, int l, int h, int c);
    VulkanMatrixMultier4x4(VulkanBackend* backend, const float* B, int l, int h, int c = 1, std::shared_ptr<VulkanImage> kernel = nullptr);
    void prepare(const VulkanCommandPool::Buffer* commandBuffer, int e, std::shared_ptr<VulkanImage> dst = nullptr, std::shared_ptr<VulkanImage> src = nullptr);

    void compute(const VulkanCommandPool::Buffer* commandBuffer) const;

    inline const VulkanImage* source() const {
        return mSource.get();
    }
    inline const VulkanImage* dest() const {
        return mDest.get();
    }

private:
    const VulkanPipeline* mPipeline;
    std::shared_ptr<VulkanLayout::DescriptorSet> mDescriptorSet;
    VulkanBackend* mBackend;

    std::shared_ptr<VulkanImage> mKernel;
    const VulkanSampler* mSampler;
    std::shared_ptr<VulkanBuffer> mConstBuffer;

    std::shared_ptr<VulkanImage> mDest;
    std::shared_ptr<VulkanImage> mSource;

    int mWidth;
    int mHeight;
    int mDepth;

    int mOutputWidth  = 0;
    int mOutputHeight = 0;
};
} // namespace MNN

#endif /* VulkanMatrixMultier4x4_hpp */
