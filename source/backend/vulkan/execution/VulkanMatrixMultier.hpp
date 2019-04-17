//
//  VulkanMatrixMultier.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef VulkanMatrixMultier_hpp
#define VulkanMatrixMultier_hpp

#include <stdio.h>
#include "VulkanBasicExecution.hpp"
namespace MNN {
class VulkanMatrixMultier : public NonCopyable {
public:
    virtual ~VulkanMatrixMultier();

    VulkanMatrixMultier(VulkanBackend* backend, const float* B, int w, int h, int c = 1);
    void prepare(int srcHeight);

    void compute(const VulkanCommandPool::Buffer* commandBuffer) const;

    inline const VulkanImage* source() const {
        return mSource.get();
    }
    inline const VulkanImage* dest() const {
        return mDest.get();
    }

private:
    const VulkanPipeline* mPipeline;
    std::shared_ptr<VulkanPipeline::DescriptorSet> mDescriptorSet;
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

#endif /* VulkanMatrixMultier_hpp */
