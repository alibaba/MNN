//
//  VulkanConvolutionWinograd.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef VulkanConvolutionWinograd_hpp
#define VulkanConvolutionWinograd_hpp

#include "VulkanConvolutionImpl.hpp"
#include "VulkanMatrixMultier4x4.hpp"

namespace MNN {
class VulkanConvolutionWinograd : public VulkanBasicExecution {
public:
    VulkanConvolutionWinograd(VulkanBackend* backend, const Convolution2DCommon* convOption, const float* weightPtr,
                              const float* biasPtr, int ci, int co);
    virtual ~VulkanConvolutionWinograd();

    virtual ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                               const VulkanCommandPool::Buffer* cmdBuffer) override;
    static bool support(const Convolution2DCommon* convOption);

private:
    std::shared_ptr<VulkanMatrixMultier4x4> mMultier;
    std::shared_ptr<VulkanImage> mBias;

    VulkanBackend* mBackend = nullptr;

    const VulkanPipeline* mSourceTransform = nullptr;
    const VulkanPipeline* mDestTransform   = nullptr;
    const VulkanSampler* mSampler          = nullptr;

    std::vector<std::shared_ptr<VulkanLayout::DescriptorSet>> mSourceTransformSet;
    std::vector<std::shared_ptr<VulkanLayout::DescriptorSet>> mDestTransformSet;
    std::shared_ptr<VulkanBuffer> mWinogradConst;

    ivec3 mTransformLocalSize;
    const Convolution2DCommon* mCommon;
    std::vector<std::shared_ptr<VulkanBuffer>> mOffsetsBuffer;

    int mUnit = 0;
};
} // namespace MNN
#endif /* VulkanConvolutionWinograd_hpp */
