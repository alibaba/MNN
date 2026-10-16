//
//  VulkanMatMul.hpp
//  MNN
//
//  Created by MNN on 2020/03/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef VulkanMatMul_hpp
#define VulkanMatMul_hpp

#include "VulkanRaster.hpp"
#include "VulkanLoop.hpp"
#ifdef MNN_VULKAN_ARM_NX
#include "VulkanArmNx.hpp"
#endif
namespace MNN {

class VulkanMatMul : public VulkanBasicExecution {
public:
    VulkanMatMul(bool transposeA, bool transposeB, Backend* vkBn, bool hasBias);
    ~ VulkanMatMul();
    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                               const VulkanCommandPool::Buffer *cmdBuffer) override;

private:
    const VulkanPipeline* mPipeline;
    std::shared_ptr<VulkanBuffer> mParam;
    std::shared_ptr<VulkanLayout::DescriptorSet> mDescribe;
    bool mTransposeA;
    bool mTransposeB;
    bool mHasBias;
#ifdef MNN_VULKAN_ARM_NX
    std::unique_ptr<VulkanArmNx> mArmNx;
    std::array<uint32_t, 3> mArmNxShape{};
#endif
};
}
#endif
