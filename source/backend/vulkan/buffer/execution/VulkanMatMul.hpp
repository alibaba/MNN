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
};
}
#endif
