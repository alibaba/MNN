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
namespace MNN {

class VulkanMatMul : public VulkanBasicExecution {
public:
    VulkanMatMul(bool transposeA, bool transposeB, Backend* vkBn, bool hasBias);
    ~ VulkanMatMul() {
        // Do nothing
    }
    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                               const VulkanCommandPool::Buffer *cmdBuffer) override;

private:
    VulkanRaster::Componet mInput;
    VulkanRaster::Componet mKernel;
    VulkanRaster::Componet mOutput;

    const VulkanPipeline* mBlitPipeline;
    const VulkanPipeline* mComputePipeline;
    std::shared_ptr<VulkanPipeline::DescriptorSet> mComputeSet;
    const VulkanPipeline* mOutputPipeline;
    std::vector<std::shared_ptr<VulkanBuffer>> mTempBuffer;
    bool mTransposeA;
    bool mTransposeB;
};
}
#endif
