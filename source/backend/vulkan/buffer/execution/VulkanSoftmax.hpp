//
//  VulkanSoftmax.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef VulkanSoftmax_hpp
#define VulkanSoftmax_hpp

#include <stdio.h>
#include "VulkanBasicExecution.hpp"
#include "VulkanRaster.hpp"

namespace MNN {
class VulkanSoftmax : public VulkanBasicExecution {
public:
    VulkanSoftmax(const Op* op, Backend* bn);
    virtual ~VulkanSoftmax();
    virtual ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                       const VulkanCommandPool::Buffer* cmdBuffer) override;

private:
    std::shared_ptr<VulkanBuffer> mConstBuffer;
    const VulkanPipeline* mSoftmaxPipeline;
    std::shared_ptr<VulkanLayout::DescriptorSet> mDescriptorSet;
    int mAxis;
    struct ConvertComponent {
        std::shared_ptr<Tensor> mTempInputTensor;
        std::shared_ptr<Tensor> mTempOutputTensor;
        VulkanRaster::Componet mInputConvert;
        VulkanRaster::Componet mOutputConvert;
    };
    std::shared_ptr<ConvertComponent> mConvert;
};

} // namespace MNN

#endif /* VulkanSoftmax_hpp */
