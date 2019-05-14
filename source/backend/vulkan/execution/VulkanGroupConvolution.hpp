//
//  VulkanGroupConvolution.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef VulkanGroupConvolution_hpp
#define VulkanGroupConvolution_hpp

#include "VulkanConvolutionImpl.hpp"
namespace MNN {
class VulkanGroupConvolution : public Execution {
public:
    VulkanGroupConvolution(const Op *op, Backend *backend);
    virtual ~VulkanGroupConvolution();
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    VulkanBackend *mBackend;
    Tensor mTempSrc;
    Tensor mTempDst;
    std::vector<Tensor *> mTempInputs;
    std::vector<Tensor *> mTempOutputs;
    const Convolution2D *mConvParameter;
    std::vector<std::tuple<std::shared_ptr<VulkanCommandPool::Buffer>, std::shared_ptr<Execution>,
                           std::shared_ptr<VulkanCommandPool::Buffer>>>
        mSubConvolutions;
};

} // namespace MNN

#endif /* VulkanGroupConvolution_hpp */
