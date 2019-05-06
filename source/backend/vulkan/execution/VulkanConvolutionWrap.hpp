//
//  VulkanConvolutionWrap.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef VulkanConvolutionWrap_hpp
#define VulkanConvolutionWrap_hpp

#include <stdio.h>
#include "VulkanConvolution.hpp"
#include "VulkanConvolutionImpl.hpp"
namespace MNN {
class VulkanConvolutionWrap : public Execution {
public:
    VulkanConvolutionWrap(const Op *op, Backend *backend);
    virtual ~VulkanConvolutionWrap();
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    std::shared_ptr<Execution> mEncodeConvolution;
    const Convolution2D *mConvParameter;
};
} // namespace MNN

#endif /* VulkanConvolutionWrap_hpp */
