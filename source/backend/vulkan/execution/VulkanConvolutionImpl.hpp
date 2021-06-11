//
//  VulkanConvolutionImpl.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef VulkanConvolutionImpl_hpp
#define VulkanConvolutionImpl_hpp
#include "VulkanBasicExecution.hpp"
namespace MNN {
class VulkanConvolutionImpl {
public:
    virtual ~VulkanConvolutionImpl() {
    }

    static VulkanBasicExecution* create(VulkanBackend* backend, const Convolution2DCommon* convOption,
                                             const std::vector<Tensor*>& input, const Tensor* output, const float* weightPtr,
                                             const float* biasPtr, int ci, int co);
    static int gImage2ColLocal;
};
} // namespace MNN
#endif /* VulkanConvolutionImpl_hpp */
