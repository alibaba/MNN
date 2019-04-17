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
    template <typename T>
    static void MNNReorderWeight(float* reorderedWeight, const T* srcWeight, int ci, int co, int kh, int kw,
                                 int uint = 4);

    static std::shared_ptr<VulkanBuffer> createBufferForSlideWindow(const VulkanBackend* backend,
                                                                    const Convolution2DCommon* convOption,
                                                                    const float* weightPtr, int ci, int co);

    static std::shared_ptr<Execution> create(VulkanBackend* backend, const Convolution2DCommon* convOption,
                                             const Tensor* input, const Tensor* output, const float* weightPtr,
                                             const float* biasPtr, int ci, int co);
};
} // namespace MNN
#endif /* VulkanConvolutionImpl_hpp */
