//
//  VulkanTensor.hpp
//  MNN
//
//  Created by MNN on 2020/03/03.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef VulkanTensor_hpp
#define VulkanTensor_hpp
#include <MNN/Tensor.hpp>
#include "core/NonCopyable.hpp"
#include "VulkanImage.hpp"
#include "VulkanBuffer.hpp"
namespace MNN {
class VulkanTensor : public NonCopyable {
public:
    ~VulkanTensor() {
    }
    VulkanTensor(const Tensor* shape, const VulkanMemoryPool& pool, bool forceBuffer = false, bool seperate = false);
    void release();
    uint64_t deviceId();

    const VulkanBuffer* buffer() const {
        return mBuffer.get();
    }
    const VulkanImage* image() const {
        return mImage.get();
    }
    uint64_t deviceId() const;

    static int getAlignSize(const Tensor* tensor);
private:
    std::shared_ptr<VulkanBuffer> mBuffer;
    std::shared_ptr<VulkanImage> mImage;
};
}
#endif