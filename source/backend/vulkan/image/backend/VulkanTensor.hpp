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
#include <array>
namespace MNN {
class VulkanTensor : public NonCopyable {
public:
    ~VulkanTensor() {
    }
    VulkanTensor(const Tensor* shape, VkFormat format, const VulkanMemoryPool& pool, const VkPhysicalDeviceLimits& limits, bool separate = false);
    void release();

    size_t imageSize() const {
        return mImage.size();
    }
    const std::array<int, 2>& blocks() const {
        return mBlocks;
    }
    const VulkanImage* image(int index = 0) const {
        return mImage[index].get();
    }
    // N, C, H, W
    static std::array<int, 4> tensorShapeFormat(const Tensor *input);

    static int getAlignSize(const Tensor* tensor);
private:
    std::vector<std::shared_ptr<VulkanImage>> mImage;
    std::array<int, 2> mBlocks;
    std::array<int, 4> mSize;
};
}
#endif
