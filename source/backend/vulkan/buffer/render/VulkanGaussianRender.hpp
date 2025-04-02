#ifndef VulkanGaussianRender_hpp
#define VulkanGaussianRender_hpp
#include "VulkanBasicExecution.hpp"
namespace MNN {
class VulkanRadixSort {
public:
    struct Content;
    VulkanRadixSort(Backend* bn, int needBit);
    ~VulkanRadixSort();
    ErrorCode onExcute(std::pair<VulkanBuffer*, VkDeviceSize> srcIndex, std::pair<VulkanBuffer*, VkDeviceSize> dstIndex, const VulkanCommandPool::Buffer *cmdBuffer, int numberPoint, std::shared_ptr<VulkanBuffer> sortNumber);
    void autoTune(std::pair<VulkanBuffer*, VkDeviceSize> srcIndex, std::pair<VulkanBuffer*, VkDeviceSize> dstIndex, int numberPoint, std::shared_ptr<VulkanBuffer> sortNumber);
private:
    Content* mContent;
    Backend* mBackend;
    int mNeedBits;
    int mPerSortBit = 4;
    int mLocalSize = 256;
    int mGroupSize = 16;
};

class VulkanRasterSort : public VulkanBasicExecution {
public:
    struct Content;
    VulkanRasterSort(Backend* bn);
    virtual ~ VulkanRasterSort();
    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                               const VulkanCommandPool::Buffer *cmdBuffer) override;
    void autoTune(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
private:
    Content* mContent;
    std::shared_ptr<VulkanRadixSort> mRadixSort;
    int mLocalSize = 1024;
    int mGroupSize = 1024;
};

};

#endif
