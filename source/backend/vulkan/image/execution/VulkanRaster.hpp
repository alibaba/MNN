#ifndef VulkanRaster_hpp
#define VulkanRaster_hpp
#include "VulkanBasicExecution.hpp"
#include "VulkanImageConverter.hpp"
namespace MNN {
class VulkanRaster : public VulkanBasicExecution {
public:
    VulkanRaster(Backend *bn) : VulkanBasicExecution(bn) {
        //Do nothing
    }
    virtual ~VulkanRaster() = default;

    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                               const VulkanCommandPool::Buffer *cmdBuffer) override;
    void onEncodeFast(const Tensor* input, const Tensor* output, const VulkanCommandPool::Buffer *cmdBuffer, bool zero);


private:
    struct ConvertInfo {
        const VulkanPipeline* pipeline = nullptr;
        std::shared_ptr<VulkanImageConverter> convert;
        std::shared_ptr<VulkanBuffer> buffer;
    };
    std::map<Tensor*, ConvertInfo> mInputBuffers;
    ConvertInfo mOutputBuffer;
    struct BlitInfo {
        const VulkanPipeline* pipeline = nullptr;
        std::shared_ptr<VulkanLayout::DescriptorSet> describe;
        std::shared_ptr<VulkanBuffer> uniform;
        VkBuffer srcBuffer;
        int srcBufferSize;
        VkBuffer dstBuffer;
        int dstBufferSize;
        ivec3 workGroup;
    };
    struct BlitImageInfo {
        std::shared_ptr<VulkanLayout::DescriptorSet> describe;
        std::shared_ptr<VulkanBuffer> uniform;
    };
    std::vector<BlitInfo> mBlits;
    std::vector<BlitImageInfo> mBlitImages;

    struct FillInfo {
        VkBuffer dstBuffer;
        int dstBufferSize;
    };
    FillInfo mZero;
    std::vector<std::shared_ptr<VulkanLayout::DescriptorSet>> mExtraDescribes;
    std::vector<std::shared_ptr<VulkanBuffer>> mExtraUniform;
};
};

#endif
