#ifndef VulkanRaster_hpp
#define VulkanRaster_hpp
#include "VulkanBasicExecution.hpp"
namespace MNN {
class VulkanRaster : public VulkanBasicExecution {
public:
    VulkanRaster(Backend *bn) : VulkanBasicExecution(bn) {
        //Do nothing
    }
    virtual ~VulkanRaster();

    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                               const VulkanCommandPool::Buffer *cmdBuffer) override;
    void onEncodeFast(const Tensor* input, const Tensor* output, const VulkanCommandPool::Buffer *cmdBuffer, bool zero);

    struct Componet {
        Tensor* real;
        std::shared_ptr<VulkanBasicExecution> exe;
    };
    static Componet create(Tensor* real, Backend* bn);

private:
    void _recycle();
    std::vector<std::shared_ptr<VulkanLayout::DescriptorSet>> mExtraDescribes;
    std::vector<std::shared_ptr<VulkanBuffer>> mExtraUniform;
    std::map<Tensor*, std::pair<MemChunk, int>> mInputBuffers;
    std::pair<MemChunk, int> mOutputBuffer;
};
};

#endif
