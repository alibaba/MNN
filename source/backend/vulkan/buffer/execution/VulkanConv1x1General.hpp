#ifndef VulkanConv1x1General_hpp
#define VulkanConv1x1General_hpp

#include <cstdint>
#include "VulkanBasicExecution.hpp"
#include "VulkanConvolution.hpp"
#include "core/ConvolutionCommon.hpp"

namespace MNN {

class VulkanConv1x1General : public VulkanBasicExecution {
public:
    VulkanConv1x1General(VulkanBackend* backend, const Convolution2DCommon* convOption, const float* biasPtr, int ci, int co,
                         std::shared_ptr<ConvolutionCommon::Int8Common> quantInfo);
    virtual ~VulkanConv1x1General();
    virtual ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                               const VulkanCommandPool::Buffer* cmdBuffer) override;
    virtual bool onClone(Backend* bn, const Op* op, VulkanBasicExecution** dst) override;

private:
    VulkanConv1x1General(VulkanBackend* backend, const Convolution2DCommon* convOption, int ci, int co,
                         std::shared_ptr<ConvolutionCommon::Int8Common> quantInfo, bool initStaticResource);
    bool _init(const float* biasPtr, bool initStaticResource);

private:
    const Convolution2DCommon* mCommon = nullptr;
    int mCi = 0;
    int mCo = 0;
    bool mIsInt4 = false;
    uint32_t mPadK = 0;
    uint32_t mPadN = 0;
    uint32_t mBlockSize = 1;
    uint32_t mBlockStride = 1;
    uint32_t mDecodeWeightStrideWords = 0;
    uint32_t mDecodeSubgroupSize = 1;

    std::shared_ptr<ConvolutionCommon::Int8Common> mQuantCommon;

    std::shared_ptr<VulkanBuffer> mQuantWeightBuffer;
    std::shared_ptr<VulkanBuffer> mQuantMetaBuffer;
    std::shared_ptr<VulkanBuffer> mBiasBuffer;

    const VulkanPipeline* mDecodePipeline = nullptr;
    const VulkanPipeline* mPackAPipeline = nullptr;
    const VulkanPipeline* mWeightToPackPipeline = nullptr;
    const VulkanPipeline* mGemmPipeline = nullptr;
    std::shared_ptr<VulkanLayout::DescriptorSet> mDecodeSet;
    std::shared_ptr<VulkanLayout::DescriptorSet> mPackASet;
    std::shared_ptr<VulkanLayout::DescriptorSet> mWeightToPackSet;
    std::shared_ptr<VulkanLayout::DescriptorSet> mGemmSet;

    std::shared_ptr<Tensor> mTempInputPacked;
    std::shared_ptr<Tensor> mTempWeightPacked;
};

} // namespace MNN

#endif // VulkanConv1x1General_hpp
