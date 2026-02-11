#ifndef VulkanConv1x1Coop_hpp
#define VulkanConv1x1Coop_hpp

#include "VulkanBasicExecution.hpp"
#include "VulkanConvolution.hpp"
#include "core/ConvolutionCommon.hpp"

namespace MNN {

class VulkanConv1x1Coop : public VulkanBasicExecution {
public:
    VulkanConv1x1Coop(VulkanBackend* backend, const Convolution2DCommon* convOption, const float* weightPtr, const float* biasPtr, int ci, int co, VulkanDevice::CoopMatInfo coopMatInfo, std::shared_ptr<ConvolutionCommon::Int8Common> quantInfo = nullptr, std::shared_ptr<ConvolutionCommon::Int8Common> weightHolder = nullptr);
    virtual ~VulkanConv1x1Coop();
    virtual ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                               const VulkanCommandPool::Buffer* cmdBuffer) override;
    virtual bool onClone(Backend* bn, const Op* op, VulkanBasicExecution** dst) override;

private:
    VulkanConv1x1Coop(VulkanBackend* backend, const Convolution2DCommon* convOption, int ci, int co,
                      uint32_t coopM, uint32_t coopN, uint32_t coopK, uint32_t subgroupSize,
                      std::shared_ptr<ConvolutionCommon::Int8Common> quantInfo,
                      std::shared_ptr<ConvolutionCommon::Int8Common> weightHolder,
                      bool initStaticResource);

    bool _init(const float* weightPtr, const float* biasPtr, bool initStaticResource);
    
    const Convolution2DCommon* mCommon;
    int mCi;
    int mCo;
    bool mIsQuant = false;
    std::shared_ptr<ConvolutionCommon::Int8Common> mQuantCommon;
    std::shared_ptr<ConvolutionCommon::Int8Common> mWeightFloatHolder;
    uint32_t mBlockSize = 0; // quant block size along K
    uint32_t mPadK = 0;
    uint32_t mPadN = 0;
    bool mQuantConverted = false;
    
    // Weight and Bias Buffer
    std::shared_ptr<VulkanBuffer> mWeightBuffer; // float/half weight (float path) or dequantized coop weight (quant path)
    std::shared_ptr<VulkanBuffer> mBiasBuffer;
    // Quant resources (static)
    std::shared_ptr<VulkanBuffer> mQuantWeightBuffer; // int8/int4 packed
    std::shared_ptr<VulkanBuffer> mQuantScaleBuffer;  // scale per block
    std::shared_ptr<VulkanBuffer> mQuantOffsetBuffer; // offset per block
    
    // Pipelines
    const VulkanPipeline* mPackPipeline;
    const VulkanPipeline* mMatMulPipeline;
    const VulkanPipeline* mUnpackPipeline;
    const VulkanPipeline* mDequantPipeline = nullptr;
    
    // Descriptor Sets
    std::shared_ptr<VulkanLayout::DescriptorSet> mPackSet;
    std::shared_ptr<VulkanLayout::DescriptorSet> mMatMulSet;
    std::shared_ptr<VulkanLayout::DescriptorSet> mUnpackSet;
    std::shared_ptr<VulkanLayout::DescriptorSet> mDequantSet;
    
    // Constant Buffers
    std::shared_ptr<VulkanBuffer> mPackConst;
    std::shared_ptr<VulkanBuffer> mMatMulConst;
    std::shared_ptr<VulkanBuffer> mUnpackConst;
    std::shared_ptr<VulkanBuffer> mDequantConst;

    // Temp Tensors
    std::shared_ptr<Tensor> mTempInput;
    std::shared_ptr<Tensor> mTempOutput;
    std::shared_ptr<Tensor> mTempWeight;
    
    // COOP Shape
    uint32_t COOP_M = 64;
    uint32_t COOP_N = 64;
    uint32_t COOP_K = 16;

    // subgroupSize
    uint32_t mSubgroupSize = 64;
};

} // namespace MNN

#endif
