#ifndef VulkanConv1x1CoopA8_hpp
#define VulkanConv1x1CoopA8_hpp

#include "VulkanBasicExecution.hpp"
#include "VulkanConvolution.hpp"
#include "core/ConvolutionCommon.hpp"

namespace MNN {

// W8A8 cooperative-matrix Conv1x1 path (per-channel asym int8 OR int4 +
// Adreno + S8S8->S32 coop matrix). M==1 reuses Coop-A16's gemv_dequant decode
// (int4/int8 variant); M>1 runs a 6-stage dynamic-quant -> S8 GEMM ->
// dequant-correction pipeline. INT4 mode adds a runtime int4->int8 unpack
// stage so the GEMM body is shared with INT8.
class VulkanConv1x1CoopA8 : public VulkanBasicExecution {
public:
    VulkanConv1x1CoopA8(VulkanBackend* backend, const Convolution2DCommon* convOption,
                        const float* biasPtr, int ci, int co,
                        VulkanDevice::CoopMatInfo coopMatInfo,
                        std::shared_ptr<ConvolutionCommon::Int8Common> quantInfo);
    virtual ~VulkanConv1x1CoopA8();

    virtual ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                               const VulkanCommandPool::Buffer* cmdBuffer) override;
    virtual bool onClone(Backend* bn, const Op* op, VulkanBasicExecution** dst) override;

private:
    VulkanConv1x1CoopA8(VulkanBackend* backend, const Convolution2DCommon* convOption, int ci, int co,
                        uint32_t coopM, uint32_t coopN, uint32_t coopK, uint32_t subgroupSize,
                        std::shared_ptr<ConvolutionCommon::Int8Common> quantInfo,
                        bool initStaticResource);

    bool _init(const float* biasPtr, bool initStaticResource);

    const Convolution2DCommon* mCommon;
    int mCi;
    int mCo;
    std::shared_ptr<ConvolutionCommon::Int8Common> mQuantCommon;
    uint32_t mPadK = 0;
    uint32_t mPadN = 0;

    // Static buffers — decode + prefill share, onClone forwards them.
    //   mQuantWeightBuffer: INT8: [padN, padK/4] uint32 (4x int8 packed).
    //                        INT4: [padN, padK/8] uint32 (8x int4 packed,
    //                              unsigned 0..15 with host +8 offset).
    //   mQuantMetaBuffer:   [padN, 2] FP interleaved (scale, offset).
    //   mSumWqBuffer:       [padN] int32, sum_k Wq[n,k]; padding rows zero.
    std::shared_ptr<VulkanBuffer> mBiasBuffer;
    std::shared_ptr<VulkanBuffer> mQuantWeightBuffer;
    std::shared_ptr<VulkanBuffer> mQuantMetaBuffer;
    std::shared_ptr<VulkanBuffer> mSumWqBuffer;

    const VulkanPipeline* mDecodePipeline = nullptr;
    std::shared_ptr<VulkanLayout::DescriptorSet> mDecodeSet;

    // INT4 prefill only: runtime nibble unpack from mQuantWeightBuffer to
    // a dynamic [padN, padK] int8 tensor (tWqInt8 in onEncode), then the
    // shared dynamic_w8a8_coop_gemm consumes it as if it were INT8.
    const VulkanPipeline* mInt4UnpackPipeline = nullptr;
    std::shared_ptr<VulkanLayout::DescriptorSet> mInt4UnpackSet;

    const VulkanPipeline* mQuantMinMaxPipeline = nullptr;
    const VulkanPipeline* mQuantReduceMinMaxPipeline = nullptr;
    const VulkanPipeline* mQuantPackPipeline = nullptr;
    const VulkanPipeline* mQuantReduceSumPipeline = nullptr;
    const VulkanPipeline* mGemmS8Pipeline = nullptr;
    const VulkanPipeline* mDequantPipeline = nullptr;

    std::shared_ptr<VulkanLayout::DescriptorSet> mQuantMinMaxSet;
    std::shared_ptr<VulkanLayout::DescriptorSet> mQuantReduceMinMaxSet;
    std::shared_ptr<VulkanLayout::DescriptorSet> mQuantPackSet;
    std::shared_ptr<VulkanLayout::DescriptorSet> mQuantReduceSumSet;
    std::shared_ptr<VulkanLayout::DescriptorSet> mGemmSet;
    std::shared_ptr<VulkanLayout::DescriptorSet> mDequantSet;

    uint32_t mCoopM = 64;
    uint32_t mCoopN = 64;
    uint32_t mCoopK = 32;
    uint32_t mSubgroupSize = 64;
    bool mIsInt4 = false;
};

} // namespace MNN

#endif
