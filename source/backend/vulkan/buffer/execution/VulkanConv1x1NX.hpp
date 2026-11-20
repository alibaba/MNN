#ifndef MNN_VULKAN_CONV1X1_NX_HPP
#define MNN_VULKAN_CONV1X1_NX_HPP

#include "VulkanBasicExecution.hpp"
#include "core/ConvolutionCommon.hpp"
#include <memory>
#include <vector>

namespace MNN {

class VulkanConv1x1General;
class VulkanNXWeightTensor;

class VulkanConv1x1NX : public VulkanBasicExecution {
public:
    static bool supported(VulkanBackend* backend, const Convolution2DCommon* common,
                          const ConvolutionCommon::Int8Common* quantCommon, int ci, int co);
    VulkanConv1x1NX(VulkanBackend* backend, const Convolution2DCommon* common, const float* bias, int ci, int co,
                    std::shared_ptr<ConvolutionCommon::Int8Common> quantCommon);
    ~VulkanConv1x1NX() override;

    ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                       const VulkanCommandPool::Buffer* command) override;
    bool onClone(Backend* backend, const Op* op, VulkanBasicExecution** dst) override;

private:
    struct Source;
    struct Weights;
    struct State;
    VulkanConv1x1NX(VulkanBackend* backend, std::shared_ptr<const Source> source,
                    std::shared_ptr<ConvolutionCommon::Int8Common> quantCommon);
    // `bucket` is the fixed chunk shape; padding rows are computed and discarded.
    bool prepare(uint32_t rows, uint32_t bucket, uint32_t padK, uint32_t padN, double& weightMs,
                 double& computeSetupMs);
    bool prepareWeights(uint32_t padK, uint32_t padN);
    // Writes padK x padN transposed weights; only the padding bytes are zeroed.
    void transposeWeights(int8_t* output, uint32_t padK, uint32_t padN) const;
    // Resident, per-op weight tensor shared by every shape and clone; null means per-shape upload.
    std::shared_ptr<VulkanNXWeightTensor> adoptSharedWeights(uint32_t padK, uint32_t padN);
    State* findPreparedState(uint32_t bucket, uint32_t padK, uint32_t padN);
    ErrorCode fallback(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                       const VulkanCommandPool::Buffer* command, const char* reason);
    // Encode through the lazily created General execution, keeping any prepared NX state alive.
    ErrorCode runGeneral(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                         const VulkanCommandPool::Buffer* command, const char* reason);

    VulkanBackend* mBackend;
    std::shared_ptr<const Source> mSource;
    std::shared_ptr<ConvolutionCommon::Int8Common> mQuantCommon;
    std::shared_ptr<const Weights> mWeights;
    // Only the current shape stays prepared. Multi-bucket caching was rejected back when every bucket
    // needed its own operand memory (0.5-1 GB over a 0.6B model's 196 convolutions); with operands
    // shared per backend the same ladder costs ~20 MB but measured only ~1.07x end to end.
    std::unique_ptr<State> mState;
    std::unique_ptr<VulkanConv1x1General> mFallback;
    VkResult mLastError = VK_SUCCESS;
    bool mResidentWeights = false;
};

} // namespace MNN
#endif
