#ifndef MNN_VULKAN_NX_MATMUL_HPP
#define MNN_VULKAN_NX_MATMUL_HPP

#include "VulkanBasicExecution.hpp"
#include <array>

// Per-op routing, weight-preparation and data-graph setup traces. They are off in production
// builds: a 0.6B model emits ~1700 lines over an eight-turn run. Configure with
// -DMNN_VULKAN_NX_DEBUG=ON to get them back.
#ifdef MNN_VULKAN_NX_DEBUG
#define MNN_NX_LOG(...) MNN_PRINT(__VA_ARGS__)
#else
#define MNN_NX_LOG(...)
#endif

namespace MNN {

// Persistent INT8 B operand (convolution weights) for the data-graph matmul.
// The content only depends on K/N, never on M, so one tensor can be adopted by every
// shape-specific graph of the same convolution and uploaded exactly once.
class VulkanNXWeightTensor {
public:
    VulkanNXWeightTensor(const VulkanDevice& device, VulkanMemoryPool& pool);
    ~VulkanNXWeightTensor();
    bool create(int k, int n);
    // `backend` is only borrowed for the staging copy and is not retained.
    bool upload(const void* data, size_t bytes, VulkanBackend* backend);
    // True when the tensor memory is host-visible and upload() writes it directly.
    bool mapped() const { return mMapped != nullptr; }

private:
    friend class VulkanNXMatMul;
    const VulkanDevice& mDevice;
    const VulkanDevice::DataGraphInfo& mApi;
    VulkanMemoryPool& mMemoryPool;
    std::array<int64_t, 3> mDimensions{};
    VkTensorDescriptionARM mDescription{};
    VkTensorARM mTensor = VK_NULL_HANDLE;
    VkTensorViewARM mView = VK_NULL_HANDLE;
    VkBuffer mBuffer = VK_NULL_HANDLE;
    MemChunk mMemory;
    VkDeviceSize mBytes = 0;
    void* mMapped = nullptr;
    int mK = 0, mN = 0;
    bool mUploaded = false;
};

// INT8 TOSA MATMUL on the Arm NX data graph: A[B,M,K] x B[1,K,N] -> C[1,M,N] (R32_SINT).
// Not op-registered; VulkanConv1x1NX owns it and drives prepareInt8/encodeInt8 directly.
class VulkanNXMatMul : public VulkanBasicExecution {
public:
    explicit VulkanNXMatMul(VulkanBackend* backend);
    ~VulkanNXMatMul() override;
    struct SetupTimes {
        double graphBuildMs = 0, tensorBufferMs = 0, shaderModuleMs = 0;
        double pipelineLayoutDescriptorMs = 0, pipelineCompileMs = 0, sessionInitMs = 0, totalMs = 0;
        uint64_t tensorBytes = 0, sessionBytes = 0;
        size_t spirvBytes = 0;
    };
    // Optional caller-owned VkPipelineCache for data-graph pipeline creation; lifetime owned by the caller.
    void setPipelineCache(VkPipelineCache cache) { mPipelineCache = cache; }
    bool prepareInt8(int m, int k, int n, VulkanNXWeightTensor* weights = nullptr);
    // Bind tensor slot `index` to a caller-owned arena range instead of allocating its own block,
    // so several shape-specific graphs share one activation/accumulator. Set before prepareInt8.
    void setTensorAlias(uint32_t index, MemChunk memory) {
        if (index < kGraphTensorSlots) {
            mAlias[index] = memory;
        }
    }
    // Donate slot `index`'s pool allocation to a caller-owned arena: this graph keeps using the memory,
    // but its destructor no longer returns it. Invalid chunk when the slot owns no allocation.
    MemChunk takeMemory(uint32_t index);
    // Barriers only cover the compute<->data-graph hand-off; the caller must guarantee no other
    // stage touches these tensors between the surrounding compute dispatches.
    void encodeInt8(VkCommandBuffer command) const;
    VkBuffer integerBuffer(uint32_t index) const { return mTensors[index].buffer; }
    VkDeviceSize integerBufferSize(uint32_t index) const { return mTensors[index].bytes; }
    const SetupTimes& setupTimes() const { return mSetupTimes; }
    VkResult lastError() const { return mLastError; }

private:
    struct TensorResource {
        std::array<int64_t, 3> dimensions{};
        VkTensorDescriptionARM description{};
        VkTensorARM tensor = VK_NULL_HANDLE;
        VkTensorViewARM view = VK_NULL_HANDLE;
        VkBuffer buffer = VK_NULL_HANDLE;
        MemChunk memory;
        VkDeviceSize bytes = 0;
        bool external = false;
        // Memory comes from a caller-owned arena: destroy the handles but never return the allocation.
        bool aliasMemory = false;
    };
    ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                       const VulkanCommandPool::Buffer* command) override;
    static std::vector<uint32_t> makeInt8Graph(int m, int k, int n);
    bool prepare(const std::vector<uint32_t>& code, int m, int k, int n);
    bool adoptWeightTensor(TensorResource& resource, VulkanNXWeightTensor& weights, int k, int n);
    bool check(VkResult result, const char* operation) const;
    bool checkCapabilities();
#ifdef MNN_VULKAN_NX_DEBUG
    void dumpTosaCapabilities();
#endif
    bool createTensor(uint32_t slot, TensorResource& resource, int rows, int columns, VkFormat format);
    bool createSession();
    bool allocate(const VkMemoryRequirements& requirements, MemChunk& memory);
    void barrier(VkCommandBuffer command, VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
                 VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess) const;

    VulkanBackend* mBackend;
    const VulkanDevice& mDevice;
    const VulkanDevice::DataGraphInfo& mApi;
    VulkanMemoryPool& mMemoryPool;
    // A, B and the accumulator C.
    static constexpr uint32_t kGraphTensorSlots = 3;
    std::array<TensorResource, kGraphTensorSlots> mTensors;
    std::array<MemChunk, kGraphTensorSlots> mAlias{};
    std::vector<MemChunk> mSessionMemory;
    VkShaderModule mShader = VK_NULL_HANDLE;
    VkDescriptorSetLayout mSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout mLayout = VK_NULL_HANDLE;
    VkPipeline mPipeline = VK_NULL_HANDLE;
    VkPipelineCache mPipelineCache = VK_NULL_HANDLE;
    VkDescriptorPool mDescriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet mSet = VK_NULL_HANDLE;
    VkDataGraphPipelineSessionARM mSession = VK_NULL_HANDLE;
    VulkanNXWeightTensor* mSharedWeights = nullptr;
    bool mReady = false;
    SetupTimes mSetupTimes;
    mutable VkResult mLastError = VK_SUCCESS;
};

} // namespace MNN
#endif