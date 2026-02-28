//
//  VulkanBackend.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef VulkanBackend_hpp
#define VulkanBackend_hpp

#include <map>
#include <MNN/ErrorCode.hpp>
#include "MNN_generated.h"
#include "VulkanRuntime.hpp"

#ifdef ENABLE_VULKAN_TIME_PROFILE
#include "VulkanTimeProfiler.hpp"
#endif

#ifdef MNN_USE_ARMV82
// FP32 <--> FP16 Function
#include "backend/arm82/Arm82OptFunc.hpp"
#define FLOAT_TO_HALF MNNQuantizeFP16
#define HALF_TO_FLOAT MNNDequantizeFP16
#else
#include "half.hpp"
#define FLOAT_TO_HALF _VKFloatToHalf
#define HALF_TO_FLOAT _VKHalfToFloat
#endif // MNN_USE_ARMV82

#ifndef MNN_USE_ARMV82
namespace MNN {
    void _VKFloatToHalf(const float* src, int16_t* dst, size_t size);
    void _VKHalfToFloat(const int16_t* src, float* dst, size_t size);
}
#endif

namespace MNN {
class VulkanBasicExecution;
typedef std::tuple<VkBuffer, VkDeviceSize, VkDeviceSize> VULKAN_TENSOR;
class VulkanBackend : public Backend {
public:
    VulkanBackend(const VulkanRuntime* runtime, const Backend::Info& info);
    virtual ~VulkanBackend();

    virtual Backend::MemObj* onAcquire(const Tensor* tensor, StorageType storageType) override;
    virtual bool onClearBuffer() override;
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op) override;
    virtual void onExecuteBegin() const override;
    virtual void onExecuteEnd() const override;
    void finish();
    virtual bool onSelectDynamicAllocator(int index, int maxIndex) override;
    virtual void onResizeBegin() override;
    virtual ErrorCode onResizeEnd() override;
    virtual void onCopyBuffer(const Tensor* srcTensor, const Tensor* dstTensor) const override;
    virtual const Runtime* getRuntime() override {
        return mRuntime;
    }
    virtual int onSync(Tensor::MapType mtype, bool toCpu, const Tensor* dstTensor) override;

    const VulkanPipelineFactory* getPipelineFactory() const;
    const VulkanPipeline* getPipeline(const std::string& key, const std::vector<VkDescriptorType>& types,
                                      const std::vector<uint32_t>& localSize = std::vector<uint32_t>(),
                                      const std::vector<uint32_t>& specConstants = std::vector<uint32_t>()) const;
    SharedPtr<VulkanPipeline> getPrivatePipeline(const std::string& key, const std::vector<VkDescriptorType>& types, const std::vector<uint32_t>& specConstants = std::vector<uint32_t>());

    const VulkanCommandPool& getPool() const {
        return (* mRuntime->mCmdPool);
    }
    const VulkanMemoryPool& getMemoryPool() const {
        return (* mRuntime->mMemoryPool);
    }
    const VulkanDevice& getDevice() const {
        return (* mRuntime->mDevice);
    }

    BufferAllocator* getDynamicMemoryPool() const {
        return mCurrentDynamicBufferPool;
    }
    virtual bool onGetTensorInfo(const Tensor* tensor, void* dstInfo) override;
    
    std::vector<uint32_t> autoTunePipeline(const VulkanPipeline* pipeline, SharedPtr<VulkanLayout::DescriptorSet> des, std::vector<int> gws);
    
    float getPipelineTime(const VulkanPipeline* pipeline, SharedPtr<VulkanLayout::DescriptorSet> des, std::vector<int> groupSize);

    bool isSupportAutotune(){
        return mUseAutoTune;
    }

    class Creator {
    public:
        virtual VulkanBasicExecution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op, Backend* backend) const = 0;
    };
    static bool addCreator(OpType t, Creator* c);

    void pushCommand(VkCommandBuffer buffer) const;
    std::shared_ptr<VulkanCommandPool::Buffer> getSingleCommand() {
        return mCmdBuffer;
    }

    inline VulkanRuntime::GPUType gpuType() const {
        return mRuntime->mGpuType;
    }

    const VulkanSampler* getCommonSampler(bool clamp = false) const {
        if (clamp) {
            return mRuntime->mClampSampler.get();
        }
        return mRuntime->mSampler.get();
    }

    const VkPhysicalDeviceProperties& proty() const {
        return device().proty();
    }
    // Buffer, offset
    std::pair<const VulkanBuffer*, size_t> getTensorBuffer(const Tensor* tensor) const;
    size_t getTensorSize(const Tensor* tensor) const;
    VULKAN_TENSOR getBuffer(const Tensor* tensor) const;
    std::shared_ptr<VulkanBuffer> allocUniform(const void* src = nullptr, int size = 0);
    void recycleUniform(std::shared_ptr<VulkanBuffer> buffer);
    void copyGPUToGPUBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size, VkDeviceSize srcOffset, VkDeviceSize dstOffset) const;
    void copyToGPUBuffer(const void* src, VkBuffer buffer, VkDeviceSize size, VkDeviceSize offset) const;
    std::shared_ptr<VulkanBuffer> createHostBuffer(size_t size) const;

    const VulkanDevice& device() const;
#ifdef ENABLE_VULKAN_TIME_PROFILE
    VulkanTimeProfiler* timeProfiler() const {
        return mTimeProfiler.get();
    }
#endif

    bool useFP16() const {
        return mUseFP16;
    }

private:
    void _finish() const;
    void _requireHostBuffer(size_t size) const;
    mutable std::shared_ptr<VulkanBuffer> mHostBuffer;

    std::shared_ptr<VulkanCommandPool::Buffer> mCmdBuffer;
    std::shared_ptr<VulkanCommandPool::Buffer> mCmdBufferForCopy;
    BufferAllocator* mCurrentDynamicBufferPool = nullptr;
    std::vector<std::shared_ptr<BufferAllocator>> mDynamicBufferPool;

    mutable std::vector<VkCommandBuffer> mCmdBuffers;
    mutable std::shared_ptr<VulkanFence> mFence;

    bool mDirect;
    const VulkanRuntime* mRuntime;
    bool mUseAutoTune = true;

    bool mUseFP16{false};

#ifdef ENABLE_VULKAN_TIME_PROFILE
    std::shared_ptr<VulkanTimeProfiler> mTimeProfiler;
#endif
};


} // namespace MNN

#endif /* VulkanBackend_hpp */
