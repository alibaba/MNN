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
#include "core/Backend.hpp"
#include <MNN/MNNSharedContext.h>
#include "MNN_generated.h"
#include "component/VulkanTensor.hpp"
#include "component/VulkanBuffer.hpp"
#include "component/VulkanCommandPool.hpp"
#include "component/VulkanDevice.hpp"
#include "component/VulkanFence.hpp"
#include "component/VulkanImage.hpp"
#include "component/VulkanInstance.hpp"
#include "component/VulkanPipeline.hpp"

namespace MNN {
class VulkanImageConverter;
class VulkanBasicExecution;

class VulkanBackend : public Backend {
public:
    VulkanBackend(const MNNVulkanContext* context, const Backend::Info& info);
    virtual ~VulkanBackend();

    virtual bool onAcquireBuffer(const Tensor* tensor, StorageType storageType) override;
    virtual bool onReleaseBuffer(const Tensor* tensor, StorageType storageType) override;
    virtual bool onClearBuffer() override;
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op) override;
    virtual std::pair<float, bool> onMeasure(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op) override;
    virtual void onExecuteBegin() const override;
    virtual void onExecuteEnd() const override;
    virtual void onResizeBegin() override;
    virtual void onResizeEnd() override;
    virtual void onCopyBuffer(const Tensor* srcTensor, const Tensor* dstTensor) const override;

    virtual bool onWaitFinish() override {
        return true;
    }
    virtual bool onAllocateBuffer() override {
        return true;
    }

    const VulkanPipeline* getPipeline(const std::string& key, const std::vector<VkDescriptorType>& types,
                                      const std::vector<uint32_t>& localSize = std::vector<uint32_t>()) const;

    const VulkanCommandPool& getPool() const {
        return (*mCmdPool);
    }
    const VulkanMemoryPool& getMemoryPool() const {
        return (*mMemoryPool);
    }
    const VulkanMemoryPool& getDynamicMemoryPool() const {
        return (*mDynamicMemoryPool);
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

    enum GPUType { ADRENO = 0, MALI = 1, OTHER = 2 };

    inline GPUType gpuType() const {
        return mGpuType;
    }

    void copyBufferToImage(const VulkanBuffer* buffer, const VulkanImage* image, VkImageLayout finalLayout = VK_IMAGE_LAYOUT_UNDEFINED) const;
    const VulkanSampler* getCommonSampler() const {
        return mSampler.get();
    }

    const VulkanTensor* findTensor(uint64_t deviceId) const;
    const VkPhysicalDeviceProperties& proty() const {
        return device().proty();
    }

    const bool success() const {
        return (nullptr != mInstance) && (mInstance->success()) && (nullptr != mDevice) && (mDevice->success());
    };

private:
    bool _supportImageSize(const Tensor* tensor);
    const VulkanDevice& device() const;
    void _finish() const;
    void _allocHostBuffer(size_t size) const;

    std::shared_ptr<VulkanPipelineFactory> mPipelineFactory;
    std::shared_ptr<VulkanCommandPool> mCmdPool;
    std::shared_ptr<VulkanCommandPool::Buffer> mCmdBuffer;
    std::shared_ptr<VulkanMemoryPool> mMemoryPool;
    std::shared_ptr<VulkanMemoryPool> mDynamicMemoryPool;
    std::shared_ptr<VulkanSampler> mSampler;

    std::map<uint64_t, std::shared_ptr<VulkanTensor>> mStaticeBuffers;
    std::map<uint64_t, std::shared_ptr<VulkanTensor>> mAllBuffers;

    mutable std::shared_ptr<VulkanBuffer> mHostBuffer;
    mutable std::vector<VkCommandBuffer> mCmdBuffers;
    mutable std::shared_ptr<VulkanFence> mFence;

    GPUType mGpuType = OTHER;

    mutable std::map<std::tuple<const Tensor*, bool, MNN_DATA_FORMAT>,
                     std::pair<std::shared_ptr<VulkanImageConverter>, std::shared_ptr<VulkanCommandPool::Buffer>>>
        mConverters;

    std::shared_ptr<VulkanInstance> mInstance;
    std::shared_ptr<VulkanDevice> mDevice;
    bool mDirect;
    float mFlops = 0.0f;
};
} // namespace MNN

#endif /* VulkanBackend_hpp */
