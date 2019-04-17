//
//  VulkanBackend.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef VulkanBackend_hpp
#define VulkanBackend_hpp

#include <list>
#include <map>
#include "Backend.hpp"
#include "MNNSharedContext.h"
#include "MNN_generated.h"
#include "VulkanBuffer.hpp"
#include "VulkanCommandPool.hpp"
#include "VulkanDevice.hpp"
#include "VulkanFence.hpp"
#include "VulkanImage.hpp"
#include "VulkanInstance.hpp"
#include "VulkanPipeline.hpp"
#include "vulkan_wrapper.h"

namespace MNN {
class VulkanImageConverter;
class VulkanTensor : public NonCopyable {
public:
    ~VulkanTensor() {
    }
    VulkanTensor(const Tensor* shape, const VulkanMemoryPool& pool, bool forceBuffer = false, bool seperate = false);
    void release();
    uint64_t deviceId();

    const VulkanBuffer* buffer() const {
        return mBuffer.get();
    }
    const VulkanImage* image() const {
        return mImage.get();
    }
    uint64_t deviceId() const;

private:
    std::shared_ptr<VulkanBuffer> mBuffer;
    std::shared_ptr<VulkanImage> mImage;
};
class VulkanBackend : public Backend {
public:
    VulkanBackend(const MNNVulkanContext* context);
    virtual ~VulkanBackend();

    virtual bool onAcquireBuffer(const Tensor* tensor, StorageType storageType) override;
    virtual bool onReleaseBuffer(const Tensor* tensor, StorageType storageType) override;
    virtual bool onClearBuffer() override;
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op) override;
    virtual void onExecuteBegin() const override;
    virtual void onExecuteEnd() const override;
    virtual void onCopyBuffer(const Tensor* srcTensor, const Tensor* dstTensor) const override;

    virtual bool onWaitFinish() override {
        return true;
    }
    virtual bool onLoadLibrary(const GpuLibrary* library) override;
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
        virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const MNN::Op* op, Backend* backend) const = 0;
    };
    static bool addCreator(OpType t, Creator* c);

    void pushCommand(VkCommandBuffer buffer) const;

    enum GPUType { ADRENO = 0, MALI = 1, OTHER = 2 };

    inline GPUType gpuType() const {
        return mGpuType;
    }

    void copyBufferToImage(const VulkanBuffer* buffer, const VulkanImage* image) const;
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
};
} // namespace MNN

#endif /* VulkanBackend_hpp */
