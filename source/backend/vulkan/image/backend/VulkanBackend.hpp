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
#include "VulkanTensor.hpp"
#include "core/TensorUtils.hpp"

namespace MNN {
class VulkanImageConverter;
class VulkanBasicExecution;

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
    virtual void onResizeBegin() override;
    virtual ErrorCode onResizeEnd() override;
    virtual void onCopyBuffer(const Tensor* srcTensor, const Tensor* dstTensor) const override;

    const VulkanPipeline* getPipeline(const std::string& key, const std::vector<VkDescriptorType>& types,
                                      const std::vector<uint32_t>& localSize = std::vector<uint32_t>()) const;

    const VulkanCommandPool& getPool() const {
        return (* mRuntime->mCmdPool);
    }
    const VulkanMemoryPool& getMemoryPool() const {
        return (* mRuntime->mMemoryPool);
    }
    const VulkanMemoryPool& getDynamicMemoryPool() const {
        return (* mDynamicMemoryPool);
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

    void copyBufferToImage(const VulkanBuffer* buffer, const VulkanImage* image, VkImageLayout finalLayout = VK_IMAGE_LAYOUT_UNDEFINED) const;
    const VulkanSampler* getCommonSampler(bool clamp = false) const {
        if (clamp) {
            return mRuntime->mClampSampler.get();
        }
        return mRuntime->mSampler.get();
    }

    const VkPhysicalDeviceProperties& proty() const {
        return device().proty();
    }

    const VulkanCommandPool::Buffer* getInitCommandBuffer() const {
        return mInitBuffer.get();
    }
private:
    bool _supportImageSize(const Tensor* tensor);
    const VulkanDevice& device() const;
    void _finish() const;
    void _allocHostBuffer(size_t size) const;

    std::shared_ptr<VulkanCommandPool::Buffer> mCmdBuffer;
    std::shared_ptr<VulkanCommandPool::Buffer> mInitBuffer;

    std::map<uint64_t, std::shared_ptr<VulkanTensor>> mAllBuffers;

    mutable std::shared_ptr<VulkanBuffer> mHostBuffer;
    mutable std::vector<VkCommandBuffer> mCmdBuffers;
    mutable std::shared_ptr<VulkanFence> mFence;


    mutable std::map<std::tuple<const Tensor::InsideDescribe::NativeInsideDescribe*, bool, MNN_DATA_FORMAT>,
                     std::pair<std::shared_ptr<VulkanImageConverter>, std::shared_ptr<VulkanCommandPool::Buffer>>>
        mConverters;

    bool mDirect;
    const VulkanRuntime* mRuntime;
    std::shared_ptr<VulkanMemoryPool> mDynamicMemoryPool;
};


} // namespace MNN

#endif /* VulkanBackend_hpp */
