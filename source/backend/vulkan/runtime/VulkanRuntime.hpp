//
//  VulkanRuntime.hpp
//  MNN
//
//  Created by MNN on b'2020/06/06'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef VulkanRuntime_hpp
#define VulkanRuntime_hpp

#include "VulkanBuffer.hpp"
#include "VulkanCommandPool.hpp"
#include "VulkanDevice.hpp"
#include "VulkanFence.hpp"
#include "VulkanImage.hpp"
#include "VulkanInstance.hpp"
#include "VulkanPipeline.hpp"
#include "core/Backend.hpp"
#define MNN_VULKAN
#include <MNN/MNNSharedContext.h>

namespace MNN {
class VulkanRuntime : public Runtime {
public:
    VulkanRuntime(const Backend::Info& info);
    virtual ~ VulkanRuntime();

    virtual Backend* onCreate(const BackendConfig* config) const override;
    enum GPUType { ADRENO = 0, MALI = 1, OTHER = 2 };
    virtual void onGabageCollect(int level) override;
    virtual float onGetMemoryInMB() override;
    int onGetRuntimeStatus(RuntimeStatus statusEnum) const override;
private:
    Backend::Info mInfo;
    std::shared_ptr<VulkanPipelineFactory> mPipelineFactory;
    std::shared_ptr<VulkanCommandPool> mCmdPool;
    std::shared_ptr<VulkanMemoryPool> mMemoryPool;
    std::shared_ptr<VulkanSampler> mSampler;
    std::shared_ptr<VulkanSampler> mClampSampler;
    std::shared_ptr<VulkanInstance> mInstance;
    std::shared_ptr<VulkanDevice> mDevice;
    float mFlops = 0.0f;
    friend class VulkanBackend;
    GPUType mGpuType = OTHER;
};
}
#endif
