//
//  VulkanRuntime.hpp
//  MNN
//
//  Created by MNN on b'2020/06/06'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef VulkanRuntime_hpp
#define VulkanRuntime_hpp
#include <queue>
#include <array>
#include <functional>
#include <unordered_map>
#include "VKCache_generated.h"
#include "VulkanBuffer.hpp"
#include "VulkanCommandPool.hpp"
#include "VulkanDevice.hpp"
#include "VulkanFence.hpp"
#include "VulkanImage.hpp"
#include "VulkanInstance.hpp"
#include "VulkanPipeline.hpp"
#include "VulkanQueryPool.hpp"
#include "core/Backend.hpp"
#define MNN_VULKAN
#include <MNN/MNNSharedContext.h>

namespace MNN {

struct VKTuneKey {
    std::string shaderName;
    std::array<uint32_t, 3> gws;

    bool operator==(const VKTuneKey & other) const {
        return (shaderName == other.shaderName) && (gws == other.gws);
    }
};

struct VKTuneValue {
    std::array<uint32_t, 3> optimalLws;
    float optimalCost;
};

struct VkTuneHash {
    size_t operator()(const VKTuneKey & key) const {
        size_t hs = std::hash<std::string>()(key.shaderName);
        size_t h0 = std::hash<uint32_t>()(key.gws[0]);
        size_t h1 = std::hash<uint32_t>()(key.gws[1]);
        size_t h2 = std::hash<uint32_t>()(key.gws[2]);
        return hs & h0 & h1 & h2;
    }
};

class VulkanRuntime : public Runtime {
public:
    virtual ~ VulkanRuntime();

    virtual Backend* onCreate(const BackendConfig* config, Backend* origin) const override;
    enum GPUType { ADRENO = 0, MALI = 1, OTHER = 2 };
    virtual void onGabageCollect(int level) override;
    virtual float onGetMemoryInMB() override;
    int onGetRuntimeStatus(RuntimeStatus statusEnum) const override;
    std::shared_ptr<VulkanBuffer> allocUniform(const void* src = nullptr, int size = 0);
    void recycleUniform(std::shared_ptr<VulkanBuffer> buffer);
    static VulkanRuntime* create(const Backend::Info& info);
    virtual std::pair<const void*, size_t> onGetCache() override;
    virtual bool onSetCache(const void* buffer, size_t size) override;

private:
    VulkanRuntime(const Backend::Info& info, std::shared_ptr<VulkanDevice> device, std::shared_ptr<VulkanInstance> instance);
    Backend::Info mInfo;
    std::shared_ptr<BufferAllocator> mBufferPool;
    std::shared_ptr<VulkanPipelineFactory> mPipelineFactory;
    std::shared_ptr<VulkanCommandPool> mCmdPool;
    std::shared_ptr<VulkanMemoryPool> mMemoryPool;
    std::shared_ptr<VulkanSampler> mSampler;
    std::shared_ptr<VulkanSampler> mClampSampler;
    std::shared_ptr<VulkanInstance> mInstance;
    std::shared_ptr<VulkanQueryPool> mQueryPool;
    std::shared_ptr<VulkanDevice> mDevice;
    std::queue<std::shared_ptr<VulkanBuffer>> mUniformCache;
    // Limit Uniform cache size = mUniformSize * mCacheUniformLimitSize B
    int mUniformSize = 512;
    int mCacheUniformLimitSize = 1024;
    float mFlops = 0.0f;
    friend class VulkanBackend;
    GPUType mGpuType = OTHER;
    int mGpuMode;
// member variables related to auto tuning
private:
    mutable std::unordered_map<VKTuneKey, VKTuneValue, VkTuneHash> mTuneMap;
    std::vector<uint8_t> mTuneBuffer;
};
}
#endif
