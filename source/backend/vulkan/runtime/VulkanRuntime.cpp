//
//  VulkanRuntime.cpp
//  MNN
//
//  Created by MNN on b'2020/06/06'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanRuntime.hpp"
#include "VulkanBackend.hpp"
namespace MNN {
class VulkanBufferAllocator : public BufferAllocator::Allocator {
public:
    VulkanBufferAllocator(const VulkanDevice& device, const VulkanMemoryPool& pool) : mDevice(device), mPool(pool) {
        // Do nothing
    }
    virtual ~ VulkanBufferAllocator() {
        // Do nothing
    }
    virtual MemChunk onAlloc(size_t size, size_t align) override {
        VulkanBuffer* newBuffer = new VulkanBuffer(mPool, false, size, nullptr, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_SHARING_MODE_EXCLUSIVE, 0);
        return MemChunk(newBuffer, 0);
    }
    virtual void onRelease(MemChunk ptr) override {
        auto p = (VulkanBuffer*)ptr.first;
        delete p;
    }
private:
    const VulkanDevice& mDevice;
    const VulkanMemoryPool& mPool;
};


float VulkanRuntime::onGetMemoryInMB() {
    return mMemoryPool->computeSize();
}

VulkanRuntime::VulkanRuntime(const Backend::Info& info) {
    mInfo = info;
    MNNVulkanContext* context = nullptr;
    if (nullptr != info.user && nullptr != info.user->sharedContext) {
       MNN_PRINT("Use user's vulkan context\n");
       context = static_cast<MNNVulkanContext*>(info.user->sharedContext);
    }
    if (NULL != context) {
        mInstance = std::make_shared<VulkanInstance>(context->pInstance);
        mDevice   = std::make_shared<VulkanDevice>(mInstance, context->pPhysicalDevice, context->pDevice,
                                                 context->iQueueFamilyIndex, context->pQueue);
    } else {
        mInstance = std::make_shared<VulkanInstance>();
        mDevice   = std::make_shared<VulkanDevice>(mInstance);
    }
    auto& dev              = *mDevice;
    mCmdPool               = std::make_shared<VulkanCommandPool>(dev);
    //GFlops, Test by mobilenet v1's ms
    static std::map<std::string, float> gFlopsMap {
        {"Mali-T860", 6.83f},
        {"Mali-T880", 6.83f},
        {"Mali-G51", 6.83f},
        {"Mali-G52", 6.83f},
        {"Mali-G71", 31.61f},
        {"Mali-G72", 31.61f},
        {"Mali-G76", 31.61f},
        {"Adreno (TM) 505", 3.19f},
        {"Adreno (TM) 506", 4.74f},
        {"Adreno (TM) 512", 14.23f},
        {"Adreno (TM) 530", 25.40f},
        {"Adreno (TM) 540", 42.74f},
        {"Adreno (TM) 615", 16.77f},
        {"Adreno (TM) 616", 18.77f},
        {"Adreno (TM) 618", 18.77f},
        {"Adreno (TM) 630", 42.74f},
        {"Adreno (TM) 640", 42.74f},
    };
    mFlops = 4.0f;//Default set as 4G, it will be larger than single-core cpu
    std::string deviceName = dev.proty().deviceName;
    //FUNC_PRINT_ALL(deviceName.c_str(), s);
    if (gFlopsMap.find(deviceName)!=gFlopsMap.end()) {
        mFlops = gFlopsMap[deviceName];
    }
    //FUNC_PRINT_ALL(mFlops, f);

    if (deviceName.find("Mali") != std::string::npos) {
        mGpuType = MALI;
    } else if (deviceName.find("Adreno") != std::string::npos) {
        mGpuType = ADRENO;
    }
    bool fp16 = true;
    if (info.user != nullptr) {
        fp16 = info.user->precision != BackendConfig::Precision_High;
    }
    mMemoryPool        = std::make_shared<VulkanMemoryPool>(dev, fp16);
    std::shared_ptr<BufferAllocator::Allocator> allocReal(new VulkanBufferAllocator(dev, *mMemoryPool));
    mBufferPool.reset(new EagerBufferAllocator(allocReal, dev.proty().limits.nonCoherentAtomSize));
    mSampler         = std::make_shared<VulkanSampler>(dev, VK_FILTER_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER);
    mClampSampler         = std::make_shared<VulkanSampler>(dev, VK_FILTER_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
    mPipelineFactory = std::make_shared<VulkanPipelineFactory>(dev);
    mQueryPool = std::make_shared<VulkanQueryPool>(dev);
}

VulkanRuntime::~VulkanRuntime() {
    mBufferPool = nullptr;
    while (!mUniformCache.empty()) {
        mUniformCache.pop();
    }
    mQueryPool = nullptr;
    mCmdPool = nullptr;
    mSampler = nullptr;
    mClampSampler = nullptr;
    mPipelineFactory = nullptr;
    mMemoryPool = nullptr;
    mDevice = nullptr;
    mInstance = nullptr;
}
std::shared_ptr<VulkanBuffer> VulkanRuntime::allocUniform(const void* src, int size) {
    std::shared_ptr<VulkanBuffer> res;
    int allocSize = size;
    if (allocSize < mUniformSize) {
        allocSize = mUniformSize;
    }
    if (mUniformCache.empty() || allocSize > mUniformSize) {
        res = std::shared_ptr<VulkanBuffer>(new VulkanBuffer(*mMemoryPool, false, allocSize, nullptr, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT));
    } else {
        res = mUniformCache.front();
        mUniformCache.pop();
    }
    if (nullptr != src) {
        auto dst = res->map();
        ::memcpy(dst, src, size);
        res->unmap();
    }
    return res;
}
void VulkanRuntime::recycleUniform(std::shared_ptr<VulkanBuffer> buffer) {
    if (buffer->size() < mUniformSize) {
        return;
    }
    if (mUniformCache.size() >= mCacheUniformLimitSize) {
        return;
    }
    mUniformCache.push(buffer);
}

void VulkanRuntime::onGabageCollect(int level) {
    mBufferPool->release(false);
    mMemoryPool->clear();
    mPipelineFactory->reset();
}

Backend* VulkanRuntime::onCreate(const BackendConfig* config) const {
    // FIXME: Use config
    return new VulkanBackend(this, mInfo);
}
int VulkanRuntime::onGetRuntimeStatus(RuntimeStatus statusEnum) const {
    switch (statusEnum) {
        case STATUS_SUPPORT_FP16: {
            return 1;
            break;
        }
        case STATUS_SUPPORT_DOT_PRODUCT: {
            return 0;
            break;
        }
        default: {
            MNN_ERROR("unsupported interface");
            break;
        }
    }
    return 0;
}
static bool _testVulkan() {
    // std::make_unique need c++14
    std::unique_ptr<VulkanInstance> instance(new VulkanInstance());
    if (nullptr == instance) {
        MNN_ERROR("Invalide device for support vulkan\n");
        return false;
    }
    if (!instance->success()) {
        MNN_ERROR("Invalide device for support vulkan\n");
        return false;
    }
    if (!instance->supportVulkan()) {
        MNN_ERROR("Invalide device for support vulkan\n");
        return false;
    }
    return true;
}

class VulkanRuntimeCreator : public RuntimeCreator {
public:
    virtual Runtime* onCreate(const Backend::Info& info) const {
        if (InitVulkan()) {
            if (_testVulkan()) {
                return new VulkanRuntime(info);
            }
        }
        return nullptr;
    }
    virtual bool onValid(Backend::Info& info) const {
        return true;
    }
};

static bool gResistor = []() {
    MNNInsertExtraRuntimeCreator(MNN_FORWARD_VULKAN, new VulkanRuntimeCreator, true);
    return false;
}();
}
