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
VulkanRuntime* VulkanRuntime::create(const Backend::Info& info) {
    MNNVulkanContext* context = nullptr;
    std::shared_ptr<VulkanDevice> device;
    std::shared_ptr<VulkanInstance> instance;
    if (nullptr != info.user && nullptr != info.user->sharedContext) {
       MNN_PRINT("Use user's vulkan context\n");
       context = static_cast<MNNVulkanContext*>(info.user->sharedContext);
    }
    if (NULL != context) {
        instance = std::make_shared<VulkanInstance>(context->pInstance);
        if (context->pInstance == VK_NULL_HANDLE) {
            MNN_ERROR("Invalide user's vulkan instance\n");
            return nullptr;
        }
        device   = std::make_shared<VulkanDevice>(instance, context->pPhysicalDevice, context->pDevice,
                                                 context->iQueueFamilyIndex, context->pQueue);
    } else {
        instance = std::make_shared<VulkanInstance>();
        if (!instance->supportVulkan()) {
            MNN_ERROR("Invalide device for support vulkan\n");
            return nullptr;
        }
        device = std::make_shared<VulkanDevice>(instance);
    }
    if (device->get() == VK_NULL_HANDLE) {
        return nullptr;
    }
    return new VulkanRuntime(info, device, instance);
}

VulkanRuntime::VulkanRuntime(const Backend::Info& info, std::shared_ptr<VulkanDevice> device, std::shared_ptr<VulkanInstance> instance) {
    mInfo = info;
    mDevice = device;
    mInstance = instance;
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

    std::vector<int> legalModeValues = {0x00000001, 0x00000002, 0x00000004,
                                        0x00000201, 0x00000202, 0x00000204};
    auto iter = std::find(legalModeValues.begin(), legalModeValues.end(), (uint32_t)mInfo.gpuMode);
    if (iter == legalModeValues.end()) {
        MNN_PRINT("The customized gpu mode is illegal for Vulkan backend. Using the default mode.\n");
        mGpuMode = 0x00000004;
    } else {
        mGpuMode = mInfo.gpuMode;
    }
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

Backend* VulkanRuntime::onCreate(const BackendConfig* config, Backend* origin) const {
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

bool VulkanRuntime::onSetCache(const void* buffer, size_t size) {
    // check the validity of the buffer
    if (nullptr == buffer) {
        mTuneBuffer.clear();
        return false;
    }

    flatbuffers::Verifier verifier(static_cast<const uint8_t*>(buffer), size);
    if (!VKCache::VerifyTuneInfoCacheBuffer(verifier)) {
        return false;
    }

    auto tuneInfoCache = VKCache::GetTuneInfoCache(buffer);
    auto tuneInfos = tuneInfoCache->TuneInfos();
    if (!tuneInfos) {
        return false;
    }
    // read from buffer, write to mTuneMap
    for (const auto & tuneInfo : * tuneInfos) {
        VKTuneKey k;
        k.shaderName = tuneInfo->shaderName()->str();
        k.gws = {tuneInfo->gws()->x(), tuneInfo->gws()->y(), tuneInfo->gws()->z()};

        VKTuneValue v;
        v.optimalLws = {tuneInfo->optimalLws()->x(), tuneInfo->optimalLws()->y(), tuneInfo->optimalLws()->z()};
        v.optimalCost = tuneInfo->optimalCost();
        mTuneMap[k] = v;
    }

    return true;
}

std::pair<const void*, size_t> VulkanRuntime::onGetCache() {
    std::unique_ptr<flatbuffers::FlatBufferBuilder> builder(new flatbuffers::FlatBufferBuilder());
    std::unique_ptr<VKCache::TuneInfoCacheT> tuneInfoCache(new VKCache::TuneInfoCacheT());

    for (const auto & kvPair : mTuneMap) {
        const VKTuneKey & k = kvPair.first;
        const VKTuneValue & v = kvPair.second;
        std::unique_ptr<VKCache::TuneInfoT> tuneInfo(new VKCache::TuneInfoT());
        tuneInfo->shaderName = k.shaderName;

        std::unique_ptr<VKCache::WorkSizeT> gwsTemp(new VKCache::WorkSizeT());
        gwsTemp->x = k.gws[0]; gwsTemp->y = k.gws[1]; gwsTemp->z = k.gws[2];
        tuneInfo->gws = std::move(gwsTemp);

        std::unique_ptr<VKCache::WorkSizeT> optimalLwsTemp(new VKCache::WorkSizeT());
        optimalLwsTemp->x = v.optimalLws[0]; optimalLwsTemp->y = v.optimalLws[1]; optimalLwsTemp->z = v.optimalLws[2];
        tuneInfo->optimalLws = std::move(optimalLwsTemp);

        tuneInfo->optimalCost = v.optimalCost;

        tuneInfoCache->TuneInfos.push_back(std::move(tuneInfo));
    }

    auto tuneInfoCacheOffset = VKCache::TuneInfoCache::Pack(*(builder.get()), tuneInfoCache.get());
    builder->Finish(tuneInfoCacheOffset);
    uint8_t *bufTemp = builder->GetBufferPointer();
    size_t size = builder->GetSize();
    mTuneBuffer.resize(size);
    ::memcpy(mTuneBuffer.data(), bufTemp, size);
    return std::make_pair(mTuneBuffer.data(), size);
}

class VulkanRuntimeCreator : public RuntimeCreator {
public:
    virtual Runtime* onCreate(const Backend::Info& info) const {
        if (InitVulkan()) {
            return VulkanRuntime::create(info);
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
