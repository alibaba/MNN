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
    mSampler         = std::make_shared<VulkanSampler>(dev, VK_FILTER_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER);
    mClampSampler         = std::make_shared<VulkanSampler>(dev, VK_FILTER_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
    mPipelineFactory = std::make_shared<VulkanPipelineFactory>(dev);
}

VulkanRuntime::~VulkanRuntime() {
    mCmdPool = nullptr;
    mSampler = nullptr;
    mClampSampler = nullptr;
    mPipelineFactory = nullptr;
    mMemoryPool = nullptr;
    mDevice = nullptr;
    mInstance = nullptr;
}

void VulkanRuntime::onGabageCollect(int level) {
    mMemoryPool->clear();
    mPipelineFactory->reset();
}

Backend* VulkanRuntime::onCreate(const BackendConfig* config) const {
    // FIXME: Use config
    return new VulkanBackend(this, mInfo);
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
