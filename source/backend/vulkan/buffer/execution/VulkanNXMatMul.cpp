#include "VulkanNXMatMul.hpp"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <initializer_list>

namespace MNN {

VulkanNXWeightTensor::VulkanNXWeightTensor(const VulkanDevice& device, VulkanMemoryPool& pool)
    : mDevice(device), mApi(mDevice.getDataGraphInfo()), mMemoryPool(pool) {}

VulkanNXWeightTensor::~VulkanNXWeightTensor() {
    if (VK_NULL_HANDLE == mDevice.get()) {
        return; // Device already gone; its destruction released every allocation.
    }
    // Recorded command buffers may still reference this tensor; teardown is not a hot path.
    vkDeviceWaitIdle(mDevice.get());
    if (mMapped && mMemory.first)
        vkUnmapMemory(mDevice.get(), static_cast<VulkanMemory*>(mMemory.first)->get());
    if (mView)
        mApi.destroyTensorView(mDevice.get(), mView, nullptr);
    if (mTensor)
        mApi.destroyTensor(mDevice.get(), mTensor, nullptr);
    if (mBuffer)
        vkDestroyBuffer(mDevice.get(), mBuffer, nullptr);
    if (mMemory.first)
        mMemoryPool.returnMemory(mMemory);
}

bool VulkanNXWeightTensor::create(int k, int n) {
    if (mTensor || k <= 0 || n <= 0)
        return false;
    if (!mApi.supported || !mApi.nxInt8Candidate || !mDevice.getInt8Support())
        return false;
    mDimensions = {{1, k, n}};
    const uint64_t elements = static_cast<uint64_t>(k) * n;
    mBytes = elements; // R8_SINT is one byte per element.
    const auto& limits = mApi.tensorProperties;
    if (limits.maxTensorDimensionCount < 3 || elements > limits.maxTensorElements ||
        static_cast<uint64_t>(k) > limits.maxPerDimensionTensorElements ||
        static_cast<uint64_t>(n) > limits.maxPerDimensionTensorElements || mBytes > limits.maxTensorSize ||
        mBytes > mDevice.proty().limits.maxStorageBufferRange) {
        MNN_NX_LOG("NX_WEIGHT SKIP: %dx%d exceeds tensor or buffer limits\n", k, n);
        return false;
    }
    mDescription.sType = VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_ARM;
    mDescription.tiling = VK_TENSOR_TILING_LINEAR_ARM;
    mDescription.format = VK_FORMAT_R8_SINT;
    mDescription.dimensionCount = 3;
    mDescription.pDimensions = mDimensions.data();
    mDescription.usage = VK_TENSOR_USAGE_DATA_GRAPH_BIT_ARM;
    VkTensorCreateInfoARM tensorInfo{VK_STRUCTURE_TYPE_TENSOR_CREATE_INFO_ARM};
    tensorInfo.pDescription = &mDescription;
    tensorInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (mApi.createTensor(mDevice.get(), &tensorInfo, nullptr, &mTensor) != VK_SUCCESS) {
        MNN_ERROR("NX_WEIGHT FAIL: create tensor\n");
        return false;
    }
    VkMemoryDedicatedRequirements tensorDedicated{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS};
    VkMemoryRequirements2 tensorRequirements{VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
    tensorRequirements.pNext = &tensorDedicated;
    VkTensorMemoryRequirementsInfoARM tensorMemoryInfo{VK_STRUCTURE_TYPE_TENSOR_MEMORY_REQUIREMENTS_INFO_ARM};
    tensorMemoryInfo.tensor = mTensor;
    mApi.getTensorMemoryRequirements(mDevice.get(), &tensorMemoryInfo, &tensorRequirements);
    if (tensorDedicated.requiresDedicatedAllocation) {
        MNN_NX_LOG("NX_WEIGHT SKIP: tensor requires dedicated allocation\n");
        return false;
    }
    VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufferInfo.size = mBytes;
    bufferInfo.usage =
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(mDevice.get(), &bufferInfo, nullptr, &mBuffer) != VK_SUCCESS) {
        MNN_ERROR("NX_WEIGHT FAIL: create alias buffer\n");
        return false;
    }
    VkMemoryDedicatedRequirements bufferDedicated{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS};
    VkMemoryRequirements2 bufferRequirements{VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
    bufferRequirements.pNext = &bufferDedicated;
    VkBufferMemoryRequirementsInfo2 bufferMemoryInfo{VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2};
    bufferMemoryInfo.buffer = mBuffer;
    auto getBufferRequirements = reinterpret_cast<PFN_vkGetBufferMemoryRequirements2>(
        vkGetDeviceProcAddr(mDevice.get(), "vkGetBufferMemoryRequirements2"));
    getBufferRequirements(mDevice.get(), &bufferMemoryInfo, &bufferRequirements);
    if (bufferDedicated.requiresDedicatedAllocation) {
        MNN_NX_LOG("NX_WEIGHT SKIP: buffer requires dedicated allocation\n");
        return false;
    }
    VkMemoryRequirements combined = tensorRequirements.memoryRequirements;
    combined.size = std::max(combined.size, bufferRequirements.memoryRequirements.size);
    combined.alignment = std::max(combined.alignment, bufferRequirements.memoryRequirements.alignment);
    combined.memoryTypeBits &= bufferRequirements.memoryRequirements.memoryTypeBits;
    if (combined.memoryTypeBits == 0 || combined.size == 0) {
        MNN_ERROR("NX_WEIGHT FAIL: no memory type satisfies tensor and buffer\n");
        return false;
    }
    // On a UMA device a host-visible tensor lets the transposed weights be written straight into the
    // memory the data graph reads, dropping the staging buffer, the per-op copy and its queue wait.
    const VkFlags hostMask = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    VkFlags request = 0;
    for (uint32_t i = 0; i < mDevice.memProty().memoryTypeCount; ++i) {
        if ((combined.memoryTypeBits & (1u << i)) &&
            (mDevice.memProty().memoryTypes[i].propertyFlags & hostMask) == hostMask) {
            request = hostMask;
            break;
        }
    }
    mMemory = mMemoryPool.allocMemory(combined, request, true);
    if (!mMemory.first) {
        MNN_ERROR("NX_WEIGHT FAIL: out of memory for %llu bytes\n", static_cast<unsigned long long>(combined.size));
        return false;
    }
    const auto memory = static_cast<VulkanMemory*>(mMemory.first)->get();
    if (vkBindBufferMemory(mDevice.get(), mBuffer, memory, mMemory.second) != VK_SUCCESS) {
        MNN_ERROR("NX_WEIGHT FAIL: bind alias buffer\n");
        return false;
    }
    VkBindTensorMemoryInfoARM bind{VK_STRUCTURE_TYPE_BIND_TENSOR_MEMORY_INFO_ARM};
    bind.tensor = mTensor;
    bind.memory = memory;
    bind.memoryOffset = mMemory.second;
    if (mApi.bindTensorMemory(mDevice.get(), 1, &bind) != VK_SUCCESS) {
        MNN_ERROR("NX_WEIGHT FAIL: bind tensor memory\n");
        return false;
    }
    VkTensorViewCreateInfoARM view{VK_STRUCTURE_TYPE_TENSOR_VIEW_CREATE_INFO_ARM};
    view.tensor = mTensor;
    view.format = VK_FORMAT_R8_SINT;
    if (mApi.createTensorView(mDevice.get(), &view, nullptr, &mView) != VK_SUCCESS) {
        MNN_ERROR("NX_WEIGHT FAIL: create tensor view\n");
        return false;
    }
    if (request && vkMapMemory(mDevice.get(), memory, mMemory.second, mBytes, 0, &mMapped) != VK_SUCCESS) {
        mMapped = nullptr; // Not fatal: upload() falls back to the staging copy.
    }
    mK = k;
    mN = n;
    return true;
}

bool VulkanNXWeightTensor::upload(const void* data, size_t bytes, VulkanBackend* backend) {
    if (!mTensor || mUploaded || !data || !backend || bytes != static_cast<size_t>(mBytes))
        return false;
    if (mMapped) {
        ::memcpy(mMapped, data, bytes);
        mUploaded = true;
        return true;
    }
    auto staging = backend->createHostBuffer(bytes);
    if (!staging || !staging->valid()) {
        MNN_ERROR("NX_WEIGHT FAIL: staging buffer unavailable\n");
        return false;
    }
    void* mapped = staging->map();
    if (!mapped) {
        MNN_ERROR("NX_WEIGHT FAIL: cannot map staging buffer\n");
        return false;
    }
    ::memcpy(mapped, data, bytes);
    staging->unmap();
    backend->copyGPUToGPUBuffer(staging->buffer(), mBuffer, mBytes, 0, 0);
    const auto result = vkQueueWaitIdle(mDevice.acquireDefaultDevQueue());
    if (result != VK_SUCCESS) {
        MNN_ERROR("NX_WEIGHT FAIL: queue wait returned %d\n", result);
        return false;
    }
    mUploaded = true;
    return true;
}

bool VulkanNXMatMul::adoptWeightTensor(TensorResource& resource, VulkanNXWeightTensor& weights, int k, int n) {
    if (!weights.mUploaded || weights.mK != k || weights.mN != n || !weights.mTensor || !weights.mView ||
        !weights.mBuffer || !weights.mMemory.first) {
        return false;
    }
    resource.dimensions = weights.mDimensions;
    resource.description = weights.mDescription;
    resource.description.pDimensions = resource.dimensions.data();
    resource.tensor = weights.mTensor;
    resource.view = weights.mView;
    resource.buffer = weights.mBuffer;
    resource.memory = weights.mMemory;
    resource.bytes = weights.mBytes;
    resource.external = true;
    return true;
}

VulkanNXMatMul::VulkanNXMatMul(VulkanBackend* backend)
    : VulkanBasicExecution(backend),
      mBackend(backend),
      mDevice(backend->getDevice()),
      mApi(mDevice.getDataGraphInfo()),
      mMemoryPool(const_cast<VulkanMemoryPool&>(backend->getMemoryPool())) {}

VulkanNXMatMul::~VulkanNXMatMul() {
    mBackend->finish();
    if (mSession)
        mApi.destroySession(mDevice.get(), mSession, nullptr);
    for (auto memory : mSessionMemory)
        mMemoryPool.returnMemory(memory);
    if (mDescriptorPool)
        vkDestroyDescriptorPool(mDevice.get(), mDescriptorPool, nullptr);
    if (mPipeline)
        vkDestroyPipeline(mDevice.get(), mPipeline, nullptr);
    if (mLayout)
        vkDestroyPipelineLayout(mDevice.get(), mLayout, nullptr);
    if (mSetLayout)
        vkDestroyDescriptorSetLayout(mDevice.get(), mSetLayout, nullptr);
    if (mShader)
        vkDestroyShaderModule(mDevice.get(), mShader, nullptr);
    for (auto& resource : mTensors) {
        if (resource.external)
            continue; // Owned by the caller's VulkanNXWeightTensor.
        if (resource.view)
            mApi.destroyTensorView(mDevice.get(), resource.view, nullptr);
        if (resource.tensor)
            mApi.destroyTensor(mDevice.get(), resource.tensor, nullptr);
        if (resource.buffer)
            vkDestroyBuffer(mDevice.get(), resource.buffer, nullptr);
        if (resource.memory.first && !resource.aliasMemory)
            mMemoryPool.returnMemory(resource.memory);
    }
}

bool VulkanNXMatMul::check(VkResult result, const char* operation) const {
    if (result != VK_SUCCESS) {
        mLastError = result;
        MNN_ERROR("GRAPH FAIL: %s returned %d\n", operation, result);
        return false;
    }
    return true;
}

#ifdef MNN_VULKAN_NX_DEBUG
// The integer path gates only on tensor format features, so without this dump we have no evidence
// of which TOSA profiles/operations the engine actually advertises beyond MATMUL.
void VulkanNXMatMul::dumpTosaCapabilities() {
    uint32_t count = 0;
    if (mApi.getQueueFamilyProperties(mDevice.physicalDevice(), mDevice.queueFamilyIndex(), &count, nullptr) !=
        VK_SUCCESS) {
        MNN_NX_LOG("NX_TOSA_CAPS: queue family property query failed\n");
        return;
    }
    std::vector<VkQueueFamilyDataGraphPropertiesARM> properties(count);
    for (auto& property : properties)
        property.sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_DATA_GRAPH_PROPERTIES_ARM;
    if (mApi.getQueueFamilyProperties(mDevice.physicalDevice(), mDevice.queueFamilyIndex(), &count,
                                      properties.data()) != VK_SUCCESS) {
        MNN_NX_LOG("NX_TOSA_CAPS: queue family property fetch failed\n");
        return;
    }
    for (uint32_t i = 0; i < count; ++i) {
        const auto& property = properties[i];
        MNN_NX_LOG("NX_TOSA_CAPS engine=%d foreign=%d operation=%s version=%u\n", property.engine.type,
                   property.engine.isForeign, property.operation.name, property.operation.version);
        VkQueueFamilyDataGraphTOSAPropertiesARM tosa{VK_STRUCTURE_TYPE_QUEUE_FAMILY_DATA_GRAPH_TOSA_PROPERTIES_ARM};
        if (mApi.getEngineOperationProperties(mDevice.physicalDevice(), mDevice.queueFamilyIndex(), &property,
                                              reinterpret_cast<VkBaseOutStructure*>(&tosa)) != VK_SUCCESS) {
            MNN_NX_LOG("NX_TOSA_CAPS operation=%s tosa properties unavailable\n", property.operation.name);
            continue;
        }
        for (uint32_t index = 0; index < tosa.profileCount; ++index)
            MNN_NX_LOG("NX_TOSA_CAPS operation=%s profile=%s quality=0x%x\n", property.operation.name,
                       tosa.pProfiles[index].name, tosa.pProfiles[index].qualityFlags);
        for (uint32_t index = 0; index < tosa.extensionCount; ++index)
            MNN_NX_LOG("NX_TOSA_CAPS operation=%s extension=%s quality=0x%x\n", property.operation.name,
                       tosa.pExtensions[index].name, tosa.pExtensions[index].qualityFlags);
    }
    // Fusing quant/epilogue needs FP32 temporaries to reproduce the shader's highp roundEven bit-exactly.
    for (VkFormat type : {VK_FORMAT_R32_SFLOAT, VK_FORMAT_R16_SFLOAT, VK_FORMAT_R8_SINT, VK_FORMAT_R32_SINT}) {
        VkTensorFormatPropertiesARM tensorFormat{VK_STRUCTURE_TYPE_TENSOR_FORMAT_PROPERTIES_ARM};
        VkFormatProperties2 format{VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2};
        format.pNext = &tensorFormat;
        mApi.getFormatProperties2(mDevice.physicalDevice(), type, &format);
        MNN_NX_LOG("NX_TENSOR_CAPS format=%d linear=0x%llx optimal=0x%llx\n", type,
                   static_cast<unsigned long long>(tensorFormat.linearTilingTensorFeatures),
                   static_cast<unsigned long long>(tensorFormat.optimalTilingTensorFeatures));
    }
}
#endif

bool VulkanNXMatMul::checkCapabilities() {
    // Statistics-capable drivers change the session bind-point layout; this path only supports
    // the plain memory bind points.
    if (!mApi.supported || !mApi.nxInt8Candidate || !mDevice.getInt8Support() || mApi.neuralStatistics)
        return false;
#ifdef MNN_VULKAN_NX_DEBUG
    dumpTosaCapabilities();
#endif
    for (VkFormat type : {VK_FORMAT_R8_SINT, VK_FORMAT_R32_SINT}) {
        VkTensorFormatPropertiesARM properties{VK_STRUCTURE_TYPE_TENSOR_FORMAT_PROPERTIES_ARM};
        VkFormatProperties2 format{VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2};
        format.pNext = &properties;
        mApi.getFormatProperties2(mDevice.physicalDevice(), type, &format);
        MNN_NX_LOG("NX_TENSOR_CAPS format=%d linear=0x%llx optimal=0x%llx\n", type,
                   static_cast<unsigned long long>(properties.linearTilingTensorFeatures),
                   static_cast<unsigned long long>(properties.optimalTilingTensorFeatures));
        if (!(properties.linearTilingTensorFeatures & VK_FORMAT_FEATURE_2_TENSOR_DATA_GRAPH_BIT_ARM)) {
            MNN_NX_LOG("NX_CONV_SKIP: LINEAR tensor format=%d unavailable\n", type);
            return false;
        }
    }
    return true;
}

bool VulkanNXMatMul::allocate(const VkMemoryRequirements& requirements, MemChunk& memory) {
    if (requirements.memoryTypeBits == 0 || requirements.size == 0) {
        MNN_ERROR("GRAPH FAIL: empty memory requirements\n");
        return false;
    }
    memory = mMemoryPool.allocMemory(requirements, 0, true);
    return memory.first != nullptr;
}

bool VulkanNXMatMul::createTensor(uint32_t slot, TensorResource& resource, int rows, int columns, VkFormat format) {
    if (rows <= 0 || columns <= 0)
        return false;
    resource.dimensions = {{1, rows, columns}};
    const uint64_t elements = static_cast<uint64_t>(rows) * columns;
    const uint32_t elementBytes = format == VK_FORMAT_R32_SINT ? 4 : 1;
    resource.bytes = elements * elementBytes;
    const auto& limits = mApi.tensorProperties;
    if (limits.maxTensorDimensionCount < 3 || elements > limits.maxTensorElements ||
        static_cast<uint64_t>(rows) > limits.maxPerDimensionTensorElements ||
        static_cast<uint64_t>(columns) > limits.maxPerDimensionTensorElements ||
        resource.bytes > limits.maxTensorSize || resource.bytes > mDevice.proty().limits.maxStorageBufferRange) {
        MNN_NX_LOG("GRAPH SKIP: tensor %dx%d elements=%llu exceeds device limits\n", rows, columns,
                   static_cast<unsigned long long>(elements));
        return false;
    }
    resource.description.sType = VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_ARM;
    resource.description.tiling = VK_TENSOR_TILING_LINEAR_ARM;
    resource.description.format = format;
    resource.description.dimensionCount = 3;
    resource.description.pDimensions = resource.dimensions.data();
    resource.description.usage = VK_TENSOR_USAGE_DATA_GRAPH_BIT_ARM;
    VkTensorCreateInfoARM tensorInfo{VK_STRUCTURE_TYPE_TENSOR_CREATE_INFO_ARM};
    tensorInfo.pDescription = &resource.description;
    tensorInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (!check(mApi.createTensor(mDevice.get(), &tensorInfo, nullptr, &resource.tensor), "create tensor"))
        return false;
    VkMemoryDedicatedRequirements tensorDedicated{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS};
    VkMemoryRequirements2 tensorRequirements{VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
    tensorRequirements.pNext = &tensorDedicated;
    VkTensorMemoryRequirementsInfoARM tensorMemoryInfo{VK_STRUCTURE_TYPE_TENSOR_MEMORY_REQUIREMENTS_INFO_ARM};
    tensorMemoryInfo.tensor = resource.tensor;
    mApi.getTensorMemoryRequirements(mDevice.get(), &tensorMemoryInfo, &tensorRequirements);
    if (tensorDedicated.requiresDedicatedAllocation) {
        MNN_NX_LOG("GRAPH SKIP: tensor requires dedicated allocation\n");
        return false;
    }
    // The quant/epilogue compute shaders address the operands through this alias buffer.
    VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufferInfo.size = resource.bytes;
    bufferInfo.usage =
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (!check(vkCreateBuffer(mDevice.get(), &bufferInfo, nullptr, &resource.buffer), "create alias buffer"))
        return false;
    VkMemoryDedicatedRequirements bufferDedicated{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS};
    VkMemoryRequirements2 bufferRequirements{VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
    bufferRequirements.pNext = &bufferDedicated;
    VkBufferMemoryRequirementsInfo2 bufferMemoryInfo{VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2};
    bufferMemoryInfo.buffer = resource.buffer;
    auto getBufferRequirements = reinterpret_cast<PFN_vkGetBufferMemoryRequirements2>(
        vkGetDeviceProcAddr(mDevice.get(), "vkGetBufferMemoryRequirements2"));
    getBufferRequirements(mDevice.get(), &bufferMemoryInfo, &bufferRequirements);
    if (bufferDedicated.requiresDedicatedAllocation) {
        MNN_NX_LOG("GRAPH SKIP: buffer requires dedicated allocation\n");
        return false;
    }
    VkMemoryRequirements combined = tensorRequirements.memoryRequirements;
    combined.size = std::max(combined.size, bufferRequirements.memoryRequirements.size);
    combined.alignment = std::max(combined.alignment, bufferRequirements.memoryRequirements.alignment);
    combined.memoryTypeBits &= bufferRequirements.memoryRequirements.memoryTypeBits;
    if (combined.memoryTypeBits == 0) {
        MNN_ERROR("GRAPH FAIL: no memory type satisfies tensor and its alias buffer\n");
        return false;
    }
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize offset = 0;
    if (mAlias[slot].first != nullptr) {
        auto* arena = static_cast<VulkanMemory*>(mAlias[slot].first);
        const auto start = static_cast<VkDeviceSize>(mAlias[slot].second);
        if (!(combined.memoryTypeBits & (1u << arena->type())) || start % combined.alignment != 0 ||
            combined.size > arena->size() || start > arena->size() - combined.size) {
            MNN_NX_LOG("ALIAS SKIP: slot=%u type=%u bits=0x%x offset=%llu need=%llu arena=%llu align=%llu\n", slot,
                       arena->type(), combined.memoryTypeBits, static_cast<unsigned long long>(start),
                       static_cast<unsigned long long>(combined.size), static_cast<unsigned long long>(arena->size()),
                       static_cast<unsigned long long>(combined.alignment));
            return false;
        }
        resource.memory = mAlias[slot];
        resource.aliasMemory = true;
        memory = arena->get();
        offset = start;
    } else if (!allocate(combined, resource.memory)) {
        return false;
    } else {
        memory = static_cast<VulkanMemory*>(resource.memory.first)->get();
        offset = resource.memory.second;
    }
    mSetupTimes.tensorBytes += combined.size;
    if (!check(vkBindBufferMemory(mDevice.get(), resource.buffer, memory, offset), "bind alias buffer"))
        return false;
    VkBindTensorMemoryInfoARM bind{VK_STRUCTURE_TYPE_BIND_TENSOR_MEMORY_INFO_ARM};
    bind.tensor = resource.tensor;
    bind.memory = memory;
    bind.memoryOffset = offset;
    if (!check(mApi.bindTensorMemory(mDevice.get(), 1, &bind), "bind tensor"))
        return false;
    VkTensorViewCreateInfoARM view{VK_STRUCTURE_TYPE_TENSOR_VIEW_CREATE_INFO_ARM};
    view.tensor = resource.tensor;
    view.format = format;
    return check(mApi.createTensorView(mDevice.get(), &view, nullptr, &resource.view), "create tensor view");
}

MemChunk VulkanNXMatMul::takeMemory(uint32_t index) {
    if (index >= kGraphTensorSlots) {
        return MemChunk();
    }
    auto& resource = mTensors[index];
    if (!resource.memory.first || resource.external || resource.aliasMemory) {
        return MemChunk();
    }
    resource.aliasMemory = true;
    return resource.memory;
}

bool VulkanNXMatMul::createSession() {
    VkDataGraphPipelineSessionCreateInfoARM info{VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SESSION_CREATE_INFO_ARM};
    info.dataGraphPipeline = mPipeline;
    if (!check(mApi.createSession(mDevice.get(), &info, nullptr, &mSession), "create session"))
        return false;
    VkDataGraphPipelineSessionBindPointRequirementsInfoARM query{
        VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_REQUIREMENTS_INFO_ARM};
    query.session = mSession;
    uint32_t count = 0;
    if (!check(mApi.getSessionBindPointRequirements(mDevice.get(), &query, &count, nullptr),
               "session bind point count"))
        return false;
    std::vector<VkDataGraphPipelineSessionBindPointRequirementARM> points(count);
    for (auto& point : points)
        point.sType = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_REQUIREMENT_ARM;
    if (!check(mApi.getSessionBindPointRequirements(mDevice.get(), &query, &count, points.data()),
               "session bind points"))
        return false;
    for (uint32_t i = 0; i < count; ++i) {
        const auto& point = points[i];
        if (point.bindPointType != VK_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_TYPE_MEMORY_ARM)
            return false;
        for (uint32_t object = 0; object < point.numObjects; ++object) {
            VkDataGraphPipelineSessionMemoryRequirementsInfoARM requirementInfo{
                VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SESSION_MEMORY_REQUIREMENTS_INFO_ARM};
            requirementInfo.session = mSession;
            requirementInfo.bindPoint = point.bindPoint;
            requirementInfo.objectIndex = object;
            VkMemoryRequirements2 requirements{VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
            mApi.getSessionMemoryRequirements(mDevice.get(), &requirementInfo, &requirements);
            MNN_NX_LOG("Session bindPoint=%d object=%u bytes=%llu\n", point.bindPoint, object,
                       static_cast<unsigned long long>(requirements.memoryRequirements.size));
            if (requirements.memoryRequirements.size == 0)
                continue;
            MemChunk memory;
            if (!allocate(requirements.memoryRequirements, memory))
                return false;
            mSessionMemory.push_back(memory);
            mSetupTimes.sessionBytes += requirements.memoryRequirements.size;
            VkBindDataGraphPipelineSessionMemoryInfoARM bind{
                VK_STRUCTURE_TYPE_BIND_DATA_GRAPH_PIPELINE_SESSION_MEMORY_INFO_ARM};
            bind.session = mSession;
            bind.bindPoint = point.bindPoint;
            bind.objectIndex = object;
            bind.memory = static_cast<VulkanMemory*>(memory.first)->get();
            bind.memoryOffset = memory.second;
            if (!check(mApi.bindSessionMemory(mDevice.get(), 1, &bind), "bind session memory"))
                return false;
        }
    }
    return true;
}

std::vector<uint32_t> VulkanNXMatMul::makeInt8Graph(int m, int k, int n) {
    if (m <= 0 || k <= 0 || n <= 0)
        return {};
    enum Id : uint32_t {
        Tosa = 1,
        UInt,
        Int8,
        Zero,
        One,
        Three,
        M,
        K,
        N,
        Int8Zero,
        Array1,
        Array3,
        ShapeZero,
        ShapeA,
        ShapeB,
        ShapeC,
        TensorZero,
        TensorA,
        TensorB,
        TensorC,
        PtrA,
        PtrB,
        PtrC,
        ZeroPoint,
        GraphType,
        InputA,
        InputB,
        Output,
        Graph,
        A,
        B,
        C,
        Bound
    };
    enum Op : uint32_t {
        Extension = 10,
        ExtInstImport = 11,
        ExtInst = 12,
        MemoryModel = 14,
        Capability = 17,
        TypeInt = 21,
        TypeArray = 28,
        TypePointer = 32,
        Constant = 43,
        ConstantComposite = 44,
        Variable = 59,
        Decorate = 71,
        TypeTensor = 4163,
        GraphEntryPoint = 4182,
        GraphBegin = 4183,
        GraphInput = 4184,
        GraphSetOutput = 4185,
        GraphEnd = 4186,
        TypeGraph = 4190
    };
    std::vector<uint32_t> code{0x07230203, 0x00010600, 0, Bound, 0};
    auto emit = [&](Op op, std::initializer_list<uint32_t> words) {
        code.push_back((static_cast<uint32_t>(words.size() + 1) << 16) | op);
        code.insert(code.end(), words.begin(), words.end());
    };
    auto emitString = [&](Op op, std::initializer_list<uint32_t> prefix, const char* text,
                          std::initializer_list<uint32_t> suffix) {
        const size_t words = (std::strlen(text) + 4) / 4;
        code.push_back((static_cast<uint32_t>(1 + prefix.size() + words + suffix.size()) << 16) | op);
        code.insert(code.end(), prefix.begin(), prefix.end());
        const size_t start = code.size();
        code.resize(start + words, 0);
        std::memcpy(code.data() + start, text, std::strlen(text));
        code.insert(code.end(), suffix.begin(), suffix.end());
    };
    emit(Capability, {1});
    emit(Capability, {39});
    emit(Capability, {4174});
    emit(Capability, {4191});
    emitString(Extension, {}, "SPV_ARM_tensors", {});
    emitString(Extension, {}, "SPV_ARM_graph", {});
    emitString(ExtInstImport, {Tosa}, "TOSA.001000.1", {});
    emit(MemoryModel, {0, 1});
    for (uint32_t binding = 0; binding < 3; ++binding) {
        emit(Decorate, {InputA + binding, 34, 0});
        emit(Decorate, {InputA + binding, 33, binding});
    }
    emit(TypeInt, {UInt, 32, 0});
    emit(TypeInt, {Int8, 8, 0});
    emit(Constant, {UInt, Zero, 0});
    emit(Constant, {UInt, One, 1});
    emit(Constant, {UInt, Three, 3});
    emit(Constant, {UInt, M, static_cast<uint32_t>(m)});
    emit(Constant, {UInt, K, static_cast<uint32_t>(k)});
    emit(Constant, {UInt, N, static_cast<uint32_t>(n)});
    emit(Constant, {Int8, Int8Zero, 0});
    emit(TypeArray, {Array1, UInt, One});
    emit(TypeArray, {Array3, UInt, Three});
    emit(ConstantComposite, {Array1, ShapeZero, One});
    emit(ConstantComposite, {Array3, ShapeA, One, M, K});
    emit(ConstantComposite, {Array3, ShapeB, One, K, N});
    emit(ConstantComposite, {Array3, ShapeC, One, M, N});
    emit(TypeTensor, {TensorZero, Int8, One, ShapeZero});
    emit(TypeTensor, {TensorA, Int8, Three, ShapeA});
    emit(TypeTensor, {TensorB, Int8, Three, ShapeB});
    emit(TypeTensor, {TensorC, UInt, Three, ShapeC});
    emit(TypePointer, {PtrA, 0, TensorA});
    emit(TypePointer, {PtrB, 0, TensorB});
    emit(TypePointer, {PtrC, 0, TensorC});
    emit(ConstantComposite, {TensorZero, ZeroPoint, Int8Zero});
    emit(TypeGraph, {GraphType, 2, TensorA, TensorB, TensorC});
    emit(Variable, {PtrA, InputA, 0});
    emit(Variable, {PtrB, InputB, 0});
    emit(Variable, {PtrC, Output, 0});
    emitString(GraphEntryPoint, {Graph}, "main", {InputA, InputB, Output});
    emit(GraphBegin, {GraphType, Graph});
    emit(GraphInput, {TensorA, A, Zero});
    emit(GraphInput, {TensorB, B, One});
    emit(ExtInst, {TensorC, C, Tosa, 6, A, B, ZeroPoint, ZeroPoint});
    emit(GraphSetOutput, {C, Zero});
    emit(GraphEnd, {});
    return code;
}

bool VulkanNXMatMul::prepareInt8(int m, int k, int n, VulkanNXWeightTensor* weights) {
    if (mTensors[0].tensor || m <= 0 || k <= 0 || n <= 0 || k > 131071)
        return false;
    mSharedWeights = weights;
    const auto start = std::chrono::steady_clock::now();
    const auto code = makeInt8Graph(m, k, n);
    mSetupTimes.graphBuildMs =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    return prepare(code, m, k, n);
}

bool VulkanNXMatMul::prepare(const std::vector<uint32_t>& code, int m, int k, int n) {
    const auto prepareStart = std::chrono::steady_clock::now();
    auto milliseconds = [](std::chrono::steady_clock::time_point start) {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    };
    mSetupTimes.spirvBytes = code.size() * sizeof(uint32_t);
    if (mTensors[0].tensor || !checkCapabilities())
        return false;
    const auto tensorStart = std::chrono::steady_clock::now();
    if (!createTensor(0, mTensors[0], m, k, VK_FORMAT_R8_SINT))
        return false;
    // Weights never depend on M, so adopt the caller's persistent tensor instead of re-uploading.
    const bool adopted = mSharedWeights && adoptWeightTensor(mTensors[1], *mSharedWeights, k, n);
    if (!adopted && !createTensor(1, mTensors[1], k, n, VK_FORMAT_R8_SINT))
        return false;
    if (!createTensor(2, mTensors[2], m, n, VK_FORMAT_R32_SINT))
        return false;
    mSetupTimes.tensorBufferMs = milliseconds(tensorStart);
    const auto layoutStart = std::chrono::steady_clock::now();
    std::array<VkDescriptorSetLayoutBinding, kGraphTensorSlots> bindings{};
    for (uint32_t i = 0; i < kGraphTensorSlots; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_TENSOR_ARM;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_ALL;
    }
    VkDescriptorSetLayoutCreateInfo setInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    setInfo.bindingCount = kGraphTensorSlots;
    setInfo.pBindings = bindings.data();
    if (!check(vkCreateDescriptorSetLayout(mDevice.get(), &setInfo, nullptr, &mSetLayout), "create set layout"))
        return false;
    VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &mSetLayout;
    if (!check(vkCreatePipelineLayout(mDevice.get(), &layoutInfo, nullptr, &mLayout), "create pipeline layout"))
        return false;
    mSetupTimes.pipelineLayoutDescriptorMs = milliseconds(layoutStart);
    const auto shaderStart = std::chrono::steady_clock::now();
    VkShaderModuleCreateInfo shader{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    shader.codeSize = code.size() * sizeof(uint32_t);
    shader.pCode = code.data();
    if (!check(vkCreateShaderModule(mDevice.get(), &shader, nullptr, &mShader), "create graph shader module"))
        return false;
    mSetupTimes.shaderModuleMs = milliseconds(shaderStart);
    std::array<VkDataGraphPipelineResourceInfoARM, kGraphTensorSlots> resources{};
    for (uint32_t i = 0; i < kGraphTensorSlots; ++i) {
        resources[i].sType = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_RESOURCE_INFO_ARM;
        resources[i].pNext = &mTensors[i].description;
        resources[i].binding = i;
    }
    VkDataGraphPipelineShaderModuleCreateInfoARM module{
        VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SHADER_MODULE_CREATE_INFO_ARM};
    module.module = mShader;
    module.pName = "main";
    VkDataGraphPipelineCreateInfoARM pipeline{VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_CREATE_INFO_ARM};
    pipeline.pNext = &module;
    pipeline.layout = mLayout;
    pipeline.resourceInfoCount = kGraphTensorSlots;
    pipeline.pResourceInfos = resources.data();
    const auto start = std::chrono::steady_clock::now();
    if (!check(mApi.createPipelines(mDevice.get(), VK_NULL_HANDLE, mPipelineCache, 1, &pipeline, nullptr, &mPipeline),
               "create data graph pipeline"))
        return false;
    mSetupTimes.pipelineCompileMs = milliseconds(start);
    MNN_NX_LOG("Graph compilation_ms=%.3f\n", mSetupTimes.pipelineCompileMs);
    const auto sessionStart = std::chrono::steady_clock::now();
    if (!createSession())
        return false;
    mSetupTimes.sessionInitMs = milliseconds(sessionStart);
    MNN_NX_LOG("Graph session_init_ms=%.3f\n", mSetupTimes.sessionInitMs);
    const auto descriptorStart = std::chrono::steady_clock::now();
    VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_TENSOR_ARM, kGraphTensorSlots};
    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    if (!check(vkCreateDescriptorPool(mDevice.get(), &poolInfo, nullptr, &mDescriptorPool), "create descriptor pool"))
        return false;
    VkDescriptorSetAllocateInfo setAllocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    setAllocate.descriptorPool = mDescriptorPool;
    setAllocate.descriptorSetCount = 1;
    setAllocate.pSetLayouts = &mSetLayout;
    if (!check(vkAllocateDescriptorSets(mDevice.get(), &setAllocate, &mSet), "allocate descriptor set"))
        return false;
    std::array<VkWriteDescriptorSetTensorARM, kGraphTensorSlots> tensorWrites{};
    std::array<VkWriteDescriptorSet, kGraphTensorSlots> writes{};
    for (uint32_t i = 0; i < kGraphTensorSlots; ++i) {
        tensorWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_TENSOR_ARM;
        tensorWrites[i].tensorViewCount = 1;
        tensorWrites[i].pTensorViews = &mTensors[i].view;
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].pNext = &tensorWrites[i];
        writes[i].dstSet = mSet;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_TENSOR_ARM;
    }
    vkUpdateDescriptorSets(mDevice.get(), kGraphTensorSlots, writes.data(), 0, nullptr);
    mSetupTimes.pipelineLayoutDescriptorMs += milliseconds(descriptorStart);
    mSetupTimes.totalMs = milliseconds(prepareStart) + mSetupTimes.graphBuildMs;
    mReady = true;
    return true;
}

void VulkanNXMatMul::encodeInt8(VkCommandBuffer command) const {
    MNN_ASSERT(mReady);
    const VkPipelineStageFlags2 computeStage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    const VkAccessFlags2 computeAccess = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
    barrier(command, computeStage, computeAccess, VK_PIPELINE_STAGE_2_DATA_GRAPH_BIT_ARM,
            VK_ACCESS_2_DATA_GRAPH_READ_BIT_ARM | VK_ACCESS_2_DATA_GRAPH_WRITE_BIT_ARM);
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_DATA_GRAPH_ARM, mPipeline);
    vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_DATA_GRAPH_ARM, mLayout, 0, 1, &mSet, 0, nullptr);
    mApi.cmdDispatch(command, mSession, nullptr);
    barrier(command, VK_PIPELINE_STAGE_2_DATA_GRAPH_BIT_ARM,
            VK_ACCESS_2_DATA_GRAPH_READ_BIT_ARM | VK_ACCESS_2_DATA_GRAPH_WRITE_BIT_ARM, computeStage, computeAccess);
}

void VulkanNXMatMul::barrier(VkCommandBuffer command, VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
                             VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess) const {
    VkMemoryBarrier2 memory{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    memory.srcStageMask = srcStage;
    memory.srcAccessMask = srcAccess;
    memory.dstStageMask = dstStage;
    memory.dstAccessMask = dstAccess;
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.memoryBarrierCount = 1;
    dependency.pMemoryBarriers = &memory;
    mApi.cmdPipelineBarrier2(command, &dependency);
}

ErrorCode VulkanNXMatMul::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                   const VulkanCommandPool::Buffer* command) {
    return NOT_SUPPORT;
}

} // namespace MNN