//
//  VulkanBackend.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanBackend.hpp"
#include <mutex>
#include "Execution.hpp"
#include "Macro.h"
#include "Tensor.hpp"
#include "TensorUtils.hpp"
#include "VulkanDevice.hpp"
#include "VulkanImageConverter.hpp"
#include "VulkanInstance.hpp"
//#define MNN_OPEN_TIME_TRACE
#include "AutoTime.hpp"
#ifdef MNN_USE_NEON
#include <arm_neon.h>
#endif

namespace MNN {
static std::map<OpType, VulkanBackend::Creator*>* gCreator = nullptr;

// –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
// Creator
// –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
static inline std::map<OpType, VulkanBackend::Creator*>* getCreatorMap() {
    if (nullptr == gCreator) {
        gCreator = new std::map<OpType, VulkanBackend::Creator*>();
    }
    return gCreator;
}

static void _copyBufferToTensor(const Tensor* dest, const VulkanBuffer* source) {
    auto sourcePtr   = source->map();
    auto dataType    = dest->getType();
    //TODO: Support other kind of dataType
    MNN_ASSERT(dataType.bits == 32);
    ::memcpy(dest->host<float>(), sourcePtr, dest->size());
    source->unmap();
}

static int _getAlignSize(const Tensor* tensor) {
    auto format      = TensorUtils::getDescribe(tensor)->dimensionFormat;
    auto elementSize = tensor->elementSize();
    // [TODO] Find a better way
    if (format == MNN_DATA_FORMAT_NCHW) {
        if (tensor->dimensions() >= 2) {
            elementSize = elementSize / tensor->channel() * ALIGN_UP4(tensor->channel());
        }
    } else if (format == MNN_DATA_FORMAT_NHWC) {
        if (tensor->dimensions() >= 3) {
            elementSize = elementSize / tensor->channel() * ALIGN_UP4(tensor->channel());
        }
    }
    return elementSize;
}

static void _copyTensorToBuffer(const Tensor* source, const VulkanBuffer* dest) {
    auto destPtr     = dest->map();
    auto dataType    = source->getType();
    //TODO: Support other kind of dataType
    MNN_ASSERT(dataType.bits == 32);
    ::memcpy(destPtr, source->host<float>(), source->size());
    dest->unmap();
}

VulkanTensor::VulkanTensor(const Tensor* shape, const VulkanMemoryPool& pool, bool forceBuffer, bool seperate) {
    auto format = TensorUtils::getDescribe(shape)->dimensionFormat;
    if (MNN_DATA_FORMAT_NC4HW4 == format && !forceBuffer) {
        mImage = std::make_shared<VulkanImage>(pool, seperate,
                                               std::vector<int>{
                                                   std::max(shape->width(), 1),
                                                   std::max(shape->height(), 1),
                                                   UP_DIV(shape->channel(), 4) * shape->batch(),
                                               },
                                               shape->getType());
    } else {
        // Compute Shader don't support uint8 / int8 / float16 / uint64, all use int32/float32
        mBuffer = std::make_shared<VulkanBuffer>(pool, seperate, _getAlignSize(shape) * sizeof(float));
    }
}
void VulkanTensor::release() {
    if (nullptr != mBuffer.get()) {
        mBuffer->release();
    }
    if (nullptr != mImage.get()) {
        mImage->release();
    }
}

uint64_t VulkanTensor::deviceId() {
    if (mImage.get()) {
        return reinterpret_cast<uint64_t>(mImage->view());
    } else {
        return reinterpret_cast<uint64_t>(mBuffer->buffer());
    }
}

VulkanBackend::VulkanBackend(const MNNVulkanContext* context) : Backend(MNN_FORWARD_VULKAN) {
    if (NULL != context) {
        mInstance = std::make_shared<VulkanInstance>(context->pInstance);
        mDevice   = std::make_shared<VulkanDevice>(mInstance, context->pPhysicalDevice, context->pDevice,
                                                 context->iQueueFamilyIndex, context->pQueue);
    } else {
        mInstance = std::make_shared<VulkanInstance>();
        mDevice   = std::make_shared<VulkanDevice>(mInstance);
    }
    auto& dev              = device();
    mCmdPool               = std::make_shared<VulkanCommandPool>(dev);
    mFence                 = std::make_shared<VulkanFence>(dev);
    std::string deviceName = dev.proty().deviceName;
    if (deviceName.find("Mali") != std::string::npos) {
        mGpuType = MALI;
    } else if (deviceName.find("Adreno") != std::string::npos) {
        mGpuType = ADRENO;
    }

    mMemoryPool        = std::make_shared<VulkanMemoryPool>(dev);
    mDynamicMemoryPool = std::make_shared<VulkanMemoryPool>(dev);
    mSampler         = std::make_shared<VulkanSampler>(dev, VK_FILTER_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER);
    mPipelineFactory = std::make_shared<VulkanPipelineFactory>(dev);
}

VulkanBackend::~VulkanBackend() {
    /*keep release order*/
    mPipelineFactory = nullptr;
    mSampler         = nullptr;

    mStaticeBuffers.clear();
    mAllBuffers.clear();

    mHostBuffer = nullptr;
    mCmdBuffers.clear();
    mFence = nullptr;
    mConverters.clear();

    mDynamicMemoryPool = nullptr;
    mMemoryPool        = nullptr;

    mCmdPool  = nullptr;
    mDevice   = nullptr;
    mInstance = nullptr;
}

bool VulkanBackend::onLoadLibrary(const GpuLibrary* library) {
    // [TODO]: Support Plugin
    return true;
}

void VulkanBackend::pushCommand(VkCommandBuffer buffer) const {
    mCmdBuffers.emplace_back(buffer);
}

const VulkanPipeline* VulkanBackend::getPipeline(const std::string& key, const std::vector<VkDescriptorType>& types,
                                                 const std::vector<uint32_t>& localSize) const {
    return mPipelineFactory->getPipeline(key, types, localSize);
}

bool VulkanBackend::_supportImageSize(const Tensor* MTensor) {
    if (UP_DIV(MTensor->channel(), 4) * MTensor->batch() > device().proty().limits.maxImageDimension3D) {
        return false;
    }
    return true;
}

bool VulkanBackend::onAcquireBuffer(const Tensor* tensor, StorageType storageType) {
    auto MTensor     = const_cast<Tensor*>(tensor);
    auto format      = TensorUtils::getDescribe(MTensor)->dimensionFormat;
    bool forceBuffer = false;
    if (MNN_DATA_FORMAT_NC4HW4 == format) {
        if (!_supportImageSize(MTensor)) {
            // forceBuffer = true;
            MNN_PRINT("Force Use Buffer because then Tensor is too Large: %d, %d, %d, %d\n", MTensor->width(),
                      MTensor->height(), MTensor->channel(), MTensor->batch());
            forceBuffer = true;
        }
    }
    if (Backend::STATIC == storageType) {
        auto newBuffer           = std::make_shared<VulkanTensor>(MTensor, getMemoryPool(), forceBuffer);
        MTensor->buffer().device = newBuffer->deviceId();
        mStaticeBuffers.insert(std::make_pair(MTensor->buffer().device, newBuffer));
    } else {
        bool seperate  = storageType == Backend::DYNAMIC_SEPERATE;
        auto newBuffer = std::make_shared<VulkanTensor>(MTensor, getDynamicMemoryPool(), forceBuffer, seperate);
        MTensor->buffer().device = newBuffer->deviceId();
        mAllBuffers.insert(std::make_pair(MTensor->buffer().device, newBuffer));
    }
    return true;
}
bool VulkanBackend::onReleaseBuffer(const Tensor* tensor, StorageType storageType) {
    auto buffer = (tensor->deviceId());
    if (Backend::DYNAMIC == storageType) {
        auto iter = mAllBuffers.find(buffer);
        MNN_ASSERT(iter != mAllBuffers.end());
        iter->second->release();
    }
    if (Backend::STATIC == storageType) {
        auto iter = mStaticeBuffers.find(buffer);
        MNN_ASSERT(iter != mStaticeBuffers.end());
        mStaticeBuffers.erase(iter);
    }
    return true;
}
bool VulkanBackend::onClearBuffer() {
    mMemoryPool->clear();
    mDynamicMemoryPool->clear();
    mAllBuffers.clear();
    return true;
}
Execution* VulkanBackend::onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                   const MNN::Op* op) {
    auto creator = getCreatorMap();
    auto iter    = creator->find(op->type());
    if (iter == creator->end()) {
        // MNN_PRINT("Vulkan don't support %d, %s: %s\n", op->type(), EnumNameOpType(op->type()),
        // op->name()->c_str());
        return nullptr;
    }
    bool valid = true;
    for (auto t : inputs) {
        if (TensorUtils::getDescribe(t)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4 && !_supportImageSize(t)) {
            valid = false;
            break;
        }
    }
    for (auto t : outputs) {
        if (TensorUtils::getDescribe(t)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4 && !_supportImageSize(t)) {
            valid = false;
            break;
        }
    }
    if (!valid) {
        return nullptr;
    }
    return iter->second->onCreate(inputs, op, this);
}
void VulkanBackend::onExecuteBegin() const {
    // FUNC_PRINT_ALL(mDynamicMemoryPool->computeSize(), f);
}
void VulkanBackend::onExecuteEnd() const {
    _finish();
}
void VulkanBackend::_finish() const {
    if (mCmdBuffers.empty()) {
        return;
    }
    AUTOTIME;
    VkSubmitInfo submit_info = {.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                                .pNext                = nullptr,
                                .waitSemaphoreCount   = 0,
                                .pWaitSemaphores      = nullptr,
                                .pWaitDstStageMask    = nullptr,
                                .commandBufferCount   = (uint32_t)mCmdBuffers.size(),
                                .pCommandBuffers      = mCmdBuffers.data(),
                                .signalSemaphoreCount = 0,
                                .pSignalSemaphores    = nullptr};
    auto fenceReal           = mFence->get();
    mFence->reset();
    CALL_VK(vkQueueSubmit(device().acquireDefaultDevQueue(), 1, &submit_info, fenceReal));

    mCmdBuffers.clear();
    auto res = mFence->wait();
    MNN_VK_CHECK(res);
}

const VulkanDevice& VulkanBackend::device() const {
    return (*mDevice);
}

void VulkanBackend::onCopyBuffer(const Tensor* srcTensor, const Tensor* dstTensor) const {
    AUTOTIME;
    if (srcTensor->host<float>() != nullptr) {
        _finish();
        auto size = _getAlignSize(srcTensor) * 4;
        // host->gpu
        _allocHostBuffer(size);
        _copyTensorToBuffer(srcTensor, mHostBuffer.get());
        auto format = TensorUtils::getDescribe(srcTensor)->dimensionFormat;
        auto key    = std::make_tuple(dstTensor, true, format);
        auto iter   = mConverters.find(key);
        if (iter == mConverters.end()) {
            auto converter = std::make_shared<VulkanImageConverter>(this);
            std::shared_ptr<VulkanCommandPool::Buffer> convertorBuffer(
                const_cast<VulkanCommandPool::Buffer*>(mCmdPool->allocBuffer()));
            convertorBuffer->begin(0);
            converter->encodeBufferToTensor(mHostBuffer->buffer(), dstTensor, mHostBuffer->size(), 0,
                                            TensorUtils::getDescribe(srcTensor)->dimensionFormat,
                                            convertorBuffer.get());
            convertorBuffer->end();
            mConverters.insert(std::make_pair(key, std::make_pair(converter, convertorBuffer)));
            iter = mConverters.find(key);
        }
        mCmdBuffers.push_back(iter->second.second->get());
    } else {
        // gpu->host
        auto size = _getAlignSize(dstTensor) * 4;
        _finish();
        _allocHostBuffer(size);
        auto format = TensorUtils::getDescribe(dstTensor)->dimensionFormat;
        auto key    = std::make_tuple(srcTensor, false, format);

        auto iter = mConverters.find(key);
        if (iter == mConverters.end()) {
            auto converter = std::make_shared<VulkanImageConverter>(this);
            std::shared_ptr<VulkanCommandPool::Buffer> convertorBuffer(
                const_cast<VulkanCommandPool::Buffer*>(mCmdPool->allocBuffer()));
            convertorBuffer->begin(0);
            converter->encodeTensorToBuffer(srcTensor, mHostBuffer->buffer(), mHostBuffer->size(), 0,
                                            TensorUtils::getDescribe(dstTensor)->dimensionFormat,
                                            convertorBuffer.get());
            convertorBuffer->end();
            mConverters.insert(std::make_pair(key, std::make_pair(converter, convertorBuffer)));
            iter = mConverters.find(key);
        }
        mCmdBuffers.push_back(iter->second.second->get());
        _finish();
        _copyBufferToTensor(dstTensor, mHostBuffer.get());
    }
}
const VulkanTensor* VulkanBackend::findTensor(uint64_t deviceId) const {
    auto iter = mAllBuffers.find(deviceId);
    if (iter != mAllBuffers.end()) {
        return iter->second.get();
    }
    return nullptr;
}

void VulkanBackend::_allocHostBuffer(size_t size) const {
    if (mHostBuffer.get() == nullptr || mHostBuffer->size() < size) {
        mHostBuffer =
            std::make_shared<VulkanBuffer>(getMemoryPool(), false, size, nullptr, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                           VK_SHARING_MODE_EXCLUSIVE, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
        mConverters.clear();
    }
}
bool VulkanBackend::addCreator(OpType t, Creator* c) {
    auto allKind = getCreatorMap();
    allKind->insert(std::make_pair(t, c));
    return true;
}

void VulkanBackend::copyBufferToImage(const VulkanBuffer* buffer, const VulkanImage* image) const {
    std::vector<int> dimVector = image->dims();
    if (image->format() != VK_FORMAT_R16G16B16A16_SFLOAT) {
        VkBufferImageCopy copyRegions;
        ::memset(&copyRegions, 0, sizeof(copyRegions));
        copyRegions.imageOffset.x                   = 0;
        copyRegions.imageOffset.y                   = 0;
        copyRegions.imageOffset.z                   = 0;
        copyRegions.imageExtent.depth               = image->depth();
        copyRegions.imageExtent.height              = image->height();
        copyRegions.imageExtent.width               = image->width();
        copyRegions.imageSubresource.layerCount     = 1;
        copyRegions.imageSubresource.mipLevel       = 0;
        copyRegions.imageSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        copyRegions.imageSubresource.baseArrayLayer = 0;

        std::unique_ptr<VulkanCommandPool::Buffer> cmdbuffer(
            const_cast<VulkanCommandPool::Buffer*>(mCmdPool->allocBuffer()));
        cmdbuffer->begin(0);
        vkCmdCopyBufferToImage(cmdbuffer->get(), buffer->buffer(), image->get(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                               1, &copyRegions);
        cmdbuffer->end();
        mCmdPool->submitAndWait(cmdbuffer->get());
    }

    const VulkanPipeline* transformPipeline = nullptr;
    std::vector<VkDescriptorType> types{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
    int localX = 16;
    int localY = 16;
    int localZ = 1;
    switch (dimVector.size()) {
        case 1:
            transformPipeline = getPipeline("glsl_buffer2Image1D_comp",
                                            /*glsl_buffer2Image1D_comp, glsl_buffer2Image1D_comp_len,*/ types);
            localX            = 256;
            localY            = 1;
            break;
        case 2:
            transformPipeline = getPipeline("glsl_buffer2Image2D_comp",
                                            /*glsl_buffer2Image2D_comp, glsl_buffer2Image2D_comp_len,*/ types);
            break;
        case 3:
            transformPipeline = getPipeline("glsl_buffer2Image3D_comp",
                                            /*glsl_buffer2Image3D_comp, glsl_buffer2Image3D_comp_len,*/ types);
            break;
        default:
            break;
    }

    std::unique_ptr<VulkanPipeline::DescriptorSet> sets(transformPipeline->createSet());
    auto constBuffer = std::make_shared<VulkanBuffer>(getMemoryPool(), false, dimVector.size() * sizeof(int),
                                                      dimVector.data(), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    sets->writeImage(image->view(), mSampler->get(), VK_IMAGE_LAYOUT_GENERAL, 0);
    sets->writeBuffer(buffer->buffer(), 1, buffer->size());
    sets->writeBuffer(constBuffer->buffer(), 2, constBuffer->size());

    std::unique_ptr<VulkanCommandPool::Buffer> cmdbuffer(
        const_cast<VulkanCommandPool::Buffer*>(mCmdPool->allocBuffer()));
    cmdbuffer->begin(0);
    transformPipeline->bind(cmdbuffer->get(), sets->get());
    vkCmdDispatch(cmdbuffer->get(), UP_DIV(image->width(), localX), UP_DIV(image->height(), localY),
                  UP_DIV(image->depth(), localZ));
    cmdbuffer->end();
    mCmdPool->submitAndWait(cmdbuffer->get());
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

// –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
// Backend Register
// –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

class VulkanBackendCreator : public BackendCreator {
    virtual Backend* onCreate(const Backend::Info& info) const {
        MNNVulkanContext* context = nullptr;
        if (nullptr != info.user && nullptr != info.user->sharedContext) {
            MNN_PRINT("Use user's vulkan context\n");
            context = static_cast<MNNVulkanContext*>(info.user->sharedContext);
        }
        auto backend = new VulkanBackend(context);
        if (!backend->success()) {
            delete backend;
            return nullptr;
        }
        return backend;
    }
};

static bool gResistor = []() {
    if (InitVulkan()) {
        if (_testVulkan()) {
            MNNInsertExtraBackendCreator(MNN_FORWARD_VULKAN, new VulkanBackendCreator);
        }
        return true;
    }
    return false;
}();

} // namespace MNN
